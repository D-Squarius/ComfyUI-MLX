from __future__ import annotations
from .k_diffusion import sampling as k_diffusion_sampling
from .extra_samplers import uni_pc
from typing import TYPE_CHECKING, Callable, NamedTuple
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.controlnet import ControlBase
import torch
from functools import partial
import collections
import json
import math
import logging
import time
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
import comfy.context_windows
import comfy.utils
import scipy.stats
import numpy


_LTX_SAMPLER_TRACE_KEY = "ltx_sampler_finite_trace"
_LTX_SAMPLER_TRACE_STOP_REASON = "LTX_SAMPLER_FINITE_TRACE_STOP"
_LTX_SAMPLER_TRACE_STATES = {}


def _ltx_sampler_trace_config(model_options):
    if model_options is None:
        return None
    config = model_options.get(_LTX_SAMPLER_TRACE_KEY)
    if not isinstance(config, dict) or not config.get("enabled"):
        return None
    if not config.get("event_path"):
        return None
    return config


def _ltx_sampler_trace_state(config):
    path = config["event_path"]
    state = _LTX_SAMPLER_TRACE_STATES.get(path)
    if state is None:
        state = {
            "event_count": 0,
            "model_call_count": 0,
            "current_model_call": None,
            "stopped": False,
        }
        _LTX_SAMPLER_TRACE_STATES[path] = state
    return state


def _ltx_sampler_trace_parse_filter(value):
    value = str(value or "").strip().lower()
    if not value or value in {"all", "full", "*"}:
        return None
    selected = set()
    for item in value.replace(",", " ").split():
        if "-" in item:
            start_text, _, end_text = item.partition("-")
            if start_text.isdigit() and end_text.isdigit():
                start, end = int(start_text), int(end_text)
                if end < start:
                    start, end = end, start
                selected.update(range(start, end + 1))
                continue
        if item.isdigit():
            selected.add(int(item))
    return selected


def _ltx_sampler_trace_value_selected(value, selector):
    selected = _ltx_sampler_trace_parse_filter(selector)
    return selected is None or int(value) in selected


def _ltx_sampler_trace_stage_selected(stage, selector):
    selector = str(selector or "").strip().lower()
    if not selector or selector in {"all", "full", "*"}:
        return True
    stage = str(stage or "").strip().lower()
    selected = {item.strip() for item in selector.replace(",", " ").split() if item.strip()}
    return stage in selected


def _ltx_sampler_trace_current_model_call(model_options):
    config = _ltx_sampler_trace_config(model_options)
    if config is None:
        return None
    return _ltx_sampler_trace_state(config).get("current_model_call")


def _ltx_sampler_trace_should_trace_block(model_options, block_index, stage):
    config = _ltx_sampler_trace_config(model_options)
    if config is None or config.get("mode") not in {"block_stage_trace", "sync_isolation_trace"}:
        return False
    current_call = _ltx_sampler_trace_current_model_call(model_options)
    if current_call is None:
        return False
    if not _ltx_sampler_trace_value_selected(current_call, config.get("trace_model_calls", "1")):
        return False
    if not _ltx_sampler_trace_value_selected(block_index, config.get("trace_blocks", "29")):
        return False
    return _ltx_sampler_trace_stage_selected(stage, config.get("trace_block_stages", "all"))


def _ltx_sampler_trace_should_use_workaround(model_options, block_index, stage, workaround_name):
    config = _ltx_sampler_trace_config(model_options)
    if config is None:
        return False
    if str(config.get("nan_workaround") or "none").strip().lower() != workaround_name:
        return False
    current_call = _ltx_sampler_trace_current_model_call(model_options)
    if current_call is None:
        return False
    if not _ltx_sampler_trace_value_selected(current_call, config.get("trace_model_calls", "all")):
        return False
    if not _ltx_sampler_trace_value_selected(block_index, config.get("trace_blocks", "all")):
        return False
    return _ltx_sampler_trace_stage_selected(stage, config.get("trace_block_stages", "all"))


def _ltx_sampler_trace_event_stage(metadata):
    metadata = metadata or {}
    stage = metadata.get("stage")
    attention = metadata.get("attention")
    if attention and stage:
        return f"{attention}.{stage}"
    return stage


def _ltx_sampler_trace_event_policy(config, model_options, metadata):
    if config is None or config.get("mode") != "sync_isolation_trace":
        return "cpu_copy_summary"
    metadata = metadata or {}
    block_index = metadata.get("block_index")
    stage = _ltx_sampler_trace_event_stage(metadata)
    if block_index is None or stage is None:
        return "cpu_copy_summary"
    if not _ltx_sampler_trace_should_trace_block(model_options, block_index, stage):
        return "cpu_copy_summary"
    action = str(config.get("sync_isolation_action") or "log_only").strip().lower()
    if action in {"sync_only", "device_finite_only", "cpu_copy_summary"}:
        return action
    return "shape_only"


def _ltx_sampler_trace_tensor_summary(value, policy="cpu_copy_summary"):
    if torch.is_tensor(value):
        summary = {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "summary_policy": policy,
        }
        if policy == "shape_only":
            return summary
        try:
            raw = value.detach()
            try:
                device_finite = torch.isfinite(raw)
                summary.update({
                    "device_finite_count": int(device_finite.sum().item()),
                    "device_nonfinite_count": int((~device_finite).sum().item()),
                    "device_nan_count": int(torch.isnan(raw).sum().item()),
                    "device_inf_count": int(torch.isinf(raw).sum().item()),
                })
            except Exception as exc:
                summary.update({
                    "device_finite_status": "error",
                    "device_finite_error": str(exc),
                })
            if policy == "device_finite_only":
                try:
                    summary["numel"] = int(raw.numel())
                    summary["all_finite"] = bool(int(summary.get("device_nonfinite_count") or 0) == 0)
                except Exception:
                    pass
                return summary
            cpu = raw.float().cpu()
            finite = torch.isfinite(cpu)
            finite_count = int(finite.sum().item())
            numel = int(cpu.numel())
            summary.update({
                "numel": numel,
                "finite_count": finite_count,
                "nonfinite_count": int((~finite).sum().item()),
                "nan_count": int(torch.isnan(cpu).sum().item()),
                "inf_count": int(torch.isinf(cpu).sum().item()),
                "all_finite": bool(finite_count == numel),
            })
            if finite_count > 0:
                finite_values = cpu[finite]
                summary.update({
                    "finite_min": float(finite_values.min().item()),
                    "finite_max": float(finite_values.max().item()),
                    "finite_mean": float(finite_values.mean().item()),
                })
            else:
                summary.update({
                    "finite_min": None,
                    "finite_max": None,
                    "finite_mean": None,
                })
        except Exception as exc:
            summary.update({
                "status": "finite_summary_error",
                "error": str(exc),
            })
        return summary
    if isinstance(value, (tuple, list)):
        return [_ltx_sampler_trace_tensor_summary(item, policy) for item in value]
    if isinstance(value, dict):
        return {str(key): _ltx_sampler_trace_tensor_summary(item, policy) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def _ltx_sampler_trace_find_nonfinite(summary, prefix=""):
    if isinstance(summary, dict) and "nonfinite_count" in summary:
        cpu_nonfinite = int(summary.get("nonfinite_count") or 0)
        device_nonfinite = int(summary.get("device_nonfinite_count") or 0)
        if cpu_nonfinite or device_nonfinite:
            return {
                "path": prefix or "tensor",
                "nonfinite_count": cpu_nonfinite,
                "device_nonfinite_count": device_nonfinite,
                "nan_count": int(summary.get("nan_count") or 0),
                "device_nan_count": int(summary.get("device_nan_count") or 0),
                "inf_count": int(summary.get("inf_count") or 0),
                "device_inf_count": int(summary.get("device_inf_count") or 0),
                "shape": summary.get("shape"),
                "dtype": summary.get("dtype"),
                "device": summary.get("device"),
            }
        return None
    if isinstance(summary, list):
        for i, item in enumerate(summary):
            found = _ltx_sampler_trace_find_nonfinite(item, f"{prefix}[{i}]" if prefix else f"[{i}]")
            if found is not None:
                return found
    if isinstance(summary, dict):
        for key, item in summary.items():
            found = _ltx_sampler_trace_find_nonfinite(item, f"{prefix}.{key}" if prefix else str(key))
            if found is not None:
                return found
    return None


def _ltx_sampler_trace_append(path, event):
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception as exc:
        logging.warning("LTX sampler finite trace failed to write %s: %s", path, exc)


def _ltx_sampler_trace_event(model_options, boundary, tensors=None, metadata=None, model_call_start=False):
    config = _ltx_sampler_trace_config(model_options)
    if config is None:
        return
    state = _ltx_sampler_trace_state(config)
    if state.get("stopped"):
        raise RuntimeError(_LTX_SAMPLER_TRACE_STOP_REASON)

    if model_call_start:
        state["current_model_call"] = state["model_call_count"]
        state["model_call_count"] += 1

    policy = _ltx_sampler_trace_event_policy(config, model_options, metadata)
    sync_isolation_action = str(config.get("sync_isolation_action") or "").strip().lower()
    if policy == "sync_only" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        policy = "shape_only"
    summaries = _ltx_sampler_trace_tensor_summary(tensors or {}, policy)
    first_nonfinite = _ltx_sampler_trace_find_nonfinite(summaries, "tensors")
    max_model_calls = int(config.get("max_model_calls") or 0)
    max_reached = bool(max_model_calls > 0 and state["model_call_count"] >= max_model_calls and boundary.endswith(".end"))
    stop = bool(first_nonfinite is not None or max_reached)
    event = {
        "event_type": "ltx_sampler_finite_trace",
        "boundary": boundary,
        "event_index": state["event_count"],
        "model_call_index": state.get("current_model_call"),
        "model_call_count": state["model_call_count"],
        "timestamp": time.time(),
        "metadata": metadata or {},
        "tensors": summaries,
        "first_nonfinite": first_nonfinite,
        "diagnostic_stop": stop,
        "summary_policy": policy,
    }
    if config.get("mode") == "sync_isolation_trace":
        event["sync_isolation_action"] = sync_isolation_action or "log_only"
    if stop:
        event["diagnostic_stop_reason"] = _LTX_SAMPLER_TRACE_STOP_REASON
        if max_reached and first_nonfinite is None:
            event["stop_detail"] = "max_model_calls_reached"
    _ltx_sampler_trace_append(config["event_path"], event)
    state["event_count"] += 1
    if stop:
        state["stopped"] = True
        raise RuntimeError(_LTX_SAMPLER_TRACE_STOP_REASON)


def add_area_dims(area, num_dims):
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area

def get_area_and_mult(conds, x_in, timestep_in):
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = list(conds['area'])
        area = add_area_dims(area, len(dims))
        if (len(area) // 2) > len(dims):
            area = area[:len(dims)] + area[len(area) // 2:(len(area) // 2) + len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        # assert (mask.shape[1:] == x_in.shape[2:])

        mask = mask[:input_x.shape[0]]
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat((input_x.shape[0] // mask.shape[0], input_x.shape[1]) + (1, ) * (mask.ndim - 1))
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds and area is not None:
        fuzz = 8
        for i in range(len(dims)):
            rr = min(fuzz, mult.shape[2 + i] // 4)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0 / rr) * (t + 1))
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], area=area)

    hooks = conds.get('hooks', None)
    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches', 'uuid', 'hooks'])
    return cond_obj(input_x, mult, conditioning, area, control, patches, conds['uuid'], hooks)

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def finalize_default_conds(model: 'BaseModel', hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]], default_conds: list[list[dict]], x_in, timestep, model_options):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for _ in default_conds:
        default_mults.append(torch.ones_like(x_in))
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            if area is not None:
                curr_default_mult: torch.Tensor = default_mults[i]
                dims = len(area) // 2
                for i in range(dims):
                    curr_default_mult = curr_default_mult.narrow(i + 2, area[i + dims], area[i])
                curr_default_mult -= cond_obj.mult
            else:
                default_mults[i] -= cond_obj.mult
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # do get_area_and_mult to get all the expected values
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue
            # replace p's mult with calculated mult
            p = p._replace(mult=mult)
            if p.hooks is not None:
                model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            hooked_to_run.setdefault(p.hooks, list())
            hooked_to_run[p.hooks] += [(p, i)]

def calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options: dict[str]):
    handler: comfy.context_windows.ContextHandlerABC = model_options.get("context_handler", None)
    if handler is None or not handler.should_use_context(model, conds, x_in, timestep, model_options):
        return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
    return handler.execute(_calc_cond_batch_outer, model, conds, x_in, timestep, model_options)

def _calc_cond_batch_outer(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _calc_cond_batch,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)

def _calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    _ltx_sampler_trace_event(
        model_options,
        "calc_cond_batch.start",
        {"x_in": x_in, "timestep": timestep},
        {"cond_count": len(conds)},
    )
    out_conds = []
    out_counts = []
    # separate conds by matching hooks
    hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]] = {}
    default_conds = []
    has_default_conds = False

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        default_c = []
        if cond is not None:
            for x in cond:
                if 'default' in x:
                    default_c.append(x)
                    has_default_conds = True
                    continue
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                if p.hooks is not None:
                    model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
                hooked_to_run.setdefault(p.hooks, list())
                hooked_to_run[p.hooks] += [(p, i)]
        default_conds.append(default_c)

    if has_default_conds:
        finalize_default_conds(model, hooked_to_run, default_conds, x_in, timestep, model_options)

    model.current_patcher.prepare_state(timestep)

    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = model.current_patcher.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                cond_shapes = collections.defaultdict(list)
                for tt in batch_amount:
                    for k, v in to_run[tt][0].conditioning.items():
                        cond_shapes[k].append(v.size())

                if model.memory_required(input_shape, cond_shapes=cond_shapes) * 1.5 < free_memory:
                    to_batch = batch_amount
                    break

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            uuids = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                mult.append(p.mult)
                c.append(p.conditioning)
                area.append(p.area)
                cond_or_uncond.append(o[1])
                uuids.append(p.uuid)
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            transformer_options = model.current_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                transformer_options = comfy.patcher_extension.merge_nested_dicts(transformer_options,
                                                                                 model_options['transformer_options'],
                                                                                 copy_dict1=False)
            if _LTX_SAMPLER_TRACE_KEY in model_options:
                transformer_options[_LTX_SAMPLER_TRACE_KEY] = model_options[_LTX_SAMPLER_TRACE_KEY]

            if patches is not None:
                transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options.get("patches", {}),
                    patches
                )

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            transformer_options["uuids"] = uuids[:]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond), transformer_options)

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond, "cond_scale": model_options.get("current_cfg_scale", None)}).chunk(batch_chunks)
            else:
                raw_output = model.apply_model(input_x, timestep_, **c)
                _ltx_sampler_trace_event(
                    model_options,
                    "calc_cond_batch.apply_model_output",
                    {"input_x": input_x, "timestep": timestep_, "raw_output": raw_output},
                    {"batch_chunks": batch_chunks, "cond_or_uncond": cond_or_uncond[:]},
                )
                output = raw_output.chunk(batch_chunks)

            _ltx_sampler_trace_event(
                model_options,
                "calc_cond_batch.output_chunks",
                {"chunks": output},
                {"batch_chunks": batch_chunks, "cond_or_uncond": cond_or_uncond[:]},
            )

            for o in range(batch_chunks):
                cond_index = cond_or_uncond[o]
                a = area[o]
                if a is None:
                    out_conds[cond_index] += output[o] * mult[o]
                    out_counts[cond_index] += mult[o]
                else:
                    out_c = out_conds[cond_index]
                    out_cts = out_counts[cond_index]
                    dims = len(a) // 2
                    for i in range(dims):
                        out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                        out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                    out_c += output[o] * mult[o]
                    out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    _ltx_sampler_trace_event(
        model_options,
        "calc_cond_batch.end",
        {"out_conds": out_conds, "out_counts": out_counts},
        {"cond_count": len(conds)},
    )
    return out_conds

def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options): #TODO: remove
    logging.warning("WARNING: The comfy.samplers.calc_cond_uncond_batch function is deprecated please use the calc_cond_batch one instead.")
    return tuple(calc_cond_batch(model, [cond, uncond], x_in, timestep, model_options))

def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    _ltx_sampler_trace_event(
        model_options,
        "cfg_function.start",
        {"x": x, "cond_pred": cond_pred, "uncond_pred": uncond_pred, "timestep": timestep},
        {"cond_scale": float(cond_scale) if isinstance(cond_scale, (float, int)) else str(cond_scale)},
    )
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "input_cond": cond, "input_uncond": uncond}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    _ltx_sampler_trace_event(
        model_options,
        "cfg_function.end",
        {"x": x, "cond_pred": cond_pred, "uncond_pred": uncond_pred, "cfg_result": cfg_result, "timestep": timestep},
        {"cond_scale": float(cond_scale) if isinstance(cond_scale, (float, int)) else str(cond_scale)},
    )
    return cfg_result

#The main sampling function shared by all the samplers
#Returns denoised
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    model_options = dict(model_options)
    model_options["current_cfg_scale"] = cond_scale
    _ltx_sampler_trace_event(
        model_options,
        "sampling_function.start",
        {"x": x, "timestep": timestep},
        {"cond_scale": float(cond_scale) if isinstance(cond_scale, (float, int)) else str(cond_scale), "seed": seed},
        model_call_start=True,
    )
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)

    _ltx_sampler_trace_event(
        model_options,
        "sampling_function.cond_predictions",
        {"x": x, "timestep": timestep, "cond_pred": out[0], "uncond_pred": out[1]},
        {"cond_scale": float(cond_scale) if isinstance(cond_scale, (float, int)) else str(cond_scale)},
    )
    result = cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)
    _ltx_sampler_trace_event(
        model_options,
        "sampling_function.end",
        {"x": x, "timestep": timestep, "denoised": result},
        {"cond_scale": float(cond_scale) if isinstance(cond_scale, (float, int)) else str(cond_scale)},
    )
    return result


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image) * latent_mask
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)

def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(float(s.sigma(ts)))

    if append_zero:
        sigs += [0.0]

    return torch.FloatTensor(sigs)

# Implemented based on: https://arxiv.org/abs/2407.12173
def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(model_sampling.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)

# from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule(model_sampling, steps, threshold_noise=0.025, linear_steps=None):
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const
            for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * model_sampling.sigma_max.cpu()

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas

def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                if mask.ndim < 4:
                    mask = comfy.utils.common_upscale(mask.unsqueeze(1), dims[-1], dims[-2], 'bilinear', 'none').squeeze(1)
                else:
                    mask = comfy.utils.common_upscale(mask, dims[-1], dims[-2], 'bilinear', 'none')

            if modified.get("set_area_to_bounds", False): #TODO: handle dim != 2
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified

def resolve_areas_and_cond_masks(conditions, h, w, device):
    logging.warning("WARNING: The comfy.samplers.resolve_areas_and_cond_masks function is deprecated please use the resolve_areas_and_cond_masks_multidim one instead.")
    return resolve_areas_and_cond_masks_multidim(conditions, [h, w], device)

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    def area_inside(a, area_cmp):
        a = add_area_dims(a, len(area_cmp) // 2)
        area_cmp = add_area_dims(area_cmp, len(a) // 2)

        a_l = len(a) // 2
        area_cmp_l = len(area_cmp) // 2
        for i in range(min(a_l, area_cmp_l)):
            if a[a_l + i] < area_cmp[area_cmp_l + i]:
                return False
        for i in range(min(a_l, area_cmp_l)):
            if (a[i] + a[a_l + i]) > (area_cmp[i] + area_cmp[area_cmp_l + i]):
                return False
        return True

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if area_inside(c_area, a):
                if smallest is None:
                    smallest = x
                elif 'area' not in smallest:
                    smallest = x
                else:
                    if math.prod(smallest['area'][:len(smallest['area']) // 2]) > math.prod(a[:len(a) // 2]):
                        smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]

def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        # handle clip hook schedule, if needed
        if 'clip_start_percent' in x:
            timestep_start = s.percent_to_sigma(max(x['clip_start_percent'], x.get('start_percent', 0.0)))
            timestep_end = s.percent_to_sigma(min(x['clip_end_percent'], x.get('end_percent', 1.0)))
        else:
            if 'start_percent' in x:
                timestep_start = s.percent_to_sigma(x['start_percent'])
            if 'end_percent' in x:
                timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n

def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n

def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4: #TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
                  "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"]

class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_options = extra_args.get("model_options", {})
        _ltx_sampler_trace_event(
            model_options,
            "ksampler.sample.start",
            {"noise": noise, "latent_image": latent_image, "sigmas": sigmas, "denoise_mask": denoise_mask},
            {"sampler_function": getattr(self.sampler_function, "__name__", type(self.sampler_function).__name__)},
        )
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))
        _ltx_sampler_trace_event(
            model_options,
            "ksampler.sample.scaled_noise",
            {"noise": noise, "latent_image": latent_image, "sigmas": sigmas},
        )

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            def k_callback(x):
                i = int(x.get("i", -1))
                _ltx_sampler_trace_event(
                    model_options,
                    "ksampler.callback",
                    {"denoised": x.get("denoised"), "x": x.get("x")},
                    {
                        "step": i,
                        "total_steps": total_steps,
                        "sigma": float(sigmas[i].detach().cpu().item()) if 0 <= i < len(sigmas) else None,
                        "next_sigma": float(sigmas[i + 1].detach().cpu().item()) if 0 <= i + 1 < len(sigmas) else None,
                    },
                )
                callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        _ltx_sampler_trace_event(
            model_options,
            "ksampler.sample.samples_before_inverse",
            {"samples": samples},
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        _ltx_sampler_trace_event(
            model_options,
            "ksampler.sample.end",
            {"samples": samples},
        )
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None, latent_shapes=None):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)

    #make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        for c in conds[k]:
            if 'hooks' in c:
                for hook in c['hooks'].hooks:
                    hook.initialize_timesteps(model)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), conds[k], 'control', lambda cond_cnets, x: cond_cnets[x])
                apply_empty_x_to_equal_area(positive, conds[k], 'gligen', lambda cond_cnets, x: cond_cnets[x])

    return conds


def preprocess_conds_hooks(conds: dict[str, list[dict[str]]]):
    # determine which ControlNets have extra_hooks that should be combined with normal hooks
    hook_replacement: dict[tuple[ControlBase, comfy.hooks.HookGroup], list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            if 'control' in kk:
                control: 'ControlBase' = kk['control']
                extra_hooks = control.get_extra_hooks()
                if len(extra_hooks) > 0:
                    hooks: comfy.hooks.HookGroup = kk.get('hooks', None)
                    to_replace = hook_replacement.setdefault((control, hooks), [])
                    to_replace.append(kk)
    # if nothing to replace, do nothing
    if len(hook_replacement) == 0:
        return

    # for optimal sampling performance, common ControlNets + hook combos should have identical hooks
    # on the cond dicts
    for key, conds_to_modify in hook_replacement.items():
        control = key[0]
        hooks = key[1]
        hooks = comfy.hooks.HookGroup.combine_all_hooks(control.get_extra_hooks() + [hooks])
        # if combined hooks are not None, set as new hooks for all relevant conds
        if hooks is not None:
            for cond in conds_to_modify:
                cond['hooks'] = hooks

def filter_registered_hooks_on_conds(conds: dict[str, list[dict[str]]], model_options: dict[str]):
    '''Modify 'hooks' on conds so that only hooks that were registered remain. Properly accounts for
    HookGroups that have the same reference.'''
    registered: comfy.hooks.HookGroup = model_options.get('registered_hooks', None)
    # if None were registered, make sure all hooks are cleaned from conds
    if registered is None:
        for k in conds:
            for kk in conds[k]:
                kk.pop('hooks', None)
        return
    # find conds that contain hooks to be replaced - group by common HookGroup refs
    hook_replacement: dict[comfy.hooks.HookGroup, list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            hooks: comfy.hooks.HookGroup = kk.get('hooks', None)
            if hooks is not None:
                if not hooks.is_subset_of(registered):
                    to_replace = hook_replacement.setdefault(hooks, [])
                    to_replace.append(kk)
    # for each hook to replace, create a new proper HookGroup and assign to all common conds
    for hooks, conds_to_modify in hook_replacement.items():
        new_hooks = hooks.new_with_common_hooks(registered)
        if len(new_hooks) == 0:
            new_hooks = None
        for kk in conds_to_modify:
            kk['hooks'] = new_hooks


def get_total_hook_groups_in_conds(conds: dict[str, list[dict[str]]]):
    hooks_set = set()
    for k in conds:
        for kk in conds[k]:
            hooks_set.add(kk.get('hooks', None))
    return len(hooks_set)


def cast_to_load_options(model_options: dict[str], device=None, dtype=None):
    '''
    If any patches from hooks, wrappers, or callbacks have .to to be called, call it.
    '''
    if model_options is None:
        return
    to_load_options = model_options.get("to_load_options", None)
    if to_load_options is None:
        return

    casts = []
    if device is not None:
        casts.append(device)
    if dtype is not None:
        casts.append(dtype)
    # if nothing to apply, do nothing
    if len(casts) == 0:
        return

    # try to call .to on patches
    if "patches" in to_load_options:
        patches = to_load_options["patches"]
        for name in patches:
            patch_list = patches[name]
            for i in range(len(patch_list)):
                if hasattr(patch_list[i], "to"):
                    for cast in casts:
                        patch_list[i] = patch_list[i].to(cast)
    if "patches_replace" in to_load_options:
        patches = to_load_options["patches_replace"]
        for name in patches:
            patch_list = patches[name]
            for k in patch_list:
                if hasattr(patch_list[k], "to"):
                    for cast in casts:
                        patch_list[k] = patch_list[k].to(cast)
    # try to call .to on any wrappers/callbacks
    wrappers_and_callbacks = ["wrappers", "callbacks"]
    for wc_name in wrappers_and_callbacks:
        if wc_name in to_load_options:
            wc: dict[str, list] = to_load_options[wc_name]
            for wc_dict in wc.values():
                for wc_list in wc_dict.values():
                    for i in range(len(wc_list)):
                        if hasattr(wc_list[i], "to"):
                            for cast in casts:
                                wc_list[i] = wc_list[i].to(cast)


class CFGGuider:
    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            if self.model_patcher.is_dynamic() and comfy.sampler_helpers.cond_has_hooks(conds[k]):
                self.model_patcher = self.model_patcher.get_non_dynamic_delegate()
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.outer_predict_noise(*args, **kwargs)

    def outer_predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.predict_noise,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, self.model_options, is_model_options=True)
        ).execute(x, timestep, model_options, seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        _ltx_sampler_trace_event(
            self.model_options,
            "cfg_guider.inner_sample.start",
            {"noise": noise, "latent_image": latent_image, "sigmas": sigmas, "denoise_mask": denoise_mask},
            {"seed": seed, "device": str(device)},
        )
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)
            _ltx_sampler_trace_event(
                self.model_options,
                "cfg_guider.inner_sample.after_process_latent_in",
                {"latent_image": latent_image},
                {"seed": seed, "device": str(device)},
            )

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
        _ltx_sampler_trace_event(
            self.model_options,
            "cfg_guider.inner_sample.after_process_conds",
            {"noise": noise, "latent_image": latent_image, "conds": self.conds},
            {"seed": seed, "device": str(device)},
        )

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}
        _ltx_sampler_trace_event(
            extra_model_options,
            "cfg_guider.inner_sample.before_sampler",
            {"noise": noise, "latent_image": latent_image, "sigmas": sigmas, "denoise_mask": denoise_mask},
            {"seed": seed, "device": str(device)},
        )

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        _ltx_sampler_trace_event(
            extra_model_options,
            "cfg_guider.inner_sample.after_sampler",
            {"samples": samples},
            {"seed": seed, "device": str(device)},
        )
        out = self.inner_model.process_latent_out(samples.to(torch.float32))
        _ltx_sampler_trace_event(
            extra_model_options,
            "cfg_guider.inner_sample.end",
            {"samples": samples, "processed_out": out},
            {"seed": seed, "device": str(device)},
        )
        return out

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        noise = noise.to(device=device, dtype=torch.float32)
        latent_image = latent_image.to(device=device, dtype=torch.float32)
        sigmas = sigmas.to(device)
        _ltx_sampler_trace_event(
            self.model_options,
            "cfg_guider.outer_sample.prepared_inputs",
            {"noise": noise, "latent_image": latent_image, "sigmas": sigmas, "denoise_mask": denoise_mask},
            {"seed": seed, "device": str(device)},
        )
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        if latent_image.is_nested:
            latent_image, latent_shapes = comfy.utils.pack_latents(latent_image.unbind())
            noise, _ = comfy.utils.pack_latents(noise.unbind())
        else:
            latent_shapes = [latent_image.shape]

        if denoise_mask is not None:
            if denoise_mask.is_nested:
                denoise_masks = denoise_mask.unbind()
                denoise_masks = denoise_masks[:len(latent_shapes)]
            else:
                denoise_masks = [denoise_mask]

            for i in range(len(denoise_masks), len(latent_shapes)):
                denoise_masks.append(torch.ones(latent_shapes[i]))

            for i in range(len(denoise_masks)):
                denoise_masks[i] = comfy.sampler_helpers.prepare_mask(denoise_masks[i], latent_shapes[i], self.model_patcher.load_device)

            if len(denoise_masks) > 1:
                denoise_mask, _ = comfy.utils.pack_latents(denoise_masks)
            else:
                denoise_mask = denoise_masks[0]
            denoise_mask = denoise_mask.float()

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds

        if len(latent_shapes) > 1:
            output = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(output, latent_shapes))
        return output


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

class SchedulerHandler(NamedTuple):
    handler: Callable[..., torch.Tensor]
    # Boolean indicates whether to call the handler like:
    #  scheduler_function(model_sampling, steps) or
    #  scheduler_function(n, sigma_min: float, sigma_max: float)
    use_ms: bool = True

SCHEDULER_HANDLERS = {
    "simple": SchedulerHandler(simple_scheduler),
    "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
    "karras": SchedulerHandler(k_diffusion_sampling.get_sigmas_karras, use_ms=False),
    "exponential": SchedulerHandler(k_diffusion_sampling.get_sigmas_exponential, use_ms=False),
    "ddim_uniform": SchedulerHandler(ddim_scheduler),
    "beta": SchedulerHandler(beta_scheduler),
    "normal": SchedulerHandler(normal_scheduler),
    "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
    "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
}
SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)

def calculate_sigmas(model_sampling: object, scheduler_name: str, steps: int) -> torch.Tensor:
    handler = SCHEDULER_HANDLERS.get(scheduler_name)
    if handler is None:
        err = f"error invalid scheduler {scheduler_name}"
        logging.error(err)
        raise ValueError(err)
    if handler.use_ms:
        return handler.handler(model_sampling, steps)
    return handler.handler(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))

def sampler_object(name):
    if name == "uni_pc":
        sampler = KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
