from __future__ import annotations

import dataclasses
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.samplers
import comfy.utils
from comfy.backends.benchmark_stats import write_event


LTX_COMFY_DIFFUSION_PREFIX = "model.diffusion_model."
LTX_COMFY_LORA_PREFIX = "diffusion_model."


def remap_ltx_comfy_transformer_key(key: str) -> str | None:
    """Map a Comfy LTX transformer checkpoint key to the MLX LTX model tree."""

    if key.startswith(LTX_COMFY_DIFFUSION_PREFIX):
        key = key[len(LTX_COMFY_DIFFUSION_PREFIX) :]
    elif key.startswith(LTX_COMFY_LORA_PREFIX):
        key = key[len(LTX_COMFY_LORA_PREFIX) :]
    else:
        return None
    replacements = (
        (".to_out.0.", ".to_out."),
        (".ff.net.0.proj.", ".ff.proj_in."),
        (".ff.net.2.", ".ff.proj_out."),
        (".linear_1.", ".linear1."),
        (".linear_2.", ".linear2."),
        ("audio_ff.net.0.proj.", "audio_ff.proj_in."),
        ("audio_ff.net.2.", "audio_ff.proj_out."),
    )
    for before, after in replacements:
        key = key.replace(before, after)
    return key


class _ComfyLTXBlockLoraSource:
    """Block LoRA source for Comfy-formatted LTX LoRA files."""

    def __init__(self, lora_path: str | Path, *, strength: float) -> None:
        import mlx.core as mx

        self.strength = float(strength)
        self._lora_data = mx.load(str(lora_path))
        self._block_keys: dict[int, dict[str, dict[str, str]]] = {}
        prefix = "transformer_blocks."
        for raw_key in self._lora_data:
            model_key = remap_ltx_comfy_transformer_key(raw_key)
            if model_key is None or not model_key.startswith(prefix):
                continue
            rest = model_key[len(prefix) :]
            idx_str, _, param_path = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            for suffix, slot in ((".lora_A.weight", "a"), (".lora_B.weight", "b")):
                if param_path.endswith(suffix):
                    param_name = param_path[: -len(suffix)]
                    self._block_keys.setdefault(block_idx, {}).setdefault(param_name, {})[slot] = raw_key
                    break

    def has_block(self, block_idx: int) -> bool:
        block = self._block_keys.get(block_idx)
        if not block:
            return False
        return any("a" in slots and "b" in slots for slots in block.values())

    def get_block_lora_dict(self, block_idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        block = self._block_keys.get(block_idx, {})
        for param_name, slots in block.items():
            if "a" not in slots or "b" not in slots:
                continue
            out[f"{param_name}.lora_A.weight"] = self._lora_data[slots["a"]]
            out[f"{param_name}.lora_B.weight"] = self._lora_data[slots["b"]]
        return out


class _ComfyLTXBlockStreamer:
    """Block streamer for Comfy monolithic LTX dev checkpoints."""

    def __init__(self, weight_path: str | Path, *, allowed_keys: set[str]) -> None:
        import mlx.core as mx

        self._weight_path = str(weight_path)
        self._allowed_keys = set(allowed_keys)
        self._weights = self._reload_dict()
        self._block_key_map: dict[int, list[tuple[str, str]]] = {}
        prefix = "transformer_blocks."
        for full_key in self._weights:
            model_key = remap_ltx_comfy_transformer_key(full_key)
            if model_key is None or model_key not in self._allowed_keys or not model_key.startswith(prefix):
                continue
            rest = model_key[len(prefix) :]
            idx_str, _, param_name = rest.partition(".")
            try:
                block_idx = int(idx_str)
            except ValueError:
                continue
            self._block_key_map.setdefault(block_idx, []).append((full_key, param_name))

        if not self._block_key_map:
            raise ValueError(f"No LTX transformer block weights found in {self._weight_path}.")

    @property
    def block_count(self) -> int:
        return len(self._block_key_map)

    def block_keys(self, idx: int) -> list[str]:
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer")
        return [param_name for _full, param_name in self._block_key_map[idx]]

    def bind(
        self,
        block: Any,
        idx: int,
        evict_previous: int | None = None,
        lora_sources: list[Any] | None = None,
    ) -> None:
        if idx not in self._block_key_map:
            raise KeyError(f"block {idx} not in streamer")
        if evict_previous is not None and evict_previous in self._block_key_map:
            for full_key, _param_name in self._block_key_map[evict_previous]:
                self._weights.pop(full_key, None)
        sample_key = self._block_key_map[idx][0][0]
        if sample_key not in self._weights:
            self._weights = self._reload_dict()
        weights = [(param_name, self._weights[full_key]) for full_key, param_name in self._block_key_map[idx]]
        if lora_sources:
            from ltx_core_mlx.loader.block_streaming import BlockStreamer

            weights = BlockStreamer._fuse_lora_into_block(weights, idx, lora_sources)
        block.load_weights(weights, strict=True)

    def _reload_dict(self) -> dict[str, Any]:
        import mlx.core as mx

        return dict(mx.load(self._weight_path))

    def close(self) -> None:
        self._weights = {}
        self._block_key_map = {}


def _load_ltx_comfy_dev_lora_transformer(
    *,
    dev_checkpoint_path: str | Path,
    lora_path: str | Path | None,
    lora_strength: float,
    low_ram_streaming: bool,
) -> Any:
    """Load Comfy LTX dev BF16 weights into the MLX LTX transformer contract."""

    import mlx.core as mx
    import mlx.utils
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core_mlx.model.transformer.model import LTXModel
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import apply_quantization

    dev_path = Path(dev_checkpoint_path).expanduser()
    if not dev_path.exists():
        raise FileNotFoundError(f"LTX dev transformer checkpoint not found: {dev_path}")
    lora_file = Path(lora_path).expanduser() if lora_path else None
    if lora_file is not None and not lora_file.exists():
        raise FileNotFoundError(f"LTX distilled LoRA not found: {lora_file}")

    dit = LTXModel()
    allowed = {k for k, v in mlx.utils.tree_flatten(dit.parameters()) if hasattr(v, "shape")}
    raw = mx.load(str(dev_path))

    def mapped_weights(*, include_blocks: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for raw_key, tensor in raw.items():
            mapped = remap_ltx_comfy_transformer_key(raw_key)
            if mapped is None or mapped not in allowed:
                continue
            is_block = mapped.startswith("transformer_blocks.")
            if include_blocks or not is_block:
                out[mapped] = tensor
        return out

    def mapped_lora(*, include_blocks: bool) -> dict[str, Any]:
        if lora_file is None:
            return {}
        lora_raw = mx.load(str(lora_file))
        out: dict[str, Any] = {}
        for raw_key, tensor in lora_raw.items():
            mapped = remap_ltx_comfy_transformer_key(raw_key)
            if mapped is None:
                continue
            base = mapped
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)] + ".weight"
                    break
            if base not in allowed:
                continue
            is_block = mapped.startswith("transformer_blocks.")
            if include_blocks or not is_block:
                out[mapped] = tensor
        return out

    if low_ram_streaming:
        dit.transformer_blocks = [dit.transformer_blocks[0]]
        weights = mapped_weights(include_blocks=False)
        if lora_file is not None:
            non_block_lora = mapped_lora(include_blocks=False)
            if non_block_lora:
                weights = apply_loras(
                    StateDict(sd=weights, size=0, dtype=set()),
                    [LoraStateDictWithStrength(StateDict(sd=non_block_lora, size=0, dtype=set()), float(lora_strength))],
                ).sd
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()), strict=False)
        streamer = _ComfyLTXBlockStreamer(dev_path, allowed_keys=allowed)
        lora_sources = (
            [_ComfyLTXBlockLoraSource(lora_file, strength=float(lora_strength))]
            if lora_file is not None
            else []
        )
        from ltx_core_mlx.loader.block_streaming import StreamingLTXModel

        dit = StreamingLTXModel(dit, streamer, lora_sources=lora_sources)
    else:
        weights = mapped_weights(include_blocks=True)
        if lora_file is not None:
            lora_weights = mapped_lora(include_blocks=True)
            if lora_weights:
                weights = apply_loras(
                    StateDict(sd=weights, size=0, dtype=set()),
                    [LoraStateDictWithStrength(StateDict(sd=lora_weights, size=0, dtype=set()), float(lora_strength))],
                ).sd
        missing = sorted(allowed.difference(weights))
        if missing:
            raise RuntimeError(
                "Comfy LTX dev checkpoint did not provide all MLX transformer weights; "
                f"missing {len(missing)} keys, first={missing[:5]}"
            )
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))

    mx.eval(dit.parameters())
    aggressive_cleanup()
    return dit


class UnsupportedIslandRequest(RuntimeError):
    """Raised when a Comfy sampler call cannot safely enter an MLX island."""


@dataclass
class ConversionCounter:
    torch_to_mlx: int = 0
    mlx_to_torch: int = 0
    torch_to_cpu: int = 0
    torch_to_mlx_seconds: float = 0.0
    mlx_to_torch_seconds: float = 0.0
    torch_to_cpu_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return dataclasses.asdict(self)


@dataclass
class MLXRuntimeCaches:
    context: dict[Any, Any] = field(default_factory=dict)
    shape_layout: dict[Any, Any] = field(default_factory=dict)
    weights: dict[Any, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.context.clear()
        self.shape_layout.clear()
        self.weights.clear()


@dataclass
class MLXMemoryPolicy:
    cache_limit_bytes: int | None = 0
    wired_limit_bytes: int | None = 0
    clear_cache_on_cleanup: bool = True

    def apply(self, mx: Any) -> None:
        if self.cache_limit_bytes is not None and hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(int(self.cache_limit_bytes))
        if self.wired_limit_bytes is not None and hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(int(self.wired_limit_bytes))

    def cleanup(self, mx: Any | None) -> None:
        if mx is None or not self.clear_cache_on_cleanup:
            return
        used_top_level_clear = False
        try:
            mx.clear_cache()
            used_top_level_clear = True
        except Exception:
            logging.debug("MLX cache cleanup failed.", exc_info=True)
        if used_top_level_clear:
            return
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "clear_cache"):
            try:
                metal.clear_cache()
            except Exception:
                logging.debug("MLX metal cache cleanup failed.", exc_info=True)


class TensorBridge:
    """Torch <-> MLX bridge for coarse island boundaries only."""

    def __init__(self, require_mlx: bool = False):
        self.require_mlx = bool(require_mlx)
        self.counter = ConversionCounter()
        self._mx: Any | None = None

    @property
    def mx(self) -> Any | None:
        if self._mx is None:
            try:
                import mlx.core as mx

                self._mx = mx
            except Exception:
                if self.require_mlx:
                    raise
                return None
        return self._mx

    def torch_to_mlx(self, tensor: torch.Tensor, dtype: Any | None = None) -> Any:
        mx = self.mx
        if mx is None:
            return tensor.detach().to("cpu")
        self.counter.torch_to_cpu += 1
        start_cpu = time.perf_counter()
        cpu_tensor = tensor.detach().to("cpu")
        self.counter.torch_to_cpu_seconds += time.perf_counter() - start_cpu
        if cpu_tensor.dtype is torch.bfloat16:
            bridge = os.environ.get("COMFY_MLX_ISLAND_TORCH_INPUT_BRIDGE", "numpy").strip().lower()
            if bridge == "torch":
                self.counter.torch_to_mlx += 1
                start_mlx = time.perf_counter()
                try:
                    out = mx.array(cpu_tensor)
                    if dtype is not None and out.dtype != dtype:
                        out = out.astype(dtype)
                    self.counter.torch_to_mlx_seconds += time.perf_counter() - start_mlx
                    return out
                except Exception:
                    self.counter.torch_to_mlx_seconds += time.perf_counter() - start_mlx
            cpu_tensor = cpu_tensor.to(torch.float32)
        elif os.environ.get("COMFY_MLX_ISLAND_TORCH_INPUT_BRIDGE", "numpy").strip().lower() == "torch":
            self.counter.torch_to_mlx += 1
            start_mlx = time.perf_counter()
            try:
                out = mx.array(cpu_tensor)
                if dtype is not None and out.dtype != dtype:
                    out = out.astype(dtype)
                self.counter.torch_to_mlx_seconds += time.perf_counter() - start_mlx
                return out
            except Exception:
                self.counter.torch_to_mlx_seconds += time.perf_counter() - start_mlx
        self.counter.torch_to_mlx += 1
        start_mlx = time.perf_counter()
        out = mx.array(cpu_tensor.numpy())
        if dtype is not None:
            out = out.astype(dtype)
        self.counter.torch_to_mlx_seconds += time.perf_counter() - start_mlx
        return out

    def mlx_to_torch(self, value: Any, *, like: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        mx = self.mx
        if mx is not None and hasattr(value, "astype"):
            self.counter.mlx_to_torch += 1
            start = time.perf_counter()
            bridge = os.environ.get("COMFY_MLX_ISLAND_TORCH_OUTPUT_BRIDGE", "numpy").strip().lower()
            if bridge == "dlpack":
                try:
                    value = torch.from_dlpack(value.astype(mx.float32))
                except Exception:
                    import numpy as np

                    value = np.asarray(value.astype(mx.float32))
            else:
                import numpy as np

                value = np.asarray(value.astype(mx.float32))
            self.counter.mlx_to_torch_seconds += time.perf_counter() - start
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(dtype=dtype or like.dtype)
        else:
            tensor = torch.from_numpy(value).to(dtype=dtype or like.dtype)
        return tensor.to(device=like.device)

    def zeros_like_model_output(self, tensor: torch.Tensor) -> Any:
        mx = self.mx
        if mx is None:
            return torch.zeros_like(tensor, device="cpu")
        model_input = self.torch_to_mlx(tensor)
        return mx.zeros_like(model_input)

    def cleanup(self, policy: MLXMemoryPolicy | None = None) -> None:
        if policy is not None:
            policy.cleanup(self.mx)


@dataclass
class MLXDenoiserCall:
    model_input: Any
    sigma: torch.Tensor
    timestep: torch.Tensor
    original_input: torch.Tensor
    context: Any | None
    control: Any | None
    transformer_options: dict[str, Any]
    extra_conds: dict[str, Any]
    cond_or_uncond: list[int] | None


@dataclass(frozen=True)
class MLXIslandAdapterContract:
    runtime: str
    model_family: str
    precision_policy: dict[str, Any]
    shared_contract_version: int = 1
    supports_shape_census: bool = True
    supports_dit_census: bool = False
    owns_transformer_weights: bool = True
    managed_torch_models: int = 0
    prompt_state_policy: str = "none"
    supported_features: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()
    required_methods: tuple[str, ...] = (
        "load_weights",
        "prepare_context",
        "pack_latents",
        "unpack_latents",
        "map_timestep",
        "forward_model_output",
        "shape_census",
        "precision_policy",
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "model_family": self.model_family,
            "precision_policy": self.precision_policy,
            "shared_contract_version": self.shared_contract_version,
            "supports_shape_census": self.supports_shape_census,
            "supports_dit_census": self.supports_dit_census,
            "owns_transformer_weights": self.owns_transformer_weights,
            "managed_torch_models": self.managed_torch_models,
            "prompt_state_policy": self.prompt_state_policy,
            "supported_features": list(self.supported_features),
            "unsupported_features": list(self.unsupported_features),
            "required_methods": list(self.required_methods),
        }


class BaseMLXDenoiserIslandRuntime:
    """Reusable runtime contract for model-specific MLX denoiser islands."""

    name = "base"
    prompt_state_policy = "none"
    supported_feature_names: tuple[str, ...] = ("lightweight_model_proxy", "model_function_wrapper")
    unsupported_extra_cond_keys: tuple[str, ...] = ()
    unsupported_feature_names: tuple[str, ...] = (
        "controlnet",
        "lora_model_patches",
        "transformer_patch_hooks",
        "context_windows",
    )

    def __init__(self, *, bridge: TensorBridge | None = None, memory_policy: MLXMemoryPolicy | None = None):
        self.bridge = bridge if bridge is not None else TensorBridge(require_mlx=False)
        self.memory_policy = memory_policy if memory_policy is not None else MLXMemoryPolicy()
        self.caches = MLXRuntimeCaches()
        self.calls = 0
        mx = self.bridge.mx
        if mx is not None:
            self.memory_policy.apply(mx)

    def load_weights(self) -> Any:
        return None

    def prepare_context(self, call: MLXDenoiserCall) -> Any:
        return call.context

    def pack_latents(self, call: MLXDenoiserCall) -> Any:
        return call.model_input

    def unpack_latents(self, value: Any, *, like: Any) -> Any:
        return value

    def map_timestep(self, call: MLXDenoiserCall) -> dict[str, Any]:
        return {
            "sigma_shape": _shape_summary(call.sigma),
            "timestep_shape": _shape_summary(call.timestep),
            "cond_or_uncond": call.cond_or_uncond,
        }

    def precision_policy(self) -> dict[str, Any]:
        return {"default": "unknown", "available": ["unknown"]}

    def shape_census(self, call: MLXDenoiserCall | None = None) -> dict[str, Any]:
        return {
            "runtime": self.name,
            "model_family": self.name,
            "precision_policy": self.precision_policy(),
            "prompt_state": self.prompt_state_report(),
            "model_input_shape": _shape_summary(call.model_input) if call is not None else None,
            "context_shape": _shape_summary(call.context) if call is not None else None,
            "timestep": self.map_timestep(call) if call is not None else None,
        }

    def adapter_contract(self) -> MLXIslandAdapterContract:
        return MLXIslandAdapterContract(
            runtime=self.name,
            model_family=self.name,
            precision_policy=self.precision_policy(),
            prompt_state_policy=self.prompt_state_policy,
            supported_features=self.supported_features(),
            unsupported_features=self.unsupported_features(),
        )

    def supported_features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.supported_feature_names))

    def unsupported_features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.unsupported_feature_names, *self.unsupported_extra_cond_keys)))

    def prompt_state_report(self) -> dict[str, Any]:
        return {
            "policy": self.prompt_state_policy,
            "scope": "none" if self.prompt_state_policy == "none" else "prompt_or_call",
            "global_prompt_cache": False,
        }

    def validate_extra_conds(self, extra_conds: dict[str, Any]) -> None:
        unsupported = sorted(key for key in self.unsupported_extra_cond_keys if extra_conds.get(key) is not None)
        if unsupported:
            raise UnsupportedIslandRequest(
                f"{self.name} MLX island does not support extra conditioning: " + ", ".join(unsupported)
            )

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        raise NotImplementedError("This MLX island runtime has no denoiser implementation yet.")

    def cleanup(self) -> None:
        self.bridge.cleanup(self.memory_policy)


class ContractMLXDenoiserRuntime(BaseMLXDenoiserIslandRuntime):
    """Shape/device contract runtime used to prove the Comfy island boundary.

    It intentionally returns a zero model prediction. This is not a generation
    implementation; it verifies that Comfy can call an MLX-owned island without
    touching a Torch transformer or per-op conversions.
    """

    name = "contract"

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        self.calls += 1
        zeros = self.bridge.zeros_like_model_output(call.original_input)
        return self.bridge.mlx_to_torch(zeros, like=call.original_input)


@dataclass
class LTXAVPromptState:
    video_tokens: Any
    audio_tokens: Any
    sigma: Any
    video_text: Any
    audio_text: Any
    video_positions: Any
    audio_positions: Any
    video_mask_tokens: Any | None
    audio_mask_tokens: Any | None
    video_timesteps: Any | None
    audio_timesteps: Any | None
    video_attention_mask: Any | None
    guide_metadata: dict[str, Any] | None
    video_shape: tuple[int, int, int]
    audio_T: int
    frame_rate: float
    batch_size: int

    def shape_bucket(self) -> list[int | float]:
        F, H, W = self.video_shape
        return [int(F), int(H), int(W), int(self.audio_T), float(self.frame_rate)]

    def to_report(self, runtime: "LTXAVIslandRuntime") -> dict[str, Any]:
        return {
            "policy": runtime.prompt_state_policy,
            "scope": "prompt_local_call",
            "global_prompt_cache": False,
            "shape_bucket": self.shape_bucket(),
            "video_tokens_shape": list(self.video_tokens.shape),
            "audio_tokens_shape": list(self.audio_tokens.shape),
            "context_video_shape": list(self.video_text.shape),
            "context_audio_shape": list(self.audio_text.shape),
            "video_positions_shape": list(self.video_positions.shape),
            "audio_positions_shape": list(self.audio_positions.shape),
            "video_denoise_mask": runtime._mask_stats(self.video_mask_tokens),
            "audio_denoise_mask": runtime._mask_stats(self.audio_mask_tokens),
            "video_timesteps_shape": _shape_summary(self.video_timesteps),
            "audio_timesteps_shape": _shape_summary(self.audio_timesteps),
            "video_attention_mask": runtime._attention_mask_stats(self.video_attention_mask),
            "guide_metadata": self.guide_metadata,
        }


class LTXAVIslandRuntime(BaseMLXDenoiserIslandRuntime):
    """LTX-specific MLX runtime for one Comfy denoiser prediction."""

    name = "ltx_av"
    prompt_state_policy = "ltx_av_prompt_local_static_state"
    supported_feature_names = (
        "lightweight_model_proxy",
        "model_function_wrapper",
        "native_comfy_nodes",
        "mlx_transformer_weights",
        "av_latents",
        "video_audio_text_split",
        "video_audio_positions",
        "video_audio_denoise_masks",
        "keyframe_guides",
        "sparse_guide_attention",
        "manual_sigmas",
        "two_stage_ltx_workflow",
        "shape_census",
    )
    unsupported_extra_cond_keys = (
        "ref_audio",
    )
    unsupported_feature_names = BaseMLXDenoiserIslandRuntime.unsupported_feature_names + (
        "torch_lora_runtime_patches",
        "context_windows",
        "reference_audio_guidance",
    )

    def __init__(
        self,
        model_path: str,
        *,
        pipeline: str = "distilled",
        dev_checkpoint_path: str | None = None,
        distilled_lora_path: str | None = None,
        distilled_lora_strength: float = 1.0,
        low_memory: bool = True,
        low_ram_streaming: bool = False,
        default_frame_rate: float = 24.0,
        context_video_dim: int = 4096,
        context_audio_dim: int = 2048,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_path = str(Path(model_path).expanduser())
        self.pipeline = str(pipeline or "distilled").strip().lower()
        self.dev_checkpoint_path = str(Path(dev_checkpoint_path).expanduser()) if dev_checkpoint_path else None
        self.distilled_lora_path = str(Path(distilled_lora_path).expanduser()) if distilled_lora_path else None
        self.distilled_lora_strength = float(distilled_lora_strength)
        self.low_memory = bool(low_memory)
        self.low_ram_streaming = bool(low_ram_streaming)
        self.default_frame_rate = float(default_frame_rate)
        self.context_video_dim = int(context_video_dim)
        self.context_audio_dim = int(context_audio_dim)
        self._pipeline: Any | None = None
        self._x0_model_cache: dict[tuple[int, int, int], Any] = {}
        self.last_event: dict[str, Any] | None = None

    def load_weights(self) -> Any:
        return self._pipeline_obj()

    def precision_policy(self) -> dict[str, Any]:
        path = self.model_path.lower()
        if "q4" in path:
            default = "q4"
        elif "q8" in path:
            default = "q8"
        else:
            default = "bf16"
        return {
            "default": default,
            "available": ["bf16", "q8", "q4"],
            "activation_dtype": "bfloat16",
            "weight_owner": "mlx",
            "pipeline": self.pipeline,
            "internal_lora": bool(self.distilled_lora_path),
            "internal_lora_strength": self.distilled_lora_strength if self.distilled_lora_path else None,
        }

    def adapter_contract(self) -> MLXIslandAdapterContract:
        return MLXIslandAdapterContract(
            runtime=self.name,
            model_family="ltx_av",
            precision_policy=self.precision_policy(),
            supports_dit_census=True,
            prompt_state_policy=self.prompt_state_policy,
            supported_features=self.supported_features(),
            unsupported_features=self.unsupported_features(),
        )

    def shape_census(self, call: MLXDenoiserCall | None = None) -> dict[str, Any]:
        out = {
            "runtime": self.name,
            "model_family": "ltx_av",
            "model_path": self.model_path,
            "pipeline": self.pipeline,
            "dev_checkpoint_path": self.dev_checkpoint_path,
            "distilled_lora_path": self.distilled_lora_path,
            "distilled_lora_strength": self.distilled_lora_strength if self.distilled_lora_path else None,
            "precision_policy": self.precision_policy(),
            "supports_dit_census": True,
            "prompt_state": self.prompt_state_report(),
            "supported_features": list(self.supported_features()),
            "unsupported_features": list(self.unsupported_features()),
        }
        if call is None:
            return out
        video_torch, audio_torch = split_ltx_av_latents(call.model_input)
        video_shape = list(video_torch.shape)
        audio_shape = list(audio_torch.shape)
        frame_rate = _extract_frame_rate(call.extra_conds, self.default_frame_rate)
        latent_bucket = (
            int(video_torch.shape[2]),
            int(video_torch.shape[3]),
            int(video_torch.shape[4]),
            int(audio_torch.shape[2]) if audio_torch.ndim == 4 else 0,
            float(frame_rate),
            int(video_torch.shape[0]),
        )
        out.update(
            {
                "model_input_shape": [video_shape, audio_shape],
                "context_shape": _shape_summary(call.context),
                "latent_layout": {
                    "video": "B,C,F,H,W",
                    "audio": "B,8,T,16",
                    "packed_by_wrapper": True,
                },
                "shape_bucket": [
                    latent_bucket[0],
                    latent_bucket[1],
                    latent_bucket[2],
                    latent_bucket[3],
                    latent_bucket[4],
                ],
                "token_counts": {
                    "video": int(video_torch.shape[2] * video_torch.shape[3] * video_torch.shape[4]) if video_torch.ndim == 5 else None,
                    "audio": int(audio_torch.shape[2]) if audio_torch.ndim == 4 else None,
                },
                "mask_metadata": {
                    "video": _torch_mask_metadata(call.extra_conds.get("denoise_mask")),
                    "audio": _torch_mask_metadata(call.extra_conds.get("audio_denoise_mask")),
                },
                "x0_model_cache": {
                    "shape_key": [latent_bucket[0], latent_bucket[1], latent_bucket[2]],
                    "hit_for_shape": (latent_bucket[0], latent_bucket[1], latent_bucket[2]) in self._x0_model_cache,
                    "cached_shape_keys": [list(key) for key in sorted(self._x0_model_cache)],
                },
                "timestep": self.map_timestep(call),
            }
        )
        return out

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for LTXAVIslandRuntime.")
        self.validate_extra_conds(call.extra_conds)

        video_torch, audio_torch = split_ltx_av_latents(call.model_input)
        if audio_torch.numel() == 0:
            raise UnsupportedIslandRequest("LTX AV MLX denoiser island requires an audio latent tensor.")
        if video_torch.ndim != 5 or audio_torch.ndim != 4:
            raise UnsupportedIslandRequest(
                f"Expected video/audio latent ranks 5/4; got {video_torch.ndim}/{audio_torch.ndim}."
            )

        state = self.prepare_prompt_state(call, video_torch=video_torch, audio_torch=audio_torch)
        F, H, W = state.video_shape
        x0_model = self._x0_model(F, H, W)
        call_kwargs = dict(
            video_latent=state.video_tokens,
            audio_latent=state.audio_tokens,
            sigma=state.sigma,
            video_text_embeds=state.video_text,
            audio_text_embeds=state.audio_text,
            video_positions=state.video_positions,
            audio_positions=state.audio_positions,
        )
        if state.video_timesteps is not None:
            call_kwargs["video_timesteps"] = state.video_timesteps
        if state.audio_timesteps is not None:
            call_kwargs["audio_timesteps"] = state.audio_timesteps
        if state.video_attention_mask is not None:
            call_kwargs["video_attention_mask"] = state.video_attention_mask
        video_x0, audio_x0 = x0_model(**call_kwargs)
        if state.video_mask_tokens is not None:
            video_x0 = video_x0 * state.video_mask_tokens + state.video_tokens * (1.0 - state.video_mask_tokens)
        if state.audio_mask_tokens is not None:
            audio_x0 = audio_x0 * state.audio_mask_tokens + state.audio_tokens * (1.0 - state.audio_mask_tokens)
        video_velocity = _x0_to_velocity(mx, state.video_tokens, video_x0, state.sigma)
        audio_velocity = _x0_to_velocity(mx, state.audio_tokens, audio_x0, state.sigma)
        video_out = self._mlx_video_tokens_to_torch(video_velocity, state.video_shape, like=video_torch)
        audio_out = self._mlx_audio_tokens_to_torch(audio_velocity, state.audio_T, like=audio_torch)
        self.calls += 1
        state_report = state.to_report(self)
        self.last_event = {
            "runtime": self.name,
            "model_path": self.model_path,
            "shape_bucket": state.shape_bucket(),
            "video_tokens": list(state.video_tokens.shape),
            "audio_tokens": list(state.audio_tokens.shape),
            "video_denoise_mask": state_report["video_denoise_mask"],
            "audio_denoise_mask": state_report["audio_denoise_mask"],
            "video_attention_mask": state_report["video_attention_mask"],
            "guide_metadata": state_report["guide_metadata"],
            "context_video_shape": list(state.video_text.shape),
            "context_audio_shape": list(state.audio_text.shape),
            "prompt_state": state_report,
            "x0_model_cache": {
                "shape_key": [int(F), int(H), int(W)],
                "hit_for_shape": (int(F), int(H), int(W)) in self._x0_model_cache,
                "cached_shape_keys": [list(key) for key in sorted(self._x0_model_cache)],
            },
            "conversions": self.bridge.counter.as_dict(),
            "memory": _mlx_memory_snapshot(mx),
        }
        return [video_out, audio_out]

    def prompt_state_report(self) -> dict[str, Any]:
        return {
            "policy": self.prompt_state_policy,
            "scope": "prompt_local_call",
            "global_prompt_cache": False,
            "static_fields": [
                "video_audio_context_split",
                "video_audio_positions",
                "denoise_mask_tokens",
                "shape_bucket",
                "frame_rate",
            ],
            "shape_cache_fields": ["positions", "x0_model_by_video_shape"],
        }

    def prepare_prompt_state(
        self,
        call: MLXDenoiserCall,
        *,
        video_torch: torch.Tensor | None = None,
        audio_torch: torch.Tensor | None = None,
    ) -> LTXAVPromptState:
        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for LTXAVIslandRuntime.")
        if video_torch is None or audio_torch is None:
            video_torch, audio_torch = split_ltx_av_latents(call.model_input)
        frame_rate = _extract_frame_rate(call.extra_conds, self.default_frame_rate)
        video_tokens, video_shape = self._torch_video_to_mlx_tokens(video_torch)
        audio_tokens, audio_T = self._torch_audio_to_mlx_tokens(audio_torch)
        F, H, W = video_shape
        batch_size = int(video_torch.shape[0])
        video_text, audio_text = self._split_context(call.context, batch_size)
        video_positions, audio_positions = self._positions(F, H, W, audio_T, frame_rate, batch_size)
        video_mask_tokens = self._denoise_mask_to_tokens(call.extra_conds.get("denoise_mask"), (F, H, W))
        audio_mask_tokens = self._audio_denoise_mask_to_tokens(call.extra_conds.get("audio_denoise_mask"), audio_T)
        video_attention_mask = None
        guide_metadata = None
        keyframe_positions = self._keyframe_positions_from_extra_conds(
            call.extra_conds.get("keyframe_idxs"),
            frame_rate=frame_rate,
            batch_size=batch_size,
        )
        if keyframe_positions is not None:
            self._reject_unsupported_keyframe_mask_values(video_mask_tokens)
            video_positions, position_metadata = self._apply_keyframe_positions(video_positions, keyframe_positions)
            video_attention_mask, guide_metadata = self._build_guide_attention_mask(
                call.extra_conds.get("guide_attention_entries"),
                total_tokens=int(video_tokens.shape[1]),
                guide_token_count=int(keyframe_positions.shape[1]),
                batch_size=batch_size,
            )
            guide_metadata = {**(guide_metadata or {}), **position_metadata}
        sigma = self.bridge.torch_to_mlx(call.sigma, dtype=mx.bfloat16)
        return LTXAVPromptState(
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            sigma=sigma,
            video_text=video_text,
            audio_text=audio_text,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_mask_tokens=video_mask_tokens,
            audio_mask_tokens=audio_mask_tokens,
            video_timesteps=self._masked_timesteps(sigma, video_mask_tokens),
            audio_timesteps=self._masked_timesteps(sigma, audio_mask_tokens),
            video_attention_mask=video_attention_mask,
            guide_metadata=guide_metadata,
            video_shape=video_shape,
            audio_T=audio_T,
            frame_rate=frame_rate,
            batch_size=batch_size,
        )

    def _pipeline_obj(self) -> Any:
        if self._pipeline is None:
            from ltx_pipelines_mlx.distilled import DistilledPipeline

            self._pipeline = DistilledPipeline(
                self.model_path,
                low_memory=self.low_memory,
                low_ram_streaming=self.low_ram_streaming,
            )
            if self.pipeline == "distilled":
                self._pipeline.load_transformer_only()
            elif self.pipeline == "dev_lora":
                if not self.dev_checkpoint_path:
                    raise RuntimeError("LTX dev_lora MLX island requires dev_checkpoint_path in the manifest.")
                self._pipeline.dit = _load_ltx_comfy_dev_lora_transformer(
                    dev_checkpoint_path=self.dev_checkpoint_path,
                    lora_path=self.distilled_lora_path,
                    lora_strength=self.distilled_lora_strength,
                    low_ram_streaming=self.low_ram_streaming,
                )
            else:
                raise RuntimeError(f"Unsupported LTX MLX island pipeline: {self.pipeline!r}")
        return self._pipeline

    def _x0_model(self, F: int, H: int, W: int) -> Any:
        key = (int(F), int(H), int(W))
        if key not in self._x0_model_cache:
            pipe = self._pipeline_obj()
            if hasattr(pipe, "distilled_x0_model_for_shape"):
                self._x0_model_cache[key] = pipe.distilled_x0_model_for_shape(F, H, W)
            else:
                pipe.load_transformer_only()
                self._x0_model_cache[key] = pipe._distilled_x0_model(F, H, W)
            if self.pipeline == "dev_lora":
                # _distilled_x0_model only means "plain X0 wrapper around pipe.dit";
                # the actual DiT was loaded above from dev + LoRA weights.
                self._x0_model_cache[key] = pipe._distilled_x0_model(F, H, W)
        return self._x0_model_cache[key]

    def _torch_video_to_mlx_tokens(self, video: torch.Tensor) -> tuple[Any, tuple[int, int, int]]:
        mx = self.bridge.mx
        video_mx = self.bridge.torch_to_mlx(video, dtype=mx.bfloat16)
        B, C, F, H, W = video_mx.shape
        if C != 128:
            raise UnsupportedIslandRequest(f"LTX AV video latent must have 128 channels; got {C}.")
        return video_mx.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C), (int(F), int(H), int(W))

    def _torch_audio_to_mlx_tokens(self, audio: torch.Tensor) -> tuple[Any, int]:
        mx = self.bridge.mx
        audio_mx = self.bridge.torch_to_mlx(audio, dtype=mx.bfloat16)
        B, C1, T, C2 = audio_mx.shape
        if C1 != 8 or C2 != 16:
            raise UnsupportedIslandRequest(f"LTX AV audio latent must have shape (B,8,T,16); got {tuple(audio.shape)}.")
        return audio_mx.transpose(0, 2, 1, 3).reshape(B, T, C1 * C2), int(T)

    def _denoise_mask_to_tokens(self, mask: Any, shape: tuple[int, int, int]) -> Any | None:
        if mask is None:
            return None
        mx = self.bridge.mx
        F, H, W = shape
        if not isinstance(mask, torch.Tensor) or mask.ndim != 5:
            raise UnsupportedIslandRequest(f"LTX AV video denoise mask must be rank 5; got {type(mask).__name__}.")
        mask_mx = self.bridge.torch_to_mlx(mask[:, :1], dtype=mx.bfloat16)
        if tuple(mask_mx.shape[-3:]) != (F, H, W):
            if tuple(mask_mx.shape[-3:]) == (F, 1, 1):
                mask_mx = mx.broadcast_to(mask_mx, (mask_mx.shape[0], 1, F, H, W))
            else:
                raise UnsupportedIslandRequest(
                    f"LTX AV video denoise mask shape {tuple(mask.shape)} does not match latent grid {(F, H, W)}."
                )
        return mask_mx.transpose(0, 2, 3, 4, 1).reshape(mask_mx.shape[0], F * H * W, 1)

    def _audio_denoise_mask_to_tokens(self, mask: Any, audio_T: int) -> Any | None:
        if mask is None:
            return None
        mx = self.bridge.mx
        if not isinstance(mask, torch.Tensor):
            raise UnsupportedIslandRequest(f"LTX AV audio denoise mask must be a tensor; got {type(mask).__name__}.")
        if mask.ndim == 4:
            mask_mx = self.bridge.torch_to_mlx(mask[:, :1], dtype=mx.bfloat16)
            if int(mask_mx.shape[2]) != int(audio_T):
                raise UnsupportedIslandRequest(f"LTX AV audio denoise mask length {mask_mx.shape[2]} does not match {audio_T}.")
            return mask_mx.transpose(0, 2, 1, 3).reshape(mask_mx.shape[0], audio_T, -1).mean(axis=-1, keepdims=True)
        if mask.ndim == 3:
            mask_mx = self.bridge.torch_to_mlx(mask, dtype=mx.bfloat16)
            if int(mask_mx.shape[1]) != int(audio_T):
                raise UnsupportedIslandRequest(f"LTX AV audio denoise mask length {mask_mx.shape[1]} does not match {audio_T}.")
            return mask_mx if int(mask_mx.shape[-1]) == 1 else mask_mx.mean(axis=-1, keepdims=True)
        raise UnsupportedIslandRequest(f"LTX AV audio denoise mask must be rank 3 or 4; got {tuple(mask.shape)}.")

    def _masked_timesteps(self, sigma: Any, mask_tokens: Any | None) -> Any | None:
        if mask_tokens is None:
            return None
        mx = self.bridge.mx
        if bool(mx.all(mask_tokens == 1.0).item()):
            return None
        return (mask_tokens * sigma[:, None, None].astype(mx.bfloat16)).squeeze(-1)

    def _keyframe_positions_from_extra_conds(self, keyframe_idxs: Any, *, frame_rate: float, batch_size: int) -> Any | None:
        if keyframe_idxs is None:
            return None
        mx = self.bridge.mx
        if not isinstance(keyframe_idxs, torch.Tensor):
            raise UnsupportedIslandRequest(
                f"LTX keyframe guide positions must be a tensor; got {type(keyframe_idxs).__name__}."
            )
        if keyframe_idxs.ndim != 4 or int(keyframe_idxs.shape[1]) != 3 or int(keyframe_idxs.shape[-1]) != 2:
            raise UnsupportedIslandRequest(
                "LTX keyframe guide positions must have shape (B,3,K,2); "
                f"got {tuple(keyframe_idxs.shape)}."
            )
        positions = self.bridge.torch_to_mlx(keyframe_idxs, dtype=mx.float32)
        if int(positions.shape[0]) == 1 and int(batch_size) != 1:
            positions = mx.repeat(positions, int(batch_size), axis=0)
        if int(positions.shape[0]) != int(batch_size):
            raise UnsupportedIslandRequest(
                f"LTX keyframe guide batch {positions.shape[0]} does not match latent batch {batch_size}."
            )
        midpoints = mx.mean(positions, axis=-1).transpose(0, 2, 1)
        scale = mx.array([1.0 / float(frame_rate), 1.0, 1.0], dtype=mx.float32)
        return midpoints * scale[None, None, :]

    def _apply_keyframe_positions(self, video_positions: Any, keyframe_positions: Any) -> tuple[Any, dict[str, Any]]:
        mx = self.bridge.mx
        guide_count = int(keyframe_positions.shape[1])
        total_tokens = int(video_positions.shape[1])
        if guide_count <= 0:
            return video_positions, {"guide_token_count": 0, "keyframe_positions_shape": list(keyframe_positions.shape)}
        if guide_count > total_tokens:
            raise UnsupportedIslandRequest(
                f"LTX keyframe guide count {guide_count} exceeds video token count {total_tokens}."
            )
        replaced = mx.concatenate([video_positions[:, : total_tokens - guide_count, :], keyframe_positions], axis=1)
        return replaced, {
            "guide_token_count": guide_count,
            "guide_start": total_tokens - guide_count,
            "keyframe_positions_shape": list(keyframe_positions.shape),
        }

    def _build_guide_attention_mask(
        self,
        guide_entries: Any,
        *,
        total_tokens: int,
        guide_token_count: int,
        batch_size: int,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        if guide_token_count <= 0:
            return None, None
        if guide_entries is None:
            return None, {
                "guide_entries": 0,
                "guide_token_count": int(guide_token_count),
                "sparse_guide_attention": False,
            }
        if not isinstance(guide_entries, list):
            raise UnsupportedIslandRequest(
                f"LTX guide attention entries must be a list; got {type(guide_entries).__name__}."
            )
        mx = self.bridge.mx
        weights = []
        entry_summaries = []
        counted_tokens = 0
        for index, entry in enumerate(guide_entries):
            if not isinstance(entry, dict):
                raise UnsupportedIslandRequest(
                    f"LTX guide attention entry {index} must be a dict; got {type(entry).__name__}."
                )
            if entry.get("pixel_mask") is not None:
                raise UnsupportedIslandRequest("LTX MLX island does not support guide pixel masks yet.")
            count = int(entry.get("pre_filter_count", 0) or 0)
            strength = float(entry.get("strength", 1.0))
            if count < 0:
                raise UnsupportedIslandRequest(f"LTX guide attention entry {index} has negative token count {count}.")
            if count == 0:
                continue
            counted_tokens += count
            weights.append(mx.full((int(batch_size), count), strength, dtype=mx.float32))
            entry_summaries.append(
                {
                    "pre_filter_count": count,
                    "strength": strength,
                    "latent_shape": _shape_summary(entry.get("latent_shape")),
                }
            )
        if counted_tokens != int(guide_token_count):
            raise UnsupportedIslandRequest(
                f"LTX guide attention token count {counted_tokens} does not match keyframe tokens {guide_token_count}."
            )
        if not weights:
            return None, {
                "guide_entries": len(guide_entries),
                "guide_token_count": int(guide_token_count),
                "sparse_guide_attention": False,
            }
        tracked_weights = mx.concatenate(weights, axis=1)
        from ltx_core_mlx.model.transformer.attention import GuideAttentionMask

        mask = GuideAttentionMask(
            total_tokens=int(total_tokens),
            guide_start=int(total_tokens) - int(guide_token_count),
            tracked_weights=tracked_weights,
        )
        return mask.to_tree(), {
            "guide_entries": len(guide_entries),
            "guide_entry_summaries": entry_summaries,
            "guide_token_count": int(guide_token_count),
            "guide_start": int(total_tokens) - int(guide_token_count),
            "guide_strengths": [item["strength"] for item in entry_summaries],
            "sparse_guide_attention": True,
        }

    def _reject_unsupported_keyframe_mask_values(self, video_mask_tokens: Any | None) -> None:
        if video_mask_tokens is None:
            return
        mx = self.bridge.mx
        mx.eval(video_mask_tokens)
        if float(mx.min(video_mask_tokens).item()) < 0.0:
            raise UnsupportedIslandRequest(
                "LTX MLX island does not support negative/dilated guide masks yet."
            )

    def _mask_stats(self, mask_tokens: Any | None) -> dict[str, Any] | None:
        if mask_tokens is None:
            return None
        mx = self.bridge.mx
        mx.eval(mask_tokens)
        return {
            "shape": list(mask_tokens.shape),
            "min": float(mx.min(mask_tokens).item()),
            "max": float(mx.max(mask_tokens).item()),
            "mean": float(mx.mean(mask_tokens.astype(mx.float32)).item()),
        }

    def _attention_mask_stats(self, attention_mask: Any | None) -> dict[str, Any] | None:
        if attention_mask is None:
            return None
        if isinstance(attention_mask, dict) and attention_mask.get("type") == "GuideAttentionMask":
            return {
                "type": attention_mask["type"],
                "total_tokens": int(attention_mask["total_tokens"]),
                "guide_start": int(attention_mask["guide_start"]),
                "tracked_count": int(attention_mask["tracked_count"]),
                "noisy_mask_shape": _shape_summary(attention_mask.get("noisy_mask")),
                "tracked_mask_shape": _shape_summary(attention_mask.get("tracked_mask")),
            }
        to_report = getattr(attention_mask, "to_report", None)
        if callable(to_report):
            return dict(to_report())
        return {
            "type": type(attention_mask).__name__,
            "shape": _shape_summary(attention_mask),
        }

    def _mlx_video_tokens_to_torch(self, tokens: Any, shape: tuple[int, int, int], *, like: torch.Tensor) -> torch.Tensor:
        F, H, W = shape
        B, _N, C = tokens.shape
        video = tokens.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)
        return self.bridge.mlx_to_torch(video, like=like)

    def _mlx_audio_tokens_to_torch(self, tokens: Any, T: int, *, like: torch.Tensor) -> torch.Tensor:
        B, _T, _C = tokens.shape
        audio = tokens.reshape(B, T, 8, 16).transpose(0, 2, 1, 3)
        return self.bridge.mlx_to_torch(audio, like=like)

    def _split_context(self, context: Any, batch_size: int) -> tuple[Any, Any]:
        mx = self.bridge.mx
        if context is None or not hasattr(context, "shape"):
            raise UnsupportedIslandRequest("LTX AV MLX denoiser island requires combined c_crossattn context.")
        expected = self.context_video_dim + self.context_audio_dim
        if int(context.shape[-1]) != expected:
            raise UnsupportedIslandRequest(f"Expected combined LTX AV context dim {expected}; got {int(context.shape[-1])}.")
        context_mx = self.bridge.torch_to_mlx(context, dtype=mx.bfloat16)
        if int(context_mx.shape[0]) == 1 and batch_size != 1:
            context_mx = mx.repeat(context_mx, batch_size, axis=0)
        if int(context_mx.shape[0]) != batch_size:
            raise UnsupportedIslandRequest(f"Context batch {context_mx.shape[0]} does not match latent batch {batch_size}.")
        return context_mx[:, :, : self.context_video_dim], context_mx[:, :, self.context_video_dim :]

    def _positions(self, F: int, H: int, W: int, audio_T: int, frame_rate: float, batch_size: int) -> tuple[Any, Any]:
        mx = self.bridge.mx
        key = ("positions", int(F), int(H), int(W), int(audio_T), float(frame_rate), int(batch_size))
        if key not in self.caches.shape_layout:
            from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

            video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
            audio_positions = compute_audio_positions(audio_T)
            if batch_size != 1:
                video_positions = mx.repeat(video_positions, batch_size, axis=0)
                audio_positions = mx.repeat(audio_positions, batch_size, axis=0)
            self.caches.shape_layout[key] = (video_positions, audio_positions)
        return self.caches.shape_layout[key]


class _LTXIslandSampling(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass


class _ZImageIslandSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
    pass


class ZImageIslandRuntime(BaseMLXDenoiserIslandRuntime):
    """Z-Image-specific MLX runtime shell for one Comfy denoiser prediction.

    The real transformer callable is supplied explicitly for now. This keeps
    The optional predictor remains for lightweight contract tests. When no
    predictor is supplied, this runtime loads the native Comfy Z-Image
    safetensor into an MLX-owned transformer adapter.
    """

    name = "z_image"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        context_dim: int = 2560,
        predictor: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_path = str(Path(model_path).expanduser()) if model_path else None
        self.context_dim = int(context_dim)
        self.predictor = predictor
        self._transformer: Any | None = None
        self.last_event: dict[str, Any] | None = None

    def load_weights(self) -> Any:
        return self._transformer_obj()

    def precision_policy(self) -> dict[str, Any]:
        model_path = (self.model_path or "").lower()
        if "q8" in model_path:
            default = "q8"
            available = ["q8"]
        else:
            default = "bf16"
            available = ["bf16"]
        return {
            "default": default,
            "available": available,
            "activation_dtype": "bfloat16",
            "weight_owner": "mlx",
        }

    def adapter_contract(self) -> MLXIslandAdapterContract:
        return MLXIslandAdapterContract(
            runtime=self.name,
            model_family="z_image",
            precision_policy=self.precision_policy(),
            supports_dit_census=True,
        )

    def shape_census(self, call: MLXDenoiserCall | None = None) -> dict[str, Any]:
        out = {
            "runtime": self.name,
            "model_family": "z_image",
            "model_path": self.model_path,
            "precision_policy": self.precision_policy(),
            "supports_dit_census": True,
        }
        transformer = self._transformer
        if transformer is not None:
            out["transformer"] = {
                "hidden_dim": int(getattr(transformer, "dim", 0) or 0),
                "heads": int(getattr(transformer, "heads", 0) or 0),
                "head_dim": int(getattr(transformer, "head_dim", 0) or 0),
                "patch_size": int(getattr(transformer, "patch_size", 0) or 0),
                "in_channels": int(getattr(transformer, "in_channels", 0) or 0),
            }
        if call is None:
            return out
        latent_torch = _single_z_image_latent(call.model_input)
        patch_size = int(getattr(transformer, "patch_size", 2) or 2)
        height = int(latent_torch.shape[-2])
        width = int(latent_torch.shape[-1])
        h_tokens = height // patch_size
        w_tokens = width // patch_size
        num_tokens = call.extra_conds.get("num_tokens")
        context_tokens = int(num_tokens) if isinstance(num_tokens, int) else (int(call.context.shape[1]) if hasattr(call.context, "shape") else None)
        out.update(
            {
                "model_input_shape": list(latent_torch.shape),
                "context_shape": _shape_summary(call.context),
                "latent_layout": "B,C,H,W",
                "token_counts": {
                    "context": context_tokens,
                    "image": h_tokens * w_tokens,
                    "stream": (context_tokens or 0) + h_tokens * w_tokens if context_tokens is not None else None,
                    "h_tokens": h_tokens,
                    "w_tokens": w_tokens,
                },
                "timestep": self.map_timestep(call),
            }
        )
        return out

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for ZImageIslandRuntime.")
        latent_torch = _single_z_image_latent(call.model_input)
        if latent_torch.ndim != 4:
            raise UnsupportedIslandRequest(
                f"Z-Image MLX denoiser island expects a rank-4 latent tensor; got {_shape_summary(call.model_input)}."
            )
        if call.context is None or not hasattr(call.context, "shape"):
            raise UnsupportedIslandRequest("Z-Image MLX denoiser island requires c_crossattn text context.")
        if int(call.context.shape[-1]) != self.context_dim:
            raise UnsupportedIslandRequest(f"Expected Z-Image context dim {self.context_dim}; got {int(call.context.shape[-1])}.")
        for unsupported in ("attention_mask", "clip_text_pooled", "ref_latents", "ref_contexts", "siglip_feats"):
            if unsupported in call.extra_conds:
                raise UnsupportedIslandRequest(f"Z-Image MLX island v1 does not support {unsupported}.")

        latent_mx = self.bridge.torch_to_mlx(latent_torch, dtype=mx.bfloat16)
        context_mx = self._context_to_mlx(call.context, latent_batch=int(latent_mx.shape[0]), num_tokens=call.extra_conds.get("num_tokens"))
        if int(context_mx.shape[0]) == 1 and int(latent_mx.shape[0]) != 1:
            context_mx = mx.repeat(context_mx, int(latent_mx.shape[0]), axis=0)
        if int(context_mx.shape[0]) != int(latent_mx.shape[0]):
            raise UnsupportedIslandRequest(
                f"Z-Image context batch {context_mx.shape[0]} does not match latent batch {latent_mx.shape[0]}."
            )
        sigma_mx = self.bridge.torch_to_mlx(call.sigma, dtype=mx.float32)
        timestep_mx = self.bridge.torch_to_mlx(call.timestep, dtype=mx.float32)
        if self.predictor is not None:
            prediction = self.predictor(
                latent=latent_mx,
                sigma=sigma_mx,
                timestep=timestep_mx,
                context=context_mx,
                call=call,
                mx=mx,
            )
            predictor_name = "injected"
        else:
            prediction = self._transformer_obj()(latent_mx, timestep_mx, context_mx, num_tokens=call.extra_conds.get("num_tokens"))
            predictor_name = "native_mlx_z_image"
        if _env_flag("COMFY_MLX_ISLAND_EVAL_BEFORE_RETURN", True):
            mx.eval(prediction)
        out = self.bridge.mlx_to_torch(prediction, like=latent_torch)
        self.calls += 1
        self.last_event = {
            "runtime": self.name,
            "model_path": self.model_path,
            "latent_shape": list(latent_torch.shape),
            "context_shape": list(call.context.shape),
            "cond_or_uncond": call.cond_or_uncond,
            "predictor": predictor_name,
            "conversions": self.bridge.counter.as_dict(),
            "memory": _mlx_memory_snapshot(mx),
        }
        return out

    def forward_denoised_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for ZImageIslandRuntime.")
        latent_torch = _single_z_image_latent(call.model_input)
        if latent_torch.ndim != 4:
            raise UnsupportedIslandRequest(
                f"Z-Image MLX denoiser island expects a rank-4 latent tensor; got {_shape_summary(call.model_input)}."
            )
        if call.context is None or not hasattr(call.context, "shape"):
            raise UnsupportedIslandRequest("Z-Image MLX denoiser island requires c_crossattn text context.")
        if int(call.context.shape[-1]) != self.context_dim:
            raise UnsupportedIslandRequest(f"Expected Z-Image context dim {self.context_dim}; got {int(call.context.shape[-1])}.")
        for unsupported in ("attention_mask", "clip_text_pooled", "ref_latents", "ref_contexts", "siglip_feats"):
            if unsupported in call.extra_conds:
                raise UnsupportedIslandRequest(f"Z-Image MLX island v1 does not support {unsupported}.")

        latent_mx = self.bridge.torch_to_mlx(latent_torch, dtype=mx.bfloat16)
        context_mx = self._context_to_mlx(call.context, latent_batch=int(latent_mx.shape[0]), num_tokens=call.extra_conds.get("num_tokens"))
        if int(context_mx.shape[0]) == 1 and int(latent_mx.shape[0]) != 1:
            context_mx = mx.repeat(context_mx, int(latent_mx.shape[0]), axis=0)
        if int(context_mx.shape[0]) != int(latent_mx.shape[0]):
            raise UnsupportedIslandRequest(
                f"Z-Image context batch {context_mx.shape[0]} does not match latent batch {latent_mx.shape[0]}."
            )
        sigma_mx = self.bridge.torch_to_mlx(call.sigma, dtype=mx.float32)
        timestep_mx = self.bridge.torch_to_mlx(call.timestep, dtype=mx.float32)
        if self.predictor is not None:
            prediction = self.predictor(
                latent=latent_mx,
                sigma=sigma_mx,
                timestep=timestep_mx,
                context=context_mx,
                call=call,
                mx=mx,
            )
            predictor_name = "injected"
        else:
            prediction = self._transformer_obj()(latent_mx, timestep_mx, context_mx, num_tokens=call.extra_conds.get("num_tokens"))
            predictor_name = "native_mlx_z_image"
        sigma_view = sigma_mx.reshape((int(sigma_mx.shape[0]),) + (1,) * (len(prediction.shape) - 1))
        denoised = latent_mx.astype(mx.float32) - prediction.astype(mx.float32) * sigma_view
        if _env_flag("COMFY_MLX_ISLAND_EVAL_BEFORE_RETURN", True):
            mx.eval(denoised)
        out = self.bridge.mlx_to_torch(denoised, like=latent_torch, dtype=torch.float32)
        self.calls += 1
        self.last_event = {
            "runtime": self.name,
            "model_path": self.model_path,
            "latent_shape": list(latent_torch.shape),
            "context_shape": list(call.context.shape),
            "cond_or_uncond": call.cond_or_uncond,
            "predictor": predictor_name,
            "return_kind": "denoised",
            "conversions": self.bridge.counter.as_dict(),
            "memory": _mlx_memory_snapshot(mx),
        }
        return out

    def sample_res_multistep_cfg1(
        self,
        *,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        sigmas: torch.Tensor,
        context: torch.Tensor,
        num_tokens: int | None,
        model_sampling: Any,
    ) -> torch.Tensor:
        """Run the CFG=1 res_multistep/simple Z-Image sampler inside MLX.

        This is a deliberately narrow benchmark candidate. It keeps Comfy's
        workflow shape but removes the per-step Torch/MLX island boundary for
        the common Z-Image Turbo path we are optimizing.
        """

        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for ZImageIslandRuntime.")
        if noise.ndim != 4 or latent_image.ndim != 4:
            raise UnsupportedIslandRequest("Native Z-Image sampler expects rank-4 noise and latent tensors.")
        if int(latent_image.shape[0]) != 1:
            raise UnsupportedIslandRequest("Native Z-Image sampler currently supports batch size 1.")
        if context.ndim != 3 or int(context.shape[-1]) != self.context_dim:
            raise UnsupportedIslandRequest(
                f"Native Z-Image sampler expected context dim {self.context_dim}; got {_shape_summary(context)}."
            )
        if len(sigmas) < 2:
            raise UnsupportedIslandRequest("Native Z-Image sampler requires at least two sigmas.")

        max_sigma = float(model_sampling.sigma_max)
        sigma0 = float(sigmas[0].detach().cpu().item())
        max_denoise = math.isclose(max_sigma, sigma0, rel_tol=1e-05) or sigma0 > max_sigma
        x_torch = model_sampling.noise_scaling(sigmas[0], noise, latent_image, max_denoise)
        x = self.bridge.torch_to_mlx(x_torch, dtype=mx.bfloat16)
        context_mx = self.bridge.torch_to_mlx(context, dtype=mx.bfloat16)
        if int(context_mx.shape[0]) == 1 and int(x.shape[0]) != 1:
            context_mx = mx.repeat(context_mx, int(x.shape[0]), axis=0)
        transformer = None if self.predictor is not None else self._transformer_obj()

        old_sigma_down: float | None = None
        old_denoised: Any | None = None
        sigma_values = [float(item.detach().cpu().item()) for item in sigmas]
        timestep_values = [
            float(model_sampling.timestep(sigmas[index : index + 1]).detach().cpu().item())
            for index in range(len(sigmas) - 1)
        ]

        for index in range(len(sigma_values) - 1):
            sigma = sigma_values[index]
            sigma_down = sigma_values[index + 1]
            sigma_mx = mx.array([sigma], dtype=mx.float32)
            timestep_mx = mx.array([timestep_values[index]], dtype=mx.float32)
            if self.predictor is not None:
                prediction = self.predictor(
                    latent=x.astype(mx.bfloat16),
                    sigma=sigma_mx,
                    timestep=timestep_mx,
                    context=context_mx,
                    call=None,
                    mx=mx,
                )
            else:
                prediction = transformer(x.astype(mx.bfloat16), timestep_mx, context_mx, num_tokens=num_tokens)
            sigma_view = sigma_mx.reshape((int(x.shape[0]),) + (1,) * (len(prediction.shape) - 1))
            denoised = x.astype(mx.float32) - prediction.astype(mx.float32) * sigma_view

            if sigma_down == 0.0 or old_denoised is None:
                d = (x.astype(mx.float32) - denoised) / mx.array(sigma, dtype=mx.float32)
                x = x.astype(mx.float32) + d * mx.array(sigma_down - sigma, dtype=mx.float32)
            else:
                t = -math.log(sigma)
                t_old = -math.log(float(old_sigma_down))
                t_next = -math.log(sigma_down)
                t_prev = -math.log(sigma_values[index - 1])
                h = t_next - t
                c2 = (t_prev - t_old) / h
                phi1 = math.expm1(-h) / (-h)
                phi2 = (phi1 - 1.0) / (-h)
                b1 = phi1 - phi2 / c2
                b2 = phi2 / c2
                if not math.isfinite(b1):
                    b1 = 0.0
                if not math.isfinite(b2):
                    b2 = 0.0
                x = (
                    mx.array(math.exp(-h), dtype=mx.float32) * x.astype(mx.float32)
                    + mx.array(h * b1, dtype=mx.float32) * denoised
                    + mx.array(h * b2, dtype=mx.float32) * old_denoised
                )

            old_denoised = denoised
            old_sigma_down = sigma_down

        mx.eval(x)
        out = self.bridge.mlx_to_torch(x, like=latent_image, dtype=torch.float32)
        self.calls += len(sigma_values) - 1
        self.last_event = {
            "runtime": self.name,
            "model_path": self.model_path,
            "return_kind": "native_res_multistep_cfg1",
            "steps": len(sigma_values) - 1,
            "latent_shape": list(latent_image.shape),
            "context_shape": list(context.shape),
            "conversions": self.bridge.counter.as_dict(),
            "memory": _mlx_memory_snapshot(mx),
        }
        return out

    def _context_to_mlx(self, context: Any, *, latent_batch: int, num_tokens: Any) -> Any:
        mx = self.bridge.mx
        if mx is None:
            raise RuntimeError("MLX is required for ZImageIslandRuntime.")
        cache_size = int(__import__("os").environ.get("COMFY_MLX_Z_IMAGE_CONTEXT_CACHE_SIZE", "8"))
        if cache_size <= 0 or not isinstance(context, torch.Tensor):
            return self.bridge.torch_to_mlx(context, dtype=mx.bfloat16)
        try:
            data_key = int(context.data_ptr())
        except Exception:
            data_key = id(context)
        key = (
            "z_image_context",
            data_key,
            tuple(int(item) for item in context.shape),
            str(context.dtype),
            str(context.device),
            int(latent_batch),
            int(num_tokens) if isinstance(num_tokens, int) else None,
        )
        cached = self.caches.context.get(key)
        if cached is not None:
            return cached
        value = self.bridge.torch_to_mlx(context, dtype=mx.bfloat16)
        if int(value.shape[0]) == 1 and int(latent_batch) != 1:
            value = mx.repeat(value, int(latent_batch), axis=0)
        if len(self.caches.context) >= cache_size:
            self.caches.context.pop(next(iter(self.caches.context)))
        self.caches.context[key] = value
        return value

    def _transformer_obj(self) -> Any:
        if self.model_path is None:
            raise UnsupportedIslandRequest("Z-Image MLX island runtime has no model_path configured.")
        if self._transformer is None:
            from comfy.backends.mlx_z_image import MLXZImageTransformer

            self._transformer = MLXZImageTransformer(self.model_path, dtype=self.bridge.mx.bfloat16)
        return self._transformer


def split_ltx_av_latents(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise UnsupportedIslandRequest(f"Expected two LTX AV latents [video, audio]; got {len(value)}.")
        video, audio = value
        if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
            raise UnsupportedIslandRequest("LTX AV latent list must contain Torch tensors.")
        return video, audio
    if isinstance(value, torch.Tensor) and value.ndim == 5:
        empty_audio = torch.zeros((value.shape[0], 8, 0, 16), device=value.device, dtype=value.dtype)
        return value, empty_audio
    raise UnsupportedIslandRequest(f"Unsupported LTX AV latent value type: {type(value).__name__}.")


def _single_z_image_latent(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise UnsupportedIslandRequest(f"Unsupported Z-Image latent value type: {_shape_summary(value)}.")


def _extract_direct_z_image_context(positive: Any) -> tuple[torch.Tensor, int | None] | None:
    if not isinstance(positive, list) or len(positive) != 1:
        return None
    row = positive[0]
    if not isinstance(row, (list, tuple)) or len(row) < 1 or not isinstance(row[0], torch.Tensor):
        return None
    context = row[0]
    metadata = row[1] if len(row) > 1 and isinstance(row[1], dict) else {}
    unsupported = {
        "area",
        "mask",
        "control",
        "gligen",
        "reference_latents",
        "clip_vision_outputs",
        "reference_latents_text_embeds",
    }
    if unsupported.intersection(metadata):
        return None
    num_tokens = None
    attention_mask = metadata.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        try:
            num_tokens = int(attention_mask.detach().sum().cpu().item())
        except Exception:
            num_tokens = None
    if num_tokens is None and hasattr(context, "shape") and len(context.shape) >= 2:
        num_tokens = int(context.shape[1])
    return context, num_tokens


def try_z_image_native_res_multistep_sampler(
    *,
    model: Any,
    noise: torch.Tensor,
    latent_image: torch.Tensor,
    positive: Any,
    negative: Any,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    disable_noise: bool,
    start_step: Any,
    last_step: Any,
    force_full_denoise: bool,
    noise_mask: Any,
    seed: int,
) -> torch.Tensor | None:
    if not _env_flag("COMFY_MLX_Z_IMAGE_NATIVE_RES_MULTISTEP", False):
        return None
    if sampler_name != "res_multistep" or scheduler != "simple":
        return None
    if not math.isclose(float(cfg), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        return None
    if not math.isclose(float(denoise), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        return None
    if disable_noise or start_step is not None or last_step is not None or force_full_denoise or noise_mask is not None:
        return None
    wrapper = getattr(model, "model_options", {}).get("model_function_wrapper")
    if not isinstance(wrapper, MLXDenoiserFunctionWrapper) or not isinstance(wrapper.runtime, ZImageIslandRuntime):
        return None
    proxy = getattr(model, "model", None)
    if getattr(proxy, "model_family", None) != "z_image":
        return None
    context_pair = _extract_direct_z_image_context(positive)
    if context_pair is None:
        return None
    context, num_tokens = context_pair
    if negative not in (None, []):
        # CFG=1 should not use negative conditioning, but reject complicated
        # negative structures so this candidate stays honest.
        if not isinstance(negative, list) or len(negative) > 1:
            return None

    sigmas = comfy.samplers.calculate_sigmas(proxy.model_sampling, scheduler, int(steps)).to(model.load_device)
    start = time.perf_counter()
    samples = wrapper.runtime.sample_res_multistep_cfg1(
        noise=noise,
        latent_image=latent_image,
        sigmas=sigmas,
        context=context,
        num_tokens=num_tokens,
        model_sampling=proxy.model_sampling,
    )
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    write_event(
        "mlx_z_image_native_sampler",
        sampler=sampler_name,
        scheduler=scheduler,
        steps=int(steps),
        seed=int(seed),
        seconds=time.perf_counter() - start,
        runtime_event=wrapper.runtime.last_event,
    )
    return samples


def _extract_frame_rate(extra_conds: dict[str, Any], default: float) -> float:
    value = extra_conds.get("frame_rate", default)
    if isinstance(value, torch.Tensor):
        value = value.detach().flatten()[0].cpu().item()
    return float(value)


def _torch_mask_metadata(mask: Any) -> dict[str, Any]:
    if mask is None:
        return {"present": False}
    out = {
        "present": True,
        "type": type(mask).__name__,
        "shape": _shape_summary(mask),
    }
    if isinstance(mask, torch.Tensor):
        out.update(
            {
                "dtype": str(mask.dtype),
                "device": str(mask.device),
                "numel": int(mask.numel()),
            }
        )
    return out


def _x0_to_velocity(mx: Any, latent: Any, x0: Any, sigma: Any) -> Any:
    eps = mx.array(1.0e-6, dtype=mx.float32)
    sigma = mx.maximum(sigma.astype(mx.float32), eps)[:, None, None]
    velocity = (latent.astype(mx.float32) - x0.astype(mx.float32)) / sigma
    return velocity.astype(latent.dtype)


def _mlx_memory_snapshot(mx: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, fn_name in (
        ("active_memory", "get_active_memory"),
        ("cache_memory", "get_cache_memory"),
        ("peak_memory", "get_peak_memory"),
    ):
        fn = getattr(mx, fn_name, None)
        if fn is None:
            metal = getattr(mx, "metal", None)
            fn = getattr(metal, fn_name, None) if metal is not None else None
        if fn is None:
            continue
        try:
            out[name] = int(fn())
        except Exception:
            pass
    return out


def _shape_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


class LightweightMLXModelProxy(torch.nn.Module):
    """Comfy MODEL-compatible object that does not own a Torch transformer."""

    is_mlx_denoiser_island_proxy = True

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        sampling_shift: float = 2.05,
        latent_format: Any | None = None,
        model_family: str = "ltx_av",
    ):
        super().__init__()
        self.dtype = dtype
        self.manual_cast_dtype = None
        self.device = torch.device("cpu")
        self.latent_format = latent_format if latent_format is not None else comfy.latent_formats.LTXAV()
        self.model_family = model_family
        if model_family == "z_image":
            self.model_sampling = _ZImageIslandSampling(None)
            self.model_sampling.set_parameters(shift=float(sampling_shift), multiplier=1.0)
        else:
            self.model_sampling = _LTXIslandSampling(None)
            self.model_sampling.set_parameters(shift=float(sampling_shift))
        self.current_patcher: comfy.model_patcher.ModelPatcher | None = None
        self.model_config = type(
            "MLXIslandModelConfig",
            (),
            {"sampling_settings": {"shift": float(sampling_shift)}, "memory_usage_factor": 1.0},
        )()

    def apply_model(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("MLX island proxy apply_model should only be called through model_function_wrapper.")

    def process_timestep(self, timestep: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return timestep

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_format.process_out(latent)

    def scale_latent_inpaint(
        self,
        *,
        x: torch.Tensor,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
    ) -> torch.Tensor:
        return latent_image

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_dtype_inference(self) -> torch.dtype:
        return self.manual_cast_dtype or self.get_dtype()

    def memory_required(self, *_args: Any, **_kwargs: Any) -> int:
        return 0

    def extra_conds_shapes(self, **_kwargs: Any) -> dict[str, list[int]]:
        return {}

    def extra_conds(self, **kwargs: Any) -> dict[str, Any]:
        if self.model_family == "z_image":
            return self._extra_conds_z_image(**kwargs)
        if self.model_family == "klein":
            return self._extra_conds_klein(**kwargs)

        out: dict[str, Any] = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        out["frame_rate"] = comfy.conds.CONDConstant(kwargs.get("frame_rate", 25))

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        audio_denoise_mask = None
        latent_shapes = kwargs.get("latent_shapes", None)
        if denoise_mask is not None and latent_shapes is not None:
            denoise_mask = comfy.utils.unpack_latents(denoise_mask, latent_shapes)
            if len(denoise_mask) > 1:
                audio_denoise_mask = denoise_mask[1]
            denoise_mask = denoise_mask[0]

        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)
        if audio_denoise_mask is not None:
            out["audio_denoise_mask"] = comfy.conds.CONDRegular(audio_denoise_mask)

        if latent_shapes is not None:
            out["latent_shapes"] = comfy.conds.CONDConstant(latent_shapes)

        keyframe_idxs = kwargs.get("keyframe_idxs", None)
        if keyframe_idxs is not None:
            out["keyframe_idxs"] = comfy.conds.CONDRegular(keyframe_idxs)

        guide_attention_entries = kwargs.get("guide_attention_entries", None)
        if guide_attention_entries is not None:
            out["guide_attention_entries"] = comfy.conds.CONDConstant(guide_attention_entries)

        ref_audio = kwargs.get("ref_audio", None)
        if ref_audio is not None:
            out["ref_audio"] = comfy.conds.CONDConstant(ref_audio)

        return out

    def _extra_conds_z_image(self, **kwargs: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out["attention_mask"] = comfy.conds.CONDRegular(attention_mask)
            out["num_tokens"] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            if "num_tokens" not in out:
                out["num_tokens"] = comfy.conds.CONDConstant(cross_attn.shape[1])

        clip_text_pooled = kwargs.get("pooled_output", None)
        if clip_text_pooled is not None:
            out["clip_text_pooled"] = comfy.conds.CONDRegular(clip_text_pooled)
        return out

    def _extra_conds_klein(self, **kwargs: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            target_text_len = 512
            if cross_attn.shape[1] < target_text_len:
                pad = torch.zeros(
                    (
                        cross_attn.shape[0],
                        target_text_len - cross_attn.shape[1],
                        cross_attn.shape[2],
                    ),
                    dtype=cross_attn.dtype,
                    device=cross_attn.device,
                )
                cross_attn = torch.cat((cross_attn, pad), dim=1)
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out["guidance"] = comfy.conds.CONDRegular(torch.FloatTensor([float(guidance)]))
        return out


_SAFE_TRANSFORMER_OPTION_KEYS = {
    "cond_or_uncond",
    "uuids",
    "sigmas",
    "sample_sigmas",
    "prefetch_dynamic_vbars",
    "wrappers",
    "callbacks",
}


def _reject_unsupported_request(c: dict[str, Any]) -> None:
    control = c.get("control", None)
    if control is not None:
        raise UnsupportedIslandRequest("MLX denoiser island does not support ControlNet/control objects yet.")

    transformer_options = c.get("transformer_options", {}) or {}
    for key in ("patches", "patches_replace", "wrappers", "callbacks"):
        if transformer_options.get(key):
            raise UnsupportedIslandRequest(f"MLX denoiser island does not support transformer_options[{key!r}] yet.")

    unknown = set(transformer_options) - _SAFE_TRANSFORMER_OPTION_KEYS - {"patches", "patches_replace"}
    if unknown:
        raise UnsupportedIslandRequest(
            "MLX denoiser island received unsupported transformer options: " + ", ".join(sorted(unknown))
        )


def _cast_cond_value(value: Any, dtype: torch.dtype, device: torch.device) -> Any:
    if hasattr(value, "dtype"):
        return comfy.model_management.cast_to_device(value, device, dtype)
    if isinstance(value, list):
        return [_cast_cond_value(v, dtype, device) for v in value]
    return value


def _slice_leading_batch(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value.narrow(0, index, 1)
        return value
    if isinstance(value, list):
        if len(value) == batch_size and not any(isinstance(v, torch.Tensor) for v in value):
            return [value[index]]
        return [_slice_leading_batch(v, index, batch_size) for v in value]
    if isinstance(value, tuple):
        return tuple(_slice_leading_batch(v, index, batch_size) for v in value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "guide_attention_entries":
                out[key] = item
            else:
                out[key] = _slice_leading_batch(item, index, batch_size)
        return out
    return value


def _repeat_leading_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size <= 1:
        return value
    repeat_shape = [batch_size] + [1] * (value.ndim - 1)
    return value.repeat(*repeat_shape)


def _branch_abs_max(value: torch.Tensor, index: int) -> float:
    branch = value.narrow(0, index, 1)
    if branch.numel() == 0:
        return 0.0
    return float(branch.detach().abs().max().item())


def _maybe_collapse_ltx_cfg1_batch(
    *,
    runtime: BaseMLXDenoiserIslandRuntime,
    x: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    model_input: Any,
    context: Any | None,
    extra_conds: dict[str, Any],
    transformer_options: dict[str, Any],
    cond_or_uncond: list[int],
    cond_scale: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any | None, dict[str, Any], dict[str, Any], list[int], dict[str, Any]]:
    if _env_flag("COMFY_MLX_LTX_DISABLE_CFG1_BATCH_COLLAPSE", False):
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "disabled",
        }
    if getattr(runtime, "name", None) != "ltx_av":
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "runtime_not_ltx_av",
        }
    try:
        cfg_value = float(cond_scale)
    except (TypeError, ValueError):
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "missing_cond_scale",
        }
    if not math.isclose(cfg_value, 1.0):
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "cfg_not_one",
            "cond_scale": cfg_value,
        }
    if len(cond_or_uncond) != 2 or sorted(cond_or_uncond) != [0, 1]:
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "not_cond_uncond_pair",
            "cond_or_uncond": cond_or_uncond,
        }
    batch_size = int(x.shape[0]) if x.ndim > 0 else 0
    if batch_size != 2:
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "batch_size_not_two",
            "batch_size": batch_size,
        }
    if not isinstance(context, torch.Tensor) or context.ndim == 0 or context.shape[0] != batch_size:
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "context_not_batch2_tensor",
            "context_shape": _shape_summary(context),
        }

    positive_index = cond_or_uncond.index(0)
    negative_index = cond_or_uncond.index(1)
    zero_threshold = float(os.environ.get("COMFY_MLX_LTX_CFG1_NEGATIVE_ZERO_ATOL", "1e-6"))
    negative_abs_max = _branch_abs_max(context, negative_index)
    if negative_abs_max > zero_threshold:
        return x, sigma, timestep, model_input, context, extra_conds, transformer_options, cond_or_uncond, {
            "applied": False,
            "reason": "negative_context_nonzero",
            "negative_context_abs_max": negative_abs_max,
            "zero_threshold": zero_threshold,
        }

    collapsed_transformer_options = _slice_leading_batch(transformer_options, positive_index, batch_size)
    collapsed_transformer_options = dict(collapsed_transformer_options)
    collapsed_transformer_options["cond_or_uncond"] = [0]
    collapsed_transformer_options["mlx_cfg1_batch_collapse_original_cond_or_uncond"] = cond_or_uncond[:]

    return (
        _slice_leading_batch(x, positive_index, batch_size),
        _slice_leading_batch(sigma, positive_index, batch_size),
        _slice_leading_batch(timestep, positive_index, batch_size),
        _slice_leading_batch(model_input, positive_index, batch_size),
        _slice_leading_batch(context, positive_index, batch_size),
        _slice_leading_batch(extra_conds, positive_index, batch_size),
        collapsed_transformer_options,
        [0],
        {
            "applied": True,
            "reason": "ltx_cfg1_zero_negative",
            "original_batch_size": batch_size,
            "positive_index": positive_index,
            "negative_index": negative_index,
            "original_cond_or_uncond": cond_or_uncond[:],
            "cond_scale": cfg_value,
            "negative_context_abs_max": negative_abs_max,
        },
    )


class MLXDenoiserFunctionWrapper:
    """Comfy `model_function_wrapper` that owns a coarse MLX denoiser island."""

    def __init__(self, model: LightweightMLXModelProxy, runtime: BaseMLXDenoiserIslandRuntime):
        self.model = model
        self.runtime = runtime
        self.return_device: torch.device | None = None
        self.calls = 0
        self.last_event: dict[str, Any] | None = None

    def __call__(self, _apply_model: Any, args: dict[str, Any]) -> torch.Tensor:
        wrapper_start = time.perf_counter()
        c = dict(args.get("c") or {})
        _reject_unsupported_request(c)

        x: torch.Tensor = args["input"]
        sigma: torch.Tensor = args["timestep"]
        xc = self.model.model_sampling.calculate_input(sigma, x)

        c_concat = c.pop("c_concat", None)
        if c_concat is not None:
            c_concat = comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)
            xc = torch.cat([xc, c_concat], dim=1)

        dtype = self.model.get_dtype_inference()
        xc = xc.to(dtype)
        device = xc.device

        context = c.pop("c_crossattn", None)
        if context is not None and hasattr(context, "dtype"):
            context = comfy.model_management.cast_to_device(context, device, dtype)

        timestep = self.model.model_sampling.timestep(sigma).float()
        extra_conds: dict[str, Any] = {}
        for key, value in c.items():
            if key in {"control", "transformer_options"}:
                continue
            extra_conds[key] = _cast_cond_value(value, dtype, device)

        timestep = self.model.process_timestep(timestep, x=x, **extra_conds)
        if "latent_shapes" in extra_conds:
            xc = comfy.utils.unpack_latents(xc, extra_conds.pop("latent_shapes"))

        transformer_options = dict(c.get("transformer_options", {}) or {})
        transformer_options["prefetch_dynamic_vbars"] = False
        cond_or_uncond = list(args.get("cond_or_uncond") or transformer_options.get("cond_or_uncond") or [])
        original_input_shape = _shape_summary(x)

        (
            x,
            sigma,
            timestep,
            xc,
            context,
            extra_conds,
            transformer_options,
            cond_or_uncond,
            batch_collapse,
        ) = _maybe_collapse_ltx_cfg1_batch(
            runtime=self.runtime,
            x=x,
            sigma=sigma,
            timestep=timestep,
            model_input=xc,
            context=context,
            extra_conds=extra_conds,
            transformer_options=transformer_options,
            cond_or_uncond=cond_or_uncond,
            cond_scale=args.get("cond_scale", None),
        )

        call = MLXDenoiserCall(
            model_input=xc,
            sigma=sigma,
            timestep=timestep,
            original_input=x,
            context=context,
            control=None,
            transformer_options=transformer_options,
            extra_conds=extra_conds,
            cond_or_uncond=cond_or_uncond,
        )
        self.runtime.validate_extra_conds(call.extra_conds)
        preprocess_seconds = time.perf_counter() - wrapper_start
        runtime_start = time.perf_counter()
        return_denoised = _env_flag("COMFY_MLX_ISLAND_RETURN_DENOISED", False)
        forward_denoised = getattr(self.runtime, "forward_denoised_output", None)
        if return_denoised and callable(forward_denoised):
            denoised = forward_denoised(call)
            model_output = None
        else:
            model_output = self.runtime.forward_model_output(call)
            denoised = None
        runtime_seconds = time.perf_counter() - runtime_start
        denoised_start = time.perf_counter()
        if denoised is None:
            if isinstance(model_output, (list, tuple)):
                model_output, _ = comfy.utils.pack_latents(model_output)
            denoised = self.model.model_sampling.calculate_denoised(sigma, model_output.float(), x)
        if batch_collapse.get("applied"):
            denoised = _repeat_leading_batch(denoised, int(batch_collapse["original_batch_size"]))
        calculate_denoised_seconds = time.perf_counter() - denoised_start
        self.calls += 1
        self.last_event = {
            "runtime": self.runtime.name,
            "input_shape": original_input_shape,
            "model_input_shape": _shape_summary(xc),
            "output_shape": _shape_summary(denoised),
            "dtype": str(dtype),
            "device": str(denoised.device),
            "cond_or_uncond": call.cond_or_uncond,
            "batch_collapse": batch_collapse,
            "conversions": self.runtime.bridge.counter.as_dict(),
            "preprocess_seconds": preprocess_seconds,
            "runtime_seconds": runtime_seconds,
            "calculate_denoised_seconds": calculate_denoised_seconds,
            "wrapper_seconds": time.perf_counter() - wrapper_start,
            "runtime_event": getattr(self.runtime, "last_event", None),
        }
        if _env_flag("COMFY_MLX_ISLAND_DEBUG_CONDITIONING", False):
            self.last_event["conditioning_debug"] = {
                "cond_or_uncond": call.cond_or_uncond,
                "context_type": type(context).__name__,
                "context_shape": _shape_summary(context),
                "extra_cond_keys": sorted(str(key) for key in extra_conds.keys()),
                "transformer_option_keys": sorted(str(key) for key in transformer_options.keys()),
                "has_control": "control" in c,
                "has_patches": "patches" in transformer_options,
            }
        write_event("mlx_denoiser_island_call", call_index=self.calls, **self.last_event)
        return denoised

    def to(self, device: torch.device | torch.dtype | str) -> "MLXDenoiserFunctionWrapper":
        torch_module = globals().get("torch")
        if torch_module is not None and isinstance(device, torch_module.dtype):
            return self
        if torch_module is not None:
            self.return_device = torch_module.device(device)
        return self

    def models(self) -> list[Any]:
        return []

    def cleanup(self) -> None:
        self.runtime.cleanup()


def create_mlx_denoiser_island_model(
    *,
    runtime: BaseMLXDenoiserIslandRuntime | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    sampling_shift: float = 2.05,
    latent_format: Any | None = None,
    model_family: str = "ltx_av",
) -> comfy.model_patcher.ModelPatcher:
    """Create a lightweight Comfy MODEL patcher backed by an MLX denoiser island."""

    load = torch.device(load_device or "cpu")
    offload = torch.device(offload_device or "cpu")
    model = LightweightMLXModelProxy(
        dtype=dtype,
        sampling_shift=sampling_shift,
        latent_format=latent_format,
        model_family=model_family,
    )
    patcher = comfy.model_patcher.ModelPatcher(model, load, offload, size=0)
    model.current_patcher = patcher
    island_runtime = runtime if runtime is not None else ContractMLXDenoiserRuntime()
    patcher.set_model_unet_function_wrapper(MLXDenoiserFunctionWrapper(model, island_runtime))
    return patcher


def create_z_image_mlx_island_model(
    *,
    runtime: ZImageIslandRuntime | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    sampling_shift: float = 3.0,
) -> comfy.model_patcher.ModelPatcher:
    """Create a lightweight Z-Image MODEL patcher backed by an MLX island runtime."""

    island_runtime = runtime if runtime is not None else ZImageIslandRuntime()
    return create_mlx_denoiser_island_model(
        runtime=island_runtime,
        load_device=load_device,
        offload_device=offload_device,
        dtype=dtype,
        sampling_shift=sampling_shift,
        latent_format=comfy.latent_formats.Flux(),
        model_family="z_image",
    )


def is_mlx_denoiser_island_model(model: Any) -> bool:
    return bool(getattr(model, "is_mlx_denoiser_island_proxy", False))
