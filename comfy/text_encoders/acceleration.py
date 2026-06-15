from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from comfy.backends.benchmark_stats import elapsed_seconds, now, redact_secrets, write_event


SIDECAR_SUFFIX = ".accel.json"
SUPPORTED_BACKENDS = {"auto", "torch_mps", "torch"}
GEMMA4_FAMILIES = {"gemma4_e2b", "gemma4_e4b", "gemma4_31b"}
QWEN_NATIVE_MTP_FAMILIES = {
    "qwen35_08b",
    "qwen35_2b",
    "qwen35_4b",
    "qwen35_9b",
    "qwen35_27b",
    "qwen36",
    "qwen36_next",
    "qwen3_next",
    "qwen3-next",
}


@dataclass(frozen=True)
class AccelerationConfig:
    mode: str = "off"
    family: str = ""
    target_clip_name: str = ""
    target_device: str = ""
    target_offload_device: str = ""
    target_initial_device: str = ""
    target_dtype: str = ""
    assistant_clip_name: str = ""
    assistant_kind: str = ""
    max_speculative_tokens: int = 0
    backend: str = "auto"
    strict: bool = False
    text_only: bool = True
    allow_multimodal: bool = False
    assistant_repo_id: str = ""
    assistant_local_path: str = ""
    assistant_revision: str = "main"
    assistant_dtype: str = "auto"
    assistant_trust_remote_code: bool = False
    use_hf_token: bool = False
    native_mtp: dict[str, Any] = field(default_factory=dict)
    draft_speculation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccelerationGate:
    supported: bool
    reason: str = ""
    route: str = "ar"
    mode: str = "off"
    family: str = ""
    backend: str = "auto"
    assistant_kind: str = ""
    assistant_clip_name: str = ""
    max_speculative_tokens: int = 0


def sidecar_candidates(model_path: str) -> list[str]:
    base = os.path.abspath(os.path.expanduser(model_path))
    root, ext = os.path.splitext(base)
    candidates = [f"{root}{SIDECAR_SUFFIX}"]
    if ext:
        candidates.append(f"{base}{SIDECAR_SUFFIX}")
    return candidates


def find_sidecar(model_path: str) -> str | None:
    for candidate in sidecar_candidates(model_path):
        if os.path.exists(candidate):
            return candidate
    return None


def load_acceleration_config_for_path(model_path: str) -> tuple[str | None, AccelerationConfig | None]:
    sidecar = find_sidecar(model_path)
    if sidecar is None:
        return None, None
    with open(sidecar, encoding="utf-8") as f:
        raw = json.load(f)
    return sidecar, parse_acceleration_config(raw)


def parse_acceleration_config(raw: dict[str, Any]) -> AccelerationConfig:
    payload = raw.get("acceleration") if isinstance(raw.get("acceleration"), dict) else raw
    mode = str(payload.get("mode", "off")).strip().lower()
    strict = bool(payload.get("strict", False)) or mode == "strict"
    if mode == "strict":
        mode = "auto"
    native_mtp = dict(payload.get("native_mtp") or {})
    draft_speculation = dict(payload.get("draft_speculation") or {})
    assistant_clip_name = str(
        payload.get("assistant_clip_name")
        or draft_speculation.get("assistant_clip_name")
        or draft_speculation.get("draft_checkpoint")
        or ""
    ).strip()
    assistant_kind = str(
        payload.get("assistant_kind")
        or draft_speculation.get("assistant_kind")
        or draft_speculation.get("draft_kind")
        or native_mtp.get("assistant_kind")
        or ""
    ).strip().lower()
    max_speculative_tokens = int(
        payload.get("max_speculative_tokens")
        or draft_speculation.get("max_speculative_tokens")
        or draft_speculation.get("draft_block_size")
        or native_mtp.get("max_speculative_tokens")
        or 0
    )
    assistant_repo_id = str(
        payload.get("assistant_repo_id")
        or draft_speculation.get("assistant_repo_id")
        or native_mtp.get("assistant_repo_id")
        or ""
    ).strip()
    assistant_local_path = str(
        payload.get("assistant_local_path")
        or draft_speculation.get("assistant_local_path")
        or native_mtp.get("assistant_local_path")
        or ""
    ).strip()
    return AccelerationConfig(
        mode=mode,
        family=str(payload.get("family") or "").strip().lower(),
        target_clip_name=str(payload.get("target_clip_name") or "").strip(),
        target_device=str(payload.get("target_device") or payload.get("load_device") or "").strip().lower(),
        target_offload_device=str(payload.get("target_offload_device") or payload.get("offload_device") or "").strip().lower(),
        target_initial_device=str(payload.get("target_initial_device") or payload.get("initial_device") or "").strip().lower(),
        target_dtype=str(payload.get("target_dtype") or payload.get("dtype") or "").strip().lower(),
        assistant_clip_name=assistant_clip_name,
        assistant_kind=assistant_kind,
        max_speculative_tokens=max_speculative_tokens,
        backend=str(payload.get("backend", "auto")).strip().lower(),
        strict=strict,
        text_only=bool(payload.get("text_only", draft_speculation.get("text_only", True))),
        allow_multimodal=bool(payload.get("allow_multimodal", draft_speculation.get("allow_multimodal", False))),
        assistant_repo_id=assistant_repo_id,
        assistant_local_path=assistant_local_path,
        assistant_revision=str(payload.get("assistant_revision") or draft_speculation.get("assistant_revision") or native_mtp.get("assistant_revision") or "main").strip(),
        assistant_dtype=str(payload.get("assistant_dtype") or draft_speculation.get("assistant_dtype") or native_mtp.get("assistant_dtype") or "auto").strip(),
        assistant_trust_remote_code=bool(payload.get("assistant_trust_remote_code") or draft_speculation.get("assistant_trust_remote_code") or native_mtp.get("assistant_trust_remote_code") or False),
        use_hf_token=bool(payload.get("use_hf_token") or draft_speculation.get("use_hf_token") or native_mtp.get("use_hf_token") or False),
        native_mtp=native_mtp,
        draft_speculation=draft_speculation,
        raw=raw,
    )


def detect_text_encoder_family(state_dict: dict[str, Any]) -> str:
    if not state_dict:
        return "unknown"
    if "model.layers.0.post_feedforward_layernorm.weight" in state_dict:
        if "model.layers.59.self_attn.q_norm.weight" in state_dict:
            return "gemma4_31b"
        if "model.layers.41.self_attn.q_norm.weight" in state_dict and "model.layers.47.self_attn.q_norm.weight" not in state_dict:
            return "gemma4_e4b"
        if "model.layers.34.self_attn.q_norm.weight" in state_dict and "model.layers.41.self_attn.q_norm.weight" not in state_dict:
            return "gemma4_e2b"
        if "model.layers.47.self_attn.q_norm.weight" in state_dict:
            return "gemma3_12b"
    if "model.language_model.layers.0.linear_attn.A_log" in state_dict and "model.language_model.layers.0.input_layernorm.weight" in state_dict:
        width = int(getattr(state_dict["model.language_model.layers.0.input_layernorm.weight"], "shape", [0])[0])
        return {
            1024: "qwen35_08b",
            2560: "qwen35_4b",
            4096: "qwen35_9b",
            5120: "qwen35_27b",
        }.get(width, "qwen35_2b")
    return "unknown"


def detect_gemma4_assistant_config(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type") or "").lower()
    architectures = {str(item).lower() for item in config.get("architectures") or []}
    if model_type == "gemma4_assistant" or "gemma4assistantforcausallm" in architectures:
        return "gemma4_assistant"
    return ""


def qwen_mtp_contract_status(model_path: str) -> AccelerationGate:
    runtime_contract = os.path.join(model_path, "mtplx_runtime.json")
    mtp_weights = os.path.join(model_path, "mtp.safetensors")
    if not os.path.exists(runtime_contract):
        return AccelerationGate(False, "missing_mtplx_runtime_contract", route="mlx_mtplx_qwen")
    if not os.path.exists(mtp_weights):
        return AccelerationGate(False, "missing_mtp_safetensors", route="mlx_mtplx_qwen")
    try:
        with open(runtime_contract, encoding="utf-8") as f:
            contract = json.load(f)
    except Exception as e:
        return AccelerationGate(False, f"invalid_mtplx_runtime_contract: {redact_secrets(e)}", route="mlx_mtplx_qwen")
    arch = str(contract.get("arch_id") or contract.get("architecture") or contract.get("model_arch") or "").lower()
    if arch and arch != "qwen3-next-mtp":
        return AccelerationGate(False, f"unsupported_mtplx_arch: {arch}", route="mlx_mtplx_qwen")
    return AccelerationGate(True, route="mlx_mtplx_qwen", mode="mtp", family="qwen3-next-mtp", backend="mlx")


def has_multimodal_tokens(tokens: Any) -> bool:
    if isinstance(tokens, dict):
        return any(has_multimodal_tokens(value) for value in tokens.values())
    if isinstance(tokens, (list, tuple)):
        for item in tokens:
            if isinstance(item, (list, tuple)):
                for part in item:
                    value = part[0] if isinstance(part, tuple) and part else part
                    if not isinstance(value, int):
                        return True
            elif not isinstance(item, int):
                return True
    return False


class TextGenerationAccelerator:
    def __init__(
        self,
        config: AccelerationConfig,
        *,
        target_clip_name: str,
        target_family: str,
        sidecar_path: str,
    ) -> None:
        self.config = config
        self.target_clip_name = target_clip_name
        self.target_family = target_family
        self.sidecar_path = sidecar_path
        self.loaded_at = time.time()
        self.last_gate: AccelerationGate | None = None
        self._assistant_model: Any | None = None
        self._assistant_cache_key: tuple[Any, ...] | None = None

    def capabilities(self) -> dict[str, Any]:
        gate = self.evaluate_gate(do_sample=False, tokens=None)
        return {
            "mode": self.config.mode,
            "strict": self.config.strict,
            "target_family": self.target_family,
            "target_clip_name": self.target_clip_name,
            "sidecar_path": self.sidecar_path,
            "supported": gate.supported,
            "route": gate.route,
            "reason": gate.reason,
        }

    def evaluate_gate(self, *, do_sample: bool, tokens: Any) -> AccelerationGate:
        cfg = self.config
        family = cfg.family or self.target_family
        if cfg.mode in {"", "off", "disabled"}:
            return AccelerationGate(False, "acceleration_off", mode="off", family=family, backend=cfg.backend)
        if cfg.backend not in SUPPORTED_BACKENDS:
            return AccelerationGate(False, f"unsupported_backend: {cfg.backend}", mode=cfg.mode, family=family, backend=cfg.backend)
        if has_multimodal_tokens(tokens) and not cfg.allow_multimodal:
            return AccelerationGate(False, "vlm_draft_not_verified", mode=cfg.mode, family=family, backend=cfg.backend)
        if cfg.native_mtp.get("enabled"):
            if family in QWEN_NATIVE_MTP_FAMILIES:
                return AccelerationGate(False, "torch_mps_qwen_native_mtp_verifier_not_implemented", route="torch_mps_qwen_native_mtp", mode=cfg.mode, family=family, backend=cfg.backend)
            if family in GEMMA4_FAMILIES:
                return AccelerationGate(False, "torch_mps_gemma4_native_mtp_verifier_not_implemented", route="torch_mps_gemma4_assistant", mode=cfg.mode, family=family, backend=cfg.backend)
            return AccelerationGate(False, "no_native_mtp_tensors", mode=cfg.mode, family=family, backend=cfg.backend)
        if cfg.draft_speculation.get("enabled") or cfg.assistant_clip_name or cfg.assistant_kind:
            if do_sample:
                return AccelerationGate(False, "sampled_speculation_not_implemented", mode=cfg.mode, family=family, backend=cfg.backend, assistant_kind=cfg.assistant_kind, assistant_clip_name=cfg.assistant_clip_name)
            if not cfg.assistant_clip_name:
                return AccelerationGate(False, "no_draft_model", mode=cfg.mode, family=family, backend=cfg.backend, assistant_kind=cfg.assistant_kind)
            if family in GEMMA4_FAMILIES and cfg.assistant_kind in {"", "mtp", "gemma4_mtp", "gemma4_assistant"}:
                if cfg.assistant_repo_id or cfg.assistant_local_path:
                    return AccelerationGate(True, route="torch_gemma4_assistant_hf", mode=cfg.mode, family=family, backend=cfg.backend, assistant_kind=cfg.assistant_kind or "gemma4_assistant", assistant_clip_name=cfg.assistant_clip_name, max_speculative_tokens=cfg.max_speculative_tokens)
                return AccelerationGate(False, "torch_mps_gemma4_assistant_verifier_not_implemented", route="torch_mps_gemma4_assistant", mode=cfg.mode, family=family, backend=cfg.backend, assistant_kind=cfg.assistant_kind or "gemma4_assistant", assistant_clip_name=cfg.assistant_clip_name, max_speculative_tokens=cfg.max_speculative_tokens)
            return AccelerationGate(False, "target_draft_pair_not_verified", mode=cfg.mode, family=family, backend=cfg.backend, assistant_kind=cfg.assistant_kind, assistant_clip_name=cfg.assistant_clip_name)
        return AccelerationGate(False, "no_acceleration_path_configured", mode=cfg.mode, family=family, backend=cfg.backend)

    def generate(self, original_generate: Callable[[], Any], *, tokens: Any, do_sample: bool, max_length: int, sampler: dict[str, Any], runtime_model: Any | None = None) -> Any:
        gate = self.evaluate_gate(do_sample=do_sample, tokens=tokens)
        self.last_gate = gate
        write_event(
            "textgen_accel_gate",
            backend="torch",
            route=gate.route,
            mode=gate.mode,
            supported=gate.supported,
            reason=gate.reason,
            family=gate.family,
            target_clip_name=self.target_clip_name,
            assistant_clip_name=gate.assistant_clip_name,
            assistant_kind=gate.assistant_kind,
            max_speculative_tokens=gate.max_speculative_tokens,
            sidecar_path=self.sidecar_path,
            do_sample=do_sample,
            max_tokens=max_length,
            sampler=sampler,
        )
        if gate.supported:
            if gate.route == "torch_gemma4_assistant_hf":
                return self._generate_gemma4_assistant_hf(
                    runtime_model=runtime_model,
                    tokens=tokens,
                    max_length=max_length,
                    sampler=sampler,
                    gate=gate,
                )
            raise RuntimeError("Internal text-generation acceleration selected a route without an implemented verifier.")
        if self.config.strict:
            raise RuntimeError(f"TextGenerate acceleration is strict but unavailable: {gate.reason}")
        start = now()
        try:
            output = original_generate()
        except BaseException as e:
            write_event(
                "textgen_accel_fallback_error",
                backend="torch",
                route=gate.route,
                reason=gate.reason,
                family=gate.family,
                error=redact_secrets(e),
                seconds=elapsed_seconds(start),
            )
            raise
        write_event(
            "textgen_accel_fallback",
            backend="torch",
            route=gate.route,
            reason=gate.reason,
            family=gate.family,
            target_clip_name=self.target_clip_name,
            assistant_clip_name=gate.assistant_clip_name,
            assistant_kind=gate.assistant_kind,
            seconds=elapsed_seconds(start),
            output_tokens=len(output) if isinstance(output, list) else None,
        )
        return output

    def _generate_gemma4_assistant_hf(self, *, runtime_model: Any | None, tokens: Any, max_length: int, sampler: dict[str, Any], gate: AccelerationGate) -> Any:
        target = _unwrap_runtime_clip(runtime_model)
        generate_with_assistant = getattr(target, "generate_with_gemma4_assistant", None)
        if generate_with_assistant is None:
            raise RuntimeError("Gemma 4 assistant MTP selected, but the loaded runtime does not expose generate_with_gemma4_assistant().")
        start = now()
        assistant = self._load_gemma4_assistant(getattr(target, "execution_device", None))
        output = generate_with_assistant(
            tokens,
            assistant_model=assistant,
            max_speculative_tokens=max(1, int(gate.max_speculative_tokens or 4)),
            max_length=max_length,
            temperature=0.0,
            top_k=int(sampler.get("top_k", 0)),
            top_p=float(sampler.get("top_p", 1.0)),
            min_p=float(sampler.get("min_p", 0.0)),
            repetition_penalty=float(sampler.get("repetition_penalty", 1.0)),
            seed=sampler.get("seed"),
            presence_penalty=float(sampler.get("presence_penalty", 0.0)),
        )
        write_event(
            "textgen_accel_generate_end",
            backend="torch",
            route=gate.route,
            family=gate.family,
            target_clip_name=self.target_clip_name,
            assistant_clip_name=gate.assistant_clip_name,
            assistant_kind=gate.assistant_kind,
            max_speculative_tokens=gate.max_speculative_tokens,
            seconds=elapsed_seconds(start),
            output_tokens=len(output) if isinstance(output, list) else None,
        )
        return output

    def _load_gemma4_assistant(self, device: Any | None) -> Any:
        cfg = self.config
        cache_key = (cfg.assistant_repo_id, cfg.assistant_local_path, cfg.assistant_revision, cfg.assistant_dtype, str(device))
        if self._assistant_model is not None and self._assistant_cache_key == cache_key:
            return self._assistant_model
        try:
            import torch
            import transformers
        except Exception as e:
            raise RuntimeError(f"Transformers and torch are required for Gemma 4 assistant MTP: {redact_secrets(e)}") from e
        model_path = cfg.assistant_local_path
        if model_path:
            model_path = os.path.abspath(os.path.expanduser(model_path))
        else:
            try:
                from huggingface_hub import snapshot_download
            except Exception as e:
                raise RuntimeError(f"huggingface_hub is required to download Gemma 4 assistant models: {redact_secrets(e)}") from e
            try:
                model_path = snapshot_download(
                    repo_id=cfg.assistant_repo_id,
                    revision=cfg.assistant_revision or None,
                    cache_dir=_default_transformers_cache_dir(),
                    local_files_only=False,
                    token=True if cfg.use_hf_token else None,
                )
            except Exception as e:
                raise RuntimeError(f"Could not resolve Gemma 4 assistant model: {redact_secrets(e)}") from e
        dtype = _resolve_torch_dtype(torch, cfg.assistant_dtype)
        load_kwargs = {}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        assistant_cls = getattr(transformers, "Gemma4AssistantForCausalLM", None)
        if assistant_cls is None:
            raise RuntimeError("Installed transformers package does not expose Gemma4AssistantForCausalLM.")
        assistant = assistant_cls.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=cfg.assistant_trust_remote_code,
            **load_kwargs,
        )
        if device is not None:
            assistant.to(device)
        assistant.eval()
        self._assistant_model = assistant
        self._assistant_cache_key = cache_key
        return assistant


def make_accelerator_for_clip(model_path: str, state_dict: dict[str, Any]) -> TextGenerationAccelerator | None:
    sidecar_path, config = load_acceleration_config_for_path(model_path)
    if config is None or sidecar_path is None:
        return None
    return TextGenerationAccelerator(
        config,
        target_clip_name=os.path.basename(model_path),
        target_family=config.family or detect_text_encoder_family(state_dict),
        sidecar_path=sidecar_path,
    )


def apply_clip_model_options_for_sidecar(model_path: str, model_options: dict[str, Any]) -> dict[str, Any]:
    _, config = load_acceleration_config_for_path(model_path)
    if config is None:
        return model_options
    options = dict(model_options)
    device = _resolve_target_device(config.target_device)
    if device is not None:
        options["load_device"] = device
    offload_device = _resolve_target_device(config.target_offload_device)
    if offload_device is not None:
        options["offload_device"] = offload_device
    elif config.target_offload_device in {"same", "load"} and device is not None:
        options["offload_device"] = device
    initial_device = _resolve_target_device(config.target_initial_device)
    if initial_device is not None:
        options["initial_device"] = initial_device
    elif config.target_initial_device in {"same", "load"} and device is not None:
        options["initial_device"] = device
    if config.target_dtype:
        import torch

        dtype = _resolve_torch_dtype(torch, config.target_dtype)
        if dtype is not None:
            options["dtype"] = dtype
    return options


def _resolve_target_device(value: str):
    value = str(value or "").strip().lower().replace("-", "_")
    if value in {"", "default", "auto"}:
        return None
    import torch
    import comfy.model_management as model_management

    if value == "cpu":
        return torch.device("cpu")
    if value in {"gpu", "accelerator", "torch"}:
        return model_management.get_torch_device()
    if value == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return model_management.get_torch_device()
    if value == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return model_management.get_torch_device()
    if value in {"same", "load"}:
        return None
    raise RuntimeError(f"Unsupported text encoder target device: {value}")


def _unwrap_runtime_clip(runtime_model: Any | None) -> Any:
    if runtime_model is None:
        return None
    clip_name = getattr(runtime_model, "clip", None)
    if clip_name and hasattr(runtime_model, clip_name):
        return getattr(runtime_model, clip_name)
    return runtime_model


def _default_transformers_cache_dir() -> str | None:
    try:
        import folder_paths
        return os.path.join(folder_paths.get_folder_paths("transformers")[0], "huggingface")
    except Exception:
        return None


def _resolve_torch_dtype(torch, name: str):
    name = str(name or "").strip().lower()
    if name in {"", "auto", "none"}:
        return None
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(name, name)
    if not hasattr(torch, attr):
        raise RuntimeError(f"Unsupported Gemma 4 assistant dtype: {name}")
    return getattr(torch, attr)
