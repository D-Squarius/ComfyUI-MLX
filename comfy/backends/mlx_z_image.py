from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from comfy.backends.mlx_dit_planner import (
    candidate_env as dit_candidate_env,
    candidate_names,
    candidate_registry_json,
    linear_spec,
    multilinear_spec,
    normalize_linear_plan,
    planned_fused_linear,
    planned_linear,
)
from comfy.backends.mlx_quant import MLXLinearWeight, detect_quantization_bits, fuse_linear_weights_by_output, make_quantized_linear_module


Z_IMAGE_TURBO_MLX_ALIAS = "Z-Image Turbo MLX BF16 Island"
Z_IMAGE_TURBO_MLX_Q8_ALIAS = "Z-Image Turbo MLX Q8 Island"
Z_IMAGE_TURBO_NATIVE_FILE = "z_image_turbo_bf16.safetensors"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent
Z_IMAGE_TURBO_MLX_Q8_DEFAULT_PATH = Path(
    os.environ.get(
        "COMFY_MLX_Z_IMAGE_Q8_MODEL_PATH",
        str(_WORKSPACE_ROOT / "Vendor" / "z-image-models" / "Z-Image-Turbo-6B-MLX-Q8"),
    )
).expanduser()


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _profile_path() -> str | None:
    path = os.environ.get("COMFY_MLX_Z_IMAGE_PROFILE_PATH")
    if path:
        return path
    if not _env_flag("COMFY_MLX_Z_IMAGE_PROFILE", False):
        return None
    return os.environ.get("COMFY_BENCH_STATS_PATH")


def _write_profile_event(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def _json_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _json_output_shape(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_json_output_shape(item) for item in value]
    return _json_shape(value)


def _mlx_arrays(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        out: list[Any] = []
        for item in value:
            out.extend(_mlx_arrays(item))
        return out
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return [value]
    return []


def _block_family(prefix: str) -> str:
    return prefix.split(".", 1)[0]


def _block_index(prefix: str) -> int | None:
    parts = prefix.split(".", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


Q8_DEFAULT_DEQUANT_PATTERNS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
Q8_SPEED_DEQUANT_PATTERNS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
Z_IMAGE_Q8_PRODUCTION_PRESET = {
    "q8_linear_impl": "hybrid_dequant",
    "q8_dequant_patterns": ",".join(Q8_DEFAULT_DEQUANT_PATTERNS),
}
Z_IMAGE_Q8_STRATEGY_PRESETS = {
    "q8_baseline_debug": {"q8_linear_impl": "quantized_matmul", "q8_dequant_patterns": ""},
    "q8_hybrid_dequant_debug": {
        "q8_linear_impl": "hybrid_dequant",
        "q8_dequant_patterns": ",".join(Q8_SPEED_DEQUANT_PATTERNS),
    },
    "q8_hybrid_dequant_plus_debug": dict(Z_IMAGE_Q8_PRODUCTION_PRESET),
}
_DIT_ENV_TO_CANDIDATE_KEY = {
    "COMFY_MLX_DIT_LINEAR_PLAN": "dit_linear_plan",
    "COMFY_MLX_Z_IMAGE_ATTENTION_LAYOUT": "attention_layout",
    "COMFY_MLX_Z_IMAGE_COMPILE": "compile",
    "COMFY_MLX_Z_IMAGE_COMPILE_BLOCK_FILTER": "compile_block_filter",
    "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "native_ops",
    "COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "bf16_compute_dtype",
    "COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS": "bf16_compute_patterns",
}


def _z_image_dit_candidate_preset(name: str) -> dict[str, str]:
    return {
        _DIT_ENV_TO_CANDIDATE_KEY[key]: value
        for key, value in dit_candidate_env(name).items()
        if key in _DIT_ENV_TO_CANDIDATE_KEY
    }


Z_IMAGE_CANDIDATE_PRESETS = {
    "none": {},
    "bf16_compile_block": {"compile": "block"},
    "bf16_compile_main_layers": {"compile": "block", "compile_block_filter": "main_layers"},
    "bf16_compile_block_alloc_reduce": {"compile": "block", "block_alloc_reduce": "1"},
    "bf16_compile_transformer": {"compile": "transformer"},
    "bf16_block_alloc_reduce": {"block_alloc_reduce": "1"},
    "bf16_native_ffn_silu_mul": {"native_ops": "ffn_silu_mul"},
    "bf16_linear_flatten_2d": {"dit_linear_plan": "flatten_2d"},
    "bf16_linear_pretransposed": {"dit_linear_plan": "pretransposed"},
    "bf16_linear_flatten_pretransposed": {"dit_linear_plan": "flatten_pretransposed"},
    "bf16_linear_addmm_bias": {"dit_linear_plan": "addmm_bias"},
    "q8_hybrid_dequant_plus_debug": dict(Z_IMAGE_Q8_PRODUCTION_PRESET),
    "q8_compile_block": {**Z_IMAGE_Q8_PRODUCTION_PRESET, "compile": "block"},
    "q8_compile_transformer": {**Z_IMAGE_Q8_PRODUCTION_PRESET, "compile": "transformer"},
    "attention_layout_variant": {"attention_layout": "head_major_rope"},
    **{
        name: _z_image_dit_candidate_preset(name)
        for name in candidate_names(implemented_only=True)
    },
}


def z_image_candidate_env_overrides(candidate: str, *, backend: str = "") -> dict[str, str]:
    if candidate not in Z_IMAGE_CANDIDATE_PRESETS:
        raise ValueError(f"Unknown Z-Image candidate preset: {candidate}")
    if backend == "native_mlx_bf16" and (candidate.startswith("q8_") or candidate.startswith("q8.")):
        return {}
    if backend == "native_mlx_q8" and (candidate.startswith("bf16_") or ".bf16." in candidate):
        return {}
    preset = dict(Z_IMAGE_CANDIDATE_PRESETS[candidate])
    if backend == "native_mlx_bf16":
        preset.pop("q8_linear_impl", None)
        preset.pop("q8_dequant_patterns", None)
    return {key: str(value) for key, value in preset.items() if str(value) != ""}


def z_image_q8_strategy_env(strategy: str) -> dict[str, str]:
    if not strategy:
        return {}
    if strategy not in Z_IMAGE_Q8_STRATEGY_PRESETS:
        raise ValueError(f"Unknown Z-Image Q8 strategy: {strategy}")
    return {key: str(value) for key, value in Z_IMAGE_Q8_STRATEGY_PRESETS[strategy].items()}


def z_image_native_q8_report_label(_result: dict[str, Any] | None, *, multiline: bool = False) -> str:
    return "ComfyUI Native MLX Island Q8"


def _q8_linear_impl() -> str:
    impl = os.environ.get("COMFY_MLX_Z_IMAGE_Q8_LINEAR_IMPL", "hybrid_dequant").strip().lower()
    if impl in {"", "default"}:
        return "hybrid_dequant"
    if impl not in {"quantized_matmul", "nn_quantized", "hybrid_dequant", "dequantized"}:
        return "hybrid_dequant"
    return impl


def _dequant_patterns() -> tuple[str, ...]:
    raw = os.environ.get("COMFY_MLX_Z_IMAGE_Q8_DEQUANT_PATTERNS")
    if raw is None:
        return Q8_DEFAULT_DEQUANT_PATTERNS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _attention_layout() -> str:
    layout = os.environ.get("COMFY_MLX_Z_IMAGE_ATTENTION_LAYOUT", "seq_major").strip().lower()
    if layout in {"", "default"}:
        return "seq_major"
    if layout not in {"seq_major", "head_major_rope"}:
        return "seq_major"
    return layout


def _compile_mode_parts() -> set[str]:
    raw = os.environ.get("COMFY_MLX_Z_IMAGE_COMPILE", "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return set()
    if raw in {"1", "true", "yes", "on", "full"}:
        return {"transformer"}
    return {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}


def _compile_block_filter() -> str:
    raw = os.environ.get("COMFY_MLX_Z_IMAGE_COMPILE_BLOCK_FILTER", "all").strip().lower()
    if raw in {"", "default"}:
        return "all"
    if raw not in {"all", "main_layers", "refiners", "none"}:
        return "all"
    return raw


def _native_ops() -> set[str]:
    raw = os.environ.get("COMFY_MLX_Z_IMAGE_NATIVE_OPS", "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return set()
    if raw in {"1", "true", "yes", "on", "all"}:
        return {
            "attn_prep_qknorm_rope_pack",
            "ffn_silu_mul",
            "ffn_packed_silu_mul_split",
            "native_bf16_self_attn",
        }
    return {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}


def _dit_linear_plan() -> str:
    return normalize_linear_plan(os.environ.get("COMFY_MLX_DIT_LINEAR_PLAN"))


def _bf16_compute_dtype_policy() -> str:
    raw = (
        os.environ.get("COMFY_MLX_DIT_BF16_COMPUTE_DTYPE")
        or os.environ.get("COMFY_MLX_Z_IMAGE_BF16_COMPUTE_DTYPE")
        or ""
    )
    value = raw.strip().lower()
    if value in {"fp16", "float16", "half"}:
        return "float16"
    if value in {"fp16_no_cast", "float16_no_cast", "half_no_cast", "float16_keep_output"}:
        return "float16_no_cast"
    return "native"


def _bf16_compute_patterns() -> tuple[str, ...]:
    raw = (
        os.environ.get("COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS")
        or os.environ.get("COMFY_MLX_Z_IMAGE_BF16_COMPUTE_PATTERNS")
        or ""
    ).strip()
    if not raw or raw.lower() in {"1", "true", "yes", "on", "all"}:
        return ()
    return tuple(part.strip() for part in raw.replace(";", ",").split(",") if part.strip())


def _matches_any_pattern(value: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in value for pattern in patterns)


def _silu(mx: Any, x: Any) -> Any:
    return x * mx.sigmoid(x)


def _rms_norm(mx: Any, x: Any, weight: Any, eps: float = 1.0e-5) -> Any:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * weight.astype(mx.float32)).astype(dtype)


def _layer_norm(mx: Any, x: Any, eps: float = 1.0e-6) -> Any:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * mx.rsqrt(var + eps)).astype(dtype)


def _timestep_embedding(mx: Any, t: Any, dim: int, max_period: int = 10000) -> Any:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


def _rope_axis(mx: Any, pos: Any, dim: int, theta: float) -> Any:
    scale = mx.linspace(0.0, float(dim - 2) / float(dim), dim // 2, dtype=mx.float32)
    omega = 1.0 / (theta**scale)
    out = pos.astype(mx.float32)[..., None] * omega[None, None]
    matrix = mx.stack([mx.cos(out), -mx.sin(out), mx.sin(out), mx.cos(out)], axis=-1)
    return matrix.reshape(*out.shape, 2, 2)


def _embed_nd(mx: Any, position_ids: Any, axes_dims: tuple[int, int, int], theta: float) -> Any:
    parts = [_rope_axis(mx, position_ids[..., axis], axes_dims[axis], theta) for axis in range(len(axes_dims))]
    return mx.concatenate(parts, axis=-3)[:, :, None]


def _apply_rope(mx: Any, x: Any, freqs_cis: Any) -> Any:
    dtype = x.dtype
    x_ = x.astype(freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, : x_.shape[2]]
    out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).astype(dtype)


def _apply_rope_head_major(mx: Any, x: Any, freqs_cis: Any) -> Any:
    dtype = x.dtype
    # x is [B, H, S, D], while cached frequencies are [B, S, 1, D/2, 2, 2].
    freqs_cis = freqs_cis.transpose(0, 2, 1, 3, 4, 5)
    x_ = x.astype(freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).astype(dtype)


def _pad_to_multiple(mx: Any, x: Any, pad_token: Any, multiple: int) -> tuple[Any, int]:
    pad_extra = (-int(x.shape[1])) % int(multiple)
    if pad_extra == 0:
        return x, 0
    pad = mx.repeat(pad_token.astype(x.dtype)[None], int(x.shape[0]), axis=0)
    pad = mx.repeat(pad, pad_extra, axis=1)
    return mx.concatenate([x, pad], axis=1), pad_extra


def _repeat_to_batch(mx: Any, x: Any, batch_size: int) -> Any:
    if int(x.shape[0]) == batch_size:
        return x
    if int(x.shape[0]) != 1:
        raise ValueError(f"Cannot repeat batch {x.shape[0]} to {batch_size}.")
    return mx.repeat(x, batch_size, axis=0)


def _torch_tensor_to_mlx(mx: Any, tensor: torch.Tensor, dtype: Any) -> Any:
    if tensor.dtype is torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return mx.array(tensor.detach().cpu().numpy()).astype(dtype)


class MLXZImageWeights:
    def __init__(self, path: str | Path, *, dtype: Any):
        import mlx.core as mx

        self.path = str(Path(path).expanduser())
        self.dtype = dtype
        self.mx = mx
        self.group_size = 64
        self.quantization_level: str | None = None
        self._weights: dict[str, Any] | None = None
        self._linear_refs: dict[tuple[str, bool], MLXLinearWeight] = {}
        self.linear_impl = _q8_linear_impl()
        self.dequant_patterns = _dequant_patterns()
        self.bf16_compute_dtype_policy = _bf16_compute_dtype_policy()
        self.bf16_compute_patterns = _bf16_compute_patterns()

    @property
    def weights(self) -> dict[str, Any]:
        if self._weights is None:
            self._weights = self._load()
        return self._weights

    def _load(self) -> dict[str, Any]:
        from safetensors import safe_open

        loaded: dict[str, Any] = {}
        path = Path(self.path)
        if path.is_dir():
            index = path / "transformer" / "model.safetensors.index.json"
            if not index.exists():
                index = path / "model.safetensors.index.json"
            data = json.loads(index.read_text(encoding="utf-8"))
            self.quantization_level = (data.get("metadata") or {}).get("quantization_level")
            weight_map = data.get("weight_map") or {}
            files = sorted(set(weight_map.values()))
            for file_name in files:
                shard = index.parent / file_name
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if weight_map.get(key) != file_name:
                            continue
                        loaded[key] = self._convert_tensor(handle.get_tensor(key))
        else:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    loaded[key] = self._convert_tensor(handle.get_tensor(key))
        return loaded

    def _convert_tensor(self, tensor: torch.Tensor) -> Any:
        if tensor.dtype is torch.bfloat16 or torch.is_floating_point(tensor):
            return _torch_tensor_to_mlx(self.mx, tensor, self.dtype)
        return self.mx.array(tensor.numpy())

    def get(self, key: str) -> Any:
        return self.weights[key]

    def maybe(self, key: str) -> Any | None:
        return self.weights.get(key)

    def has(self, key: str) -> bool:
        return key in self.weights

    def first(self, *keys: str) -> Any:
        for key in keys:
            value = self.maybe(key)
            if value is not None:
                return value
        raise KeyError("Missing all candidate Z-Image weight keys: " + ", ".join(keys))

    def first_base(self, *bases: str) -> str:
        for base in bases:
            if self.has(base + ".weight"):
                return base
        raise KeyError("Missing all candidate Z-Image linear keys: " + ", ".join(bases))

    def linear_ref(self, base: str, *, bias: bool = True) -> MLXLinearWeight:
        cache_key = (base, bool(bias))
        cached = self._linear_refs.get(cache_key)
        if cached is not None:
            return cached

        weight = self.get(base + ".weight")
        scales = self.maybe(base + ".scales")
        biases = self.maybe(base + ".biases")
        bias_weight = self.maybe(base + ".bias") if bias else None
        bits = None
        if scales is not None:
            bits = int(self.quantization_level) if self.quantization_level is not None else detect_quantization_bits(weight, scales, self.group_size)
        impl = self.linear_impl if scales is not None else "dense"
        module = None
        if scales is not None and impl in {"dequantized", "hybrid_dequant"}:
            should_dequantize = impl == "dequantized" or _matches_any_pattern(base, self.dequant_patterns)
            if should_dequantize:
                weight = self.mx.dequantize(
                    weight,
                    scales,
                    biases,
                    group_size=self.group_size,
                    bits=bits,
                    mode="affine",
                    dtype=self.dtype,
                )
                scales = None
                biases = None
                impl = "dequantized"
            else:
                impl = "quantized_matmul"
        ref = MLXLinearWeight(
            weight=weight,
            bias=bias_weight,
            scales=scales,
            biases=biases,
            group_size=self.group_size,
            bits=bits,
            name=base,
            impl=impl,
        )
        if scales is not None and impl == "nn_quantized":
            module = make_quantized_linear_module(self.mx, ref)
            ref = MLXLinearWeight(
                weight=ref.weight,
                bias=ref.bias,
                scales=ref.scales,
                biases=ref.biases,
                group_size=ref.group_size,
                bits=ref.bits,
                mode=ref.mode,
                name=ref.name,
                impl=impl,
                module=module,
            )
        bf16_compute_allowed = not self.bf16_compute_patterns or _matches_any_pattern(base, self.bf16_compute_patterns)
        if (
            scales is None
            and bf16_compute_allowed
            and self.bf16_compute_dtype_policy in {"float16", "float16_no_cast"}
            and weight.dtype == self.mx.bfloat16
        ):
            ref = MLXLinearWeight(
                weight=weight.astype(self.mx.float16),
                bias=bias_weight.astype(self.mx.float16) if bias_weight is not None else None,
                group_size=self.group_size,
                name=base,
                impl="dense_fp16_compute" if self.bf16_compute_dtype_policy == "float16" else "dense_fp16_compute_no_cast",
                compute_dtype=self.mx.float16,
                output_dtype=self.mx.bfloat16 if self.bf16_compute_dtype_policy == "float16" else None,
            )
        self._linear_refs[cache_key] = ref
        return ref

    def first_linear_ref(self, *bases: str, bias: bool = True) -> MLXLinearWeight:
        return self.linear_ref(self.first_base(*bases), bias=bias)

    def linear(self, x: Any, base: str, *, bias: bool = True) -> Any:
        return self.linear_ref(base, bias=bias)(self.mx, x)


class MLXZImageBlock:
    def __init__(self, weights: MLXZImageWeights, prefix: str, *, modulation: bool, dim: int = 3840, heads: int = 30):
        self.weights = weights
        self.prefix = prefix
        self.modulation = bool(modulation)
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.use_fused_qkv = _env_flag("COMFY_MLX_Z_IMAGE_FUSE_QKV", False)
        self.use_fused_ffn = _env_flag("COMFY_MLX_Z_IMAGE_FUSE_FFN", False)
        self.use_block_alloc_reduce = _env_flag("COMFY_MLX_Z_IMAGE_BLOCK_ALLOC_REDUCE", False)
        self.attention_layout = _attention_layout()
        self.linear_plan = _dit_linear_plan()
        self.compile_modes = _compile_mode_parts()
        self.compile_block_filter = _compile_block_filter()
        self.native_ops = _native_ops()
        self.use_packed_ffn_activation = "ffn_packed_silu_mul_split" in self.native_ops
        self._compiled_modulated_call: Any | None = None
        self._compiled_plain_call: Any | None = None
        self._compiled_ffn_call: Any | None = None
        self._compiled_ffn_disabled = False
        self._compile_disabled = False
        self.profile_path = _profile_path()
        self.profile_call_index: int | None = None
        self.attention_q_norm_weight = self._maybe_cache_norm_weight(self._first_w("attention.q_norm.weight", "attention.norm_q.weight"))
        self.attention_k_norm_weight = self._maybe_cache_norm_weight(self._first_w("attention.k_norm.weight", "attention.norm_k.weight"))
        self.attention_out = self._plan_linear(
            self.weights.first_linear_ref(f"{self.prefix}.attention.out", f"{self.prefix}.attention.to_out.0"),
            role="attention_out",
        )
        if self.weights.has(f"{self.prefix}.attention.to_q.weight"):
            self.attention_q = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.attention.to_q", bias=False), role="attention_q")
            self.attention_k = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.attention.to_k", bias=False), role="attention_k")
            self.attention_v = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.attention.to_v", bias=False), role="attention_v")
            self.attention_qkv = (
                planned_fused_linear(
                    self.weights.mx,
                    [self.attention_q, self.attention_k, self.attention_v],
                    name=f"{self.prefix}.attention.to_qkv_fused",
                    plan=self.linear_plan,
                    role="attention_qkv",
                )
                if self.use_fused_qkv
                else None
            )
        else:
            self.attention_q = None
            self.attention_k = None
            self.attention_v = None
            self.attention_qkv = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.attention.qkv"), role="attention_qkv")
        self.feed_forward_w1 = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.feed_forward.w1", bias=False), role="ffn_gate")
        self.feed_forward_w2 = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.feed_forward.w2", bias=False), role="ffn_down")
        self.feed_forward_w3 = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.feed_forward.w3", bias=False), role="ffn_up")
        self.feed_forward_w1_w3 = (
            planned_fused_linear(
                self.weights.mx,
                [self.feed_forward_w1, self.feed_forward_w3],
                name=f"{self.prefix}.feed_forward.w1_w3_fused",
                plan=self.linear_plan,
                role="ffn_gate_up",
            )
            if self.use_fused_ffn or self.use_packed_ffn_activation
            else None
        )
        self.attention_norm1_weight = self._maybe_cache_norm_weight(self._w("attention_norm1.weight"))
        self.attention_norm2_weight = self._maybe_cache_norm_weight(self._w("attention_norm2.weight"))
        self.ffn_norm1_weight = self._maybe_cache_norm_weight(self._w("ffn_norm1.weight"))
        self.ffn_norm2_weight = self._maybe_cache_norm_weight(self._w("ffn_norm2.weight"))
        self.adaln = self._plan_linear(self.weights.linear_ref(f"{self.prefix}.adaLN_modulation.0"), role="adaln") if self.modulation else None

    def _plan_linear(self, linear: MLXLinearWeight, *, role: str) -> Any:
        return planned_linear(self.weights.mx, linear, plan=self.linear_plan, role=role)

    def _maybe_cache_norm_weight(self, weight: Any) -> Any:
        return weight.astype(self.weights.mx.float32) if self.use_block_alloc_reduce else weight

    def _profile(self, mx: Any, stage: str, fn: Any, *, input_value: Any | None = None, linears: tuple[MLXLinearWeight, ...] = ()) -> Any:
        if not self.profile_path:
            return fn()
        start = time.perf_counter()
        out = fn()
        arrays = _mlx_arrays(out)
        if arrays:
            mx.eval(*arrays)
        _write_profile_event(
            self.profile_path,
            {
                "event": "z_image_block_profile",
                "block": self.prefix,
                "block_family": _block_family(self.prefix),
                "block_index": _block_index(self.prefix),
                "call_index": self.profile_call_index,
                "stage": stage,
                "input_shape": _json_shape(input_value),
                "output_shape": _json_output_shape(out),
                "linear_backend": linears[0].impl if linears else None,
                "linear_names": [linear.name for linear in linears],
                "quantized": any(linear.quantized for linear in linears),
                "bits": linears[0].bits if linears else None,
                "group_size": linears[0].group_size if linears else None,
                "seconds": time.perf_counter() - start,
                "fused_qkv": self.use_fused_qkv,
                "fused_ffn": self.use_fused_ffn,
                "packed_ffn_activation": self.use_packed_ffn_activation,
                "block_alloc_reduce": self.use_block_alloc_reduce,
                "attention_layout": self.attention_layout,
                "linear_plan": self.linear_plan,
                "compile_block_filter": self.compile_block_filter,
                "native_ops": sorted(self.native_ops),
            },
        )
        return out

    def shape_census(self, *, stream_tokens: int) -> dict[str, Any]:
        attention: dict[str, Any] = {
            "out": linear_spec(self.attention_out, role="attention_out").to_json(),
            "heads": self.heads,
            "head_dim": self.head_dim,
            "layout": self.attention_layout,
        }
        if self.attention_q is not None:
            attention.update(
                {
                    "q": linear_spec(self.attention_q, role="attention_q").to_json(),
                    "k": linear_spec(self.attention_k, role="attention_k").to_json(),
                    "v": linear_spec(self.attention_v, role="attention_v").to_json(),
                    "qkv_group": multilinear_spec(
                        f"{self.prefix}.attention.qkv_group",
                        "attention_qkv",
                        [self.attention_q, self.attention_k, self.attention_v],
                        ["attention_q", "attention_k", "attention_v"],
                    ).to_json(),
                }
            )
        else:
            attention["qkv"] = linear_spec(self.attention_qkv, role="attention_qkv").to_json()

        return {
            "name": self.prefix,
            "family": _block_family(self.prefix),
            "index": _block_index(self.prefix),
            "stream_tokens": int(stream_tokens),
            "hidden_dim": self.dim,
            "attention": attention,
            "ffn": {
                "gate": linear_spec(self.feed_forward_w1, role="ffn_gate").to_json(),
                "up": linear_spec(self.feed_forward_w3, role="ffn_up").to_json(),
                "down": linear_spec(self.feed_forward_w2, role="ffn_down").to_json(),
                "gate_up_group": multilinear_spec(
                    f"{self.prefix}.feed_forward.gate_up_group",
                    "ffn_gate_up",
                    [self.feed_forward_w1, self.feed_forward_w3],
                    ["ffn_gate", "ffn_up"],
                ).to_json(),
            },
            "linear_plan": self.linear_plan,
            "attention_layout": self.attention_layout,
            "layout_contracts": {
                "stream": "BTH",
                "qkv_projection": "BTH",
                "qk_norm_rope_input": "BTHD",
                "sdpa_input": "BHTD",
                "ffn_activation": "BTH",
            },
            "planner_candidates": candidate_registry_json(),
        }

    def _compile_allowed_for_prefix(self) -> bool:
        if self.compile_block_filter == "none":
            return False
        if self.compile_block_filter == "main_layers":
            return self.prefix.startswith("layers.")
        if self.compile_block_filter == "refiners":
            return self.prefix.startswith("noise_refiner.") or self.prefix.startswith("context_refiner.")
        return True

    def _w(self, suffix: str) -> Any:
        return self.weights.get(f"{self.prefix}.{suffix}")

    def _maybe(self, suffix: str) -> Any | None:
        return self.weights.maybe(f"{self.prefix}.{suffix}")

    def _first_w(self, *suffixes: str) -> Any:
        return self.weights.first(*(f"{self.prefix}.{suffix}" for suffix in suffixes))

    def _linear(self, suffix: str, x: Any, *, bias: bool = True) -> Any:
        return self.weights.linear(x, f"{self.prefix}.{suffix}", bias=bias)

    def _first_linear(self, x: Any, *suffixes: str, bias: bool = True) -> Any:
        return self.weights.linear(x, self.weights.first_base(*(f"{self.prefix}.{suffix}" for suffix in suffixes)), bias=bias)

    def _attention(self, mx: Any, x: Any, mask: Any, freqs_cis: Any) -> Any:
        bsz, seqlen, _ = x.shape
        q_size = self.heads * self.head_dim
        k_size = self.heads * self.head_dim
        if self.attention_q is not None and not self.use_fused_qkv:
            q, k, v = self._profile(
                mx,
                "qkv_projection_split",
                lambda: (self.attention_q(mx, x), self.attention_k(mx, x), self.attention_v(mx, x)),
                input_value=x,
                linears=(self.attention_q, self.attention_k, self.attention_v),
            )
        else:
            q, k, v = self._profile(
                mx,
                "qkv_projection_fused",
                lambda: mx.split(self.attention_qkv(mx, x), [q_size, q_size + k_size], axis=-1),
                input_value=x,
                linears=(self.attention_qkv,),
            )
        if "attn_prep_qknorm_rope_pack" in self.native_ops:
            q = q.reshape(bsz, seqlen, self.heads, self.head_dim)
            k = k.reshape(bsz, seqlen, self.heads, self.head_dim)
            v = v.reshape(bsz, seqlen, self.heads, self.head_dim)
            q, k, v = self._profile(
                mx,
                "attn_prep_qknorm_rope_pack",
                lambda: self._attn_prep_qknorm_rope_pack(mx, q, k, v, freqs_cis),
                input_value=q,
            )
        elif self.attention_layout == "head_major_rope":
            q = q.reshape(bsz, seqlen, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(bsz, seqlen, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(bsz, seqlen, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            q, k = self._profile(
                mx,
                "qk_norm_rope_head_major",
                lambda: (
                    _apply_rope_head_major(mx, _rms_norm(mx, q, self.attention_q_norm_weight, eps=1.0e-5), freqs_cis),
                    _apply_rope_head_major(mx, _rms_norm(mx, k, self.attention_k_norm_weight, eps=1.0e-5), freqs_cis),
                ),
                input_value=q,
            )
        else:
            q = q.reshape(bsz, seqlen, self.heads, self.head_dim)
            k = k.reshape(bsz, seqlen, self.heads, self.head_dim)
            v = v.reshape(bsz, seqlen, self.heads, self.head_dim)
            q, k = self._profile(
                mx,
                "qk_norm_rope",
                lambda: (
                    _apply_rope(mx, _rms_norm(mx, q, self.attention_q_norm_weight, eps=1.0e-5), freqs_cis),
                    _apply_rope(mx, _rms_norm(mx, k, self.attention_k_norm_weight, eps=1.0e-5), freqs_cis),
                ),
                input_value=q,
            )
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
        if mask is not None:
            raise ValueError("Z-Image MLX island v1 does not support attention masks.")
        if "sdpa_contiguous_inputs" in self.native_ops:
            q, k, v = self._profile(
                mx,
                "sdpa_contiguous_inputs",
                lambda: (mx.contiguous(q), mx.contiguous(k), mx.contiguous(v)),
                input_value=q,
            )
        out = self._profile(mx, "sdpa", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5), input_value=q)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seqlen, self.dim)
        if "attn_output_contiguous" in self.native_ops:
            out = self._profile(mx, "attention_output_contiguous", lambda: mx.contiguous(out), input_value=out)
        return self._profile(
            mx,
            "attention_output_projection",
            lambda: self.attention_out(mx, out),
            input_value=out,
            linears=(self.attention_out,),
        )

    def _feed_forward_eager(self, mx: Any, x: Any) -> Any:
        if self.use_packed_ffn_activation:
            return self.feed_forward_w2(mx, self._ffn_packed_silu_mul_split(mx, self.feed_forward_w1_w3(mx, x)))
        if self.use_fused_ffn:
            w1, w3 = mx.split(self.feed_forward_w1_w3(mx, x), [int(self.feed_forward_w1.weight.shape[0])], axis=-1)
        else:
            w1, w3 = self.feed_forward_w1(mx, x), self.feed_forward_w3(mx, x)
        return self.feed_forward_w2(mx, self._ffn_silu_mul(mx, w1, w3))

    def _feed_forward_compiled(self, mx: Any, x: Any) -> Any | None:
        if "ffn_region" not in self.compile_modes or self.profile_path or self._compiled_ffn_disabled:
            return None
        compile_fn = getattr(mx, "compile", None)
        if compile_fn is None:
            self._compiled_ffn_disabled = True
            return None
        try:
            if self._compiled_ffn_call is None:
                self._compiled_ffn_call = compile_fn(lambda x: self._feed_forward_eager(mx, x))
            return self._compiled_ffn_call(x)
        except Exception:
            self._compiled_ffn_call = None
            self._compiled_ffn_disabled = True
            return None

    def _feed_forward(self, mx: Any, x: Any) -> Any:
        compiled = self._feed_forward_compiled(mx, x)
        if compiled is not None:
            return compiled

        if self.use_packed_ffn_activation:
            ffn_act = self._profile(
                mx,
                "ffn_gate_up_activation_packed_native",
                lambda: self._ffn_packed_silu_mul_split(mx, self.feed_forward_w1_w3(mx, x)),
                input_value=x,
                linears=(self.feed_forward_w1_w3,),
            )
            return self._profile(
                mx,
                "ffn_down_projection",
                lambda: self.feed_forward_w2(mx, ffn_act),
                input_value=ffn_act,
                linears=(self.feed_forward_w2,),
            )

        if self.use_fused_ffn:
            w1, w3 = self._profile(
                mx,
                "ffn_gate_up_projection_fused",
                lambda: mx.split(self.feed_forward_w1_w3(mx, x), [int(self.feed_forward_w1.weight.shape[0])], axis=-1),
                input_value=x,
                linears=(self.feed_forward_w1_w3,),
            )
        else:
            w1, w3 = self._profile(
                mx,
                "ffn_gate_up_projection_split",
                lambda: (self.feed_forward_w1(mx, x), self.feed_forward_w3(mx, x)),
                input_value=x,
                linears=(self.feed_forward_w1, self.feed_forward_w3),
            )
        ffn_act_stage = "ffn_silu_mul_native" if "ffn_silu_mul" in self.native_ops else "ffn_silu_mul"
        ffn_act = self._profile(mx, ffn_act_stage, lambda: self._ffn_silu_mul(mx, w1, w3), input_value=w1)
        return self._profile(
            mx,
            "ffn_down_projection",
            lambda: self.feed_forward_w2(mx, ffn_act),
            input_value=w1,
            linears=(self.feed_forward_w2,),
        )

    def _ffn_silu_mul(self, mx: Any, gate: Any, up: Any) -> Any:
        if "ffn_silu_mul" not in self.native_ops:
            return _silu(mx, gate) * up
        from comfy.backends.mlx_dit_native import kernels as dit_native

        return dit_native.ffn_silu_mul(gate, up)

    def _ffn_packed_silu_mul_split(self, mx: Any, packed_gate_up: Any) -> Any:
        from comfy.backends.mlx_dit_native import kernels as dit_native

        hidden_dim = int(self.feed_forward_w1.weight.shape[0])
        return dit_native.ffn_packed_silu_mul_split(packed_gate_up, hidden_dim)

    def _attn_prep_qknorm_rope_pack(self, mx: Any, q: Any, k: Any, v: Any, freqs_cis: Any) -> Any:
        from comfy.backends.mlx_dit_native import kernels as dit_native

        return dit_native.attn_prep_qknorm_rope_pack(
            q,
            k,
            v,
            self.attention_q_norm_weight,
            self.attention_k_norm_weight,
            freqs_cis,
            eps=1.0e-5,
        )

    def _rmsnorm_residual_gate(self, mx: Any, value: Any, weight: Any, residual: Any, gate: Any) -> Any:
        if "rmsnorm_residual_gate" not in self.native_ops:
            return residual + gate * _rms_norm(mx, value, weight, eps=1.0e-5)
        from comfy.backends.mlx_dit_native import kernels as dit_native

        return dit_native.rmsnorm_residual_gate(value, weight, residual, gate, eps=1.0e-5)

    def _call_impl(self, mx: Any, x: Any, mask: Any, freqs_cis: Any, adaln_input: Any | None = None) -> Any:
        if self.modulation:
            if adaln_input is None:
                raise ValueError(f"{self.prefix} requires AdaLN input.")
            adaln = self.adaln(mx, adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(adaln, 4, axis=-1)
            attn_in = _rms_norm(mx, x, self.attention_norm1_weight, eps=1.0e-5) * (1.0 + scale_msa[:, None])
            attn_out = self._attention(mx, attn_in, mask, freqs_cis)
            gate_msa = mx.tanh(gate_msa[:, None])
            x = self._rmsnorm_residual_gate(mx, attn_out, self.attention_norm2_weight, x, gate_msa)
            ffn_in = _rms_norm(mx, x, self.ffn_norm1_weight, eps=1.0e-5) * (1.0 + scale_mlp[:, None])
            ffn_out = self._feed_forward(mx, ffn_in)
            gate_mlp = mx.tanh(gate_mlp[:, None])
            x = self._rmsnorm_residual_gate(mx, ffn_out, self.ffn_norm2_weight, x, gate_mlp)
            return x

        if adaln_input is not None:
            raise ValueError(f"{self.prefix} does not use AdaLN input.")
        attn_out = self._attention(mx, _rms_norm(mx, x, self.attention_norm1_weight, eps=1.0e-5), mask, freqs_cis)
        x = x + _rms_norm(mx, attn_out, self.attention_norm2_weight, eps=1.0e-5)
        ffn_out = self._feed_forward(mx, _rms_norm(mx, x, self.ffn_norm1_weight, eps=1.0e-5))
        return x + _rms_norm(mx, ffn_out, self.ffn_norm2_weight, eps=1.0e-5)

    def __call__(self, mx: Any, x: Any, mask: Any, freqs_cis: Any, adaln_input: Any | None = None) -> Any:
        compile_block = ("block" in self.compile_modes or "blocks" in self.compile_modes) and self._compile_allowed_for_prefix()
        if compile_block and not self.profile_path and mask is None and not self._compile_disabled:
            compile_fn = getattr(mx, "compile", None)
            if compile_fn is None:
                self._compile_disabled = True
            else:
                try:
                    if self.modulation:
                        if self._compiled_modulated_call is None:
                            self._compiled_modulated_call = compile_fn(lambda x, freqs_cis, adaln_input: self._call_impl(mx, x, None, freqs_cis, adaln_input))
                        return self._compiled_modulated_call(x, freqs_cis, adaln_input)
                    if self._compiled_plain_call is None:
                        self._compiled_plain_call = compile_fn(lambda x, freqs_cis: self._call_impl(mx, x, None, freqs_cis, None))
                    return self._compiled_plain_call(x, freqs_cis)
                except Exception:
                    self._compiled_modulated_call = None
                    self._compiled_plain_call = None
                    self._compile_disabled = True

        return self._call_impl(mx, x, mask, freqs_cis, adaln_input)


class MLXZImageTransformer:
    """Native-Comfy-key Z-Image Turbo transformer implemented with MLX arrays."""

    def __init__(self, model_path: str | Path, *, dtype: Any | None = None):
        import mlx.core as mx

        self.mx = mx
        self.dtype = dtype if dtype is not None else mx.bfloat16
        self.weights = MLXZImageWeights(model_path, dtype=self.dtype)
        self.dim = 3840
        self.in_channels = 16
        self.patch_size = 2
        self.heads = 30
        self.head_dim = 128
        self.axes_dims = (32, 48, 48)
        self.rope_theta = 256.0
        self.time_scale = 1000.0
        self.pad_tokens_multiple = 32 if self.weights.maybe("cap_pad_token") is not None else None
        self.noise_refiner = [MLXZImageBlock(self.weights, f"noise_refiner.{idx}", modulation=True) for idx in range(2)]
        self.context_refiner = [MLXZImageBlock(self.weights, f"context_refiner.{idx}", modulation=False) for idx in range(2)]
        self.layers = [MLXZImageBlock(self.weights, f"layers.{idx}", modulation=True) for idx in range(self._count_layers())]
        self._freqs_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._context_cache: dict[tuple[int, tuple[int, ...], str, int | None], Any] = {}
        self._context_cache_max = max(0, int(os.environ.get("COMFY_MLX_Z_IMAGE_CONTEXT_CACHE_SIZE", "8")))
        self.profile_path = _profile_path()
        self._call_index = 0
        self.compile_modes = _compile_mode_parts()
        self._compiled_forwards: dict[tuple[Any, ...], Any] = {}
        self._compile_disabled = False

    def _profile(self, stage: str, fn: Any, *, input_value: Any | None = None) -> Any:
        if not self.profile_path:
            return fn()
        start = time.perf_counter()
        out = fn()
        arrays = _mlx_arrays(out)
        if arrays:
            self.mx.eval(*arrays)
        _write_profile_event(
            self.profile_path,
            {
                "event": "z_image_transformer_profile",
                "call_index": self._call_index,
                "stage": stage,
                "input_shape": _json_shape(input_value),
                "output_shape": _json_output_shape(out),
                "seconds": time.perf_counter() - start,
                "linear_backend": self.weights.linear_impl,
                "compile_modes": sorted(self.compile_modes),
            },
        )
        return out

    def _count_layers(self) -> int:
        count = 0
        while (
            self.weights.maybe(f"layers.{count}.attention.qkv.weight") is not None
            or self.weights.maybe(f"layers.{count}.attention.to_q.weight") is not None
        ):
            count += 1
        if count == 0:
            raise ValueError("Z-Image MLX transformer found no layers.* attention keys.")
        return count

    def _timestep_embed(self, t: Any) -> Any:
        mx = self.mx
        t_freq = _timestep_embedding(mx, t, 256).astype(self.dtype)
        h = self.weights.linear(t_freq, self.weights.first_base("t_embedder.mlp.0", "t_embedder.linear1"))
        h = _silu(mx, h)
        return self.weights.linear(h, self.weights.first_base("t_embedder.mlp.2", "t_embedder.linear2"))

    def _cap_embed(self, cap_feats: Any) -> Any:
        mx = self.mx
        cap = _rms_norm(mx, cap_feats.astype(self.dtype), self.weights.get("cap_embedder.0.weight"), eps=1.0e-5)
        cap = self.weights.linear(cap, "cap_embedder.1")
        if self.pad_tokens_multiple is not None:
            cap, _ = _pad_to_multiple(mx, cap, self.weights.get("cap_pad_token"), self.pad_tokens_multiple)
        return cap

    def _cached_cap_embed(self, cap_feats: Any, num_tokens: int | None) -> Any:
        if self._context_cache_max <= 0:
            return self._profile("cap_embed", lambda: self._cap_embed(cap_feats), input_value=cap_feats)
        key = (id(cap_feats), tuple(int(item) for item in cap_feats.shape), str(cap_feats.dtype), num_tokens)
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached
        cap = self._profile("cap_embed", lambda: self._cap_embed(cap_feats), input_value=cap_feats)
        if len(self._context_cache) >= self._context_cache_max:
            self._context_cache.pop(next(iter(self._context_cache)))
        self._context_cache[key] = cap
        return cap

    def _cap_positions(self, length: int, batch_size: int) -> Any:
        mx = self.mx
        pos = mx.zeros((batch_size, length, 3), dtype=mx.float32)
        pos = mx.concatenate(
            [
                mx.repeat(mx.arange(length, dtype=mx.float32)[None, :, None] + 1.0, batch_size, axis=0),
                pos[:, :, 1:],
            ],
            axis=-1,
        )
        return pos

    def _patchify(self, x: Any) -> tuple[Any, tuple[int, int]]:
        mx = self.mx
        bsz, channels, height, width = x.shape
        pad_h = (-int(height)) % self.patch_size
        pad_w = (-int(width)) % self.patch_size
        if pad_h or pad_w:
            padded = mx.zeros((bsz, channels, int(height) + pad_h, int(width) + pad_w), dtype=x.dtype)
            padded[:, :, :height, :width] = x
            x = padded
            height, width = x.shape[-2:]
        h_tokens = int(height) // self.patch_size
        w_tokens = int(width) // self.patch_size
        p = self.patch_size
        x = x.reshape(bsz, channels, h_tokens, p, w_tokens, p).transpose(0, 2, 4, 3, 5, 1)
        return x.reshape(bsz, h_tokens * w_tokens, p * p * channels), (int(height), int(width))

    def _x_positions(self, cap_len: int, h_tokens: int, w_tokens: int, batch_size: int) -> Any:
        mx = self.mx
        t = mx.full((h_tokens * w_tokens,), float(cap_len + 1), dtype=mx.float32)
        h = mx.tile(mx.repeat(mx.arange(h_tokens, dtype=mx.float32), w_tokens), 1)
        w = mx.tile(mx.arange(w_tokens, dtype=mx.float32), h_tokens)
        pos = mx.stack([t, h, w], axis=-1)[None]
        return mx.repeat(pos, batch_size, axis=0)

    def _cached_freqs(self, cap_len: int, h_tokens: int, w_tokens: int, batch_size: int, x_pad_extra: int) -> tuple[Any, Any, Any]:
        key = (
            int(batch_size),
            int(cap_len),
            int(h_tokens),
            int(w_tokens),
            int(x_pad_extra),
            tuple(int(item) for item in self.axes_dims),
            float(self.rope_theta),
        )
        cached = self._freqs_cache.get(key)
        if cached is not None:
            return cached
        cap_pos = self._cap_positions(cap_len, batch_size)
        x_pos = self._x_positions(cap_len, h_tokens, w_tokens, batch_size)
        if x_pad_extra:
            x_pos = self.mx.concatenate([x_pos, self.mx.zeros((batch_size, x_pad_extra, 3), dtype=self.mx.float32)], axis=1)
        cap_freqs = _embed_nd(self.mx, cap_pos, self.axes_dims, self.rope_theta)
        x_freqs = _embed_nd(self.mx, x_pos, self.axes_dims, self.rope_theta)
        freqs = self.mx.concatenate([cap_freqs, x_freqs], axis=1)
        self._freqs_cache[key] = (cap_freqs, x_freqs, freqs)
        return self._freqs_cache[key]

    def _unpatchify(self, x: Any, height: int, width: int) -> Any:
        bsz = x.shape[0]
        p = self.patch_size
        h_tokens = int(height) // p
        w_tokens = int(width) // p
        x = x.reshape(bsz, h_tokens, w_tokens, p, p, self.in_channels)
        x = x.transpose(0, 5, 1, 3, 2, 4)
        return x.reshape(bsz, self.in_channels, height, width)

    def _final_layer(self, img: Any, adaln_input: Any) -> Any:
        mx = self.mx
        scale = self.weights.linear(
            _silu(mx, adaln_input),
            self.weights.first_base("final_layer.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.0"),
        )
        img = _layer_norm(mx, img, eps=1.0e-6) * (1.0 + scale[:, None])
        return self.weights.linear(img, self.weights.first_base("final_layer.linear", "all_final_layer.2-1.linear"))

    def _forward_impl(self, latent: Any, timestep: Any, context: Any, num_tokens: int | None = None) -> Any:
        mx = self.mx
        latent = latent.astype(self.dtype)
        context = context.astype(self.dtype)
        bsz, channels, height, width = latent.shape
        if int(channels) != self.in_channels:
            raise ValueError(f"Z-Image latent must have {self.in_channels} channels; got {channels}.")

        # Match comfy.ldm.lumina.model.NextDiT._forward: t = 1.0 - timesteps,
        # then TimestepEmbedder receives t * time_scale.
        t = (1.0 - timestep.astype(mx.float32)) * self.time_scale
        adaln_input = self._profile("timestep_embed", lambda: self._timestep_embed(t), input_value=t)
        x_patches, padded_shape = self._profile("patchify", lambda: self._patchify(latent), input_value=latent)
        x = self._profile(
            "x_embed",
            lambda: self.weights.linear(x_patches.astype(self.dtype), self.weights.first_base("x_embedder", "all_x_embedder.2-1")),
            input_value=x_patches,
        )
        original_img_len = int(x.shape[1])
        cap = self._cached_cap_embed(context, num_tokens)
        if num_tokens is not None:
            num_tokens = max(1, min(int(num_tokens), int(cap.shape[1])))
        cap_len = int(cap.shape[1])

        x_pad_extra = 0
        if self.pad_tokens_multiple is not None:
            x, x_pad_extra = _pad_to_multiple(mx, x, self.weights.get("x_pad_token"), self.pad_tokens_multiple)
        h_tokens = padded_shape[0] // self.patch_size
        w_tokens = padded_shape[1] // self.patch_size
        cap_freqs, x_freqs, freqs = self._profile(
            "cached_freqs",
            lambda: self._cached_freqs(cap_len, h_tokens, w_tokens, int(bsz), x_pad_extra),
        )

        for layer in self.context_refiner:
            cap = layer(mx, cap, None, cap_freqs, None)
        for layer in self.noise_refiner:
            x = layer(mx, x, None, x_freqs, adaln_input)

        img = mx.concatenate([cap, x], axis=1)
        for layer in self.layers:
            img = layer(mx, img, None, freqs, adaln_input)

        img = self._profile("final_layer", lambda: self._final_layer(img, adaln_input), input_value=img)
        img = img[:, cap_len : cap_len + original_img_len]
        out = self._profile("unpatchify", lambda: self._unpatchify(img, padded_shape[0], padded_shape[1]), input_value=img)
        out = out[:, :, :height, :width]
        return -out

    def __call__(self, latent: Any, timestep: Any, context: Any, *, num_tokens: int | None = None) -> Any:
        self._call_index += 1
        for layer in [*self.context_refiner, *self.noise_refiner, *self.layers]:
            layer.profile_call_index = self._call_index

        compile_enabled = "transformer" in self.compile_modes
        if compile_enabled and not self.profile_path and not self._compile_disabled:
            compile_fn = getattr(self.mx, "compile", None)
            if compile_fn is None:
                self._compile_disabled = True
            else:
                try:
                    key = (
                        tuple(int(item) for item in latent.shape),
                        tuple(int(item) for item in context.shape),
                        str(latent.dtype),
                        str(context.dtype),
                        int(num_tokens) if num_tokens is not None else None,
                        self.weights.linear_impl,
                        self.weights.dequant_patterns,
                        self.layers[0].attention_layout if self.layers else "seq_major",
                        tuple(sorted(self.compile_modes)),
                        tuple(int(item) for item in self.axes_dims),
                        float(self.rope_theta),
                    )
                    compiled_forward = self._compiled_forwards.get(key)
                    if compiled_forward is None:
                        compiled_forward = compile_fn(self._forward_impl)
                        self._compiled_forwards[key] = compiled_forward
                    return compiled_forward(latent, timestep, context, num_tokens)
                except Exception:
                    self._compiled_forwards.clear()
                    self._compile_disabled = True

        return self._forward_impl(latent, timestep, context, num_tokens)


def resolve_mlx_z_image_model_path(name: str) -> str | None:
    import folder_paths

    if name == Z_IMAGE_TURBO_MLX_ALIAS:
        return folder_paths.get_full_path("diffusion_models", Z_IMAGE_TURBO_NATIVE_FILE)
    if name == Z_IMAGE_TURBO_MLX_Q8_ALIAS:
        env_path = Path(
            str(
                os.environ.get(
                    "COMFY_MLX_Z_IMAGE_Q8_MODEL_PATH",
                    os.environ.get("COMFY_MLX_Z_IMAGE_Q8_PATH", Z_IMAGE_TURBO_MLX_Q8_DEFAULT_PATH),
                )
            )
        ).expanduser()
        return str(env_path) if env_path.exists() else None

    candidate = Path(name).expanduser()
    if candidate.exists() and candidate.suffix == ".json":
        data = json.loads(candidate.read_text(encoding="utf-8"))
        model_path = data.get("model_path") or data.get("diffusion_model") or data.get("package_path")
        if model_path:
            model_candidate = Path(model_path).expanduser()
            if not model_candidate.is_absolute():
                model_candidate = candidate.parent / model_candidate
            return str(model_candidate.resolve())

    if name.endswith(".mlx_z_image.json"):
        path = folder_paths.get_full_path("diffusion_models", name)
        if path is None:
            return None
        return resolve_mlx_z_image_model_path(path)
    return None


def list_mlx_z_image_choices() -> list[str]:
    import folder_paths

    choices = {Z_IMAGE_TURBO_MLX_ALIAS}
    if resolve_mlx_z_image_model_path(Z_IMAGE_TURBO_MLX_Q8_ALIAS) is not None:
        choices.add(Z_IMAGE_TURBO_MLX_Q8_ALIAS)
    for root in folder_paths.get_folder_paths("diffusion_models"):
        base = Path(root)
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.name.endswith(".mlx_z_image.json"):
                choices.add(child.name)
    return sorted(choices)
