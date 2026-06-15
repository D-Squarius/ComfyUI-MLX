from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

import comfy.latent_formats
from comfy.backends.benchmark_stats import write_event
from comfy.backends.mlx_denoiser_island import (
    BaseMLXDenoiserIslandRuntime,
    MLXDenoiserCall,
    MLXIslandAdapterContract,
    UnsupportedIslandRequest,
    _mlx_memory_snapshot,
    _shape_summary,
    create_mlx_denoiser_island_model,
)
from comfy.backends.mlx_quant import MLXLinearWeight, detect_quantization_bits, make_quantized_linear_module


KLEIN_4B_MLX_8BIT_ALIAS = "FLUX.2 Klein 4B MLX 8-bit Island"
KLEIN_4B_MLX_4BIT_ALIAS = "FLUX.2 Klein 4B MLX 4-bit Island"
KLEIN_9B_MLX_8BIT_ALIAS = "FLUX.2 Klein 9B MLX 8-bit Island"
KLEIN_9B_MLX_4BIT_ALIAS = "FLUX.2 Klein 9B MLX 4-bit Island"
KLEIN_4B_MLX_BF16_ALIAS = "FLUX.2 Klein 4B BF16 MLX Island"
KLEIN_9B_MLX_BF16_ALIAS = "FLUX.2 Klein 9B BF16 MLX Island"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent
KLEIN_COMFY_MODEL_ROOT = _REPO_ROOT / "models" / "diffusion_models"


def _klein_mlx_vendor_root() -> Path:
    return Path(os.environ.get("COMFY_MLX_KLEIN_MODEL_ROOT", str(_WORKSPACE_ROOT / "Vendor" / "klein-mlx-models"))).expanduser()


def _klein_default_paths() -> dict[str, Path]:
    vendor_root = _klein_mlx_vendor_root()
    return {
        KLEIN_4B_MLX_BF16_ALIAS: KLEIN_COMFY_MODEL_ROOT / "flux-2-klein-4b.safetensors",
        KLEIN_4B_MLX_8BIT_ALIAS: vendor_root / "flux2-klein-4b-8bit",
        KLEIN_4B_MLX_4BIT_ALIAS: vendor_root / "flux2-klein-4b-4bit",
        KLEIN_9B_MLX_BF16_ALIAS: KLEIN_COMFY_MODEL_ROOT / "flux-2-klein-9b.safetensors",
        KLEIN_9B_MLX_8BIT_ALIAS: vendor_root / "flux2-klein-9b-8bit",
        KLEIN_9B_MLX_4BIT_ALIAS: vendor_root / "flux2-klein-9b-4bit",
    }


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _candidate_flags() -> dict[str, Any]:
    return {
        "cache_static_rope_ids": _env_flag("COMFY_MLX_KLEIN_CACHE_STATIC_ROPE_IDS"),
        "cache_timestep_embedding": _env_flag("COMFY_MLX_KLEIN_CACHE_TIMESTEP_EMBEDDING"),
        "disable_extra_memory_snapshot": _env_flag("COMFY_MLX_KLEIN_DISABLE_MEMORY_SNAPSHOT"),
        "single_eval_boundary": _env_flag("COMFY_MLX_KLEIN_SINGLE_EVAL_BOUNDARY"),
        "profile": _env_flag("COMFY_MLX_KLEIN_PROFILE"),
        "profile_eval_stages": _env_flag("COMFY_MLX_KLEIN_PROFILE_EVAL_STAGES"),
        "q_linear_impl": _env_linear_impl(),
    }


def _record_stage(profile: dict[str, Any] | None, name: str, seconds: float) -> None:
    if profile is None:
        return
    profile.setdefault("stages", []).append({"name": name, "seconds": float(seconds)})


def _eval_tree(mx: Any, *values: Any) -> None:
    arrays = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            arrays.append(value)

    for value in values:
        collect(value)
    if arrays:
        mx.eval(*arrays)


def list_mlx_klein_choices() -> list[str]:
    return list(_klein_default_paths())


def resolve_mlx_klein_model_path(name: str) -> Path | None:
    return _klein_default_paths().get(name)


def _env_linear_impl() -> str:
    impl = os.environ.get("COMFY_MLX_KLEIN_Q_LINEAR_IMPL", "dequantized").strip().lower()
    if impl in {"", "default"}:
        return "dequantized"
    if impl not in {"quantized_matmul", "nn_quantized", "dequantized"}:
        return "quantized_matmul"
    return impl


def _torch_tensor_to_mlx(mx: Any, tensor: torch.Tensor, dtype: Any) -> Any:
    if tensor.dtype is torch.bfloat16:
        tensor = tensor.to(torch.float32)
    elif torch.is_floating_point(tensor) and tensor.dtype is not torch.float32:
        tensor = tensor.to(torch.float32)
    out = mx.array(tensor.detach().cpu().numpy())
    return out.astype(dtype) if torch.is_floating_point(tensor) else out


def _silu(mx: Any, x: Any) -> Any:
    return x * mx.sigmoid(x)


def _layer_norm(mx: Any, x: Any, eps: float = 1.0e-6) -> Any:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) * (xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * mx.rsqrt(var + eps)).astype(dtype)


def _rms_norm(mx: Any, x: Any, weight: Any, eps: float = 1.0e-5) -> Any:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * weight.astype(mx.float32)).astype(dtype)


def _timestep_embedding(mx: Any, timesteps: Any, dim: int) -> Any:
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    args = timesteps[:, None].astype(mx.float32) * freqs[None]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


def _rope_cos_sin(mx: Any, ids: Any, axes_dims: tuple[int, ...], theta: float) -> tuple[Any, Any]:
    cos_parts = []
    sin_parts = []
    pos = ids.astype(mx.float32)
    for axis, dim in enumerate(axes_dims):
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta**scale)
        out = pos[..., axis, None] * omega[None]
        cos_parts.append(mx.cos(out))
        sin_parts.append(mx.sin(out))
    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def _apply_rope_bhsd(mx: Any, query: Any, key: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    out_dtype = query.dtype
    cos = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

    def mix(x: Any) -> Any:
        xf = x.astype(mx.float32)
        pair = xf.reshape(*xf.shape[:-1], -1, 2)
        real = pair[..., 0]
        imag = pair[..., 1]
        out0 = real * cos - imag * sin
        out1 = imag * cos + real * sin
        return mx.stack([out0, out1], axis=-1).reshape(*xf.shape).astype(out_dtype)

    return mix(query), mix(key)


def _linear_input_dim(linear: MLXLinearWeight) -> int:
    if linear.quantized:
        bits = linear.bits
        if bits is None:
            bits = detect_quantization_bits(linear.weight, linear.scales, linear.group_size)
        return int(linear.weight.shape[1]) * 32 // int(bits)
    return int(linear.weight.shape[1])


def _split_mod_params(mx: Any, value: Any, sets: int) -> tuple[tuple[Any, Any, Any], ...]:
    if value.ndim == 2:
        value = mx.expand_dims(value, axis=1)
    chunks = mx.split(value, 3 * sets, axis=-1)
    return tuple(tuple(chunks[3 * idx : 3 * (idx + 1)]) for idx in range(sets))


def _indices_for_prefix(weights: dict[str, Any], prefix: str) -> list[int]:
    indices = set()
    needle = prefix + "."
    for key in weights:
        if not key.startswith(needle):
            continue
        rest = key[len(needle) :]
        try:
            indices.add(int(rest.split(".", 1)[0]))
        except Exception:
            pass
    return sorted(indices)


class MLXKleinWeights:
    def __init__(self, path: str | Path, *, dtype: Any):
        import mlx.core as mx

        self.path = str(Path(path).expanduser())
        self.dtype = dtype
        self.mx = mx
        self.group_size = 64
        self.quantization_level: str | None = None
        self.quantization_group_size: int = 64
        self.linear_impl = _env_linear_impl()
        self.weight_format = "unknown"
        self._weights: dict[str, Any] | None = None
        self._linear_refs: dict[str, MLXLinearWeight] = {}

    @property
    def weights(self) -> dict[str, Any]:
        if self._weights is None:
            self._weights = self._load()
        return self._weights

    def _load(self) -> dict[str, Any]:
        from safetensors import safe_open

        path = Path(self.path)
        if path.is_file():
            self.weight_format = "comfy_bf16_safetensor"
            self.linear_impl = "dense"
            raw: dict[str, Any] = {}
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    raw[key] = self._convert_tensor(handle.get_tensor(key))
            return self._normalize_comfy_dense_weights(raw)

        index = path / "transformer" / "model.safetensors.index.json" if path.is_dir() else None
        if index is None or not index.exists():
            raise UnsupportedIslandRequest(f"Klein MLX transformer index not found under {path}.")
        self.weight_format = "mlx_quantized_package"
        data = json.loads(index.read_text(encoding="utf-8"))
        metadata = data.get("metadata") or {}
        self.quantization_level = metadata.get("quantization_level")
        try:
            self.quantization_group_size = int(metadata.get("quantization_group_size") or self.group_size)
        except Exception:
            self.quantization_group_size = self.group_size
        weight_map = data.get("weight_map") or {}
        loaded: dict[str, Any] = {}
        for file_name in sorted(set(weight_map.values())):
            shard = index.parent / file_name
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if weight_map.get(key) == file_name:
                        loaded[key] = self._convert_tensor(handle.get_tensor(key))
        return loaded

    def _normalize_comfy_dense_weights(self, raw: dict[str, Any]) -> dict[str, Any]:
        mx = self.mx
        out: dict[str, Any] = {}

        def copy(dst: str, src: str) -> None:
            out[dst] = raw[src]

        def split_qkv(src: str, q_dst: str, k_dst: str, v_dst: str) -> None:
            q, k, v = mx.split(raw[src], 3, axis=0)
            out[q_dst] = q
            out[k_dst] = k
            out[v_dst] = v

        def swap_scale_shift(src: str) -> Any:
            shift, scale = mx.split(raw[src], 2, axis=0)
            return mx.concatenate([scale, shift], axis=0)

        copy("x_embedder.weight", "img_in.weight")
        copy("context_embedder.weight", "txt_in.weight")
        copy("time_guidance_embed.linear_1.weight", "time_in.in_layer.weight")
        copy("time_guidance_embed.linear_2.weight", "time_in.out_layer.weight")
        copy("double_stream_modulation_img.linear.weight", "double_stream_modulation_img.lin.weight")
        copy("double_stream_modulation_txt.linear.weight", "double_stream_modulation_txt.lin.weight")
        copy("single_stream_modulation.linear.weight", "single_stream_modulation.lin.weight")
        out["norm_out.linear.weight"] = swap_scale_shift("final_layer.adaLN_modulation.1.weight")
        copy("proj_out.weight", "final_layer.linear.weight")

        double_indices = _indices_for_prefix(raw, "double_blocks")
        for index in double_indices:
            src = f"double_blocks.{index}"
            dst = f"transformer_blocks.{index}"
            split_qkv(f"{src}.img_attn.qkv.weight", f"{dst}.attn.to_q.weight", f"{dst}.attn.to_k.weight", f"{dst}.attn.to_v.weight")
            split_qkv(
                f"{src}.txt_attn.qkv.weight",
                f"{dst}.attn.add_q_proj.weight",
                f"{dst}.attn.add_k_proj.weight",
                f"{dst}.attn.add_v_proj.weight",
            )
            copy(f"{dst}.attn.to_out.weight", f"{src}.img_attn.proj.weight")
            copy(f"{dst}.attn.to_add_out.weight", f"{src}.txt_attn.proj.weight")
            copy(f"{dst}.attn.norm_q.weight", f"{src}.img_attn.norm.query_norm.scale")
            copy(f"{dst}.attn.norm_k.weight", f"{src}.img_attn.norm.key_norm.scale")
            copy(f"{dst}.attn.norm_added_q.weight", f"{src}.txt_attn.norm.query_norm.scale")
            copy(f"{dst}.attn.norm_added_k.weight", f"{src}.txt_attn.norm.key_norm.scale")
            copy(f"{dst}.ff.linear_in.weight", f"{src}.img_mlp.0.weight")
            copy(f"{dst}.ff.linear_out.weight", f"{src}.img_mlp.2.weight")
            copy(f"{dst}.ff_context.linear_in.weight", f"{src}.txt_mlp.0.weight")
            copy(f"{dst}.ff_context.linear_out.weight", f"{src}.txt_mlp.2.weight")

        single_indices = _indices_for_prefix(raw, "single_blocks")
        for index in single_indices:
            src = f"single_blocks.{index}"
            dst = f"single_transformer_blocks.{index}.attn"
            copy(f"{dst}.to_qkv_mlp_proj.weight", f"{src}.linear1.weight")
            copy(f"{dst}.to_out.weight", f"{src}.linear2.weight")
            copy(f"{dst}.norm_q.weight", f"{src}.norm.query_norm.scale")
            copy(f"{dst}.norm_k.weight", f"{src}.norm.key_norm.scale")

        return out

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

    def linear_ref(self, base: str) -> MLXLinearWeight:
        cached = self._linear_refs.get(base)
        if cached is not None:
            return cached
        weight = self.get(base + ".weight")
        scales = self.maybe(base + ".scales")
        biases = self.maybe(base + ".biases")
        bits = None
        group_size = int(self.quantization_group_size)
        impl = "dense"
        if scales is not None:
            bits = int(self.quantization_level) if self.quantization_level is not None else detect_quantization_bits(weight, scales, group_size)
            impl = self.linear_impl
            if impl == "dequantized":
                weight = self.mx.dequantize(
                    weight,
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                    dtype=self.dtype,
                )
                scales = None
                biases = None
        ref = MLXLinearWeight(
            weight=weight,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            name=base,
            impl=impl if scales is not None else "dense",
        )
        if scales is not None and impl == "nn_quantized":
            module = make_quantized_linear_module(self.mx, ref)
            ref = MLXLinearWeight(
                weight=ref.weight,
                scales=ref.scales,
                biases=ref.biases,
                group_size=ref.group_size,
                bits=ref.bits,
                name=ref.name,
                impl=impl,
                module=module,
            )
        self._linear_refs[base] = ref
        return ref

    def linear(self, x: Any, base: str) -> Any:
        return self.linear_ref(base)(self.mx, x)


class MLXKleinDoubleBlock:
    def __init__(self, weights: MLXKleinWeights, index: int, *, dim: int, heads: int, head_dim: int, mlp_dim: int):
        self.weights = weights
        self.mx = weights.mx
        self.prefix = f"transformer_blocks.{index}"
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.mlp_dim = int(mlp_dim)
        attn = self.prefix + ".attn"
        ff = self.prefix + ".ff"
        ff_context = self.prefix + ".ff_context"
        self.to_q = weights.linear_ref(attn + ".to_q")
        self.to_k = weights.linear_ref(attn + ".to_k")
        self.to_v = weights.linear_ref(attn + ".to_v")
        self.add_q = weights.linear_ref(attn + ".add_q_proj")
        self.add_k = weights.linear_ref(attn + ".add_k_proj")
        self.add_v = weights.linear_ref(attn + ".add_v_proj")
        self.to_out = weights.linear_ref(attn + ".to_out.0" if weights.has(attn + ".to_out.0.weight") else attn + ".to_out")
        self.to_add_out = weights.linear_ref(attn + ".to_add_out")
        self.norm_q = weights.get(attn + ".norm_q.weight")
        self.norm_k = weights.get(attn + ".norm_k.weight")
        self.norm_added_q = weights.get(attn + ".norm_added_q.weight")
        self.norm_added_k = weights.get(attn + ".norm_added_k.weight")
        self.ff_in = weights.linear_ref(ff + ".linear_in")
        self.ff_out = weights.linear_ref(ff + ".linear_out")
        self.ff_context_in = weights.linear_ref(ff_context + ".linear_in")
        self.ff_context_out = weights.linear_ref(ff_context + ".linear_out")

    def _ff(self, x: Any, linear_in: MLXLinearWeight, linear_out: MLXLinearWeight) -> Any:
        mx = self.mx
        hidden = linear_in(mx, x)
        gate, up = mx.split(hidden, 2, axis=-1)
        return linear_out(mx, _silu(mx, gate) * up)

    def _qkv(self, x: Any, q: MLXLinearWeight, k: MLXLinearWeight, v: MLXLinearWeight, nq: Any, nk: Any) -> tuple[Any, Any, Any]:
        mx = self.mx
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        query = q(mx, x).reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        key = k(mx, x).reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        value = v(mx, x).reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        query = _rms_norm(mx, query, nq)
        key = _rms_norm(mx, key, nk)
        return query, key, value

    def __call__(self, img: Any, txt: Any, img_mod: tuple[tuple[Any, Any, Any], ...], txt_mod: tuple[tuple[Any, Any, Any], ...], rope: tuple[Any, Any]) -> tuple[Any, Any]:
        mx = self.mx
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = img_mod
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = txt_mod

        img_norm = (1 + scale_msa) * _layer_norm(mx, img) + shift_msa
        txt_norm = (1 + c_scale_msa) * _layer_norm(mx, txt) + c_shift_msa

        iq, ik, iv = self._qkv(img_norm, self.to_q, self.to_k, self.to_v, self.norm_q, self.norm_k)
        tq, tk, tv = self._qkv(txt_norm, self.add_q, self.add_k, self.add_v, self.norm_added_q, self.norm_added_k)
        query = mx.concatenate([tq, iq], axis=2)
        key = mx.concatenate([tk, ik], axis=2)
        value = mx.concatenate([tv, iv], axis=2)
        query, key = _apply_rope_bhsd(mx, query, key, rope[0], rope[1])
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(attn.shape[0], -1, self.dim)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + gate_msa * self.to_out(mx, img_attn)
        txt = txt + c_gate_msa * self.to_add_out(mx, txt_attn)

        img_ff = (1 + scale_mlp) * _layer_norm(mx, img) + shift_mlp
        txt_ff = (1 + c_scale_mlp) * _layer_norm(mx, txt) + c_shift_mlp
        img = img + gate_mlp * self._ff(img_ff, self.ff_in, self.ff_out)
        txt = txt + c_gate_mlp * self._ff(txt_ff, self.ff_context_in, self.ff_context_out)
        return txt, img


class MLXKleinSingleBlock:
    def __init__(self, weights: MLXKleinWeights, index: int, *, dim: int, heads: int, head_dim: int, mlp_dim: int):
        self.weights = weights
        self.mx = weights.mx
        self.prefix = f"single_transformer_blocks.{index}"
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.mlp_dim = int(mlp_dim)
        attn = self.prefix + ".attn"
        self.to_qkv_mlp = weights.linear_ref(attn + ".to_qkv_mlp_proj")
        self.to_out = weights.linear_ref(attn + ".to_out")
        self.norm_q = weights.get(attn + ".norm_q.weight")
        self.norm_k = weights.get(attn + ".norm_k.weight")

    def __call__(self, x: Any, mod: tuple[Any, Any, Any], rope: tuple[Any, Any]) -> Any:
        mx = self.mx
        mod_shift, mod_scale, mod_gate = mod
        x_norm = (1 + mod_scale) * _layer_norm(mx, x) + mod_shift
        proj = self.to_qkv_mlp(mx, x_norm)
        qkv, mlp_hidden = mx.split(proj, [self.dim * 3], axis=-1)
        query, key, value = mx.split(qkv, 3, axis=-1)
        batch, seq_len = int(query.shape[0]), int(query.shape[1])
        query = query.reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        query = _rms_norm(mx, query, self.norm_q)
        key = _rms_norm(mx, key, self.norm_k)
        query, key = _apply_rope_bhsd(mx, query, key, rope[0], rope[1])
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        gate, up = mx.split(mlp_hidden, 2, axis=-1)
        mlp_hidden = _silu(mx, gate) * up
        out = self.to_out(mx, mx.concatenate([attn, mlp_hidden], axis=-1))
        return x + mod_gate * out


class MLXKleinTransformer:
    def __init__(self, path: str | Path, *, dtype: Any):
        self.weights = MLXKleinWeights(path, dtype=dtype)
        self.mx = self.weights.mx
        self.dtype = dtype
        self.x_embedder = self.weights.linear_ref("x_embedder")
        self.context_embedder = self.weights.linear_ref("context_embedder")
        self.time_linear_1 = self.weights.linear_ref("time_guidance_embed.linear_1")
        self.time_linear_2 = self.weights.linear_ref("time_guidance_embed.linear_2")
        self.mod_img = self.weights.linear_ref("double_stream_modulation_img.linear")
        self.mod_txt = self.weights.linear_ref("double_stream_modulation_txt.linear")
        self.mod_single = self.weights.linear_ref("single_stream_modulation.linear")
        self.norm_out = self.weights.linear_ref("norm_out.linear")
        self.proj_out = self.weights.linear_ref("proj_out")
        self.hidden_dim = int(self.x_embedder.weight.shape[0])
        self.head_dim = int(self.weights.get("transformer_blocks.0.attn.norm_q.weight").shape[0])
        self.heads = self.hidden_dim // self.head_dim
        self.context_dim = _linear_input_dim(self.context_embedder)
        self.in_channels = _linear_input_dim(self.x_embedder)
        self.mlp_dim = _linear_input_dim(self.weights.linear_ref("transformer_blocks.0.ff.linear_out"))
        self.double_blocks = [
            MLXKleinDoubleBlock(self.weights, index, dim=self.hidden_dim, heads=self.heads, head_dim=self.head_dim, mlp_dim=self.mlp_dim)
            for index in self._block_indices("transformer_blocks")
        ]
        self.single_blocks = [
            MLXKleinSingleBlock(self.weights, index, dim=self.hidden_dim, heads=self.heads, head_dim=self.head_dim, mlp_dim=self.mlp_dim)
            for index in self._block_indices("single_transformer_blocks")
        ]
        self.axes_dims = (32, 32, 32, 32)
        self.rope_theta = 2000.0

    def _block_indices(self, prefix: str) -> list[int]:
        indices = set()
        needle = prefix + "."
        for key in self.weights.weights:
            if not key.startswith(needle):
                continue
            rest = key[len(needle) :]
            try:
                indices.add(int(rest.split(".", 1)[0]))
            except Exception:
                pass
        return sorted(indices)

    def _img_ids(self, batch: int, height: int, width: int) -> Any:
        mx = self.mx
        ids = mx.zeros((height, width, 4), dtype=mx.float32)
        ids = mx.concatenate(
            [
                ids[..., 0:1],
                mx.broadcast_to(mx.arange(height, dtype=mx.float32)[:, None, None], (height, width, 1)),
                mx.broadcast_to(mx.arange(width, dtype=mx.float32)[None, :, None], (height, width, 1)),
                ids[..., 3:4],
            ],
            axis=-1,
        )
        ids = ids.reshape(height * width, 4)
        if batch != 1:
            ids = mx.repeat(ids[None], batch, axis=0)
        return ids

    def _txt_ids(self, batch: int, tokens: int) -> Any:
        mx = self.mx
        ids = mx.zeros((tokens, 4), dtype=mx.float32)
        ids = mx.concatenate(
            [
                ids[:, 0:3],
                mx.arange(tokens, dtype=mx.float32)[:, None],
            ],
            axis=-1,
        )
        if batch != 1:
            ids = mx.repeat(ids[None], batch, axis=0)
        return ids

    def _modulation(self, linear: MLXLinearWeight, temb: Any, sets: int) -> tuple[tuple[Any, Any, Any], ...]:
        return _split_mod_params(self.mx, linear(self.mx, _silu(self.mx, temb)), sets)

    def _timestep_embed(
        self,
        timestep: Any,
        *,
        batch: int,
        static_cache: dict[Any, Any] | None,
        cache_enabled: bool,
        cache_key: Any | None,
    ) -> tuple[Any, bool]:
        mx = self.mx
        if timestep.ndim == 0:
            timestep = mx.full((batch,), timestep, dtype=self.dtype)
        timestep = timestep.reshape(-1).astype(self.dtype)
        if int(timestep.shape[0]) == 1 and batch != 1:
            timestep = mx.repeat(timestep, batch, axis=0)
        key = None
        if cache_enabled and static_cache is not None and cache_key is not None:
            key = (
                "temb",
                cache_key,
                int(batch),
                int(self.hidden_dim),
                str(self.dtype),
                str(self.weights.path),
            )
            cached = static_cache.get(key)
            if cached is not None:
                return cached, True
        timestep = timestep * mx.where(mx.max(timestep) <= 1.0, mx.array(1000.0, dtype=self.dtype), mx.array(1.0, dtype=self.dtype))
        temb = self.time_linear_2(mx, _silu(mx, self.time_linear_1(mx, _timestep_embedding(mx, timestep.astype(mx.float32), 256))))
        temb = temb.astype(self.dtype)
        if key is not None and static_cache is not None:
            mx.eval(temb)
            if len(static_cache) >= 16:
                static_cache.pop(next(iter(static_cache)))
            static_cache[key] = temb
        return temb, False

    def _rope(
        self,
        *,
        batch: int,
        height: int,
        width: int,
        tokens: int,
        static_cache: dict[Any, Any] | None,
        cache_enabled: bool,
    ) -> tuple[tuple[Any, Any], bool]:
        mx = self.mx
        key = None
        if cache_enabled and static_cache is not None:
            key = (
                "rope",
                int(batch),
                int(height),
                int(width),
                int(tokens),
                tuple(int(item) for item in self.axes_dims),
                float(self.rope_theta),
                str(self.dtype),
                str(self.weights.path),
            )
            cached = static_cache.get(key)
            if cached is not None:
                return cached, True
        img_ids = self._img_ids(batch, height, width)
        txt_ids = self._txt_ids(batch, tokens)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        img_rope = _rope_cos_sin(mx, img_ids, self.axes_dims, self.rope_theta)
        txt_rope = _rope_cos_sin(mx, txt_ids, self.axes_dims, self.rope_theta)
        rope = (
            mx.concatenate([txt_rope[0], img_rope[0]], axis=0),
            mx.concatenate([txt_rope[1], img_rope[1]], axis=0),
        )
        if key is not None and static_cache is not None:
            mx.eval(rope[0], rope[1])
            if len(static_cache) >= 16:
                static_cache.pop(next(iter(static_cache)))
            static_cache[key] = rope
        return rope, False

    def __call__(
        self,
        latent: Any,
        timestep: Any,
        context: Any,
        guidance: Any | None = None,
        *,
        static_cache: dict[Any, Any] | None = None,
        candidate_flags: dict[str, Any] | None = None,
        timestep_cache_key: Any | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        mx = self.mx
        candidate_flags = candidate_flags or {}
        profile_eval = bool(candidate_flags.get("profile_eval_stages"))

        stage_start = time.perf_counter()
        if latent.ndim != 4:
            raise UnsupportedIslandRequest(f"Klein MLX transformer expects latent [B,C,H,W], got {latent.shape}.")
        batch, channels, height, width = [int(v) for v in latent.shape]
        if channels != self.in_channels:
            raise UnsupportedIslandRequest(f"Klein MLX transformer expected {self.in_channels} latent channels, got {channels}.")
        img = latent.transpose(0, 2, 3, 1).reshape(batch, height * width, channels)
        context = context.astype(self.dtype)
        if int(context.shape[-1]) != self.context_dim:
            raise UnsupportedIslandRequest(
                f"Klein MLX transformer expected context dim {self.context_dim}, got {int(context.shape[-1])}."
            )
        _record_stage(profile, "shape_setup", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        temb, temb_cache_hit = self._timestep_embed(
            timestep,
            batch=batch,
            static_cache=static_cache,
            cache_enabled=bool(candidate_flags.get("cache_timestep_embedding")),
            cache_key=timestep_cache_key,
        )
        if profile_eval:
            mx.eval(temb)
        _record_stage(profile, "timestep_embed", time.perf_counter() - stage_start)
        if profile is not None:
            profile["timestep_cache_hit"] = bool(temb_cache_hit)

        stage_start = time.perf_counter()
        img = self.x_embedder(mx, img)
        txt = self.context_embedder(mx, context)
        if profile_eval:
            mx.eval(img, txt)
        _record_stage(profile, "x_text_embedding", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        rope, rope_cache_hit = self._rope(
            batch=batch,
            height=height,
            width=width,
            tokens=int(txt.shape[1]),
            static_cache=static_cache,
            cache_enabled=bool(candidate_flags.get("cache_static_rope_ids")),
        )
        if profile_eval:
            mx.eval(rope[0], rope[1])
        _record_stage(profile, "rope_setup", time.perf_counter() - stage_start)
        if profile is not None:
            profile["rope_cache_hit"] = bool(rope_cache_hit)

        stage_start = time.perf_counter()
        img_mod = self._modulation(self.mod_img, temb, 2)
        txt_mod = self._modulation(self.mod_txt, temb, 2)
        if profile_eval:
            _eval_tree(mx, img_mod, txt_mod)
        _record_stage(profile, "double_modulation", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        for block in self.double_blocks:
            txt, img = block(img, txt, img_mod, txt_mod, rope)
        if profile_eval:
            mx.eval(txt, img)
        _record_stage(profile, "double_blocks", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        stream = mx.concatenate([txt, img], axis=1)
        single_mod = self._modulation(self.mod_single, temb, 1)[0]
        if profile_eval:
            _eval_tree(mx, stream, single_mod)
        _record_stage(profile, "single_modulation", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        for block in self.single_blocks:
            stream = block(stream, single_mod, rope)
        if profile_eval:
            mx.eval(stream)
        _record_stage(profile, "single_blocks", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        img = stream[:, int(txt.shape[1]) :]
        scale_shift = self.norm_out(mx, _silu(mx, temb))
        scale, shift = mx.split(scale_shift, 2, axis=-1)
        img = _layer_norm(mx, img) * (1 + scale[:, None, :]) + shift[:, None, :]
        out = self.proj_out(mx, img)
        out = out.reshape(batch, height, width, self.in_channels).transpose(0, 3, 1, 2)
        if profile_eval:
            mx.eval(out)
        _record_stage(profile, "final_layer", time.perf_counter() - stage_start)
        return out

    def shape_census(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "mlp_dim": self.mlp_dim,
            "context_dim": self.context_dim,
            "latent_channels": self.in_channels,
            "double_blocks": len(self.double_blocks),
            "single_blocks": len(self.single_blocks),
            "quantization_level": self.weights.quantization_level,
            "quantization_group_size": self.weights.quantization_group_size,
            "linear_impl": self.weights.linear_impl,
            "weight_format": self.weights.weight_format,
        }


class KleinIslandRuntime(BaseMLXDenoiserIslandRuntime):
    name = "klein_mlx_island"
    prompt_state_policy = "prompt_local"
    unsupported_extra_cond_keys = (
        "attention_mask",
        "c_concat",
        "control",
        "ref_latents",
        "reference_latents",
        "clip_vision_outputs",
    )
    unsupported_feature_names = (
        "controlnet",
        "lora_model_patches",
        "transformer_patch_hooks",
        "context_windows",
        "image_edit_reference_latents",
        "attention_masks",
    )

    def __init__(self, model_path: str | Path | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_path = str(Path(model_path).expanduser()) if model_path is not None else None
        self._transformer: MLXKleinTransformer | None = None
        self.last_event: dict[str, Any] | None = None
        self._static_cache: dict[Any, Any] = {}
        self._last_context_cache_hit = False

    def _transformer_obj(self) -> MLXKleinTransformer:
        if self.model_path is None:
            raise UnsupportedIslandRequest("Klein MLX island runtime has no model_path configured.")
        if self._transformer is None:
            self._transformer = MLXKleinTransformer(self.model_path, dtype=self.bridge.mx.bfloat16)
        return self._transformer

    def precision_policy(self) -> dict[str, Any]:
        bits = None
        if self._transformer is not None:
            bits = self._transformer.weights.quantization_level
        default = "q8" if bits == "8" else "q4" if bits == "4" else "bf16"
        return {"default": default, "available": [default], "runtime_label": "Klein MLX Island"}

    def adapter_contract(self) -> MLXIslandAdapterContract:
        return MLXIslandAdapterContract(
            runtime=self.name,
            model_family="klein",
            precision_policy=self.precision_policy(),
            supports_dit_census=True,
            prompt_state_policy=self.prompt_state_policy,
            supported_features=self.supported_features(),
            unsupported_features=self.unsupported_features(),
        )

    def shape_census(self, call: MLXDenoiserCall | None = None) -> dict[str, Any]:
        model = self._transformer.shape_census() if self._transformer is not None else None
        return {
            "runtime": self.name,
            "model_family": "klein",
            "precision_policy": self.precision_policy(),
            "model_path": self.model_path,
            "model": model,
            "model_input_shape": _shape_summary(call.model_input) if call is not None else None,
            "context_shape": _shape_summary(call.context) if call is not None else None,
            "timestep_shape": _shape_summary(call.timestep) if call is not None else None,
        }

    def _prepare_context(self, call: MLXDenoiserCall, *, latent_batch: int) -> Any:
        if call.context is None:
            raise UnsupportedIslandRequest("Klein MLX island requires Flux2 text context.")
        mx = self.bridge.mx
        context = call.context
        key = (
            id(context),
            tuple(int(item) for item in context.shape),
            str(context.dtype),
            str(context.device),
            int(latent_batch),
        )
        cached = self.caches.context.get(key)
        if cached is not None:
            self._last_context_cache_hit = True
            return cached
        self._last_context_cache_hit = False
        value = self.bridge.torch_to_mlx(context, dtype=mx.bfloat16)
        if int(value.shape[0]) == 1 and int(latent_batch) != 1:
            value = mx.repeat(value, int(latent_batch), axis=0)
        if len(self.caches.context) >= 2:
            self.caches.context.pop(next(iter(self.caches.context)))
        self.caches.context[key] = value
        return value

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        self.calls += 1
        if not isinstance(call.model_input, torch.Tensor) or call.model_input.ndim != 4:
            raise UnsupportedIslandRequest(f"Klein MLX island expects one Flux2 latent tensor; got {_shape_summary(call.model_input)}.")
        mx = self.bridge.mx
        candidate_flags = _candidate_flags()
        profile = {"stages": []} if candidate_flags.get("profile") else None
        start = time.perf_counter()
        stage_start = time.perf_counter()
        latent_mx = self.bridge.torch_to_mlx(call.model_input, dtype=mx.bfloat16)
        _record_stage(profile, "torch_to_mlx_latent", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        context_mx = self._prepare_context(call, latent_batch=int(call.model_input.shape[0]))
        _record_stage(profile, "prepare_context", time.perf_counter() - stage_start)
        timestep_mx = self.bridge.torch_to_mlx(call.timestep, dtype=mx.bfloat16)
        guidance = call.extra_conds.get("guidance")
        guidance_mx = self.bridge.torch_to_mlx(guidance, dtype=mx.bfloat16) if isinstance(guidance, torch.Tensor) else None
        timestep_key = None
        if candidate_flags.get("cache_timestep_embedding") and isinstance(call.timestep, torch.Tensor):
            timestep_key = tuple(round(float(item), 8) for item in call.timestep.detach().cpu().float().reshape(-1).tolist())
        prediction = self._transformer_obj()(
            latent_mx,
            timestep_mx,
            context_mx,
            guidance=guidance_mx,
            static_cache=self._static_cache,
            candidate_flags=candidate_flags,
            timestep_cache_key=timestep_key,
            profile=profile,
        )
        stage_start = time.perf_counter()
        if not candidate_flags.get("single_eval_boundary"):
            mx.eval(prediction)
            _record_stage(profile, "mlx_eval", time.perf_counter() - stage_start)
        else:
            _record_stage(profile, "mlx_eval_skipped", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        out = self.bridge.mlx_to_torch(prediction, like=call.original_input, dtype=call.original_input.dtype)
        _record_stage(profile, "mlx_to_torch_output", time.perf_counter() - stage_start)
        memory = None if candidate_flags.get("disable_extra_memory_snapshot") else _mlx_memory_snapshot(mx)
        self.last_event = {
            "runtime": self.name,
            "call_index": self.calls,
            "model_path": self.model_path,
            "input_shape": _shape_summary(call.model_input),
            "context_shape": _shape_summary(call.context),
            "timestep_shape": _shape_summary(call.timestep),
            "output_shape": _shape_summary(out),
            "seconds": time.perf_counter() - start,
            "conversions": self.bridge.counter.as_dict(),
            "memory": memory,
            "shape_census": self.shape_census(call),
            "candidate_flags": candidate_flags,
            "context_cache_hit": self._last_context_cache_hit,
            "profile": profile,
        }
        write_event("mlx_klein_island_forward", **self.last_event)
        return out


def create_klein_mlx_island_model(
    *,
    runtime: KleinIslandRuntime | None = None,
    load_device: torch.device | str | None = None,
    offload_device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    sampling_shift: float = 2.02,
) -> Any:
    island_runtime = runtime if runtime is not None else KleinIslandRuntime()
    return create_mlx_denoiser_island_model(
        runtime=island_runtime,
        load_device=load_device,
        offload_device=offload_device,
        dtype=dtype,
        sampling_shift=sampling_shift,
        latent_format=comfy.latent_formats.Flux2(),
        model_family="klein",
    )
