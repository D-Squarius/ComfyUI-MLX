from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


THREADGROUP_SIZE = 256


def _mx():
    return importlib.import_module("mlx.core")


def _source(name: str) -> str:
    path = Path(__file__).with_name("metal") / f"{name}.metal"
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=32)
def _elementwise_kernel(name: str, source_name: str):
    mx = _mx()
    return mx.fast.metal_kernel(
        name=name,
        input_names=["a", "b"],
        output_names=["out"],
        source=_source(source_name),
        ensure_row_contiguous=True,
    )


def _native_thread_count(hidden_size: int) -> int:
    threads = 1
    limit = min(int(hidden_size), THREADGROUP_SIZE)
    while threads * 2 <= limit:
        threads *= 2
    return threads


def _broadcast_mode(reference: Any, operand: Any) -> tuple[int, int]:
    if len(reference.shape) < 2:
        raise ValueError("native broadcast kernels require inputs with at least 2 dimensions.")
    hidden_size = int(reference.shape[-1])
    if int(operand.shape[-1]) != hidden_size:
        raise ValueError(f"native broadcast hidden size mismatch: {operand.shape[-1]} vs {hidden_size}.")
    if tuple(operand.shape) == tuple(reference.shape):
        return 2, 1
    if int(operand.size) == hidden_size:
        return 0, 1
    if len(reference.shape) >= 3 and len(operand.shape) == len(reference.shape) and int(operand.shape[-2]) == 1:
        rows_per_batch = int(reference.shape[-2])
        expected_size = int(reference.size // rows_per_batch)
        if int(operand.size) == expected_size:
            return 1, rows_per_batch
    raise ValueError(f"native broadcast shape {operand.shape} is incompatible with reference shape {reference.shape}.")


@lru_cache(maxsize=64)
def _rmsnorm_residual_gate_kernel(hidden_size: int, gate_mode: int, gate_rows_per_batch: int, eps: float):
    mx = _mx()
    threads = _native_thread_count(hidden_size)
    source = (
        _source("rmsnorm_residual_gate")
        .replace("{{HIDDEN_SIZE}}", str(int(hidden_size)))
        .replace("{{THREADS}}", str(int(threads)))
        .replace("{{GATE_MODE}}", str(int(gate_mode)))
        .replace("{{GATE_ROWS_PER_BATCH}}", str(int(gate_rows_per_batch)))
        .replace("{{EPS}}", repr(float(eps)))
    )
    return mx.fast.metal_kernel(
        name=(
            "mlx_dit_rmsnorm_residual_gate"
            f"_h{int(hidden_size)}_g{int(gate_mode)}r{int(gate_rows_per_batch)}_e{abs(hash(float(eps))) & 0xffff:x}"
        ),
        input_names=["x", "weight", "residual", "gate"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _attn_prep_qknorm_rope_pack_kernel(head_dim: int, tokens: int, heads: int, freq_batch: int, eps: float):
    mx = _mx()
    source = (
        _source("attn_prep_qknorm_rope_pack")
        .replace("{{HEAD_DIM}}", str(int(head_dim)))
        .replace("{{TOKENS}}", str(int(tokens)))
        .replace("{{HEADS}}", str(int(heads)))
        .replace("{{FREQ_BATCH}}", str(int(freq_batch)))
        .replace("{{EPS}}", repr(float(eps)))
    )
    return mx.fast.metal_kernel(
        name=(
            "mlx_dit_attn_prep_qknorm_rope_pack"
            f"_h{int(heads)}_d{int(head_dim)}_t{int(tokens)}_fb{int(freq_batch)}"
        ),
        input_names=["q", "k", "v", "q_weight", "k_weight", "freqs"],
        output_names=["q_out", "k_out", "v_out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _native_bf16_self_attention_kernel(head_dim: int, tokens: int, heads: int, freq_batch: int, eps: float, scale: float):
    mx = _mx()
    source = (
        _source("native_bf16_self_attention")
        .replace("{{HEAD_DIM}}", str(int(head_dim)))
        .replace("{{TOKENS}}", str(int(tokens)))
        .replace("{{HEADS}}", str(int(heads)))
        .replace("{{FREQ_BATCH}}", str(int(freq_batch)))
        .replace("{{EPS}}", repr(float(eps)))
        .replace("{{SCALE}}", repr(float(scale)))
    )
    return mx.fast.metal_kernel(
        name=f"mlx_dit_native_bf16_self_attention_h{int(heads)}_d{int(head_dim)}_t{int(tokens)}_fb{int(freq_batch)}",
        input_names=["q", "k", "v", "q_weight", "k_weight", "freqs"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def ffn_silu_mul(gate: Any, up: Any):
    if gate.shape != up.shape:
        raise ValueError(f"ffn_silu_mul requires matching shapes; got {gate.shape} and {up.shape}.")
    kernel = _elementwise_kernel("mlx_dit_ffn_silu_mul", "ffn_silu_mul")
    return kernel(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


@lru_cache(maxsize=32)
def _ffn_packed_silu_mul_split_kernel(hidden_dim: int):
    mx = _mx()
    source = _source("ffn_packed_silu_mul_split").replace("{{HIDDEN_DIM}}", str(int(hidden_dim)))
    return mx.fast.metal_kernel(
        name=f"mlx_dit_ffn_packed_silu_mul_split_h{int(hidden_dim)}",
        input_names=["gu"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def ffn_packed_silu_mul_split(gu: Any, hidden_dim: int | None = None):
    if len(gu.shape) < 2:
        raise ValueError(f"ffn_packed_silu_mul_split requires [..., 2H] input; got {gu.shape}.")
    packed_dim = int(gu.shape[-1])
    if hidden_dim is None:
        if packed_dim % 2 != 0:
            raise ValueError(f"ffn_packed_silu_mul_split requires an even packed dim; got {packed_dim}.")
        hidden_dim = packed_dim // 2
    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0 or packed_dim != hidden_dim * 2:
        raise ValueError(
            "ffn_packed_silu_mul_split requires last dim to equal 2 * hidden_dim; "
            f"got shape {gu.shape}, hidden_dim={hidden_dim}."
        )
    out_shape = (*tuple(gu.shape[:-1]), hidden_dim)
    kernel = _ffn_packed_silu_mul_split_kernel(hidden_dim)
    return kernel(
        inputs=[gu],
        template=[("T", gu.dtype)],
        grid=(int(gu.size) // 2, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[gu.dtype],
    )[0]


def rmsnorm_residual_gate(x: Any, weight: Any, residual: Any, gate: Any, *, eps: float = 1e-5):
    if len(x.shape) < 2:
        raise ValueError("rmsnorm_residual_gate requires an input with at least 2 dimensions.")
    hidden_size = int(x.shape[-1])
    if int(weight.shape[-1]) != hidden_size:
        raise ValueError(f"rmsnorm_residual_gate weight hidden size mismatch: {weight.shape[-1]} vs {hidden_size}.")
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError(f"rmsnorm_residual_gate residual shape mismatch: {x.shape} vs {residual.shape}.")
    gate_mode, gate_rows_per_batch = _broadcast_mode(x, gate)
    outer = int(x.size // hidden_size)
    kernel = _rmsnorm_residual_gate_kernel(hidden_size, gate_mode, gate_rows_per_batch, float(eps))
    threads = _native_thread_count(hidden_size)
    return kernel(
        inputs=[x, weight, residual, gate],
        template=[("T", x.dtype)],
        grid=(outer * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _validate_attn_prep_inputs(q: Any, k: Any, v: Any, q_weight: Any, k_weight: Any, freqs_cis: Any) -> tuple[int, int, int, int]:
    if len(q.shape) != 4:
        raise ValueError(f"attn_prep_qknorm_rope_pack requires q with shape [B,T,H,D]; got {q.shape}.")
    if tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError(f"attn_prep_qknorm_rope_pack requires matching q/k/v shapes; got {q.shape}, {k.shape}, {v.shape}.")
    batch, tokens, heads, head_dim = (int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3]))
    if head_dim % 2 != 0:
        raise ValueError(f"attn_prep_qknorm_rope_pack requires an even head dim; got {head_dim}.")
    if int(q_weight.shape[-1]) != head_dim or int(k_weight.shape[-1]) != head_dim:
        raise ValueError(
            "attn_prep_qknorm_rope_pack norm weights must match head dim; "
            f"got {q_weight.shape}, {k_weight.shape}, head dim {head_dim}."
        )
    if len(freqs_cis.shape) != 6:
        raise ValueError(f"attn_prep_qknorm_rope_pack requires freqs shape [B,T,1,D/2,2,2]; got {freqs_cis.shape}.")
    if int(freqs_cis.shape[0]) not in {1, batch}:
        raise ValueError(f"attn_prep_qknorm_rope_pack freqs batch {freqs_cis.shape[0]} is incompatible with batch {batch}.")
    if int(freqs_cis.shape[1]) != tokens or int(freqs_cis.shape[3]) != head_dim // 2:
        raise ValueError(
            "attn_prep_qknorm_rope_pack freqs token/head dims do not match q; "
            f"got freqs {freqs_cis.shape}, q {q.shape}."
        )
    return batch, tokens, heads, head_dim


def attn_prep_qknorm_rope_pack_eager(mx: Any, q: Any, k: Any, v: Any, q_weight: Any, k_weight: Any, freqs_cis: Any, *, eps: float = 1e-5):
    _validate_attn_prep_inputs(q, k, v, q_weight, k_weight, freqs_cis)
    dtype = q.dtype

    def _norm(x: Any, weight: Any) -> Any:
        xf = x.astype(mx.float32)
        out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + float(eps))
        return (out * weight.astype(mx.float32)).astype(dtype)

    def _rope_head_major(x: Any) -> Any:
        freqs = freqs_cis.transpose(0, 2, 1, 3, 4, 5)
        x_ = x.astype(freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
        out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
        return out.reshape(*x.shape).astype(dtype)

    q_head = q.transpose(0, 2, 1, 3)
    k_head = k.transpose(0, 2, 1, 3)
    v_head = v.transpose(0, 2, 1, 3)
    return _rope_head_major(_norm(q_head, q_weight)), _rope_head_major(_norm(k_head, k_weight)), v_head


def attn_prep_qknorm_rope_pack(q: Any, k: Any, v: Any, q_weight: Any, k_weight: Any, freqs_cis: Any, *, eps: float = 1e-5):
    batch, tokens, heads, head_dim = _validate_attn_prep_inputs(q, k, v, q_weight, k_weight, freqs_cis)
    kernel = _attn_prep_qknorm_rope_pack_kernel(head_dim, tokens, heads, int(freqs_cis.shape[0]), float(eps))
    out_shape = (batch, heads, tokens, head_dim)
    return tuple(
        kernel(
            inputs=[q, k, v, q_weight, k_weight, freqs_cis],
            template=[("T", q.dtype)],
            grid=(batch * heads * tokens, 1, 1),
            threadgroup=(THREADGROUP_SIZE, 1, 1),
            output_shapes=[out_shape, out_shape, out_shape],
            output_dtypes=[q.dtype, k.dtype, v.dtype],
        )
    )


def _native_attention_max_tokens() -> int:
    raw = os.environ.get("COMFY_MLX_DIT_NATIVE_ATTENTION_MAX_TOKENS", "256").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 256


def native_bf16_self_attention_reference(
    mx: Any,
    q: Any,
    k: Any,
    v: Any,
    q_weight: Any,
    k_weight: Any,
    freqs_cis: Any,
    *,
    scale: float,
    eps: float = 1e-5,
):
    q_head, k_head, v_head = attn_prep_qknorm_rope_pack_eager(mx, q, k, v, q_weight, k_weight, freqs_cis, eps=eps)
    out = mx.fast.scaled_dot_product_attention(q_head, k_head, v_head, scale=float(scale))
    return out.transpose(0, 2, 1, 3)


def native_bf16_self_attention(
    q: Any,
    k: Any,
    v: Any,
    q_weight: Any,
    k_weight: Any,
    freqs_cis: Any,
    *,
    scale: float,
    eps: float = 1e-5,
):
    batch, tokens, heads, head_dim = _validate_attn_prep_inputs(q, k, v, q_weight, k_weight, freqs_cis)
    max_tokens = _native_attention_max_tokens()
    if tokens > max_tokens:
        raise ValueError(
            "native_bf16_self_attention prototype is disabled for large token counts; "
            f"tokens={tokens}, max={max_tokens}. This avoids running a naïve O(T^2*D) feasibility kernel on production shapes."
        )
    kernel = _native_bf16_self_attention_kernel(head_dim, tokens, heads, int(freqs_cis.shape[0]), float(eps), float(scale))
    return kernel(
        inputs=[q, k, v, q_weight, k_weight, freqs_cis],
        template=[("T", q.dtype)],
        grid=(batch * tokens * heads * head_dim, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def verify_available_kernels(kernel_names: tuple[str, ...]) -> None:
    mx = _mx()
    if not kernel_names:
        return
    a = mx.array([[0.25, -0.5, 0.75, -1.0]], dtype=mx.float32)
    b = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)
    if any(name in kernel_names for name in ("ffn_silu_mul", "all")):
        mx.eval(ffn_silu_mul(a, b))
    if any(name in kernel_names for name in ("ffn_packed_silu_mul_split", "all")):
        packed = mx.concatenate([a, b], axis=-1)
        mx.eval(ffn_packed_silu_mul_split(packed, int(a.shape[-1])))
    if any(name in kernel_names for name in ("rmsnorm_residual_gate", "all")):
        x = mx.array([[[0.25, -0.5, 0.75, -1.0]]], dtype=mx.float32)
        weight = mx.ones((4,), dtype=mx.float32)
        gate = mx.ones((1, 1, 4), dtype=mx.float32)
        mx.eval(rmsnorm_residual_gate(x, weight, x, gate))
    if any(name in kernel_names for name in ("attn_prep_qknorm_rope_pack", "all")):
        q = mx.array([[[[0.25, -0.5, 0.75, -1.0]]]], dtype=mx.float32)
        w = mx.ones((4,), dtype=mx.float32)
        freqs = mx.array([[[[[[1.0, -0.0], [0.0, 1.0]], [[1.0, -0.0], [0.0, 1.0]]]]]], dtype=mx.float32)
        out = attn_prep_qknorm_rope_pack(q, q, q, w, w, freqs)
        mx.eval(*out)
    if any(name in kernel_names for name in ("native_bf16_self_attention", "all")):
        q = mx.array([[[[0.25, -0.5, 0.75, -1.0]], [[0.5, 0.25, -0.25, 1.0]]]], dtype=mx.float32)
        w = mx.ones((4,), dtype=mx.float32)
        freqs = mx.array(
            [
                [
                    [[[[1.0, -0.0], [0.0, 1.0]], [[1.0, -0.0], [0.0, 1.0]]]],
                    [[[[1.0, -0.0], [0.0, 1.0]], [[1.0, -0.0], [0.0, 1.0]]]],
                ]
            ],
            dtype=mx.float32,
        )
        mx.eval(native_bf16_self_attention(q, q, q, w, w, freqs, scale=0.5))
