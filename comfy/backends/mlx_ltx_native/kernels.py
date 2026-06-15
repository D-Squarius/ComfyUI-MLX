from __future__ import annotations

import importlib
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


@lru_cache(maxsize=32)
def _rmsnorm_residual_kernel(hidden_size: int, eps: float):
    mx = _mx()
    source = _source("rmsnorm_residual").replace("{{HIDDEN_SIZE}}", str(int(hidden_size))).replace(
        "{{EPS}}", repr(float(eps))
    )
    return mx.fast.metal_kernel(
        name=f"mlx_ltx_rmsnorm_residual_h{int(hidden_size)}",
        input_names=["x", "weight", "residual"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=32)
def _rope_pair_kernel(head_dim: int):
    mx = _mx()
    source = _source("rope_pair").replace("{{HEAD_DIM}}", str(int(head_dim)))
    return mx.fast.metal_kernel(
        name=f"mlx_ltx_rope_pair_h{int(head_dim)}",
        input_names=["x", "cos", "sin"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=32)
def _rope_split_kernel(head_dim: int):
    mx = _mx()
    source = _source("rope_split").replace("{{HEAD_DIM}}", str(int(head_dim)))
    return mx.fast.metal_kernel(
        name=f"mlx_ltx_rope_split_h{int(head_dim)}",
        input_names=["x", "cos", "sin"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _gelu_approx_kernel():
    mx = _mx()
    return mx.fast.metal_kernel(
        name="mlx_ltx_gelu_approx",
        input_names=["x"],
        output_names=["out"],
        source=_source("gelu_approx"),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _scale_shift_kernel(
    hidden_size: int,
    scale_mode: int,
    scale_rows_per_batch: int,
    shift_mode: int,
    shift_rows_per_batch: int,
):
    mx = _mx()
    source = (
        _source("scale_shift")
        .replace("{{HIDDEN_SIZE}}", str(int(hidden_size)))
        .replace("{{SCALE_MODE}}", str(int(scale_mode)))
        .replace("{{SCALE_ROWS_PER_BATCH}}", str(int(scale_rows_per_batch)))
        .replace("{{SHIFT_MODE}}", str(int(shift_mode)))
        .replace("{{SHIFT_ROWS_PER_BATCH}}", str(int(shift_rows_per_batch)))
    )
    return mx.fast.metal_kernel(
        name=(
            "mlx_ltx_scale_shift"
            f"_h{int(hidden_size)}_s{int(scale_mode)}r{int(scale_rows_per_batch)}"
            f"_t{int(shift_mode)}r{int(shift_rows_per_batch)}"
        ),
        input_names=["x", "scale", "shift"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=64)
def _gated_residual_kernel(hidden_size: int, gate_mode: int, gate_rows_per_batch: int):
    mx = _mx()
    source = (
        _source("gated_residual")
        .replace("{{HIDDEN_SIZE}}", str(int(hidden_size)))
        .replace("{{GATE_MODE}}", str(int(gate_mode)))
        .replace("{{GATE_ROWS_PER_BATCH}}", str(int(gate_rows_per_batch)))
    )
    return mx.fast.metal_kernel(
        name=f"mlx_ltx_gated_residual_h{int(hidden_size)}_g{int(gate_mode)}r{int(gate_rows_per_batch)}",
        input_names=["residual", "value", "gate"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


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


def ffn_silu_mul(gate: Any, up: Any):
    mx = _mx()
    if gate.shape != up.shape:
        raise ValueError(f"ffn_silu_mul requires matching shapes; got {gate.shape} and {up.shape}.")
    kernel = _elementwise_kernel("mlx_ltx_ffn_silu_mul", "ffn_silu_mul")
    return kernel(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


def gelu_approx(x: Any):
    kernel = _gelu_approx_kernel()
    return kernel(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def scale_shift(x: Any, scale: Any, shift: Any):
    if len(x.shape) < 2:
        raise ValueError("scale_shift requires an input with at least 2 dimensions.")
    hidden_size = int(x.shape[-1])
    scale_mode, scale_rows_per_batch = _broadcast_mode(x, scale)
    shift_mode, shift_rows_per_batch = _broadcast_mode(x, shift)
    kernel = _scale_shift_kernel(
        hidden_size,
        scale_mode,
        scale_rows_per_batch,
        shift_mode,
        shift_rows_per_batch,
    )
    return kernel(
        inputs=[x, scale, shift],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def gated_residual(residual: Any, value: Any, gate: Any):
    if residual.shape != value.shape:
        raise ValueError(f"gated_residual requires matching residual/value shapes; got {residual.shape} and {value.shape}.")
    if len(residual.shape) < 2:
        raise ValueError("gated_residual requires an input with at least 2 dimensions.")
    hidden_size = int(residual.shape[-1])
    gate_mode, gate_rows_per_batch = _broadcast_mode(residual, gate)
    kernel = _gated_residual_kernel(hidden_size, gate_mode, gate_rows_per_batch)
    return kernel(
        inputs=[residual, value, gate],
        template=[("T", residual.dtype)],
        grid=(residual.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[residual.dtype],
    )[0]


def residual_add(x: Any, residual: Any):
    if x.shape != residual.shape:
        raise ValueError(f"residual_add requires matching shapes; got {x.shape} and {residual.shape}.")
    kernel = _elementwise_kernel("mlx_ltx_residual_add", "residual_add")
    return kernel(
        inputs=[x, residual],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def rmsnorm_residual(x: Any, weight: Any, residual: Any, *, eps: float = 1e-6):
    if len(x.shape) < 2:
        raise ValueError("rmsnorm_residual requires an input with at least 2 dimensions.")
    hidden_size = int(x.shape[-1])
    if int(weight.shape[-1]) != hidden_size:
        raise ValueError(f"rmsnorm_residual weight hidden size mismatch: {weight.shape[-1]} vs {hidden_size}.")
    if x.shape != residual.shape:
        raise ValueError(f"rmsnorm_residual residual shape mismatch: {x.shape} vs {residual.shape}.")
    outer = int(x.size // hidden_size)
    kernel = _rmsnorm_residual_kernel(hidden_size, float(eps))
    return kernel(
        inputs=[x, weight, residual],
        template=[("T", x.dtype)],
        grid=(outer, hidden_size, 1),
        threadgroup=(1, min(hidden_size, THREADGROUP_SIZE), 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def rope_pair(x: Any, cos: Any, sin: Any):
    if len(x.shape) < 1:
        raise ValueError("rope_pair requires a non-scalar input.")
    head_dim = int(x.shape[-1])
    if head_dim % 2:
        raise ValueError(f"rope_pair requires an even head dimension; got {head_dim}.")
    if int(cos.shape[-1]) not in {head_dim, head_dim // 2}:
        raise ValueError(f"rope_pair cos shape {cos.shape} is incompatible with head dim {head_dim}.")
    if int(sin.shape[-1]) not in {head_dim, head_dim // 2}:
        raise ValueError(f"rope_pair sin shape {sin.shape} is incompatible with head dim {head_dim}.")
    pairs = int(x.size // 2)
    kernel = _rope_pair_kernel(head_dim)
    return kernel(
        inputs=[x, cos, sin],
        template=[("T", x.dtype)],
        grid=(pairs, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def rope_split(x: Any, cos: Any, sin: Any):
    if len(x.shape) < 1:
        raise ValueError("rope_split requires a non-scalar input.")
    head_dim = int(x.shape[-1])
    if head_dim % 2:
        raise ValueError(f"rope_split requires an even head dimension; got {head_dim}.")
    half = head_dim // 2
    if int(cos.shape[-1]) != half:
        raise ValueError(f"rope_split cos shape {cos.shape} is incompatible with head dim {head_dim}.")
    if int(sin.shape[-1]) != half:
        raise ValueError(f"rope_split sin shape {sin.shape} is incompatible with head dim {head_dim}.")
    kernel = _rope_split_kernel(head_dim)
    return kernel(
        inputs=[x, cos, sin],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def verify_available_kernels(kernel_names: tuple[str, ...]) -> None:
    mx = _mx()
    if not kernel_names:
        return
    a = mx.array([[0.25, -0.5, 0.75, -1.0]], dtype=mx.float32)
    b = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)
    if any(name in kernel_names for name in ("ffn_elementwise", "all_safe")):
        mx.eval(gelu_approx(a), ffn_silu_mul(a, b))
    if any(name in kernel_names for name in ("norm", "attention_prelude", "all_safe")):
        scale = mx.zeros((1, 1, 4), dtype=mx.float32)
        shift = mx.ones((1, 1, 4), dtype=mx.float32)
        mx.eval(scale_shift(a.reshape(1, 1, 4), scale, shift), gated_residual(a.reshape(1, 1, 4), b.reshape(1, 1, 4), scale))
    if any(name in kernel_names for name in ("rope", "attention_prelude", "all_safe")):
        cos = mx.ones((2,), dtype=mx.float32)
        sin = mx.zeros((2,), dtype=mx.float32)
        mx.eval(rope_pair(a, cos, sin), rope_split(a, cos, sin))
