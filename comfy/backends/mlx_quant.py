from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def detect_quantization_bits(weight: Any, scales: Any, group_size: int = 64) -> int:
    weight_cols = int(weight.shape[-1])
    scales_cols = int(scales.shape[-1])
    if scales_cols <= 0:
        raise ValueError("Quantized weight scales must have a non-zero last dimension.")
    return round(weight_cols * 32 / (scales_cols * int(group_size)))


@dataclass(frozen=True)
class MLXLinearWeight:
    weight: Any
    bias: Any | None = None
    scales: Any | None = None
    biases: Any | None = None
    group_size: int = 64
    bits: int | None = None
    mode: str = "affine"
    name: str = ""
    impl: str = "quantized_matmul"
    module: Any | None = None
    compute_dtype: Any | None = None
    output_dtype: Any | None = None

    @property
    def quantized(self) -> bool:
        return self.scales is not None

    def __call__(self, mx: Any, x: Any) -> Any:
        if self.scales is not None:
            bits = self.bits
            if bits is None:
                bits = detect_quantization_bits(self.weight, self.scales, self.group_size)
            if self.impl == "nn_quantized" and self.module is not None:
                out = self.module(x)
            else:
                out = mx.quantized_matmul(
                    x,
                    self.weight,
                    scales=self.scales,
                    biases=self.biases,
                    transpose=True,
                    group_size=self.group_size,
                    bits=bits,
                    mode=self.mode,
                )
        else:
            if self.compute_dtype is not None:
                out = x.astype(self.compute_dtype) @ self.weight.T
            else:
                out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        if self.output_dtype is not None:
            out = out.astype(self.output_dtype)
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantized": self.quantized,
            "impl": self.impl,
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "compute_dtype": None if self.compute_dtype is None else str(self.compute_dtype),
            "output_dtype": None if self.output_dtype is None else str(self.output_dtype),
        }


def mlx_linear(
    mx: Any,
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    scales: Any | None = None,
    biases: Any | None = None,
    group_size: int = 64,
    bits: int | None = None,
    mode: str = "affine",
) -> Any:
    return MLXLinearWeight(
        weight=weight,
        bias=bias,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )(mx, x)


def make_quantized_linear_module(mx: Any, linear: MLXLinearWeight) -> Any:
    if not linear.quantized:
        raise ValueError("nn.QuantizedLinear can only wrap quantized linear weights.")
    bits = _resolved_bits(linear)
    if bits is None:
        raise ValueError("Quantized linear bit width is required for nn.QuantizedLinear.")
    import mlx.nn as nn

    input_dims = int(linear.weight.shape[1]) * 32 // int(bits)
    output_dims = int(linear.weight.shape[0])
    module = nn.QuantizedLinear(
        input_dims,
        output_dims,
        bias=linear.bias is not None,
        group_size=linear.group_size,
        bits=bits,
        mode=linear.mode,
    )
    module.weight = linear.weight
    module.scales = linear.scales
    if linear.biases is not None:
        module.biases = linear.biases
    if linear.bias is not None:
        module.bias = linear.bias
    return module


def _resolved_bits(linear: MLXLinearWeight) -> int | None:
    if linear.scales is None:
        return None
    if linear.bits is not None:
        return int(linear.bits)
    return detect_quantization_bits(linear.weight, linear.scales, linear.group_size)


def _concat_optional(mx: Any, values: Sequence[Any | None], *, axis: int, field: str) -> Any | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Cannot fuse linear weights with mixed {field} presence.")
    return mx.concatenate(list(values), axis=axis)


def fuse_linear_weights_by_output(mx: Any, linears: Sequence[MLXLinearWeight], *, name: str = "") -> MLXLinearWeight:
    """Fuse same-input linear projections by concatenating their output rows.

    This works for dense weights and MLX packed affine quantized weights because
    both layouts use output rows on axis 0. The caller remains responsible for
    splitting the fused projection output back into the original output sizes.
    """

    if not linears:
        raise ValueError("Cannot fuse an empty linear list.")

    first = linears[0]
    quantized = first.quantized
    in_shape = tuple(first.weight.shape[1:])
    group_size = int(first.group_size)
    bits = _resolved_bits(first)
    mode = first.mode

    for linear in linears[1:]:
        if bool(linear.quantized) != bool(quantized):
            raise ValueError("Cannot fuse mixed dense and quantized linear weights.")
        if tuple(linear.weight.shape[1:]) != in_shape:
            raise ValueError(
                f"Cannot fuse linear weights with different input shapes: {in_shape} vs {tuple(linear.weight.shape[1:])}."
            )
        if int(linear.group_size) != group_size:
            raise ValueError(f"Cannot fuse quantized linear weights with different group sizes: {group_size} vs {linear.group_size}.")
        if linear.mode != mode:
            raise ValueError(f"Cannot fuse quantized linear weights with different modes: {mode} vs {linear.mode}.")
        linear_bits = _resolved_bits(linear)
        if linear_bits != bits:
            raise ValueError(f"Cannot fuse quantized linear weights with different bit widths: {bits} vs {linear_bits}.")

    return MLXLinearWeight(
        weight=mx.concatenate([linear.weight for linear in linears], axis=0),
        bias=_concat_optional(mx, [linear.bias for linear in linears], axis=0, field="bias"),
        scales=_concat_optional(mx, [linear.scales for linear in linears], axis=0, field="scales"),
        biases=_concat_optional(mx, [linear.biases for linear in linears], axis=0, field="quantized biases"),
        group_size=group_size,
        bits=bits,
        mode=mode,
        name=name,
        impl=first.impl,
        compute_dtype=first.compute_dtype,
        output_dtype=first.output_dtype,
    )
