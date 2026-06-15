from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from comfy.backends.mlx_quant import MLXLinearWeight, fuse_linear_weights_by_output


LINEAR_PLAN_CURRENT = "current"
LINEAR_PLAN_FLATTEN_2D = "flatten_2d"
LINEAR_PLAN_PRETRANSPOSED = "pretransposed"
LINEAR_PLAN_FLATTEN_PRETRANSPOSED = "flatten_pretransposed"
LINEAR_PLAN_ADDDMM_BIAS = "addmm_bias"
LINEAR_PLAN_PAD_M64 = "pad_m64"
VALID_LINEAR_PLANS = {
    LINEAR_PLAN_CURRENT,
    LINEAR_PLAN_FLATTEN_2D,
    LINEAR_PLAN_PRETRANSPOSED,
    LINEAR_PLAN_FLATTEN_PRETRANSPOSED,
    LINEAR_PLAN_ADDDMM_BIAS,
    LINEAR_PLAN_PAD_M64,
}

LAYOUT_BTH = "BTH"
LAYOUT_BTHD = "BTHD"
LAYOUT_BHTD = "BHTD"
LAYOUT_M2D = "M2D"
LAYOUT_B_T_3HD = "BT3HD"
VALID_LAYOUTS = {
    LAYOUT_BTH,
    LAYOUT_BTHD,
    LAYOUT_BHTD,
    LAYOUT_M2D,
    LAYOUT_B_T_3HD,
}


@dataclass(frozen=True)
class LayoutContract:
    name: str
    input_layouts: tuple[str, ...]
    output_layouts: tuple[str, ...]
    output_contiguous: bool
    may_allocate_temp: bool
    supports_bf16: bool = True
    supports_q8: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_layouts": list(self.input_layouts),
            "output_layouts": list(self.output_layouts),
            "output_contiguous": self.output_contiguous,
            "may_allocate_temp": self.may_allocate_temp,
            "supports_bf16": self.supports_bf16,
            "supports_q8": self.supports_q8,
        }


@dataclass(frozen=True)
class OpCandidate:
    name: str
    family: str
    contract: LayoutContract
    env: tuple[tuple[str, str], ...] = ()
    implemented: bool = True
    default_enabled: bool = True
    custom_kernel: bool = False
    description: str = ""
    target_stages: tuple[str, ...] = ()
    decision: str = ""

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "contract": self.contract.to_json(),
            "env": self.env_dict(),
            "implemented": self.implemented,
            "default_enabled": self.default_enabled,
            "custom_kernel": self.custom_kernel,
            "description": self.description,
            "target_stages": list(self.target_stages),
            "decision": self.decision,
        }


def _candidate(
    name: str,
    family: str,
    *,
    contract: LayoutContract,
    env: dict[str, str] | None = None,
    implemented: bool = True,
    default_enabled: bool = True,
    custom_kernel: bool = False,
    description: str = "",
    target_stages: Sequence[str] = (),
    decision: str = "",
) -> OpCandidate:
    return OpCandidate(
        name=name,
        family=family,
        contract=contract,
        env=tuple(sorted((env or {}).items())),
        implemented=implemented,
        default_enabled=default_enabled,
        custom_kernel=custom_kernel,
        description=description,
        target_stages=tuple(target_stages),
        decision=decision,
    )


LINEAR_BTH_TO_BTH = LayoutContract(
    name="linear.bth_to_bth",
    input_layouts=(LAYOUT_BTH,),
    output_layouts=(LAYOUT_BTH,),
    output_contiguous=True,
    may_allocate_temp=False,
    supports_bf16=True,
    supports_q8=True,
)
LINEAR_M2D_TO_BTH = LayoutContract(
    name="linear.m2d_to_bth",
    input_layouts=(LAYOUT_BTH,),
    output_layouts=(LAYOUT_M2D, LAYOUT_BTH),
    output_contiguous=True,
    may_allocate_temp=True,
    supports_bf16=True,
    supports_q8=False,
)
ATTENTION_PREP_BTH_TO_BHTD = LayoutContract(
    name="attention.prep_bth_to_bhtd",
    input_layouts=(LAYOUT_BTH,),
    output_layouts=(LAYOUT_BHTD, LAYOUT_BHTD, LAYOUT_BHTD),
    output_contiguous=True,
    may_allocate_temp=True,
    supports_bf16=True,
    supports_q8=True,
)
ATTENTION_CORE_BTHD_TO_BTHD = LayoutContract(
    name="attention.core_bthd_to_bthd",
    input_layouts=(LAYOUT_BTHD, LAYOUT_BTHD, LAYOUT_BTHD),
    output_layouts=(LAYOUT_BTHD,),
    output_contiguous=True,
    may_allocate_temp=False,
    supports_bf16=True,
    supports_q8=False,
)
NORM_TO_LINEAR_INPUT = LayoutContract(
    name="norm.modulated_to_linear_input",
    input_layouts=(LAYOUT_BTH,),
    output_layouts=(LAYOUT_BTH,),
    output_contiguous=True,
    may_allocate_temp=True,
    supports_bf16=True,
    supports_q8=False,
)
RESIDUAL_BTH = LayoutContract(
    name="residual.gated_add_bth",
    input_layouts=(LAYOUT_BTH, LAYOUT_BTH),
    output_layouts=(LAYOUT_BTH,),
    output_contiguous=True,
    may_allocate_temp=False,
    supports_bf16=True,
    supports_q8=False,
)
Q8_DEQUANT_TO_LINEAR = LayoutContract(
    name="q8.dequant_to_linear_layout",
    input_layouts=(LAYOUT_BTH,),
    output_layouts=(LAYOUT_BTH,),
    output_contiguous=True,
    may_allocate_temp=True,
    supports_bf16=False,
    supports_q8=True,
)


OP_CANDIDATES: dict[str, OpCandidate] = {
    "linear.bf16.flatten_2d": _candidate(
        "linear.bf16.flatten_2d",
        "linear",
        contract=LINEAR_M2D_TO_BTH,
        env={"COMFY_MLX_DIT_LINEAR_PLAN": LINEAR_PLAN_FLATTEN_2D},
        description="Flatten [B,T,H] to [B*T,H] before dense BF16 matmul, then restore [B,T,O].",
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
    ),
    "linear.bf16.preoriented": _candidate(
        "linear.bf16.preoriented",
        "linear",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_DIT_LINEAR_PLAN": LINEAR_PLAN_PRETRANSPOSED},
        description="Cache a contiguous transposed BF16 weight for dense linear projections.",
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
    ),
    "linear.bf16.flatten_preoriented": _candidate(
        "linear.bf16.flatten_preoriented",
        "linear",
        contract=LINEAR_M2D_TO_BTH,
        env={"COMFY_MLX_DIT_LINEAR_PLAN": LINEAR_PLAN_FLATTEN_PRETRANSPOSED},
        description="Combine flattened 2D lowering with cached contiguous transposed BF16 weights.",
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
    ),
    "linear.bf16.pad_m64": _candidate(
        "linear.bf16.pad_m64",
        "linear",
        contract=LINEAR_M2D_TO_BTH,
        env={"COMFY_MLX_DIT_LINEAR_PLAN": LINEAR_PLAN_PAD_M64},
        implemented=True,
        default_enabled=False,
        description=(
            "Pad flattened dense BF16 linear inputs to an M multiple of 64, run MLX matmul, then slice back before "
            "restoring the original token shape."
        ),
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
        decision="internal candidate: tests whether M=4128 tail handling is costing Steel GEMM throughput",
    ),
    "linear.bf16.fp16_compute": _candidate(
        "linear.bf16.fp16_compute",
        "linear",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16"},
        implemented=True,
        default_enabled=False,
        description=(
            "Relaxed BF16 candidate: keep BF16 model semantics at the adapter boundary, but run dense DiT linears "
            "through FP16 compute and cast outputs back to BF16."
        ),
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
        decision="internal candidate: only acceptable if speed and image-quality gates both pass; not strict BF16 arithmetic",
    ),
    "linear.bf16.fp16_compute_no_cast": _candidate(
        "linear.bf16.fp16_compute_no_cast",
        "linear",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16_no_cast"},
        implemented=True,
        default_enabled=False,
        description=(
            "Relaxed precision candidate: run dense DiT linears in FP16 and leave outputs FP16 to avoid BF16 castback. "
            "This is exploratory and not strict BF16 arithmetic."
        ),
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
        decision="internal candidate: only acceptable if speed and image-quality gates both pass; not strict BF16 arithmetic",
    ),
    "linear.bf16.fp16_ffn_compute": _candidate(
        "linear.bf16.fp16_ffn_compute",
        "linear",
        contract=LINEAR_BTH_TO_BTH,
        env={
            "COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16",
            "COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS": "feed_forward",
        },
        implemented=True,
        default_enabled=False,
        description=(
            "Relaxed BF16 candidate: run only FFN dense linears through FP16 compute and cast FFN outputs back to BF16."
        ),
        target_stages=("ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
        decision="internal candidate: narrower than global FP16 compute; only acceptable if speed and image-quality gates both pass",
    ),
    "linear.bf16.fp16_ffn_compute_no_cast": _candidate(
        "linear.bf16.fp16_ffn_compute_no_cast",
        "linear",
        contract=LINEAR_BTH_TO_BTH,
        env={
            "COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16_no_cast",
            "COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS": "feed_forward",
        },
        implemented=True,
        default_enabled=False,
        description=(
            "Relaxed precision candidate: run only FFN dense linears in FP16 and leave their outputs FP16. "
            "This is exploratory and not strict BF16 arithmetic."
        ),
        target_stages=("ffn_gate_up", "ffn_down", "ffn_full", "full_block"),
        decision="internal candidate: validate carefully because global no-cast caused nonfinite sampler output",
    ),
    "attention.prep.mlx_layout_pack": _candidate(
        "attention.prep.mlx_layout_pack",
        "attention",
        contract=ATTENTION_PREP_BTH_TO_BHTD,
        env={"COMFY_MLX_Z_IMAGE_ATTENTION_LAYOUT": "head_major_rope"},
        description="Use the existing MLX head-major prep path to reduce attention layout churn before SDPA.",
        target_stages=("qk_norm_rope_head_major", "sdpa", "full_block"),
    ),
    "attention.prep.metal_qknorm_rope_pack": _candidate(
        "attention.prep.metal_qknorm_rope_pack",
        "attention",
        contract=ATTENTION_PREP_BTH_TO_BHTD,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "attn_prep_qknorm_rope_pack"},
        implemented=True,
        default_enabled=False,
        custom_kernel=True,
        description="Metal feasibility target: fuse Q/K norm, RoPE, and BHTD pack for SDPA.",
        target_stages=("attn_prep_qknorm_rope_pack", "full_block"),
        decision=(
            "failed default gate: isolated prep was about 56% faster, but 1024 BF16 end-to-end smoke regressed "
            "to about 103.61s total / 97.32s sampler"
        ),
    ),
    "attention.core.native_bf16_self_attn": _candidate(
        "attention.core.native_bf16_self_attn",
        "attention",
        contract=ATTENTION_CORE_BTHD_TO_BTHD,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "native_bf16_self_attn"},
        implemented=True,
        default_enabled=False,
        custom_kernel=True,
        description=(
            "Native BF16 full self-attention feasibility prototype. It is exact but naïve and token-limited; "
            "used only to test whether this op boundary is worth deeper tiled attention work."
        ),
        target_stages=("attention_core_native", "attention_core_plus_out_projection", "full_block"),
        decision="research only: bounded naïve prototype for deciding whether deeper tiled native attention is worth building",
    ),
    "attention.sdpa.contiguous_inputs": _candidate(
        "attention.sdpa.contiguous_inputs",
        "attention",
        contract=ATTENTION_PREP_BTH_TO_BHTD,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "sdpa_contiguous_inputs"},
        implemented=True,
        default_enabled=False,
        description=(
            "Force q/k/v to contiguous BHTD buffers immediately before MLX fast SDPA. "
            "This falsifies whether non-contiguous SDPA inputs are costing the DiT attention path."
        ),
        target_stages=("sdpa_contiguous_inputs", "sdpa", "full_block"),
        decision="internal candidate: benchmark-only unless exact-shape block and sampler gates prove a BF16 win",
    ),
    "attention.output.contiguous_before_projection": _candidate(
        "attention.output.contiguous_before_projection",
        "attention",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "attn_output_contiguous"},
        implemented=True,
        default_enabled=False,
        description=(
            "Force the post-SDPA [B,T,H*D] attention output to a contiguous buffer before the output projection. "
            "This falsifies hidden projection materialization after the SDPA transpose/reshape."
        ),
        target_stages=("attention_output_contiguous", "attention_output_projection", "full_block"),
        decision="internal candidate: default-disabled unless exact-shape block and sampler gates prove a BF16 win",
    ),
    "ffn.bf16.compiled_region": _candidate(
        "ffn.bf16.compiled_region",
        "ffn",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_Z_IMAGE_COMPILE": "ffn_region"},
        implemented=True,
        default_enabled=False,
        description="Compile only the stable BF16 FFN region w1/w3 -> silu_mul -> w2; internal gate only.",
        target_stages=("ffn_region", "ffn_full", "full_block", "full_transformer"),
        decision="internal candidate: default-disabled until exact-shape and sampler gates prove a BF16 win",
    ),
    "ffn.bf16.packed_gate_up_activation": _candidate(
        "ffn.bf16.packed_gate_up_activation",
        "ffn",
        contract=LINEAR_BTH_TO_BTH,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "ffn_packed_silu_mul_split"},
        implemented=True,
        default_enabled=False,
        custom_kernel=True,
        description="Pack FFN gate/up projection and fuse split+silu*up in a reusable native epilogue.",
        target_stages=(
            "ffn_gate_up_activation_split",
            "ffn_gate_up_activation_packed_split",
            "ffn_gate_up_activation_packed_native",
            "ffn_region",
            "ffn_full",
            "full_block",
            "full_transformer",
        ),
        decision="internal candidate: default-disabled until packed FFN microbench and sampler gates prove a BF16 win",
    ),
    "norm.rmsnorm_residual_gate": _candidate(
        "norm.rmsnorm_residual_gate",
        "norm",
        contract=RESIDUAL_BTH,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "rmsnorm_residual_gate"},
        implemented=True,
        default_enabled=False,
        custom_kernel=True,
        description="Fuse RMSNorm, gate multiply, and residual add after attention/FFN branches.",
        target_stages=("full_block", "full_transformer"),
        decision="internal candidate: row-wise reduction kernel must clear block and sampler gates before default use",
    ),
    "norm.modulated_rmsnorm_to_linear_input": _candidate(
        "norm.modulated_rmsnorm_to_linear_input",
        "norm",
        contract=NORM_TO_LINEAR_INPUT,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "modulated_rmsnorm_to_linear_input"},
        implemented=False,
        custom_kernel=True,
        description="Planned Metal feasibility target: fuse RMSNorm and modulation into contiguous linear input.",
        target_stages=("full_block",),
    ),
    "residual.gated_add": _candidate(
        "residual.gated_add",
        "residual",
        contract=RESIDUAL_BTH,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "gated_residual_add"},
        implemented=False,
        custom_kernel=True,
        description="Planned Metal feasibility target: fuse residual plus gate times branch output.",
        target_stages=("full_block",),
    ),
    "q8.dequant_pack_to_matmul_layout": _candidate(
        "q8.dequant_pack_to_matmul_layout",
        "q8",
        contract=Q8_DEQUANT_TO_LINEAR,
        env={"COMFY_MLX_Z_IMAGE_NATIVE_OPS": "q8_dequant_pack_to_matmul_layout"},
        implemented=False,
        custom_kernel=True,
        description="Planned Q8 feasibility target: dequantize directly into the selected matmul-ready layout.",
        target_stages=("qkv_projection", "attention_output_projection", "ffn_gate_up", "ffn_down", "full_block"),
    ),
}


def candidate_names(*, implemented_only: bool = False, default_only: bool = False) -> tuple[str, ...]:
    names = tuple(OP_CANDIDATES)
    if implemented_only:
        names = tuple(name for name in names if OP_CANDIDATES[name].implemented)
    if default_only:
        names = tuple(name for name in names if OP_CANDIDATES[name].default_enabled)
    return names


def default_candidate_names() -> tuple[str, ...]:
    return candidate_names(implemented_only=True, default_only=True)


def candidate_spec(name: str) -> OpCandidate | None:
    return OP_CANDIDATES.get(name)


def candidate_registry_json(*, implemented_only: bool = False, default_only: bool = False) -> dict[str, Any]:
    return {name: OP_CANDIDATES[name].to_json() for name in candidate_names(implemented_only=implemented_only, default_only=default_only)}


def candidate_env(name: str) -> dict[str, str]:
    candidate = candidate_spec(name)
    if candidate is None or not candidate.implemented:
        return {}
    return candidate.env_dict()


@dataclass(frozen=True)
class LinearSpec:
    name: str
    role: str
    in_dim: int
    out_dim: int
    dtype: str
    quantized: bool
    bits: int | None = None
    group_size: int | None = None
    impl: str = "dense"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "dtype": self.dtype,
            "quantized": self.quantized,
            "bits": self.bits,
            "group_size": self.group_size,
            "impl": self.impl,
        }


@dataclass(frozen=True)
class MultiLinearSpec:
    name: str
    role: str
    members: tuple[LinearSpec, ...]

    @property
    def in_dim(self) -> int:
        return self.members[0].in_dim if self.members else 0

    @property
    def out_dim(self) -> int:
        return sum(member.out_dim for member in self.members)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "members": [member.to_json() for member in self.members],
        }


@dataclass(frozen=True)
class AttentionSpec:
    name: str
    heads: int
    head_dim: int
    layout: str
    q: LinearSpec
    k: LinearSpec
    v: LinearSpec
    out: LinearSpec

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "layout": self.layout,
            "q": self.q.to_json(),
            "k": self.k.to_json(),
            "v": self.v.to_json(),
            "out": self.out.to_json(),
        }


@dataclass(frozen=True)
class FFNSpec:
    name: str
    gate: LinearSpec
    up: LinearSpec
    down: LinearSpec

    @property
    def hidden_dim(self) -> int:
        return self.gate.out_dim

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_dim": self.hidden_dim,
            "gate": self.gate.to_json(),
            "up": self.up.to_json(),
            "down": self.down.to_json(),
        }


@dataclass(frozen=True)
class BlockSpec:
    name: str
    family: str
    index: int | None
    stream_tokens: int
    hidden_dim: int
    attention: AttentionSpec
    ffn: FFNSpec

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "index": self.index,
            "stream_tokens": self.stream_tokens,
            "hidden_dim": self.hidden_dim,
            "attention": self.attention.to_json(),
            "ffn": self.ffn.to_json(),
        }


def normalize_linear_plan(value: str | None) -> str:
    if value is None:
        return LINEAR_PLAN_CURRENT
    plan = value.strip().lower()
    if plan in {"", "0", "false", "no", "off", "default", "none"}:
        return LINEAR_PLAN_CURRENT
    if plan not in VALID_LINEAR_PLANS:
        return LINEAR_PLAN_CURRENT
    return plan


def linear_spec(linear: Any, *, role: str = "") -> LinearSpec:
    quantized = bool(getattr(linear, "quantized", False))
    weight = linear.weight
    if quantized:
        bits = int(linear.bits or 8)
        in_dim = int(weight.shape[1]) * 32 // bits
    else:
        bits = None
        in_dim = int(weight.shape[1])
    return LinearSpec(
        name=str(getattr(linear, "name", "")),
        role=role,
        in_dim=in_dim,
        out_dim=int(weight.shape[0]),
        dtype=str(getattr(weight, "dtype", "")),
        quantized=quantized,
        bits=bits,
        group_size=int(getattr(linear, "group_size", 0) or 0) if quantized else None,
        impl=str(getattr(linear, "impl", "dense")),
    )


class PlannedLinear:
    """Opt-in execution-plan wrapper around an MLXLinearWeight.

    The wrapper intentionally delegates metadata and weight attributes so it can
    still be used by existing fusion/census helpers. Quantized linears currently
    fall back to the base implementation unless a later Q8 planner adds a
    measured strategy.
    """

    def __init__(self, mx: Any, base: MLXLinearWeight, *, plan: str, role: str = ""):
        self.base = base
        self.plan = normalize_linear_plan(plan)
        self.role = role
        self._prepared_weight_t = None
        if self.plan in {LINEAR_PLAN_PRETRANSPOSED, LINEAR_PLAN_FLATTEN_PRETRANSPOSED, LINEAR_PLAN_ADDDMM_BIAS} and not base.quantized:
            self._prepared_weight_t = mx.contiguous(base.weight.T)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def quantized(self) -> bool:
        return self.base.quantized

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any | None:
        return self.base.bias

    @property
    def scales(self) -> Any | None:
        return self.base.scales

    @property
    def biases(self) -> Any | None:
        return self.base.biases

    @property
    def group_size(self) -> int:
        return self.base.group_size

    @property
    def bits(self) -> int | None:
        return self.base.bits

    @property
    def mode(self) -> str:
        return self.base.mode

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def impl(self) -> str:
        return f"{self.base.impl}:{self.plan}" if self.plan != LINEAR_PLAN_CURRENT else self.base.impl

    def metadata(self) -> dict[str, Any]:
        data = dict(self.base.metadata())
        data["plan"] = self.plan
        data["role"] = self.role
        return data

    def _weight_t(self, mx: Any) -> Any:
        if self._prepared_weight_t is None:
            return self.base.weight.T
        return self._prepared_weight_t

    def __call__(self, mx: Any, x: Any) -> Any:
        if self.base.quantized or self.plan == LINEAR_PLAN_CURRENT:
            return self.base(mx, x)

        weight_t = self._weight_t(mx)
        if self.plan == LINEAR_PLAN_PRETRANSPOSED:
            out = x @ weight_t
            if self.base.bias is not None:
                out = out + self.base.bias
            return out

        if self.plan in {LINEAR_PLAN_FLATTEN_2D, LINEAR_PLAN_FLATTEN_PRETRANSPOSED, LINEAR_PLAN_ADDDMM_BIAS} and len(x.shape) > 2:
            original_shape = tuple(x.shape[:-1])
            x_2d = x.reshape((-1, int(x.shape[-1])))
            if self.plan == LINEAR_PLAN_ADDDMM_BIAS and self.base.bias is not None:
                out_2d = mx.addmm(self.base.bias, x_2d, weight_t)
            else:
                out_2d = x_2d @ weight_t
                if self.base.bias is not None:
                    out_2d = out_2d + self.base.bias
            return out_2d.reshape((*original_shape, int(self.base.weight.shape[0])))

        if self.plan == LINEAR_PLAN_PAD_M64:
            original_shape = tuple(x.shape[:-1]) if len(x.shape) > 2 else (int(x.shape[0]),)
            x_2d = x.reshape((-1, int(x.shape[-1]))) if len(x.shape) > 2 else x
            original_m = int(x_2d.shape[0])
            pad_m = (-original_m) % 64
            if pad_m:
                pad = mx.zeros((pad_m, int(x_2d.shape[-1])), dtype=x_2d.dtype)
                x_mat = mx.concatenate([x_2d, pad], axis=0)
            else:
                x_mat = x_2d
            out_2d = x_mat @ weight_t
            if self.base.bias is not None:
                out_2d = out_2d + self.base.bias
            if pad_m:
                out_2d = out_2d[:original_m]
            if len(x.shape) > 2:
                return out_2d.reshape((*original_shape, int(self.base.weight.shape[0])))
            return out_2d

        out = x @ weight_t
        if self.base.bias is not None:
            out = out + self.base.bias
        return out


def planned_linear(mx: Any, linear: MLXLinearWeight, *, plan: str, role: str = "") -> Any:
    normalized = normalize_linear_plan(plan)
    if normalized == LINEAR_PLAN_CURRENT:
        return linear
    return PlannedLinear(mx, linear, plan=normalized, role=role)


def planned_fused_linear(mx: Any, linears: Sequence[Any], *, name: str, plan: str, role: str = "") -> Any:
    bases = [linear.base if isinstance(linear, PlannedLinear) else linear for linear in linears]
    fused = fuse_linear_weights_by_output(mx, bases, name=name)
    return planned_linear(mx, fused, plan=plan, role=role)


def multilinear_spec(name: str, role: str, linears: Iterable[Any], roles: Sequence[str]) -> MultiLinearSpec:
    members = tuple(linear_spec(linear, role=member_role) for linear, member_role in zip(linears, roles, strict=False))
    return MultiLinearSpec(name=name, role=role, members=members)
