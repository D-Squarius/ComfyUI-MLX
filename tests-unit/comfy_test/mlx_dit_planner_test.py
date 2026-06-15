from __future__ import annotations

import pytest

from comfy.backends.mlx_dit_planner import (
    ATTENTION_CORE_BTHD_TO_BTHD,
    ATTENTION_PREP_BTH_TO_BHTD,
    LINEAR_PLAN_CURRENT,
    LAYOUT_BHTD,
    LAYOUT_BTH,
    LinearSpec,
    candidate_env,
    candidate_names,
    candidate_registry_json,
    candidate_spec,
    default_candidate_names,
    linear_spec,
    normalize_linear_plan,
    planned_fused_linear,
    planned_linear,
)
from comfy.backends.mlx_quant import MLXLinearWeight


mx = pytest.importorskip("mlx.core")


def test_normalize_linear_plan_rejects_unknown_values():
    assert normalize_linear_plan(None) == LINEAR_PLAN_CURRENT
    assert normalize_linear_plan("") == LINEAR_PLAN_CURRENT
    assert normalize_linear_plan("default") == LINEAR_PLAN_CURRENT
    assert normalize_linear_plan("flatten_2d") == "flatten_2d"
    assert normalize_linear_plan("pad_m64") == "pad_m64"
    assert normalize_linear_plan("not-a-real-plan") == LINEAR_PLAN_CURRENT


def test_linear_spec_for_dense_weight():
    weight = mx.ones((4, 3), dtype=mx.float32)
    ref = MLXLinearWeight(weight=weight, name="proj.weight", impl="dense")
    spec = linear_spec(ref, role="q")

    assert isinstance(spec, LinearSpec)
    assert spec.name == "proj.weight"
    assert spec.role == "q"
    assert spec.in_dim == 3
    assert spec.out_dim == 4
    assert not spec.quantized


def test_layout_contract_serialization():
    data = ATTENTION_PREP_BTH_TO_BHTD.to_json()

    assert data["input_layouts"] == [LAYOUT_BTH]
    assert data["output_layouts"] == [LAYOUT_BHTD, LAYOUT_BHTD, LAYOUT_BHTD]
    assert data["output_contiguous"] is True
    assert data["supports_bf16"] is True

    core = ATTENTION_CORE_BTHD_TO_BTHD.to_json()
    assert core["input_layouts"] == ["BTHD", "BTHD", "BTHD"]
    assert core["output_layouts"] == ["BTHD"]
    assert core["supports_bf16"] is True
    assert core["supports_q8"] is False


def test_candidate_registry_exposes_implemented_and_planned_candidates():
    registry = candidate_registry_json()
    implemented = candidate_registry_json(implemented_only=True)

    assert "linear.bf16.flatten_2d" in registry
    assert "linear.bf16.pad_m64" in implemented
    assert "attention.prep.mlx_layout_pack" in implemented
    assert "attention.prep.metal_qknorm_rope_pack" in registry
    assert "attention.prep.metal_qknorm_rope_pack" in implemented
    assert "attention.core.native_bf16_self_attn" in implemented
    assert "attention.sdpa.contiguous_inputs" in implemented
    assert "attention.output.contiguous_before_projection" in implemented
    assert "linear.bf16.fp16_compute" in implemented
    assert "linear.bf16.fp16_compute_no_cast" in implemented
    assert "norm.rmsnorm_residual_gate" in implemented
    assert "attention.prep.metal_qknorm_rope_pack" not in candidate_registry_json(default_only=True)
    assert "attention.core.native_bf16_self_attn" not in candidate_registry_json(default_only=True)
    assert registry["attention.prep.metal_qknorm_rope_pack"]["custom_kernel"] is True
    assert registry["attention.prep.metal_qknorm_rope_pack"]["default_enabled"] is False
    assert "failed default gate" in registry["attention.prep.metal_qknorm_rope_pack"]["decision"]
    assert registry["attention.core.native_bf16_self_attn"]["custom_kernel"] is True
    assert registry["attention.sdpa.contiguous_inputs"]["custom_kernel"] is False
    assert registry["attention.output.contiguous_before_projection"]["custom_kernel"] is False
    assert registry["norm.rmsnorm_residual_gate"]["custom_kernel"] is True
    assert registry["norm.rmsnorm_residual_gate"]["default_enabled"] is False


def test_default_candidate_names_excludes_failed_research_candidate():
    assert "attention.prep.metal_qknorm_rope_pack" in candidate_names(implemented_only=True)
    assert "attention.prep.metal_qknorm_rope_pack" not in default_candidate_names()
    assert "attention.core.native_bf16_self_attn" in candidate_names(implemented_only=True)
    assert "attention.core.native_bf16_self_attn" not in default_candidate_names()
    assert "attention.sdpa.contiguous_inputs" in candidate_names(implemented_only=True)
    assert "attention.sdpa.contiguous_inputs" not in default_candidate_names()
    assert "attention.output.contiguous_before_projection" in candidate_names(implemented_only=True)
    assert "attention.output.contiguous_before_projection" not in default_candidate_names()
    assert "linear.bf16.fp16_compute" in candidate_names(implemented_only=True)
    assert "linear.bf16.fp16_compute" not in default_candidate_names()
    assert "linear.bf16.fp16_compute_no_cast" in candidate_names(implemented_only=True)
    assert "linear.bf16.fp16_compute_no_cast" not in default_candidate_names()
    assert "linear.bf16.fp16_ffn_compute" in candidate_names(implemented_only=True)
    assert "linear.bf16.fp16_ffn_compute" not in default_candidate_names()
    assert "linear.bf16.fp16_ffn_compute_no_cast" in candidate_names(implemented_only=True)
    assert "linear.bf16.fp16_ffn_compute_no_cast" not in default_candidate_names()
    assert "linear.bf16.pad_m64" in candidate_names(implemented_only=True)
    assert "linear.bf16.pad_m64" not in default_candidate_names()
    assert "ffn.bf16.compiled_region" in candidate_names(implemented_only=True)
    assert "ffn.bf16.compiled_region" not in default_candidate_names()
    assert "ffn.bf16.packed_gate_up_activation" in candidate_names(implemented_only=True)
    assert "ffn.bf16.packed_gate_up_activation" not in default_candidate_names()
    assert "norm.rmsnorm_residual_gate" in candidate_names(implemented_only=True)
    assert "norm.rmsnorm_residual_gate" not in default_candidate_names()


def test_candidate_env_only_enables_implemented_candidates():
    assert candidate_env("linear.bf16.flatten_2d") == {"COMFY_MLX_DIT_LINEAR_PLAN": "flatten_2d"}
    assert candidate_env("attention.prep.mlx_layout_pack") == {"COMFY_MLX_Z_IMAGE_ATTENTION_LAYOUT": "head_major_rope"}
    assert candidate_env("attention.prep.metal_qknorm_rope_pack") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "attn_prep_qknorm_rope_pack"
    }
    assert candidate_env("attention.core.native_bf16_self_attn") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "native_bf16_self_attn"
    }
    assert candidate_env("attention.sdpa.contiguous_inputs") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "sdpa_contiguous_inputs"
    }
    assert candidate_env("attention.output.contiguous_before_projection") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "attn_output_contiguous"
    }
    assert candidate_env("ffn.bf16.compiled_region") == {"COMFY_MLX_Z_IMAGE_COMPILE": "ffn_region"}
    assert candidate_env("ffn.bf16.packed_gate_up_activation") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "ffn_packed_silu_mul_split"
    }
    assert candidate_env("norm.rmsnorm_residual_gate") == {
        "COMFY_MLX_Z_IMAGE_NATIVE_OPS": "rmsnorm_residual_gate"
    }
    assert candidate_env("linear.bf16.fp16_compute") == {"COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16"}
    assert candidate_env("linear.bf16.fp16_compute_no_cast") == {"COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16_no_cast"}
    assert candidate_env("linear.bf16.fp16_ffn_compute") == {
        "COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16",
        "COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS": "feed_forward",
    }
    assert candidate_env("linear.bf16.fp16_ffn_compute_no_cast") == {
        "COMFY_MLX_DIT_BF16_COMPUTE_DTYPE": "float16_no_cast",
        "COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS": "feed_forward",
    }
    assert candidate_env("linear.bf16.pad_m64") == {"COMFY_MLX_DIT_LINEAR_PLAN": "pad_m64"}
    assert candidate_spec("missing") is None


def test_dense_linear_can_use_fp16_compute_and_bf16_output():
    x = mx.arange(12, dtype=mx.bfloat16).reshape(1, 3, 4) / 10.0
    weight = mx.ones((5, 4), dtype=mx.float16)
    ref = MLXLinearWeight(
        weight=weight,
        name="proj.weight",
        impl="dense_fp16_compute",
        compute_dtype=mx.float16,
        output_dtype=mx.bfloat16,
    )

    out = ref(mx, x)
    mx.eval(out)

    assert out.shape == (1, 3, 5)
    assert out.dtype == mx.bfloat16
    assert ref.metadata()["compute_dtype"].endswith("float16")
    assert ref.metadata()["output_dtype"].endswith("bfloat16")


@pytest.mark.parametrize("plan", ["flatten_2d", "pretransposed", "flatten_pretransposed", "addmm_bias", "pad_m64"])
def test_planned_dense_linear_matches_current(plan):
    x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4) / 10.0
    weight = mx.arange(20, dtype=mx.float32).reshape(5, 4) / 7.0
    bias = mx.arange(5, dtype=mx.float32) / 11.0
    ref = MLXLinearWeight(weight=weight, bias=bias, name="test.proj", impl="dense")

    expected = ref(mx, x)
    candidate = planned_linear(mx, ref, plan=plan, role="test")(mx, x)
    mx.eval(expected, candidate)

    assert mx.allclose(candidate, expected, rtol=1.0e-5, atol=1.0e-5).item()
    assert planned_linear(mx, ref, plan=plan, role="test").metadata()["plan"] == plan


def test_pad_m64_plan_slices_padded_linear_output_back_to_tokens():
    x = mx.arange(68, dtype=mx.float32).reshape(1, 17, 4) / 10.0
    weight = mx.arange(20, dtype=mx.float32).reshape(5, 4) / 7.0
    ref = MLXLinearWeight(weight=weight, name="proj.weight", impl="dense")

    out = planned_linear(mx, ref, plan="pad_m64", role="q")(mx, x)
    expected = ref(mx, x)
    mx.eval(out, expected)

    assert out.shape == (1, 17, 5)
    assert mx.allclose(out, expected, rtol=1.0e-5, atol=1.0e-5).item()


def test_planned_fused_linear_matches_split_outputs():
    x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4) / 10.0
    left = MLXLinearWeight(weight=mx.arange(20, dtype=mx.float32).reshape(5, 4) / 9.0, name="left", impl="dense")
    right = MLXLinearWeight(weight=mx.arange(12, dtype=mx.float32).reshape(3, 4) / 13.0, name="right", impl="dense")

    fused = planned_fused_linear(mx, [left, right], name="left_right", plan="flatten_pretransposed", role="pair")
    expected = mx.concatenate([left(mx, x), right(mx, x)], axis=-1)
    actual = fused(mx, x)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5).item()
    assert fused.metadata()["plan"] == "flatten_pretransposed"
