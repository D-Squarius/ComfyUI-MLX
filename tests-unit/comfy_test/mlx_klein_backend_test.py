from __future__ import annotations

import torch
import pytest

import comfy.latent_formats
from comfy.backends.mlx_denoiser_island import (
    ContractMLXDenoiserRuntime,
    MLXDenoiserFunctionWrapper,
    UnsupportedIslandRequest,
    is_mlx_denoiser_island_model,
)
from comfy.backends.mlx_klein import (
    KLEIN_4B_MLX_8BIT_ALIAS,
    KLEIN_4B_MLX_4BIT_ALIAS,
    KLEIN_4B_MLX_BF16_ALIAS,
    KLEIN_9B_MLX_8BIT_ALIAS,
    KLEIN_9B_MLX_4BIT_ALIAS,
    KLEIN_9B_MLX_BF16_ALIAS,
    KleinIslandRuntime,
    MLXKleinWeights,
    _candidate_flags,
    create_klein_mlx_island_model,
    list_mlx_klein_choices,
    resolve_mlx_klein_model_path,
)


def _wrapper(model_patcher) -> MLXDenoiserFunctionWrapper:
    return model_patcher.model_options["model_function_wrapper"]


def test_mlx_klein_aliases_are_available_and_resolve_to_vendor_paths():
    choices = list_mlx_klein_choices()

    assert choices == [
        KLEIN_4B_MLX_BF16_ALIAS,
        KLEIN_4B_MLX_8BIT_ALIAS,
        KLEIN_4B_MLX_4BIT_ALIAS,
        KLEIN_9B_MLX_BF16_ALIAS,
        KLEIN_9B_MLX_8BIT_ALIAS,
        KLEIN_9B_MLX_4BIT_ALIAS,
    ]
    assert resolve_mlx_klein_model_path(KLEIN_4B_MLX_BF16_ALIAS).name == "flux-2-klein-4b.safetensors"
    assert resolve_mlx_klein_model_path(KLEIN_4B_MLX_8BIT_ALIAS).name == "flux2-klein-4b-8bit"
    assert resolve_mlx_klein_model_path("flux-2-klein-4b.safetensors") is None


def test_mlx_klein_vendor_root_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_MLX_KLEIN_MODEL_ROOT", str(tmp_path))

    assert resolve_mlx_klein_model_path(KLEIN_4B_MLX_8BIT_ALIAS) == tmp_path / "flux2-klein-4b-8bit"
    assert resolve_mlx_klein_model_path(KLEIN_9B_MLX_4BIT_ALIAS) == tmp_path / "flux2-klein-9b-4bit"


def test_klein_island_model_uses_flux2_latent_format_and_no_torch_transformer_models():
    model = create_klein_mlx_island_model(runtime=ContractMLXDenoiserRuntime())
    wrapper = _wrapper(model)

    assert is_mlx_denoiser_island_model(model.model)
    assert model.model.model_family == "klein"
    assert isinstance(model.model.latent_format, comfy.latent_formats.Flux2)
    assert wrapper.models() == []
    assert model.model_patches_models() == []


def test_klein_extra_conds_pad_flux2_context_and_keep_guidance():
    model = create_klein_mlx_island_model(runtime=ContractMLXDenoiserRuntime())
    context = torch.randn(1, 9, 7680)

    out = model.model.extra_conds(cross_attn=context, guidance=1.0)

    assert out["c_crossattn"].cond.shape == (1, 512, 7680)
    assert torch.equal(out["c_crossattn"].cond[:, :9], context)
    assert torch.count_nonzero(out["c_crossattn"].cond[:, 9:]) == 0
    assert out["guidance"].cond.shape == (1,)
    assert out["guidance"].cond.item() == 1.0


def test_klein_runtime_rejects_unsupported_attention_mask():
    runtime = KleinIslandRuntime(model_path="/tmp/missing-klein")

    with pytest.raises(UnsupportedIslandRequest, match="attention_mask"):
        runtime.validate_extra_conds({"attention_mask": torch.ones(1, 512)})


def test_klein_candidate_flags_are_internal_env_gates(monkeypatch):
    monkeypatch.setenv("COMFY_MLX_KLEIN_CACHE_STATIC_ROPE_IDS", "1")
    monkeypatch.setenv("COMFY_MLX_KLEIN_CACHE_TIMESTEP_EMBEDDING", "true")
    monkeypatch.setenv("COMFY_MLX_KLEIN_DISABLE_MEMORY_SNAPSHOT", "yes")
    monkeypatch.setenv("COMFY_MLX_KLEIN_SINGLE_EVAL_BOUNDARY", "on")
    monkeypatch.setenv("COMFY_MLX_KLEIN_PROFILE", "1")
    monkeypatch.setenv("COMFY_MLX_KLEIN_Q_LINEAR_IMPL", "quantized_matmul")

    flags = _candidate_flags()

    assert flags["cache_static_rope_ids"] is True
    assert flags["cache_timestep_embedding"] is True
    assert flags["disable_extra_memory_snapshot"] is True
    assert flags["single_eval_boundary"] is True
    assert flags["profile"] is True
    assert flags["q_linear_impl"] == "quantized_matmul"


def test_klein_dense_comfy_keys_normalize_to_island_transformer_roles():
    import mlx.core as mx

    weights = MLXKleinWeights("/tmp/fake.safetensors", dtype=mx.bfloat16)
    raw = {
        "img_in.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "txt_in.weight": mx.zeros((2, 3), dtype=mx.bfloat16),
        "time_in.in_layer.weight": mx.zeros((2, 4), dtype=mx.bfloat16),
        "time_in.out_layer.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "double_stream_modulation_img.lin.weight": mx.zeros((12, 2), dtype=mx.bfloat16),
        "double_stream_modulation_txt.lin.weight": mx.zeros((12, 2), dtype=mx.bfloat16),
        "single_stream_modulation.lin.weight": mx.zeros((6, 2), dtype=mx.bfloat16),
        "final_layer.adaLN_modulation.1.weight": mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.bfloat16),
        "final_layer.linear.weight": mx.zeros((128, 2), dtype=mx.bfloat16),
        "double_blocks.0.img_attn.qkv.weight": mx.zeros((6, 2), dtype=mx.bfloat16),
        "double_blocks.0.txt_attn.qkv.weight": mx.zeros((6, 2), dtype=mx.bfloat16),
        "double_blocks.0.img_attn.proj.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "double_blocks.0.txt_attn.proj.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "double_blocks.0.img_attn.norm.query_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
        "double_blocks.0.img_attn.norm.key_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
        "double_blocks.0.txt_attn.norm.query_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
        "double_blocks.0.txt_attn.norm.key_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
        "double_blocks.0.img_mlp.0.weight": mx.zeros((4, 2), dtype=mx.bfloat16),
        "double_blocks.0.img_mlp.2.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "double_blocks.0.txt_mlp.0.weight": mx.zeros((4, 2), dtype=mx.bfloat16),
        "double_blocks.0.txt_mlp.2.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "single_blocks.0.linear1.weight": mx.zeros((8, 2), dtype=mx.bfloat16),
        "single_blocks.0.linear2.weight": mx.zeros((2, 4), dtype=mx.bfloat16),
        "single_blocks.0.norm.query_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
        "single_blocks.0.norm.key_norm.scale": mx.zeros((1,), dtype=mx.bfloat16),
    }

    out = weights._normalize_comfy_dense_weights(raw)

    assert out["x_embedder.weight"].shape == (2, 2)
    assert out["transformer_blocks.0.attn.to_q.weight"].shape == (2, 2)
    assert out["transformer_blocks.0.attn.to_k.weight"].shape == (2, 2)
    assert out["transformer_blocks.0.attn.to_v.weight"].shape == (2, 2)
    assert out["transformer_blocks.0.attn.add_q_proj.weight"].shape == (2, 2)
    assert out["transformer_blocks.0.ff.linear_in.weight"].shape == (4, 2)
    assert out["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"].shape == (8, 2)
    assert out["norm_out.linear.weight"].shape == (4, 2)
    assert out["norm_out.linear.weight"].tolist() == [[5, 6], [7, 8], [1, 2], [3, 4]]


def test_klein_wrapper_contract_returns_finite_flux2_latent_without_torch_apply_model():
    runtime = ContractMLXDenoiserRuntime()
    model = create_klein_mlx_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 128, 32, 32)
    sigma = torch.tensor([0.5])
    context = torch.randn(1, 512, 7680)

    def forbidden_apply_model(*_args, **_kwargs):
        raise AssertionError("Torch apply_model must not run for Klein MLX island.")

    out = wrapper(
        forbidden_apply_model,
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [0],
            "c": {
                "c_crossattn": context,
                "guidance": torch.tensor([1.0]),
                "transformer_options": {
                    "cond_or_uncond": [0],
                    "sigmas": sigma,
                },
            },
        },
    )

    assert runtime.calls == 1
    assert wrapper.calls == 1
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert wrapper.last_event["runtime"] == "contract"
