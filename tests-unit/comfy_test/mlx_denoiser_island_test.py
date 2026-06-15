import pytest
import torch

import comfy.latent_formats
import comfy.utils
from comfy.backends.mlx_quant import MLXLinearWeight, detect_quantization_bits, fuse_linear_weights_by_output, make_quantized_linear_module, mlx_linear
from comfy.backends.mlx_denoiser_island import (
    BaseMLXDenoiserIslandRuntime,
    ContractMLXDenoiserRuntime,
    LTXAVIslandRuntime,
    MLXDenoiserCall,
    MLXDenoiserFunctionWrapper,
    TensorBridge,
    UnsupportedIslandRequest,
    ZImageIslandRuntime,
    create_mlx_denoiser_island_model,
    create_z_image_mlx_island_model,
    is_mlx_denoiser_island_model,
    split_ltx_av_latents,
    try_z_image_native_res_multistep_sampler,
)
from comfy.backends.mlx_z_image import (
    MLXZImageWeights,
    Q8_DEFAULT_DEQUANT_PATTERNS,
    Z_IMAGE_TURBO_MLX_Q8_ALIAS,
    _apply_rope,
    _apply_rope_head_major,
    _attention_layout,
    _compile_block_filter,
    _compile_mode_parts,
    _write_profile_event,
    resolve_mlx_z_image_model_path,
    z_image_candidate_env_overrides,
    z_image_native_q8_report_label,
    z_image_q8_strategy_env,
)


def _wrapper(model_patcher) -> MLXDenoiserFunctionWrapper:
    return model_patcher.model_options["model_function_wrapper"]


def test_mlx_denoiser_island_proxy_reports_no_managed_torch_models():
    runtime = ContractMLXDenoiserRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)

    assert is_mlx_denoiser_island_model(model.model)
    assert wrapper.models() == []
    assert model.model_patches_models() == []

    model.model_patches_to(torch.device("cpu"))

    assert _wrapper(model) is wrapper
    assert wrapper.return_device == torch.device("cpu")


def test_mlx_denoiser_island_runtime_adapter_contract_is_explicit():
    runtime = ContractMLXDenoiserRuntime()
    contract = runtime.adapter_contract().to_json()

    assert contract["runtime"] == "contract"
    assert contract["managed_torch_models"] == 0
    assert contract["owns_transformer_weights"] is True
    assert set(contract["required_methods"]) == {
        "load_weights",
        "prepare_context",
        "pack_latents",
        "unpack_latents",
        "map_timestep",
        "forward_model_output",
        "shape_census",
        "precision_policy",
    }


def test_mlx_denoiser_island_proxy_lightweight_model_contract():
    model = create_mlx_denoiser_island_model()
    wrapper = _wrapper(model)

    assert isinstance(model.model.latent_format, comfy.latent_formats.LTXAV)
    assert wrapper.to(torch.float16) is wrapper
    assert wrapper.return_device is None

    wrapper.to("cpu")

    assert wrapper.return_device == torch.device("cpu")
    assert model.model.memory_required((1, 128, 1, 4, 4)) == 0
    assert model.model.extra_conds_shapes() == {}


def test_mlx_denoiser_island_proxy_extra_conds_preserves_ltx_av_metadata():
    model = create_mlx_denoiser_island_model()
    video = torch.randn(1, 128, 1, 2, 2)
    audio = torch.randn(1, 8, 3, 16)
    _packed, latent_shapes = comfy.utils.pack_latents([video, audio])
    context = torch.randn(1, 1024, 6144)

    out = model.model.extra_conds(cross_attn=context, frame_rate=24, latent_shapes=latent_shapes)

    assert out["c_crossattn"].cond is context
    assert out["frame_rate"].cond == 24
    assert out["latent_shapes"].cond == latent_shapes


def test_mlx_denoiser_island_wrapper_contract_returns_finite_denoised_tensor():
    runtime = ContractMLXDenoiserRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(2, 128, 1, 4, 4)
    sigma = torch.tensor([0.5, 0.5])
    context = torch.randn(2, 8, 16)

    def forbidden_apply_model(*_args, **_kwargs):
        raise AssertionError("Torch apply_model must not run for the MLX island contract.")

    out = wrapper(
        forbidden_apply_model,
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [0, 1],
            "c": {
                "c_crossattn": context,
                "transformer_options": {
                    "cond_or_uncond": [0, 1],
                    "sigmas": sigma,
                },
            },
        },
    )

    assert runtime.calls == 1
    assert wrapper.calls == 1
    assert out.shape == x.shape
    assert out.device == x.device
    assert torch.isfinite(out).all()
    assert torch.equal(out, x)
    assert wrapper.last_event["cond_or_uncond"] == [0, 1]
    assert wrapper.last_event["input_shape"] == list(x.shape)


class _RecordingLTXRuntime(BaseMLXDenoiserIslandRuntime):
    name = "ltx_av"

    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward_model_output(self, call: MLXDenoiserCall) -> torch.Tensor:
        self.calls += 1
        self.last_call = call
        return torch.zeros_like(call.original_input)


def test_ltx_cfg1_zero_negative_batch_collapses_before_runtime():
    runtime = _RecordingLTXRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(2, 128, 1, 2, 2)
    sigma = torch.tensor([0.5, 0.5])
    negative_context = torch.zeros(1, 8, 16)
    positive_context = torch.randn(1, 8, 16)
    context = torch.cat([negative_context, positive_context], dim=0)

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [1, 0],
            "cond_scale": 1.0,
            "c": {
                "c_crossattn": context,
                "transformer_options": {
                    "cond_or_uncond": [1, 0],
                    "sigmas": sigma,
                },
            },
        },
    )

    assert runtime.calls == 1
    assert runtime.last_call is not None
    assert runtime.last_call.original_input.shape[0] == 1
    assert runtime.last_call.context.shape[0] == 1
    assert runtime.last_call.cond_or_uncond == [0]
    assert out.shape == x.shape
    assert torch.equal(out[0], out[1])
    assert wrapper.last_event["cond_or_uncond"] == [0]
    assert wrapper.last_event["batch_collapse"]["applied"] is True
    assert wrapper.last_event["batch_collapse"]["original_cond_or_uncond"] == [1, 0]


def test_ltx_batch_collapse_does_not_apply_when_cfg_is_not_one():
    runtime = _RecordingLTXRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(2, 128, 1, 2, 2)
    sigma = torch.tensor([0.5, 0.5])
    context = torch.cat([torch.zeros(1, 8, 16), torch.randn(1, 8, 16)], dim=0)

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [1, 0],
            "cond_scale": 2.0,
            "c": {
                "c_crossattn": context,
                "transformer_options": {
                    "cond_or_uncond": [1, 0],
                    "sigmas": sigma,
                },
            },
        },
    )

    assert runtime.calls == 1
    assert runtime.last_call.original_input.shape[0] == 2
    assert out.shape == x.shape
    assert wrapper.last_event["cond_or_uncond"] == [1, 0]
    assert wrapper.last_event["batch_collapse"]["applied"] is False
    assert wrapper.last_event["batch_collapse"]["reason"] == "cfg_not_one"


def test_tensor_bridge_mlx_to_torch_dlpack_output_bridge(monkeypatch):
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    monkeypatch.setenv("COMFY_MLX_ISLAND_TORCH_OUTPUT_BRIDGE", "dlpack")
    bridge = TensorBridge(require_mlx=True)
    like = torch.zeros((2, 3), dtype=torch.bfloat16)

    out = bridge.mlx_to_torch(mx.ones((2, 3), dtype=mx.float32), like=like)

    assert out.shape == like.shape
    assert out.dtype == torch.bfloat16
    assert torch.all(out == 1)
    assert bridge.counter.mlx_to_torch == 1


def test_tensor_bridge_torch_input_bridge_preserves_bfloat16(monkeypatch):
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    monkeypatch.setenv("COMFY_MLX_ISLAND_TORCH_INPUT_BRIDGE", "torch")
    bridge = TensorBridge(require_mlx=True)

    out = bridge.torch_to_mlx(torch.ones((2, 3), dtype=torch.bfloat16), dtype=mx.bfloat16)

    assert out.shape == (2, 3)
    assert out.dtype == mx.bfloat16
    assert bridge.counter.torch_to_cpu == 1
    assert bridge.counter.torch_to_mlx == 1


def test_mlx_denoiser_island_rejects_control_before_runtime_call():
    runtime = ContractMLXDenoiserRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 128, 1, 2, 2)
    sigma = torch.tensor([1.0])

    with pytest.raises(UnsupportedIslandRequest, match="ControlNet"):
        wrapper(
            lambda *_args, **_kwargs: None,
            {
                "input": x,
                "timestep": sigma,
                "cond_or_uncond": [0],
                "c": {
                    "control": object(),
                    "transformer_options": {"cond_or_uncond": [0]},
                },
            },
        )

    assert runtime.calls == 0
    assert wrapper.calls == 0


def test_mlx_denoiser_island_rejects_patch_hooks_before_runtime_call():
    runtime = ContractMLXDenoiserRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 128, 1, 2, 2)
    sigma = torch.tensor([1.0])

    with pytest.raises(UnsupportedIslandRequest, match="patches_replace"):
        wrapper(
            lambda *_args, **_kwargs: None,
            {
                "input": x,
                "timestep": sigma,
                "cond_or_uncond": [0],
                "c": {
                    "transformer_options": {
                        "cond_or_uncond": [0],
                        "patches_replace": {"dit": {("double_block", 0): object()}},
                    },
                },
            },
        )

    assert runtime.calls == 0
    assert wrapper.calls == 0


def test_mlx_denoiser_island_allows_empty_hooks_and_rejects_nonempty_callbacks():
    runtime = ContractMLXDenoiserRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 128, 1, 2, 2)
    sigma = torch.tensor([1.0])

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [0],
            "c": {
                "transformer_options": {
                    "cond_or_uncond": [0],
                    "wrappers": {},
                    "callbacks": {},
                },
            },
        },
    )

    assert torch.equal(out, x)
    assert runtime.calls == 1

    with pytest.raises(UnsupportedIslandRequest, match="callbacks"):
        wrapper(
            lambda *_args, **_kwargs: None,
            {
                "input": x,
                "timestep": sigma,
                "cond_or_uncond": [0],
                "c": {
                    "transformer_options": {
                        "cond_or_uncond": [0],
                        "callbacks": {"on_cfg": object()},
                    },
                },
            },
        )


def test_mlx_denoiser_island_model_sampling_is_comfy_compatible():
    model = create_mlx_denoiser_island_model(sampling_shift=1.25)
    model_sampling = model.get_model_object("model_sampling")
    x = torch.randn(1, 128, 1, 2, 2)
    sigma = torch.tensor([0.25])

    assert hasattr(model_sampling, "percent_to_sigma")
    assert model_sampling.calculate_input(sigma, x).shape == x.shape
    assert model_sampling.calculate_denoised(sigma, torch.zeros_like(x), x).shape == x.shape


def test_z_image_island_proxy_uses_flux_latent_format_and_shift():
    model = create_z_image_mlx_island_model(runtime=ZImageIslandRuntime(predictor=lambda **kwargs: kwargs["latent"]))

    assert isinstance(model.model.latent_format, comfy.latent_formats.Flux)
    assert model.get_model_object("model_sampling").shift == 3.0
    assert model.get_model_object("model_sampling").multiplier == 1.0
    assert _wrapper(model).models() == []


def test_z_image_wrapper_can_return_mlx_denoised_output(monkeypatch):
    pytest.importorskip("mlx.core")

    def fake_predictor(*, latent, **_kwargs):
        return latent * 0.25

    monkeypatch.setenv("COMFY_MLX_ISLAND_RETURN_DENOISED", "1")
    runtime = ZImageIslandRuntime(predictor=fake_predictor)
    model = create_z_image_mlx_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)
    sigma = torch.tensor([0.5])
    context = torch.randn(1, 4, 2560, dtype=torch.bfloat16)

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [0],
            "c": {
                "c_crossattn": context,
                "transformer_options": {
                    "cond_or_uncond": [0],
                    "sigmas": sigma,
                },
            },
        },
    )
    expected = model.model.model_sampling.calculate_denoised(sigma, (x * 0.25).float(), x)

    assert out.dtype == torch.float32
    assert torch.allclose(out, expected, atol=1e-2, rtol=1e-2)
    assert wrapper.last_event["calculate_denoised_seconds"] < 0.01
    assert wrapper.last_event["runtime_event"]["return_kind"] == "denoised"


def test_z_image_island_runtime_shape_census_and_precision_policy():
    runtime = ZImageIslandRuntime(predictor=lambda **kwargs: kwargs["latent"])
    x = torch.zeros(1, 16, 128, 128)
    call = MLXDenoiserCall(
        model_input=x,
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=torch.zeros(1, 25, 2560),
        control=None,
        transformer_options={},
        extra_conds={"num_tokens": 25},
        cond_or_uncond=[0],
    )

    contract = runtime.adapter_contract().to_json()
    census = runtime.shape_census(call)

    assert contract["model_family"] == "z_image"
    assert contract["precision_policy"]["default"] == "bf16"
    assert contract["precision_policy"]["available"] == ["bf16"]
    assert census["supports_dit_census"] is True
    assert census["model_input_shape"] == [1, 16, 128, 128]
    assert census["token_counts"]["image"] == 4096
    assert census["token_counts"]["stream"] == 4121

    q8_runtime = ZImageIslandRuntime(model_path="/tmp/Z-Image-Turbo-6B-MLX-Q8")
    q8_contract = q8_runtime.adapter_contract().to_json()
    assert q8_contract["precision_policy"]["default"] == "q8"
    assert q8_contract["precision_policy"]["available"] == ["q8"]


def test_z_image_island_runtime_rejects_missing_predictor():
    pytest.importorskip("mlx.core")
    runtime = ZImageIslandRuntime()
    x = torch.randn(1, 16, 8, 8)
    call = MLXDenoiserCall(
        model_input=x,
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=torch.randn(1, 4, 2560),
        control=None,
        transformer_options={},
        extra_conds={},
        cond_or_uncond=[0],
    )

    with pytest.raises(UnsupportedIslandRequest, match="no model_path"):
        runtime.forward_model_output(call)


def test_z_image_island_proxy_extra_conds_matches_lumina_context_metadata():
    model = create_z_image_mlx_island_model(runtime=ZImageIslandRuntime(predictor=lambda **kwargs: kwargs["latent"]))
    context = torch.randn(1, 17, 2560)

    out = model.model.extra_conds(cross_attn=context)

    assert out["c_crossattn"].cond is context
    assert out["num_tokens"].cond == 17


def test_z_image_island_runtime_rejects_unsupported_attention_mask():
    pytest.importorskip("mlx.core")
    runtime = ZImageIslandRuntime(predictor=lambda **kwargs: kwargs["latent"])
    x = torch.randn(1, 16, 8, 8)
    call = MLXDenoiserCall(
        model_input=x,
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=torch.randn(1, 4, 2560),
        control=None,
        transformer_options={},
        extra_conds={"attention_mask": torch.ones(1, 4)},
        cond_or_uncond=[0],
    )

    with pytest.raises(UnsupportedIslandRequest, match="attention_mask"):
        runtime.forward_model_output(call)


def test_z_image_island_runtime_rejects_wrong_context_dim():
    pytest.importorskip("mlx.core")
    runtime = ZImageIslandRuntime(predictor=lambda **kwargs: kwargs["latent"])
    x = torch.randn(1, 16, 8, 8)
    call = MLXDenoiserCall(
        model_input=x,
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=torch.randn(1, 4, 128),
        control=None,
        transformer_options={},
        extra_conds={},
        cond_or_uncond=[0],
    )

    with pytest.raises(UnsupportedIslandRequest, match="Z-Image context dim"):
        runtime.forward_model_output(call)


def test_z_image_island_wrapper_contract_returns_finite_denoised_tensor():
    pytest.importorskip("mlx.core")

    def fake_predictor(*, latent, mx, **_kwargs):
        return mx.zeros_like(latent)

    runtime = ZImageIslandRuntime(predictor=fake_predictor)
    model = create_z_image_mlx_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(1, 16, 8, 8)
    sigma = torch.tensor([0.5])
    context = torch.randn(1, 12, 2560)

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [0],
            "c": {
                "c_crossattn": context,
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
    assert out.device == x.device
    assert torch.isfinite(out).all()
    assert wrapper.last_event["runtime"] == "z_image"
    assert runtime.last_event["context_shape"] == [1, 12, 2560]


def test_z_image_native_res_multistep_sampler_is_env_gated(monkeypatch):
    monkeypatch.delenv("COMFY_MLX_Z_IMAGE_NATIVE_RES_MULTISTEP", raising=False)

    assert (
        try_z_image_native_res_multistep_sampler(
            model=object(),
            noise=torch.zeros(1, 16, 4, 4),
            latent_image=torch.zeros(1, 16, 4, 4),
            positive=[],
            negative=[],
            steps=8,
            cfg=1.0,
            sampler_name="res_multistep",
            scheduler="simple",
            denoise=1.0,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            seed=1,
        )
        is None
    )


def test_z_image_native_res_multistep_sampler_runtime_smoke():
    pytest.importorskip("mlx.core")

    def fake_predictor(*, latent, mx, **_kwargs):
        return mx.zeros_like(latent)

    runtime = ZImageIslandRuntime(predictor=fake_predictor)
    model = create_z_image_mlx_island_model(runtime=runtime)
    noise = torch.randn(1, 16, 4, 4)
    latent = torch.zeros_like(noise)
    context = torch.randn(1, 4, 2560)
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    out = runtime.sample_res_multistep_cfg1(
        noise=noise,
        latent_image=latent,
        sigmas=sigmas,
        context=context,
        num_tokens=4,
        model_sampling=model.model.model_sampling,
    )

    assert out.shape == latent.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert runtime.calls == 2
    assert runtime.last_event["return_kind"] == "native_res_multistep_cfg1"
    assert runtime.last_event["steps"] == 2


def test_z_image_unet_loader_alias_returns_island_model(monkeypatch, tmp_path):
    import folder_paths
    import nodes
    from comfy.backends.mlx_z_image import Z_IMAGE_TURBO_MLX_ALIAS

    model_file = tmp_path / "z_image_turbo_bf16.safetensors"
    model_file.write_bytes(b"placeholder")
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: str(model_file) if folder == "diffusion_models" else None)

    model, = nodes.UNETLoader().load_unet(Z_IMAGE_TURBO_MLX_ALIAS, "default")

    assert is_mlx_denoiser_island_model(model.model)
    assert _wrapper(model).models() == []


def test_z_image_q8_alias_resolves_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_Q8_MODEL_PATH", str(tmp_path))

    assert resolve_mlx_z_image_model_path(Z_IMAGE_TURBO_MLX_Q8_ALIAS) == str(tmp_path)


def test_z_image_q8_alias_keeps_legacy_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_Q8_PATH", str(tmp_path))

    assert resolve_mlx_z_image_model_path(Z_IMAGE_TURBO_MLX_Q8_ALIAS) == str(tmp_path)


def test_ltx_av_island_runtime_shape_census_and_precision_policy():
    runtime = LTXAVIslandRuntime("ltx-2.3-mlx-q4", bridge=runtime_bridge_no_mlx())
    video = torch.zeros(1, 128, 1, 8, 8)
    audio = torch.zeros(1, 8, 16, 16)
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=video,
        context=torch.zeros(1, 1024, 6144),
        control=None,
        transformer_options={},
        extra_conds={"frame_rate": 24},
        cond_or_uncond=[0],
    )

    contract = runtime.adapter_contract().to_json()
    census = runtime.shape_census(call)

    assert contract["model_family"] == "ltx_av"
    assert contract["precision_policy"]["default"] == "q4"
    assert contract["prompt_state_policy"] == "ltx_av_prompt_local_static_state"
    assert "video_audio_positions" in contract["supported_features"]
    assert "keyframe_guides" in contract["supported_features"]
    assert "sparse_guide_attention" in contract["supported_features"]
    assert "guide_attention_entries" not in contract["unsupported_features"]
    assert census["model_input_shape"] == [[1, 128, 1, 8, 8], [1, 8, 16, 16]]
    assert census["token_counts"] == {"video": 64, "audio": 16}
    assert census["shape_bucket"] == [1, 8, 8, 16, 24.0]
    assert census["prompt_state"]["global_prompt_cache"] is False
    assert census["mask_metadata"]["video"] == {"present": False}
    assert census["x0_model_cache"]["hit_for_shape"] is False


def runtime_bridge_no_mlx():
    from comfy.backends.mlx_denoiser_island import TensorBridge

    return TensorBridge(require_mlx=False)


def test_mlx_quant_detects_q8_group64():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    weight = mx.zeros((3840, 960), dtype=mx.uint32)
    scales = mx.zeros((3840, 60), dtype=mx.bfloat16)

    assert detect_quantization_bits(weight, scales, group_size=64) == 8


def test_mlx_quantized_linear_dispatch_matches_mlx_matmul():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    dense = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 10.0
    q_weight, scales, biases = mx.quantize(dense, group_size=64, bits=8)
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 7.0

    out = mlx_linear(mx, x, q_weight, scales=scales, biases=biases, group_size=64, bits=8)
    expected = mx.quantized_matmul(x, q_weight, scales=scales, biases=biases, transpose=True, group_size=64, bits=8)

    assert mx.allclose(out, expected).item()


def test_mlx_fused_dense_linear_matches_split_outputs():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 17.0
    w1 = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 19.0
    w2 = mx.arange(384, dtype=mx.float32).reshape(6, 64) / 23.0
    b1 = mx.arange(4, dtype=mx.float32) / 29.0
    b2 = mx.arange(6, dtype=mx.float32) / 31.0
    split = mx.concatenate([mlx_linear(mx, x, w1, b1), mlx_linear(mx, x, w2, b2)], axis=-1)
    fused = fuse_linear_weights_by_output(
        mx,
        [MLXLinearWeight(w1, bias=b1), MLXLinearWeight(w2, bias=b2)],
        name="dense_fused",
    )

    assert mx.allclose(fused(mx, x), split, atol=1.0e-5).item()


def test_mlx_fused_quantized_linear_matches_split_outputs():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    dense1 = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 13.0
    dense2 = mx.arange(384, dtype=mx.float32).reshape(6, 64) / 17.0
    q1, scales1, biases1 = mx.quantize(dense1, group_size=64, bits=8)
    q2, scales2, biases2 = mx.quantize(dense2, group_size=64, bits=8)
    x = mx.arange(192, dtype=mx.float32).reshape(3, 64) / 11.0
    split = mx.concatenate(
        [
            mlx_linear(mx, x, q1, scales=scales1, biases=biases1, group_size=64, bits=8),
            mlx_linear(mx, x, q2, scales=scales2, biases=biases2, group_size=64, bits=8),
        ],
        axis=-1,
    )
    fused = fuse_linear_weights_by_output(
        mx,
        [
            MLXLinearWeight(q1, scales=scales1, biases=biases1, group_size=64, bits=8),
            MLXLinearWeight(q2, scales=scales2, biases=biases2, group_size=64, bits=8),
        ],
        name="q8_fused",
    )

    assert mx.allclose(fused(mx, x), split).item()


def test_mlx_dequantized_linear_matches_quantized_output():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    dense = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 13.0
    q_weight, scales, biases = mx.quantize(dense, group_size=64, bits=8)
    dequantized = mx.dequantize(q_weight, scales, biases, group_size=64, bits=8, dtype=mx.bfloat16)
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 7.0

    quantized = MLXLinearWeight(q_weight, scales=scales, biases=biases, group_size=64, bits=8)(mx, x)
    dequant = MLXLinearWeight(dequantized, impl="dequantized")(mx, x)

    assert mx.allclose(dequant, quantized, rtol=1.0e-3, atol=2.0e-2).item()


def test_mlx_nn_quantized_linear_matches_quantized_matmul():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    dense = mx.arange(256, dtype=mx.float32).reshape(4, 64) / 10.0
    q_weight, scales, biases = mx.quantize(dense, group_size=64, bits=8)
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 7.0
    ref = MLXLinearWeight(q_weight, scales=scales, biases=biases, group_size=64, bits=8)
    nn_ref = MLXLinearWeight(
        q_weight,
        scales=scales,
        biases=biases,
        group_size=64,
        bits=8,
        impl="nn_quantized",
        module=make_quantized_linear_module(mx, ref),
    )

    assert mx.allclose(nn_ref(mx, x), ref(mx, x)).item()


def test_z_image_candidate_presets_scope_to_backend():
    assert z_image_candidate_env_overrides("q8_hybrid_dequant_plus_debug", backend="native_mlx_bf16") == {}
    assert z_image_candidate_env_overrides("bf16_compile_block", backend="native_mlx_q8") == {}
    assert z_image_candidate_env_overrides("linear.bf16.flatten_2d", backend="native_mlx_q8") == {}
    assert z_image_candidate_env_overrides("bf16_compile_main_layers", backend="native_mlx_bf16") == {
        "compile": "block",
        "compile_block_filter": "main_layers",
    }
    assert z_image_candidate_env_overrides("ffn.bf16.compiled_region", backend="native_mlx_bf16") == {"compile": "ffn_region"}
    assert z_image_candidate_env_overrides("ffn.bf16.packed_gate_up_activation", backend="native_mlx_bf16") == {
        "native_ops": "ffn_packed_silu_mul_split"
    }
    assert z_image_candidate_env_overrides("linear.bf16.fp16_compute", backend="native_mlx_bf16") == {"bf16_compute_dtype": "float16"}
    assert z_image_candidate_env_overrides("linear.bf16.fp16_ffn_compute", backend="native_mlx_bf16") == {
        "bf16_compute_dtype": "float16",
        "bf16_compute_patterns": "feed_forward",
    }
    assert z_image_candidate_env_overrides("linear.bf16.fp16_ffn_compute_no_cast", backend="native_mlx_bf16") == {
        "bf16_compute_dtype": "float16_no_cast",
        "bf16_compute_patterns": "feed_forward",
    }
    assert z_image_candidate_env_overrides("ffn.bf16.packed_gate_up_activation", backend="native_mlx_q8") == {}
    assert z_image_candidate_env_overrides("linear.bf16.flatten_2d", backend="native_mlx_bf16") == {"dit_linear_plan": "flatten_2d"}
    assert z_image_candidate_env_overrides("linear.bf16.pad_m64", backend="native_mlx_bf16") == {"dit_linear_plan": "pad_m64"}
    assert z_image_candidate_env_overrides("bf16_compile_block_alloc_reduce", backend="native_mlx_bf16") == {
        "compile": "block",
        "block_alloc_reduce": "1",
    }
    assert z_image_candidate_env_overrides("q8_hybrid_dequant_plus_debug", backend="native_mlx_q8") == {
        "q8_linear_impl": "hybrid_dequant",
        "q8_dequant_patterns": "attention.to_q,attention.to_k,attention.to_v,attention.to_out,feed_forward.w1,feed_forward.w2,feed_forward.w3",
    }
    assert z_image_candidate_env_overrides("attention_layout_variant", backend="native_mlx_bf16") == {"attention_layout": "head_major_rope"}


def test_z_image_q8_strategy_presets_are_explicit():
    assert z_image_q8_strategy_env("q8_baseline_debug") == {"q8_linear_impl": "quantized_matmul", "q8_dequant_patterns": ""}
    assert z_image_q8_strategy_env("q8_hybrid_dequant_debug") == {
        "q8_linear_impl": "hybrid_dequant",
        "q8_dequant_patterns": "attention.to_q,attention.to_k,attention.to_v,feed_forward.w1,feed_forward.w2,feed_forward.w3",
    }
    assert z_image_q8_strategy_env("q8_hybrid_dequant_plus_debug") == {
        "q8_linear_impl": "hybrid_dequant",
        "q8_dequant_patterns": ",".join(Q8_DEFAULT_DEQUANT_PATTERNS),
    }


def test_z_image_q8_report_label_is_single_production_mode():
    assert z_image_native_q8_report_label({"q8_linear_impl": "hybrid_dequant"}) == "ComfyUI Native MLX Island Q8"


def test_z_image_attention_layout_and_compile_env(monkeypatch):
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_ATTENTION_LAYOUT", "head_major_rope")
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_COMPILE", "block,transformer")
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_COMPILE_BLOCK_FILTER", "main_layers")

    assert _attention_layout() == "head_major_rope"
    assert _compile_mode_parts() == {"block", "transformer"}
    assert _compile_block_filter() == "main_layers"


def test_z_image_head_major_rope_matches_seq_major():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    x = mx.arange(1 * 5 * 3 * 8, dtype=mx.float32).reshape(1, 5, 3, 8) / 17.0
    freqs = mx.tile(mx.eye(2, dtype=mx.float32).reshape(1, 1, 1, 1, 2, 2), (1, 5, 1, 4, 1, 1))
    seq = _apply_rope(mx, x, freqs)
    head = _apply_rope_head_major(mx, x.transpose(0, 2, 1, 3), freqs).transpose(0, 2, 1, 3)

    assert mx.allclose(seq, head).item()


def test_z_image_profile_event_writer_schema(tmp_path):
    path = tmp_path / "profile.jsonl"

    _write_profile_event(
        str(path),
        {
            "event": "z_image_block_profile",
            "block": "layers.0",
            "block_family": "layers",
            "block_index": 0,
            "stage": "qkv_projection_split",
            "input_shape": [1, 1056, 3840],
            "output_shape": [[1, 1056, 3840]],
            "linear_backend": "quantized_matmul",
            "quantized": True,
            "bits": 8,
            "seconds": 0.1,
        },
    )

    event = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert event["event"] == "z_image_block_profile"
    assert event["block_index"] == 0
    assert event["linear_backend"] == "quantized_matmul"


def test_z_image_indexed_q8_weights_load_quant_metadata(tmp_path):
    pytest.importorskip("mlx.core")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    package = tmp_path / "z_image_q8"
    transformer = package / "transformer"
    transformer.mkdir(parents=True)
    shard = transformer / "0.safetensors"
    safetensors_torch.save_file(
        {
            "linear.weight": torch.zeros((4, 2), dtype=torch.uint32),
            "linear.scales": torch.ones((4, 1), dtype=torch.bfloat16),
            "linear.biases": torch.zeros((4, 1), dtype=torch.bfloat16),
            "linear.bias": torch.zeros((4,), dtype=torch.bfloat16),
        },
        str(shard),
    )
    (transformer / "model.safetensors.index.json").write_text(
        """
{
  "metadata": {"quantization_level": "8"},
  "weight_map": {
    "linear.weight": "0.safetensors",
    "linear.scales": "0.safetensors",
    "linear.biases": "0.safetensors",
    "linear.bias": "0.safetensors"
  }
}
""".strip(),
        encoding="utf-8",
    )

    import mlx.core as mx

    weights = MLXZImageWeights(package, dtype=mx.bfloat16)

    assert weights.quantization_level is None
    assert weights.get("linear.weight").shape == (4, 2)
    assert weights.quantization_level == "8"
    assert weights.has("linear.scales")
    assert weights.linear_ref("linear") is weights.linear_ref("linear")
    assert weights.linear_ref("linear").bits == 8


def test_z_image_q8_linear_impl_env_selects_dequantized(monkeypatch, tmp_path):
    pytest.importorskip("mlx.core")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    package = tmp_path / "z_image_q8"
    transformer = package / "transformer"
    transformer.mkdir(parents=True)
    shard = transformer / "0.safetensors"
    dense = torch.arange(256, dtype=torch.float32).reshape(4, 64) / 10.0
    import mlx.core as mx

    q_weight, scales, biases = mx.quantize(mx.array(dense.numpy()), group_size=64, bits=8)
    safetensors_torch.save_file(
        {
            "linear.weight": torch.from_numpy(__import__("numpy").array(q_weight)),
            "linear.scales": torch.from_numpy(__import__("numpy").array(scales)).to(torch.bfloat16),
            "linear.biases": torch.from_numpy(__import__("numpy").array(biases)).to(torch.bfloat16),
        },
        str(shard),
    )
    (transformer / "model.safetensors.index.json").write_text(
        """
{
  "metadata": {"quantization_level": "8"},
  "weight_map": {
    "linear.weight": "0.safetensors",
    "linear.scales": "0.safetensors",
    "linear.biases": "0.safetensors"
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_Q8_LINEAR_IMPL", "hybrid_dequant")
    monkeypatch.setenv("COMFY_MLX_Z_IMAGE_Q8_DEQUANT_PATTERNS", "linear")

    weights = MLXZImageWeights(package, dtype=mx.bfloat16)
    ref = weights.linear_ref("linear")

    assert not ref.quantized
    assert ref.impl == "dequantized"


def test_z_image_dense_bf16_can_select_fp16_compute(monkeypatch, tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import mlx.core as mx

    model_file = tmp_path / "z_image_dense.safetensors"
    safetensors_torch.save_file(
        {
            "linear.weight": torch.ones((4, 3), dtype=torch.bfloat16),
            "linear.bias": torch.zeros((4,), dtype=torch.bfloat16),
        },
        str(model_file),
    )
    monkeypatch.setenv("COMFY_MLX_DIT_BF16_COMPUTE_DTYPE", "float16")

    weights = MLXZImageWeights(model_file, dtype=mx.bfloat16)
    ref = weights.linear_ref("linear")
    x = mx.ones((1, 2, 3), dtype=mx.bfloat16)
    out = ref(mx, x)
    mx.eval(out)

    assert weights.bf16_compute_dtype_policy == "float16"
    assert ref.impl == "dense_fp16_compute"
    assert ref.weight.dtype == mx.float16
    assert out.dtype == mx.bfloat16


def test_z_image_bf16_compute_patterns_scope_fp16_to_ffn(monkeypatch, tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import mlx.core as mx

    model_file = tmp_path / "z_image_dense_patterns.safetensors"
    safetensors_torch.save_file(
        {
            "layers.0.attention.to_q.weight": torch.ones((4, 3), dtype=torch.bfloat16),
            "layers.0.feed_forward.w1.weight": torch.ones((4, 3), dtype=torch.bfloat16),
        },
        str(model_file),
    )
    monkeypatch.setenv("COMFY_MLX_DIT_BF16_COMPUTE_DTYPE", "float16")
    monkeypatch.setenv("COMFY_MLX_DIT_BF16_COMPUTE_PATTERNS", "feed_forward")

    weights = MLXZImageWeights(model_file, dtype=mx.bfloat16)
    attention_ref = weights.linear_ref("layers.0.attention.to_q", bias=False)
    ffn_ref = weights.linear_ref("layers.0.feed_forward.w1", bias=False)

    assert weights.bf16_compute_dtype_policy == "float16"
    assert weights.bf16_compute_patterns == ("feed_forward",)
    assert attention_ref.impl == "dense"
    assert attention_ref.weight.dtype == mx.bfloat16
    assert ffn_ref.impl == "dense_fp16_compute"
    assert ffn_ref.weight.dtype == mx.float16


def test_z_image_q8_default_uses_selected_fast_dequant_patterns(monkeypatch, tmp_path):
    pytest.importorskip("mlx.core")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    monkeypatch.delenv("COMFY_MLX_Z_IMAGE_Q8_LINEAR_IMPL", raising=False)
    monkeypatch.delenv("COMFY_MLX_Z_IMAGE_Q8_DEQUANT_PATTERNS", raising=False)

    package = tmp_path / "z_image_q8_default"
    transformer = package / "transformer"
    transformer.mkdir(parents=True)
    shard = transformer / "0.safetensors"
    dense = torch.arange(256, dtype=torch.float32).reshape(4, 64) / 10.0
    import mlx.core as mx

    q_weight, scales, biases = mx.quantize(mx.array(dense.numpy()), group_size=64, bits=8)
    tensors = {}
    weight_map = {}
    for base in ("attention.to_out", "unmatched.linear"):
        tensors[f"{base}.weight"] = torch.from_numpy(__import__("numpy").array(q_weight))
        tensors[f"{base}.scales"] = torch.from_numpy(__import__("numpy").array(scales)).to(torch.bfloat16)
        tensors[f"{base}.biases"] = torch.from_numpy(__import__("numpy").array(biases)).to(torch.bfloat16)
        weight_map[f"{base}.weight"] = "0.safetensors"
        weight_map[f"{base}.scales"] = "0.safetensors"
        weight_map[f"{base}.biases"] = "0.safetensors"
    safetensors_torch.save_file(tensors, str(shard))
    (transformer / "model.safetensors.index.json").write_text(
        __import__("json").dumps({"metadata": {"quantization_level": "8"}, "weight_map": weight_map}),
        encoding="utf-8",
    )

    weights = MLXZImageWeights(package, dtype=mx.bfloat16)
    matched = weights.linear_ref("attention.to_out")
    unmatched = weights.linear_ref("unmatched.linear")

    assert weights.linear_impl == "hybrid_dequant"
    assert weights.dequant_patterns == Q8_DEFAULT_DEQUANT_PATTERNS
    assert not matched.quantized
    assert matched.impl == "dequantized"
    assert unmatched.quantized
    assert unmatched.impl == "quantized_matmul"


def test_z_image_runtime_caches_context_conversion():
    pytest.importorskip("mlx.core")

    def fake_predictor(*, latent, mx, **_kwargs):
        return mx.zeros_like(latent)

    runtime = ZImageIslandRuntime(predictor=fake_predictor)
    x = torch.randn(1, 16, 8, 8)
    context = torch.randn(1, 4, 2560)
    call = MLXDenoiserCall(
        model_input=x,
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=context,
        control=None,
        transformer_options={},
        extra_conds={},
        cond_or_uncond=[0],
    )

    runtime.forward_model_output(call)
    runtime.forward_model_output(call)

    assert runtime.bridge.counter.torch_to_mlx == 7
    conversions = runtime.bridge.counter.as_dict()
    assert conversions["torch_to_cpu_seconds"] >= 0.0
    assert conversions["torch_to_mlx_seconds"] >= 0.0
    assert conversions["mlx_to_torch_seconds"] >= 0.0


def test_z_image_island_runtime_accepts_single_tensor_container():
    pytest.importorskip("mlx.core")

    def fake_predictor(*, latent, mx, **_kwargs):
        return mx.zeros_like(latent)

    runtime = ZImageIslandRuntime(predictor=fake_predictor)
    x = torch.randn(1, 16, 8, 8)
    call = MLXDenoiserCall(
        model_input=[x],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=x,
        context=torch.randn(1, 4, 2560),
        control=None,
        transformer_options={},
        extra_conds={},
        cond_or_uncond=[0],
    )

    out = runtime.forward_model_output(call)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mlx_denoiser_island_inpaint_scaling_keeps_clean_latent():
    model = create_mlx_denoiser_island_model()
    x = torch.randn(1, 128, 1, 2, 2)
    noise = torch.randn_like(x)
    latent_image = torch.randn_like(x)
    sigma = torch.tensor([0.9])

    scaled = model.model.scale_latent_inpaint(x=x, sigma=sigma, noise=noise, latent_image=latent_image)

    assert scaled is latent_image


def test_ltx_av_island_runtime_builds_masked_token_timesteps():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    runtime = LTXAVIslandRuntime("/tmp/nonexistent")
    mask = torch.ones(1, 1, 2, 2, 2)
    mask[:, :, 0] = 0.0

    tokens = runtime._denoise_mask_to_tokens(mask, (2, 2, 2))
    timesteps = runtime._masked_timesteps(mx.array([0.5], dtype=mx.bfloat16), tokens)
    mx.eval(tokens, timesteps)

    assert list(tokens.shape) == [1, 8, 1]
    assert list(timesteps.shape) == [1, 8]
    assert float(mx.min(tokens).item()) == 0.0
    assert float(mx.max(tokens).item()) == 1.0
    assert float(mx.min(timesteps).item()) == 0.0
    assert float(mx.max(timesteps).item()) == pytest.approx(0.5)


def test_ltx_av_shape_census_reports_mask_metadata():
    runtime = LTXAVIslandRuntime("ltx-2.3-mlx-q4", bridge=runtime_bridge_no_mlx())
    video = torch.zeros(1, 128, 2, 4, 4)
    audio = torch.zeros(1, 8, 8, 16)
    video_mask = torch.ones(1, 1, 2, 4, 4)
    audio_mask = torch.ones(1, 1, 8, 16)
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=video,
        context=torch.zeros(1, 1024, 6144),
        control=None,
        transformer_options={},
        extra_conds={"frame_rate": 24, "denoise_mask": video_mask, "audio_denoise_mask": audio_mask},
        cond_or_uncond=[0],
    )

    census = runtime.shape_census(call)

    assert census["mask_metadata"]["video"]["present"] is True
    assert census["mask_metadata"]["video"]["shape"] == [1, 1, 2, 4, 4]
    assert census["mask_metadata"]["audio"]["present"] is True
    assert census["mask_metadata"]["audio"]["shape"] == [1, 1, 8, 16]


def test_ltx_cfg1_batch_collapse_preserves_guide_attention_entries():
    runtime = _RecordingLTXRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    x = torch.randn(2, 128, 1, 2, 2)
    sigma = torch.tensor([1.0, 1.0])
    context = torch.cat([torch.zeros(1, 8, 16), torch.randn(1, 8, 16)], dim=0)
    guide_entries = [
        {"pre_filter_count": 4, "strength": 0.7, "pixel_mask": None, "latent_shape": [1, 2, 2]},
        {"pre_filter_count": 4, "strength": 0.6, "pixel_mask": None, "latent_shape": [1, 2, 2]},
    ]

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": x,
            "timestep": sigma,
            "cond_or_uncond": [1, 0],
            "cond_scale": 1.0,
            "c": {
                "c_crossattn": context,
                "guide_attention_entries": guide_entries,
                "transformer_options": {
                    "cond_or_uncond": [1, 0],
                    "sigmas": sigma,
                },
            },
        },
    )

    assert runtime.calls == 1
    assert runtime.last_call.cond_or_uncond == [0]
    assert runtime.last_call.original_input.shape[0] == 1
    assert len(runtime.last_call.extra_conds["guide_attention_entries"]) == 2
    assert runtime.last_call.extra_conds["guide_attention_entries"][0]["strength"] == 0.7
    assert runtime.last_call.extra_conds["guide_attention_entries"][1]["strength"] == 0.6
    assert out.shape == x.shape


def test_ltx_keyframe_positions_replace_tail_positions():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    video = torch.zeros(1, 128, 2, 2, 2)
    audio = torch.zeros(1, 8, 3, 16)
    keyframe_idxs = torch.tensor(
        [[[[0.0, 0.0], [24.0, 24.0]], [[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0], [14.0, 16.0]]]]
    )
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=video,
        context=torch.zeros(1, 4, 6144),
        control=None,
        transformer_options={},
        extra_conds={
            "frame_rate": 24,
            "keyframe_idxs": keyframe_idxs,
            "guide_attention_entries": [
                {"pre_filter_count": 1, "strength": 0.7, "pixel_mask": None, "latent_shape": [1, 1, 1]},
                {"pre_filter_count": 1, "strength": 0.7, "pixel_mask": None, "latent_shape": [1, 1, 1]},
            ],
        },
        cond_or_uncond=[0],
    )

    state = runtime.prepare_prompt_state(call)
    expected_tail = mx.array([[[0.0, 3.0, 11.0], [1.0, 7.0, 15.0]]], dtype=mx.float32)
    mx.eval(state.video_positions, expected_tail)

    assert mx.allclose(state.video_positions[:, -2:, :], expected_tail).item()
    assert state.guide_metadata["guide_token_count"] == 2
    assert state.guide_metadata["guide_start"] == 6
    assert state.to_report(runtime)["video_attention_mask"]["tracked_count"] == 2


def test_ltx_guide_attention_sparse_mask_matches_dense_reference():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from ltx_core_mlx.model.transformer.attention import GuideAttentionMask, _attention_with_guide_mask

    q = mx.arange(1 * 2 * 5 * 4, dtype=mx.float32).reshape(1, 2, 5, 4) / 17.0
    k = mx.arange(1 * 2 * 5 * 4, dtype=mx.float32).reshape(1, 2, 5, 4) / 19.0
    v = mx.arange(1 * 2 * 5 * 4, dtype=mx.float32).reshape(1, 2, 5, 4) / 23.0
    weights = mx.array([[0.7, 0.5]], dtype=mx.float32)
    sparse_mask = GuideAttentionMask(total_tokens=5, guide_start=3, tracked_weights=weights)
    sparse = _attention_with_guide_mask(q, k, v, scale=0.5, guide_mask=sparse_mask)

    log_weights = mx.log(weights)
    dense_noisy = mx.concatenate(
        [mx.zeros((1, 1, 3, 3), dtype=mx.float32), mx.broadcast_to(log_weights[:, None, None, :], (1, 1, 3, 2))],
        axis=-1,
    )
    dense_tracked = mx.concatenate(
        [
            mx.broadcast_to(log_weights[:, None, :, None], (1, 1, 2, 3)),
            mx.zeros((1, 1, 2, 2), dtype=mx.float32),
        ],
        axis=-1,
    )
    dense_mask = mx.concatenate([dense_noisy, dense_tracked], axis=2)
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5, mask=dense_mask)

    assert mx.allclose(sparse, dense, atol=1.0e-5, rtol=1.0e-5).item()


def test_ltx_guide_attention_rejects_pixel_masks():
    pytest.importorskip("mlx.core")

    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    with pytest.raises(UnsupportedIslandRequest, match="pixel masks"):
        runtime._build_guide_attention_mask(
            [{"pre_filter_count": 1, "strength": 0.7, "pixel_mask": torch.ones(1, 1)}],
            total_tokens=4,
            guide_token_count=1,
            batch_size=1,
        )


def test_ltx_keyframe_guides_reject_negative_denoise_masks():
    pytest.importorskip("mlx.core")

    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    video = torch.zeros(1, 128, 1, 2, 2)
    audio = torch.zeros(1, 8, 3, 16)
    video_mask = torch.ones(1, 1, 1, 2, 2)
    video_mask[:, :, :, -1, -1] = -1.0
    keyframe_idxs = torch.tensor([[[[0.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]]])
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=video,
        context=torch.zeros(1, 4, 6144),
        control=None,
        transformer_options={},
        extra_conds={
            "frame_rate": 24,
            "denoise_mask": video_mask,
            "keyframe_idxs": keyframe_idxs,
            "guide_attention_entries": [
                {"pre_filter_count": 1, "strength": 0.7, "pixel_mask": None, "latent_shape": [1, 1, 1]},
            ],
        },
        cond_or_uncond=[0],
    )

    with pytest.raises(UnsupportedIslandRequest, match="negative/dilated"):
        runtime.prepare_prompt_state(call)


def test_ltx_av_prompt_state_is_prompt_local_not_context_cached():
    pytest.importorskip("mlx.core")

    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    video = torch.zeros(1, 128, 1, 2, 2)
    audio = torch.zeros(1, 8, 3, 16)
    context_a = torch.zeros(1, 4, 6144)
    context_b = torch.ones(1, 4, 6144)

    def make_call(context):
        return MLXDenoiserCall(
            model_input=[video, audio],
            sigma=torch.tensor([0.5]),
            timestep=torch.tensor([0.5]),
            original_input=video,
            context=context,
            control=None,
            transformer_options={},
            extra_conds={"frame_rate": 24},
            cond_or_uncond=[0],
        )

    state_a = runtime.prepare_prompt_state(make_call(context_a))
    state_b = runtime.prepare_prompt_state(make_call(context_b))

    assert state_a.shape_bucket() == [1, 2, 2, 3, 24.0]
    assert state_b.shape_bucket() == [1, 2, 2, 3, 24.0]
    assert runtime.caches.context == {}
    assert state_a.to_report(runtime)["global_prompt_cache"] is False


def test_ltx_av_latent_split_accepts_list_and_rejects_tensor_without_audio():
    video = torch.randn(1, 128, 1, 2, 2)
    audio = torch.randn(1, 8, 3, 16)

    split_video, split_audio = split_ltx_av_latents([video, audio])

    assert split_video is video
    assert split_audio is audio

    tensor_video, tensor_audio = split_ltx_av_latents(video)

    assert tensor_video is video
    assert tensor_audio.shape == (1, 8, 0, 16)


class _ListOutputRuntime(BaseMLXDenoiserIslandRuntime):
    name = "list_output"

    def forward_model_output(self, call: MLXDenoiserCall):
        video, audio = call.model_input
        self.calls += 1
        return [torch.ones_like(video), torch.ones_like(audio)]


def test_mlx_denoiser_island_wrapper_packs_list_model_output_before_sampler_math():
    runtime = _ListOutputRuntime()
    model = create_mlx_denoiser_island_model(runtime=runtime)
    wrapper = _wrapper(model)
    video = torch.randn(1, 128, 1, 2, 2)
    audio = torch.randn(1, 8, 3, 16)
    packed, latent_shapes = comfy.utils.pack_latents([video, audio])
    sigma = torch.tensor([0.5])
    context = torch.randn(1, 4, 6144)

    out = wrapper(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected Torch apply_model call")),
        {
            "input": packed,
            "timestep": sigma,
            "cond_or_uncond": [0],
            "c": {
                "c_crossattn": context,
                "latent_shapes": latent_shapes,
                "transformer_options": {
                    "cond_or_uncond": [0],
                    "sigmas": sigma,
                },
            },
        },
    )
    expected_model_output, _ = comfy.utils.pack_latents([torch.ones_like(video), torch.ones_like(audio)])
    expected = model.model.model_sampling.calculate_denoised(sigma, expected_model_output, packed)

    assert runtime.calls == 1
    assert torch.equal(out, expected)
    assert wrapper.last_event["model_input_shape"] == [list(video.shape), list(audio.shape)]


def test_ltx_av_runtime_rejects_wrong_combined_context_dim():
    pytest.importorskip("mlx.core")
    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    video = torch.randn(1, 128, 1, 2, 2)
    audio = torch.randn(1, 8, 3, 16)
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=torch.zeros(1, 1, video.numel() + audio.numel()),
        context=torch.randn(1, 4, 128),
        control=None,
        transformer_options={},
        extra_conds={},
        cond_or_uncond=[0],
    )

    with pytest.raises(UnsupportedIslandRequest, match="combined LTX AV context dim"):
        runtime.forward_model_output(call)


def test_ltx_av_runtime_token_context_plumbing_with_fake_x0_model():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    class FakeX0:
        def __call__(self, *, video_latent, audio_latent, sigma, **_kwargs):
            sigma = sigma.astype(mx.float32)[:, None, None]
            return (
                video_latent.astype(mx.float32) - sigma,
                audio_latent.astype(mx.float32) - sigma,
            )

    runtime = LTXAVIslandRuntime("/does/not/load/in/this/test")
    runtime._x0_model = lambda *_shape: FakeX0()
    video = torch.randn(1, 128, 1, 2, 2)
    audio = torch.randn(1, 8, 3, 16)
    call = MLXDenoiserCall(
        model_input=[video, audio],
        sigma=torch.tensor([0.5]),
        timestep=torch.tensor([0.5]),
        original_input=torch.zeros(1, 1, video.numel() + audio.numel()),
        context=torch.randn(1, 4, 6144),
        control=None,
        transformer_options={},
        extra_conds={"frame_rate": torch.tensor([24.0])},
        cond_or_uncond=[0],
    )

    video_out, audio_out = runtime.forward_model_output(call)

    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    assert torch.allclose(video_out.float(), torch.ones_like(video).float(), atol=1e-3, rtol=1e-3)
    assert torch.allclose(audio_out.float(), torch.ones_like(audio).float(), atol=1e-3, rtol=1e-3)
    assert runtime.last_event["shape_bucket"] == [1, 2, 2, 3, 24.0]


def test_distilled_upscaler_helper_preserves_expected_shape_with_fake_modules():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from ltx_pipelines_mlx.distilled import DistilledPipeline

    class FakeVAEEncoder:
        def denormalize_latent(self, value):
            return value

        def normalize_latent(self, value):
            return value

    class FakeUpsampler:
        def __call__(self, value):
            return mx.repeat(mx.repeat(value, 2, axis=3), 2, axis=4)

    pipe = DistilledPipeline.__new__(DistilledPipeline)
    pipe.load_upscaler_only = lambda: None
    pipe.image_conditioner = type("FakeImageConditioner", (), {})()
    pipe.vae_encoder = FakeVAEEncoder()
    pipe.upsampler = FakeUpsampler()

    video = mx.zeros((1, 128, 2, 3, 5), dtype=mx.bfloat16)
    out = pipe.upscale_distilled_video_latent(video)

    assert tuple(out.shape) == (1, 128, 2, 6, 10)
