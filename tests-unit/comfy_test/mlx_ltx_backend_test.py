import os
import json
import sys
import types
from pathlib import Path

import pytest

import comfy.backends
from comfy.backends.mlx_denoiser_island import is_mlx_denoiser_island_model, remap_ltx_comfy_transformer_key
from comfy.backends.mlx_ltx_backend import (
    MLXLTXClipProxy,
    MLXLTXBackend,
    MLX_LTX_LATENT_UPSCALER_ALIASES,
    MLXLTXRunConfig,
    MLXLTXVAEProxy,
    extract_mlx_ltx_frame_rate,
    extract_mlx_ltx_prompt,
    infer_ltx_dimensions,
    is_mlx_ltx_audio_proxy,
    is_mlx_ltx_image_proxy,
    is_mlx_ltx_manifest_path,
    is_mlx_ltx_model,
    is_mlx_ltx_vae,
    load_mlx_ltx_checkpoint,
    load_mlx_ltx_manifest,
    ltx_duration_seconds,
    make_model_spec,
    make_mlx_ltx_audio_placeholder,
    validate_ltx_frame_count,
)


def test_builtin_mlx_ltx_backend_registration_does_not_import_optional_ltx_modules():
    sys.modules.pop("ltx_pipelines_mlx", None)
    sys.modules.pop("ltx_pipelines_mlx.distilled", None)

    backends = {backend.name for backend in comfy.backends.list_backends()}

    assert "mlx_ltx" in backends
    assert "ltx_pipelines_mlx" not in sys.modules
    assert "ltx_pipelines_mlx.distilled" not in sys.modules


def test_mlx_ltx_non_apple_platform_reports_unavailable(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    status = MLXLTXBackend().capabilities()

    assert status.available is False
    assert "Apple Silicon" in status.reason


def test_mlx_ltx_frame_count_validation():
    assert validate_ltx_frame_count(49) == 49
    assert ltx_duration_seconds(49, 24) == pytest.approx(49 / 24)
    with pytest.raises(ValueError, match="8k \\+ 1"):
        validate_ltx_frame_count(50)


def test_mlx_ltx_make_model_spec_normalizes_snapshot_controls():
    spec = make_model_spec(
        repo_id=" dgrauet/ltx-2.3-mlx-q4 ",
        revision=" main ",
        variant="q4",
        pipeline="distilled",
        allow_patterns="*.json,*.safetensors",
        ignore_patterns="*.md",
        local_files_only=True,
        use_hf_token=True,
    )

    assert spec.repo_id == "dgrauet/ltx-2.3-mlx-q4"
    assert spec.revision == "main"
    assert spec.variant == "q4"
    assert spec.pipeline == "distilled"
    assert spec.allow_patterns == ("*.json", "*.safetensors")
    assert spec.ignore_patterns == ("*.md",)
    assert spec.local_files_only is True
    assert spec.use_hf_token is True


def test_ltx_comfy_dev_and_lora_keys_remap_to_mlx_transformer_contract():
    assert (
        remap_ltx_comfy_transformer_key("model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight")
        == "transformer_blocks.0.attn1.to_out.weight"
    )
    assert (
        remap_ltx_comfy_transformer_key("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias")
        == "transformer_blocks.0.ff.proj_in.bias"
    )
    assert (
        remap_ltx_comfy_transformer_key("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight")
        == "adaln_single.emb.timestep_embedder.linear1.weight"
    )
    assert (
        remap_ltx_comfy_transformer_key("diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_B.weight")
        == "transformer_blocks.0.audio_ff.proj_out.lora_B.weight"
    )
    assert remap_ltx_comfy_transformer_key("vae.decoder.weight") is None


def test_ltx_latent_upscaler_alias_uses_available_dev_lora_manifest():
    assert (
        MLX_LTX_LATENT_UPSCALER_ALIASES["LTX-2.3 x2 Spatial Upscaler v1.1"]
        == "ltx23_dev_lora_mlx_bf16_native.mlx_ltx.json"
    )


def test_mlx_ltx_resolve_local_model_path_does_not_require_huggingface(tmp_path):
    spec = make_model_spec(repo_id=str(tmp_path))

    resolved = MLXLTXBackend().resolve_model_path(spec)

    assert resolved == os.path.abspath(tmp_path)


def test_mlx_ltx_resolve_hf_snapshot_uses_spec_controls(monkeypatch):
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/mlx-ltx-cache/models--ltx"

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    spec = make_model_spec(
        repo_id="dgrauet/ltx-2.3-mlx-q4",
        revision="deadbeef",
        allow_patterns=["*.json"],
        ignore_patterns=["*.md"],
        local_files_only=True,
        use_hf_token=True,
    )

    resolved = MLXLTXBackend().resolve_model_path(spec, cache_dir="/tmp/mlx-ltx-cache")

    assert resolved == "/tmp/mlx-ltx-cache/models--ltx"
    assert calls == {
        "repo_id": "dgrauet/ltx-2.3-mlx-q4",
        "revision": "deadbeef",
        "cache_dir": "/tmp/mlx-ltx-cache",
        "local_files_only": True,
        "allow_patterns": ("*.json",),
        "ignore_patterns": ("*.md",),
        "token": True,
    }


def test_mlx_ltx_defaults_low_ram_for_q8_and_bf16():
    backend = MLXLTXBackend()

    backend._unified_memory_gb = lambda: 32
    assert backend._default_low_ram(make_model_spec(repo_id="repo", variant="q4")) is False
    assert backend._default_low_ram(make_model_spec(repo_id="repo", variant="q8")) is True
    assert backend._default_low_ram(make_model_spec(repo_id="repo", variant="bf16")) is True
    backend._unified_memory_gb = lambda: 64
    assert backend._default_low_ram(make_model_spec(repo_id="repo", variant="q8")) is False
    assert backend._default_low_memory(make_model_spec(repo_id="repo", variant="q8")) is False


def test_mlx_ltx_manifest_parsing_and_proxy_checkpoint(tmp_path):
    manifest = tmp_path / "ltx.mlx_ltx.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "dgrauet/ltx-2.3-mlx-q8",
                "execution_mode": "legacy_media",
                "revision": "main",
                "variant": "q8",
                "pipeline": "distilled",
                "local_files_only": True,
                "low_memory": False,
                "low_ram_streaming": False,
                "stage1_steps": 8,
                "stage2_steps": 3,
                "frame_rate": 24,
                "gemma_model_id": "mlx-community/gemma-test",
                "media_passthrough": True,
                "prompt_cache_size": 4,
                "stage_snapshot_cache": False,
                "mlx_cache_limit_gb": 2.5,
                "profile_level": "block_group",
                "metal_capture": False,
                "block_profile_interval": 6,
                "eval_every": 6,
                "stage1_eval_every": 0,
                "stage2_eval_every": 8,
                "decode_profile": True,
                "compile_dit": True,
                "stage1_compile_dit": False,
                "stage2_compile_dit": True,
                "compile_block_stack": True,
                "compile_x0": True,
                "compile_shapeless": False,
                "flatten_attention_projections": True,
                "dequantize_video_ffn": True,
                "native_metal_kernels": True,
                "native_kernel_profile": True,
                "native_kernel_verify": False,
                "native_kernel_fallback": True,
                "native_kernel_set": "norm",
            }
        ),
        encoding="utf-8",
    )

    spec, run_config = load_mlx_ltx_manifest(manifest)
    model, clip, vae = load_mlx_ltx_checkpoint(manifest)

    assert is_mlx_ltx_manifest_path(manifest)
    assert spec.repo_id == "dgrauet/ltx-2.3-mlx-q8"
    assert spec.variant == "q8"
    assert spec.gemma_model_id == "mlx-community/gemma-test"
    assert run_config.low_memory is False
    assert run_config.low_ram is False
    assert run_config.media_passthrough is True
    assert run_config.prompt_cache_size == 4
    assert run_config.stage_snapshot_cache is False
    assert run_config.mlx_cache_limit_gb == 2.5
    assert run_config.profile_level == "block_group"
    assert run_config.metal_capture is False
    assert run_config.block_profile_interval == 6
    assert run_config.eval_every == 6
    assert run_config.stage1_eval_every == 0
    assert run_config.stage2_eval_every == 8
    assert run_config.decode_profile is True
    assert run_config.compile_dit is True
    assert run_config.stage1_compile_dit is False
    assert run_config.stage2_compile_dit is True
    assert run_config.compile_block_stack is True
    assert run_config.compile_x0 is True
    assert run_config.compile_shapeless is False
    assert run_config.flatten_attention_projections is True
    assert run_config.dequantize_video_ffn is True
    assert run_config.native_metal_kernels is True
    assert run_config.native_kernel_profile is True
    assert run_config.native_kernel_verify is False
    assert run_config.native_kernel_fallback is True
    assert run_config.native_kernel_set == "norm"
    assert spec.metadata["media_passthrough"] is True
    assert spec.metadata["prompt_cache_size"] == 4
    assert spec.metadata["profile_level"] == "block_group"
    assert spec.metadata["stage1_eval_every"] == 0
    assert spec.metadata["stage2_eval_every"] == 8
    assert spec.metadata["compile_dit"] is True
    assert spec.metadata["stage1_compile_dit"] is False
    assert spec.metadata["stage2_compile_dit"] is True
    assert spec.metadata["compile_block_stack"] is True
    assert spec.metadata["compile_x0"] is True
    assert spec.metadata["flatten_attention_projections"] is True
    assert spec.metadata["dequantize_video_ffn"] is True
    assert spec.metadata["native_metal_kernels"] is True
    assert spec.metadata["native_kernel_set"] == "norm"
    assert spec.metadata["execution_mode"] == "legacy_media"
    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("hello"))
    assert extract_mlx_ltx_prompt(conditioning) == "hello"


def test_mlx_ltx_dev_lora_manifest_parses_internal_lora_fields(tmp_path):
    (tmp_path / "dev.safetensors").write_text("dev", encoding="utf-8")
    (tmp_path / "lora.safetensors").write_text("lora", encoding="utf-8")
    manifest = tmp_path / "ltx-dev-lora.mlx_ltx.json"
    manifest.write_text(
        json.dumps(
            {
                "local_path": str(tmp_path),
                "variant": "bf16",
                "pipeline": "dev_lora",
                "execution_mode": "denoiser_island",
                "dev_checkpoint_path": "dev.safetensors",
                "distilled_lora_path": "lora.safetensors",
                "distilled_lora_strength": 0.5,
                "low_ram_streaming": True,
                "low_memory": True,
            }
        ),
        encoding="utf-8",
    )

    spec, run_config = load_mlx_ltx_manifest(manifest)

    assert spec.pipeline == "dev_lora"
    assert run_config.pipeline == "dev_lora"
    assert spec.metadata["dev_checkpoint_path"] == str((tmp_path / "dev.safetensors").resolve())
    assert spec.metadata["distilled_lora_path"] == str((tmp_path / "lora.safetensors").resolve())
    assert spec.metadata["distilled_lora_strength"] == 0.5
    assert run_config.low_ram is True


def test_mlx_ltx_clip_proxy_skips_empty_negative_prompt_without_gemma(monkeypatch, tmp_path):
    def fail_encode(*args, **kwargs):
        raise AssertionError("empty negative prompt should not run Gemma encode")

    clip = MLXLTXClipProxy(
        make_model_spec(repo_id="repo", variant="q4"),
        MLXLTXRunConfig(prompt="positive prompt", output_path=str(tmp_path / "out.mp4")),
    )
    monkeypatch.setattr(clip, "_encode_prompt_context", fail_encode)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    context, metadata = conditioning[0]
    assert tuple(context.shape) == (1, 1024, 6144)
    assert metadata["mlx_ltx_context_skipped"] == "negative_prompt_cfg1_unused_zero_context"


def test_mlx_ltx_folder_loader_returns_denoiser_island_without_torch_transformer(tmp_path):
    model_dir = tmp_path / "ltx-2.3-mlx-q4"
    model_dir.mkdir()
    for name in ("config.json", "split_model.json", "transformer-distilled-1.1.safetensors", "connector.safetensors"):
        (model_dir / name).write_text("{}", encoding="utf-8")

    model, clip, vae = load_mlx_ltx_checkpoint(model_dir)

    assert is_mlx_denoiser_island_model(model.model)
    assert model.model_patches_models() == []
    assert model.model_options["model_function_wrapper"].models() == []
    assert not is_mlx_ltx_model(model)
    assert isinstance(clip, MLXLTXClipProxy)
    assert is_mlx_ltx_vae(vae)


def test_mlx_ltx_clip_proxy_returns_real_context_shape(monkeypatch, tmp_path):
    import torch

    model_dir = tmp_path / "ltx-2.3-mlx-q4"
    model_dir.mkdir()
    for name in ("config.json", "split_model.json", "transformer-distilled-1.1.safetensors", "connector.safetensors"):
        (model_dir / name).write_text("{}", encoding="utf-8")

    _model, clip, _vae = load_mlx_ltx_checkpoint(model_dir)
    monkeypatch.setattr(
        MLXLTXClipProxy,
        "_encode_prompt_context",
        lambda self, prompt: (
            torch.zeros(1, 1024, 6144),
            {
                "mlx_ltx_context_video_shape": [1, 1024, 4096],
                "mlx_ltx_context_audio_shape": [1, 1024, 2048],
                "mlx_ltx_context_shape": [1, 1024, 6144],
            },
        ),
    )

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("native prompt"))

    assert conditioning[0][0].shape == (1, 1024, 6144)
    assert conditioning[0][1]["mlx_ltx_prompt"] == "native prompt"
    assert conditioning[0][1]["mlx_ltx_context_video_shape"] == [1, 1024, 4096]


def test_mlx_ltx_clip_proxy_skips_obvious_negative_prompt_context(monkeypatch, tmp_path):
    model_dir = tmp_path / "ltx-2.3-mlx-q4"
    model_dir.mkdir()
    for name in ("config.json", "split_model.json", "transformer-distilled-1.1.safetensors", "connector.safetensors"):
        (model_dir / name).write_text("{}", encoding="utf-8")

    _model, clip, _vae = load_mlx_ltx_checkpoint(model_dir)

    def fail_encode(_self, prompt):
        raise AssertionError(f"negative prompt should not encode Gemma context: {prompt}")

    monkeypatch.setattr(MLXLTXClipProxy, "_encode_prompt_context", fail_encode)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("black frame, white frame, silent audio, broken video"))

    assert conditioning[0][0].shape == (1, 1024, 6144)
    assert conditioning[0][1]["conditioning_mode"] == "metadata_only_negative"
    assert conditioning[0][1]["mlx_ltx_context_skipped"] == "negative_prompt_cfg1_unused_zero_context"
    assert extract_mlx_ltx_prompt(conditioning) == "black frame, white frame, silent audio, broken video"


def test_mlx_ltx_generate_video_uses_distilled_pipeline(monkeypatch, tmp_path):
    calls = {}
    init_calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            init_calls.append(kwargs)

        def generate_and_save(self, **kwargs):
            calls["generate"] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"fake mp4")
            return kwargs["output_path"]

    backend = MLXLTXBackend()
    monkeypatch.setattr(backend, "_import_mlx_core", lambda: types.SimpleNamespace(default_device=lambda: "gpu"))
    monkeypatch.setattr(backend, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(backend, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")
    spec = make_model_spec(repo_id="dgrauet/ltx-2.3-mlx-q4", variant="q4")
    output = tmp_path / "out.mp4"

    result = backend.generate_video_to_file(
        spec,
        "prompt",
        str(output),
        width=512,
        height=512,
        num_frames=49,
        frame_rate=24,
        stage1_steps=8,
        stage2_steps=3,
        seed=123,
    )

    assert output.exists()
    assert result.output_path == str(output.resolve())
    assert calls["init"]["model_dir"] == "/models/ltx"
    assert calls["init"]["low_memory"] is False
    assert calls["init"]["low_ram_streaming"] is False
    assert calls["init"]["gemma_model_id"] == "mlx-community/gemma-3-12b-it-4bit"
    assert calls["generate"]["prompt"] == "prompt"
    assert calls["generate"]["num_frames"] == 49
    assert calls["generate"]["seed"] == 123

    backend.generate_video_to_file(spec, "prompt", str(tmp_path / "out2.mp4"), width=512, height=512, num_frames=49)

    assert len(init_calls) == 1


def test_mlx_ltx_prompt_cache_reuses_evaluated_embeddings(monkeypatch, tmp_path):
    calls = {"encode": 0, "load_text_encoder": 0}

    class FakePipeline:
        def __init__(self, **kwargs):
            self.low_memory = False

        def _load_text_encoder(self):
            calls["load_text_encoder"] += 1

        def _encode_text(self, prompt):
            calls["encode"] += 1
            return (f"video:{prompt}", f"audio:{prompt}")

        def generate_two_stage(self, prompt, **kwargs):
            self._load_text_encoder()
            self._encode_text(prompt)
            return "video_latent", "audio_latent"

        def _decode_and_save_video(self, video_latent, audio_latent, output_path, *, frame_rate):
            Path(output_path).write_bytes(b"fake mp4")
            return output_path

    stats_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("COMFY_BENCH_STATS_PATH", str(stats_path))
    backend = MLXLTXBackend()
    monkeypatch.setattr(backend, "_import_mlx_core", lambda: types.SimpleNamespace(default_device=lambda: "gpu", eval=lambda *args: None))
    monkeypatch.setattr(backend, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(backend, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")
    spec = make_model_spec(repo_id="dgrauet/ltx-2.3-mlx-q4", variant="q4", prompt_cache_size=2)

    backend.generate_video_to_file(spec, "same prompt", str(tmp_path / "one.mp4"), width=512, height=512, num_frames=49)
    backend.generate_video_to_file(spec, "same prompt", str(tmp_path / "two.mp4"), width=512, height=512, num_frames=49)

    assert calls["encode"] == 1
    assert calls["load_text_encoder"] == 1
    events = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines()]
    prompt_events = [event for event in events if event.get("stage") == "prompt_encode"]
    assert [event["model"]["prompt_cache_hit"] for event in prompt_events] == [False, True]


def test_mlx_ltx_empty_cache_is_explicit_only(monkeypatch, tmp_path):
    calls = {"clear": 0}
    fake_mx = types.SimpleNamespace(default_device=lambda: "gpu", clear_cache=lambda: calls.__setitem__("clear", calls["clear"] + 1))

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def generate_and_save(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"fake mp4")
            return kwargs["output_path"]

    backend = MLXLTXBackend()
    monkeypatch.setattr(backend, "_import_mlx_core", lambda: fake_mx)
    monkeypatch.setattr(backend, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(backend, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")

    backend.generate_video_to_file(make_model_spec(repo_id="repo"), "prompt", str(tmp_path / "out.mp4"))
    assert calls["clear"] == 0

    backend.empty_cache()
    assert calls["clear"] == 1


def test_mlx_ltx_metal_capture_and_profile_metadata(monkeypatch, tmp_path):
    calls = {"start": 0, "stop": 0}

    class FakeMetal:
        @staticmethod
        def start_capture():
            calls["start"] += 1
            return "/tmp/capture.gputrace"

        @staticmethod
        def stop_capture():
            calls["stop"] += 1
            return "/tmp/capture.gputrace"

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def generate_and_save(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"fake mp4")
            return kwargs["output_path"]

    stats_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("COMFY_BENCH_STATS_PATH", str(stats_path))
    backend = MLXLTXBackend()
    monkeypatch.setattr(
        backend,
        "_import_mlx_core",
        lambda: types.SimpleNamespace(default_device=lambda: "gpu", metal=FakeMetal),
    )
    monkeypatch.setattr(backend, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(backend, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")
    spec = make_model_spec(
        repo_id="repo",
        profile_level="block_group",
        metal_capture=True,
        block_profile_interval=4,
        eval_every=4,
        stage1_eval_every=0,
        stage2_eval_every=8,
        decode_profile=True,
        compile_dit=True,
        stage1_compile_dit=False,
        stage2_compile_dit=True,
        compile_block_stack=True,
        compile_x0=True,
    )

    result = backend.generate_video_to_file(spec, "prompt", str(tmp_path / "out.mp4"))

    assert Path(result.output_path).exists()
    assert calls == {"start": 1, "stop": 1}
    events = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines()]
    generate_start = next(event for event in events if event.get("event_type") == "generate_start")
    assert generate_start["model"]["profile_level"] == "block_group"
    assert generate_start["model"]["eval_every"] == 4
    assert generate_start["model"]["stage1_eval_every"] == 0
    assert generate_start["model"]["stage2_eval_every"] == 8
    assert generate_start["model"]["compile_dit"] is True
    assert generate_start["model"]["stage1_compile_dit"] is False
    assert generate_start["model"]["stage2_compile_dit"] is True
    assert generate_start["model"]["compile_block_stack"] is True
    assert generate_start["model"]["compile_x0"] is True
    capture_events = [event for event in events if event.get("event_type") == "mlx_ltx_metal_capture"]
    assert [event["action"] for event in capture_events] == ["start", "stop"]


def test_mlx_ltx_profile_callback_does_not_attach_none_eval_every():
    backend = MLXLTXBackend()
    target = types.SimpleNamespace(_mlx_ltx_eval_every=4)
    config = MLXLTXRunConfig(prompt="prompt", output_path="/tmp/out.mp4")

    backend._set_profile_attrs(target, config, lambda **kwargs: None, stage="stage1_denoise")

    assert not hasattr(target, "_mlx_ltx_eval_every")


def test_mlx_ltx_native_kernel_flags_are_lazy_and_attached(monkeypatch, tmp_path):
    import comfy.backends.mlx_ltx_backend as mlx_ltx_backend

    calls = {"record": 0, "verify": 0, "install": 0}

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def record_status(self, model_metadata):
            calls["record"] += 1

        def verify_microbench(self):
            calls["verify"] += 1

        def patch_ltx_modules(self):
            class Context:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Context()

        def metadata(self):
            return {
                "native_kernel_available": True,
                "native_kernel_reason": "fake",
                "native_kernel_installed_targets": calls["install"],
                "native_kernel_used_count": 0,
                "native_kernel_fallback_count": 0,
                "native_kernel_verify_count": calls["verify"],
                "native_kernel_verify_seconds": 0.0,
                "native_kernel_names": ["norm"],
            }

        def install_on(self, target):
            calls["install"] += 1
            setattr(target, "_fake_native_kernel_runtime", True)

    made = {}

    def fake_make_runtime(**kwargs):
        made.update(kwargs)
        return FakeRuntime(**kwargs)

    class FakePipeline:
        def __init__(self, **kwargs):
            self.dit = types.SimpleNamespace()

        def _encode_text(self, prompt):
            return ("video_embeds", "audio_embeds")

        def generate_two_stage(self, prompt, **kwargs):
            self._encode_text(prompt)
            return "video_latent", "audio_latent"

        def _decode_and_save_video(self, video_latent, audio_latent, output_path, *, frame_rate):
            Path(output_path).write_bytes(b"fake mp4")
            return output_path

    stats_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("COMFY_BENCH_STATS_PATH", str(stats_path))
    monkeypatch.setattr(mlx_ltx_backend, "make_native_kernel_runtime", fake_make_runtime)
    backend = MLXLTXBackend()
    monkeypatch.setattr(backend, "_import_mlx_core", lambda: types.SimpleNamespace(default_device=lambda: "gpu", eval=lambda *args: None))
    monkeypatch.setattr(backend, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(backend, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")
    spec = make_model_spec(
        repo_id="repo",
        native_metal_kernels=True,
        native_kernel_profile=True,
        native_kernel_verify=True,
        native_kernel_fallback=True,
        native_kernel_set="norm",
    )

    backend.generate_video_to_file(spec, "prompt", str(tmp_path / "out.mp4"))

    assert made == {
        "native_metal_kernels": True,
        "native_kernel_set": "norm",
        "native_kernel_profile": True,
        "native_kernel_verify": True,
        "native_kernel_fallback": True,
    }
    assert calls["record"] == 1
    assert calls["verify"] == 1
    assert calls["install"] >= 2
    events = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines()]
    generate_start = next(event for event in events if event.get("event_type") == "generate_start")
    assert generate_start["model"]["native_metal_kernels"] is True
    assert generate_start["model"]["native_kernel_set"] == "norm"
    assert generate_start["model"]["native_kernel_available"] is True


def test_mlx_ltx_native_kernel_runtime_reports_missing_mlx(monkeypatch):
    import comfy.backends.mlx_ltx_native_kernels as native_kernels

    real_import = native_kernels.importlib.import_module

    def fake_import(name):
        if name == "mlx.core":
            raise ImportError("no mlx")
        return real_import(name)

    monkeypatch.setattr(native_kernels.importlib, "import_module", fake_import)
    runtime = native_kernels.make_native_kernel_runtime(
        native_metal_kernels=True,
        native_kernel_set="norm",
        native_kernel_profile=False,
        native_kernel_verify=True,
        native_kernel_fallback=True,
    )

    availability = runtime.availability()

    assert availability.available is False
    assert "no mlx" in availability.reason
    assert runtime.metadata()["native_kernel_available"] is False


def test_mlx_ltx_native_kernel_runtime_disables_unsafe_rope():
    import comfy.backends.mlx_ltx_native_kernels as native_kernels

    runtime = native_kernels.make_native_kernel_runtime(
        native_metal_kernels=True,
        native_kernel_set="rope",
        native_kernel_profile=False,
        native_kernel_verify=True,
        native_kernel_fallback=True,
    )

    availability = runtime.availability()
    runtime.verify_microbench()

    assert availability.available is False
    assert "GPU-recovery" in availability.reason
    assert runtime.metadata()["native_kernel_verify_count"] == 0


def test_mlx_ltx_native_kernel_runtime_patches_ltx_modules(monkeypatch):
    import comfy.backends.mlx_ltx_native_kernels as native_kernels

    fake_rope = types.ModuleType("ltx_core_mlx.model.transformer.rope")
    fake_attention = types.ModuleType("ltx_core_mlx.model.transformer.attention")
    fake_feed_forward = types.ModuleType("ltx_core_mlx.model.transformer.feed_forward")

    def original_rope(x, cos, sin):
        return "original_rope"

    fake_rope.apply_rope_split = original_rope
    fake_attention.apply_rope_split = original_rope

    class FakeFeedForward:
        def __init__(self):
            self.proj_in = lambda x: f"proj_in:{x}"
            self.proj_out = lambda x: f"proj_out:{x}"

        def __call__(self, x):
            return "original_ffn"

    fake_feed_forward.FeedForward = FakeFeedForward
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.model.transformer.rope", fake_rope)
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.model.transformer.attention", fake_attention)
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.model.transformer.feed_forward", fake_feed_forward)

    fake_kernels = types.ModuleType("comfy.backends.mlx_ltx_native.kernels")
    fake_kernels.rope_split = lambda x, cos, sin: "native_rope"
    fake_kernels.gelu_approx = lambda x: f"gelu:{x}"
    monkeypatch.setitem(sys.modules, "comfy.backends.mlx_ltx_native.kernels", fake_kernels)

    runtime = native_kernels.make_native_kernel_runtime(
        native_metal_kernels=True,
        native_kernel_set="rope",
        native_kernel_profile=False,
        native_kernel_verify=True,
        native_kernel_fallback=True,
    )
    monkeypatch.setattr(runtime, "availability", lambda: native_kernels.NativeKernelAvailability(True, "available", "test"))

    with runtime.patch_ltx_modules():
        assert fake_rope.apply_rope_split(None, None, None) == "native_rope"
        assert fake_attention.apply_rope_split(None, None, None) == "native_rope"
        assert FakeFeedForward()("x") == "original_ffn"

    assert fake_rope.apply_rope_split is original_rope
    assert fake_attention.apply_rope_split is original_rope
    assert FakeFeedForward()("x") == "original_ffn"
    assert runtime.metadata()["native_kernel_used_count"] == 2


def test_mlx_ltx_native_kernel_runtime_patches_ffn(monkeypatch):
    import comfy.backends.mlx_ltx_native_kernels as native_kernels

    fake_feed_forward = types.ModuleType("ltx_core_mlx.model.transformer.feed_forward")

    class FakeFeedForward:
        def __init__(self):
            self.proj_in = lambda x: f"proj_in:{x}"
            self.proj_out = lambda x: f"proj_out:{x}"

        def __call__(self, x):
            return "original_ffn"

    fake_feed_forward.FeedForward = FakeFeedForward
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.model.transformer.feed_forward", fake_feed_forward)

    fake_kernels = types.ModuleType("comfy.backends.mlx_ltx_native.kernels")
    fake_kernels.gelu_approx = lambda x: f"gelu:{x}"
    monkeypatch.setitem(sys.modules, "comfy.backends.mlx_ltx_native.kernels", fake_kernels)

    runtime = native_kernels.make_native_kernel_runtime(
        native_metal_kernels=True,
        native_kernel_set="ffn_elementwise",
        native_kernel_profile=False,
        native_kernel_verify=True,
        native_kernel_fallback=True,
    )
    monkeypatch.setattr(runtime, "availability", lambda: native_kernels.NativeKernelAvailability(True, "available", "test"))

    with runtime.patch_ltx_modules():
        assert FakeFeedForward()("x") == "proj_out:gelu:proj_in:x"

    assert FakeFeedForward()("x") == "original_ffn"
    assert runtime.metadata()["native_kernel_used_count"] == 1


def test_mlx_ltx_native_kernel_runtime_patches_norm_helpers(monkeypatch):
    import comfy.backends.mlx_ltx_native_kernels as native_kernels

    fake_transformer = types.ModuleType("ltx_core_mlx.model.transformer.transformer")

    class FakeBlock:
        def _scale_shift(self, x, scale, shift):
            return f"original_scale_shift:{x}:{scale}:{shift}"

        def _gated_residual(self, residual, value, gate):
            return f"original_gated_residual:{residual}:{value}:{gate}"

    fake_transformer.BasicAVTransformerBlock = FakeBlock
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.model.transformer.transformer", fake_transformer)

    fake_kernels = types.ModuleType("comfy.backends.mlx_ltx_native.kernels")
    fake_kernels.scale_shift = lambda x, scale, shift: f"native_scale_shift:{x}:{scale}:{shift}"
    fake_kernels.gated_residual = lambda residual, value, gate: f"native_gated_residual:{residual}:{value}:{gate}"
    monkeypatch.setitem(sys.modules, "comfy.backends.mlx_ltx_native.kernels", fake_kernels)

    runtime = native_kernels.make_native_kernel_runtime(
        native_metal_kernels=True,
        native_kernel_set="norm",
        native_kernel_profile=False,
        native_kernel_verify=True,
        native_kernel_fallback=True,
    )
    monkeypatch.setattr(runtime, "availability", lambda: native_kernels.NativeKernelAvailability(True, "available", "test"))

    block = FakeBlock()
    with runtime.patch_ltx_modules():
        assert block._scale_shift("x", "s", "t") == "native_scale_shift:x:s:t"
        assert block._gated_residual("r", "v", "g") == "native_gated_residual:r:v:g"

    assert block._scale_shift("x", "s", "t") == "original_scale_shift:x:s:t"
    assert block._gated_residual("r", "v", "g") == "original_gated_residual:r:v:g"
    assert runtime.metadata()["native_kernel_used_count"] == 2


def test_mlx_ltx_conditioning_and_latent_helpers():
    import torch

    conditioning = [[None, {"mlx_ltx_prompt": "a prompt", "frame_rate": 24.0}]]
    latent = {"samples": torch.zeros([1, 128, 7, 16, 16]), "downscale_ratio_spacial": 32}
    placeholder = make_mlx_ltx_audio_placeholder(49, 24, 1)

    assert extract_mlx_ltx_prompt(conditioning) == "a prompt"
    assert extract_mlx_ltx_frame_rate(conditioning) == 24.0
    assert infer_ltx_dimensions(latent) == (512, 512, 49)
    assert placeholder["type"] == "mlx_ltx_audio_placeholder"


def test_mlx_ltx_checkpoint_loader_and_clip_text_encode_use_existing_nodes(monkeypatch, tmp_path):
    import folder_paths
    import nodes

    manifest = tmp_path / "ltx.mlx_ltx.json"
    manifest.write_text(json.dumps({"repo_id": "dgrauet/ltx-2.3-mlx-q4", "execution_mode": "legacy_media"}), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: str(manifest))
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: str(manifest))

    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint("ltx.mlx_ltx.json")
    conditioning = nodes.CLIPTextEncode().encode(clip, "native prompt")[0]

    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)
    assert extract_mlx_ltx_prompt(conditioning) == "native prompt"


def test_mlx_ltx_checkpoint_loader_defaults_to_island_for_folder(monkeypatch, tmp_path):
    import folder_paths
    import nodes

    model_dir = tmp_path / "ltx-2.3-mlx-q4"
    model_dir.mkdir()
    for name in ("config.json", "split_model.json", "transformer-distilled-1.1.safetensors", "connector.safetensors"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda folder: [str(tmp_path)] if folder == "checkpoints" else [])
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: None)

    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint("ltx-2.3-mlx-q4")

    assert is_mlx_denoiser_island_model(model.model)
    assert isinstance(clip, MLXLTXClipProxy)
    assert is_mlx_ltx_vae(vae)


def test_ltxv_scheduler_one_step_stretch_stays_finite():
    import torch
    from comfy_extras.nodes_lt import LTXVScheduler

    sigmas, = LTXVScheduler.execute(1, 2.05, 0.95, True, 0.1, None)

    assert torch.isfinite(sigmas).all()
    assert sigmas.tolist() == pytest.approx([1.0, 0.0])


def test_ltxv_scheduler_distilled_stage_presets_match_ltx_2_3_tables():
    from comfy_extras.nodes_lt import LTXVScheduler

    stage1, = LTXVScheduler.execute(8, 2.05, 0.95, True, 0.1, None, "ltx_2_3_distilled_stage1")
    stage2, = LTXVScheduler.execute(3, 2.05, 0.95, True, 0.1, None, "ltx_2_3_distilled_stage2")

    assert stage1.tolist() == pytest.approx(
        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    )
    assert stage2.tolist() == pytest.approx([0.909375, 0.725, 0.421875, 0.0])


def test_mlx_ltx_sampler_custom_advanced_dispatches_without_torch_sampling(monkeypatch):
    import comfy.backends.mlx_ltx_backend as mlx_ltx_backend
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

    called = {}
    model = types.SimpleNamespace(is_mlx_ltx_proxy=True)
    guider = types.SimpleNamespace(
        model_patcher=model,
        original_conds={"positive": [{"mlx_ltx_prompt": "prompt"}], "negative": [{"mlx_ltx_prompt": ""}]},
        cfg=1.0,
    )
    noise = types.SimpleNamespace(seed=123)
    latent = {"samples": object()}

    def fake_run(model_arg, positive, negative, latent_arg, *, seed=None, cfg=None):
        called.update({"model": model_arg, "positive": positive, "negative": negative, "latent": latent_arg, "seed": seed, "cfg": cfg})
        return {"samples": "media", "mlx_ltx_media_path": "/tmp/out.mp4"}

    monkeypatch.setattr(mlx_ltx_backend, "run_mlx_ltx_sampling", fake_run)

    result = SamplerCustomAdvanced.execute(noise, guider, object(), object(), latent)

    assert result[0]["samples"] == "media"
    assert result[1]["samples"] == "media"
    assert called["model"] is model
    assert called["latent"] is latent
    assert called["seed"] == 123


def test_mlx_ltx_vae_proxy_decodes_video_only_on_decode(monkeypatch):
    pytest.importorskip("mlx.core")
    import torch

    calls = []

    def fake_decode_video(self, latent, *, tiled):
        calls.append(("video", tiled, tuple(latent.shape)))
        return torch.zeros(1, 1, 8, 8, 3)

    monkeypatch.setattr(MLXLTXVAEProxy, "_decode_video", fake_decode_video)
    vae = MLXLTXVAEProxy(make_model_spec(repo_id="/models/ltx"), MLXLTXRunConfig(prompt="", output_path=""))

    assert calls == []
    images = vae.decode(torch.zeros(1, 128, 1, 1, 1))

    assert images.shape == (1, 1, 8, 8, 3)
    assert calls == [("video", False, (1, 128, 1, 1, 1))]


def test_mlx_ltx_vae_proxy_encodes_i2v_image_tensor(monkeypatch):
    import torch

    calls = []

    def fake_encode_video(self, pixels):
        calls.append(tuple(pixels.shape))
        return torch.ones(1, 128, 1, 1, 1, dtype=pixels.dtype, device=pixels.device)

    monkeypatch.setattr(MLXLTXVAEProxy, "_encode_video", fake_encode_video)
    vae = MLXLTXVAEProxy(make_model_spec(repo_id="/models/ltx"), MLXLTXRunConfig(prompt="", output_path=""))

    encoded = vae.encode(torch.zeros(1, 32, 32, 3))

    assert vae.downscale_index_formula == (8, 32, 32)
    assert encoded.shape == (1, 128, 1, 1, 1)
    assert calls == [(1, 32, 32, 3)]


def test_ltxv_img_to_video_inplace_uses_mlx_vae_encode(monkeypatch):
    import torch
    from comfy_extras.nodes_lt import LTXVImgToVideoInplace

    calls = []

    def fake_encode_video(self, pixels):
        calls.append(tuple(pixels.shape))
        return torch.ones(1, 128, 1, 1, 1, dtype=pixels.dtype, device=pixels.device)

    monkeypatch.setattr(MLXLTXVAEProxy, "_encode_video", fake_encode_video)
    vae = MLXLTXVAEProxy(make_model_spec(repo_id="/models/ltx"), MLXLTXRunConfig(prompt="", output_path=""))
    latent = {"samples": torch.zeros(1, 128, 1, 1, 1)}
    image = torch.zeros(1, 32, 32, 3)

    output = LTXVImgToVideoInplace.execute(vae, image, latent, strength=1.0)[0]

    assert calls == [(1, 32, 32, 3)]
    assert torch.equal(output["samples"], torch.ones_like(output["samples"]))
    assert output["noise_mask"].shape == (1, 1, 1, 1, 1)
    assert torch.equal(output["noise_mask"], torch.zeros_like(output["noise_mask"]))


def test_mlx_ltx_vae_proxy_decode_tiled_dispatches_video(monkeypatch):
    import torch
    import nodes

    calls = []

    def fake_decode_tiled(self, latent, *, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        calls.append((tuple(latent.shape), tile_x, tile_y, overlap, tile_t, overlap_t))
        return torch.zeros(1, 2, 8, 8, 3)

    monkeypatch.setattr(MLXLTXVAEProxy, "decode_tiled", fake_decode_tiled)
    vae = MLXLTXVAEProxy(make_model_spec(repo_id="/models/ltx"), MLXLTXRunConfig(prompt="", output_path=""))
    latent = {"samples": torch.zeros(1, 128, 1, 1, 1)}

    images = nodes.VAEDecodeTiled().decode(vae, latent, 512)[0]

    assert images.shape == (2, 8, 8, 3)
    assert calls == [((1, 128, 1, 1, 1), 16, 16, 2, None, None)]


def test_mlx_ltx_audio_decode_node_uses_vae_proxy(monkeypatch):
    import torch
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode

    calls = []

    def fake_decode(self, latent):
        calls.append(tuple(latent.shape))
        return torch.zeros(1, 12, 2)

    monkeypatch.setattr(MLXLTXVAEProxy, "decode", fake_decode)
    vae = MLXLTXVAEProxy(make_model_spec(repo_id="/models/ltx"), MLXLTXRunConfig(prompt="", output_path=""))
    latent = {"samples": torch.zeros(1, 8, 2, 16)}

    audio = LTXVAudioVAEDecode.execute(latent, vae)[0]

    assert calls == [(1, 8, 2, 16)]
    assert audio["waveform"].shape == (1, 2, 12)
    assert audio["sample_rate"] == 48000


def test_mlx_ltx_latent_upscale_preserves_video_noise_mask(monkeypatch):
    pytest.importorskip("mlx.core")
    pytest.importorskip("ltx_pipelines_mlx")
    import torch
    import comfy.backends.mlx_ltx_backend as mlx_ltx_backend

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def upscale_distilled_video_latent(self, video_mx):
            import mlx.core as mx

            arr = mx.repeat(video_mx, repeats=2, axis=3)
            return mx.repeat(arr, repeats=2, axis=4)

    monkeypatch.setattr(mlx_ltx_backend, "DistilledPipeline", FakePipeline, raising=False)
    monkeypatch.setattr("ltx_pipelines_mlx.DistilledPipeline", FakePipeline)
    monkeypatch.setattr(mlx_ltx_backend._get_mlx_ltx_backend(), "resolve_model_path", lambda spec: "/models/ltx")

    upscaler = mlx_ltx_backend.MLXLTXLatentUpscalerProxy(
        make_model_spec(repo_id="/models/ltx"),
        MLXLTXRunConfig(prompt="", output_path=""),
    )
    samples = {
        "samples": torch.zeros(1, 128, 1, 2, 2),
        "noise_mask": torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]]),
    }

    output = mlx_ltx_backend.run_mlx_ltx_latent_upscale(upscaler, samples)

    assert output["samples"].shape == (1, 128, 1, 4, 4)
    assert output["noise_mask"].shape == (1, 1, 1, 4, 4)
    assert set(output["noise_mask"].flatten().tolist()) == {0.0, 1.0}


def test_mlx_ltx_media_unwrap_uses_existing_decode_nodes(monkeypatch):
    import comfy.backends.mlx_ltx_backend as mlx_ltx_backend
    import nodes
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode

    components = types.SimpleNamespace(images="frames", audio={"waveform": "audio", "sample_rate": 48000})
    monkeypatch.setattr(mlx_ltx_backend, "mlx_ltx_media_components", lambda latent: components)
    latent = {"samples": object(), "mlx_ltx_media_path": "/tmp/out.mp4", "mlx_ltx_metadata": {"media_passthrough": False}}

    assert nodes.VAEDecode().decode(None, latent) == ("frames",)
    assert LTXVAudioVAEDecode.execute(latent, None)[0] == {"waveform": "audio", "sample_rate": 48000}


def test_mlx_ltx_media_proxy_preserves_video_file_through_create_video(monkeypatch):
    import nodes
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode
    from comfy_extras.nodes_video import CreateVideo

    class FakeVideo:
        def __init__(self, path):
            self.path = path

    fake_backend = MLXLTXBackend()
    monkeypatch.setattr(fake_backend, "output_as_comfy_video", lambda path: FakeVideo(path))
    monkeypatch.setattr(comfy.backends, "get_backend", lambda name: fake_backend)
    latent = {
        "samples": object(),
        "mlx_ltx_media_path": "/tmp/out.mp4",
        "mlx_ltx_metadata": {"media_passthrough": True},
        "frame_rate": 24,
    }

    image_proxy = nodes.VAEDecode().decode(None, latent)[0]
    audio_proxy = LTXVAudioVAEDecode.execute(latent, None)[0]

    assert is_mlx_ltx_image_proxy(image_proxy)
    assert is_mlx_ltx_audio_proxy(audio_proxy)

    result = CreateVideo.execute(image_proxy, 24, audio_proxy)
    video = result[0]

    assert isinstance(video, FakeVideo)
    assert video.path == "/tmp/out.mp4"


def test_mlx_ltx_output_as_comfy_video_uses_existing_video_input(monkeypatch):
    class FakeVideo:
        def __init__(self, path):
            self.path = path

        def get_components(self):
            return {"images": "frames", "audio": "audio"}

    fake_latest = types.ModuleType("comfy_api.latest")
    fake_latest.InputImpl = types.SimpleNamespace(VideoFromFile=FakeVideo)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", fake_latest)

    backend = MLXLTXBackend()
    video = backend.output_as_comfy_video("/tmp/out.mp4")
    components = backend.output_components("/tmp/out.mp4")

    assert isinstance(video, FakeVideo)
    assert video.path == "/tmp/out.mp4"
    assert components == {"images": "frames", "audio": "audio"}
