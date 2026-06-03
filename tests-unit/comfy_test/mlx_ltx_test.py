import json
import os
import sys
import types
from pathlib import Path

import pytest

from comfy.ldm.lightricks.mlx import (
    MLXLTXRuntime,
    MLX_LTX_REQUIRED_MODEL_FILES,
    extract_mlx_ltx_frame_rate,
    extract_mlx_ltx_prompt,
    find_mlx_ltx_checkpoint_folder,
    infer_ltx_dimensions,
    is_mlx_ltx_audio_proxy,
    is_mlx_ltx_image_proxy,
    is_mlx_ltx_manifest_path,
    is_mlx_ltx_model_dir,
    is_mlx_ltx_model,
    is_mlx_ltx_vae,
    list_mlx_ltx_checkpoint_folders,
    load_mlx_ltx_checkpoint,
    load_mlx_ltx_manifest,
    make_model_spec,
    make_mlx_ltx_audio_placeholder,
    run_mlx_ltx_sampling,
    validate_ltx_frame_count,
)


def write_fake_mlx_ltx_model_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    for filename in MLX_LTX_REQUIRED_MODEL_FILES:
        (path / filename).write_text("{}", encoding="utf-8")
    return path


def test_import_does_not_import_optional_ltx_modules():
    sys.modules.pop("ltx_pipelines_mlx", None)
    sys.modules.pop("ltx_pipelines_mlx.distilled", None)

    import comfy.ldm.lightricks.mlx  # noqa: F401

    assert "ltx_pipelines_mlx" not in sys.modules
    assert "ltx_pipelines_mlx.distilled" not in sys.modules


def test_mlx_ltx_frame_count_validation():
    assert validate_ltx_frame_count(49) == 49
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


def test_mlx_ltx_resolve_local_model_path_does_not_require_huggingface(tmp_path):
    spec = make_model_spec(repo_id=str(tmp_path))

    resolved = MLXLTXRuntime().resolve_model_path(spec)

    assert resolved == os.path.abspath(tmp_path)


def test_mlx_ltx_model_directory_detection_and_proxy_checkpoint(tmp_path):
    model_dir = write_fake_mlx_ltx_model_dir(tmp_path / "ltx-2.3-mlx-q4")

    assert is_mlx_ltx_model_dir(model_dir)
    assert list_mlx_ltx_checkpoint_folders([str(tmp_path)]) == ["ltx-2.3-mlx-q4"]
    assert find_mlx_ltx_checkpoint_folder("ltx-2.3-mlx-q4", [str(tmp_path)]) == str(model_dir)

    model, clip, vae = load_mlx_ltx_checkpoint(model_dir)

    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)
    assert model.spec.local_path == str(model_dir.resolve())
    assert model.spec.variant == "q4"
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("hello"))
    assert extract_mlx_ltx_prompt(conditioning) == "hello"


def test_mlx_ltx_model_directory_rejects_missing_spatial_upscaler(tmp_path):
    model_dir = write_fake_mlx_ltx_model_dir(tmp_path / "ltx-2.3-mlx-q4")
    (model_dir / "spatial_upscaler_x2_v1_1.safetensors").unlink()

    assert not is_mlx_ltx_model_dir(model_dir)
    assert list_mlx_ltx_checkpoint_folders([str(tmp_path)]) == []
    assert find_mlx_ltx_checkpoint_folder("ltx-2.3-mlx-q4", [str(tmp_path)]) is None


def test_mlx_ltx_model_directory_rejects_older_unversioned_distilled_transformer(tmp_path):
    model_dir = write_fake_mlx_ltx_model_dir(tmp_path / "ltx-2.3-mlx-q4")
    (model_dir / "transformer-distilled-1.1.safetensors").unlink()
    (model_dir / "transformer-distilled.safetensors").write_text("{}", encoding="utf-8")

    assert not is_mlx_ltx_model_dir(model_dir)
    assert list_mlx_ltx_checkpoint_folders([str(tmp_path)]) == []
    assert find_mlx_ltx_checkpoint_folder("ltx-2.3-mlx-q4", [str(tmp_path)]) is None


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

    resolved = MLXLTXRuntime().resolve_model_path(spec, cache_dir="/tmp/mlx-ltx-cache")

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


def test_mlx_ltx_manifest_parsing_and_proxy_checkpoint(tmp_path):
    manifest = tmp_path / "ltx.mlx_ltx.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "dgrauet/ltx-2.3-mlx-q8",
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
    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize("hello"))
    assert extract_mlx_ltx_prompt(conditioning) == "hello"


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

    runtime = MLXLTXRuntime()
    monkeypatch.setattr(runtime, "_import_mlx_core", lambda: types.SimpleNamespace(default_device=lambda: "gpu"))
    monkeypatch.setattr(runtime, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(runtime, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")
    spec = make_model_spec(repo_id="dgrauet/ltx-2.3-mlx-q4", variant="q4")
    output = tmp_path / "out.mp4"

    result = runtime.generate_video_to_file(
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

    runtime.generate_video_to_file(
        spec,
        "prompt",
        str(tmp_path / "out2.mp4"),
        width=512,
        height=512,
        num_frames=49,
        frame_rate=24,
        stage1_steps=8,
        stage2_steps=3,
        seed=123,
    )

    assert len(init_calls) == 1


def test_mlx_ltx_empty_cache_is_explicit_only(monkeypatch, tmp_path):
    calls = {"clear": 0}
    fake_mx = types.SimpleNamespace(default_device=lambda: "gpu", clear_cache=lambda: calls.__setitem__("clear", calls["clear"] + 1))

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def generate_and_save(self, **kwargs):
            Path(kwargs["output_path"]).write_bytes(b"fake mp4")
            return kwargs["output_path"]

    runtime = MLXLTXRuntime()
    monkeypatch.setattr(runtime, "_import_mlx_core", lambda: fake_mx)
    monkeypatch.setattr(runtime, "_import_distilled_pipeline", lambda: FakePipeline)
    monkeypatch.setattr(runtime, "resolve_model_path", lambda spec, cache_dir=None: "/models/ltx")

    runtime.generate_video_to_file(
        make_model_spec(repo_id="repo"),
        "prompt",
        str(tmp_path / "out.mp4"),
        width=512,
        height=512,
        num_frames=49,
        frame_rate=24,
        stage1_steps=8,
        stage2_steps=3,
        seed=42,
    )
    assert calls["clear"] == 0

    runtime.empty_cache()
    assert calls["clear"] == 1


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
    manifest.write_text(json.dumps({"repo_id": "dgrauet/ltx-2.3-mlx-q4"}), encoding="utf-8")
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: str(manifest))
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: str(manifest))

    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint("ltx.mlx_ltx.json")
    conditioning = nodes.CLIPTextEncode().encode(clip, "native prompt")[0]

    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)
    assert extract_mlx_ltx_prompt(conditioning) == "native prompt"


def test_mlx_ltx_checkpoint_loader_lists_and_loads_model_folder(monkeypatch, tmp_path):
    import folder_paths
    import nodes

    checkpoints_dir = tmp_path / "checkpoints"
    write_fake_mlx_ltx_model_dir(checkpoints_dir / "ltx-2.3-mlx-q4")
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "checkpoints", ([str(checkpoints_dir)], folder_paths.supported_checkpoint_extensions))
    folder_paths.filename_list_cache.clear()

    ckpt_choices = nodes.CheckpointLoaderSimple.INPUT_TYPES()["required"]["ckpt_name"][0]
    model, _clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint("ltx-2.3-mlx-q4")

    assert "ltx-2.3-mlx-q4" in ckpt_choices
    assert is_mlx_ltx_model(model)
    assert is_mlx_ltx_vae(vae)


def test_mlx_ltx_checkpoint_loader_keeps_regular_checkpoint_path(monkeypatch, tmp_path):
    import comfy.sd
    import folder_paths
    import nodes

    ckpt_path = tmp_path / "model.safetensors"
    ckpt_path.write_bytes(b"fake")
    monkeypatch.setattr(folder_paths, "get_full_path", lambda folder, name: str(ckpt_path))
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda folder: [])
    monkeypatch.setattr(comfy.sd, "load_checkpoint_guess_config", lambda *args, **kwargs: ("model", "clip", "vae", "extra"))

    assert nodes.CheckpointLoaderSimple().load_checkpoint("model.safetensors") == ("model", "clip", "vae")


def test_mlx_ltx_sampler_custom_advanced_dispatches_without_torch_sampling(monkeypatch):
    import comfy.ldm.lightricks.mlx as mlx_ltx
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

    monkeypatch.setattr(mlx_ltx, "run_mlx_ltx_sampling", fake_run)

    result = SamplerCustomAdvanced.execute(noise, guider, object(), object(), latent)

    assert result[0]["samples"] == "media"
    assert result[1]["samples"] == "media"
    assert called["model"] is model
    assert called["latent"] is latent
    assert called["seed"] == 123


def test_mlx_ltx_media_unwrap_uses_existing_decode_nodes(monkeypatch):
    import comfy.ldm.lightricks.mlx as mlx_ltx
    import nodes
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode

    components = types.SimpleNamespace(images="frames", audio={"waveform": "audio", "sample_rate": 48000})
    monkeypatch.setattr(mlx_ltx, "mlx_ltx_media_components", lambda latent: components)
    latent = {"samples": object(), "mlx_ltx_media_path": "/tmp/out.mp4", "mlx_ltx_metadata": {"media_passthrough": False}}

    assert nodes.VAEDecode().decode(None, latent) == ("frames",)
    assert LTXVAudioVAEDecode.execute(latent, None)[0] == {"waveform": "audio", "sample_rate": 48000}


def test_mlx_ltx_media_passthrough_proxies_existing_video_node(monkeypatch):
    import comfy.ldm.lightricks.mlx as mlx_ltx
    import nodes
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode

    latent = {"samples": object(), "mlx_ltx_media_path": "/tmp/out.mp4", "mlx_ltx_metadata": {"media_passthrough": True}}

    image_proxy = nodes.VAEDecode().decode(None, latent)[0]
    audio_proxy = LTXVAudioVAEDecode.execute(latent, None)[0]

    assert is_mlx_ltx_image_proxy(image_proxy)
    assert is_mlx_ltx_audio_proxy(audio_proxy)
    assert image_proxy.media_path == "/tmp/out.mp4"
    assert audio_proxy.media_path == "/tmp/out.mp4"
