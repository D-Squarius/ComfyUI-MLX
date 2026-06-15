import json
import os
import sys
import types

import pytest

import comfy.backends
import comfy.text_encoders.mlx as mlx_module
from comfy.backends.mlx_backend import MLXBackend, MLXModelHandle, make_model_spec, redact_secrets
from comfy.text_encoders.mlx import (
    MLXGenerationOutput,
    MLXGenerationRequest,
    MLXManifest,
    MLXRuntimeAdapter,
    _collect_vlm_images,
    _filter_mlx_lm_generate_kwargs,
    _filter_mlx_lm_stream_kwargs,
    load_manifest,
    load_mlx_clip,
)


@pytest.fixture(autouse=True)
def no_processing_interrupt_import(monkeypatch):
    monkeypatch.setattr(mlx_module, "_throw_if_interrupted", lambda: None)


def test_builtin_mlx_backend_registration_does_not_import_optional_modules():
    sys.modules.pop("mlx", None)
    sys.modules.pop("mlx_lm", None)

    backends = {backend.name for backend in comfy.backends.list_backends()}

    assert "mlx" in backends
    assert "mlx" not in sys.modules
    assert "mlx_lm" not in sys.modules


def test_non_apple_platform_reports_unavailable(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    status = MLXBackend().capabilities()

    assert status.available is False
    assert "Apple Silicon" in status.reason


def test_redact_hf_tokens_and_authorization_values():
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    fake_bearer = "abcdefghijklmnop"
    text = f"failed token={fake_hf_token} Authorization Bearer {fake_bearer}"

    redacted = redact_secrets(text)

    assert fake_hf_token not in redacted
    assert fake_bearer not in redacted
    assert "hf_***" in redacted


def test_make_model_spec_normalizes_patterns():
    spec = make_model_spec(
        repo_id=" mlx-community/test ",
        revision=" main ",
        allow_patterns="*.json, *.safetensors\n*.txt",
        ignore_patterns="*.msgpack",
        local_files_only=True,
        use_hf_token=True,
    )

    assert spec.repo_id == "mlx-community/test"
    assert spec.revision == "main"
    assert spec.allow_patterns == ("*.json", "*.safetensors", "*.txt")
    assert spec.ignore_patterns == ("*.msgpack",)
    assert spec.local_files_only is True
    assert spec.use_hf_token is True


def test_resolve_local_model_path_does_not_require_huggingface(tmp_path):
    spec = make_model_spec(repo_id=str(tmp_path))

    resolved = MLXBackend().resolve_model_path(spec)

    assert resolved == os.path.abspath(tmp_path)


def test_resolve_hf_snapshot_uses_manifest_controls(monkeypatch):
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/mlx-cache/models--test"

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    spec = make_model_spec(
        repo_id="mlx-community/test",
        revision="deadbeef",
        allow_patterns=["*.json", "*.safetensors"],
        ignore_patterns=["*.md"],
        local_files_only=True,
        use_hf_token=True,
    )

    resolved = MLXBackend().resolve_model_path(spec, cache_dir="/tmp/mlx-cache")

    assert resolved == "/tmp/mlx-cache/models--test"
    assert calls == {
        "repo_id": "mlx-community/test",
        "revision": "deadbeef",
        "cache_dir": "/tmp/mlx-cache",
        "local_files_only": True,
        "allow_patterns": ("*.json", "*.safetensors"),
        "ignore_patterns": ("*.md",),
        "token": True,
    }


def test_mlx_manifest_parses_hf_snapshot_settings(tmp_path):
    manifest_path = tmp_path / "qwen.mlx.json"
    manifest_path.write_text(
        """
        {
          "backend": "mlx_lm",
          "repo_id": "mlx-community/Qwen3-0.6B-4bit",
          "revision": "abc123",
          "allow_patterns": ["*.json", "*.safetensors"],
          "ignore_patterns": "*.md,*.txt",
          "local_files_only": true,
          "use_hf_token": true,
          "generation": {"max_kv_size": 2048}
        }
        """,
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path), "mlx_lm")

    assert manifest.task == "lm"
    assert manifest.library == "mlx_lm"
    assert manifest.spec.repo_id == "mlx-community/Qwen3-0.6B-4bit"
    assert manifest.spec.revision == "abc123"
    assert manifest.spec.allow_patterns == ("*.json", "*.safetensors")
    assert manifest.spec.ignore_patterns == ("*.md", "*.txt")
    assert manifest.spec.local_files_only is True
    assert manifest.spec.use_hf_token is True
    assert manifest.generation == {"max_kv_size": 2048}


def test_mlx_manifest_accepts_mtplx_library_and_generation_controls(tmp_path):
    manifest_path = tmp_path / "mtplx.mlx.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "mlx_lm",
                "library": "mtplx",
                "repo_id": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
                "load": {"mtp": True},
                "generation": {
                    "generation_mode": "mtp",
                    "speculative_depth": 3,
                    "min_speculative_depth": 1,
                    "verify_strategy": "capture_commit",
                    "verify_core": "linear-gdn-from-conv-tape",
                    "mtp_hidden_variant": "post_norm",
                    "mtp_cache_policy": "persistent",
                    "mtp_history_policy": "committed",
                    "draft_sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                    "prompt_encoding": "mtplx_prompt_case",
                    "profile": "sustained",
                    "strict_contract": True,
                    "runtime_mode": "in_process",
                    "unsupported_sampling_policy": "error",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path), "mlx_lm")

    assert manifest.task == "lm"
    assert manifest.library == "mtplx"
    assert manifest.load == {"mtp": True}
    assert manifest.generation["generation_mode"] == "mtp"
    assert manifest.generation["speculative_depth"] == 3
    assert manifest.generation["verify_core"] == "linear-gdn-from-conv-tape"
    assert manifest.generation["draft_sampler"]["top_k"] == 20
    assert manifest.generation["prompt_encoding"] == "mtplx_prompt_case"


def test_mlx_manifest_rejects_ambiguous_model_sources(tmp_path):
    manifest_path = tmp_path / "bad.mlx.json"
    manifest_path.write_text(
        '{"backend": "mlx_lm", "repo_id": "mlx-community/test", "local_path": "./local"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="should not set both"):
        load_manifest(str(manifest_path), "mlx_lm")


def test_mlx_manifest_rejects_inline_tokens(tmp_path):
    manifest_path = tmp_path / "bad.mlx"
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    manifest_path.write_text(
        f'{{"backend": "mlx", "repo_id": "mlx-community/test", "token": "{fake_hf_token}"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain token"):
        load_manifest(str(manifest_path), "mlx_lm")


def test_mlx_manifest_rejects_secret_key_aliases(tmp_path):
    manifest_path = tmp_path / "bad.mlx"
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    manifest_path.write_text(
        f'{{"backend": "mlx_lm", "repo_id": "mlx-community/test", "load": {{"private_token": "{fake_hf_token}"}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain token"):
        load_manifest(str(manifest_path), "mlx_lm")


def test_mlx_manifest_rejects_loader_task_mismatch(tmp_path):
    manifest_path = tmp_path / "bad.mlx.json"
    manifest_path.write_text(
        '{"backend": "mlx_lm", "repo_id": "mlx-community/test", "task": "tts"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match loader type"):
        load_manifest(str(manifest_path), "mlx_lm")


@pytest.mark.parametrize(
    ("loader_type", "task"),
    [
        ("mlx_lm", "lm"),
        ("mlx_vlm", "vlm"),
        ("mlx_audio_asr", "audio_asr"),
        ("mlx_audio_tts", "audio_tts"),
        ("mlx_embeddings", "embeddings"),
    ],
)
def test_load_mlx_clip_uses_core_loader_for_each_modality(monkeypatch, tmp_path, loader_type, task):
    manifest_path = tmp_path / f"{loader_type}.mlx.json"
    manifest_path.write_text(
        f'{{"backend": "{loader_type}", "repo_id": "mlx-community/test", "cache_dir": "{tmp_path / "hf"}"}}',
        encoding="utf-8",
    )

    class FakeBackend:
        generation_lock = contextlib_null()

        def __init__(self):
            self.resolved = []

        def capabilities(self):
            return types.SimpleNamespace(available=True, reason="")

        def resolve_model_path(self, spec, cache_dir=None):
            self.resolved.append((spec, cache_dir))
            return "/resolved/model"

        def register_handle(self, handle):
            return handle

        def unload(self, handle):
            handle.model = None
            handle.processor = None

    fake_backend = FakeBackend()
    loads = []

    def fake_load_handle(backend, manifest, model_path):
        loads.append((backend, manifest, model_path))
        return MLXModelHandle(
            backend="mlx",
            spec=manifest.spec,
            model=object(),
            processor=object(),
            resolved_path=model_path,
            modality=manifest.task,
        )

    monkeypatch.setattr(mlx_module, "get_backend", lambda name: fake_backend)
    monkeypatch.setattr(mlx_module, "_load_handle", fake_load_handle)

    adapter = load_mlx_clip(str(manifest_path), loader_type)

    assert adapter.is_mlx_runtime is True
    assert adapter.mlx_modality == task
    assert fake_backend.resolved[0][0].repo_id == "mlx-community/test"
    assert fake_backend.resolved[0][1] == str(tmp_path / "hf")
    assert loads[0][1].task == task
    assert loads[0][2] == "/resolved/model"


def test_text_generate_adapter_round_trips_text_and_audio():
    class FakeBackend:
        generation_lock = contextlib_null()

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_audio_tts",
        task="audio_tts",
        library="mlx_audio",
        spec=make_model_spec(repo_id="fake"),
    )
    handle = type("Handle", (), {"model": None, "processor": None})()
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)
    audio = {"waveform": object(), "sample_rate": 24000}
    output = MLXGenerationOutput("Generated audio.", audio=audio)

    assert adapter.is_mlx_runtime is True
    assert adapter.supports_textgenerate is True
    assert adapter.supports_conditioning is False
    assert isinstance(adapter.tokenize("hello"), MLXGenerationRequest)
    assert adapter.decode(output) == "Generated audio."
    assert adapter.decode_audio(output) is audio
    with pytest.raises(RuntimeError, match="TextGenerate"):
        adapter.encode_from_tokens({})


def test_mlx_audio_asr_requires_audio_input():
    class FakeBackend:
        generation_lock = contextlib_null()

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_audio_asr",
        task="audio_asr",
        library="mlx_audio",
        spec=make_model_spec(repo_id="fake"),
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=object(), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    with pytest.raises(ValueError, match="needs an AUDIO input"):
        adapter.generate(adapter.tokenize("transcribe this"))


def test_mlx_lm_stream_kwargs_filter_outer_only_options():
    class FakeUtils:
        @staticmethod
        def generate_step(prompt, model, temp=0.0, top_p=1.0, min_p=0.0, max_kv_size=None):
            return ()

    fake_mlx_lm = types.SimpleNamespace(utils=FakeUtils)

    filtered = _filter_mlx_lm_stream_kwargs(
        fake_mlx_lm,
        {
            "max_tokens": 8,
            "temp": 0.0,
            "top_p": 0.9,
            "min_p": 0.05,
            "max_kv_size": 128,
            "verbose": False,
            "unknown": True,
        },
    )

    assert filtered == {
        "max_tokens": 8,
        "temp": 0.0,
        "top_p": 0.9,
        "min_p": 0.05,
        "max_kv_size": 128,
    }


def test_mlx_lm_stream_kwargs_build_sampler_for_new_mlx_lm():
    sampler_calls = []

    class FakeGenerateModule:
        @staticmethod
        def generate_step(prompt, model, *, max_tokens=256, sampler=None, max_kv_size=None):
            return ()

        @staticmethod
        def make_sampler(temp=0.0, top_p=0.0, min_p=0.0, top_k=0):
            sampler_calls.append((temp, top_p, min_p, top_k))
            return "sampler"

    filtered = _filter_mlx_lm_stream_kwargs(
        FakeGenerateModule,
        {
            "max_tokens": 8,
            "temp": 0.0,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 64,
            "max_kv_size": 128,
            "verbose": False,
        },
    )

    assert sampler_calls == [(0.0, 0.9, 0.05, 64)]
    assert filtered == {"max_tokens": 8, "max_kv_size": 128, "sampler": "sampler"}


def test_mlx_lm_generate_kwargs_filter_inner_generate_step_options():
    def generate(model, tokenizer, prompt, max_tokens=100, verbose=False, **kwargs):
        return "text"

    class FakeUtils:
        @staticmethod
        def generate_step(prompt, model, temp=0.0, top_p=1.0, min_p=0.0, max_kv_size=None):
            return ()

    fake_mlx_lm = types.SimpleNamespace(generate=generate, utils=FakeUtils)

    filtered = _filter_mlx_lm_generate_kwargs(
        fake_mlx_lm,
        {
            "max_tokens": 8,
            "temp": 0.0,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 64,
            "max_kv_size": 128,
            "verbose": False,
        },
    )

    assert filtered == {
        "max_tokens": 8,
        "temp": 0.0,
        "top_p": 0.9,
        "min_p": 0.05,
        "max_kv_size": 128,
        "verbose": False,
    }


def test_mlx_lm_defaults_to_non_stream_and_reports_clean_timing(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeTokenizer:
        def encode(self, text):
            return str(text).split()

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return f"templated {messages[0]['content']}"

    class FakeMLXLM:
        @staticmethod
        def generate(model, tokenizer, prompt, **kwargs):
            assert model == "model"
            assert prompt == "templated hello world"
            assert kwargs["max_tokens"] == 3
            assert kwargs["temp"] == 0.0
            assert kwargs["verbose"] is False
            return "one two three"

        @staticmethod
        def generate_step(prompt, model, temp=0.0, top_p=1.0, min_p=0.0, max_kv_size=None):
            return ()

        @staticmethod
        def stream_generate(*args, **kwargs):
            raise AssertionError("default MLX LM path should use non-stream generation")

    events = []
    monkeypatch.setattr(mlx_module, "_import_dependency", lambda module, extra: FakeMLXLM if module == "mlx_lm" else None)
    monkeypatch.setattr(mlx_module, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mlx_lm",
        spec=make_model_spec(repo_id="fake"),
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="model", processor=FakeTokenizer())
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("hello world"), do_sample=False, max_length=3)

    assert output.text == "one two three"
    start_event = next(event for event in events if event["event_type"] == "generate_start")
    end_event = next(event for event in events if event["event_type"] == "generate_end")
    assert start_event["stream"] is False
    assert start_event["template_mode"] == "chat_template"
    assert start_event["prompt_tokens"] == 3
    assert end_event["generation_mode"] == "non_stream"
    assert end_event["first_token_seconds"] is None
    assert end_event["output_tokens"] == 3
    assert end_event["token_count_source"] == "fallback_tokenizer_after_generation"
    assert end_event["preprocess_seconds"] >= 0
    assert end_event["token_count_seconds"] >= 0


def test_mtplx_optional_import_is_lazy_for_regular_mlx_lm(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeTokenizer:
        def encode(self, text):
            return [1, 2]

    class FakeMLXLM:
        @staticmethod
        def generate(model, tokenizer, prompt, **kwargs):
            return "ok"

        @staticmethod
        def generate_step(prompt, model, temp=0.0):
            return ()

    imported = []

    def fake_import(module, extra):
        imported.append(module)
        if module == "mlx_lm":
            return FakeMLXLM
        raise AssertionError(f"unexpected optional import: {module}")

    monkeypatch.setattr(mlx_module, "_import_dependency", fake_import)
    manifest = MLXManifest(
        manifest_path="regular.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mlx_lm",
        spec=make_model_spec(repo_id="fake"),
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="model", processor=FakeTokenizer())
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    assert adapter.generate(adapter.tokenize("hello"), do_sample=False).text == "ok"
    assert "mtplx.runtime" not in imported


def test_mtplx_load_accepts_verified_contract(monkeypatch):
    class FakeBackend:
        def register_handle(self, handle):
            return handle

    class FakeTokenizer:
        pass

    class FakeRuntime:
        tokenizer = FakeTokenizer()

    class FakeMTPLXRuntimeModule:
        @staticmethod
        def inspect(path):
            assert path == "/models/mtplx"
            return {
                "compatibility": {
                    "can_run": True,
                    "runtime_contract_path": "/models/mtplx/mtplx_runtime.json",
                }
            }

        @staticmethod
        def load(path, mtp=True, mtp_adapter=None):
            assert path == "/models/mtplx"
            assert mtp is True
            assert mtp_adapter is None
            return FakeRuntime()

    monkeypatch.setattr(mlx_module, "_import_dependency", lambda module, extra: FakeMTPLXRuntimeModule)
    manifest = MLXManifest(
        manifest_path="mtplx.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mtplx",
        spec=make_model_spec(repo_id="fake"),
        load={"mtp": True},
        generation={"strict_contract": True},
    )

    handle = mlx_module._load_handle(FakeBackend(), manifest, "/models/mtplx")

    assert isinstance(handle.model, FakeRuntime)
    assert isinstance(handle.processor, FakeTokenizer)
    assert handle.mtplx_metadata["runtime_mode"] == "in_process"


def test_mtplx_load_rejects_unverified_or_no_mtp_contract(monkeypatch):
    class FakeBackend:
        def register_handle(self, handle):
            return handle

    class FakeMTPLXRuntimeModule:
        @staticmethod
        def inspect(path):
            return {"compatibility": {"can_run": False, "reason": "no MTP heads"}}

    monkeypatch.setattr(mlx_module, "_import_dependency", lambda module, extra: FakeMTPLXRuntimeModule)
    manifest = MLXManifest(
        manifest_path="mtplx.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mtplx",
        spec=make_model_spec(repo_id="fake"),
        generation={"strict_contract": True},
    )

    with pytest.raises(comfy.backends.BackendUnavailableError, match="strict_contract|compatibility"):
        mlx_module._load_handle(FakeBackend(), manifest, "/models/not-mtplx")


def test_mtplx_generation_maps_stats_to_events(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeTokenizer:
        def encode(self, text):
            assert "hello" in text
            return [10, 11]

    class FakeSamplerConfig:
        def __init__(self, temperature=0.0, top_p=1.0, top_k=0):
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k

    class FakeStats:
        def to_dict(self):
            return {
                "generated_tokens": 4,
                "speculative_depth": 3,
                "accepted_drafts": 6,
                "drafted_tokens": 8,
                "requested_speculative_depth": 3,
                "accepted_by_depth": [4, 2, 0],
                "drafted_by_depth": [4, 3, 1],
                "mean_accept_probability_by_depth": [0.9, 0.5, 0.1],
                "prompt_tps": 120.0,
                "prompt_target_prefill_time_s": 0.11,
                "prompt_mtp_history_time_s": 0.09,
                "cached_tokens": 12,
                "new_prefill_tokens": 2,
                "session_cache_hit": True,
                "verify_core": "linear-gdn-from-conv-tape",
                "mtp_history_policy": "committed",
                "prefill_chunks": 1,
                "peak_memory_bytes": 1234,
                "prompt_eval_time_s": 0.2,
                "draft_time_s": 0.3,
                "verify_time_s": 0.4,
                "decode_tok_s": 40.0,
                "end_to_end_tok_s": 30.0,
            }

    class FakeGeneration:
        @staticmethod
        def generate_mtpk(runtime, prompt_ids, **kwargs):
            assert runtime == "runtime"
            assert prompt_ids == [10, 11]
            assert kwargs["speculative_depth"] == 3
            assert kwargs["min_speculative_depth"] == 1
            assert kwargs["verify_strategy"] == "capture_commit"
            assert kwargs["verify_core"] == "linear-gdn-from-conv-tape"
            assert kwargs["mtp_hidden_variant"] == "post_norm"
            assert kwargs["mtp_cache_policy"] == "persistent"
            assert kwargs["mtp_history_policy"] == "committed"
            assert kwargs["draft_core"] == "stock"
            assert kwargs["draft_sampler"].temperature == 0.6
            assert kwargs["draft_sampler"].top_p == 0.95
            assert kwargs["draft_sampler"].top_k == 20
            kwargs["token_callback"]([1, 2])
            return types.SimpleNamespace(text="reply", stats=FakeStats())

    class FakeSampling:
        SamplerConfig = FakeSamplerConfig

    class FakeProfile:
        def env_dict(self):
            return {"MTPLX_FAKE_PROFILE": "1"}

    class FakeProfiles:
        applied = []
        restored = []

        @staticmethod
        def get_profile(name):
            assert name == "performance-cold"
            return FakeProfile()

        @classmethod
        def apply_profile_env(cls, name):
            assert name == "performance-cold"
            cls.applied.append(name)
            return {"MTPLX_FAKE_PROFILE": None}

        @classmethod
        def restore_profile_env(cls, previous):
            cls.restored.append(previous)

    def fake_import(module, extra):
        if module == "mtplx.generation":
            return FakeGeneration
        if module == "mtplx.sampling":
            return FakeSampling
        if module == "mtplx.profiles":
            return FakeProfiles
        raise AssertionError(module)

    events = []
    monkeypatch.setattr(mlx_module, "_import_dependency", fake_import)
    monkeypatch.setattr(mlx_module, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))
    manifest = MLXManifest(
        manifest_path="mtplx.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mtplx",
        spec=make_model_spec(repo_id="fake"),
        generation={
            "generation_mode": "mtp",
            "speculative_depth": 3,
            "min_speculative_depth": 1,
            "verify_strategy": "capture_commit",
            "verify_core": "linear-gdn-from-conv-tape",
            "mtp_hidden_variant": "post_norm",
            "mtp_cache_policy": "persistent",
            "mtp_history_policy": "committed",
            "draft_sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "profile": "performance-cold",
            "unsupported_sampling_policy": "ignore",
        },
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="runtime", processor=FakeTokenizer())
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("hello"), do_sample=False, max_length=4)

    end_event = next(event for event in events if event["event_type"] == "generate_end")
    assert output.text == "reply"
    assert end_event["generation_mode"] == "mtplx_mtp"
    assert end_event["output_tokens"] == 4
    assert end_event["speculative_depth"] == 3
    assert end_event["accepted_drafts"] == 6
    assert end_event["drafted_tokens"] == 8
    assert end_event["acceptance_rate"] == 0.75
    assert end_event["decode_tok_s"] == 40.0
    assert end_event["verify_core"] == "linear-gdn-from-conv-tape"
    assert end_event["accepted_by_depth"] == [4, 2, 0]
    assert end_event["drafted_by_depth"] == [4, 3, 1]
    assert end_event["prompt_target_prefill_time_s"] == 0.11
    assert end_event["session_cache_hit"] is True
    assert end_event["draft_sampler_temperature"] == 0.6
    assert FakeProfiles.applied == ["performance-cold"]
    assert FakeProfiles.restored == [{"MTPLX_FAKE_PROFILE": None}]


def test_mtplx_prompt_case_encoding_uses_mtplx_schema(monkeypatch):
    captured = {}

    class FakePromptCase:
        def __init__(self, **kwargs):
            captured["case"] = kwargs
            self.prompt = kwargs["prompt"]

    class FakeSchema:
        PromptCase = FakePromptCase

        @staticmethod
        def encode_prompt_case(tokenizer, case, *, chat_template=False, enable_thinking=None):
            captured["tokenizer"] = tokenizer
            captured["chat_template"] = chat_template
            captured["enable_thinking"] = enable_thinking
            captured["prompt"] = case.prompt
            return [7, 8, 9]

    monkeypatch.setattr(
        mlx_module,
        "_import_dependency",
        lambda module, extra: FakeSchema if module == "mtplx.benchmarks.schema" else pytest.fail(module),
    )

    tokens = mlx_module._mtplx_encode_prompt(
        object(),
        "hello",
        {
            "prompt_encoding": "mtplx_prompt_case",
            "max_tokens": 12,
            "chat_template": True,
            "enable_thinking": False,
        },
    )

    assert tokens == [7, 8, 9]
    assert captured["case"]["id"] == "comfy_textgenerate"
    assert captured["case"]["max_tokens"] == 12
    assert captured["prompt"] == "hello"
    assert captured["chat_template"] is True
    assert captured["enable_thinking"] is False


def test_mtplx_generation_rejects_unsupported_sampling_by_default(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeTokenizer:
        def encode(self, text):
            return [1]

    manifest = MLXManifest(
        manifest_path="mtplx.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mtplx",
        spec=make_model_spec(repo_id="fake"),
        generation={"generation_mode": "mtp"},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="runtime", processor=FakeTokenizer())
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    with pytest.raises(comfy.backends.BackendUnavailableError, match="sampling controls"):
        adapter.generate(adapter.tokenize("hello"), do_sample=True, min_p=0.05, repetition_penalty=1.05)


def test_mtplx_http_bridge_formats_openai_request(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    captured = {}

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "hello from server"}}],
                    "usage": {"completion_tokens": 3},
                    "mtplx_stats": {"generated_tokens": 3, "accepted_drafts": 2, "drafted_tokens": 4, "speculative_depth": 2},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    monkeypatch.setattr(mlx_module.urllib.request, "urlopen", fake_urlopen)
    manifest = MLXManifest(
        manifest_path="mtplx_http.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mtplx",
        spec=make_model_spec(repo_id="fake"),
        generation={
            "runtime_mode": "http",
            "base_url": "http://127.0.0.1:18083",
            "generation_mode": "mtp",
            "speculative_depth": 2,
            "unsupported_sampling_policy": "ignore",
        },
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=mlx_module.MTPLXHTTPRuntime("http://127.0.0.1:18083", "/tmp/model"), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("prompt"), do_sample=False, max_length=7)

    assert output.text == "hello from server"
    assert captured["url"] == "http://127.0.0.1:18083/v1/chat/completions"
    assert captured["payload"]["messages"][0]["content"] == "prompt"
    assert captured["payload"]["generation_mode"] == "mtp"
    assert captured["payload"]["depth"] == 2
    assert output.metadata["acceptance_rate"] == 0.5


def test_mlx_lm_stream_mode_counts_chunks_and_throttles_interrupts(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeTokenizer:
        def encode(self, text):
            return list(range(len(str(text).split())))

    class FakeMLXLM:
        @staticmethod
        def stream_generate(model, tokenizer, prompt, **kwargs):
            assert kwargs["max_tokens"] == 16
            for index in range(16):
                yield types.SimpleNamespace(text="x", token=index)

        @staticmethod
        def generate(*args, **kwargs):
            raise AssertionError("stream manifest option should use stream_generate")

    interrupt_checks = []
    events = []
    monkeypatch.setattr(mlx_module, "_import_dependency", lambda module, extra: FakeMLXLM if module == "mlx_lm" else None)
    monkeypatch.setattr(mlx_module, "_throw_if_interrupted", lambda: interrupt_checks.append(True))
    monkeypatch.setattr(mlx_module, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_lm",
        task="lm",
        library="mlx_lm",
        spec=make_model_spec(repo_id="fake"),
        generation={"stream": True, "interrupt_poll_interval": 8, "interrupt_poll_seconds": 999},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="model", processor=FakeTokenizer())
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("hello"), max_length=16)

    end_event = next(event for event in events if event["event_type"] == "generate_end")
    assert output.text == "x" * 16
    assert end_event["stream"] is True
    assert end_event["generation_mode"] == "stream"
    assert end_event["output_tokens"] == 16
    assert end_event["token_count_source"] == "mlx_stream_chunk_token_ids"
    assert end_event["first_token_seconds"] is not None
    assert len(interrupt_checks) == 3


def test_mlx_audio_tts_returns_audio_dict():
    class FakeBackend:
        generation_lock = contextlib_null()

    class FakeTTS:
        def generate(self, text):
            assert text == "say it"
            return [
                types.SimpleNamespace(audio=[0.0, 0.25], text="hello ", sample_rate=16000),
                types.SimpleNamespace(audio=[0.5], text="world", sample_rate=16000),
            ]

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_audio_tts",
        task="audio_tts",
        library="mlx_audio",
        spec=make_model_spec(repo_id="fake"),
        audio={"sample_rate": 16000},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=FakeTTS(), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("say it"))

    assert output.text == "hello world"
    assert output.audio["sample_rate"] == 16000
    assert tuple(output.audio["waveform"].shape) == (1, 1, 3)


def test_mlx_audio_tts_accepts_raw_audio_arrays():
    class FakeBackend:
        generation_lock = contextlib_null()

    class FakeRawTTS:
        def generate(self, text):
            import numpy as np

            assert text == "say it"
            return np.array([0.0, 0.25, -0.25], dtype=np.float32)

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_audio_tts",
        task="audio_tts",
        library="mlx_audio",
        spec=make_model_spec(repo_id="fake"),
        audio={"sample_rate": 22050},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=FakeRawTTS(), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize("say it"))

    assert output.text == "Generated audio at 22050 Hz."
    assert output.audio["sample_rate"] == 22050
    assert tuple(output.audio["waveform"].shape) == (1, 1, 3)


def test_mlx_embeddings_json_cap_is_enforced():
    class FakeBackend:
        generation_lock = contextlib_null()

    class FakeEmbeddings:
        def process(self, inputs, processor=None, **kwargs):
            assert inputs == [{"text": "embed me"}]
            return {"embedding": list(range(128))}

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_embeddings",
        task="embeddings",
        library="mlx_embeddings",
        spec=make_model_spec(repo_id="fake"),
        embeddings={"max_output_json_bytes": 16},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=FakeEmbeddings(), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    with pytest.raises(ValueError, match="max_output_json_bytes"):
        adapter.generate(adapter.tokenize("embed me"))


def test_mlx_embeddings_fallback_uses_mlx_embeddings_generate(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

    class FakeArray:
        shape = (2, 2)

        def tolist(self):
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakeEmbeddingsModule:
        @staticmethod
        def generate(model, processor, texts, images=None, max_length=512):
            assert model == "model"
            assert processor == "processor"
            assert texts == ["cat", "dog"]
            assert images is None
            assert max_length == 32
            return types.SimpleNamespace(text_embeds=FakeArray())

    monkeypatch.setattr(mlx_module, "_import_dependency", lambda module, extra: FakeEmbeddingsModule)

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_embeddings",
        task="embeddings",
        library="mlx_embeddings",
        spec=make_model_spec(repo_id="fake"),
        embeddings={"max_length": 32},
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="model", processor="processor")
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(adapter.tokenize('[{"text": "cat"}, {"text": "dog"}]'))

    assert output.text
    assert json.loads(output.text) == {"shape": [2, 2], "embeddings": [[0.1, 0.2], [0.3, 0.4]]}


def test_mlx_audio_asr_uses_initial_prompt(monkeypatch):
    class FakeBackend:
        generation_lock = contextlib_null()

    class FakeASR:
        def generate(self, audio_path, **kwargs):
            assert audio_path == "audio.wav"
            assert kwargs["initial_prompt"] == "domain hint"
            return types.SimpleNamespace(text="transcribed")

    monkeypatch.setattr(mlx_module, "_audio_to_temp_wav", lambda audio: "audio.wav")
    monkeypatch.setattr(mlx_module, "_remove_temp_file", lambda path: None)

    manifest = MLXManifest(
        manifest_path="fake.mlx",
        loader_type="mlx_audio_asr",
        task="audio_asr",
        library="mlx_audio",
        spec=make_model_spec(repo_id="fake"),
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model=FakeASR(), processor=None)
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    output = adapter.generate(MLXGenerationRequest(prompt="domain hint", audio={}))

    assert output.text == "transcribed"


def test_mlx_vlm_samples_video_frames():
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL.Image")
    video = torch.zeros((48, 2, 2, 3), dtype=torch.float32)
    request = MLXGenerationRequest(prompt="describe", video=video)

    images = _collect_vlm_images(request, {"frame_stride": 24, "max_frames": 1})

    assert len(images) == 1
    assert images[0].size == (2, 2)


def test_mlx_vlm_loads_and_caches_mtp_drafter_from_generation_manifest(monkeypatch):
    class FakeDrafter:
        config = types.SimpleNamespace(block_size=6)

        def __init__(self):
            self.accept_lens = []

    class FakeBackend:
        generation_lock = contextlib_null()

        def memory_stats(self):
            return types.SimpleNamespace(stats={})

    class FakeProcessor:
        def encode(self, text):
            return [1, 2]

    class FakeMLXVLM:
        calls = []

        @staticmethod
        def generate(model, processor, prompt, image=None, audio=None, **kwargs):
            assert model == "target"
            assert isinstance(kwargs["draft_model"], FakeDrafter)
            assert kwargs["draft_kind"] == "mtp"
            assert kwargs["draft_block_size"] == 6
            FakeMLXVLM.calls.append(kwargs["draft_model"])
            kwargs["draft_model"].accept_lens = [4]
            return types.SimpleNamespace(text="drafted reply", generation_tokens=5)

    class FakeDrafters:
        load_calls = []

        @staticmethod
        def load_drafter(path_or_repo, kind=None):
            FakeDrafters.load_calls.append((path_or_repo, kind))
            assert path_or_repo == "mlx-community/gemma-4-E4B-it-assistant-bf16"
            assert kind == "mtp"
            return FakeDrafter(), "mtp"

    def fake_import(module, extra):
        if module == "mlx_vlm":
            return FakeMLXVLM
        if module == "mlx_vlm.speculative.drafters":
            return FakeDrafters
        raise AssertionError(module)

    events = []
    monkeypatch.setattr(mlx_module, "_import_dependency", fake_import)
    monkeypatch.setattr(mlx_module, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))
    manifest = MLXManifest(
        manifest_path="gemma4.mlx",
        loader_type="mlx_vlm",
        task="vlm",
        library="mlx_vlm",
        spec=make_model_spec(repo_id="fake"),
        generation={
            "draft_model": "mlx-community/gemma-4-E4B-it-assistant-bf16",
            "draft_kind": "mtp",
            "draft_block_size": 6,
        },
    )
    handle = MLXModelHandle(backend="mlx", spec=manifest.spec, model="target", processor=FakeProcessor(), modality="vlm")
    adapter = MLXRuntimeAdapter(FakeBackend(), manifest, handle)

    first = adapter.generate(adapter.tokenize("hello"), do_sample=False, max_length=8)
    second = adapter.generate(adapter.tokenize("hello again"), do_sample=False, max_length=8)

    assert first.text == "drafted reply"
    assert second.metadata["generation_mode"] == "mlx_vlm_mtp_draft"
    assert second.metadata["assistant_model"] == "mlx-community/gemma-4-E4B-it-assistant-bf16"
    assert second.metadata["assistant_kind"] == "mtp"
    assert second.metadata["speculative_depth"] == 6
    assert second.metadata["output_tokens"] == 5
    assert second.metadata["speculative_rounds"] == 1
    assert second.metadata["max_draft_tokens_per_round"] == 5
    assert second.metadata["accepted_drafts"] == 4
    assert second.metadata["drafted_tokens"] == 4
    assert second.metadata["acceptance_rate"] == 1.0
    assert FakeDrafters.load_calls == [("mlx-community/gemma-4-E4B-it-assistant-bf16", "mtp")]
    assert [event["event_type"] for event in events if event["event_type"].startswith("draft_load_")] == ["draft_load_start", "draft_load_end"]


def test_mlx_backend_unload_clears_handle_without_mlx_installed():
    backend = MLXBackend()
    handle = MLXModelHandle(backend="mlx", spec=make_model_spec(repo_id="fake"), model=object(), processor=object())

    backend.unload(handle)

    assert handle.model is None
    assert handle.processor is None


def test_mlx_backend_interrupt_check_is_best_effort(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)

    MLXBackend()._throw_if_interrupted()


def test_text_generate_keeps_text_output_and_passes_optional_audio():
    from comfy_extras.nodes_textgen import TextGenerate

    audio = {"waveform": object(), "sample_rate": 24000}

    class FakeClip:
        def tokenize(self, prompt, **kwargs):
            return {"prompt": prompt, "audio": kwargs.get("audio")}

        def generate(self, tokens, **kwargs):
            return MLXGenerationOutput(f"reply: {tokens['prompt']}", audio=audio)

        def decode(self, generated):
            return generated.text

        def decode_audio(self, generated):
            return generated.audio

    result = TextGenerate.execute(
        FakeClip(),
        "hello",
        32,
        {"sampling_mode": "off"},
    )

    assert result.args[0] == "reply: hello"
    assert result.args[1] is audio


def test_text_generate_defaults_sampling_mode_when_api_omits_dynamic_combo():
    from comfy_extras.nodes_textgen import TextGenerate

    class FakeClip:
        def tokenize(self, prompt, **kwargs):
            return prompt

        def generate(self, tokens, **kwargs):
            assert kwargs["do_sample"] is False
            return tokens

        def decode(self, generated):
            return generated

    result = TextGenerate.execute(FakeClip(), "hello", 8)

    assert result.args[0] == "hello"


def test_text_generate_errors_on_non_generating_clip():
    from comfy_extras.nodes_textgen import TextGenerate

    with pytest.raises(TypeError, match="tokenize/generate/decode"):
        TextGenerate.execute(object(), "hello", 32, {"sampling_mode": "off"})


class contextlib_null:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
