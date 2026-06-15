import json
import sys
import types

import pytest

import comfy.backends
import comfy.backends.gguf_backend as gguf_module
import comfy.text_encoders.baseline as baseline_module
from comfy.backends.benchmark_stats import event_enabled, redact_secrets
from comfy.backends.gguf_backend import GGUFBackend, GGUFModelHandle
from comfy.backends.transformers_backend import TransformersBackend, TransformersModelHandle
from comfy.text_encoders.baseline import (
    BaselineGenerationOutput,
    BaselineManifest,
    BaselineRuntimeAdapter,
    load_baseline_clip,
    load_manifest,
)


def test_builtin_benchmark_backends_do_not_import_optional_runtime_modules():
    sys.modules.pop("llama_cpp", None)

    backends = {backend.name for backend in comfy.backends.list_backends()}

    assert {"gguf", "mlx", "transformers"}.issubset(backends)
    assert "llama_cpp" not in sys.modules


def test_gguf_manifest_parses_strict_benchmark_metadata(tmp_path):
    manifest_path = tmp_path / "qwen.gguf.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "gguf_lm",
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "revision": "main",
                "quantization": "Q4_K_M",
                "comparison_group": "qwen-4bit",
                "family": "Qwen2.5 Instruct",
                "size_b": 0.5,
                "baseline_class": "strict_quantized",
                "fairness_notes": "closest Q4 counterpart",
                "fair_comparison": True,
                "local_files_only": True,
                "load": {"n_ctx": 2048},
                "generation": {"stop": ["<|im_end|>"]},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path), "gguf_lm")

    assert manifest.backend_name == "gguf"
    assert manifest.spec.repo_id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    assert manifest.spec.metadata["filename"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    assert manifest.spec.metadata["quantization"] == "Q4_K_M"
    assert manifest.spec.metadata["comparison_group"] == "qwen-4bit"
    assert manifest.spec.metadata["family"] == "Qwen2.5 Instruct"
    assert manifest.spec.metadata["size_b"] == 0.5
    assert manifest.spec.metadata["baseline_class"] == "strict_quantized"
    assert manifest.spec.metadata["fairness_notes"] == "closest Q4 counterpart"
    assert manifest.spec.metadata["fair_comparison"] is True
    assert manifest.load == {"n_ctx": 2048}
    assert manifest.generation == {"stop": ["<|im_end|>"]}


def test_transformers_manifest_parses_loader_options(tmp_path):
    manifest_path = tmp_path / "qwen.transformers.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "transformers_lm",
                "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
                "revision": "main",
                "quantization": "GPTQ-Int4",
                "comparison_group": "qwen-4bit",
                "trust_remote_code": True,
                "load": {"device": "mps", "dtype": "float16"},
                "tokenizer": {"use_fast": True},
                "assistant": {
                    "repo_id": "google/gemma-4-E2B-it-assistant",
                    "assistant_kind": "gemma4_assistant",
                    "strict": True,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(str(manifest_path), "transformers_lm")

    assert manifest.backend_name == "transformers"
    assert manifest.spec.metadata["trust_remote_code"] is True
    assert manifest.spec.metadata["load"] == {"device": "mps", "dtype": "float16"}
    assert manifest.spec.metadata["tokenizer"] == {"use_fast": True}
    assert manifest.spec.metadata["assistant"]["assistant_kind"] == "gemma4_assistant"
    assert manifest.assistant["strict"] is True


def test_baseline_manifest_rejects_inline_tokens(tmp_path):
    manifest_path = tmp_path / "bad.gguf.json"
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    manifest_path.write_text(
        f'{{"backend": "gguf_lm", "repo_id": "test/model", "token": "{fake_hf_token}"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain token"):
        load_manifest(str(manifest_path), "gguf_lm")


def test_load_baseline_clip_uses_core_backend_loader(monkeypatch, tmp_path):
    manifest_path = tmp_path / "qwen.gguf.json"
    manifest_path.write_text(
        '{"backend": "gguf_lm", "repo_id": "Qwen/test", "filename": "model.gguf"}',
        encoding="utf-8",
    )

    class FakeBackend:
        def __init__(self):
            self.loaded = []

        def capabilities(self):
            return types.SimpleNamespace(available=True, reason="")

        def load_llm(self, spec, cache_dir=None):
            self.loaded.append((spec, cache_dir))
            return types.SimpleNamespace(model="model", processor=None)

        def generate_text(self, handle, prompt, **kwargs):
            return types.SimpleNamespace(text=f"reply: {prompt}", stats_json="{}")

        def unload(self, handle):
            handle.model = None

    fake_backend = FakeBackend()
    monkeypatch.setattr(baseline_module, "get_backend", lambda name: fake_backend)

    adapter = load_baseline_clip(str(manifest_path), "gguf_lm")
    output = adapter.generate(adapter.tokenize("hello"), do_sample=False, max_length=8)

    assert adapter.supports_textgenerate is True
    assert fake_backend.loaded[0][0].metadata["filename"] == "model.gguf"
    assert isinstance(output, BaselineGenerationOutput)
    assert adapter.decode(output) == "reply: hello"


def test_baseline_adapter_rejects_conditioning_use():
    manifest = BaselineManifest(
        manifest_path="fake.gguf.json",
        loader_type="gguf_lm",
        backend_name="gguf",
        task="lm",
        spec=types.SimpleNamespace(metadata={}),
    )
    adapter = BaselineRuntimeAdapter(types.SimpleNamespace(unload=lambda handle: None), manifest, object())

    with pytest.raises(RuntimeError, match="TextGenerate"):
        adapter.encode_from_tokens({})


def test_gguf_snapshot_resolution_uses_filename(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    expected = snapshot / "model.gguf"
    expected.write_text("fake", encoding="utf-8")
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(snapshot)

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    spec = baseline_module.GGUFModelSpec(
        repo_id="test/repo",
        revision="main",
        metadata={"filename": "model.gguf"},
    )

    resolved = GGUFBackend().resolve_model_path(spec, cache_dir=str(tmp_path / "cache"))

    assert resolved == str(expected)
    assert calls["allow_patterns"] == ("model.gguf",)
    assert calls["local_files_only"] is False


def test_gguf_backend_reports_gpu_offload_support():
    fake_llama_cpp = types.SimpleNamespace(llama_supports_gpu_offload=lambda: True)

    assert GGUFBackend()._supports_gpu_offload(fake_llama_cpp) is True


def test_transformers_backend_synchronizes_mps_timing():
    calls = []
    fake_torch = types.SimpleNamespace(mps=types.SimpleNamespace(synchronize=lambda: calls.append("sync")))

    TransformersBackend()._synchronize_device(fake_torch, "mps")

    assert calls == ["sync"]


def test_transformers_backend_synchronizes_cuda_timing():
    calls = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("sync"),
        )
    )

    TransformersBackend()._synchronize_device(fake_torch, "cuda")

    assert calls == ["sync"]


def test_transformers_backend_accepts_float8_dtype_alias(monkeypatch):
    fake_torch = types.SimpleNamespace(float8_e4m3fn=object())
    backend = TransformersBackend()
    monkeypatch.setattr(backend, "_import_torch", lambda: fake_torch)

    assert backend._resolve_torch_dtype("fp8") is fake_torch.float8_e4m3fn


def test_transformers_backend_passes_assistant_model_to_greedy_generate():
    torch = pytest.importorskip("torch")
    assistant = object()
    calls = []

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "templated prompt"

        def __call__(self, text, return_tensors="pt"):
            assert text == "templated prompt"
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded"

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.tensor([[1, 2, 3, 4]])

    spec = baseline_module.TransformersModelSpec(
        repo_id="google/gemma-4-E2B-it",
        revision="main",
        metadata={
            "load": {"dtype": "float16"},
            "assistant": {"repo_id": "google/gemma-4-E2B-it-assistant", "assistant_kind": "gemma4_assistant"},
        },
    )
    handle = TransformersModelHandle(
        backend="transformers",
        spec=spec,
        model=FakeModel(),
        processor=FakeTokenizer(),
        resolved_path="/tmp/target",
        modality="llm",
        device="cpu",
        assistant_model=assistant,
        assistant_model_id="google/gemma-4-E2B-it-assistant",
        assistant_kind="gemma4_assistant",
    )

    result = TransformersBackend().generate_text(
        handle,
        "hello",
        max_tokens=8,
        do_sample=False,
        generation_options={"generation_mode": "mtp", "max_speculative_tokens": 4},
    )
    stats = json.loads(result.stats_json)

    assert calls[0]["assistant_model"] is assistant
    assert calls[0]["num_assistant_tokens"] == 4
    assert "generation_mode" not in calls[0]
    assert result.text == "decoded"
    assert stats["generation_mode"] == "transformers_gemma4_assistant_mtp"
    assert stats["assistant_enabled"] is True


def test_transformers_backend_rejects_strict_sampled_assistant_generation():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1]])}

    spec = baseline_module.TransformersModelSpec(
        repo_id="target",
        metadata={"assistant": {"repo_id": "assistant", "strict": True}},
    )
    handle = TransformersModelHandle(
        backend="transformers",
        spec=spec,
        model=object(),
        processor=FakeTokenizer(),
        resolved_path="/tmp/target",
        modality="llm",
        device="cpu",
        assistant_model=object(),
        assistant_model_id="assistant",
        assistant_kind="gemma4_assistant",
    )

    with pytest.raises(baseline_module.BackendUnavailableError, match="greedy"):
        TransformersBackend().generate_text(handle, "hello", do_sample=True)


def test_gguf_generation_reports_prompt_template_metadata(monkeypatch):
    class FakeModel:
        def tokenize(self, data, add_bos=True):
            return [1, 2, 3]

        def create_completion(self, prompt, stream=False, **options):
            assert prompt == "raw prompt"
            assert stream is False
            return {"choices": [{"text": "one two"}], "usage": {"completion_tokens": 2}}

    events = []
    monkeypatch.setattr(gguf_module, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))
    spec = baseline_module.GGUFModelSpec(repo_id="test/repo", revision="main", metadata={"model_id": "fake"})
    handle = GGUFModelHandle(backend="gguf", spec=spec, model=FakeModel(), processor=None, resolved_path="/tmp/fake.gguf", modality="llm")

    result = GGUFBackend().generate_text(handle, "raw prompt", max_tokens=8, do_sample=False, use_chat_template=False)

    assert result.text == "one two"
    end_event = next(event for event in events if event["event_type"] == "generate_end")
    assert end_event["template_mode"] == "raw"
    assert end_event["stream"] is False
    assert end_event["generation_mode"] == "non_stream"
    assert end_event["output_tokens"] == 2
    assert end_event["token_count_source"] == "llama_cpp_usage_completion_tokens"
    assert len(end_event["prompt_sha256"]) == 64


def test_benchmark_stats_redacts_tokens():
    fake_hf_token = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    text = redact_secrets(f"failed token={fake_hf_token}")

    assert fake_hf_token not in text


def test_benchmark_stats_event_filter(monkeypatch):
    monkeypatch.delenv("COMFY_BENCH_DISABLED_EVENTS", raising=False)
    assert event_enabled("mlx_denoiser_island_call")

    monkeypatch.setenv("COMFY_BENCH_DISABLED_EVENTS", "mlx_denoiser_island_call,other")
    assert not event_enabled("mlx_denoiser_island_call")
    assert event_enabled("generate_end")

    monkeypatch.setenv("COMFY_BENCH_DISABLED_EVENTS", "all")
    assert not event_enabled("generate_end")
