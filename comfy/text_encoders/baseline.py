from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Any

from comfy.backends import BackendUnavailableError, ModelSpec, get_backend
from comfy.backends.benchmark_stats import redact_secrets
from comfy.backends.gguf_backend import GGUFModelSpec
from comfy.backends.transformers_backend import TransformersModelSpec


BASELINE_LOADER_TYPES = ("gguf_lm", "transformers_lm")
BASELINE_BACKENDS = {
    "gguf_lm": "gguf",
    "transformers_lm": "transformers",
}
_TOKEN_KEYS = {"token", "hf_token", "access_token", "auth_token", "private_token", "hf_api_token", "authorization", "api_key", "secret"}


@dataclass(frozen=True)
class BaselineManifest:
    manifest_path: str
    loader_type: str
    backend_name: str
    task: str
    spec: ModelSpec
    cache_dir: str | None = None
    load: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    assistant: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineGenerationRequest:
    prompt: str
    skip_template: bool = False


@dataclass(frozen=True)
class BaselineGenerationOutput:
    text: str
    stats_json: str = ""


def is_baseline_loader_type(loader_type: str) -> bool:
    return loader_type in BASELINE_LOADER_TYPES


def load_baseline_clip(manifest_path: str, loader_type: str, model_options: dict[str, Any] | None = None) -> BaselineRuntimeAdapter:
    manifest = load_manifest(manifest_path, loader_type)
    backend = get_backend(manifest.backend_name)
    status = backend.capabilities()
    if not status.available:
        raise BackendUnavailableError(status.reason)

    if model_options and model_options.get("load_device") is not None:
        manifest = _with_cpu_device(manifest)

    handle = backend.load_llm(manifest.spec, cache_dir=manifest.cache_dir)
    return BaselineRuntimeAdapter(backend, manifest, handle)


def load_manifest(manifest_path: str, loader_type: str) -> BaselineManifest:
    if loader_type not in BASELINE_LOADER_TYPES:
        raise ValueError(f"Unknown benchmark loader type: {loader_type}")
    if not _is_baseline_manifest_path(manifest_path):
        raise ValueError("Benchmark CLIPLoader types expect a .gguf.json, .transformers.json, or .baseline.json manifest.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not read benchmark manifest {manifest_path}: {redact_secrets(e)}") from e

    if not isinstance(raw, dict):
        raise ValueError("Benchmark manifests must be JSON objects.")
    _reject_inline_tokens(raw)

    manifest_backend = str(raw.get("backend", raw.get("type", loader_type))).strip().lower()
    if manifest_backend != loader_type:
        raise ValueError(f"Benchmark manifest backend/type {manifest_backend!r} does not match loader type {loader_type!r}.")

    task = str(raw.get("task") or raw.get("modality") or "lm").strip().lower()
    if task not in {"lm", "llm", "text"}:
        raise ValueError(f"Unsupported benchmark manifest task: {task}")

    repo_id = str(raw.get("repo_id") or raw.get("model") or "").strip()
    local_path = _normalize_local_path(raw.get("local_path"), manifest_path)
    if repo_id and local_path:
        raise ValueError("Benchmark manifest should not set both repo_id/model and local_path.")
    if not repo_id and not local_path:
        raise ValueError("Benchmark manifest needs repo_id or local_path.")

    metadata = {
        "manifest_path": manifest_path,
        "loader_type": loader_type,
        "backend": BASELINE_BACKENDS[loader_type],
        "model_id": str(raw.get("model_id") or raw.get("id") or os.path.basename(manifest_path)),
        "comparison_group": str(raw.get("comparison_group") or ""),
        "family": str(raw.get("family") or ""),
        "size_b": raw.get("size_b"),
        "baseline_class": str(raw.get("baseline_class") or ""),
        "fairness_notes": str(raw.get("fairness_notes") or ""),
        "quantization": str(raw.get("quantization") or ""),
        "fair_comparison": bool(raw.get("fair_comparison", False)),
        "filename": _none_if_blank(raw.get("filename") or raw.get("gguf_filename")),
        "required_files": raw.get("required_files"),
        "context_length": raw.get("context_length"),
        "trust_remote_code": bool(raw.get("trust_remote_code", False)),
        "load": _dict_or_empty(raw.get("load")),
        "tokenizer": _dict_or_empty(raw.get("tokenizer")),
        "assistant": _dict_or_empty(raw.get("assistant")),
    }
    metadata = {key: value for key, value in metadata.items() if value not in ("", None)}

    spec_cls = GGUFModelSpec if loader_type == "gguf_lm" else TransformersModelSpec
    spec = spec_cls(
        repo_id=repo_id or local_path,
        revision=_none_if_blank(raw.get("revision")),
        local_path=local_path,
        modality="lm",
        allow_patterns=_as_tuple(raw.get("allow_patterns")),
        ignore_patterns=_as_tuple(raw.get("ignore_patterns")),
        local_files_only=bool(raw.get("local_files_only", False)),
        use_hf_token=bool(raw.get("use_hf_token", False)),
        metadata=metadata,
    )

    cache_dir = raw.get("cache_dir")
    if cache_dir:
        cache_dir = _normalize_cache_dir(str(cache_dir), loader_type)

    return BaselineManifest(
        manifest_path=manifest_path,
        loader_type=loader_type,
        backend_name=BASELINE_BACKENDS[loader_type],
        task="lm",
        spec=spec,
        cache_dir=cache_dir,
        load=_dict_or_empty(raw.get("load")),
        tokenizer=_dict_or_empty(raw.get("tokenizer")),
        generation=_dict_or_empty(raw.get("generation")),
        assistant=_dict_or_empty(raw.get("assistant")),
        benchmark=_dict_or_empty(raw.get("benchmark")),
    )


class BaselineRuntimeAdapter:
    def __init__(self, backend, manifest: BaselineManifest, handle) -> None:
        self.backend = backend
        self.manifest = manifest
        self.handle = handle
        self.supports_textgenerate = True
        self.supports_conditioning = False
        self.is_benchmark_runtime = True
        self.benchmark_backend = manifest.backend_name

    def tokenize(self, text, return_word_ids=False, **kwargs):
        return BaselineGenerationRequest(
            prompt=text,
            skip_template=bool(kwargs.get("skip_template", False)),
        )

    def generate(
        self,
        tokens,
        do_sample=True,
        max_length=256,
        temperature=0.7,
        top_k=64,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.05,
        presence_penalty=0.0,
        seed=None,
    ):
        request = tokens if isinstance(tokens, BaselineGenerationRequest) else BaselineGenerationRequest(prompt=str(tokens))
        result = self.backend.generate_text(
            self.handle,
            request.prompt,
            max_tokens=int(max_length),
            do_sample=bool(do_sample),
            temperature=float(temperature if do_sample else 0.0),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=seed,
            use_chat_template=not request.skip_template,
            generation_options=self.manifest.generation,
        )
        return BaselineGenerationOutput(text=result.text, stats_json=getattr(result, "stats_json", ""))

    def decode(self, generated, skip_special_tokens=True):
        if isinstance(generated, BaselineGenerationOutput):
            return generated.text
        if isinstance(generated, str):
            return generated
        return getattr(generated, "text", str(generated))

    def decode_audio(self, generated):
        return None

    def clone(self):
        return self

    def add_patches(self, *args, **kwargs):
        return ()

    def encode_from_tokens(self, *args, **kwargs):
        raise RuntimeError("This benchmark runtime adapter is only compatible with TextGenerate, not conditioning nodes.")

    def encode_from_tokens_scheduled(self, *args, **kwargs):
        raise RuntimeError("This benchmark runtime adapter is only compatible with TextGenerate, not conditioning nodes.")

    def unload(self) -> None:
        self.backend.unload(self.handle)
        gc.collect()


def _with_cpu_device(manifest: BaselineManifest) -> BaselineManifest:
    raw_load = dict(manifest.load)
    if manifest.loader_type == "transformers_lm":
        raw_load["device"] = "cpu"
    spec = type(manifest.spec)(
        repo_id=manifest.spec.repo_id,
        revision=manifest.spec.revision,
        local_path=manifest.spec.local_path,
        modality=manifest.spec.modality,
        allow_patterns=manifest.spec.allow_patterns,
        ignore_patterns=manifest.spec.ignore_patterns,
        local_files_only=manifest.spec.local_files_only,
        use_hf_token=manifest.spec.use_hf_token,
        metadata={**manifest.spec.metadata, "load": raw_load},
    )
    return BaselineManifest(
        manifest_path=manifest.manifest_path,
        loader_type=manifest.loader_type,
        backend_name=manifest.backend_name,
        task=manifest.task,
        spec=spec,
        cache_dir=manifest.cache_dir,
        load=raw_load,
        tokenizer=manifest.tokenizer,
        generation=manifest.generation,
        assistant=manifest.assistant,
        benchmark=manifest.benchmark,
    )


def _normalize_local_path(value, manifest_path: str) -> str | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    value = os.path.expanduser(value)
    if not os.path.isabs(value):
        value = os.path.join(os.path.dirname(manifest_path), value)
    return os.path.abspath(value)


def _normalize_cache_dir(value: str, loader_type: str) -> str:
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return value
    import folder_paths

    folder_name = "gguf" if loader_type == "gguf_lm" else "transformers"
    return os.path.join(folder_paths.get_folder_paths(folder_name)[0], value)


def _none_if_blank(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _as_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace("\n", ",").split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _dict_or_empty(value) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reject_inline_tokens(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _TOKEN_KEYS:
                raise ValueError("Benchmark manifests must not contain token values; use use_hf_token instead.")
            _reject_inline_tokens(item)
    elif isinstance(value, list):
        for item in value:
            _reject_inline_tokens(item)


def _is_baseline_manifest_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".gguf.json") or lower.endswith(".transformers.json") or lower.endswith(".baseline.json")
