from __future__ import annotations

import gc
import importlib
import inspect
import json
import logging
import os
import platform
import re
import threading
import time
import weakref
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import psutil

from .base import BackendCapabilities, BackendUnavailableError, MemoryStats, ModelSpec, RuntimeHandle

HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_\-]{8,}")
AUTHORIZATION_BEARER_RE = re.compile(r"(?i)\bauthorization\s*[:=]?\s*bearer\s+([A-Za-z0-9._\-+/=]{8,})")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|access_token|authorization|bearer)\b(\s*[:=]\s*|\s+)([A-Za-z0-9._\-+/=]{8,})"
)


def redact_secrets(value: Any) -> str:
    text = str(value)
    text = HF_TOKEN_RE.sub("hf_***", text)
    text = AUTHORIZATION_BEARER_RE.sub("Authorization Bearer ***", text)
    return SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _split_patterns(patterns: str | Iterable[str] | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        values = []
        for part in patterns.replace("\n", ",").split(","):
            part = part.strip()
            if part:
                values.append(part)
        return tuple(values)
    return tuple(p for p in patterns if p)


def _call_with_supported_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(*args, **kwargs)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(*args, **supported)


def _filter_mlx_lm_stream_kwargs(mlx_lm, kwargs: dict[str, Any]) -> dict[str, Any]:
    generate_step = _find_mlx_lm_callable(mlx_lm, "generate_step")
    if generate_step is None:
        filtered = dict(kwargs)
        filtered.pop("verbose", None)
        return filtered

    signature = inspect.signature(generate_step)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)

    allowed = set(signature.parameters)
    filtered = dict(kwargs)
    if "sampler" in allowed and "sampler" not in filtered:
        make_sampler = _find_mlx_lm_callable(mlx_lm, "make_sampler")
        if make_sampler is not None:
            sampler_kwargs = {
                key: filtered.pop(key)
                for key in ("temp", "top_p", "min_p", "top_k", "min_tokens_to_keep")
                if key in filtered
            }
            filtered["sampler"] = _call_with_supported_kwargs(make_sampler, **sampler_kwargs)
    return {key: value for key, value in filtered.items() if key == "max_tokens" or key in allowed}


def _find_mlx_lm_callable(mlx_lm, name: str):
    candidate = getattr(mlx_lm, name, None)
    if callable(candidate):
        return candidate
    for attr in ("generate", "utils"):
        module = getattr(mlx_lm, attr, None)
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    for module_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            module = importlib.import_module(module_name)
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
        except Exception:
            pass
    return None


def _default_cache_dir() -> str:
    import folder_paths

    root = folder_paths.get_folder_paths("mlx")[0]
    return os.path.join(root, "huggingface")


@dataclass(frozen=True)
class MLXModelSpec(ModelSpec):
    pass


@dataclass(eq=False)
class MLXModelHandle(RuntimeHandle):
    loaded_at: float = field(default_factory=time.time)
    generation_count: int = 0


@dataclass(frozen=True)
class MLXGenerationResult:
    text: str
    stats_json: str


class MLXBackend:
    name = "mlx"

    def __init__(self) -> None:
        self._generation_lock = threading.RLock()
        self._handles: weakref.WeakSet[MLXModelHandle] = weakref.WeakSet()

    @property
    def generation_lock(self):
        return self._generation_lock

    def register_handle(self, handle: MLXModelHandle) -> MLXModelHandle:
        self._handles.add(handle)
        return handle

    def capabilities(self) -> BackendCapabilities:
        if not is_apple_silicon():
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason="MLX backend requires Apple Silicon macOS.",
                modalities=("llm", "vlm", "tts", "asr", "embeddings"),
                device="unsupported",
                unified_memory=False,
            )
        try:
            self._import_mlx_core()
        except BackendUnavailableError as e:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=str(e),
                modalities=("llm", "vlm", "tts", "asr", "embeddings"),
                device="apple_silicon",
                unified_memory=True,
            )
        return BackendCapabilities(
            name=self.name,
            available=True,
            reason="MLX core is importable on Apple Silicon.",
            modalities=("llm", "vlm", "tts", "asr", "embeddings"),
            device="apple_silicon",
            unified_memory=True,
        )

    def _import_mlx_core(self):
        if not is_apple_silicon():
            raise BackendUnavailableError("MLX backend requires Apple Silicon macOS.")
        try:
            import mlx.core as mx
        except Exception as e:
            raise BackendUnavailableError(
                "MLX is not installed. Install an optional MLX extra such as `pip install -e .[mlx-llm]`."
            ) from e
        return mx

    def _import_mlx_lm(self):
        self._import_mlx_core()
        try:
            import mlx_lm
        except Exception as e:
            raise BackendUnavailableError(
                "mlx-lm is not installed. Install it with `pip install -e .[mlx-llm]`."
            ) from e
        return mlx_lm

    def resolve_model_path(
        self,
        spec: MLXModelSpec,
        cache_dir: str | None = None,
    ) -> str:
        if spec.local_path:
            return os.path.abspath(os.path.expanduser(spec.local_path))
        repo_or_path = os.path.expanduser(spec.repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise BackendUnavailableError(
                "huggingface_hub is not installed. Install it with `pip install huggingface_hub`."
            ) from e

        try:
            return snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision or None,
                cache_dir=cache_dir or _default_cache_dir(),
                local_files_only=spec.local_files_only,
                allow_patterns=spec.allow_patterns or None,
                ignore_patterns=spec.ignore_patterns or None,
                token=True if spec.use_hf_token else None,
            )
        except Exception as e:
            raise BackendUnavailableError(redact_secrets(e)) from e

    def load_llm(self, spec: MLXModelSpec, cache_dir: str | None = None) -> MLXModelHandle:
        mlx_lm = self._import_mlx_lm()
        model_path = self.resolve_model_path(spec, cache_dir=cache_dir)
        try:
            load = getattr(mlx_lm, "load")
            loaded = _call_with_supported_kwargs(load, model_path)
            if not isinstance(loaded, tuple) or len(loaded) < 2:
                raise BackendUnavailableError("mlx_lm.load did not return a model/tokenizer tuple.")
            model, tokenizer = loaded[0], loaded[1]
            handle = MLXModelHandle(
                backend=self.name,
                spec=spec,
                model=model,
                processor=tokenizer,
                resolved_path=model_path,
                modality="llm",
            )
            return self.register_handle(handle)
        except BackendUnavailableError:
            raise
        except Exception as e:
            raise BackendUnavailableError(redact_secrets(e)) from e

    def generate_text(
        self,
        handle: MLXModelHandle,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        use_chat_template: bool = True,
    ) -> MLXGenerationResult:
        if handle.model is None or handle.processor is None:
            raise BackendUnavailableError("MLX model handle has been unloaded.")

        mlx_lm = self._import_mlx_lm()
        formatted_prompt = self._format_prompt(handle.processor, prompt, use_chat_template)
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "verbose": False,
        }

        if seed is not None and seed >= 0:
            self._seed(seed)

        start = time.perf_counter()
        text_parts: list[str] = []
        with self._generation_lock:
            try:
                stream_generate = getattr(mlx_lm, "stream_generate", None)
                if stream_generate is not None:
                    stream_kwargs = _filter_mlx_lm_stream_kwargs(mlx_lm, generation_kwargs)
                    stream = _call_with_supported_kwargs(
                        stream_generate,
                        handle.model,
                        handle.processor,
                        prompt=formatted_prompt,
                        **stream_kwargs,
                    )
                    for chunk in stream:
                        self._throw_if_interrupted()
                        text_parts.append(getattr(chunk, "text", str(chunk)))
                else:
                    generate = getattr(mlx_lm, "generate")
                    generated = _call_with_supported_kwargs(
                        generate,
                        handle.model,
                        handle.processor,
                        prompt=formatted_prompt,
                        **generation_kwargs,
                    )
                    self._throw_if_interrupted()
                    text_parts.append(getattr(generated, "text", str(generated)))
            except BaseException as e:
                if e.__class__.__name__ == "InterruptProcessingException":
                    raise
                raise BackendUnavailableError(redact_secrets(e)) from e

        handle.generation_count += 1
        text = "".join(text_parts)
        stats = {
            "backend": self.name,
            "modality": handle.modality,
            "repo_id": handle.spec.repo_id,
            "revision": handle.spec.revision,
            "resolved_path": handle.resolved_path,
            "seconds": round(time.perf_counter() - start, 4),
            "characters": len(text),
            "generations": handle.generation_count,
            "memory": self.memory_stats().stats,
        }
        return MLXGenerationResult(text=text, stats_json=json.dumps(stats, indent=2, sort_keys=True))

    def _format_prompt(self, tokenizer, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            return prompt

    def _seed(self, seed: int) -> None:
        try:
            mx = self._import_mlx_core()
            if hasattr(mx, "random") and hasattr(mx.random, "seed"):
                mx.random.seed(seed)
        except Exception:
            pass

    def _throw_if_interrupted(self) -> None:
        try:
            import comfy.model_management

            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception as e:
            logging.debug("Could not check ComfyUI processing interruption state: %s", e)

    def unload(self, handle: MLXModelHandle) -> None:
        handle.model = None
        handle.processor = None
        gc.collect()
        self.empty_cache()

    def unload_all(self) -> None:
        for handle in list(self._handles):
            self.unload(handle)

    def empty_cache(self) -> None:
        try:
            mx = self._import_mlx_core()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                metal = getattr(mx, "metal", None)
                if metal is not None and hasattr(metal, "clear_cache"):
                    metal.clear_cache()
        except Exception:
            pass
        gc.collect()

    def memory_stats(self) -> MemoryStats:
        stats: dict[str, Any] = {
            "process_rss": psutil.Process(os.getpid()).memory_info().rss,
        }
        try:
            mx = self._import_mlx_core()
            for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
                if hasattr(mx, name):
                    stats[name.replace("get_", "")] = getattr(mx, name)()
            metal = getattr(mx, "metal", None)
            if metal is not None:
                for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
                    key = name.replace("get_", "")
                    if key not in stats and hasattr(metal, name):
                        stats[key] = getattr(metal, name)()
        except Exception as e:
            stats["mlx_unavailable"] = redact_secrets(e)
        return MemoryStats(backend=self.name, stats=stats)

    def status_json(self) -> str:
        capabilities = self.capabilities()
        payload = {
            "capabilities": asdict(capabilities),
            "memory": self.memory_stats().stats,
            "serialized_execution": True,
            "loaded_handles": len(list(self._handles)),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def make_model_spec(
    repo_id: str,
    revision: str = "",
    local_path: str = "",
    modality: str = "llm",
    allow_patterns: str | Iterable[str] | None = None,
    ignore_patterns: str | Iterable[str] | None = None,
    local_files_only: bool = False,
    use_hf_token: bool = False,
) -> MLXModelSpec:
    return MLXModelSpec(
        repo_id=repo_id.strip(),
        revision=revision.strip() or None,
        local_path=local_path.strip() or None,
        modality=modality,
        allow_patterns=_split_patterns(allow_patterns),
        ignore_patterns=_split_patterns(ignore_patterns),
        local_files_only=local_files_only,
        use_hf_token=use_hf_token,
    )
