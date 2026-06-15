from __future__ import annotations

import gc
import glob
import hashlib
import os
import platform
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from .base import BackendCapabilities, BackendUnavailableError, MemoryStats, ModelSpec, RuntimeHandle
from .benchmark_stats import elapsed_seconds, now, process_memory, redact_secrets, write_event


def _default_cache_dir() -> str:
    import folder_paths

    root = folder_paths.get_folder_paths("gguf")[0]
    return os.path.join(root, "huggingface")


@dataclass(frozen=True)
class GGUFModelSpec(ModelSpec):
    pass


@dataclass(eq=False)
class GGUFModelHandle(RuntimeHandle):
    loaded_at: float = field(default_factory=time.time)
    generation_count: int = 0


@dataclass(frozen=True)
class GGUFGenerationResult:
    text: str
    stats_json: str


class GGUFBackend:
    name = "gguf"

    def __init__(self) -> None:
        self._generation_lock = threading.RLock()
        self._handles: weakref.WeakSet[GGUFModelHandle] = weakref.WeakSet()

    @property
    def generation_lock(self):
        return self._generation_lock

    def register_handle(self, handle: GGUFModelHandle) -> GGUFModelHandle:
        self._handles.add(handle)
        return handle

    def capabilities(self) -> BackendCapabilities:
        try:
            self._import_llama_cpp()
        except BackendUnavailableError as e:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=str(e),
                modalities=("llm",),
                device=self._device_label(),
                unified_memory=platform.system() == "Darwin",
            )
        return BackendCapabilities(
            name=self.name,
            available=True,
            reason="llama-cpp-python is importable.",
            modalities=("llm",),
            device=self._device_label(),
            unified_memory=platform.system() == "Darwin",
        )

    def _device_label(self) -> str:
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            return "apple_silicon_metal_if_built"
        return platform.machine().lower() or "unknown"

    def _import_llama_cpp(self):
        try:
            import llama_cpp
        except Exception as e:
            raise BackendUnavailableError(
                "llama-cpp-python is not installed. Install the optional GGUF benchmark extra and ensure Metal support is enabled on macOS."
            ) from e
        return llama_cpp

    def resolve_model_path(self, spec: GGUFModelSpec, cache_dir: str | None = None) -> str:
        if spec.local_path:
            path = os.path.abspath(os.path.expanduser(spec.local_path))
            if not os.path.exists(path):
                raise BackendUnavailableError(f"GGUF local_path does not exist: {path}")
            return path

        repo_or_path = os.path.expanduser(spec.repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        filename = str(spec.metadata.get("filename") or spec.metadata.get("gguf_filename") or "").strip()
        allow_patterns = spec.allow_patterns or ((filename,) if filename else ("*.gguf",))

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise BackendUnavailableError("huggingface_hub is required to download GGUF benchmark models.") from e

        try:
            snapshot = snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision or None,
                cache_dir=cache_dir or _default_cache_dir(),
                local_files_only=spec.local_files_only,
                allow_patterns=allow_patterns,
                ignore_patterns=spec.ignore_patterns or None,
                token=True if spec.use_hf_token else None,
            )
        except Exception as e:
            raise BackendUnavailableError(redact_secrets(e)) from e

        if filename:
            path = os.path.join(snapshot, filename)
            if os.path.exists(path):
                return path
            matches = glob.glob(os.path.join(snapshot, "**", filename), recursive=True)
            if matches:
                return matches[0]
            raise BackendUnavailableError(f"GGUF file {filename!r} was not found in downloaded snapshot.")

        matches = sorted(glob.glob(os.path.join(snapshot, "**", "*.gguf"), recursive=True))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise BackendUnavailableError("Downloaded snapshot did not contain a GGUF file.")
        raise BackendUnavailableError("GGUF manifest must specify filename when a snapshot contains multiple GGUF files.")

    def load_llm(self, spec: GGUFModelSpec, cache_dir: str | None = None) -> GGUFModelHandle:
        llama_cpp = self._import_llama_cpp()
        model_path = self.resolve_model_path(spec, cache_dir=cache_dir)
        load_options = dict(spec.metadata.get("load") or {})
        load_options.setdefault("n_ctx", int(spec.metadata.get("context_length") or 2048))
        load_options.setdefault("verbose", False)
        if platform.system() == "Darwin":
            load_options.setdefault("n_gpu_layers", -1)
        gpu_offload_requested = int(load_options.get("n_gpu_layers") or 0) != 0
        metal_available = self._supports_gpu_offload(llama_cpp)
        if platform.system() == "Darwin" and gpu_offload_requested and metal_available is False:
            raise BackendUnavailableError(
                "llama-cpp-python is importable but was built without GPU/Metal offload support; rebuild it with Metal before benchmarking GGUF."
            )
        metadata = self._metadata(spec, load_options, metal_available, gpu_offload_requested)

        start = now()
        write_event(
            "load_start",
            backend=self.name,
            model=metadata,
            resolved_path=model_path,
            metal_available=metal_available,
            gpu_offload_requested=gpu_offload_requested,
            n_gpu_layers=load_options.get("n_gpu_layers"),
        )
        try:
            model = llama_cpp.Llama(model_path=model_path, **load_options)
        except Exception as e:
            write_event("load_error", backend=self.name, model=metadata, error=redact_secrets(e))
            raise BackendUnavailableError(redact_secrets(e)) from e

        handle = GGUFModelHandle(
            backend=self.name,
            spec=spec,
            model=model,
            processor=None,
            resolved_path=model_path,
            modality="llm",
        )
        write_event(
            "load_end",
            backend=self.name,
            model=metadata,
            resolved_path=model_path,
            seconds=elapsed_seconds(start),
            memory=self.memory_stats().stats,
            metal_available=metal_available,
            gpu_offload_requested=gpu_offload_requested,
            n_gpu_layers=load_options.get("n_gpu_layers"),
        )
        return self.register_handle(handle)

    def generate_text(
        self,
        handle: GGUFModelHandle,
        prompt: str,
        max_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 64,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        use_chat_template: bool = False,
        generation_options: dict[str, Any] | None = None,
    ) -> GGUFGenerationResult:
        if handle.model is None:
            raise BackendUnavailableError("GGUF model handle has been unloaded.")

        options = dict(generation_options or {})
        options.update(
            {
                "max_tokens": int(max_tokens),
                "temperature": float(temperature if do_sample else 0.0),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repeat_penalty": float(repetition_penalty),
            }
        )
        if seed is not None and seed >= 0:
            options["seed"] = int(seed)

        prompt_tokens = self._count_tokens(handle.model, prompt)
        template_mode = "chat_template" if use_chat_template else "raw"
        prompt_sha256 = _prompt_sha256(prompt)
        start = now()
        write_event(
            "generate_start",
            backend=self.name,
            model=self._metadata(handle.spec, {}, None, None),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            prompt_sha256=prompt_sha256,
            template_mode=template_mode,
            stream=False,
        )
        with self._generation_lock:
            try:
                if use_chat_template:
                    result = handle.model.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        stream=False,
                        **options,
                    )
                    text = result["choices"][0]["message"].get("content", "")
                else:
                    result = handle.model.create_completion(prompt=prompt, stream=False, **options)
                    text = result["choices"][0].get("text", "")
            except Exception as e:
                write_event(
                    "generate_error",
                    backend=self.name,
                    model=self._metadata(handle.spec, {}, None, None),
                    prompt_sha256=prompt_sha256,
                    template_mode=template_mode,
                    stream=False,
                    error=redact_secrets(e),
                )
                raise BackendUnavailableError(redact_secrets(e)) from e

        handle.generation_count += 1
        output_tokens, token_count_source = self._completion_token_stats(result, text, handle.model)
        seconds = elapsed_seconds(start)
        stats = {
            "backend": self.name,
            "modality": handle.modality,
            "repo_id": handle.spec.repo_id,
            "revision": handle.spec.revision,
            "resolved_path": handle.resolved_path,
            "device": self._device_label(),
            "seconds": seconds,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "token_count_source": token_count_source,
            "tokens_per_second": round(output_tokens / seconds, 4) if seconds > 0 and output_tokens else None,
            "prompt_sha256": prompt_sha256,
            "template_mode": template_mode,
            "stream": False,
            "generation_mode": "non_stream",
            "characters": len(text),
            "generations": handle.generation_count,
            "memory": self.memory_stats().stats,
        }
        event_stats = dict(stats)
        event_stats.pop("backend", None)
        write_event("generate_end", backend=self.name, model=self._metadata(handle.spec, {}, None, None), **event_stats)
        import json

        return GGUFGenerationResult(text=text, stats_json=json.dumps(stats, indent=2, sort_keys=True))

    def _supports_gpu_offload(self, llama_cpp) -> bool | None:
        for target in (llama_cpp, getattr(llama_cpp, "llama_cpp", None)):
            func = getattr(target, "llama_supports_gpu_offload", None)
            if callable(func):
                try:
                    return bool(func())
                except Exception:
                    pass
        for target in (llama_cpp, getattr(llama_cpp, "llama_cpp", None)):
            func = getattr(target, "llama_print_system_info", None)
            if callable(func):
                try:
                    info = func()
                    if isinstance(info, bytes):
                        info = info.decode("utf-8", errors="replace")
                    text = str(info).lower()
                    if "metal" in text:
                        return "metal = 1" in text or "metal: yes" in text or "metal enabled" in text
                except Exception:
                    pass
        return None

    def _metadata(self, spec: GGUFModelSpec, load_options: dict[str, Any], metal_available: bool | None, gpu_offload_requested: bool | None) -> dict[str, Any]:
        metadata = dict(spec.metadata)
        metadata.update(
            {
                "backend_device": self._device_label(),
                "metal_available": metal_available,
                "gpu_offload_requested": gpu_offload_requested,
                "n_gpu_layers": load_options.get("n_gpu_layers"),
            }
        )
        return {key: value for key, value in metadata.items() if value not in ("", None)}

    def _count_tokens(self, model, text: str) -> int | None:
        try:
            return len(model.tokenize(text.encode("utf-8"), add_bos=True))
        except Exception:
            return None

    def _completion_token_stats(self, result: dict[str, Any], text: str, model) -> tuple[int, str]:
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
            return int(usage["completion_tokens"]), "llama_cpp_usage_completion_tokens"
        count = self._count_tokens(model, text)
        return int(count or 0), "fallback_tokenizer_after_generation"

    def unload(self, handle: GGUFModelHandle) -> None:
        handle.model = None
        handle.processor = None
        gc.collect()

    def unload_all(self) -> None:
        for handle in list(self._handles):
            self.unload(handle)

    def empty_cache(self) -> None:
        gc.collect()

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(backend=self.name, stats=process_memory())


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()
