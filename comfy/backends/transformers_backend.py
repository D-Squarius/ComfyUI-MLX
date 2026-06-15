from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from .base import BackendCapabilities, BackendUnavailableError, MemoryStats, ModelSpec, RuntimeHandle
from .benchmark_stats import elapsed_seconds, now, process_memory, redact_secrets, write_event


def _default_cache_dir() -> str:
    import folder_paths

    root = folder_paths.get_folder_paths("transformers")[0]
    return os.path.join(root, "huggingface")


@dataclass(frozen=True)
class TransformersModelSpec(ModelSpec):
    pass


@dataclass(eq=False)
class TransformersModelHandle(RuntimeHandle):
    loaded_at: float = field(default_factory=time.time)
    generation_count: int = 0
    device: str = "unknown"
    assistant_model: Any | None = None
    assistant_model_id: str = ""
    assistant_kind: str = ""


@dataclass(frozen=True)
class TransformersGenerationResult:
    text: str
    stats_json: str


class TransformersBackend:
    name = "transformers"

    def __init__(self) -> None:
        self._generation_lock = threading.RLock()
        self._handles: weakref.WeakSet[TransformersModelHandle] = weakref.WeakSet()

    @property
    def generation_lock(self):
        return self._generation_lock

    def register_handle(self, handle: TransformersModelHandle) -> TransformersModelHandle:
        self._handles.add(handle)
        return handle

    def capabilities(self) -> BackendCapabilities:
        try:
            self._import_transformers()
        except BackendUnavailableError as e:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=str(e),
                modalities=("llm",),
                device="unknown",
                unified_memory=False,
            )
        return BackendCapabilities(
            name=self.name,
            available=True,
            reason="transformers is importable.",
            modalities=("llm",),
            device=self._select_device("auto"),
            unified_memory=self._select_device("auto") == "mps",
        )

    def _import_transformers(self):
        try:
            import transformers
        except Exception as e:
            raise BackendUnavailableError("transformers is not installed.") from e
        return transformers

    def resolve_model_path(self, spec: TransformersModelSpec, cache_dir: str | None = None) -> str:
        if spec.local_path:
            path = os.path.abspath(os.path.expanduser(spec.local_path))
            if not os.path.exists(path):
                raise BackendUnavailableError(f"Transformers local_path does not exist: {path}")
            return path

        repo_or_path = os.path.expanduser(spec.repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise BackendUnavailableError("huggingface_hub is required to download Transformers benchmark models.") from e

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

    def resolve_assistant_model_path(
        self,
        assistant: dict[str, Any],
        spec: TransformersModelSpec,
        cache_dir: str | None = None,
    ) -> str:
        local_path = assistant.get("local_path")
        if local_path:
            path = os.path.abspath(os.path.expanduser(str(local_path)))
            if not os.path.exists(path):
                raise BackendUnavailableError(f"Transformers assistant local_path does not exist: {path}")
            return path

        repo_id = str(assistant.get("repo_id") or assistant.get("model") or "").strip()
        if not repo_id:
            raise BackendUnavailableError("Transformers assistant config requires repo_id, model, or local_path.")

        repo_or_path = os.path.expanduser(repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise BackendUnavailableError("huggingface_hub is required to download Transformers assistant models.") from e

        try:
            return snapshot_download(
                repo_id=repo_id,
                revision=str(assistant.get("revision") or spec.revision or "") or None,
                cache_dir=cache_dir or _default_cache_dir(),
                local_files_only=bool(assistant.get("local_files_only", spec.local_files_only)),
                allow_patterns=_tuple_or_none(assistant.get("allow_patterns")),
                ignore_patterns=_tuple_or_none(assistant.get("ignore_patterns")),
                token=True if bool(assistant.get("use_hf_token", spec.use_hf_token)) else None,
            )
        except Exception as e:
            raise BackendUnavailableError(redact_secrets(e)) from e

    def load_llm(self, spec: TransformersModelSpec, cache_dir: str | None = None) -> TransformersModelHandle:
        transformers = self._import_transformers()
        model_path = self.resolve_model_path(spec, cache_dir=cache_dir)
        load_options = dict(spec.metadata.get("load") or {})
        tokenizer_options = dict(spec.metadata.get("tokenizer") or {})
        assistant_options = dict(spec.metadata.get("assistant") or {})
        trust_remote_code = bool(spec.metadata.get("trust_remote_code", load_options.pop("trust_remote_code", False)))
        device = self._select_device(str(load_options.pop("device", "auto")))
        dtype_name = str(load_options.pop("dtype", load_options.pop("torch_dtype", "auto")))
        torch_dtype = self._resolve_torch_dtype(dtype_name)
        torch = self._import_torch()

        start = now()
        metadata = self._metadata(spec, device=device, dtype_name=dtype_name)
        write_event("load_start", backend=self.name, model=metadata, resolved_path=model_path, torch_device=device, torch_dtype=dtype_name)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
                **tokenizer_options,
            )
            model_kwargs = dict(load_options)
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
                **model_kwargs,
            )
            if not getattr(model, "hf_device_map", None):
                model.to(device)
            model.eval()
            assistant_model = None
            assistant_model_id = ""
            assistant_kind = ""
            if assistant_options:
                assistant_model_id = str(
                    assistant_options.get("repo_id")
                    or assistant_options.get("model")
                    or assistant_options.get("local_path")
                    or ""
                )
                assistant_path = self.resolve_assistant_model_path(assistant_options, spec, cache_dir=cache_dir)
                assistant_load_options = self._assistant_load_options(assistant_options)
                assistant_trust_remote_code = bool(assistant_options.get("trust_remote_code", trust_remote_code))
                assistant_dtype_name = str(assistant_options.get("dtype", assistant_options.get("torch_dtype", dtype_name)))
                assistant_dtype = self._resolve_torch_dtype(assistant_dtype_name)
                if assistant_dtype is not None:
                    assistant_load_options["torch_dtype"] = assistant_dtype
                assistant_cls, detected_kind = self._assistant_model_class(
                    transformers,
                    assistant_path,
                    trust_remote_code=assistant_trust_remote_code,
                )
                assistant_kind = str(assistant_options.get("assistant_kind") or assistant_options.get("kind") or detected_kind or "")
                assistant_start = now()
                write_event(
                    "assistant_load_start",
                    backend=self.name,
                    model=metadata,
                    assistant_model=assistant_model_id,
                    assistant_kind=assistant_kind,
                    resolved_path=assistant_path,
                    torch_device=device,
                    torch_dtype=assistant_dtype_name,
                )
                assistant_model = assistant_cls.from_pretrained(
                    assistant_path,
                    trust_remote_code=assistant_trust_remote_code,
                    local_files_only=True,
                    **assistant_load_options,
                )
                if not getattr(assistant_model, "hf_device_map", None):
                    assistant_model.to(device)
                assistant_model.eval()
                write_event(
                    "assistant_load_end",
                    backend=self.name,
                    model=metadata,
                    assistant_model=assistant_model_id,
                    assistant_kind=assistant_kind,
                    resolved_path=assistant_path,
                    seconds=elapsed_seconds(assistant_start),
                    memory=self.memory_stats().stats,
                    torch_device=device,
                    torch_dtype=assistant_dtype_name,
                )
        except Exception as e:
            write_event("load_error", backend=self.name, model=metadata, error=redact_secrets(e))
            raise BackendUnavailableError(redact_secrets(e)) from e

        handle = TransformersModelHandle(
            backend=self.name,
            spec=spec,
            model=model,
            processor=tokenizer,
            resolved_path=model_path,
            modality="llm",
            device=device,
            assistant_model=assistant_model,
            assistant_model_id=assistant_model_id,
            assistant_kind=assistant_kind,
        )
        write_event(
            "load_end",
            backend=self.name,
            model=metadata,
            resolved_path=model_path,
            seconds=elapsed_seconds(start),
            memory=self.memory_stats().stats,
            torch_device=device,
            torch_dtype=dtype_name,
            assistant_model=assistant_model_id,
            assistant_kind=assistant_kind,
            mps_available=self._mps_available(torch),
        )
        return self.register_handle(handle)

    def generate_text(
        self,
        handle: TransformersModelHandle,
        prompt: str,
        max_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 64,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        use_chat_template: bool = True,
        generation_options: dict[str, Any] | None = None,
    ) -> TransformersGenerationResult:
        if handle.model is None or handle.processor is None:
            raise BackendUnavailableError("Transformers model handle has been unloaded.")

        torch = self._import_torch()
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))

        prompt_text = self._format_prompt(handle.processor, prompt, use_chat_template)
        template_mode = "chat_template" if use_chat_template and prompt_text != prompt else "raw"
        prompt_sha256 = _prompt_sha256(prompt_text)
        inputs = handle.processor(prompt_text, return_tensors="pt")
        inputs = {key: value.to(handle.device) for key, value in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        options = self._generation_options(generation_options)
        options.update(
            {
                "max_new_tokens": int(max_tokens),
                "do_sample": bool(do_sample),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "pad_token_id": handle.processor.eos_token_id,
            }
        )
        if do_sample:
            options["temperature"] = float(temperature)
        generation_mode = "non_stream"
        assistant_enabled = False
        assistant_policy = str(options.pop("assistant_sampling_policy", "fallback")).strip().lower()
        strict_option = bool(options.pop("strict", False))
        speculative_tokens = options.pop("max_speculative_tokens", None)
        if handle.assistant_model is not None:
            strict_assistant = bool(strict_option or (handle.spec.metadata.get("assistant") or {}).get("strict", False))
            if do_sample:
                if strict_assistant or assistant_policy in {"error", "strict"}:
                    raise BackendUnavailableError("Transformers Gemma 4 assistant MTP currently supports greedy TextGenerate only.")
                generation_mode = "transformers_ar_sampling_fallback"
            else:
                if speculative_tokens is not None:
                    options.setdefault("num_assistant_tokens", int(speculative_tokens))
                options["assistant_model"] = handle.assistant_model
                generation_mode = "transformers_gemma4_assistant_mtp"
                assistant_enabled = True

        self._synchronize_device(torch, handle.device)
        start = now()
        write_event(
            "generate_start",
            backend=self.name,
            model=self._metadata(handle.spec, device=handle.device, dtype_name=str(handle.spec.metadata.get("load", {}).get("dtype", ""))),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            prompt_sha256=prompt_sha256,
            template_mode=template_mode,
            stream=False,
            torch_device=handle.device,
            generation_mode=generation_mode,
            assistant_model=handle.assistant_model_id,
            assistant_kind=handle.assistant_kind,
            assistant_enabled=assistant_enabled,
        )
        with self._generation_lock:
            try:
                inference_context = getattr(torch, "inference_mode", torch.no_grad)
                with inference_context():
                    output_ids = handle.model.generate(**inputs, **options)
                self._synchronize_device(torch, handle.device)
            except Exception as e:
                write_event(
                    "generate_error",
                    backend=self.name,
                    model=self._metadata(handle.spec, device=handle.device, dtype_name=str(handle.spec.metadata.get("load", {}).get("dtype", ""))),
                    prompt_sha256=prompt_sha256,
                    template_mode=template_mode,
                    stream=False,
                    error=redact_secrets(e),
                )
                raise BackendUnavailableError(redact_secrets(e)) from e

        generated = output_ids[0][prompt_tokens:]
        text = handle.processor.decode(generated, skip_special_tokens=True)
        output_tokens = int(generated.shape[-1])
        seconds = elapsed_seconds(start)
        handle.generation_count += 1
        stats = {
            "backend": self.name,
            "modality": handle.modality,
            "repo_id": handle.spec.repo_id,
            "revision": handle.spec.revision,
            "resolved_path": handle.resolved_path,
            "device": handle.device,
            "torch_device": handle.device,
            "torch_dtype": str(handle.spec.metadata.get("load", {}).get("dtype", "")),
            "seconds": seconds,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "token_count_source": "generated_tensor_length",
            "tokens_per_second": round(output_tokens / seconds, 4) if seconds > 0 and output_tokens else None,
            "prompt_sha256": prompt_sha256,
            "template_mode": template_mode,
            "stream": False,
            "generation_mode": generation_mode,
            "assistant_model": handle.assistant_model_id,
            "assistant_kind": handle.assistant_kind,
            "assistant_enabled": assistant_enabled,
            "acceptance_rate": None,
            "accepted_drafts": None,
            "drafted_tokens": None,
            "characters": len(text),
            "generations": handle.generation_count,
            "memory": self.memory_stats().stats,
        }
        event_stats = dict(stats)
        event_stats.pop("backend", None)
        write_event(
            "generate_end",
            backend=self.name,
            model=self._metadata(handle.spec, device=handle.device, dtype_name=str(handle.spec.metadata.get("load", {}).get("dtype", ""))),
            mps_available=self._mps_available(torch),
            mps_synchronized=handle.device == "mps",
            cuda_synchronized=handle.device.startswith("cuda"),
            **event_stats,
        )
        return TransformersGenerationResult(text=text, stats_json=json.dumps(stats, indent=2, sort_keys=True))

    def _format_prompt(self, tokenizer, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return prompt

    def _import_torch(self):
        try:
            import torch
        except Exception as e:
            raise BackendUnavailableError("torch is required for the Transformers benchmark backend.") from e
        return torch

    def _select_device(self, requested: str) -> str:
        requested = requested.strip().lower()
        if requested and requested != "auto":
            return requested
        try:
            torch = self._import_torch()
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _metadata(self, spec: TransformersModelSpec, device: str, dtype_name: str) -> dict[str, Any]:
        metadata = dict(spec.metadata)
        assistant = metadata.get("assistant") if isinstance(metadata.get("assistant"), dict) else {}
        metadata.update(
            {
                "backend_device": device,
                "torch_device": device,
                "torch_dtype": dtype_name or "auto",
                "mps_requested": device == "mps",
                "assistant_model": assistant.get("repo_id") or assistant.get("model") or assistant.get("local_path") or "",
                "assistant_kind": assistant.get("assistant_kind") or assistant.get("kind") or "",
            }
        )
        return metadata

    def _assistant_load_options(self, assistant: dict[str, Any]) -> dict[str, Any]:
        load_options = dict(assistant.get("load") or {})
        for key, value in assistant.items():
            if key not in {
                "repo_id",
                "model",
                "local_path",
                "revision",
                "allow_patterns",
                "ignore_patterns",
                "local_files_only",
                "use_hf_token",
                "assistant_kind",
                "kind",
                "device",
                "dtype",
                "torch_dtype",
                "trust_remote_code",
                "strict",
                "load",
            }:
                load_options[key] = value
        return load_options

    def _assistant_model_class(self, transformers, model_path: str, trust_remote_code: bool):
        detected_kind = ""
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            detected_kind = str(getattr(config, "model_type", "") or "")
            architectures = getattr(config, "architectures", None) or []
            if detected_kind == "gemma4_assistant" or "Gemma4AssistantForCausalLM" in architectures:
                assistant_cls = getattr(transformers, "Gemma4AssistantForCausalLM", None)
                if assistant_cls is not None:
                    return assistant_cls, "gemma4_assistant"
        except Exception:
            pass
        return transformers.AutoModelForCausalLM, detected_kind

    def _generation_options(self, generation_options: dict[str, Any] | None) -> dict[str, Any]:
        options = dict(generation_options or {})
        unsupported_policy = options.pop("unsupported_sampling_policy", None)
        if unsupported_policy is not None:
            options.setdefault("assistant_sampling_policy", unsupported_policy)
        for key in (
            "assistant",
            "assistant_model",
            "draft_model",
            "draft_kind",
            "generation_mode",
            "runtime_mode",
            "profile",
            "verify_strategy",
        ):
            options.pop(key, None)
        return options

    def _mps_available(self, torch) -> bool:
        try:
            return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            return False

    def _synchronize_device(self, torch, device: str) -> None:
        if device.startswith("cuda"):
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available() and hasattr(torch.cuda, "synchronize"):
                    torch.cuda.synchronize()
            except Exception:
                pass
            return
        if device != "mps":
            return
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    def _resolve_torch_dtype(self, name: str):
        name = name.strip().lower()
        if name in {"", "auto", "none"}:
            return None
        torch = self._import_torch()
        aliases = {
            "fp16": "float16",
            "float16": "float16",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp8": "float8_e4m3fn",
            "float8": "float8_e4m3fn",
            "float8_e4m3fn": "float8_e4m3fn",
            "float8_e5m2": "float8_e5m2",
            "float8_e4m3fnuz": "float8_e4m3fnuz",
            "float8_e5m2fnuz": "float8_e5m2fnuz",
            "fp32": "float32",
            "float32": "float32",
        }
        attr = aliases.get(name, name)
        if not hasattr(torch, attr):
            raise BackendUnavailableError(f"Unsupported Transformers torch dtype: {name}")
        return getattr(torch, attr)

    def unload(self, handle: TransformersModelHandle) -> None:
        handle.model = None
        handle.processor = None
        handle.assistant_model = None
        gc.collect()
        try:
            torch = self._import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) is not None and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def unload_all(self) -> None:
        for handle in list(self._handles):
            self.unload(handle)

    def empty_cache(self) -> None:
        gc.collect()
        try:
            torch = self._import_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) is not None and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def memory_stats(self) -> MemoryStats:
        stats = process_memory()
        try:
            torch = self._import_torch()
            if torch.cuda.is_available():
                stats["cuda_memory_allocated"] = int(torch.cuda.memory_allocated())
                stats["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                if hasattr(torch.mps, "current_allocated_memory"):
                    stats["mps_current_allocated_memory"] = int(torch.mps.current_allocated_memory())
                if hasattr(torch.mps, "driver_allocated_memory"):
                    stats["mps_driver_allocated_memory"] = int(torch.mps.driver_allocated_memory())
        except Exception as e:
            stats["transformers_memory_unavailable"] = redact_secrets(e)
        return MemoryStats(backend=self.name, stats=stats)


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()


def _tuple_or_none(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace("\n", ",").split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())
