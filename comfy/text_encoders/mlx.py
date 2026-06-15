from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import tempfile
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from comfy.backends import BackendUnavailableError, get_backend
from comfy.backends.benchmark_stats import elapsed_seconds, now, write_event
from comfy.backends.mlx_backend import MLXModelHandle, MLXModelSpec, redact_secrets


MLX_LOADER_TYPES = ("mlx_lm", "mlx_vlm", "mlx_audio_asr", "mlx_audio_tts", "mlx_embeddings")
MLX_LOADER_TASKS = {
    "mlx_lm": "lm",
    "mlx_vlm": "vlm",
    "mlx_audio_asr": "audio_asr",
    "mlx_audio_tts": "audio_tts",
    "mlx_embeddings": "embeddings",
}
MLX_TASK_LIBRARIES = {
    "lm": "mlx_lm",
    "vlm": "mlx_vlm",
    "audio_asr": "mlx_audio",
    "audio_tts": "mlx_audio",
    "embeddings": "mlx_embeddings",
    "rerank": "mlx_embeddings",
}
MLX_TASK_ALIASES = {
    "llm": "lm",
    "text": "lm",
    "tts": "audio_tts",
    "asr": "audio_asr",
    "stt": "audio_asr",
    "embedding": "embeddings",
}
_TOKEN_KEYS = {"token", "hf_token", "access_token", "auth_token", "private_token", "hf_api_token", "authorization", "api_key", "secret"}
_DEFAULT_MAX_JSON_BYTES = 1_048_576
_MTPLX_SUPPORTED_VERIFY_STRATEGIES = {"batched", "capture_commit", "graphbank", "graphbank_capture_commit"}
_MTPLX_SUPPORTED_RUNTIME_MODES = {"in_process", "http"}
_MTPLX_DEFAULT_VERIFY_CORE = "linear-gdn-from-conv-tape"
_MTPLX_DEFAULT_HIDDEN_VARIANT = "post_norm"
_MTPLX_DEFAULT_CACHE_POLICY = "persistent"
_MTPLX_DEFAULT_HISTORY_POLICY = "committed"
_MTPLX_DEFAULT_DRAFT_CORE = "stock"
_MTPLX_STATS_METADATA_KEYS = (
    "tok_s",
    "decode_elapsed_s",
    "decode_tok_s",
    "end_to_end_tok_s",
    "prompt_eval_time_s",
    "prompt_tps",
    "prompt_target_prefill_time_s",
    "prompt_mtp_history_time_s",
    "prompt_target_prefill_tok_s",
    "prompt_mtp_history_tok_s",
    "target_forward_time_s",
    "verify_time_s",
    "verify_forward_time_s",
    "verify_eval_time_s",
    "verify_logits_eval_time_s",
    "verify_hidden_eval_time_s",
    "verify_joint_eval_time_s",
    "verify_target_distribution_time_s",
    "draft_time_s",
    "mtp_history_policy",
    "mtp_history_window_tokens",
    "cached_tokens",
    "new_prefill_tokens",
    "session_cache_hit",
    "cache_miss_reason",
    "session_restore_mode",
    "snapshot_time_s",
    "accept_time_s",
    "rollback_time_s",
    "repair_time_s",
    "commit_time_s",
    "capture_commit_time_s",
    "clear_cache_events",
    "clear_cache_time_s",
    "speculative_depth",
    "requested_speculative_depth",
    "accepted_by_depth",
    "drafted_by_depth",
    "accept_probability_sum_by_depth",
    "mean_accept_probability_by_depth",
    "skipped_drafts",
    "bonus_tokens",
    "correction_tokens",
    "verify_calls",
    "peak_memory_bytes",
    "prefill_chunk_size",
    "prefill_chunks",
    "prefill_route",
    "runtime_mtp_enabled",
    "draft_head_installed",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "full_logits_tokens_emitted",
    "final_logits_tokens_emitted",
    "logits_tokens_emitted",
)
_MTPLX_OPTION_METADATA_KEYS = (
    "verify_strategy",
    "verify_core",
    "mtp_hidden_variant",
    "mtp_cache_policy",
    "mtp_history_policy",
    "draft_core",
    "draft_margin_threshold",
    "min_speculative_depth",
    "prompt_encoding",
)


@dataclass(frozen=True)
class MLXManifest:
    manifest_path: str
    loader_type: str
    task: str
    library: str
    spec: MLXModelSpec
    cache_dir: str | None = None
    load: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, Any] = field(default_factory=dict)


@dataclass
class MLXGenerationRequest:
    prompt: str
    skip_template: bool = False
    thinking: bool = False
    image: Any = None
    video: Any = None
    audio: Any = None


@dataclass
class MLXGenerationOutput:
    text: str
    audio: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLXPreparedLMRequest:
    prompt: str
    prompt_tokens: int | None
    kwargs: dict[str, Any]
    stream: bool
    template_mode: str
    prompt_sha256: str
    interrupt_poll_interval: int
    interrupt_poll_seconds: float


def is_mlx_loader_type(loader_type: str) -> bool:
    return loader_type in MLX_LOADER_TYPES


def load_mlx_clip(manifest_path: str, loader_type: str, model_options: dict[str, Any] | None = None) -> MLXRuntimeAdapter:
    manifest = load_manifest(manifest_path, loader_type)
    backend = get_backend("mlx")
    status = backend.capabilities()
    if not status.available:
        raise BackendUnavailableError(status.reason)

    if model_options and model_options.get("load_device") is not None:
        logging.warning("MLX CLIPLoader ignores the torch device override; MLX selects its own Apple Silicon device.")

    _throw_if_interrupted()
    model_path = backend.resolve_model_path(manifest.spec, cache_dir=manifest.cache_dir)
    _throw_if_interrupted()
    start = now()
    write_event("load_start", backend="mlx", model=_benchmark_metadata(manifest), resolved_path=model_path)
    handle = _load_handle(backend, manifest, model_path)
    write_event(
        "load_end",
        backend="mlx",
        model=_benchmark_metadata(manifest),
        resolved_path=model_path,
        seconds=elapsed_seconds(start),
        memory=_backend_memory_stats(backend),
    )
    _throw_if_interrupted()
    return MLXRuntimeAdapter(backend, manifest, handle)


def load_manifest(manifest_path: str, loader_type: str) -> MLXManifest:
    if loader_type not in MLX_LOADER_TASKS:
        raise ValueError(f"Unknown MLX loader type: {loader_type}")
    if not _is_mlx_manifest_path(manifest_path):
        raise ValueError("MLX CLIPLoader types expect a .mlx or .mlx.json manifest in models/text_encoders.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not read MLX manifest {manifest_path}: {redact_secrets(e)}") from e

    if not isinstance(raw, dict):
        raise ValueError("MLX manifests must be JSON objects.")
    _reject_inline_tokens(raw)

    manifest_backend = str(raw.get("backend", raw.get("type", "mlx"))).strip().lower()
    if manifest_backend not in {"mlx", loader_type}:
        raise ValueError(f"MLX manifest backend/type {manifest_backend!r} does not match loader type {loader_type!r}.")

    task = _normalize_task(raw.get("task") or raw.get("modality") or MLX_LOADER_TASKS[loader_type])
    if task != MLX_LOADER_TASKS[loader_type]:
        raise ValueError(f"MLX manifest task {task!r} does not match loader type {loader_type!r}.")
    if task not in set(MLX_TASK_LIBRARIES):
        raise ValueError(f"Unsupported MLX manifest task: {task}")
    library = str(raw.get("library") or MLX_TASK_LIBRARIES[task]).strip()

    repo_id = str(raw.get("repo_id") or raw.get("model") or "").strip()
    local_path = _normalize_local_path(raw.get("local_path"), manifest_path)
    if repo_id and local_path:
        raise ValueError("MLX manifest should not set both repo_id/model and local_path.")
    if not repo_id and not local_path:
        raise ValueError("MLX manifest needs repo_id or local_path.")

    spec = MLXModelSpec(
        repo_id=repo_id or local_path,
        revision=_none_if_blank(raw.get("revision")),
        local_path=local_path,
        modality=task,
        allow_patterns=_as_tuple(raw.get("allow_patterns")),
        ignore_patterns=_as_tuple(raw.get("ignore_patterns")),
        local_files_only=bool(raw.get("local_files_only", False)),
        use_hf_token=bool(raw.get("use_hf_token", False)),
        metadata={
            "manifest_path": manifest_path,
            "library": library,
            "model_id": str(raw.get("model_id") or raw.get("id") or os.path.basename(manifest_path)),
            "comparison_group": str(raw.get("comparison_group") or ""),
            "family": str(raw.get("family") or ""),
            "size_b": raw.get("size_b"),
            "baseline_class": str(raw.get("baseline_class") or ""),
            "fairness_notes": str(raw.get("fairness_notes") or ""),
            "quantization": str(raw.get("quantization") or ""),
            "fair_comparison": bool(raw.get("fair_comparison", False)),
        },
    )

    cache_dir = raw.get("cache_dir")
    if cache_dir:
        cache_dir = _normalize_cache_dir(str(cache_dir))

    return MLXManifest(
        manifest_path=manifest_path,
        loader_type=loader_type,
        task=task,
        library=library,
        spec=spec,
        cache_dir=cache_dir,
        load=_dict_or_empty(raw.get("load")),
        generation=_dict_or_empty(raw.get("generation")),
        audio=_dict_or_empty(raw.get("audio")),
        video=_dict_or_empty(raw.get("video")),
        embeddings=_dict_or_empty(raw.get("embeddings") or raw.get("runtime")),
    )


class MLXRuntimeAdapter:
    def __init__(self, backend, manifest: MLXManifest, handle: MLXModelHandle) -> None:
        self.backend = backend
        self.manifest = manifest
        self.handle = handle
        self.is_mlx_runtime = True
        self.mlx_modality = manifest.task
        self.supports_textgenerate = True
        self.supports_conditioning = False
        self._cached_mlx_lm = None
        self._cached_vlm_drafters = {}

    def tokenize(self, text, return_word_ids=False, **kwargs):
        return MLXGenerationRequest(
            prompt=text,
            skip_template=bool(kwargs.get("skip_template", False)),
            thinking=bool(kwargs.get("thinking", False)),
            image=kwargs.get("image"),
            video=kwargs.get("video"),
            audio=kwargs.get("audio"),
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
        request = tokens if isinstance(tokens, MLXGenerationRequest) else MLXGenerationRequest(prompt=str(tokens))
        preprocess_start = now()
        prepared_lm = None
        prompt_for_stats = request.prompt
        prompt_tokens = _token_count(self.handle.processor, prompt_for_stats)
        template_mode = "raw" if request.skip_template else "chat_template"
        stream = None
        prompt_sha256 = _prompt_sha256(prompt_for_stats)
        if self.manifest.task == "lm":
            prepared_lm = self._prepare_lm_request(request, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty)
            prompt_for_stats = prepared_lm.prompt
            prompt_tokens = prepared_lm.prompt_tokens
            template_mode = prepared_lm.template_mode
            stream = prepared_lm.stream
            prompt_sha256 = prepared_lm.prompt_sha256
        preprocess_seconds = elapsed_seconds(preprocess_start)

        lock_wait_start = now()
        with self.backend.generation_lock:
            lock_wait_seconds = elapsed_seconds(lock_wait_start)
            _throw_if_interrupted()
            if seed is not None and seed >= 0:
                self._seed(seed)
                if prepared_lm is not None:
                    prepared_lm.kwargs["seed"] = int(seed)
            write_event(
                "generate_start",
                backend="mlx",
                model=_benchmark_metadata(self.manifest),
                modality=self.manifest.task,
                prompt_tokens=prompt_tokens,
                max_tokens=max_length,
                preprocess_seconds=preprocess_seconds,
                lock_wait_seconds=lock_wait_seconds,
                prompt_sha256=prompt_sha256,
                template_mode=template_mode,
                stream=stream,
            )
            generation_start = now()
            try:
                if self.manifest.task == "lm":
                    output = self._generate_mtplx_lm(prepared_lm) if _is_mtplx_manifest(self.manifest) else self._generate_lm(prepared_lm)
                elif self.manifest.task == "vlm":
                    output = self._generate_vlm(request, do_sample, max_length, temperature, top_p)
                elif self.manifest.task == "audio_asr":
                    output = self._generate_audio_asr(request, do_sample, max_length, temperature)
                elif self.manifest.task == "audio_tts":
                    output = self._generate_audio_tts(request)
                elif self.manifest.task in {"embeddings", "rerank"}:
                    output = self._generate_embeddings(request)
                else:
                    raise BackendUnavailableError(f"Unsupported MLX task: {self.manifest.task}")
            except BaseException as e:
                write_event(
                    "generate_error",
                    backend="mlx",
                    model=_benchmark_metadata(self.manifest),
                    modality=self.manifest.task,
                    preprocess_seconds=preprocess_seconds,
                    lock_wait_seconds=lock_wait_seconds,
                    prompt_sha256=prompt_sha256,
                    template_mode=template_mode,
                    stream=stream,
                    error=redact_secrets(e),
                )
                raise
            generation_seconds = elapsed_seconds(generation_start)

        token_count_start = now()
        output_tokens = output.metadata.get("output_tokens") if output.metadata else None
        token_count_source = output.metadata.get("token_count_source") if output.metadata else None
        if output_tokens is None:
            output_tokens = _token_count(self.handle.processor, output.text)
            token_count_source = "fallback_tokenizer_after_generation" if output_tokens is not None else "unavailable"
        token_count_seconds = elapsed_seconds(token_count_start)
        first_token_seconds = output.metadata.get("first_token_seconds") if output.metadata else None
        write_event(
            "generate_end",
            backend="mlx",
            model=_benchmark_metadata(self.manifest),
            modality=self.manifest.task,
            seconds=generation_seconds,
            generation_seconds=generation_seconds,
            preprocess_seconds=preprocess_seconds,
            lock_wait_seconds=lock_wait_seconds,
            token_count_seconds=token_count_seconds,
            first_token_seconds=first_token_seconds,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            token_count_source=token_count_source,
            tokens_per_second=round(output_tokens / generation_seconds, 4) if generation_seconds > 0 and output_tokens else None,
            prompt_sha256=prompt_sha256,
            template_mode=template_mode,
            stream=stream,
            generation_mode=output.metadata.get("generation_mode") if output.metadata else None,
            **_generation_event_metadata(output.metadata if output.metadata else {}),
            characters=len(output.text),
            memory=_backend_memory_stats(self.backend),
        )
        return output

    def decode(self, generated, skip_special_tokens=True):
        if isinstance(generated, MLXGenerationOutput):
            return generated.text
        if isinstance(generated, str):
            return generated
        return getattr(generated, "text", str(generated))

    def decode_audio(self, generated):
        if isinstance(generated, MLXGenerationOutput):
            return generated.audio
        return None

    def clone(self):
        return self

    def add_patches(self, *args, **kwargs):
        return ()

    def encode_from_tokens(self, *args, **kwargs):
        raise RuntimeError("This MLX runtime adapter is only compatible with TextGenerate, not conditioning nodes.")

    def encode_from_tokens_scheduled(self, *args, **kwargs):
        raise RuntimeError("This MLX runtime adapter is only compatible with TextGenerate, not conditioning nodes.")

    def unload(self) -> None:
        self.backend.unload(self.handle)

    def _mlx_lm_module(self):
        if self._cached_mlx_lm is None:
            self._cached_mlx_lm = _import_dependency("mlx_lm", "mlx-llm")
        return self._cached_mlx_lm

    def _prepare_lm_request(self, request, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty=0.0):
        use_template = not request.skip_template
        prompt = _format_prompt(self.handle.processor, request.prompt, use_template)
        kwargs = dict(self.manifest.generation)
        stream = bool(kwargs.pop("stream", False) or kwargs.pop("measure_first_token", False))
        interrupt_poll_interval = max(1, int(kwargs.pop("interrupt_poll_interval", 8)))
        interrupt_poll_seconds = max(0.0, float(kwargs.pop("interrupt_poll_seconds", 0.05)))
        kwargs.update({
            "max_tokens": int(max_length),
            "temp": float(temperature if do_sample else 0.0),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
        })
        if repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        if _is_mtplx_manifest(self.manifest):
            if not do_sample or float(min_p) == 0.0:
                kwargs.pop("min_p", None)
            if presence_penalty != 0.0:
                kwargs["presence_penalty"] = float(presence_penalty)
            kwargs["stream"] = stream
            return MLXPreparedLMRequest(
                prompt=prompt,
                prompt_tokens=_token_count(self.handle.processor, prompt),
                kwargs=kwargs,
                stream=stream,
                template_mode="chat_template" if use_template and prompt != request.prompt else "raw",
                prompt_sha256=_prompt_sha256(prompt),
                interrupt_poll_interval=interrupt_poll_interval,
                interrupt_poll_seconds=interrupt_poll_seconds,
            )

        mlx_lm = self._mlx_lm_module()
        if stream:
            kwargs = _filter_mlx_lm_stream_kwargs(mlx_lm, kwargs)
        else:
            kwargs.setdefault("verbose", False)
            kwargs = _filter_mlx_lm_generate_kwargs(mlx_lm, kwargs)
        return MLXPreparedLMRequest(
            prompt=prompt,
            prompt_tokens=_token_count(self.handle.processor, prompt),
            kwargs=kwargs,
            stream=stream,
            template_mode="chat_template" if use_template and prompt != request.prompt else "raw",
            prompt_sha256=_prompt_sha256(prompt),
            interrupt_poll_interval=interrupt_poll_interval,
            interrupt_poll_seconds=interrupt_poll_seconds,
        )

    def _generate_lm(self, prepared: MLXPreparedLMRequest):
        mlx_lm = self._mlx_lm_module()
        if not prepared.stream:
            try:
                generate = getattr(mlx_lm, "generate")
                generated = _call_with_supported_kwargs(
                    generate,
                    self.handle.model,
                    self.handle.processor,
                    prompt=prepared.prompt,
                    **prepared.kwargs,
                )
                _throw_if_interrupted()
            except BaseException as e:
                _raise_unless_interrupted(e)
            if isinstance(generated, str):
                text = generated
            else:
                text = getattr(generated, "text", str(generated))
            return MLXGenerationOutput(
                text=text,
                metadata={
                    "first_token_seconds": None,
                    "output_tokens": None,
                    "token_count_source": "fallback_tokenizer_after_generation",
                    "generation_mode": "non_stream",
                },
            )

        parts = []
        stream_start = now()
        first_token_seconds = None
        chunk_count = 0
        token_count = 0
        saw_token_ids = True
        last_interrupt_check = stream_start
        try:
            stream = mlx_lm.stream_generate(self.handle.model, self.handle.processor, prompt=prepared.prompt, **prepared.kwargs)
            for chunk in stream:
                chunk_count += 1
                if first_token_seconds is None:
                    first_token_seconds = elapsed_seconds(stream_start)
                token = getattr(chunk, "token", None)
                if token is None:
                    saw_token_ids = False
                else:
                    token_count += 1
                parts.append(getattr(chunk, "text", str(chunk)))
                should_poll_by_count = chunk_count % prepared.interrupt_poll_interval == 0
                should_poll_by_time = elapsed_seconds(last_interrupt_check) >= prepared.interrupt_poll_seconds
                if should_poll_by_count or should_poll_by_time:
                    _throw_if_interrupted()
                    last_interrupt_check = now()
        except BaseException as e:
            _raise_unless_interrupted(e)
        if saw_token_ids:
            output_tokens = token_count
            token_count_source = "mlx_stream_chunk_token_ids"
        else:
            output_tokens = chunk_count
            token_count_source = "mlx_stream_chunk_count"
        return MLXGenerationOutput(
            text="".join(parts),
            metadata={
                "first_token_seconds": first_token_seconds,
                "output_tokens": output_tokens,
                "token_count_source": token_count_source,
                "generation_mode": "stream",
            },
        )

    def _generate_mtplx_lm(self, prepared: MLXPreparedLMRequest):
        runtime_mode = _normalize_mtplx_runtime_mode(prepared.kwargs.get("runtime_mode", "in_process"))
        if runtime_mode == "http":
            return _generate_mtplx_http(prepared, self.manifest)

        generation_mode = _normalize_mtplx_generation_mode(prepared.kwargs.get("generation_mode", "mtp"))
        unsupported_policy = str(prepared.kwargs.get("unsupported_sampling_policy", "error")).strip().lower()
        _validate_mtplx_sampling(prepared.kwargs, unsupported_policy)

        mtplx_generation = _import_dependency("mtplx.generation", "mtplx")
        mtplx_sampling = _import_dependency("mtplx.sampling", "mtplx")
        prompt_encoding = str(prepared.kwargs.get("prompt_encoding", "tokenizer")).strip().lower()
        prompt_ids = _mtplx_encode_prompt(self.handle.processor, prepared.prompt, prepared.kwargs)
        sampler = _mtplx_sampler_config(
            mtplx_sampling,
            temperature=prepared.kwargs.get("temp", prepared.kwargs.get("temperature", 0.0)),
            top_p=prepared.kwargs.get("top_p", 1.0),
            top_k=prepared.kwargs.get("top_k", 0),
        )
        draft_sampler = _mtplx_draft_sampler_config(mtplx_sampling, prepared.kwargs)
        stream_start = now()
        first_token_seconds = None
        callback_token_count = 0
        last_interrupt_check = stream_start

        def token_callback(token_ids):
            nonlocal first_token_seconds, callback_token_count, last_interrupt_check
            count = len(token_ids) if isinstance(token_ids, (list, tuple)) else 1
            callback_token_count += count
            if first_token_seconds is None and count:
                first_token_seconds = elapsed_seconds(stream_start)
            should_poll_by_count = callback_token_count % prepared.interrupt_poll_interval == 0
            should_poll_by_time = elapsed_seconds(last_interrupt_check) >= prepared.interrupt_poll_seconds
            if should_poll_by_count or should_poll_by_time:
                _throw_if_interrupted()
                last_interrupt_check = now()

        try:
            with _temporary_mtplx_profile(prepared.kwargs.get("profile")):
                if generation_mode == "ar":
                    generated = _call_with_supported_kwargs(
                        mtplx_generation.generate_ar,
                        self.handle.model,
                        prompt_ids,
                        max_tokens=int(prepared.kwargs["max_tokens"]),
                        sampler=sampler,
                        seed=int(prepared.kwargs.get("seed", 0)),
                        token_callback=token_callback,
                    )
                else:
                    verify_strategy = str(prepared.kwargs.get("verify_strategy", "capture_commit")).strip()
                    if verify_strategy not in _MTPLX_SUPPORTED_VERIFY_STRATEGIES:
                        raise ValueError(f"Unsupported MTPLX verify_strategy: {verify_strategy}")
                    verify_core = str(prepared.kwargs.get("verify_core", _MTPLX_DEFAULT_VERIFY_CORE)).strip()
                    mtp_hidden_variant = str(prepared.kwargs.get("mtp_hidden_variant", _MTPLX_DEFAULT_HIDDEN_VARIANT)).strip()
                    mtp_cache_policy = str(prepared.kwargs.get("mtp_cache_policy", _MTPLX_DEFAULT_CACHE_POLICY)).strip()
                    mtp_history_policy = str(prepared.kwargs.get("mtp_history_policy", _MTPLX_DEFAULT_HISTORY_POLICY)).strip()
                    draft_core = str(prepared.kwargs.get("draft_core", _MTPLX_DEFAULT_DRAFT_CORE)).strip()
                    generated = _call_with_supported_kwargs(
                        mtplx_generation.generate_mtpk,
                        self.handle.model,
                        prompt_ids,
                        max_tokens=int(prepared.kwargs["max_tokens"]),
                        sampler=sampler,
                        speculative_depth=int(prepared.kwargs.get("speculative_depth", 3)),
                        seed=int(prepared.kwargs.get("seed", 0)),
                        draft_sampler=draft_sampler,
                        mtp_hidden_variant=mtp_hidden_variant,
                        mtp_cache_policy=mtp_cache_policy,
                        mtp_history_policy=mtp_history_policy,
                        min_speculative_depth=int(prepared.kwargs.get("min_speculative_depth", 1)),
                        verify_strategy=verify_strategy,
                        verify_core=verify_core,
                        draft_core=draft_core,
                        draft_margin_threshold=_float_or_none(prepared.kwargs.get("draft_margin_threshold")),
                        token_callback=token_callback,
                    )
            _throw_if_interrupted()
        except BaseException as e:
            _raise_unless_interrupted(e)

        stats = _mtplx_stats_dict(generated)
        output_tokens = stats.get("generated_tokens") or callback_token_count or None
        text = _extract_text(generated)
        metadata = _mtplx_output_metadata(
            stats,
            generation_mode=generation_mode,
            first_token_seconds=first_token_seconds,
            output_tokens=output_tokens,
            runtime_mode=runtime_mode,
            profile=prepared.kwargs.get("profile"),
        )
        for key, value in _mtplx_generation_options_metadata(
            prepared.kwargs,
            generation_mode=generation_mode,
            prompt_encoding=prompt_encoding,
            draft_sampler=draft_sampler,
        ).items():
            metadata.setdefault(key, value)
        return MLXGenerationOutput(text=text, metadata=metadata)

    def _generate_vlm(self, request, do_sample, max_length, temperature, top_p):
        mlx_vlm = _import_dependency("mlx_vlm", "mlx-vlm")
        prompt = request.prompt
        images = _collect_vlm_images(request, self.manifest.video)
        audio_paths = []
        if request.audio is not None:
            audio_paths.append(_audio_to_temp_wav(request.audio))
        try:
            from mlx_vlm.prompt_utils import apply_chat_template

            prompt = _call_with_supported_kwargs(
                apply_chat_template,
                self.handle.processor,
                getattr(self.handle, "config", None),
                request.prompt,
                add_generation_prompt=True,
                num_images=len(images),
                num_audios=len(audio_paths),
                enable_thinking=request.thinking,
            )
        except Exception:
            prompt = request.prompt

        kwargs = dict(self.manifest.generation)
        kwargs.update({
            "max_tokens": int(max_length),
            "temperature": float(temperature if do_sample else 0.0),
            "top_p": float(top_p),
            "verbose": False,
        })
        draft_metadata = self._prepare_vlm_drafter_kwargs(kwargs)
        try:
            generated = _call_with_supported_kwargs(
                mlx_vlm.generate,
                self.handle.model,
                self.handle.processor,
                prompt,
                image=images or None,
                audio=audio_paths or None,
                **kwargs,
            )
            _throw_if_interrupted()
        except TypeError:
            try:
                generated = _call_with_supported_kwargs(
                    mlx_vlm.generate,
                    self.handle.model,
                    self.handle.processor,
                    prompt,
                    image=images,
                    images=images,
                    audio=audio_paths or None,
                    **kwargs,
                )
            except BaseException as e:
                _raise_unless_interrupted(e)
        except BaseException as e:
            _raise_unless_interrupted(e)
        finally:
            for audio_path in audio_paths:
                _remove_temp_file(audio_path)
        metadata = dict(draft_metadata)
        if metadata:
            metadata["generation_mode"] = "mlx_vlm_mtp_draft" if str(metadata.get("assistant_kind") or "").lower() == "mtp" else "mlx_vlm_draft"
        generation_tokens = getattr(generated, "generation_tokens", None)
        if generation_tokens is not None:
            metadata["output_tokens"] = int(generation_tokens)
            metadata["token_count_source"] = "mlx_vlm_generation_result"
        metadata.update(_vlm_draft_acceptance_metadata(kwargs.get("draft_model"), metadata))
        return MLXGenerationOutput(text=getattr(generated, "text", str(generated)), metadata=metadata)

    def _prepare_vlm_drafter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        draft_model = kwargs.get("draft_model")
        if draft_model is None and "draft" in kwargs:
            draft_model = kwargs.pop("draft")
            kwargs["draft_model"] = draft_model
        if draft_model in (None, ""):
            return {}

        draft_kind = _none_if_blank(kwargs.get("draft_kind"))
        metadata = {
            "assistant_kind": draft_kind,
            "assistant_model": str(draft_model),
            "speculative_depth": kwargs.get("draft_block_size") or kwargs.get("max_speculative_tokens"),
        }
        if isinstance(draft_model, str):
            draft_ref = draft_model.strip()
            if not draft_ref:
                kwargs.pop("draft_model", None)
                return {}
            cache_key = (draft_ref, draft_kind or "")
            cached = self._cached_vlm_drafters.get(cache_key)
            if cached is None:
                start = now()
                write_event(
                    "draft_load_start",
                    backend="mlx",
                    modality=self.manifest.task,
                    assistant_model=draft_ref,
                    assistant_kind=draft_kind,
                )
                drafters = _import_dependency("mlx_vlm.speculative.drafters", "mlx-vlm")
                loaded = _call_with_supported_kwargs(drafters.load_drafter, draft_ref, kind=draft_kind)
                if isinstance(loaded, tuple):
                    drafter, resolved_kind = loaded[0], loaded[1] if len(loaded) > 1 else draft_kind
                else:
                    drafter, resolved_kind = loaded, draft_kind
                cached = (drafter, resolved_kind)
                self._cached_vlm_drafters[cache_key] = cached
                write_event(
                    "draft_load_end",
                    backend="mlx",
                    modality=self.manifest.task,
                    assistant_model=draft_ref,
                    assistant_kind=resolved_kind,
                    seconds=elapsed_seconds(start),
                    memory=_backend_memory_stats(self.backend),
                )
            drafter, resolved_kind = cached
            kwargs["draft_model"] = drafter
            if resolved_kind:
                kwargs["draft_kind"] = resolved_kind
                metadata["assistant_kind"] = resolved_kind
            metadata["assistant_model"] = draft_ref
        return metadata

    def _generate_audio_asr(self, request, do_sample, max_length, temperature):
        if request.audio is None:
            raise ValueError("MLX audio ASR needs an AUDIO input on TextGenerate.")
        audio_path = _audio_to_temp_wav(request.audio)
        try:
            kwargs = dict(self.manifest.generation)
            if request.prompt:
                kwargs.setdefault("initial_prompt", request.prompt)
            kwargs.setdefault("max_tokens", int(max_length))
            kwargs.setdefault("temperature", float(temperature if do_sample else 0.0))
            generated = _call_audio_generate(self.handle.model, audio_path, kwargs)
            _throw_if_interrupted()
        except BaseException as e:
            _raise_unless_interrupted(e)
        finally:
            _remove_temp_file(audio_path)
        return MLXGenerationOutput(text=_extract_text(generated))

    def _generate_audio_tts(self, request):
        kwargs = dict(self.manifest.generation)
        kwargs.update(self.manifest.audio)
        text = kwargs.pop("text", request.prompt)
        try:
            generated = _call_tts_generate(self.handle.model, text, kwargs)
            audio, sample_rate, text_parts = _collect_tts_audio(generated, self.manifest.audio)
            _throw_if_interrupted()
        except BaseException as e:
            _raise_unless_interrupted(e)
        text_summary = "".join(text_parts).strip()
        if not text_summary:
            text_summary = f"Generated audio at {sample_rate} Hz."
        return MLXGenerationOutput(text=text_summary, audio=audio)

    def _generate_embeddings(self, request):
        inputs = _embedding_inputs(request)
        try:
            if hasattr(self.handle.model, "process"):
                result = self.handle.model.process(inputs, processor=self.handle.processor, **self.manifest.embeddings)
            else:
                result = _run_mlx_embeddings_generate(
                    self.handle.model,
                    self.handle.processor,
                    inputs,
                    request,
                    self.manifest.embeddings,
                )
            _throw_if_interrupted()
        except BaseException as e:
            _raise_unless_interrupted(e)
        output = json.dumps(_embedding_output_to_jsonable(result), indent=2)
        max_bytes = int(self.manifest.embeddings.get("max_output_json_bytes", _DEFAULT_MAX_JSON_BYTES))
        if max_bytes > 0 and len(output.encode("utf-8")) > max_bytes:
            raise ValueError(f"MLX embeddings JSON output exceeded max_output_json_bytes={max_bytes}.")
        return MLXGenerationOutput(text=output)

    def _seed(self, seed: int) -> None:
        try:
            mx = _import_dependency("mlx.core", "mlx")
            if hasattr(mx, "random") and hasattr(mx.random, "seed"):
                mx.random.seed(seed)
        except Exception:
            pass


def _load_handle(backend, manifest: MLXManifest, model_path: str) -> MLXModelHandle:
    try:
        if manifest.task == "lm":
            if _is_mtplx_manifest(manifest):
                model, processor = _load_mtplx_model(model_path, manifest)
            else:
                mlx_lm = _import_dependency("mlx_lm", "mlx-llm")
                loaded = _call_with_supported_kwargs(mlx_lm.load, model_path, **manifest.load)
                model, processor = loaded[0], loaded[1]
        elif manifest.task == "vlm":
            mlx_vlm = _import_dependency("mlx_vlm", "mlx-vlm")
            model, processor = _call_with_supported_kwargs(mlx_vlm.load, model_path, **manifest.load)
        elif manifest.task == "audio_tts":
            model = _load_tts_model(model_path, manifest.load)
            processor = None
        elif manifest.task == "audio_asr":
            model = _load_asr_model(model_path, manifest.load)
            processor = None
        elif manifest.task in {"embeddings", "rerank"}:
            model, processor = _load_embedding_model(model_path, manifest.load)
        else:
            raise BackendUnavailableError(f"Unsupported MLX task: {manifest.task}")
    except BackendUnavailableError:
        raise
    except Exception as e:
        raise BackendUnavailableError(redact_secrets(e)) from e

    handle = MLXModelHandle(
        backend="mlx",
        spec=manifest.spec,
        model=model,
        processor=processor,
        resolved_path=model_path,
        modality=manifest.task,
    )
    if manifest.task == "vlm":
        try:
            from mlx_vlm.utils import load_config

            handle.config = load_config(model_path)
        except Exception:
            handle.config = None
    if _is_mtplx_manifest(manifest):
        handle.mtplx_metadata = getattr(model, "mtplx_metadata", {})
    return backend.register_handle(handle)


@dataclass
class MTPLXHTTPRuntime:
    base_url: str
    model_path: str
    mtplx_metadata: dict[str, Any] = field(default_factory=dict)


def _is_mtplx_manifest(manifest: MLXManifest) -> bool:
    return manifest.task == "lm" and str(manifest.library).strip().lower() == "mtplx"


def _normalize_mtplx_runtime_mode(value: Any) -> str:
    mode = str(value or "in_process").strip().lower().replace("-", "_")
    if mode not in _MTPLX_SUPPORTED_RUNTIME_MODES:
        raise ValueError(f"MTPLX runtime_mode must be one of {sorted(_MTPLX_SUPPORTED_RUNTIME_MODES)}")
    return mode


def _normalize_mtplx_generation_mode(value: Any) -> str:
    mode = str(value or "mtp").strip().lower().replace("-", "_")
    if mode in {"mtpk", "speculative"}:
        mode = "mtp"
    if mode not in {"ar", "mtp"}:
        raise ValueError("MTPLX generation_mode must be 'ar' or 'mtp'.")
    return mode


def _load_mtplx_model(model_path: str, manifest: MLXManifest):
    generation = dict(manifest.generation)
    runtime_mode = _normalize_mtplx_runtime_mode(generation.get("runtime_mode", "in_process"))
    strict_contract = bool(generation.get("strict_contract", True))
    if runtime_mode == "http":
        base_url = str(generation.get("base_url") or "http://127.0.0.1:18083").rstrip("/")
        return MTPLXHTTPRuntime(
            base_url=base_url,
            model_path=model_path,
            mtplx_metadata={"runtime_mode": "http", "strict_contract": strict_contract},
        ), None

    mtplx_runtime = _import_dependency("mtplx.runtime", "mtplx")
    inspection = _inspect_mtplx_model(mtplx_runtime, model_path)
    _validate_mtplx_contract(inspection, strict_contract=strict_contract)
    load_options = dict(manifest.load)
    mtp = bool(load_options.pop("mtp", True))
    mtp_adapter = _none_if_blank(load_options.pop("mtp_adapter", None))
    try:
        runtime = _call_with_supported_kwargs(
            mtplx_runtime.load,
            model_path,
            mtp=mtp,
            mtp_adapter=mtp_adapter,
            **load_options,
        )
    except BaseException as e:
        _raise_unless_interrupted(e)
    runtime.mtplx_metadata = {
        "runtime_mode": "in_process",
        "strict_contract": strict_contract,
        "contract": inspection,
        "mtp": mtp,
        "mtp_adapter": mtp_adapter,
    }
    return runtime, getattr(runtime, "tokenizer", None)


def _inspect_mtplx_model(mtplx_runtime, model_path: str) -> dict[str, Any]:
    inspect_fn = getattr(mtplx_runtime, "inspect", None)
    if callable(inspect_fn):
        try:
            inspected = inspect_fn(model_path)
            return _to_jsonable(inspected.to_dict() if hasattr(inspected, "to_dict") else inspected)
        except BaseException as e:
            _raise_unless_interrupted(e)
    try:
        artifacts = _import_dependency("mtplx.artifacts", "mtplx")
        inspected = artifacts.inspect_model(model_path)
        return _to_jsonable(inspected.to_dict() if hasattr(inspected, "to_dict") else inspected)
    except BaseException as e:
        _raise_unless_interrupted(e)


def _validate_mtplx_contract(inspection: dict[str, Any], strict_contract: bool) -> None:
    compatibility = inspection.get("compatibility") if isinstance(inspection.get("compatibility"), dict) else {}
    can_run = compatibility.get("can_run")
    if can_run is None:
        can_run = inspection.get("passes_primary_gate")
    if can_run is None:
        tier = str(compatibility.get("tier") or compatibility.get("support_level") or "").lower()
        can_run = tier in {"verified", "verified_native", "verified-native"}
    runtime_contract = (
        compatibility.get("runtime_contract")
        or inspection.get("runtime_contract_data")
        or inspection.get("runtime_contract_path")
        or compatibility.get("runtime_contract_path")
    )
    mtp_supported = str(compatibility.get("mtp_supported") or inspection.get("mtp_supported") or "").strip().lower()
    if mtp_supported in {"no", "false", "unsupported"}:
        can_run = False
    recognized = compatibility.get("recognized", inspection.get("architecture_recognized"))
    arch_id = compatibility.get("arch_id") or inspection.get("mtp_arch")
    runtime_contract = (
        runtime_contract
        or compatibility.get("runtime_contract_data")
        or inspection.get("runtime_contract")
    )
    if strict_contract and not (bool(can_run) and bool(runtime_contract)):
        support = f" arch_id={arch_id!r} recognized={recognized!r}" if arch_id or recognized is not None else ""
        raise BackendUnavailableError(
            "MTPLX strict_contract requires a verified MTP model with mtplx_runtime.json. "
            f"{support} Inspection: {redact_secrets(json.dumps(_to_jsonable(inspection), sort_keys=True)[:1000])}"
        )
    if can_run is False:
        reason = (
            compatibility.get("message")
            or compatibility.get("reason")
            or compatibility.get("runtime_contract_error")
            or inspection.get("runtime_contract_error")
            or "model is not MTPLX-compatible"
        )
        raise BackendUnavailableError(f"MTPLX model compatibility check failed: {redact_secrets(reason)}")


def _load_tts_model(model_path: str, kwargs: dict[str, Any]):
    try:
        from mlx_audio.tts.utils import load_model
    except Exception:
        try:
            from mlx_audio.tts.utils import load as load_model
        except Exception as e:
            raise BackendUnavailableError("mlx-audio TTS is not installed. Install `pip install -e .[mlx-audio]`.") from e
    return _call_with_supported_kwargs(load_model, model_path, **kwargs)


def _load_asr_model(model_path: str, kwargs: dict[str, Any]):
    try:
        from mlx_audio.stt import load
    except Exception:
        try:
            from mlx_audio.stt.utils import load
        except Exception as e:
            raise BackendUnavailableError("mlx-audio STT is not installed. Install `pip install -e .[mlx-audio]`.") from e
    return _call_with_supported_kwargs(load, model_path, **kwargs)


def _load_embedding_model(model_path: str, kwargs: dict[str, Any]):
    try:
        from mlx_embeddings import load
    except Exception:
        try:
            from mlx_embeddings.utils import load
        except Exception as e:
            raise BackendUnavailableError("mlx-embeddings is not installed. Install `pip install -e .[mlx-embeddings]`.") from e
    return _call_with_supported_kwargs(load, model_path, **kwargs)


def _call_audio_generate(model, audio_path: str, kwargs: dict[str, Any]):
    try:
        return _call_with_supported_kwargs(model.generate, audio_path, **kwargs)
    except TypeError:
        return _call_with_supported_kwargs(model.generate, audio=audio_path, **kwargs)


def _call_tts_generate(model, text: str, kwargs: dict[str, Any]):
    try:
        return _call_with_supported_kwargs(model.generate, text, **kwargs)
    except TypeError:
        return _call_with_supported_kwargs(model.generate, text=text, **kwargs)


def _collect_tts_audio(generated, audio_options: dict[str, Any]):
    import torch

    sample_rate = int(audio_options.get("sample_rate") or 24000)
    tensors = []
    text_parts = []
    iterator = generated if _is_tts_result_iterable(generated) else [generated]
    for result in iterator:
        _throw_if_interrupted()
        if hasattr(result, "text"):
            text_parts.append(str(result.text))
        result_audio = result if _is_audio_array_like(result) else getattr(result, "audio", None)
        if result_audio is None and isinstance(result, dict):
            result_audio = result.get("audio")
        if result_audio is None:
            continue
        sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
        tensors.append(_array_to_audio_tensor(result_audio))

    if not tensors:
        return None, sample_rate, text_parts
    waveform = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
    return {"waveform": waveform, "sample_rate": sample_rate}, sample_rate, text_parts


def _is_tts_result_iterable(value) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if _is_audio_array_like(value):
        return False
    return hasattr(value, "__iter__")


def _is_audio_array_like(value) -> bool:
    if value is None:
        return False
    return hasattr(value, "shape") and not isinstance(value, (str, bytes, dict))


def _array_to_audio_tensor(value):
    import numpy as np
    import torch

    array = np.array(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        if array.shape[0] > array.shape[1]:
            array = array.T
        array = array.reshape(1, array.shape[0], array.shape[1])
    elif array.ndim == 3:
        pass
    else:
        array = array.reshape(1, 1, -1)
    return torch.tensor(array.tolist(), dtype=torch.float32)


def _audio_to_temp_wav(audio: dict[str, Any]) -> str:
    import wave

    import folder_paths
    import numpy as np

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="mlx_audio_", dir=temp_dir)
    os.close(fd)
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform.reshape(1, -1)
    if waveform.shape[0] > waveform.shape[-1]:
        waveform = waveform.T
    pcm = (np.clip(waveform.T, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return path


def _collect_vlm_images(request: MLXGenerationRequest, video_options: dict[str, Any]) -> list[Any]:
    images = []
    images.extend(_tensor_to_pil_images(request.image))
    video = request.video
    if video is not None:
        stride = int(video_options.get("frame_stride", 24))
        max_frames = int(video_options.get("max_frames", 16))
        frames = video[::max(1, stride)]
        if max_frames > 0:
            frames = frames[:max_frames]
        images.extend(_tensor_to_pil_images(frames))
    return images


def _tensor_to_pil_images(image) -> list[Any]:
    if image is None:
        return []
    import numpy as np
    from PIL import Image

    tensor = image.detach().cpu().clamp(0, 1)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    images = []
    for item in tensor:
        try:
            raw = item.numpy()
        except RuntimeError:
            raw = item.tolist()
        array = (np.array(raw, dtype=np.float32) * 255.0).round().astype(np.uint8)
        images.append(Image.fromarray(array))
    return images


def _embedding_inputs(request: MLXGenerationRequest):
    try:
        parsed = json.loads(request.prompt)
        inputs = parsed
    except Exception:
        inputs = [{"text": request.prompt}]
    if request.image is not None and isinstance(inputs, list):
        pil_images = _tensor_to_pil_images(request.image)
        for index, image in enumerate(pil_images):
            if index < len(inputs) and isinstance(inputs[index], dict):
                inputs[index].setdefault("image", image)
    return inputs


def _run_mlx_embeddings_generate(model, processor, inputs, request: MLXGenerationRequest, kwargs: dict[str, Any]):
    mlx_embeddings = _import_dependency("mlx_embeddings", "mlx-embeddings")
    generation_kwargs = dict(kwargs)
    generation_kwargs.pop("max_output_json_bytes", None)
    texts, images = _embedding_texts_and_images(inputs, request)
    return _call_with_supported_kwargs(
        mlx_embeddings.generate,
        model,
        processor,
        texts,
        images=images,
        **generation_kwargs,
    )


def _embedding_texts_and_images(inputs, request: MLXGenerationRequest):
    if isinstance(inputs, list):
        texts = []
        images = []
        for item in inputs:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
                if "image" in item:
                    images.append(item["image"])
            else:
                texts.append(str(item))
        return texts, images or None
    if isinstance(inputs, dict):
        if "documents" in inputs or "query" in inputs:
            return inputs, None
        image = inputs.get("image")
        text = str(inputs.get("text", request.prompt))
        return text, image
    return str(request.prompt), None


def _embedding_output_to_jsonable(value):
    embeddings = getattr(value, "text_embeds", value)
    jsonable = _to_jsonable(embeddings)
    shape = _shape_of(embeddings)
    if shape is not None:
        return {
            "shape": shape,
            "embeddings": jsonable,
        }
    return jsonable


def _shape_of(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(part) for part in shape]
    except Exception:
        return list(shape)


def _validate_mtplx_sampling(kwargs: dict[str, Any], policy: str) -> None:
    unsupported = {}
    if float(kwargs.get("min_p", 0.0) or 0.0) != 0.0:
        unsupported["min_p"] = kwargs.get("min_p")
    if float(kwargs.get("presence_penalty", 0.0) or 0.0) != 0.0:
        unsupported["presence_penalty"] = kwargs.get("presence_penalty")
    if float(kwargs.get("repetition_penalty", 1.0) or 1.0) != 1.0:
        unsupported["repetition_penalty"] = kwargs.get("repetition_penalty")
    if not unsupported:
        return
    if policy in {"ignore", "drop"}:
        for key in unsupported:
            kwargs.pop(key, None)
        return
    if policy != "error":
        raise ValueError("MTPLX unsupported_sampling_policy must be 'error' or 'ignore'.")
    details = ", ".join(f"{key}={value}" for key, value in sorted(unsupported.items()))
    raise BackendUnavailableError(f"MTPLX does not support these TextGenerate sampling controls yet: {details}")


def _mtplx_sampler_config(mtplx_sampling, *, temperature: Any, top_p: Any, top_k: Any):
    return mtplx_sampling.SamplerConfig(
        temperature=float(temperature or 0.0),
        top_p=float(top_p if top_p not in (None, "") else 1.0),
        top_k=int(top_k or 0),
    )


def _mtplx_draft_sampler_config(mtplx_sampling, kwargs: dict[str, Any]):
    draft = kwargs.get("draft_sampler")
    if draft in (None, "", False):
        if not any(key in kwargs for key in ("draft_temperature", "draft_top_p", "draft_top_k")):
            return None
        draft = {}
    if not isinstance(draft, dict):
        raise ValueError("MTPLX draft_sampler must be an object with temperature, top_p, and top_k.")
    return _mtplx_sampler_config(
        mtplx_sampling,
        temperature=draft.get("temperature", draft.get("temp", kwargs.get("draft_temperature", kwargs.get("temp", 0.0)))),
        top_p=draft.get("top_p", kwargs.get("draft_top_p", kwargs.get("top_p", 1.0))),
        top_k=draft.get("top_k", kwargs.get("draft_top_k", kwargs.get("top_k", 0))),
    )


def _mtplx_encode_prompt(tokenizer, prompt: str, kwargs: dict[str, Any]) -> list[int]:
    encoding = str(kwargs.get("prompt_encoding", "tokenizer")).strip().lower().replace("-", "_")
    if encoding in {"tokenizer", "raw", ""}:
        return _mtplx_encode(tokenizer, prompt)
    if encoding not in {"mtplx_prompt_case", "prompt_case", "chat"}:
        raise ValueError("MTPLX prompt_encoding must be 'tokenizer' or 'mtplx_prompt_case'.")
    try:
        schema = _import_dependency("mtplx.benchmarks.schema", "mtplx")
        prompt_case = schema.PromptCase(
            id=str(kwargs.get("prompt_case_id") or "comfy_textgenerate"),
            category=str(kwargs.get("prompt_case_category") or "comfy"),
            prompt=prompt,
            max_tokens=int(kwargs.get("max_tokens", 256)),
        )
        encoded = schema.encode_prompt_case(
            tokenizer,
            prompt_case,
            chat_template=bool(kwargs.get("chat_template", True)),
            enable_thinking=bool(kwargs.get("enable_thinking", kwargs.get("thinking", False))),
        )
    except AttributeError as e:
        raise BackendUnavailableError("Installed MTPLX does not expose PromptCase prompt encoding.") from e
    except TypeError:
        encoded = schema.encode_prompt_case(tokenizer, prompt_case)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list):
        raise BackendUnavailableError("MTPLX prompt case encoder did not return a token id list.")
    return [int(token) for token in encoded]


def _mtplx_encode(tokenizer, prompt: str) -> list[int]:
    if tokenizer is None:
        raise BackendUnavailableError("MTPLX runtime did not expose a tokenizer.")
    try:
        encoded = tokenizer.encode(prompt)
    except Exception:
        encoded = tokenizer(prompt)
        if isinstance(encoded, dict):
            encoded = encoded.get("input_ids")
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list):
        raise BackendUnavailableError("MTPLX tokenizer did not return a token id list.")
    return [int(token) for token in encoded]


def _mtplx_stats_dict(generated) -> dict[str, Any]:
    stats = getattr(generated, "stats", None)
    if stats is None and isinstance(generated, dict):
        stats = generated.get("stats") or generated.get("mtplx_stats")
    if hasattr(stats, "to_dict"):
        return _to_jsonable(stats.to_dict())
    return _to_jsonable(stats or {})


def _mtplx_output_metadata(
    stats: dict[str, Any],
    *,
    generation_mode: str,
    first_token_seconds: float | None,
    output_tokens: int | None,
    runtime_mode: str,
    profile: Any,
) -> dict[str, Any]:
    drafted_tokens = int(stats.get("drafted_tokens") or 0)
    accepted_drafts = int(stats.get("accepted_drafts") or 0)
    acceptance_rate = accepted_drafts / drafted_tokens if drafted_tokens > 0 else None
    metadata = {
        "first_token_seconds": first_token_seconds,
        "output_tokens": output_tokens,
        "token_count_source": "mtplx_generation_stats" if output_tokens is not None else "fallback_tokenizer_after_generation",
        "generation_mode": f"mtplx_{generation_mode}",
        "mtplx_runtime_mode": runtime_mode,
        "mtplx_profile": str(profile or ""),
        "speculative_depth": stats.get("speculative_depth"),
        "requested_speculative_depth": stats.get("requested_speculative_depth"),
        "accepted_drafts": accepted_drafts,
        "rejected_drafts": stats.get("rejected_drafts"),
        "drafted_tokens": drafted_tokens,
        "acceptance_rate": acceptance_rate,
        "prompt_eval_time_s": stats.get("prompt_eval_time_s"),
        "draft_time_s": stats.get("draft_time_s"),
        "verify_time_s": stats.get("verify_time_s"),
        "decode_tok_s": stats.get("decode_tok_s") or stats.get("tok_s"),
        "end_to_end_tok_s": stats.get("end_to_end_tok_s"),
        "mtplx_stats": stats,
    }
    for key in _MTPLX_STATS_METADATA_KEYS:
        value = stats.get(key)
        if value not in ("", None):
            metadata[key] = value
    if metadata.get("decode_tok_s") in ("", None) and stats.get("tok_s") not in ("", None):
        metadata["decode_tok_s"] = stats.get("tok_s")
    return metadata


def _mtplx_generation_options_metadata(
    kwargs: dict[str, Any],
    *,
    generation_mode: str,
    prompt_encoding: str,
    draft_sampler: Any,
) -> dict[str, Any]:
    metadata = {"prompt_encoding": prompt_encoding}
    if generation_mode == "mtp":
        defaults = {
            "verify_strategy": "capture_commit",
            "verify_core": _MTPLX_DEFAULT_VERIFY_CORE,
            "mtp_hidden_variant": _MTPLX_DEFAULT_HIDDEN_VARIANT,
            "mtp_cache_policy": _MTPLX_DEFAULT_CACHE_POLICY,
            "mtp_history_policy": _MTPLX_DEFAULT_HISTORY_POLICY,
            "draft_core": _MTPLX_DEFAULT_DRAFT_CORE,
            "min_speculative_depth": 1,
        }
        for key, default in defaults.items():
            metadata[key] = kwargs.get(key, default)
        if kwargs.get("draft_margin_threshold") not in ("", None):
            metadata["draft_margin_threshold"] = kwargs.get("draft_margin_threshold")
        if draft_sampler is not None:
            metadata["draft_sampler_temperature"] = getattr(draft_sampler, "temperature", None)
            metadata["draft_sampler_top_p"] = getattr(draft_sampler, "top_p", None)
            metadata["draft_sampler_top_k"] = getattr(draft_sampler, "top_k", None)
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def _vlm_draft_acceptance_metadata(draft_model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    lens = getattr(draft_model, "accept_lens", None)
    if not lens:
        return {}
    accept_lens = [max(0, int(value)) for value in list(lens)]
    block_total = metadata.get("speculative_depth")
    if block_total in ("", None):
        block_total = getattr(getattr(draft_model, "config", None), "block_size", None)
    try:
        block_total_int = int(block_total)
    except Exception:
        block_total_int = 0
    max_draft_per_round = max(block_total_int - 1, 0)
    output_tokens = metadata.get("output_tokens")
    drafted_tokens = 0
    if max_draft_per_round > 0 and output_tokens not in ("", None):
        remaining_output = max(0, int(output_tokens) - 1)
        emitted_after_first = 0
        for accepted in accept_lens:
            remaining = max(0, remaining_output - emitted_after_first)
            if remaining <= 0:
                break
            drafted_tokens += min(max_draft_per_round, remaining)
            emitted_after_first += min(accepted + 1, remaining)
    elif max_draft_per_round > 0:
        drafted_tokens = len(accept_lens) * max_draft_per_round
    accepted_drafts = sum(min(value, max_draft_per_round) if max_draft_per_round else value for value in accept_lens)
    rejected_drafts = max(drafted_tokens - accepted_drafts, 0) if drafted_tokens else None
    return {
        "speculative_rounds": len(accept_lens),
        "max_draft_tokens_per_round": max_draft_per_round or None,
        "accepted_drafts": accepted_drafts,
        "rejected_drafts": rejected_drafts,
        "drafted_tokens": drafted_tokens or None,
        "acceptance_rate": (accepted_drafts / drafted_tokens) if drafted_tokens else None,
        "mean_accepted_drafts": accepted_drafts / len(accept_lens),
    }


def _generation_event_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "mtplx_runtime_mode",
        "mtplx_profile",
        "speculative_rounds",
        "max_draft_tokens_per_round",
        "accepted_drafts",
        "rejected_drafts",
        "drafted_tokens",
        "acceptance_rate",
        "mean_accepted_drafts",
        "assistant_kind",
        "assistant_model",
        "draft_sampler_temperature",
        "draft_sampler_top_p",
        "draft_sampler_top_k",
        *_MTPLX_OPTION_METADATA_KEYS,
        *_MTPLX_STATS_METADATA_KEYS,
    )
    return {key: metadata.get(key) for key in allowed if metadata.get(key) not in ("", None)}


@contextmanager
def _temporary_mtplx_profile(profile_name):
    profile_name = _none_if_blank(profile_name)
    if profile_name is None:
        yield
        return
    profiles = _import_dependency("mtplx.profiles", "mtplx")
    profile = profiles.get_profile(profile_name)
    apply_profile_env = getattr(profiles, "apply_profile_env", None)
    restore_profile_env = getattr(profiles, "restore_profile_env", None)
    if callable(apply_profile_env) and callable(restore_profile_env):
        previous = apply_profile_env(profile_name)
        try:
            yield
        finally:
            restore_profile_env(previous)
        return
    env = profile.env_dict() if hasattr(profile, "env_dict") else dict(getattr(profile, "env", ()))
    previous = {key: os.environ.get(key) for key in env}
    try:
        for key, value in env.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _generate_mtplx_http(prepared: MLXPreparedLMRequest, manifest: MLXManifest) -> MLXGenerationOutput:
    runtime = _normalize_mtplx_runtime_mode(prepared.kwargs.get("runtime_mode", "http"))
    if runtime != "http":
        raise ValueError("Internal error: _generate_mtplx_http called for non-http runtime.")
    _validate_mtplx_sampling(prepared.kwargs, str(prepared.kwargs.get("unsupported_sampling_policy", "error")).strip().lower())
    base_url = str(prepared.kwargs.get("base_url") or getattr(manifest, "base_url", "") or "http://127.0.0.1:18083").rstrip("/")
    generation_mode = _normalize_mtplx_generation_mode(prepared.kwargs.get("generation_mode", "mtp"))
    payload = {
        "model": manifest.spec.metadata.get("model_id") or "mtplx",
        "messages": [{"role": "user", "content": prepared.prompt}],
        "max_tokens": int(prepared.kwargs.get("max_tokens", 256)),
        "temperature": float(prepared.kwargs.get("temp", prepared.kwargs.get("temperature", 0.0))),
        "top_p": float(prepared.kwargs.get("top_p", 1.0)),
        "top_k": int(prepared.kwargs.get("top_k", 0)),
        "generation_mode": generation_mode,
        "depth": int(prepared.kwargs.get("speculative_depth", 3)),
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        start = now()
        with urllib.request.urlopen(request, timeout=float(prepared.kwargs.get("http_timeout", 900))) as response:
            body = response.read().decode("utf-8")
        _throw_if_interrupted()
        parsed = json.loads(body)
    except BaseException as e:
        _raise_unless_interrupted(e)
    choices = parsed.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    text = message.get("content") or choice.get("text") or ""
    stats = parsed.get("mtplx_stats") or parsed.get("stats") or {}
    usage = parsed.get("usage") or {}
    output_tokens = stats.get("generated_tokens") or usage.get("completion_tokens")
    metadata = _mtplx_output_metadata(
        _to_jsonable(stats),
        generation_mode=generation_mode,
        first_token_seconds=None,
        output_tokens=output_tokens,
        runtime_mode="http",
        profile=prepared.kwargs.get("profile"),
    )
    metadata.setdefault("http_wall_seconds", elapsed_seconds(start))
    return MLXGenerationOutput(text=text, metadata=metadata)


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


def _filter_mlx_lm_generate_kwargs(mlx_lm, kwargs: dict[str, Any]) -> dict[str, Any]:
    filtered = _filter_mlx_lm_stream_kwargs(mlx_lm, kwargs)
    generate = getattr(mlx_lm, "generate", None)
    if not callable(generate):
        return filtered
    try:
        signature = inspect.signature(generate)
    except Exception:
        return filtered
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if "verbose" in kwargs and (accepts_var_kwargs or "verbose" in signature.parameters):
        filtered["verbose"] = kwargs["verbose"]
    if accepts_var_kwargs:
        return filtered
    allowed = set(signature.parameters)
    return {key: value for key, value in filtered.items() if key in allowed}


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


def _extract_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "transcript", "result"):
            if key in value:
                return str(value[key])
        return json.dumps(_to_jsonable(value), indent=2)
    if hasattr(value, "text"):
        return str(value.text)
    if hasattr(value, "segments"):
        return json.dumps(_to_jsonable(value.segments), indent=2)
    return str(value)


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "value": str(value)}
    if hasattr(value, "__dict__"):
        return _to_jsonable(value.__dict__)
    return str(value)


def _benchmark_metadata(manifest: MLXManifest) -> dict[str, Any]:
    metadata = dict(manifest.spec.metadata)
    metadata.update(
        {
            "loader_type": manifest.loader_type,
            "backend": "mlx",
            "backend_device": "apple_silicon_metal",
            "mlx_device": "metal",
            "task": manifest.task,
            "library": manifest.library,
            "model_id": metadata.get("model_id") or os.path.basename(manifest.manifest_path),
            "repo_id": manifest.spec.repo_id,
            "revision": manifest.spec.revision,
            "comparison_group": metadata.get("comparison_group", ""),
            "quantization": metadata.get("quantization", ""),
            "fair_comparison": bool(metadata.get("fair_comparison", False)),
            "package_versions": _package_versions("mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "mlx-audio", "mlx-embeddings", "mtplx"),
        }
    )
    return {key: value for key, value in metadata.items() if value not in ("", None)}


def _package_versions(*packages: str) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()


def _token_count(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.encode(text)
        return len(encoded)
    except Exception:
        pass
    try:
        encoded = tokenizer(text)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded
        return len(input_ids)
    except Exception:
        return None


def _backend_memory_stats(backend) -> dict[str, Any]:
    try:
        memory_stats = getattr(backend, "memory_stats", None)
        if memory_stats is None:
            return {}
        return dict(memory_stats().stats)
    except Exception:
        return {}


def _format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        return prompt


def _call_with_supported_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(*args, **kwargs)
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(*args, **supported)


def _import_dependency(module_name: str, extra_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as e:
        message = redact_secrets(e)
        raise BackendUnavailableError(
            f"Could not import {module_name}. Install `pip install -e .[{extra_name}]` "
            f"or fix the optional dependency error: {message}"
        ) from e


def _throw_if_interrupted() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except Exception as e:
        logging.debug("Could not check ComfyUI processing interruption state: %s", e)


def _raise_unless_interrupted(error: BaseException):
    if error.__class__.__name__ == "InterruptProcessingException":
        raise error
    raise BackendUnavailableError(redact_secrets(error)) from error


def _remove_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _normalize_local_path(value, manifest_path: str) -> str | None:
    value = _none_if_blank(value)
    if value is None:
        return None
    value = os.path.expanduser(value)
    if not os.path.isabs(value):
        value = os.path.join(os.path.dirname(manifest_path), value)
    return os.path.abspath(value)


def _normalize_cache_dir(value: str) -> str:
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return value
    import folder_paths

    return os.path.join(folder_paths.get_folder_paths("mlx")[0], value)


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _none_if_blank(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _normalize_task(value) -> str:
    task = str(value).strip().lower()
    return MLX_TASK_ALIASES.get(task, task)


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
                raise ValueError("MLX manifests must not contain token values; use use_hf_token instead.")
            _reject_inline_tokens(item)
    elif isinstance(value, list):
        for item in value:
            _reject_inline_tokens(item)


def _is_mlx_manifest_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".mlx") or lower.endswith(".mlx.json")


def cleanup_mlx_adapter(adapter: MLXRuntimeAdapter) -> None:
    adapter.unload()
    gc.collect()


MLXTextGenerateAdapter = MLXRuntimeAdapter
