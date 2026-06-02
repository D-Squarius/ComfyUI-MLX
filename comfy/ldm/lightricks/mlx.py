from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class MLXLTXUnavailable(RuntimeError):
    pass


MLX_LTX_REQUIRED_MODEL_FILES = frozenset(
    {
        "config.json",
        "split_model.json",
        "transformer-distilled.safetensors",
        "vae_decoder.safetensors",
        "vae_encoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
        "connector.safetensors",
    }
)


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
    return tuple(patterns)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _redact(value: Any) -> str:
    text = str(value)
    if "hf_" not in text:
        return text
    parts = text.split("hf_")
    return parts[0] + "hf_***" + "hf_***".join(part[8:] for part in parts[1:])


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def validate_ltx_frame_count(num_frames: int) -> int:
    frames = int(num_frames)
    if frames <= 0:
        raise ValueError("LTX frame count must be positive.")
    if (frames - 1) % 8 != 0:
        raise ValueError(f"LTX frame count must be 8k + 1; got {frames}.")
    return frames


@dataclass(frozen=True)
class MLXLTXModelSpec:
    repo_id: str
    revision: str | None = None
    local_path: str | None = None
    variant: str = "q4"
    pipeline: str = "distilled"
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    local_files_only: bool = False
    use_hf_token: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MLXLTXRunConfig:
    prompt: str = ""
    output_path: str = ""
    width: int = 512
    height: int = 512
    num_frames: int = 49
    frame_rate: float = 24.0
    pipeline: str = "distilled"
    stage1_steps: int = 8
    stage2_steps: int = 3
    seed: int = 42
    low_memory: bool | None = None
    low_ram: bool | None = None
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    media_passthrough: bool = True
    prompt_cache_size: int = 4

    def validated(self) -> "MLXLTXRunConfig":
        validate_ltx_frame_count(self.num_frames)
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("LTX width and height must be positive.")
        if float(self.frame_rate) <= 0:
            raise ValueError("LTX frame_rate must be positive.")
        if int(self.stage1_steps) <= 0 or int(self.stage2_steps) < 0:
            raise ValueError("LTX stage step counts must be non-negative and stage1 must be positive.")
        if int(self.prompt_cache_size) < 0:
            raise ValueError("LTX prompt_cache_size must be non-negative.")
        return self


@dataclass(frozen=True)
class MLXLTXGenerationResult:
    output_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MLXLTXModelProxy:
    """Model proxy returned by CheckpointLoaderSimple for MLX LTX checkpoints."""

    is_mlx_ltx_proxy = True

    def __init__(self, spec: MLXLTXModelSpec, run_config: MLXLTXRunConfig):
        self.spec = spec
        self.run_config = run_config
        self.model_options: dict[str, Any] = {}
        self.hook_mode = None
        self.model = _MLXLTXLatentModelProxy()

    def clone(self):
        clone = MLXLTXModelProxy(self.spec, self.run_config)
        clone.model_options = dict(self.model_options)
        clone.hook_mode = self.hook_mode
        return clone

    def is_dynamic(self) -> bool:
        return False

    def get_non_dynamic_delegate(self):
        return self

    def add_object_patch(self, name: str, value: Any) -> None:
        self.model_options[name] = value

    def get_model_object(self, name: str) -> Any:
        if name == "model_sampling":
            return _MLXLTXSamplingProxy()
        raise KeyError(name)

    def model_dtype(self):
        try:
            import torch

            return torch.float32
        except Exception:
            return None


class MLXLTXClipProxy:
    is_mlx_ltx_clip_proxy = True

    def __init__(self, spec: MLXLTXModelSpec, run_config: MLXLTXRunConfig):
        self.spec = spec
        self.run_config = run_config

    def tokenize(self, text: str) -> str:
        return str(text)

    def encode_from_tokens_scheduled(self, tokens: str):
        return [[None, {"mlx_ltx_prompt": str(tokens), "prompt": str(tokens)}]]


class MLXLTXVAEProxy:
    is_mlx_ltx_vae_proxy = True
    latent_channels = 128

    def __init__(self, spec: MLXLTXModelSpec, run_config: MLXLTXRunConfig):
        self.spec = spec
        self.run_config = run_config
        self.first_stage_model = _MLXLTXAudioVAEConfigProxy()

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 32


class _MLXLTXLatentModelProxy:
    def process_latent_out(self, latent):
        return latent


class _MLXLTXSamplingProxy:
    def percent_to_sigma(self, percent: float) -> float:
        return float(percent)


class _MLXLTXAudioVAEConfigProxy:
    latent_frequency_bins = 128
    output_sample_rate = 48000

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        return max(1, int(round(float(frames_number) / max(float(frame_rate), 1.0) * 24)))


class MLXLTXRuntime:
    def __init__(self) -> None:
        self._generation_lock = threading.RLock()
        self._pipeline_cache: dict[tuple[Any, ...], Any] = {}
        self._resolved_path_cache: dict[tuple[Any, ...], str] = {}

    def _import_mlx_core(self):
        if not is_apple_silicon():
            raise MLXLTXUnavailable("MLX LTX requires Apple Silicon macOS.")
        try:
            import mlx.core as mx
        except Exception as e:
            raise MLXLTXUnavailable("MLX is not installed in this ComfyUI environment.") from e
        return mx

    def _import_distilled_pipeline(self):
        self._import_mlx_core()
        try:
            module = importlib.import_module("ltx_pipelines_mlx.distilled")
            return getattr(module, "DistilledPipeline")
        except Exception as e:
            raise MLXLTXUnavailable("ltx_pipelines_mlx is not installed in this ComfyUI environment.") from e

    def resolve_model_path(self, spec: MLXLTXModelSpec, cache_dir: str | None = None) -> str:
        if spec.local_path:
            return os.path.abspath(os.path.expanduser(spec.local_path))
        repo_or_path = os.path.expanduser(spec.repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        cache_key = (
            spec.repo_id,
            spec.revision or "",
            cache_dir or _default_cache_dir(),
            spec.local_files_only,
            spec.allow_patterns,
            spec.ignore_patterns,
            spec.use_hf_token,
        )
        if cache_key in self._resolved_path_cache:
            return self._resolved_path_cache[cache_key]

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise MLXLTXUnavailable("huggingface_hub is required to download MLX LTX models.") from e

        try:
            resolved = snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision or None,
                cache_dir=cache_dir or _default_cache_dir(),
                local_files_only=spec.local_files_only,
                allow_patterns=spec.allow_patterns or None,
                ignore_patterns=spec.ignore_patterns or None,
                token=True if spec.use_hf_token else None,
            )
        except Exception as e:
            raise MLXLTXUnavailable(_redact(e)) from e
        self._resolved_path_cache[cache_key] = resolved
        return resolved

    def generate_video_to_file(
        self,
        spec: MLXLTXModelSpec,
        prompt: str,
        output_path: str,
        *,
        width: int,
        height: int,
        num_frames: int,
        frame_rate: float,
        stage1_steps: int,
        stage2_steps: int,
        seed: int,
        low_memory: bool | None = None,
        low_ram: bool | None = None,
        gemma_model_id: str | None = None,
        media_passthrough: bool = True,
        cache_dir: str | None = None,
    ) -> MLXLTXGenerationResult:
        config = MLXLTXRunConfig(
            prompt=prompt,
            output_path=output_path,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            pipeline=spec.pipeline,
            stage1_steps=int(stage1_steps),
            stage2_steps=int(stage2_steps),
            seed=int(seed),
            low_memory=low_memory if low_memory is not None else self._default_low_memory(spec),
            low_ram=low_ram if low_ram is not None else self._default_low_ram(spec),
            gemma_model_id=gemma_model_id or spec.gemma_model_id,
            media_passthrough=media_passthrough,
        ).validated()
        if config.pipeline != "distilled":
            raise MLXLTXUnavailable("Only the distilled LTX MLX pipeline is currently supported.")

        output = Path(config.output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        model_path = self.resolve_model_path(spec, cache_dir=cache_dir)
        pipeline_cls = self._import_distilled_pipeline()

        cache_key = (
            os.path.abspath(model_path),
            spec.revision or "",
            spec.variant,
            config.pipeline,
            bool(config.low_memory),
            bool(config.low_ram),
            config.gemma_model_id,
        )
        with self._generation_lock:
            pipe = self._pipeline_cache.get(cache_key)
            if pipe is None:
                pipe = _call_with_supported_kwargs(
                    pipeline_cls,
                    model_dir=model_path,
                    gemma_model_id=config.gemma_model_id,
                    low_memory=config.low_memory,
                    low_ram_streaming=bool(config.low_ram),
                )
                self._pipeline_cache[cache_key] = pipe
            generated = _call_with_supported_kwargs(
                pipe.generate_and_save,
                prompt=config.prompt,
                output_path=str(output),
                height=config.height,
                width=config.width,
                num_frames=config.num_frames,
                frame_rate=config.frame_rate,
                seed=config.seed,
                stage1_steps=config.stage1_steps,
                stage2_steps=config.stage2_steps,
            )

        if generated:
            output = Path(str(generated)).expanduser().resolve()
        return MLXLTXGenerationResult(
            output_path=str(output),
            metadata={
                "backend": "mlx_ltx",
                "repo_id": spec.repo_id,
                "variant": spec.variant,
                "width": config.width,
                "height": config.height,
                "num_frames": config.num_frames,
                "frame_rate": config.frame_rate,
                "stage1_steps": config.stage1_steps,
                "stage2_steps": config.stage2_steps,
                "media_passthrough": config.media_passthrough,
            },
        )

    def output_as_comfy_video(self, output_path: str):
        from comfy_api.latest import InputImpl

        return InputImpl.VideoFromFile(output_path)

    def output_components(self, output_path: str):
        return self.output_as_comfy_video(output_path).get_components()

    def unload_all(self) -> None:
        self._pipeline_cache.clear()
        self.empty_cache()

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

    @staticmethod
    def _default_low_ram(spec: MLXLTXModelSpec) -> bool:
        if spec.metadata.get("low_ram_streaming") is not None:
            return bool(spec.metadata["low_ram_streaming"])
        return False

    @staticmethod
    def _default_low_memory(spec: MLXLTXModelSpec) -> bool:
        if spec.metadata.get("low_memory") is not None:
            return bool(spec.metadata["low_memory"])
        return False


class MLXLTXImageProxy:
    is_mlx_ltx_image_proxy = True

    def __init__(self, latent: dict[str, Any]):
        self.latent = latent
        self.media_path = str(latent["mlx_ltx_media_path"])

    def materialize(self):
        return mlx_ltx_media_components(self.latent).images

    def __getattr__(self, name: str) -> Any:
        return getattr(self.materialize(), name)

    def __getitem__(self, item):
        return self.materialize()[item]


class MLXLTXAudioProxy:
    is_mlx_ltx_audio_proxy = True

    def __init__(self, latent: dict[str, Any]):
        self.latent = latent
        self.media_path = str(latent["mlx_ltx_media_path"])

    def materialize(self):
        return mlx_ltx_media_components(self.latent).audio

    def __getattr__(self, name: str) -> Any:
        return getattr(self.materialize(), name)

    def __getitem__(self, item):
        return self.materialize()[item]


def make_model_spec(
    repo_id: str,
    revision: str = "",
    local_path: str = "",
    variant: str = "q4",
    pipeline: str = "distilled",
    allow_patterns: str | Iterable[str] | None = None,
    ignore_patterns: str | Iterable[str] | None = None,
    local_files_only: bool = False,
    use_hf_token: bool = False,
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    low_memory: bool | None = None,
    low_ram_streaming: bool | None = None,
    media_passthrough: bool = True,
    prompt_cache_size: int = 4,
) -> MLXLTXModelSpec:
    metadata = {
        "media_passthrough": bool(media_passthrough),
        "prompt_cache_size": int(prompt_cache_size),
    }
    if low_memory is not None:
        metadata["low_memory"] = bool(low_memory)
    if low_ram_streaming is not None:
        metadata["low_ram_streaming"] = bool(low_ram_streaming)
    return MLXLTXModelSpec(
        repo_id=repo_id.strip(),
        revision=revision.strip() or None,
        local_path=local_path.strip() or None,
        variant=variant,
        pipeline=pipeline,
        gemma_model_id=gemma_model_id,
        allow_patterns=_split_patterns(allow_patterns),
        ignore_patterns=_split_patterns(ignore_patterns),
        local_files_only=local_files_only,
        use_hf_token=use_hf_token,
        metadata=metadata,
    )


def is_mlx_ltx_manifest_path(path: str | os.PathLike[str]) -> bool:
    return str(path).lower().endswith(".mlx_ltx.json")


def is_mlx_ltx_model_dir(path: str | os.PathLike[str]) -> bool:
    model_dir = Path(path).expanduser()
    if not model_dir.is_dir():
        return False
    return all((model_dir / filename).exists() for filename in MLX_LTX_REQUIRED_MODEL_FILES)


def infer_mlx_ltx_variant_from_path(path: str | os.PathLike[str]) -> str:
    name = Path(path).name.lower()
    if "q4" in name or "int4" in name:
        return "q4"
    if "q8" in name or "int8" in name:
        return "q8"
    return "bf16"


def list_mlx_ltx_checkpoint_folders(search_paths: Iterable[str]) -> list[str]:
    folders: list[str] = []
    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue
        for root, subdirs, _files in os.walk(search_path, followlinks=True):
            subdirs[:] = [name for name in subdirs if name != ".git"]
            if is_mlx_ltx_model_dir(root):
                relative_path = os.path.relpath(root, start=search_path)
                if relative_path != ".":
                    folders.append(relative_path)
                subdirs[:] = []
    return sorted(set(folders))


def find_mlx_ltx_checkpoint_folder(ckpt_name: str, search_paths: Iterable[str]) -> str | None:
    safe_name = os.path.relpath(os.path.join("/", ckpt_name), "/")
    for search_path in search_paths:
        model_dir = os.path.join(search_path, safe_name)
        if is_mlx_ltx_model_dir(model_dir):
            return model_dir
    return None


def load_mlx_ltx_model_dir(path: str | os.PathLike[str]):
    model_dir = Path(path).expanduser().resolve()
    if not is_mlx_ltx_model_dir(model_dir):
        raise ValueError(f"MLX LTX model folder is missing required files: {model_dir}")
    variant = infer_mlx_ltx_variant_from_path(model_dir)
    spec = make_model_spec(repo_id=str(model_dir), local_path=str(model_dir), variant=variant, pipeline="distilled")
    run_config = MLXLTXRunConfig(pipeline="distilled", media_passthrough=True).validated()
    return MLXLTXModelProxy(spec, run_config), MLXLTXClipProxy(spec, run_config), MLXLTXVAEProxy(spec, run_config)


def load_mlx_ltx_manifest(path: str | os.PathLike[str]) -> tuple[MLXLTXModelSpec, MLXLTXRunConfig]:
    manifest_path = Path(path).expanduser()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_id = str(payload.get("repo_id") or payload.get("model_id") or "").strip()
    local_path = str(payload.get("local_path") or "").strip()
    if not repo_id and not local_path:
        raise ValueError(f"MLX LTX manifest {manifest_path} needs `repo_id` or `local_path`.")

    variant = str(payload.get("variant") or payload.get("quantization") or "q4")
    pipeline = str(payload.get("pipeline") or "distilled")
    gemma_model_id = str(payload.get("gemma_model_id") or "mlx-community/gemma-3-12b-it-4bit")
    low_memory = _coerce_optional_bool(payload.get("low_memory", None))
    low_ram_streaming = _coerce_optional_bool(payload.get("low_ram_streaming", payload.get("low_ram", None)))
    media_passthrough = bool(payload.get("media_passthrough", True))
    prompt_cache_size = int(payload.get("prompt_cache_size", 4))

    spec = make_model_spec(
        repo_id=repo_id or local_path,
        revision=str(payload.get("revision") or "").strip(),
        local_path=local_path,
        variant=variant,
        pipeline=pipeline,
        allow_patterns=payload.get("allow_patterns"),
        ignore_patterns=payload.get("ignore_patterns"),
        local_files_only=bool(payload.get("local_files_only", False)),
        use_hf_token=bool(payload.get("use_hf_token", False)),
        gemma_model_id=gemma_model_id,
        low_memory=low_memory,
        low_ram_streaming=low_ram_streaming,
        media_passthrough=media_passthrough,
        prompt_cache_size=prompt_cache_size,
    )
    run_config = MLXLTXRunConfig(
        width=int(payload.get("width", 512)),
        height=int(payload.get("height", 512)),
        num_frames=int(payload.get("num_frames", 49)),
        frame_rate=float(payload.get("frame_rate", 24.0)),
        pipeline=pipeline,
        stage1_steps=int(payload.get("stage1_steps", 8)),
        stage2_steps=int(payload.get("stage2_steps", 3)),
        seed=int(payload.get("seed", 42)),
        low_memory=low_memory,
        low_ram=low_ram_streaming,
        gemma_model_id=gemma_model_id,
        media_passthrough=media_passthrough,
        prompt_cache_size=prompt_cache_size,
    )
    return spec, run_config.validated()


def load_mlx_ltx_checkpoint(path: str | os.PathLike[str]):
    if is_mlx_ltx_model_dir(path):
        return load_mlx_ltx_model_dir(path)
    spec, run_config = load_mlx_ltx_manifest(path)
    return MLXLTXModelProxy(spec, run_config), MLXLTXClipProxy(spec, run_config), MLXLTXVAEProxy(spec, run_config)


def is_mlx_ltx_model(model: Any) -> bool:
    return bool(getattr(model, "is_mlx_ltx_proxy", False))


def is_mlx_ltx_vae(vae: Any) -> bool:
    return bool(getattr(vae, "is_mlx_ltx_vae_proxy", False))


def is_mlx_ltx_media_latent(latent: Any) -> bool:
    return isinstance(latent, dict) and bool(latent.get("mlx_ltx_media_path"))


def is_mlx_ltx_image_proxy(value: Any) -> bool:
    return bool(getattr(value, "is_mlx_ltx_image_proxy", False))


def is_mlx_ltx_audio_proxy(value: Any) -> bool:
    return bool(getattr(value, "is_mlx_ltx_audio_proxy", False))


def make_mlx_ltx_image_proxy(latent: dict[str, Any]) -> MLXLTXImageProxy:
    return MLXLTXImageProxy(latent)


def make_mlx_ltx_audio_proxy(latent: dict[str, Any]) -> MLXLTXAudioProxy:
    return MLXLTXAudioProxy(latent)


def mlx_ltx_media_passthrough_enabled(latent: dict[str, Any]) -> bool:
    metadata = latent.get("mlx_ltx_metadata") if isinstance(latent.get("mlx_ltx_metadata"), dict) else {}
    return bool(metadata.get("media_passthrough", True))


def mlx_ltx_video_from_proxy(image_proxy: MLXLTXImageProxy):
    return _RUNTIME.output_as_comfy_video(image_proxy.media_path)


def materialize_mlx_ltx_image_proxy(image_proxy: MLXLTXImageProxy):
    return image_proxy.materialize()


def materialize_mlx_ltx_audio_proxy(audio_proxy: MLXLTXAudioProxy):
    return audio_proxy.materialize()


def is_mlx_ltx_audio_placeholder(latent: Any) -> bool:
    return isinstance(latent, dict) and latent.get("type") == "mlx_ltx_audio_placeholder"


def mlx_ltx_media_components(latent: dict[str, Any]):
    return _RUNTIME.output_components(str(latent["mlx_ltx_media_path"]))


def make_mlx_ltx_audio_placeholder(frames_number: int, frame_rate: int | float, batch_size: int = 1) -> dict[str, Any]:
    try:
        import torch

        samples = torch.empty(0)
    except Exception:
        samples = None
    return {
        "samples": samples,
        "type": "mlx_ltx_audio_placeholder",
        "frames_number": int(frames_number),
        "frame_rate": float(frame_rate),
        "batch_size": int(batch_size),
    }


def run_mlx_ltx_sampling(
    model: MLXLTXModelProxy,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    *,
    seed: int | None = None,
    cfg: float | None = None,
) -> dict[str, Any]:
    if not is_mlx_ltx_model(model):
        raise TypeError("run_mlx_ltx_sampling requires an MLXLTXModelProxy.")

    prompt = extract_mlx_ltx_prompt(positive) or ""
    width, height, frames = infer_ltx_dimensions(latent)
    frame_rate = extract_mlx_ltx_frame_rate(positive) or extract_mlx_ltx_frame_rate(negative) or model.run_config.frame_rate
    result = _RUNTIME.generate_video_to_file(
        model.spec,
        prompt,
        _native_temp_output_path(model.spec),
        width=width,
        height=height,
        num_frames=frames,
        frame_rate=frame_rate,
        stage1_steps=model.run_config.stage1_steps,
        stage2_steps=model.run_config.stage2_steps,
        seed=int(seed if seed is not None else model.run_config.seed),
        low_memory=model.run_config.low_memory,
        low_ram=model.run_config.low_ram,
        gemma_model_id=model.run_config.gemma_model_id,
        media_passthrough=model.run_config.media_passthrough,
    )
    try:
        import torch

        samples = torch.empty(0)
    except Exception:
        samples = latent.get("samples")
    return {
        "samples": samples,
        "type": "mlx_ltx_media",
        "mlx_ltx_media_path": result.output_path,
        "mlx_ltx_media_kind": "av",
        "mlx_ltx_metadata": {
            **result.metadata,
            "prompt": prompt,
            "negative_prompt": extract_mlx_ltx_prompt(negative) or "",
            "cfg": cfg,
        },
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "num_frames": frames,
    }


def extract_mlx_ltx_prompt(conditioning: Any) -> str | None:
    for metadata in _conditioning_metadata(conditioning):
        for key in ("mlx_ltx_prompt", "prompt", "text"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
    return None


def extract_mlx_ltx_frame_rate(conditioning: Any) -> float | None:
    for metadata in _conditioning_metadata(conditioning):
        value = metadata.get("frame_rate")
        if value is not None:
            return float(value)
    return None


def infer_ltx_dimensions(latent: dict[str, Any]) -> tuple[int, int, int]:
    if latent.get("width") and latent.get("height") and latent.get("num_frames"):
        return int(latent["width"]), int(latent["height"]), int(latent["num_frames"])
    samples = latent.get("samples")
    if hasattr(samples, "is_nested") and samples.is_nested:
        samples = samples.unbind()[0]
    if not hasattr(samples, "shape") or len(samples.shape) < 5:
        raise ValueError("MLX LTX sampling needs a video latent with shape [B, C, F, H, W].")
    latent_frames = int(samples.shape[2])
    height = int(samples.shape[3]) * int(latent.get("downscale_ratio_spacial", 32))
    width = int(samples.shape[4]) * int(latent.get("downscale_ratio_spacial", 32))
    frames = (latent_frames - 1) * 8 + 1
    return width, height, frames


def unload_all() -> None:
    _RUNTIME.unload_all()


def empty_cache() -> None:
    _RUNTIME.empty_cache()


def _call_with_supported_kwargs(func, *args, **kwargs):
    try:
        import inspect

        signature = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return func(*args, **kwargs)
        supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return func(*args, **supported)
    except (TypeError, ValueError):
        return func(*args, **kwargs)


def _default_cache_dir() -> str:
    import folder_paths

    root = folder_paths.get_folder_paths("mlx_ltx")[0]
    return os.path.join(root, "huggingface")


def _conditioning_metadata(conditioning: Any) -> Iterable[dict[str, Any]]:
    if isinstance(conditioning, dict):
        yield conditioning
        return
    if not isinstance(conditioning, (list, tuple)):
        return
    for item in conditioning:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
            yield item[1]


def _native_temp_output_path(spec: MLXLTXModelSpec) -> str:
    import folder_paths

    safe_repo = "".join(ch if ch.isalnum() else "_" for ch in spec.repo_id)[-80:]
    directory = Path(folder_paths.get_temp_directory()) / "mlx_ltx"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"{safe_repo}_{time.time_ns()}.mp4")


_RUNTIME = MLXLTXRuntime()
