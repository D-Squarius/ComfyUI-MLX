from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import psutil

from .base import BackendCapabilities, BackendUnavailableError, MemoryStats, ModelSpec
from .benchmark_stats import elapsed_seconds, now, redact_secrets, write_event
from .mlx_backend import is_apple_silicon
from .mlx_ltx_native_kernels import coerce_native_kernel_set, make_native_kernel_runtime


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


def _looks_like_ltx_negative_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    if not lowered.strip():
        return True
    markers = (
        "black frame",
        "white frame",
        "silent audio",
        "broken video",
        "corrupted",
        "static image",
        "cartoon",
        "childish",
        "ugly",
        "low quality",
        "worst quality",
    )
    return any(marker in lowered for marker in markers)


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


MLX_LTX_PROFILE_LEVELS = {"off", "stage", "block_group", "block_deep"}
MLX_LTX_DENOISE_SYNC_POLICIES = {"each_step", "end", "none", "interval"}
MLX_LTX_EXECUTION_MODES = {"denoiser_island", "legacy_media"}

MLX_LTX_DEFAULT_FOLDER_BY_VARIANT = {
    "q4": "ltx-2.3-mlx-q4",
    "q8": "ltx-2.3-mlx-q8",
    "bf16": "ltx-2.3-mlx",
}

MLX_LTX_CHECKPOINT_ALIASES = {
    "LTX-2.3 22B Distilled 1.1 MLX Q4 Transformer": "ltx-2.3-mlx-q4",
    "LTX-2.3 22B Distilled 1.1 MLX Q8 Transformer": "ltx-2.3-mlx-q8",
    "LTX-2.3 22B Distilled 1.1 MLX BF16 Transformer": "ltx-2.3-mlx",
    "LTX-2.3 22B Dev+LoRA MLX BF16 Transformer": "ltx23_dev_lora_mlx_bf16_native.mlx_ltx.json",
}

MLX_LTX_TEXT_ENCODER_ALIASES = {
    "Gemma 3 12B IT 4-bit MLX text encoder": "ltx-2.3-mlx-q4",
}

MLX_LTX_AUDIO_VAE_ALIASES = {
    "LTX-2.3 Audio VAE": "ltx-2.3-mlx-q4",
}

MLX_LTX_LATENT_UPSCALER_ALIASES = {
    "LTX-2.3 x2 Spatial Upscaler v1.1": "ltx23_dev_lora_mlx_bf16_native.mlx_ltx.json",
}


def _coerce_profile_level(value: Any) -> str:
    level = str(value or "stage").strip().lower()
    if level not in MLX_LTX_PROFILE_LEVELS:
        raise ValueError(f"MLX LTX profile_level must be one of {sorted(MLX_LTX_PROFILE_LEVELS)}; got {level!r}.")
    return level


def _coerce_denoise_sync_policy(value: Any) -> str:
    policy = str(value or "each_step").strip().lower()
    if policy not in MLX_LTX_DENOISE_SYNC_POLICIES:
        raise ValueError(
            f"MLX LTX denoise_sync_policy must be one of {sorted(MLX_LTX_DENOISE_SYNC_POLICIES)}; got {policy!r}."
        )
    return policy


def _coerce_execution_mode(value: Any) -> str:
    mode = str(value or "denoiser_island").strip().lower().replace("-", "_")
    if mode in {"island", "mlx_island"}:
        mode = "denoiser_island"
    if mode in {"legacy", "media", "full_pipeline"}:
        mode = "legacy_media"
    if mode not in MLX_LTX_EXECUTION_MODES:
        raise ValueError(f"MLX LTX execution_mode must be one of {sorted(MLX_LTX_EXECUTION_MODES)}; got {mode!r}.")
    return mode


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _manifest_asset_path(manifest_path: Path, value: Any, folder_type: str | None = None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    local = (manifest_path.parent / candidate).resolve()
    if local.exists():
        return str(local)
    if folder_type is not None:
        try:
            import folder_paths

            resolved = folder_paths.get_full_path(folder_type, text)
            if resolved is not None:
                return str(Path(resolved).expanduser().resolve())
        except Exception:
            pass
    return str(local)


def _default_cache_dir() -> str:
    import folder_paths

    root = folder_paths.get_folder_paths("mlx_ltx")[0]
    return os.path.join(root, "huggingface")


def _alias_to_mlx_ltx_folder(name: str, alias_map: dict[str, str] | None = None) -> str:
    value = str(name).strip()
    aliases = {}
    aliases.update(MLX_LTX_CHECKPOINT_ALIASES)
    aliases.update(MLX_LTX_TEXT_ENCODER_ALIASES)
    aliases.update(MLX_LTX_AUDIO_VAE_ALIASES)
    aliases.update(MLX_LTX_LATENT_UPSCALER_ALIASES)
    if alias_map:
        aliases.update(alias_map)
    return aliases.get(value, value)


def is_mlx_ltx_folder_path(path: str | os.PathLike[str]) -> bool:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        return False
    required = (
        "config.json",
        "split_model.json",
        "transformer-distilled-1.1.safetensors",
        "connector.safetensors",
    )
    return all((folder / name).exists() for name in required)


def is_mlx_ltx_manifest_path(path: str | os.PathLike[str]) -> bool:
    return str(path).lower().endswith(".mlx_ltx.json")


def is_mlx_ltx_reference_path(path: str | os.PathLike[str]) -> bool:
    return is_mlx_ltx_manifest_path(path) or is_mlx_ltx_folder_path(path)


def _infer_variant_from_path(path: str | os.PathLike[str]) -> str:
    name = Path(path).name.lower()
    if "q8" in name:
        return "q8"
    if "q4" in name:
        return "q4"
    return "bf16"


def resolve_mlx_ltx_checkpoint_path(name: str, alias_map: dict[str, str] | None = None) -> str | None:
    import folder_paths

    candidate_name = _alias_to_mlx_ltx_folder(name, alias_map=alias_map)
    expanded = Path(candidate_name).expanduser()
    if expanded.exists() and is_mlx_ltx_reference_path(expanded):
        return str(expanded.resolve())

    relative = os.path.relpath(os.path.join("/", candidate_name), "/")
    for root in folder_paths.get_folder_paths("checkpoints"):
        folder = Path(root) / relative
        if is_mlx_ltx_folder_path(folder):
            return str(folder.resolve())

    file_path = folder_paths.get_full_path("checkpoints", candidate_name)
    if file_path is not None and is_mlx_ltx_reference_path(file_path):
        return file_path
    return None


def list_mlx_ltx_checkpoint_choices(*, aliases: bool = True) -> list[str]:
    import folder_paths

    choices: set[str] = set()
    if aliases:
        choices.update(MLX_LTX_CHECKPOINT_ALIASES)
    for root in folder_paths.get_folder_paths("checkpoints"):
        base = Path(root)
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if is_mlx_ltx_folder_path(child):
                choices.add(child.name)
    return sorted(choices)


def validate_ltx_frame_count(num_frames: int) -> int:
    frames = int(num_frames)
    if frames <= 0:
        raise ValueError("LTX frame count must be positive.")
    if (frames - 1) % 8 != 0:
        raise ValueError(f"LTX frame count must be 8k + 1; got {frames}.")
    return frames


def ltx_duration_seconds(num_frames: int, frame_rate: float) -> float:
    frames = validate_ltx_frame_count(num_frames)
    fps = float(frame_rate)
    if fps <= 0:
        raise ValueError("LTX frame_rate must be positive.")
    return frames / fps


@dataclass(frozen=True)
class MLXLTXModelSpec(ModelSpec):
    variant: str = "q4"
    pipeline: str = "distilled"
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"


@dataclass(frozen=True)
class MLXLTXRunConfig:
    prompt: str
    output_path: str
    width: int = 512
    height: int = 512
    num_frames: int = 49
    frame_rate: float = 24.0
    pipeline: str = "distilled"
    stage1_steps: int = 8
    stage2_steps: int = 3
    seed: int = 42
    low_memory: bool | None = None
    low_ram: bool = False
    tile_frames: int = 1
    tile_spatial: int = 1
    tile_overlap: int = 2
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    image: str | None = None
    audio: str | None = None
    media_passthrough: bool = True
    prompt_cache_size: int = 8
    stage_snapshot_cache: bool = False
    mlx_cache_limit_gb: float | None = None
    profile_level: str = "stage"
    metal_capture: bool = False
    block_profile_interval: int = 8
    eval_every: int | None = None
    stage1_eval_every: int | None = None
    stage2_eval_every: int | None = None
    decode_profile: bool = False
    compile_dit: bool = False
    stage1_compile_dit: bool | None = None
    stage2_compile_dit: bool | None = None
    compile_block_stack: bool = False
    compile_x0: bool = False
    compile_shapeless: bool = False
    fuse_attention_projections: bool = False
    flatten_attention_projections: bool = False
    dequantize_video_ffn: bool = False
    native_metal_kernels: bool = False
    native_kernel_profile: bool = False
    native_kernel_verify: bool = True
    native_kernel_fallback: bool = True
    native_kernel_set: str = "off"
    ffmpeg_video_encoder: str = "libx264"
    ffmpeg_video_bitrate: str = "12M"
    cache_text_kv: bool = False
    text_kv_cache_size: int = 2048
    audio_debug: bool = False
    audio_debug_dir: str | None = None
    audio_debug_dump_arrays: bool = True
    denoise_sync_policy: str = "each_step"
    denoise_sync_interval: int = 2

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
        if int(self.text_kv_cache_size) < 0:
            raise ValueError("LTX text_kv_cache_size must be non-negative.")
        if self.mlx_cache_limit_gb is not None and float(self.mlx_cache_limit_gb) < 0:
            raise ValueError("LTX mlx_cache_limit_gb must be non-negative.")
        _coerce_profile_level(self.profile_level)
        if int(self.block_profile_interval) < 0:
            raise ValueError("LTX block_profile_interval must be non-negative.")
        if self.eval_every is not None and int(self.eval_every) < 0:
            raise ValueError("LTX eval_every must be non-negative.")
        if self.stage1_eval_every is not None and int(self.stage1_eval_every) < 0:
            raise ValueError("LTX stage1_eval_every must be non-negative.")
        if self.stage2_eval_every is not None and int(self.stage2_eval_every) < 0:
            raise ValueError("LTX stage2_eval_every must be non-negative.")
        _coerce_denoise_sync_policy(self.denoise_sync_policy)
        if int(self.denoise_sync_interval) <= 0:
            raise ValueError("LTX denoise_sync_interval must be positive.")
        coerce_native_kernel_set(self.native_kernel_set)
        if bool(self.native_metal_kernels) and coerce_native_kernel_set(self.native_kernel_set) == "off":
            raise ValueError("LTX native_metal_kernels requires native_kernel_set to be a concrete kernel set.")
        return self


@dataclass(frozen=True)
class MLXLTXGenerationResult:
    output_path: str
    stats_json: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _combined_ltx_context_to_torch(video_embeds: Any, audio_embeds: Any):
    import numpy as np
    import torch
    import mlx.core as mx

    video_shape = list(video_embeds.shape)
    audio_shape = list(audio_embeds.shape)
    if len(video_shape) != 3 or len(audio_shape) != 3:
        raise ValueError(f"MLX LTX context must be rank-3 video/audio embeddings; got {video_shape}/{audio_shape}.")
    if video_shape[:2] != audio_shape[:2]:
        raise ValueError(f"MLX LTX video/audio context batch-token shapes differ: {video_shape}/{audio_shape}.")
    if int(video_shape[-1]) != 4096 or int(audio_shape[-1]) != 2048:
        raise ValueError(f"MLX LTX context dims must be 4096+2048; got {video_shape[-1]}+{audio_shape[-1]}.")
    combined = mx.concatenate([video_embeds.astype(mx.float32), audio_embeds.astype(mx.float32)], axis=-1)
    mx.eval(combined)
    context = torch.from_numpy(np.asarray(combined)).to(dtype=torch.float32)
    return context, {
        "mlx_ltx_context_video_shape": video_shape,
        "mlx_ltx_context_audio_shape": audio_shape,
        "mlx_ltx_context_shape": list(context.shape),
    }


def _torch_latent_to_mlx(latent: Any, mx: Any, *, dtype: Any) -> Any:
    import torch

    latent_cpu = latent.detach().to("cpu", dtype=torch.float32).contiguous()
    return mx.array(latent_cpu.numpy()).astype(dtype)


def _mlx_video_pixels_to_torch_images(pixels: Any, *, np: Any, mx: Any, torch: Any):
    images = mx.clip((pixels.astype(mx.float32) + 1.0) * 0.5, 0.0, 1.0)
    images = images.transpose(0, 2, 3, 4, 1)
    return torch.from_numpy(np.asarray(images.astype(mx.float32))).to(dtype=torch.float32)


def _island_prompt_cache_key(
    model_path: str,
    spec: MLXLTXModelSpec,
    config: MLXLTXRunConfig,
    prompt: str,
) -> tuple[Any, ...]:
    return (
        "denoiser_island_context",
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        os.path.abspath(model_path),
        spec.repo_id,
        spec.revision or "",
        spec.variant,
        config.pipeline,
        config.gemma_model_id,
        MLXLTXBackend._package_version("ltx-pipelines-mlx"),
        MLXLTXBackend._package_version("ltx-core-mlx"),
        MLXLTXBackend._package_version("mlx"),
    )


class MLXLTXModelProxy:
    """Internal model proxy returned by CheckpointLoaderSimple for .mlx_ltx.json manifests."""

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
        prompt = str(tokens)
        if self.spec.metadata.get("execution_mode") == "legacy_media":
            return [[None, {"mlx_ltx_prompt": prompt, "prompt": prompt}]]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        if _looks_like_ltx_negative_prompt(prompt):
            import torch

            context = torch.zeros(1, 1024, 6144, dtype=torch.float32)
            write_event(
                "mlx_ltx_clip_encode_skip",
                prompt_hash=prompt_hash,
                reason="negative_prompt_cfg1_unused_zero_context",
                context_shape=list(context.shape),
            )
            return [
                [
                    context,
                    {
                        "mlx_ltx_prompt": prompt,
                        "prompt": prompt,
                        "conditioning_mode": "metadata_only_negative",
                        "mlx_ltx_context_skipped": "negative_prompt_cfg1_unused_zero_context",
                        "mlx_ltx_context_shape": list(context.shape),
                    },
                ]
            ]

        encode_start = now()
        write_event("mlx_ltx_clip_encode_start", prompt_hash=prompt_hash, prompt_chars=len(prompt))
        context, metadata = self._encode_prompt_context(prompt)
        write_event(
            "mlx_ltx_clip_encode_end",
            prompt_hash=prompt_hash,
            seconds=elapsed_seconds(encode_start),
            context_shape=list(context.shape) if hasattr(context, "shape") else None,
            prompt_cache_hit=metadata.get("prompt_cache_hit"),
        )
        return [[context, {"mlx_ltx_prompt": prompt, "prompt": prompt, **metadata}]]

    def _encode_prompt_context(self, prompt: str):
        backend = _get_mlx_ltx_backend()
        model_path = backend.resolve_model_path(self.spec)
        cache_key = _island_prompt_cache_key(model_path, self.spec, self.run_config, prompt)
        cached = backend._prompt_cache_get(cache_key, self.run_config.prompt_cache_size)
        if cached is not None:
            context, metadata = cached
            return context, {**metadata, "prompt_cache_hit": True}

        import mlx.core as mx
        from ltx_pipelines_mlx import DistilledPipeline

        pipe = DistilledPipeline(
            model_path,
            low_memory=True if self.run_config.low_memory is None else bool(self.run_config.low_memory),
            low_ram_streaming=bool(self.run_config.low_ram),
        )
        try:
            encode_start = now()
            pipe._load_text_encoder()
            video_embeds, audio_embeds = pipe._encode_text(prompt)
            mx.eval(video_embeds, audio_embeds)
            context, metadata = _combined_ltx_context_to_torch(video_embeds, audio_embeds)
            metadata = {
                **metadata,
                "conditioning_mode": "real",
                "prompt_cache_hit": False,
                "prompt_encode_seconds": elapsed_seconds(encode_start),
            }
            backend._prompt_cache_put(cache_key, (context, metadata), self.run_config.prompt_cache_size)
            return context, metadata
        finally:
            prompt_encoder = getattr(pipe, "prompt_encoder", None)
            if prompt_encoder is not None and hasattr(prompt_encoder, "free"):
                prompt_encoder.free()
            del pipe
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass


class MLXLTXVAEProxy:
    is_mlx_ltx_vae_proxy = True
    latent_channels = 128
    audio_sample_rate = 48000
    downscale_index_formula = (8, 32, 32)

    def __init__(self, spec: MLXLTXModelSpec, run_config: MLXLTXRunConfig):
        self.spec = spec
        self.run_config = run_config
        self.first_stage_model = _MLXLTXAudioVAEConfigProxy()

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 32

    def encode(self, pixels):
        import torch

        if not isinstance(pixels, torch.Tensor):
            raise TypeError("MLX LTX VAE encode expects a Torch image tensor.")
        if pixels.ndim != 4 or int(pixels.shape[-1]) < 3:
            raise ValueError(f"MLX LTX VAE encode expects [frames,height,width,channels]; got {tuple(pixels.shape)}.")
        return self._encode_video(pixels)

    def decode(self, latent):
        import torch

        if not isinstance(latent, torch.Tensor):
            raise TypeError("MLX LTX VAE decode expects a Torch latent tensor.")
        if latent.ndim == 5:
            return self._decode_video(latent, tiled=False)
        if latent.ndim == 4:
            return self._decode_audio(latent)
        raise ValueError(f"MLX LTX VAE cannot decode latent shape {tuple(latent.shape)}.")

    def _encode_video(self, pixels):
        import numpy as np
        import torch
        import mlx.core as mx

        pipe = self._decode_pipeline()
        try:
            pixel_cpu = pixels[..., :3].detach().to("cpu", dtype=torch.float32).contiguous()
            video_mx = mx.array(pixel_cpu.numpy())
            video_mx = (video_mx * 2.0 - 1.0).transpose(3, 0, 1, 2)[None, ...].astype(mx.bfloat16)
            encoder = pipe.image_conditioner.load()
            latent = encoder.encode(video_mx)
            mx.eval(latent)
            latent_torch = torch.from_numpy(np.asarray(latent.astype(mx.float32)))
            return latent_torch.to(dtype=pixels.dtype, device=pixels.device)
        finally:
            try:
                pipe.image_conditioner.free()
            except Exception:
                pass
            del pipe
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

    def decode_tiled(self, latent, *, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        import torch

        if not isinstance(latent, torch.Tensor):
            raise TypeError("MLX LTX VAE tiled decode expects a Torch latent tensor.")
        if latent.ndim != 5:
            raise ValueError(f"MLX LTX tiled video decode expects rank-5 latent; got {tuple(latent.shape)}.")
        return self._decode_video(latent, tiled=True)

    def _decode_video(self, latent, *, tiled: bool):
        import numpy as np
        import torch
        import mlx.core as mx

        pipe = self._decode_pipeline()
        try:
            video_mx = _torch_latent_to_mlx(latent, mx, dtype=mx.bfloat16)
            decoder = pipe.video_decoder_block.load()
            if tiled:
                from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling

                tiling = _compute_decode_tiling(video_mx.shape, frame_rate=float(self.run_config.frame_rate))
                chunks = []
                for chunk in decoder.tiled_decode(video_mx, tiling):
                    chunks.append(_mlx_video_pixels_to_torch_images(chunk, np=np, mx=mx, torch=torch))
                if not chunks:
                    raise RuntimeError("MLX LTX tiled video decode produced no frames.")
                images = torch.cat(chunks, dim=1)
            else:
                pixels = decoder.decode(video_mx)
                mx.eval(pixels)
                images = _mlx_video_pixels_to_torch_images(pixels, np=np, mx=mx, torch=torch)
            return images.to(device=latent.device)
        finally:
            del pipe
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

    def _decode_audio(self, latent):
        import numpy as np
        import torch
        import mlx.core as mx

        pipe = self._decode_pipeline()
        try:
            audio_mx = _torch_latent_to_mlx(latent, mx, dtype=mx.bfloat16)
            decoder, vocoder = pipe.audio_decoder_block.load()
            mel = decoder.decode(audio_mx)
            waveform = vocoder(mel)
            mx.eval(waveform)
            waveform_np = np.asarray(waveform.astype(mx.float32))
            waveform_torch = torch.from_numpy(waveform_np).to(dtype=torch.float32, device=latent.device)
            if waveform_torch.ndim != 3:
                raise RuntimeError(f"MLX LTX audio decode returned unexpected shape {tuple(waveform_torch.shape)}.")
            return waveform_torch.movedim(1, -1)
        finally:
            del pipe
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

    def _decode_pipeline(self):
        from ltx_pipelines_mlx import DistilledPipeline

        backend = _get_mlx_ltx_backend()
        model_path = backend.resolve_model_path(self.spec)
        return DistilledPipeline(
            model_path,
            low_memory=True if self.run_config.low_memory is None else bool(self.run_config.low_memory),
            low_ram_streaming=bool(self.run_config.low_ram),
        )


class MLXLTXLatentUpscalerProxy:
    is_mlx_ltx_latent_upscaler_proxy = True

    def __init__(self, spec: MLXLTXModelSpec, run_config: MLXLTXRunConfig):
        self.spec = spec
        self.run_config = run_config


class _MLXLTXLatentModelProxy:
    def process_latent_out(self, latent):
        return latent


class _MLXLTXSamplingProxy:
    def percent_to_sigma(self, percent: float) -> float:
        return float(percent)


class _MLXLTXAudioVAEConfigProxy:
    latent_frequency_bins = 16
    output_sample_rate = 48000

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        try:
            from ltx_core_mlx.utils.positions import compute_audio_token_count

            return int(compute_audio_token_count(int(frames_number), frame_rate=float(frame_rate)))
        except Exception:
            pass
        return max(1, int(round(float(frames_number) / max(float(frame_rate), 1.0) * 24)))


class MLXLTXBackend:
    name = "mlx_ltx"

    def __init__(self) -> None:
        self._generation_lock = threading.RLock()
        self._pipeline_cache: dict[tuple[Any, ...], Any] = {}
        self._resolved_path_cache: dict[tuple[Any, ...], str] = {}
        self._prompt_cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()

    def capabilities(self) -> BackendCapabilities:
        if not is_apple_silicon():
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason="MLX LTX backend requires Apple Silicon macOS.",
                modalities=("video", "video_audio"),
                device="unsupported",
                unified_memory=False,
            )
        try:
            self._import_mlx_core()
            self._import_distilled_pipeline()
        except BackendUnavailableError as e:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=str(e),
                modalities=("video", "video_audio"),
                device="apple_silicon",
                unified_memory=True,
            )
        return BackendCapabilities(
            name=self.name,
            available=True,
            reason="MLX core and ltx-pipelines-mlx are importable on Apple Silicon.",
            modalities=("video", "video_audio"),
            device="apple_silicon",
            unified_memory=True,
        )

    def _import_mlx_core(self):
        if not is_apple_silicon():
            raise BackendUnavailableError("MLX LTX backend requires Apple Silicon macOS.")
        try:
            import mlx.core as mx
        except Exception as e:
            raise BackendUnavailableError(
                "MLX is not installed. Install MLX plus the ltx-2-mlx packages in the local test environment."
            ) from e
        return mx

    def _import_distilled_pipeline(self):
        self._import_mlx_core()
        try:
            module = importlib.import_module("ltx_pipelines_mlx.distilled")
            return getattr(module, "DistilledPipeline")
        except Exception as e:
            raise BackendUnavailableError(
                "ltx-pipelines-mlx is not installed. Install dgrauet/ltx-2-mlx optional packages for LTX MLX tests."
            ) from e

    def resolve_model_path(self, spec: MLXLTXModelSpec, cache_dir: str | None = None) -> str:
        if spec.local_path:
            return os.path.abspath(os.path.expanduser(spec.local_path))
        repo_or_path = os.path.expanduser(spec.repo_id)
        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)
        cache_key = self._resolved_path_cache_key(spec, cache_dir)
        if cache_key in self._resolved_path_cache:
            return self._resolved_path_cache[cache_key]

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise BackendUnavailableError(
                "huggingface_hub is not installed. Install it with `pip install huggingface_hub`."
            ) from e

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
            self._resolved_path_cache[cache_key] = resolved
            return resolved
        except Exception as e:
            raise BackendUnavailableError(redact_secrets(e)) from e

    def generate_video_to_file(
        self,
        spec: MLXLTXModelSpec,
        prompt: str,
        output_path: str,
        *,
        width: int = 512,
        height: int = 512,
        num_frames: int = 49,
        frame_rate: float = 24.0,
        pipeline: str | None = None,
        stage1_steps: int = 8,
        stage2_steps: int = 3,
        seed: int = 42,
        low_memory: bool | None = None,
        low_ram: bool | None = None,
        tile_frames: int = 1,
        tile_spatial: int = 1,
        tile_overlap: int = 2,
        gemma_model_id: str | None = None,
        image: str | None = None,
        audio: str | None = None,
        cache_dir: str | None = None,
        media_passthrough: bool | None = None,
        prompt_cache_size: int | None = None,
        stage_snapshot_cache: bool | None = None,
        mlx_cache_limit_gb: float | None = None,
        profile_level: str | None = None,
        metal_capture: bool | None = None,
        block_profile_interval: int | None = None,
        eval_every: int | None = None,
        stage1_eval_every: int | None = None,
        stage2_eval_every: int | None = None,
        decode_profile: bool | None = None,
        compile_dit: bool | None = None,
        stage1_compile_dit: bool | None = None,
        stage2_compile_dit: bool | None = None,
        compile_block_stack: bool | None = None,
        compile_x0: bool | None = None,
        compile_shapeless: bool | None = None,
        fuse_attention_projections: bool | None = None,
        flatten_attention_projections: bool | None = None,
        dequantize_video_ffn: bool | None = None,
        native_metal_kernels: bool | None = None,
        native_kernel_profile: bool | None = None,
        native_kernel_verify: bool | None = None,
        native_kernel_fallback: bool | None = None,
        native_kernel_set: str | None = None,
        ffmpeg_video_encoder: str | None = None,
        ffmpeg_video_bitrate: str | None = None,
        cache_text_kv: bool | None = None,
        text_kv_cache_size: int | None = None,
        audio_debug: bool | None = None,
        audio_debug_dir: str | None = None,
        audio_debug_dump_arrays: bool | None = None,
        denoise_sync_policy: str | None = None,
        denoise_sync_interval: int | None = None,
    ) -> MLXLTXGenerationResult:
        run_config = MLXLTXRunConfig(
            prompt=prompt,
            output_path=output_path,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            pipeline=pipeline or spec.pipeline or "distilled",
            stage1_steps=int(stage1_steps),
            stage2_steps=int(stage2_steps),
            seed=int(seed),
            low_memory=low_memory if low_memory is not None else self._default_low_memory(spec),
            low_ram=bool(low_ram if low_ram is not None else self._default_low_ram(spec)),
            tile_frames=int(tile_frames),
            tile_spatial=int(tile_spatial),
            tile_overlap=int(tile_overlap),
            gemma_model_id=gemma_model_id or spec.gemma_model_id,
            image=image,
            audio=audio,
            media_passthrough=bool(
                media_passthrough if media_passthrough is not None else spec.metadata.get("media_passthrough", True)
            ),
            prompt_cache_size=int(prompt_cache_size if prompt_cache_size is not None else spec.metadata.get("prompt_cache_size", 8)),
            stage_snapshot_cache=bool(
                stage_snapshot_cache if stage_snapshot_cache is not None else spec.metadata.get("stage_snapshot_cache", False)
            ),
            mlx_cache_limit_gb=(
                float(mlx_cache_limit_gb)
                if mlx_cache_limit_gb is not None
                else (float(spec.metadata["mlx_cache_limit_gb"]) if spec.metadata.get("mlx_cache_limit_gb") is not None else None)
            ),
            profile_level=_coerce_profile_level(profile_level if profile_level is not None else spec.metadata.get("profile_level", "stage")),
            metal_capture=bool(
                metal_capture
                if metal_capture is not None
                else _coerce_optional_bool(spec.metadata.get("metal_capture", False))
            ),
            block_profile_interval=int(
                block_profile_interval
                if block_profile_interval is not None
                else spec.metadata.get("block_profile_interval", 8)
            ),
            eval_every=(
                int(eval_every)
                if eval_every is not None
                else _coerce_optional_int(spec.metadata.get("eval_every", None))
            ),
            stage1_eval_every=(
                int(stage1_eval_every)
                if stage1_eval_every is not None
                else _coerce_optional_int(spec.metadata.get("stage1_eval_every", None))
            ),
            stage2_eval_every=(
                int(stage2_eval_every)
                if stage2_eval_every is not None
                else _coerce_optional_int(spec.metadata.get("stage2_eval_every", None))
            ),
            decode_profile=bool(
                decode_profile
                if decode_profile is not None
                else _coerce_optional_bool(spec.metadata.get("decode_profile", False))
            ),
            compile_dit=bool(
                compile_dit
                if compile_dit is not None
                else _coerce_optional_bool(spec.metadata.get("compile_dit", False))
            ),
            stage1_compile_dit=(
                bool(stage1_compile_dit)
                if stage1_compile_dit is not None
                else _coerce_optional_bool(spec.metadata.get("stage1_compile_dit", None))
            ),
            stage2_compile_dit=(
                bool(stage2_compile_dit)
                if stage2_compile_dit is not None
                else _coerce_optional_bool(spec.metadata.get("stage2_compile_dit", None))
            ),
            compile_block_stack=bool(
                compile_block_stack
                if compile_block_stack is not None
                else _coerce_optional_bool(spec.metadata.get("compile_block_stack", False))
            ),
            compile_x0=bool(
                compile_x0
                if compile_x0 is not None
                else _coerce_optional_bool(spec.metadata.get("compile_x0", False))
            ),
            compile_shapeless=bool(
                compile_shapeless
                if compile_shapeless is not None
                else _coerce_optional_bool(spec.metadata.get("compile_shapeless", False))
            ),
            fuse_attention_projections=bool(
                fuse_attention_projections
                if fuse_attention_projections is not None
                else _coerce_optional_bool(spec.metadata.get("fuse_attention_projections", False))
            ),
            flatten_attention_projections=bool(
                flatten_attention_projections
                if flatten_attention_projections is not None
                else _coerce_optional_bool(spec.metadata.get("flatten_attention_projections", False))
            ),
            dequantize_video_ffn=bool(
                dequantize_video_ffn
                if dequantize_video_ffn is not None
                else _coerce_optional_bool(spec.metadata.get("dequantize_video_ffn", False))
            ),
            native_metal_kernels=bool(
                native_metal_kernels
                if native_metal_kernels is not None
                else _coerce_optional_bool(spec.metadata.get("native_metal_kernels", False))
            ),
            native_kernel_profile=bool(
                native_kernel_profile
                if native_kernel_profile is not None
                else _coerce_optional_bool(spec.metadata.get("native_kernel_profile", False))
            ),
            native_kernel_verify=bool(
                native_kernel_verify
                if native_kernel_verify is not None
                else _coerce_optional_bool(spec.metadata.get("native_kernel_verify", True))
            ),
            native_kernel_fallback=bool(
                native_kernel_fallback
                if native_kernel_fallback is not None
                else _coerce_optional_bool(spec.metadata.get("native_kernel_fallback", True))
            ),
            native_kernel_set=coerce_native_kernel_set(
                native_kernel_set if native_kernel_set is not None else spec.metadata.get("native_kernel_set", "off")
            ),
            ffmpeg_video_encoder=str(
                ffmpeg_video_encoder
                if ffmpeg_video_encoder is not None
                else spec.metadata.get("ffmpeg_video_encoder", "libx264")
            ),
            ffmpeg_video_bitrate=str(
                ffmpeg_video_bitrate
                if ffmpeg_video_bitrate is not None
                else spec.metadata.get("ffmpeg_video_bitrate", "12M")
            ),
            cache_text_kv=bool(
                cache_text_kv if cache_text_kv is not None else _coerce_optional_bool(spec.metadata.get("cache_text_kv", False))
            ),
            text_kv_cache_size=int(
                text_kv_cache_size if text_kv_cache_size is not None else spec.metadata.get("text_kv_cache_size", 2048)
            ),
            audio_debug=bool(
                audio_debug
                if audio_debug is not None
                else _coerce_optional_bool(spec.metadata.get("audio_debug", False))
            ),
            audio_debug_dir=audio_debug_dir if audio_debug_dir is not None else spec.metadata.get("audio_debug_dir", None),
            audio_debug_dump_arrays=bool(
                audio_debug_dump_arrays
                if audio_debug_dump_arrays is not None
                else _coerce_optional_bool(spec.metadata.get("audio_debug_dump_arrays", True))
            ),
            denoise_sync_policy=_coerce_denoise_sync_policy(
                denoise_sync_policy if denoise_sync_policy is not None else spec.metadata.get("denoise_sync_policy", "each_step")
            ),
            denoise_sync_interval=int(
                denoise_sync_interval
                if denoise_sync_interval is not None
                else spec.metadata.get("denoise_sync_interval", 2)
            ),
        ).validated()
        if run_config.pipeline != "distilled":
            raise BackendUnavailableError("Native MLX LTX benchmarking currently supports only the distilled pipeline.")

        output = Path(run_config.output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        resolve_start = now()
        model_path = self.resolve_model_path(spec, cache_dir=cache_dir)
        DistilledPipeline = self._import_distilled_pipeline()
        mx = self._import_mlx_core()
        self._apply_mlx_cache_limit(mx, run_config.mlx_cache_limit_gb)

        model_metadata = {
            "backend": self.name,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "variant": spec.variant,
            "pipeline": run_config.pipeline,
            "resolved_path": model_path,
            "width": run_config.width,
            "height": run_config.height,
            "num_frames": run_config.num_frames,
            "frame_rate": run_config.frame_rate,
            "duration_seconds": ltx_duration_seconds(run_config.num_frames, run_config.frame_rate),
            "stage1_steps": run_config.stage1_steps,
            "stage2_steps": run_config.stage2_steps,
            "low_memory": run_config.low_memory,
            "low_ram": run_config.low_ram,
            "tile_frames": run_config.tile_frames,
            "tile_spatial": run_config.tile_spatial,
            "gemma_model_id": run_config.gemma_model_id,
            "media_passthrough": run_config.media_passthrough,
            "prompt_cache_size": run_config.prompt_cache_size,
            "stage_snapshot_cache": run_config.stage_snapshot_cache,
            "mlx_cache_limit_gb": run_config.mlx_cache_limit_gb,
            "profile_level": run_config.profile_level,
            "metal_capture": run_config.metal_capture,
            "block_profile_interval": run_config.block_profile_interval,
            "eval_every": run_config.eval_every,
            "stage1_eval_every": run_config.stage1_eval_every,
            "stage2_eval_every": run_config.stage2_eval_every,
            "decode_profile": run_config.decode_profile,
            "compile_dit": run_config.compile_dit,
            "stage1_compile_dit": run_config.stage1_compile_dit,
            "stage2_compile_dit": run_config.stage2_compile_dit,
            "compile_block_stack": run_config.compile_block_stack,
            "compile_x0": run_config.compile_x0,
            "compile_shapeless": run_config.compile_shapeless,
            "fuse_attention_projections": run_config.fuse_attention_projections,
            "flatten_attention_projections": run_config.flatten_attention_projections,
            "dequantize_video_ffn": run_config.dequantize_video_ffn,
            "native_metal_kernels": run_config.native_metal_kernels,
            "native_kernel_profile": run_config.native_kernel_profile,
            "native_kernel_verify": run_config.native_kernel_verify,
            "native_kernel_fallback": run_config.native_kernel_fallback,
            "native_kernel_set": run_config.native_kernel_set,
            "ffmpeg_video_encoder": run_config.ffmpeg_video_encoder,
            "ffmpeg_video_bitrate": run_config.ffmpeg_video_bitrate,
            "cache_text_kv": run_config.cache_text_kv,
            "text_kv_cache_size": run_config.text_kv_cache_size,
            "audio_debug": run_config.audio_debug,
            "audio_debug_dir": self._audio_debug_dir(run_config, output) if self._audio_debug_enabled(run_config) else "",
            "audio_debug_dump_arrays": run_config.audio_debug_dump_arrays,
            "denoise_sync_policy": run_config.denoise_sync_policy,
            "denoise_sync_interval": run_config.denoise_sync_interval,
            "backend_device": "mlx_metal",
            "mlx_device": self._mlx_device_name(mx),
            "metal_available": True,
        }
        native_runtime = make_native_kernel_runtime(
            native_metal_kernels=run_config.native_metal_kernels,
            native_kernel_set=run_config.native_kernel_set,
            native_kernel_profile=run_config.native_kernel_profile,
            native_kernel_verify=run_config.native_kernel_verify,
            native_kernel_fallback=run_config.native_kernel_fallback,
        )
        self._write_stage_end("model_path_resolve", resolve_start, model_metadata)

        load_start = now()
        write_event("load_start", model=model_metadata)
        try:
            lookup_start = now()
            cache_key = self._pipeline_cache_key(model_path, spec, run_config)
            with self._generation_lock:
                pipe = self._pipeline_cache.get(cache_key)
                cache_hit = pipe is not None
                if pipe is None:
                    pipe_kwargs = {
                        "model_dir": model_path,
                        "gemma_model_id": run_config.gemma_model_id,
                        "low_memory": run_config.low_memory,
                        "low_ram_streaming": run_config.low_ram,
                        "tile_count": self._tile_count(run_config),
                    }
                    pipe = _call_with_supported_kwargs(DistilledPipeline, **pipe_kwargs)
                    self._pipeline_cache[cache_key] = pipe
            self._write_stage_end(
                "pipeline_cache_lookup",
                lookup_start,
                {**model_metadata, "pipeline_cache_hit": cache_hit},
            )
        except Exception as e:
            write_event("load_error", model=model_metadata, error=redact_secrets(e), seconds=elapsed_seconds(load_start))
            raise BackendUnavailableError(redact_secrets(e)) from e
        write_event(
            "load_end",
            model={**model_metadata, "pipeline_cache_hit": cache_hit},
            seconds=elapsed_seconds(load_start),
            memory=self.memory_stats().stats,
        )

        generation_start = now()
        native_runtime.record_status(model_metadata)
        native_runtime.verify_microbench()
        model_metadata.update(native_runtime.metadata())
        write_event("generate_start", model=model_metadata)
        metal_capture_started = False
        try:
            generate_kwargs = {
                "prompt": prompt,
                "output_path": str(output),
                "height": run_config.height,
                "width": run_config.width,
                "num_frames": run_config.num_frames,
                "frame_rate": run_config.frame_rate,
                "seed": run_config.seed,
                "stage1_steps": run_config.stage1_steps,
                "stage2_steps": run_config.stage2_steps,
                "image": run_config.image,
                "audio": run_config.audio,
                "denoise_sync_policy": run_config.denoise_sync_policy,
                "denoise_sync_interval": run_config.denoise_sync_interval,
            }
            with self._generation_lock:
                metal_capture_started = self._start_metal_capture(mx, model_metadata, run_config)
                generated = self._generate_with_stage_instrumentation(
                    pipe=pipe,
                    run_config=run_config,
                    model_metadata=model_metadata,
                    model_path=model_path,
                    spec=spec,
                    generate_kwargs=generate_kwargs,
                    output=output,
                    native_runtime=native_runtime,
                )
                if generated:
                    output = Path(str(generated)).expanduser().resolve()
        except BaseException as e:
            write_event("generate_error", model=model_metadata, error=redact_secrets(e), seconds=elapsed_seconds(generation_start))
            if e.__class__.__name__ == "InterruptProcessingException":
                raise
            raise BackendUnavailableError(redact_secrets(e)) from e
        finally:
            if metal_capture_started:
                self._stop_metal_capture(mx, model_metadata)

        generation_seconds = elapsed_seconds(generation_start)
        model_metadata.update(self._collect_compile_counters(pipe))
        model_metadata.update(native_runtime.metadata())
        stats = {
            **model_metadata,
            "output_path": str(output),
            "generation_seconds": generation_seconds,
            "pipeline_cache_size": len(self._pipeline_cache),
            "prompt_cache_size": len(self._prompt_cache),
            "memory": self.memory_stats().stats,
        }
        write_event(
            "generate_end",
            model=model_metadata,
            seconds=generation_seconds,
            media_path=str(output),
            output_file_size=output.stat().st_size if output.exists() else 0,
            media_passthrough=run_config.media_passthrough,
            prompt_cache_size=run_config.prompt_cache_size,
            memory=stats["memory"],
        )
        return MLXLTXGenerationResult(output_path=str(output), stats_json=json.dumps(stats, indent=2, sort_keys=True), metadata=stats)

    def _collect_compile_counters(self, pipe: Any) -> dict[str, int]:
        totals = {
            "compile_block_stack_used_count": 0,
            "compile_block_stack_fallback_count": 0,
        }
        seen: set[int] = set()
        candidates = [
            pipe,
            getattr(pipe, "dit", None),
            getattr(getattr(pipe, "dit", None), "model", None),
            getattr(getattr(getattr(pipe, "dit", None), "model", None), "model", None),
        ]
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            for key in tuple(totals):
                value = getattr(candidate, f"_mlx_ltx_{key}", None)
                if value is not None:
                    try:
                        totals[key] += int(value)
                    except Exception:
                        pass
        return totals

    def output_as_comfy_video(self, output_path: str):
        try:
            from comfy_api.latest import InputImpl
        except Exception as e:
            raise BackendUnavailableError("Comfy video input helpers are not importable in this environment.") from e
        return InputImpl.VideoFromFile(output_path)

    def output_components(self, output_path: str):
        start = now()
        try:
            return self.output_as_comfy_video(output_path).get_components()
        finally:
            self._write_stage_end("media_materialize", start, {"backend": self.name, "media_path": output_path})

    def _default_low_ram(self, spec: MLXLTXModelSpec) -> bool:
        variant = (spec.variant or spec.metadata.get("variant") or "").lower()
        if spec.metadata.get("low_ram_streaming") is not None:
            return bool(spec.metadata.get("low_ram_streaming"))
        if self._unified_memory_gb() >= 64:
            return False
        return variant in {"q8", "bf16", "fp16"}

    def _default_low_memory(self, spec: MLXLTXModelSpec) -> bool:
        if spec.metadata.get("low_memory") is not None:
            return bool(spec.metadata.get("low_memory"))
        variant = (spec.variant or spec.metadata.get("variant") or "").lower()
        if self._unified_memory_gb() >= 64 and variant in {"q4", "q8"}:
            return False
        return variant in {"q8", "bf16", "fp16"}

    def _unified_memory_gb(self) -> float:
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0.0

    def _pipeline_cache_key(self, model_path: str, spec: MLXLTXModelSpec, config: MLXLTXRunConfig) -> tuple[Any, ...]:
        return (
            os.path.abspath(model_path),
            spec.revision or "",
            spec.variant,
            config.pipeline,
            bool(config.low_memory),
            bool(config.low_ram),
            config.tile_frames,
            config.tile_spatial,
            config.tile_overlap,
            config.gemma_model_id,
            bool(config.dequantize_video_ffn),
            bool(config.native_metal_kernels),
            config.native_kernel_set,
        )

    def _tile_count(self, config: MLXLTXRunConfig):
        if config.tile_frames <= 1 and config.tile_spatial <= 1:
            return None
        try:
            module = importlib.import_module("ltx_core_mlx.tiling")
            tile_count_config = getattr(module, "TileCountConfig")
            dimension_config = getattr(module, "DimensionTilingConfig")
        except Exception as e:
            raise BackendUnavailableError("ltx-core-mlx tiling helpers are not importable; disable LTX tiling for this run.") from e
        return tile_count_config(
            frames=dimension_config(count=config.tile_frames, overlap=config.tile_overlap),
            height=dimension_config(count=config.tile_spatial, overlap=config.tile_overlap),
            width=dimension_config(count=config.tile_spatial, overlap=config.tile_overlap),
        )

    def _mlx_device_name(self, mx) -> str:
        try:
            default_device = getattr(mx, "default_device", None)
            if callable(default_device):
                return str(default_device())
        except Exception:
            pass
        return "mlx"

    def unload_all(self) -> None:
        self._pipeline_cache.clear()
        self._prompt_cache.clear()
        self.empty_cache()

    def empty_cache(self) -> None:
        self.clear_mlx_cache()
        gc.collect()

    def clear_mlx_cache(self) -> None:
        try:
            mx = self._import_mlx_core()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
                return
            metal = getattr(mx, "metal", None)
            if metal is not None and hasattr(metal, "clear_cache"):
                metal.clear_cache()
        except Exception:
            pass

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
        payload = {
            "capabilities": asdict(self.capabilities()),
            "memory": self.memory_stats().stats,
            "platform": platform.platform(),
            "serialized_execution": True,
            "user_facing_nodes": False,
            "resident_pipeline_cache_size": len(self._pipeline_cache),
            "prompt_cache_size": len(self._prompt_cache),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _generate_with_stage_instrumentation(
        self,
        *,
        pipe: Any,
        run_config: MLXLTXRunConfig,
        model_metadata: dict[str, Any],
        model_path: str,
        spec: MLXLTXModelSpec,
        generate_kwargs: dict[str, Any],
        output: Path,
        native_runtime: Any,
    ) -> str | None:
        if not (hasattr(pipe, "generate_two_stage") and hasattr(pipe, "_decode_and_save_video")):
            return _call_with_supported_kwargs(pipe.generate_and_save, **generate_kwargs)

        prompt_cache_key = self._prompt_cache_key(model_path, spec, run_config)
        cached_embeds = self._prompt_cache_get(prompt_cache_key, run_config.prompt_cache_size)
        prompt_cache_hit = cached_embeds is not None
        captured_embeds: dict[str, tuple[Any, Any]] = {}
        original_encode = getattr(pipe, "_encode_text")
        original_load_text_encoder = getattr(pipe, "_load_text_encoder", None)
        original_load = getattr(pipe, "load", None)
        module = importlib.import_module(pipe.__class__.__module__)
        original_denoise_loop = getattr(module, "denoise_loop", None)
        self._configure_pipeline_profile(pipe, run_config, model_metadata, native_runtime)

        def timed_encode(prompt: str):
            encode_start = now()
            if prompt_cache_hit and prompt == run_config.prompt and cached_embeds is not None:
                result = cached_embeds
            else:
                result = original_encode(prompt)
                self._materialize_mlx_arrays(result)
                if prompt == run_config.prompt:
                    captured_embeds["value"] = result
            self._write_stage_end(
                "prompt_encode",
                encode_start,
                {**model_metadata, "prompt_cache_hit": prompt_cache_hit},
            )
            if isinstance(result, (tuple, list)) and len(result) > 1:
                self._write_audio_debug_array("audio_embeds", result[1], run_config, model_metadata)
            return result

        def no_load_text_encoder(*args, **kwargs):
            return None

        def timed_load_text_encoder(*args, **kwargs):
            if original_load_text_encoder is None:
                return None
            load_start = now()
            result = original_load_text_encoder(*args, **kwargs)
            self._write_stage_end("text_encoder_load", load_start, model_metadata)
            return result

        denoise_counter = {"count": 0}

        def timed_denoise_loop(*args, **kwargs):
            denoise_counter["count"] += 1
            stage = "stage1_denoise" if denoise_counter["count"] == 1 else "stage2_denoise"
            model_arg = kwargs.get("model")
            if model_arg is None and args:
                model_arg = args[0]
            self._configure_denoise_model_profile(model_arg, run_config, stage, model_metadata, native_runtime)
            audio_state = kwargs.get("audio_state")
            if audio_state is None and len(args) > 2:
                audio_state = args[2]
            self._write_audio_debug_array(f"{stage}_audio_state_in", audio_state, run_config, model_metadata)
            stage_start = now()
            with native_runtime.patch_ltx_modules():
                result = original_denoise_loop(*args, **kwargs)
            self._materialize_denoise_result(result)
            if hasattr(result, "audio_latent"):
                self._write_audio_debug_array(f"{stage}_audio_latent_out", getattr(result, "audio_latent"), run_config, model_metadata)
            self._write_stage_end(stage, stage_start, model_metadata)
            return result

        def wrap_upsampler_once():
            upsampler = getattr(pipe, "upsampler", None)
            if upsampler is not None:
                if getattr(upsampler, "_mlx_ltx_timed_wrapper", False):
                    upsampler = getattr(upsampler, "_wrapped")
                setattr(pipe, "upsampler", _TimedCallable(upsampler, self, "upsample", model_metadata))

        def timed_load(*args, **kwargs):
            load_start = now()
            result = original_load(*args, **kwargs)
            wrap_upsampler_once()
            self._write_stage_end("transformer_load", load_start, model_metadata)
            return result

        try:
            setattr(pipe, "_encode_text", timed_encode)
            if prompt_cache_hit and original_load_text_encoder is not None:
                setattr(pipe, "_load_text_encoder", no_load_text_encoder)
            elif original_load_text_encoder is not None:
                setattr(pipe, "_load_text_encoder", timed_load_text_encoder)
            if original_load is not None:
                setattr(pipe, "load", timed_load)
                wrap_upsampler_once()
            if original_denoise_loop is not None:
                setattr(module, "denoise_loop", timed_denoise_loop)

            video_latent, audio_latent = _call_with_supported_kwargs(
                pipe.generate_two_stage,
                prompt=run_config.prompt,
                height=run_config.height,
                width=run_config.width,
                num_frames=run_config.num_frames,
                frame_rate=run_config.frame_rate,
                seed=run_config.seed,
                stage1_steps=run_config.stage1_steps,
                stage2_steps=run_config.stage2_steps,
                image=run_config.image,
                audio=run_config.audio,
                denoise_sync_policy=run_config.denoise_sync_policy,
                denoise_sync_interval=run_config.denoise_sync_interval,
            )
            if not prompt_cache_hit and "value" in captured_embeds:
                self._prompt_cache_put(prompt_cache_key, captured_embeds["value"], run_config.prompt_cache_size)
            self._write_audio_debug_array("final_audio_latent", audio_latent, run_config, model_metadata)

            old_video_encoder = os.environ.get("LTX2_FFMPEG_VIDEO_ENCODER")
            old_video_bitrate = os.environ.get("LTX2_FFMPEG_VIDEO_BITRATE")
            if run_config.ffmpeg_video_encoder:
                os.environ["LTX2_FFMPEG_VIDEO_ENCODER"] = run_config.ffmpeg_video_encoder
            if run_config.ffmpeg_video_bitrate:
                os.environ["LTX2_FFMPEG_VIDEO_BITRATE"] = run_config.ffmpeg_video_bitrate
            try:
                decode_start = now()
                if self._audio_debug_enabled(run_config):
                    generated = self._debug_decode_and_save_video(
                        pipe=pipe,
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        output=output,
                        run_config=run_config,
                        model_metadata=model_metadata,
                    )
                else:
                    generated = _call_with_supported_kwargs(
                        pipe._decode_and_save_video,
                        video_latent,
                        audio_latent,
                        str(output),
                        frame_rate=run_config.frame_rate,
                    )
                self._write_stage_end("backend_decode_mux", decode_start, model_metadata)
            finally:
                if old_video_encoder is None:
                    os.environ.pop("LTX2_FFMPEG_VIDEO_ENCODER", None)
                else:
                    os.environ["LTX2_FFMPEG_VIDEO_ENCODER"] = old_video_encoder
                if old_video_bitrate is None:
                    os.environ.pop("LTX2_FFMPEG_VIDEO_BITRATE", None)
                else:
                    os.environ["LTX2_FFMPEG_VIDEO_BITRATE"] = old_video_bitrate
            return generated or str(output)
        finally:
            setattr(pipe, "_encode_text", original_encode)
            if original_load_text_encoder is not None:
                setattr(pipe, "_load_text_encoder", original_load_text_encoder)
            if original_load is not None:
                setattr(pipe, "load", original_load)
            if original_denoise_loop is not None:
                setattr(module, "denoise_loop", original_denoise_loop)

    def _write_stage_end(self, stage: str, start: float, model_metadata: dict[str, Any]) -> None:
        write_event(
            "mlx_ltx_stage_end",
            stage=stage,
            seconds=elapsed_seconds(start),
            model=model_metadata,
            memory=self.memory_stats().stats,
        )

    def _audio_debug_enabled(self, config: MLXLTXRunConfig) -> bool:
        env = os.environ.get("LTX_MLX_AUDIO_DEBUG", "").strip().lower()
        if env in {"1", "true", "yes", "on"}:
            return True
        if env in {"0", "false", "no", "off"}:
            return False
        return bool(config.audio_debug)

    def _audio_debug_dir(self, config: MLXLTXRunConfig, output: Path) -> str:
        configured = (config.audio_debug_dir or os.environ.get("LTX_MLX_AUDIO_DEBUG_DIR", "")).strip()
        if configured:
            return str(Path(configured).expanduser().resolve())
        stats_file = os.environ.get("COMFY_BENCH_STATS_PATH", "").strip()
        if stats_file:
            return str((Path(stats_file).expanduser().resolve().parent / "mlx_audio_debug" / output.stem).resolve())
        return str((output.parent / "mlx_audio_debug" / output.stem).resolve())

    def _debug_decode_and_save_video(
        self,
        *,
        pipe: Any,
        video_latent: Any,
        audio_latent: Any,
        output: Path,
        run_config: MLXLTXRunConfig,
        model_metadata: dict[str, Any],
    ) -> str:
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_pipelines_mlx.utils._orchestration import save_waveform

        debug_dir = Path(self._audio_debug_dir(run_config, output))
        debug_dir.mkdir(parents=True, exist_ok=True)

        audio_decode_start = now()
        decoder, vocoder = pipe.audio_decoder_block.load()
        mel = decoder.decode(audio_latent)
        self._materialize_mlx_arrays(mel)
        self._write_audio_debug_array("decoded_mel", mel, run_config, model_metadata)
        waveform = vocoder(mel)
        self._materialize_mlx_arrays(waveform)
        self._write_audio_debug_array("decoded_waveform", waveform, run_config, model_metadata)
        self._write_stage_end("audio_decode", audio_decode_start, model_metadata)
        if run_config.low_memory:
            aggressive_cleanup()

        wav_path = debug_dir / f"{output.stem}.audio.wav"
        save_start = now()
        save_waveform(waveform, str(wav_path), sample_rate=48000)
        self._write_stage_end("save_waveform", save_start, model_metadata)
        self._write_audio_debug_artifact("temp_wav", wav_path, run_config, model_metadata)

        mux_start = now()
        pipe.video_decoder_block.decode_and_stream(
            video_latent,
            str(output),
            frame_rate=run_config.frame_rate,
            audio_path=str(wav_path),
        )
        self._write_stage_end("video_decode_mux", mux_start, model_metadata)
        self._write_audio_debug_artifact("final_mp4", output, run_config, model_metadata)

        cleanup_start = now()
        if run_config.low_memory:
            aggressive_cleanup()
        self._write_stage_end("decode_cleanup", cleanup_start, model_metadata)
        return str(output)

    def _write_audio_debug_array(
        self,
        name: str,
        value: Any,
        config: MLXLTXRunConfig,
        model_metadata: dict[str, Any],
    ) -> None:
        if not self._audio_debug_enabled(config) or value is None:
            return
        if hasattr(value, "latent"):
            value = getattr(value, "latent")
        output = Path(config.output_path).expanduser().resolve()
        debug_dir = Path(self._audio_debug_dir(config, output))
        debug_dir.mkdir(parents=True, exist_ok=True)
        stats, np_value = self._audio_debug_array_stats(value)
        if not stats:
            return
        artifact_path = ""
        if config.audio_debug_dump_arrays and np_value is not None:
            artifact = debug_dir / f"{name}.npz"
            try:
                import numpy as np

                np.savez_compressed(artifact, value=np_value)
                artifact_path = str(artifact)
            except Exception as e:
                stats["dump_error"] = redact_secrets(e)
        payload = {
            "name": name,
            "debug_dir": str(debug_dir),
            "path": artifact_path,
            "model": model_metadata,
            **stats,
        }
        self._append_audio_debug_jsonl(debug_dir, payload)
        write_event("mlx_ltx_audio_debug", **payload)

    def _write_audio_debug_artifact(
        self,
        name: str,
        path: Path,
        config: MLXLTXRunConfig,
        model_metadata: dict[str, Any],
    ) -> None:
        if not self._audio_debug_enabled(config):
            return
        output = Path(config.output_path).expanduser().resolve()
        debug_dir = Path(self._audio_debug_dir(config, output))
        payload = {
            "name": name,
            "debug_dir": str(debug_dir),
            "path": str(path),
            "model": model_metadata,
            "file_size": path.stat().st_size if path.exists() else 0,
        }
        self._append_audio_debug_jsonl(debug_dir, payload)
        write_event("mlx_ltx_audio_debug", **payload)

    def _audio_debug_array_stats(self, value: Any) -> tuple[dict[str, Any], Any | None]:
        try:
            import numpy as np

            mx = self._import_mlx_core()
            self._materialize_mlx_arrays(value)
            np_value = np.array(value.astype(mx.float32), dtype=np.float32)
            finite = np_value[np.isfinite(np_value)]
            stats: dict[str, Any] = {
                "shape": [int(dim) for dim in getattr(value, "shape", ())],
                "dtype": str(getattr(value, "dtype", "")),
                "size": int(np_value.size),
                "nonfinite_count": int(np_value.size - finite.size),
            }
            if finite.size:
                stats.update(
                    {
                        "min": round(float(np.min(finite)), 8),
                        "max": round(float(np.max(finite)), 8),
                        "mean": round(float(np.mean(finite)), 8),
                        "rms": round(float(np.sqrt(np.mean(np.square(finite, dtype=np.float64)))), 8),
                        "absmax": round(float(np.max(np.abs(finite))), 8),
                    }
                )
            return stats, np_value
        except Exception as e:
            return {"stats_error": redact_secrets(e)}, None

    @staticmethod
    def _append_audio_debug_jsonl(debug_dir: Path, payload: dict[str, Any]) -> None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            with (debug_dir / "audio_debug.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        except Exception:
            pass

    def _configure_pipeline_profile(
        self,
        pipe: Any,
        config: MLXLTXRunConfig,
        model_metadata: dict[str, Any],
        native_runtime: Any | None = None,
    ) -> None:
        callback = self._make_profile_callback(model_metadata) if self._profile_callback_enabled(config) else None
        for target in (pipe, getattr(pipe, "dit", None)):
            self._set_profile_attrs(target, config, callback, stage=None, native_runtime=native_runtime)

    def _configure_denoise_model_profile(
        self,
        model: Any,
        config: MLXLTXRunConfig,
        stage: str,
        model_metadata: dict[str, Any],
        native_runtime: Any | None = None,
    ) -> None:
        callback = self._make_profile_callback(model_metadata) if self._profile_callback_enabled(config) else None
        targets = [model, getattr(model, "model", None)]
        inner = getattr(getattr(model, "model", None), "model", None)
        if inner is not None:
            targets.append(inner)
        for target in targets:
            self._set_profile_attrs(target, config, callback, stage=stage, native_runtime=native_runtime)

    def _set_profile_attrs(
        self,
        target: Any,
        config: MLXLTXRunConfig,
        callback: Any,
        stage: str | None,
        native_runtime: Any | None = None,
    ) -> None:
        if target is None:
            return
        stage_eval_every = config.eval_every
        if stage == "stage1_denoise" and config.stage1_eval_every is not None:
            stage_eval_every = config.stage1_eval_every
        elif stage == "stage2_denoise" and config.stage2_eval_every is not None:
            stage_eval_every = config.stage2_eval_every
        stage_compile_dit = config.compile_dit
        if stage == "stage1_denoise" and config.stage1_compile_dit is not None:
            stage_compile_dit = config.stage1_compile_dit
        elif stage == "stage2_denoise" and config.stage2_compile_dit is not None:
            stage_compile_dit = config.stage2_compile_dit
        try:
            setattr(target, "_mlx_ltx_profile_callback", callback)
            setattr(target, "_mlx_ltx_profile_level", config.profile_level)
            setattr(target, "_mlx_ltx_profile_stage", stage)
            setattr(target, "_mlx_ltx_block_profile_interval", int(config.block_profile_interval))
            if stage_eval_every is None:
                if hasattr(target, "_mlx_ltx_eval_every"):
                    delattr(target, "_mlx_ltx_eval_every")
            else:
                setattr(target, "_mlx_ltx_eval_every", int(stage_eval_every))
            setattr(target, "_mlx_ltx_stage1_eval_every", config.stage1_eval_every)
            setattr(target, "_mlx_ltx_stage2_eval_every", config.stage2_eval_every)
            setattr(target, "_mlx_ltx_decode_profile", bool(config.decode_profile))
            setattr(target, "_mlx_ltx_compile_dit", bool(stage_compile_dit))
            setattr(target, "_mlx_ltx_stage1_compile_dit", config.stage1_compile_dit)
            setattr(target, "_mlx_ltx_stage2_compile_dit", config.stage2_compile_dit)
            setattr(target, "_mlx_ltx_compile_block_stack", bool(config.compile_block_stack))
            setattr(target, "_mlx_ltx_compile_x0", bool(config.compile_x0))
            setattr(target, "_mlx_ltx_compile_shapeless", bool(config.compile_shapeless))
            setattr(target, "_mlx_ltx_fuse_attention_projections", bool(config.fuse_attention_projections))
            setattr(target, "_mlx_ltx_flatten_attention_projections", bool(config.flatten_attention_projections))
            setattr(target, "_mlx_ltx_dequantize_video_ffn", bool(config.dequantize_video_ffn))
            setattr(target, "_mlx_ltx_native_metal_kernels", bool(config.native_metal_kernels))
            setattr(target, "_mlx_ltx_native_kernel_profile", bool(config.native_kernel_profile))
            setattr(target, "_mlx_ltx_native_kernel_verify", bool(config.native_kernel_verify))
            setattr(target, "_mlx_ltx_native_kernel_fallback", bool(config.native_kernel_fallback))
            setattr(target, "_mlx_ltx_native_kernel_set", config.native_kernel_set)
            if native_runtime is not None:
                native_runtime.install_on(target)
            setattr(target, "_mlx_ltx_cache_text_kv", bool(config.cache_text_kv))
            setattr(target, "_mlx_ltx_text_kv_cache_size", int(config.text_kv_cache_size))
            prompt_hash = hashlib.sha256(config.prompt.encode("utf-8")).hexdigest()
            setattr(
                target,
                "_mlx_ltx_text_kv_cache_scope",
                (
                    prompt_hash,
                    int(config.width),
                    int(config.height),
                    int(config.num_frames),
                    float(config.frame_rate),
                    str(config.gemma_model_id),
                ),
            )
        except Exception:
            pass

    def _profile_callback_enabled(self, config: MLXLTXRunConfig) -> bool:
        return config.profile_level in {"block_group", "block_deep"} or bool(config.decode_profile)

    def _make_profile_callback(self, model_metadata: dict[str, Any]):
        def profile_callback(event: Any) -> None:
            payload = dict(event) if isinstance(event, dict) else {"message": redact_secrets(event)}
            profile_name = str(payload.pop("profile_name", payload.pop("name", "unknown")))
            seconds = payload.pop("seconds", None)
            write_event(
                "mlx_ltx_profile_event",
                profile_name=profile_name,
                seconds=seconds,
                profile_level=model_metadata.get("profile_level"),
                model=model_metadata,
                memory=self.memory_stats().stats,
                **payload,
            )

        return profile_callback

    def _start_metal_capture(self, mx: Any, model_metadata: dict[str, Any], config: MLXLTXRunConfig) -> bool:
        if not config.metal_capture:
            return False
        metal = getattr(mx, "metal", None)
        start_capture = getattr(metal, "start_capture", None) if metal is not None else None
        if not callable(start_capture):
            write_event(
                "mlx_ltx_metal_capture",
                action="unavailable",
                model=model_metadata,
                reason="mlx.core.metal.start_capture is not available.",
            )
            return False
        try:
            capture_dir = Path(config.output_path).expanduser().resolve().parent / "metal_captures"
            capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = capture_dir / f"{Path(config.output_path).stem}.gputrace"
            try:
                result = start_capture(str(capture_path))
            except TypeError:
                result = start_capture()
            write_event("mlx_ltx_metal_capture", action="start", model=model_metadata, capture=result or str(capture_path))
            return True
        except Exception as e:
            write_event("mlx_ltx_metal_capture", action="error", model=model_metadata, error=redact_secrets(e))
            return False

    def _stop_metal_capture(self, mx: Any, model_metadata: dict[str, Any]) -> None:
        metal = getattr(mx, "metal", None)
        stop_capture = getattr(metal, "stop_capture", None) if metal is not None else None
        if not callable(stop_capture):
            return
        try:
            result = stop_capture()
            write_event("mlx_ltx_metal_capture", action="stop", model=model_metadata, capture=result)
        except Exception as e:
            write_event("mlx_ltx_metal_capture", action="stop_error", model=model_metadata, error=redact_secrets(e))

    def _resolved_path_cache_key(self, spec: MLXLTXModelSpec, cache_dir: str | None) -> tuple[Any, ...]:
        return (
            spec.repo_id,
            spec.revision or "",
            cache_dir or _default_cache_dir(),
            spec.local_files_only,
            spec.allow_patterns,
            spec.ignore_patterns,
            spec.use_hf_token,
        )

    def _prompt_cache_key(self, model_path: str, spec: MLXLTXModelSpec, config: MLXLTXRunConfig) -> tuple[Any, ...]:
        return (
            hashlib.sha256(config.prompt.encode("utf-8")).hexdigest(),
            "",
            os.path.abspath(model_path),
            spec.repo_id,
            spec.revision or "",
            spec.variant,
            config.pipeline,
            config.gemma_model_id,
            self._package_version("ltx-pipelines-mlx"),
            self._package_version("ltx-core-mlx"),
            self._package_version("mlx"),
        )

    def _prompt_cache_get(self, key: tuple[Any, ...], limit: int) -> tuple[Any, Any] | None:
        if limit <= 0:
            return None
        value = self._prompt_cache.get(key)
        if value is not None:
            self._prompt_cache.move_to_end(key)
        return value

    def _prompt_cache_put(self, key: tuple[Any, ...], value: tuple[Any, Any], limit: int) -> None:
        if limit <= 0:
            return
        self._prompt_cache[key] = value
        self._prompt_cache.move_to_end(key)
        while len(self._prompt_cache) > limit:
            self._prompt_cache.popitem(last=False)

    def _apply_mlx_cache_limit(self, mx: Any, cache_limit_gb: float | None) -> None:
        if cache_limit_gb is None or not hasattr(mx, "set_cache_limit"):
            return
        try:
            mx.set_cache_limit(int(float(cache_limit_gb) * (1024**3)))
        except Exception as e:
            write_event("mlx_ltx_cache_limit_error", error=redact_secrets(e))

    def _materialize_mlx_arrays(self, value: Any) -> None:
        try:
            mx = self._import_mlx_core()
            arrays = value if isinstance(value, (tuple, list)) else (value,)
            if hasattr(mx, "eval"):
                mx.eval(*arrays)
        except Exception:
            pass

    def _materialize_denoise_result(self, value: Any) -> None:
        arrays = []
        for name in ("video_latent", "audio_latent"):
            if hasattr(value, name):
                arrays.append(getattr(value, name))
        if arrays:
            self._materialize_mlx_arrays(tuple(arrays))

    @staticmethod
    def _package_version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not_installed"


class _TimedCallable:
    _mlx_ltx_timed_wrapper = True

    def __init__(self, wrapped: Any, backend: MLXLTXBackend, stage: str, model_metadata: dict[str, Any]) -> None:
        self._wrapped = wrapped
        self._backend = backend
        self._stage = stage
        self._model_metadata = model_metadata

    def __call__(self, *args, **kwargs):
        start = now()
        result = self._wrapped(*args, **kwargs)
        self._backend._materialize_mlx_arrays(result)
        self._backend._write_stage_end(self._stage, start, self._model_metadata)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


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
    prompt_cache_size: int = 8,
    stage_snapshot_cache: bool = False,
    mlx_cache_limit_gb: float | None = None,
    profile_level: str = "stage",
    metal_capture: bool = False,
    block_profile_interval: int = 8,
    eval_every: int | None = None,
    stage1_eval_every: int | None = None,
    stage2_eval_every: int | None = None,
    decode_profile: bool = False,
    compile_dit: bool = False,
    stage1_compile_dit: bool | None = None,
    stage2_compile_dit: bool | None = None,
    compile_block_stack: bool = False,
    compile_x0: bool = False,
    compile_shapeless: bool = False,
    fuse_attention_projections: bool = False,
    flatten_attention_projections: bool = False,
    dequantize_video_ffn: bool = False,
    native_metal_kernels: bool = False,
    native_kernel_profile: bool = False,
    native_kernel_verify: bool = True,
    native_kernel_fallback: bool = True,
    native_kernel_set: str = "off",
    ffmpeg_video_encoder: str = "libx264",
    ffmpeg_video_bitrate: str = "12M",
    cache_text_kv: bool = False,
    text_kv_cache_size: int = 2048,
    audio_debug: bool = False,
    audio_debug_dir: str | None = None,
    audio_debug_dump_arrays: bool = True,
    denoise_sync_policy: str = "each_step",
    denoise_sync_interval: int = 2,
    execution_mode: str = "denoiser_island",
    dev_checkpoint_path: str | None = None,
    distilled_lora_path: str | None = None,
    distilled_lora_strength: float | None = None,
) -> MLXLTXModelSpec:
    metadata = {
        "variant": variant,
        "pipeline": pipeline,
        "execution_mode": _coerce_execution_mode(execution_mode),
        "media_passthrough": bool(media_passthrough),
        "prompt_cache_size": int(prompt_cache_size),
        "stage_snapshot_cache": bool(stage_snapshot_cache),
        "mlx_cache_limit_gb": mlx_cache_limit_gb,
        "profile_level": _coerce_profile_level(profile_level),
        "metal_capture": bool(metal_capture),
        "block_profile_interval": int(block_profile_interval),
        "eval_every": eval_every,
        "stage1_eval_every": stage1_eval_every,
        "stage2_eval_every": stage2_eval_every,
        "decode_profile": bool(decode_profile),
        "compile_dit": bool(compile_dit),
        "stage1_compile_dit": stage1_compile_dit,
        "stage2_compile_dit": stage2_compile_dit,
        "compile_block_stack": bool(compile_block_stack),
        "compile_x0": bool(compile_x0),
        "compile_shapeless": bool(compile_shapeless),
        "fuse_attention_projections": bool(fuse_attention_projections),
        "flatten_attention_projections": bool(flatten_attention_projections),
        "dequantize_video_ffn": bool(dequantize_video_ffn),
        "native_metal_kernels": bool(native_metal_kernels),
        "native_kernel_profile": bool(native_kernel_profile),
        "native_kernel_verify": bool(native_kernel_verify),
        "native_kernel_fallback": bool(native_kernel_fallback),
        "native_kernel_set": coerce_native_kernel_set(native_kernel_set),
        "ffmpeg_video_encoder": str(ffmpeg_video_encoder),
        "ffmpeg_video_bitrate": str(ffmpeg_video_bitrate),
        "cache_text_kv": bool(cache_text_kv),
        "text_kv_cache_size": int(text_kv_cache_size),
        "audio_debug": bool(audio_debug),
        "audio_debug_dir": str(audio_debug_dir or ""),
        "audio_debug_dump_arrays": bool(audio_debug_dump_arrays),
        "denoise_sync_policy": _coerce_denoise_sync_policy(denoise_sync_policy),
        "denoise_sync_interval": int(denoise_sync_interval),
    }
    if dev_checkpoint_path:
        metadata["dev_checkpoint_path"] = str(dev_checkpoint_path)
    if distilled_lora_path:
        metadata["distilled_lora_path"] = str(distilled_lora_path)
    if distilled_lora_strength is not None:
        metadata["distilled_lora_strength"] = float(distilled_lora_strength)
    if low_memory is not None:
        metadata["low_memory"] = bool(low_memory)
    if low_ram_streaming is not None:
        metadata["low_ram_streaming"] = bool(low_ram_streaming)
    return MLXLTXModelSpec(
        repo_id=repo_id.strip(),
        revision=revision.strip() or None,
        local_path=local_path.strip() or None,
        modality="video",
        allow_patterns=_split_patterns(allow_patterns),
        ignore_patterns=_split_patterns(ignore_patterns),
        local_files_only=local_files_only,
        use_hf_token=use_hf_token,
        variant=variant,
        pipeline=pipeline,
        gemma_model_id=gemma_model_id,
        metadata=metadata,
    )


def load_mlx_ltx_manifest(path: str | os.PathLike[str]) -> tuple[MLXLTXModelSpec, MLXLTXRunConfig]:
    manifest_path = Path(path).expanduser()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_id = str(payload.get("repo_id") or payload.get("model_id") or "").strip()
    local_path = str(payload.get("local_path") or "").strip()
    if not repo_id and not local_path:
        raise ValueError(f"MLX LTX manifest {manifest_path} needs `repo_id` or `local_path`.")
    variant = str(payload.get("variant") or payload.get("quantization") or "q4")
    pipeline = str(payload.get("pipeline") or "distilled")
    execution_mode = _coerce_execution_mode(payload.get("execution_mode", payload.get("backend_mode", "denoiser_island")))
    gemma_model_id = str(payload.get("gemma_model_id") or "mlx-community/gemma-3-12b-it-4bit")
    low_memory = _coerce_optional_bool(payload.get("low_memory", None))
    low_ram_streaming = _coerce_optional_bool(payload.get("low_ram_streaming", payload.get("low_ram", None)))
    media_passthrough = bool(payload.get("media_passthrough", True))
    prompt_cache_size = int(payload.get("prompt_cache_size", 8))
    stage_snapshot_cache = bool(payload.get("stage_snapshot_cache", False))
    mlx_cache_limit_gb = payload.get("mlx_cache_limit_gb", None)
    if mlx_cache_limit_gb is not None:
        mlx_cache_limit_gb = float(mlx_cache_limit_gb)
    profile_level = _coerce_profile_level(payload.get("profile_level", "stage"))
    metal_capture = bool(_coerce_optional_bool(payload.get("metal_capture", False)))
    block_profile_interval = int(payload.get("block_profile_interval", 8))
    eval_every = _coerce_optional_int(payload.get("eval_every", None))
    stage1_eval_every = _coerce_optional_int(payload.get("stage1_eval_every", None))
    stage2_eval_every = _coerce_optional_int(payload.get("stage2_eval_every", None))
    decode_profile = bool(_coerce_optional_bool(payload.get("decode_profile", False)))
    compile_dit = bool(_coerce_optional_bool(payload.get("compile_dit", False)))
    stage1_compile_dit = _coerce_optional_bool(payload.get("stage1_compile_dit", None))
    stage2_compile_dit = _coerce_optional_bool(payload.get("stage2_compile_dit", None))
    compile_block_stack = bool(_coerce_optional_bool(payload.get("compile_block_stack", False)))
    compile_x0 = bool(_coerce_optional_bool(payload.get("compile_x0", False)))
    compile_shapeless = bool(_coerce_optional_bool(payload.get("compile_shapeless", False)))
    fuse_attention_projections = bool(_coerce_optional_bool(payload.get("fuse_attention_projections", False)))
    flatten_attention_projections = bool(_coerce_optional_bool(payload.get("flatten_attention_projections", False)))
    dequantize_video_ffn = bool(_coerce_optional_bool(payload.get("dequantize_video_ffn", False)))
    native_metal_kernels = bool(_coerce_optional_bool(payload.get("native_metal_kernels", False)))
    native_kernel_profile = bool(_coerce_optional_bool(payload.get("native_kernel_profile", False)))
    native_kernel_verify = bool(_coerce_optional_bool(payload.get("native_kernel_verify", True)))
    native_kernel_fallback = bool(_coerce_optional_bool(payload.get("native_kernel_fallback", True)))
    native_kernel_set = coerce_native_kernel_set(payload.get("native_kernel_set", "off"))
    ffmpeg_video_encoder = str(payload.get("ffmpeg_video_encoder", "libx264"))
    ffmpeg_video_bitrate = str(payload.get("ffmpeg_video_bitrate", "12M"))
    cache_text_kv = bool(_coerce_optional_bool(payload.get("cache_text_kv", False)))
    text_kv_cache_size = int(payload.get("text_kv_cache_size", 2048))
    audio_debug = bool(_coerce_optional_bool(payload.get("audio_debug", False)))
    audio_debug_dir = str(payload.get("audio_debug_dir") or "").strip() or None
    audio_debug_dump_arrays = bool(_coerce_optional_bool(payload.get("audio_debug_dump_arrays", True)))
    denoise_sync_policy = _coerce_denoise_sync_policy(payload.get("denoise_sync_policy", "each_step"))
    denoise_sync_interval = int(payload.get("denoise_sync_interval", 2))
    dev_checkpoint_path = _manifest_asset_path(
        manifest_path,
        payload.get("dev_checkpoint_path") or payload.get("dev_checkpoint") or payload.get("dev_transformer"),
        "checkpoints",
    )
    distilled_lora_path = _manifest_asset_path(
        manifest_path,
        payload.get("distilled_lora_path") or payload.get("distilled_lora") or payload.get("lora_path"),
        "loras",
    )
    distilled_lora_strength = (
        float(payload.get("distilled_lora_strength"))
        if payload.get("distilled_lora_strength") is not None
        else (float(payload.get("lora_strength")) if payload.get("lora_strength") is not None else None)
    )
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
        stage_snapshot_cache=stage_snapshot_cache,
        mlx_cache_limit_gb=mlx_cache_limit_gb,
        profile_level=profile_level,
        metal_capture=metal_capture,
        block_profile_interval=block_profile_interval,
        eval_every=eval_every,
        stage1_eval_every=stage1_eval_every,
        stage2_eval_every=stage2_eval_every,
        decode_profile=decode_profile,
        compile_dit=compile_dit,
        stage1_compile_dit=stage1_compile_dit,
        stage2_compile_dit=stage2_compile_dit,
        compile_block_stack=compile_block_stack,
        compile_x0=compile_x0,
        compile_shapeless=compile_shapeless,
        fuse_attention_projections=fuse_attention_projections,
        flatten_attention_projections=flatten_attention_projections,
        dequantize_video_ffn=dequantize_video_ffn,
        native_metal_kernels=native_metal_kernels,
        native_kernel_profile=native_kernel_profile,
        native_kernel_verify=native_kernel_verify,
        native_kernel_fallback=native_kernel_fallback,
        native_kernel_set=native_kernel_set,
        ffmpeg_video_encoder=ffmpeg_video_encoder,
        ffmpeg_video_bitrate=ffmpeg_video_bitrate,
        cache_text_kv=cache_text_kv,
        text_kv_cache_size=text_kv_cache_size,
        audio_debug=audio_debug,
        audio_debug_dir=audio_debug_dir,
        audio_debug_dump_arrays=audio_debug_dump_arrays,
        denoise_sync_policy=denoise_sync_policy,
        denoise_sync_interval=denoise_sync_interval,
        execution_mode=execution_mode,
        dev_checkpoint_path=dev_checkpoint_path,
        distilled_lora_path=distilled_lora_path,
        distilled_lora_strength=distilled_lora_strength,
    )
    run_config = MLXLTXRunConfig(
        prompt="",
        output_path="",
        width=int(payload.get("width", 512)),
        height=int(payload.get("height", 512)),
        num_frames=int(payload.get("num_frames", 49)),
        frame_rate=float(payload.get("frame_rate", 24.0)),
        pipeline=pipeline,
        stage1_steps=int(payload.get("stage1_steps", 8)),
        stage2_steps=int(payload.get("stage2_steps", 3)),
        seed=int(payload.get("seed", 42)),
        low_memory=low_memory,
        low_ram=bool(low_ram_streaming) if low_ram_streaming is not None else False,
        tile_frames=int(payload.get("tile_frames", 1)),
        tile_spatial=int(payload.get("tile_spatial", 1)),
        tile_overlap=int(payload.get("tile_overlap", 2)),
        gemma_model_id=gemma_model_id,
        media_passthrough=media_passthrough,
        prompt_cache_size=prompt_cache_size,
        stage_snapshot_cache=stage_snapshot_cache,
        mlx_cache_limit_gb=mlx_cache_limit_gb,
        profile_level=profile_level,
        metal_capture=metal_capture,
        block_profile_interval=block_profile_interval,
        eval_every=eval_every,
        stage1_eval_every=stage1_eval_every,
        stage2_eval_every=stage2_eval_every,
        decode_profile=decode_profile,
        compile_dit=compile_dit,
        stage1_compile_dit=stage1_compile_dit,
        stage2_compile_dit=stage2_compile_dit,
        compile_block_stack=compile_block_stack,
        compile_x0=compile_x0,
        compile_shapeless=compile_shapeless,
        fuse_attention_projections=fuse_attention_projections,
        flatten_attention_projections=flatten_attention_projections,
        dequantize_video_ffn=dequantize_video_ffn,
        native_metal_kernels=native_metal_kernels,
        native_kernel_profile=native_kernel_profile,
        native_kernel_verify=native_kernel_verify,
        native_kernel_fallback=native_kernel_fallback,
        native_kernel_set=native_kernel_set,
        ffmpeg_video_encoder=ffmpeg_video_encoder,
        ffmpeg_video_bitrate=ffmpeg_video_bitrate,
        cache_text_kv=cache_text_kv,
        text_kv_cache_size=text_kv_cache_size,
        audio_debug=audio_debug,
        audio_debug_dir=audio_debug_dir,
        audio_debug_dump_arrays=audio_debug_dump_arrays,
        denoise_sync_policy=denoise_sync_policy,
        denoise_sync_interval=denoise_sync_interval,
    )
    return spec, run_config.validated()


def load_mlx_ltx_reference(path: str | os.PathLike[str]) -> tuple[MLXLTXModelSpec, MLXLTXRunConfig]:
    reference_path = Path(path).expanduser()
    if is_mlx_ltx_manifest_path(reference_path):
        return load_mlx_ltx_manifest(reference_path)
    if not is_mlx_ltx_folder_path(reference_path):
        raise ValueError(f"MLX LTX reference must be a .mlx_ltx.json manifest or full model folder: {reference_path}")

    variant = _infer_variant_from_path(reference_path)
    spec = make_model_spec(
        repo_id=reference_path.name,
        local_path=str(reference_path),
        variant=variant,
        pipeline="distilled",
        local_files_only=True,
        execution_mode="denoiser_island",
    )
    run_config = MLXLTXRunConfig(
        prompt="",
        output_path="",
        pipeline="distilled",
        low_memory=True,
        low_ram=False,
        media_passthrough=True,
    )
    return spec, run_config.validated()


def load_mlx_ltx_checkpoint(path: str | os.PathLike[str]):
    spec, run_config = load_mlx_ltx_reference(path)
    if spec.metadata.get("execution_mode") == "legacy_media":
        return MLXLTXModelProxy(spec, run_config), MLXLTXClipProxy(spec, run_config), MLXLTXVAEProxy(spec, run_config)

    from comfy.backends.mlx_denoiser_island import LTXAVIslandRuntime, create_mlx_denoiser_island_model

    model_path = _get_mlx_ltx_backend().resolve_model_path(spec)
    runtime = LTXAVIslandRuntime(
        model_path,
        pipeline=run_config.pipeline,
        dev_checkpoint_path=spec.metadata.get("dev_checkpoint_path"),
        distilled_lora_path=spec.metadata.get("distilled_lora_path"),
        distilled_lora_strength=float(spec.metadata.get("distilled_lora_strength", 1.0)),
        low_memory=True if run_config.low_memory is None else bool(run_config.low_memory),
        low_ram_streaming=bool(run_config.low_ram),
        default_frame_rate=float(run_config.frame_rate),
    )
    model = create_mlx_denoiser_island_model(runtime=runtime)
    model.model.mlx_ltx_pipeline = run_config.pipeline
    model.model.mlx_ltx_internal_lora_name = (
        Path(str(spec.metadata.get("distilled_lora_path"))).name if spec.metadata.get("distilled_lora_path") else None
    )
    model.model.mlx_ltx_internal_lora_strength = float(spec.metadata.get("distilled_lora_strength", 0.0) or 0.0)
    return model, MLXLTXClipProxy(spec, run_config), MLXLTXVAEProxy(spec, run_config)


def is_mlx_ltx_model(model: Any) -> bool:
    return bool(getattr(model, "is_mlx_ltx_proxy", False))


def is_mlx_ltx_vae(vae: Any) -> bool:
    return bool(getattr(vae, "is_mlx_ltx_vae_proxy", False))


def is_mlx_ltx_latent_upscaler(upscale_model: Any) -> bool:
    return bool(getattr(upscale_model, "is_mlx_ltx_latent_upscaler_proxy", False))


def load_mlx_ltx_latent_upscaler(path: str | os.PathLike[str]):
    spec, run_config = load_mlx_ltx_reference(path)
    return MLXLTXLatentUpscalerProxy(spec, run_config)


def is_mlx_ltx_media_latent(latent: Any) -> bool:
    return isinstance(latent, dict) and bool(latent.get("mlx_ltx_media_path"))


class MLXLTXImageProxy:
    is_mlx_ltx_image_proxy = True

    def __init__(self, latent: dict[str, Any]):
        self.latent = latent
        self.media_path = str(latent["mlx_ltx_media_path"])
        self.frame_rate = float(latent.get("frame_rate", 24.0))

    def materialize(self):
        return mlx_ltx_media_components(self.latent).images

    def materialize_components(self):
        return mlx_ltx_media_components(self.latent)

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
    backend = _get_mlx_ltx_backend()
    return backend.output_as_comfy_video(image_proxy.media_path)


def materialize_mlx_ltx_image_proxy(image_proxy: MLXLTXImageProxy):
    return image_proxy.materialize()


def materialize_mlx_ltx_audio_proxy(audio_proxy: MLXLTXAudioProxy):
    return audio_proxy.materialize()


def is_mlx_ltx_audio_placeholder(latent: Any) -> bool:
    return isinstance(latent, dict) and latent.get("type") == "mlx_ltx_audio_placeholder"


def mlx_ltx_media_components(latent: dict[str, Any]):
    backend = _get_mlx_ltx_backend()
    return backend.output_components(str(latent["mlx_ltx_media_path"]))


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


def run_mlx_ltx_latent_upscale(upscale_model: MLXLTXLatentUpscalerProxy, samples: dict[str, Any]) -> dict[str, Any]:
    if not is_mlx_ltx_latent_upscaler(upscale_model):
        raise TypeError("run_mlx_ltx_latent_upscale requires an MLXLTXLatentUpscalerProxy.")

    import numpy as np
    import torch
    import mlx.core as mx
    from ltx_pipelines_mlx import DistilledPipeline

    video = samples["samples"]
    nested_audio = None
    if hasattr(video, "is_nested") and video.is_nested:
        video, nested_audio = video.unbind()
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("MLX LTX latent upscaler expects a video latent tensor shaped [B,C,F,H,W].")

    backend = _get_mlx_ltx_backend()
    model_path = backend.resolve_model_path(upscale_model.spec)
    pipe = DistilledPipeline(
        model_path,
        low_memory=True if upscale_model.run_config.low_memory is None else bool(upscale_model.run_config.low_memory),
        low_ram_streaming=bool(upscale_model.run_config.low_ram),
    )
    try:
        video_cpu = video.detach().to("cpu", dtype=torch.float32)
        video_mx = mx.array(video_cpu.numpy()).astype(mx.bfloat16)
        upscaled = pipe.upscale_distilled_video_latent(video_mx)
        mx.eval(upscaled)
        upscaled_torch = torch.from_numpy(np.asarray(upscaled.astype(mx.float32))).to(dtype=video.dtype, device=video.device)
    finally:
        del pipe
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    output = samples.copy()
    if nested_audio is not None:
        import comfy.nested_tensor

        output["samples"] = comfy.nested_tensor.NestedTensor((upscaled_torch, nested_audio))
    else:
        output["samples"] = upscaled_torch

    noise_mask = output.get("noise_mask")
    if isinstance(noise_mask, torch.Tensor) and noise_mask.ndim == 5:
        if noise_mask.shape[-2:] == video.shape[-2:] and noise_mask.shape[-2:] != upscaled_torch.shape[-2:]:
            b, c, f, h, w = noise_mask.shape
            mask_2d = noise_mask.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            mask_2d = torch.nn.functional.interpolate(mask_2d, size=upscaled_torch.shape[-2:], mode="nearest")
            noise_mask = mask_2d.reshape(b, f, c, upscaled_torch.shape[-2], upscaled_torch.shape[-1]).permute(0, 2, 1, 3, 4)
        output["noise_mask"] = noise_mask.to(device=upscaled_torch.device)
    return output


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
    output_path = _native_temp_output_path(model.spec)
    backend = _get_mlx_ltx_backend()
    result = backend.generate_video_to_file(
        model.spec,
        prompt,
        output_path,
        width=width,
        height=height,
        num_frames=frames,
        frame_rate=frame_rate,
        pipeline=model.run_config.pipeline,
        stage1_steps=model.run_config.stage1_steps,
        stage2_steps=model.run_config.stage2_steps,
        seed=int(seed if seed is not None else model.run_config.seed),
        low_memory=model.run_config.low_memory,
        low_ram=model.run_config.low_ram,
        tile_frames=model.run_config.tile_frames,
        tile_spatial=model.run_config.tile_spatial,
        tile_overlap=model.run_config.tile_overlap,
        gemma_model_id=model.run_config.gemma_model_id,
        media_passthrough=model.run_config.media_passthrough,
        prompt_cache_size=model.run_config.prompt_cache_size,
        stage_snapshot_cache=model.run_config.stage_snapshot_cache,
        mlx_cache_limit_gb=model.run_config.mlx_cache_limit_gb,
        profile_level=model.run_config.profile_level,
        metal_capture=model.run_config.metal_capture,
        block_profile_interval=model.run_config.block_profile_interval,
        eval_every=model.run_config.eval_every,
        stage1_eval_every=model.run_config.stage1_eval_every,
        stage2_eval_every=model.run_config.stage2_eval_every,
        decode_profile=model.run_config.decode_profile,
        compile_dit=model.run_config.compile_dit,
        stage1_compile_dit=model.run_config.stage1_compile_dit,
        stage2_compile_dit=model.run_config.stage2_compile_dit,
        compile_block_stack=model.run_config.compile_block_stack,
        compile_x0=model.run_config.compile_x0,
        compile_shapeless=model.run_config.compile_shapeless,
        fuse_attention_projections=model.run_config.fuse_attention_projections,
        flatten_attention_projections=model.run_config.flatten_attention_projections,
        dequantize_video_ffn=model.run_config.dequantize_video_ffn,
        native_metal_kernels=model.run_config.native_metal_kernels,
        native_kernel_profile=model.run_config.native_kernel_profile,
        native_kernel_verify=model.run_config.native_kernel_verify,
        native_kernel_fallback=model.run_config.native_kernel_fallback,
        native_kernel_set=model.run_config.native_kernel_set,
        ffmpeg_video_encoder=model.run_config.ffmpeg_video_encoder,
        ffmpeg_video_bitrate=model.run_config.ffmpeg_video_bitrate,
        cache_text_kv=model.run_config.cache_text_kv,
        text_kv_cache_size=model.run_config.text_kv_cache_size,
        audio_debug=model.run_config.audio_debug,
        audio_debug_dir=model.run_config.audio_debug_dir,
        audio_debug_dump_arrays=model.run_config.audio_debug_dump_arrays,
        denoise_sync_policy=model.run_config.denoise_sync_policy,
        denoise_sync_interval=model.run_config.denoise_sync_interval,
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
    stamp = f"{time.time_ns()}"
    directory = Path(folder_paths.get_temp_directory()) / "mlx_ltx_native"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"{safe_repo}_{stamp}.mp4")


def _get_mlx_ltx_backend() -> MLXLTXBackend:
    from comfy.backends import get_backend

    backend = get_backend("mlx_ltx")
    if not isinstance(backend, MLXLTXBackend):
        raise TypeError("Registered mlx_ltx backend has an unexpected type.")
    return backend
