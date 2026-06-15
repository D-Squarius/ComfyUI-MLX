from __future__ import annotations

import importlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_stats import redact_secrets, write_event


NATIVE_KERNEL_SETS = {
    "off",
    "rope",
    "norm",
    "attention_prelude",
    "ffn_elementwise",
    "decode",
    "all_safe",
}

UNSAFE_NATIVE_KERNEL_SETS = {
    "rope": "disabled after local GPU-recovery crashes in LTX q4 full-workload runs",
    "attention_prelude": "disabled because the current implementation shares the unsafe RoPE kernel path",
}


def coerce_native_kernel_set(value: Any) -> str:
    kernel_set = str(value or "off").strip().lower()
    if kernel_set not in NATIVE_KERNEL_SETS:
        raise ValueError(
            f"MLX LTX native_kernel_set must be one of {sorted(NATIVE_KERNEL_SETS)}; got {kernel_set!r}."
        )
    return kernel_set


def native_kernel_names_for_set(kernel_set: str) -> tuple[str, ...]:
    kernel_set = coerce_native_kernel_set(kernel_set)
    if kernel_set == "off":
        return ()
    if kernel_set == "all_safe":
        return ("ffn_elementwise",)
    return (kernel_set,)


@dataclass
class NativeKernelAvailability:
    available: bool
    reason: str = ""
    mlx_version: str = ""


@dataclass
class NativeKernelStats:
    availability_checked: bool = False
    available: bool = False
    reason: str = ""
    installed_targets: int = 0
    used_count: int = 0
    fallback_count: int = 0
    verify_count: int = 0
    verify_seconds: float = 0.0
    kernel_names: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, Any]:
        return {
            "native_kernel_available": self.available,
            "native_kernel_reason": self.reason,
            "native_kernel_installed_targets": self.installed_targets,
            "native_kernel_used_count": self.used_count,
            "native_kernel_fallback_count": self.fallback_count,
            "native_kernel_verify_count": self.verify_count,
            "native_kernel_verify_seconds": round(self.verify_seconds, 6),
            "native_kernel_names": list(self.kernel_names),
        }


@dataclass
class NativeKernelRuntime:
    enabled: bool = False
    kernel_set: str = "off"
    profile: bool = False
    verify: bool = True
    fallback: bool = True
    stats: NativeKernelStats = field(default_factory=NativeKernelStats)

    def __post_init__(self) -> None:
        self.kernel_set = coerce_native_kernel_set(self.kernel_set)
        self.enabled = bool(self.enabled) and self.kernel_set != "off"
        self.stats.kernel_names = native_kernel_names_for_set(self.kernel_set) if self.enabled else ()

    def availability(self) -> NativeKernelAvailability:
        if not self.enabled:
            self.stats.availability_checked = True
            self.stats.available = False
            self.stats.reason = "disabled"
            return NativeKernelAvailability(False, "disabled")
        unsafe_reason = UNSAFE_NATIVE_KERNEL_SETS.get(self.kernel_set)
        if unsafe_reason is not None:
            self.stats.availability_checked = True
            self.stats.available = False
            self.stats.reason = unsafe_reason
            return NativeKernelAvailability(False, unsafe_reason)
        try:
            mx = importlib.import_module("mlx.core")
            fast = getattr(mx, "fast", None)
            if fast is None or not callable(getattr(fast, "metal_kernel", None)):
                raise RuntimeError("mlx.core.fast.metal_kernel is unavailable")
            version = _package_version("mlx")
            self.stats.availability_checked = True
            self.stats.available = True
            self.stats.reason = "available"
            return NativeKernelAvailability(True, "available", version)
        except Exception as e:
            reason = redact_secrets(e)
            self.stats.availability_checked = True
            self.stats.available = False
            self.stats.reason = str(reason)
            return NativeKernelAvailability(False, str(reason))

    def install_on(self, target: Any) -> None:
        if target is None:
            return
        try:
            setattr(target, "_mlx_ltx_native_kernel_runtime", self)
            setattr(target, "_mlx_ltx_native_metal_kernels", bool(self.enabled))
            setattr(target, "_mlx_ltx_native_kernel_set", self.kernel_set)
            setattr(target, "_mlx_ltx_native_kernel_profile", bool(self.profile))
            setattr(target, "_mlx_ltx_native_kernel_verify", bool(self.verify))
            setattr(target, "_mlx_ltx_native_kernel_fallback", bool(self.fallback))
            setattr(target, "_mlx_ltx_native_kernel_names", self.stats.kernel_names)
            if self.enabled:
                self.stats.installed_targets += 1
        except Exception:
            if not self.fallback:
                raise
            self.stats.fallback_count += 1

    def record_status(self, model_metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        availability = self.availability()
        write_event(
            "mlx_ltx_native_kernel_status",
            model={
                **model_metadata,
                **self.stats.as_metadata(),
                "native_kernel_mlx_version": availability.mlx_version,
            },
            available=availability.available,
            reason=availability.reason,
            kernel_set=self.kernel_set,
            kernel_names=list(self.stats.kernel_names),
        )

    def verify_microbench(self) -> None:
        if not self.enabled or not self.verify:
            return
        availability = self.availability()
        if not availability.available:
            if not self.fallback:
                raise RuntimeError(availability.reason)
            return
        start = time.perf_counter()
        try:
            kernels = importlib.import_module("comfy.backends.mlx_ltx_native.kernels")

            kernels.verify_available_kernels(self.stats.kernel_names)
            self.stats.verify_count += 1
        except Exception:
            self.stats.fallback_count += 1
            if not self.fallback:
                raise
        finally:
            self.stats.verify_seconds += time.perf_counter() - start

    @contextmanager
    def patch_ltx_modules(self):
        patches: list[tuple[Any, str, Any]] = []
        if not self.enabled:
            yield
            return
        availability = self.availability()
        if not availability.available:
            if not self.fallback:
                raise RuntimeError(availability.reason)
            self.stats.fallback_count += 1
            yield
            return
        try:
            kernels = importlib.import_module("comfy.backends.mlx_ltx_native.kernels")

            if any(name in self.stats.kernel_names for name in ("rope", "attention_prelude")):
                rope_module = importlib.import_module("ltx_core_mlx.model.transformer.rope")
                attention_module = importlib.import_module("ltx_core_mlx.model.transformer.attention")
                rope_wrapper = self._wrap_kernel("rope_split", kernels.rope_split, rope_module.apply_rope_split)
                for module in (rope_module, attention_module):
                    patches.append((module, "apply_rope_split", getattr(module, "apply_rope_split")))
                    setattr(module, "apply_rope_split", rope_wrapper)

            if "norm" in self.stats.kernel_names:
                transformer_module = importlib.import_module("ltx_core_mlx.model.transformer.transformer")
                block_cls = getattr(transformer_module, "BasicAVTransformerBlock")
                original_scale_shift = getattr(block_cls, "_scale_shift")
                original_gated_residual = getattr(block_cls, "_gated_residual")

                def scale_shift_call(instance, x, scale, shift):
                    try:
                        result = kernels.scale_shift(x, scale, shift)
                        self.stats.used_count += 1
                        return result
                    except Exception:
                        self.stats.fallback_count += 1
                        if not self.fallback:
                            raise
                        return original_scale_shift(instance, x, scale, shift)

                def gated_residual_call(instance, residual, value, gate):
                    try:
                        result = kernels.gated_residual(residual, value, gate)
                        self.stats.used_count += 1
                        return result
                    except Exception:
                        self.stats.fallback_count += 1
                        if not self.fallback:
                            raise
                        return original_gated_residual(instance, residual, value, gate)

                patches.append((block_cls, "_scale_shift", original_scale_shift))
                patches.append((block_cls, "_gated_residual", original_gated_residual))
                setattr(block_cls, "_scale_shift", scale_shift_call)
                setattr(block_cls, "_gated_residual", gated_residual_call)

            if "ffn_elementwise" in self.stats.kernel_names:
                feed_forward_module = importlib.import_module("ltx_core_mlx.model.transformer.feed_forward")
                feed_forward_cls = getattr(feed_forward_module, "FeedForward")
                original_call = feed_forward_cls.__call__

                def ffn_call(instance, x):
                    try:
                        projected = instance.proj_in(x)
                        activated = kernels.gelu_approx(projected)
                        self.stats.used_count += 1
                        return instance.proj_out(activated)
                    except Exception:
                        self.stats.fallback_count += 1
                        if not self.fallback:
                            raise
                        return original_call(instance, x)

                patches.append((feed_forward_cls, "__call__", original_call))
                setattr(feed_forward_cls, "__call__", ffn_call)
            yield
        except Exception:
            self.stats.fallback_count += 1
            if not self.fallback:
                raise
            yield
        finally:
            for target, name, original in reversed(patches):
                try:
                    setattr(target, name, original)
                except Exception:
                    pass

    def _wrap_kernel(self, name: str, native_func: Any, fallback_func: Any):
        def wrapper(*args, **kwargs):
            try:
                result = native_func(*args, **kwargs)
                self.stats.used_count += 1
                return result
            except Exception:
                self.stats.fallback_count += 1
                if not self.fallback:
                    raise
                return fallback_func(*args, **kwargs)

        wrapper.__name__ = f"comfy_native_{name}"
        return wrapper

    def metadata(self) -> dict[str, Any]:
        return self.stats.as_metadata()


def make_native_kernel_runtime(
    *,
    native_metal_kernels: bool,
    native_kernel_set: str,
    native_kernel_profile: bool,
    native_kernel_verify: bool,
    native_kernel_fallback: bool,
) -> NativeKernelRuntime:
    return NativeKernelRuntime(
        enabled=bool(native_metal_kernels),
        kernel_set=native_kernel_set,
        profile=bool(native_kernel_profile),
        verify=bool(native_kernel_verify),
        fallback=bool(native_kernel_fallback),
    )


def metal_source_path(name: str) -> Path:
    return Path(__file__).with_name("mlx_ltx_native") / "metal" / f"{name}.metal"


def _package_version(package: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return "not_installed"
