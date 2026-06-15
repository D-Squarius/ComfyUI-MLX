from __future__ import annotations

import logging
import threading

from .base import RuntimeBackend

_BACKENDS: dict[str, RuntimeBackend] = {}
_BUILTINS_LOADED = False
_LOCK = threading.RLock()


def register_backend(backend: RuntimeBackend) -> None:
    with _LOCK:
        _BACKENDS[backend.name] = backend


def ensure_builtin_backends() -> None:
    global _BUILTINS_LOADED
    with _LOCK:
        if _BUILTINS_LOADED:
            return
        from .gguf_backend import GGUFBackend
        from .mlx_backend import MLXBackend
        from .mlx_ltx_backend import MLXLTXBackend
        from .transformers_backend import TransformersBackend

        register_backend(GGUFBackend())
        register_backend(MLXBackend())
        register_backend(MLXLTXBackend())
        register_backend(TransformersBackend())
        _BUILTINS_LOADED = True


def get_backend(name: str) -> RuntimeBackend:
    ensure_builtin_backends()
    with _LOCK:
        return _BACKENDS[name]


def list_backends() -> list[RuntimeBackend]:
    ensure_builtin_backends()
    with _LOCK:
        return list(_BACKENDS.values())


def unload_backend_models() -> None:
    for backend in list_backends():
        try:
            backend.unload_all()
        except Exception as e:
            logging.debug("Backend %s unload failed: %s", backend.name, e)


def empty_backend_caches() -> None:
    for backend in list_backends():
        try:
            backend.empty_cache()
        except Exception as e:
            logging.debug("Backend %s cache cleanup failed: %s", backend.name, e)
