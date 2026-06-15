from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    available: bool
    reason: str = ""
    modalities: tuple[str, ...] = ()
    device: str = "unknown"
    unified_memory: bool = False


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None = None
    local_path: str | None = None
    modality: str = "llm"
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    local_files_only: bool = False
    use_hf_token: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class RuntimeHandle:
    backend: str
    spec: ModelSpec
    model: Any
    processor: Any = None
    resolved_path: str | None = None
    modality: str = "llm"


@dataclass(frozen=True)
class GenerationChunk:
    text: str
    token_count: int | None = None
    is_final: bool = False


@dataclass(frozen=True)
class MemoryStats:
    backend: str
    stats: dict[str, Any]


class RuntimeBackend(Protocol):
    name: str

    def capabilities(self) -> BackendCapabilities:
        ...

    def unload_all(self) -> None:
        ...

    def empty_cache(self) -> None:
        ...
