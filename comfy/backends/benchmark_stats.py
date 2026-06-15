from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import psutil


HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_\-]{8,}")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|access_token|authorization|bearer|api_key|secret)\b(\s*[:=]\s*|\s+)([A-Za-z0-9._\-+/=]{8,})"
)

_LOCK = threading.RLock()


def redact_secrets(value: Any) -> str:
    text = str(value)
    text = HF_TOKEN_RE.sub("hf_***", text)
    return SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def stats_path() -> str:
    return os.environ.get("COMFY_BENCH_STATS_PATH", "").strip()


def benchmark_enabled() -> bool:
    return bool(stats_path())


def event_enabled(event_type: str) -> bool:
    raw = os.environ.get("COMFY_BENCH_DISABLED_EVENTS", "").strip()
    if not raw:
        return True
    disabled = {part.strip() for part in raw.split(",") if part.strip()}
    return "all" not in disabled and str(event_type) not in disabled


def now() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def process_memory() -> dict[str, int]:
    try:
        info = psutil.Process(os.getpid()).memory_info()
        return {"process_rss": int(info.rss)}
    except Exception:
        return {}


def torch_mps_memory() -> dict[str, int]:
    try:
        import torch

        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            return {}
        stats: dict[str, int] = {}
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["mps_current_allocated_memory"] = int(torch.mps.current_allocated_memory())
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["mps_driver_allocated_memory"] = int(torch.mps.driver_allocated_memory())
        return stats
    except Exception:
        return {}


def memory_snapshot() -> dict[str, int]:
    return {**process_memory(), **torch_mps_memory()}


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._latest: dict[str, int] = {}
        self._peaks: dict[str, int] = {}

    def start(self) -> None:
        if not benchmark_enabled():
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, name="comfy-bench-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()
        return self.snapshot()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._latest,
                **{f"{key}_peak": value for key, value in self._peaks.items()},
            }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        current = memory_snapshot()
        with self._lock:
            self._latest = current
            for key, value in current.items():
                if isinstance(value, int):
                    self._peaks[key] = max(value, self._peaks.get(key, 0))


def write_event(event_type: str, **payload: Any) -> None:
    path = stats_path()
    if not path or not event_enabled(event_type):
        return
    event = {
        "event_type": event_type,
        "time": time.time(),
        "pid": os.getpid(),
        "memory": process_memory(),
    }
    event.update(_jsonable(payload))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(event, sort_keys=True, default=str)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return redact_secrets(value)
