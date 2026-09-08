"""Small in-process metrics collector with no external backend."""

from __future__ import annotations

from contextlib import contextmanager
from collections import deque
import math
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class OperationMetric:
    name: str
    duration_ms: float
    attributes: dict[str, Any]


class TelemetryCollector:
    """Thread-safe collector that can later feed an adaptive policy."""

    def __init__(self, *, max_events: int | None = 1024) -> None:
        if max_events is not None and (type(max_events) is not int or max_events < 1):
            raise ValueError('max_events must be positive or None')
        self._metrics = deque(maxlen=max_events)
        self.dropped_events = 0
        self._lock = Lock()

    def record(self, name: str, duration_ms: float, **attributes: Any) -> None:
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError('duration must be finite and nonnegative')
        metric = OperationMetric(name, duration_ms, dict(attributes))
        with self._lock:
            if self._metrics.maxlen is not None and len(self._metrics) == self._metrics.maxlen:
                self.dropped_events += 1
            self._metrics.append(metric)

    @contextmanager
    def measure(self, name: str, **attributes: Any) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.record(name, (perf_counter() - started) * 1000, **attributes)

    def snapshot(self) -> tuple[OperationMetric, ...]:
        with self._lock:
            return tuple(self._metrics)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(metric) for metric in self.snapshot()]

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()
