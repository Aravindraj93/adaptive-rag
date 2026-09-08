"""Portable, dependency-free host profiling for future retrieval planning."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any


def _memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return pages * page_size
    except (OSError, ValueError):
        pass

    if platform.system() == "Windows":
        try:
            import ctypes

            value = ctypes.c_ulonglong()
            if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(value)):
                return int(value.value) * 1024
        except (AttributeError, OSError):
            pass
    return None


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Stable hardware facts useful to an adaptive retrieval planner."""

    logical_cpus: int
    physical_memory_bytes: int | None
    architecture: str
    operating_system: str
    python_version: str
    processor: str

    @property
    def physical_memory_gib(self) -> float | None:
        if self.physical_memory_bytes is None:
            return None
        return self.physical_memory_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_hardware() -> HardwareProfile:
    """Inspect the current machine without running expensive probes."""

    return HardwareProfile(
        logical_cpus=os.cpu_count() or 1,
        physical_memory_bytes=_memory_bytes(),
        architecture=platform.machine() or "unknown",
        operating_system=platform.platform(),
        python_version=platform.python_version(),
        processor=platform.processor() or "unknown",
    )

