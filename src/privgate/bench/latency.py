"""Per-document latency statistics and a description of the machine they were measured on.

Latency is wall time around ``detect()`` only (no I/O, no evaluation), measured with
``time.perf_counter_ns``. The first :data:`WARMUP_DOCS` documents of a run are excluded:
they include lazy initialisation (model weights paged in, regex caches) that a long-running
gateway pays once. Numbers depend on this laptop and on what else it was doing; they are
comparable between detectors of one run, not across machines.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Sequence

from privgate.bench.bootstrap import percentile

WARMUP_DOCS = 20
PACKAGES = ("privgate", "spacy", "presidio-analyzer", "en-core-web-lg", "pyarrow")


def latency_stats(
    timings_ns: Sequence[int], chars: Sequence[int], warmup: int = WARMUP_DOCS
) -> dict:
    """Median, p95 and mean milliseconds per document, and characters per second."""
    if len(timings_ns) != len(chars):
        raise ValueError("one timing per document is required")
    times, sizes = list(timings_ns[warmup:]), list(chars[warmup:])
    if not times:
        raise ValueError(f"need more than {warmup} documents to measure latency")
    ms = sorted(t / 1e6 for t in times)
    total_s = sum(times) / 1e9
    return {
        "documents_timed": len(times),
        "warmup_documents_excluded": min(warmup, len(timings_ns)),
        "median_ms": percentile(ms, 0.5),
        "p95_ms": percentile(ms, 0.95),
        "mean_ms": sum(ms) / len(ms),
        "chars_per_second": sum(sizes) / total_s if total_s else 0.0,
    }


def cpu_model() -> str:
    """Human-readable CPU name (Windows registry, /proc/cpuinfo, or platform fallback)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
        "privgate_commit": _git_commit(),
    }
