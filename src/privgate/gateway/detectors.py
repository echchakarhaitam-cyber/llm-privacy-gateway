"""Detector configurations the gateway can run.

Any name known to the benchmark registry (``regex``, ``crf``, ``regex+presidio``,
``crf+regex+presidio``, ...) plus ad-hoc unions of the runnable detectors written with ``+``
(e.g. ``crf+regex``), combined by the same union ensemble the benchmark uses.
"""

from __future__ import annotations

from privgate.bench.registry import ALL_DETECTORS, RUNNABLE, make_detector
from privgate.detectors.base import Detector


def build_detector(name: str) -> Detector:
    if name in ALL_DETECTORS:
        return make_detector(name)
    parts = name.split("+")
    if len(parts) > 1 and len(set(parts)) == len(parts) and all(p in RUNNABLE for p in parts):
        from privgate.detectors.ensemble import EnsembleDetector

        return EnsembleDetector([make_detector(p) for p in parts], name=name)
    raise ValueError(f"unknown detector {name!r}; known: {ALL_DETECTORS} or a '+' union of "
                     f"{RUNNABLE}")


def member_names(detector: Detector | None) -> tuple[str, ...]:
    """Names of the detectors behind ``detector`` (for the audit log and /healthz)."""
    if detector is None:
        return ()
    members = getattr(detector, "detectors", None)
    if members:
        return tuple(str(getattr(d, "name", type(d).__name__)) for d in members)
    return (str(getattr(detector, "name", type(detector).__name__)),)
