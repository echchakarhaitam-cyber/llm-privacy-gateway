"""Detector names used by the benchmark and how to build them (heavy imports are lazy)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from privgate.detectors.base import Detector

BASE_DETECTORS = ("regex", "spacy", "presidio")
ENSEMBLES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "regex+spacy": ("regex", "spacy"),
    "regex+presidio": ("regex", "presidio"),
})
ALL_DETECTORS = BASE_DETECTORS + tuple(ENSEMBLES)


def members(name: str) -> tuple[str, ...]:
    """Base detectors that make up ``name`` (itself for a base detector)."""
    if name in ENSEMBLES:
        return ENSEMBLES[name]
    if name in BASE_DETECTORS:
        return (name,)
    raise ValueError(f"unknown detector {name!r}; expected one of {ALL_DETECTORS}")


def make_detector(name: str, presidio_threshold: float = 0.0) -> Detector:
    """A configured detector; ensembles are built from freshly made members."""
    if name == "regex":
        from privgate.detectors.regex import RegexDetector

        return RegexDetector()
    if name == "spacy":
        from privgate.detectors.spacy_ner import SpacyDetector

        return SpacyDetector()
    if name == "presidio":
        from privgate.detectors.presidio import PresidioDetector

        return PresidioDetector(score_threshold=presidio_threshold)
    if name in ENSEMBLES:
        from privgate.detectors.ensemble import EnsembleDetector

        return EnsembleDetector([make_detector(m, presidio_threshold) for m in ENSEMBLES[name]],
                                name=name)
    raise ValueError(f"unknown detector {name!r}; expected one of {ALL_DETECTORS}")
