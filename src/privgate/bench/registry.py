"""Detector names used by the benchmark and how to build them (heavy imports are lazy).

Milestone 3 detectors (off-the-shelf) and Milestone 4 detectors (trained here) are listed
separately so that ``python -m privgate.bench all`` and the Milestone 3 tables keep exactly
their original scope. Trained models are loaded from :data:`MODEL_DIRS`, written by
``scripts/train_ml.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from privgate.detectors.base import Detector

BASE_DETECTORS = ("regex", "spacy", "presidio")
ML_DETECTORS = ("crf", "logreg")
M3_ENSEMBLES = ("regex+spacy", "regex+presidio")
ENSEMBLES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "regex+spacy": ("regex", "spacy"),
    "regex+presidio": ("regex", "presidio"),
    "crf+regex+presidio": ("crf", "regex", "presidio"),
})
M3_DETECTORS = BASE_DETECTORS + M3_ENSEMBLES
RUNNABLE = BASE_DETECTORS + ML_DETECTORS
ALL_DETECTORS = RUNNABLE + tuple(ENSEMBLES)
MODEL_DIRS: Mapping[str, Path] = MappingProxyType({
    "crf": Path("models/m4/crf"),
    "logreg": Path("models/m4/logreg"),
})


def members(name: str) -> tuple[str, ...]:
    """Detectors that make up ``name`` (itself for a non-ensemble detector)."""
    if name in ENSEMBLES:
        return ENSEMBLES[name]
    if name in RUNNABLE:
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
    if name == "crf":
        from privgate.ml.crf import CRFDetector

        return CRFDetector.load(MODEL_DIRS["crf"])
    if name == "logreg":
        from privgate.ml.logreg import TokenLogRegDetector

        return TokenLogRegDetector.load(MODEL_DIRS["logreg"])
    if name in ENSEMBLES:
        from privgate.detectors.ensemble import EnsembleDetector

        return EnsembleDetector([make_detector(m, presidio_threshold) for m in ENSEMBLES[name]],
                                name=name)
    raise ValueError(f"unknown detector {name!r}; expected one of {ALL_DETECTORS}")
