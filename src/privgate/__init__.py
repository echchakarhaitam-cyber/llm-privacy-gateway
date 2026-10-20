"""privgate: measured PII detection and a fail-closed gateway for LLM requests."""

from privgate.detectors import Detector, RegexDetector
from privgate.evaluation import (
    CharReport,
    Counts,
    EvaluationReport,
    ModeReport,
    evaluate_corpus,
    evaluate_document,
)
from privgate.spans import EntityType, Span, resolve_overlaps

__version__ = "0.1.0"

__all__ = [
    "CharReport",
    "Counts",
    "Detector",
    "EntityType",
    "EvaluationReport",
    "ModeReport",
    "RegexDetector",
    "Span",
    "evaluate_corpus",
    "evaluate_document",
    "resolve_overlaps",
]
