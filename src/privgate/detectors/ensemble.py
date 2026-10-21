"""Union ensemble: every member's spans, then :func:`~privgate.spans.resolve_overlaps`.

A union maximises coverage (the privacy goal: a character hidden by any member stays
hidden), at the cost of precision. Overlaps between members are resolved with the same
deterministic rule as everywhere else (longer span, then higher score, ...).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from privgate.detectors.base import Detector
from privgate.spans import EntityType, Span, resolve_overlaps


class EnsembleDetector:
    """Runs its members in order and combines their spans."""

    def __init__(self, detectors: Sequence[Detector], name: str):
        if not detectors:
            raise ValueError("an ensemble needs at least one detector")
        if not name:
            raise ValueError("an ensemble needs a name")
        self.detectors = tuple(detectors)
        self.name = name
        self.supported_labels: frozenset[EntityType] = frozenset().union(
            *(d.supported_labels for d in self.detectors))

    def detect(self, text: str) -> list[Span]:
        return self.combine(d.detect(text) for d in self.detectors)

    @staticmethod
    def combine(member_spans: Iterable[Iterable[Span]]) -> list[Span]:
        """The ensemble's output given each member's output (used to reuse cached runs)."""
        return resolve_overlaps(span for spans in member_spans for span in spans)
