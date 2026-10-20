"""Character spans: the common currency of every detector and every metric.

A span marks ``text[start:end]`` (``end`` exclusive, Python slice convention) as one
entity of a given type. Detectors return spans, gold annotations are spans, the
evaluation compares spans, and the gateway masks spans. Keeping a single, validated
representation avoids the classic off-by-one disagreements between components.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    """Entity labels used across the project.

    Dataset-specific labels are mapped onto this smaller, stable set so that
    detectors with different native taxonomies can be compared on equal terms.
    """

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    DATE = "DATE"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    ID_NUMBER = "ID_NUMBER"
    OTHER = "OTHER"


@dataclass(frozen=True, order=True)
class Span:
    """A labelled half-open character interval ``[start, end)``.

    ``score`` is the detector's confidence in ``[0, 1]`` (1.0 for rule-based hits and
    gold annotations). ``source`` names the detector that produced the span; it is
    informative only.
    """

    start: int
    end: int
    label: EntityType
    score: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Span.{name} must be an int, got {value!r}")
        if self.start < 0:
            raise ValueError(f"Span.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Span.end must be > start, got [{self.start}, {self.end})")
        if not isinstance(self.label, EntityType):
            try:
                object.__setattr__(self, "label", EntityType(self.label))
            except ValueError as exc:
                raise ValueError(f"unknown entity label {self.label!r}") from exc
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(f"Span.score must be a number, got {self.score!r}")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Span.score must be in [0, 1], got {self.score}")
        object.__setattr__(self, "score", float(self.score))

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap_length(self, other: Span) -> int:
        """Number of characters shared with ``other`` (0 if disjoint)."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def overlaps(self, other: Span) -> bool:
        return self.overlap_length(other) > 0

    def text(self, of: str) -> str:
        """The covered substring of ``of``; raises if the span lies outside the text."""
        if self.end > len(of):
            raise ValueError(f"span [{self.start}, {self.end}) exceeds text length {len(of)}")
        return of[self.start : self.end]


def resolve_overlaps(
    spans: Iterable[Span], priority: Sequence[EntityType] | None = None
) -> list[Span]:
    """Return a non-overlapping subset of ``spans``, sorted by position.

    Greedy rule, applied in this order until one decides:
      1. the longer span wins (it usually contains the shorter, e.g. a URL containing an
         email);
      2. the higher score wins;
      3. the label listed earlier in ``priority`` wins (labels not listed rank last);
      4. the span that starts earlier wins, then the one that ends earlier, then the
         label whose name sorts first.
    Every tie is broken, so the result does not depend on the input order.
    """
    rank = {label: i for i, label in enumerate(priority or ())}
    last = len(rank)

    def key(s: Span) -> tuple[int, float, int, int, int, str]:
        return (-s.length, -s.score, rank.get(s.label, last), s.start, s.end, s.label.value)

    kept: list[Span] = []
    for span in sorted(spans, key=key):
        if all(not span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end, s.label.value))
