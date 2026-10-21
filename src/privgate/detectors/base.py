"""The interface every PII detector implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from privgate.spans import EntityType, Span


@runtime_checkable
class Detector(Protocol):
    """Finds personal data in a text.

    ``detect`` must return non-overlapping spans sorted by start offset, with offsets
    relative to the exact string it received. It must not modify the text.

    ``supported_labels`` lists the entity types the detector can emit; evaluation uses it
    to separate "the tool missed what it targets" from "the tool does not target this".
    """

    name: str
    supported_labels: frozenset[EntityType]

    def detect(self, text: str) -> list[Span]: ...
