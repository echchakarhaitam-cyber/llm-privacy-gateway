"""The interface every PII detector implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from privgate.spans import Span


@runtime_checkable
class Detector(Protocol):
    """Finds personal data in a text.

    ``detect`` must return non-overlapping spans sorted by start offset, with offsets
    relative to the exact string it received. It must not modify the text.
    """

    name: str

    def detect(self, text: str) -> list[Span]: ...
