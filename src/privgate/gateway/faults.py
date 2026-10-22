"""Fault injection for the RQ9 experiment and the tests. NEVER imported by the gateway itself.

A :class:`FaultInjector` holds the fault of the request currently in flight (the experiment
sends requests one at a time). :class:`FaultyDetector` and :func:`faulty_upstream` consult it:

``exception``    the detector raises (the message quotes the text, like a real parser error)
``hang``         the detector sleeps ``hang_s`` (longer than the gateway's timeout) first
``malformed``    the detector returns spans that break its contract (rotating through
                 out-of-bounds, overlapping, and not-a-span variants)
``upstream_5xx`` the upstream answers with a 503 (after recording the payload it received)

:class:`ReplayDetector` returns stored spans for known texts. Running the real CRF + Presidio
pipeline 40,000 times to measure fault handling would only measure the detectors' speed; the
gateway's behaviour under faults depends on what the detector returns, not on how it got it.
An unknown text raises, which the gateway treats as a detector failure (fail closed).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from privgate.gateway.upstream import MockUpstream, UpstreamError
from privgate.spans import EntityType, Span

FAULT_KINDS = ("exception", "hang", "malformed", "upstream_5xx")
MIXED = "mixed"
MALFORMED_VARIANTS = ("out_of_bounds", "overlap", "not_a_span")


class FaultInjector:
    """The fault to apply to the request in flight (``None`` = no fault)."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.malformed_used = 0


class ReplayDetector:
    """Detector that replays stored spans keyed by the exact text."""

    def __init__(self, spans_by_text: Mapping[str, Sequence[Span]], name: str = "replay"):
        self.name = name
        self.supported_labels = frozenset(EntityType)
        self._spans = {text: tuple(spans) for text, spans in spans_by_text.items()}

    def detect(self, text: str) -> list[Span]:
        return list(self._spans[text])         # KeyError for an unknown text -> fail closed


class FaultyDetector:
    def __init__(self, inner, injector: FaultInjector, hang_s: float = 0.25):
        self.inner = inner
        self.injector = injector
        self.hang_s = hang_s
        self.name = getattr(inner, "name", "detector")
        self.supported_labels = getattr(inner, "supported_labels", frozenset(EntityType))

    def detect(self, text: str):
        kind = self.injector.current
        if kind == "exception":
            raise RuntimeError(f"injected detector failure while reading: {text[:60]}")
        if kind == "hang":
            time.sleep(self.hang_s)
        elif kind == "malformed":
            n = len(text)
            variant = MALFORMED_VARIANTS[self.injector.malformed_used % len(MALFORMED_VARIANTS)]
            self.injector.malformed_used += 1
            if variant == "overlap" and n >= 2:
                return [Span(0, n, EntityType.PERSON), Span(n // 2, n, EntityType.EMAIL)]
            if variant == "not_a_span":
                return [(0, min(n, 5), "PERSON")]
            return [Span(max(0, n - 3), n + 5, EntityType.PERSON)]     # out of bounds
        return self.inner.detect(text)


def faulty_upstream(injector: FaultInjector, **kw) -> MockUpstream:
    """A MockUpstream that fails with a 503 while ``injector.current == 'upstream_5xx'``."""
    def failure(_payload: dict):
        return UpstreamError("injected upstream failure", 503) \
            if injector.current == "upstream_5xx" else None
    return MockUpstream(failure=failure, **kw)
