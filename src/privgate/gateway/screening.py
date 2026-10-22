"""The screening pipeline: detect -> validate -> apply policy -> mask -> verify.

Everything here is FAIL CLOSED: any outcome other than "every string was screened, masked and
the masking was verified to round-trip" raises, and the caller must then NOT contact the
upstream. The rules (each has a unit test in tests/test_gateway_screening.py):

========================  ==========================================================
reason                    when
========================  ==========================================================
``not_ready``             no detector could be loaded
``detector_error``        the detector raised any exception
``detector_timeout``      screening the request took longer than the time budget
``malformed_output``      the detector returned something that is not a list of
                          :class:`~privgate.spans.Span` that are inside the text, sorted
                          and non-overlapping (the :class:`Detector` contract)
``mapping_error``         masking failed, produced inconsistent edits, or the masked
                          text does not unmask back to the original (placeholder collision)
``internal_error``        any other exception in the screening path
========================  ==========================================================

All strings of a request are masked in ONE pass over their concatenation (separated by a
newline), so one request has one mapping and a value gets the same placeholder in every
message. Detectors still see each string on its own, with offsets relative to it.

Detectors are not documented as thread-safe, so by default one worker thread runs them
(``workers=1``). A detector that hangs keeps that thread busy until it returns: later
requests then time out too and are blocked (availability suffers, privacy does not).
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event

from privgate.gateway.policy import Policy
from privgate.gateway.schema import ChatRequest
from privgate.masking import Masker, MaskingError, unmask
from privgate.spans import Span

DEFAULT_TIMEOUT_S = 2.0
SEPARATOR = "\n"


class ScreeningUnavailable(Exception):
    """Screening could not be completed; the request must be blocked (503)."""

    def __init__(self, reason: str, cause_class: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.cause_class = cause_class


class PolicyViolation(Exception):
    """The request contains an entity type the policy blocks (403). Labels only, no values."""

    def __init__(self, labels: list[str]):
        super().__init__(", ".join(labels))
        self.labels = tuple(labels)


@dataclass(frozen=True)
class Screened:
    """A fully screened request. ``mapping`` is as sensitive as the text: keep it local."""

    request: ChatRequest
    mapping: Mapping[str, str]
    counts: Mapping[str, int]
    detect_ms: float
    mask_ms: float


def validate_spans(text: str, spans: object) -> list[Span]:
    """The detector contract, enforced. Raises ``ScreeningUnavailable('malformed_output')``."""
    if not isinstance(spans, (list, tuple)):
        raise ScreeningUnavailable("malformed_output", "NotAList")
    previous_end = 0
    for s in spans:
        if not isinstance(s, Span):
            raise ScreeningUnavailable("malformed_output", "NotASpan")
        if s.start < 0 or s.end <= s.start or s.end > len(text):
            raise ScreeningUnavailable("malformed_output", "SpanOutOfBounds")
        if s.start < previous_end:
            raise ScreeningUnavailable("malformed_output", "SpansOverlapOrUnsorted")
        previous_end = s.end
    return list(spans)


class Screener:
    def __init__(self, detector, policy: Policy | None = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S, workers: int = 1,
                 masker: Masker | None = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.detector = detector
        self.policy = policy or Policy()
        self.timeout_s = timeout_s
        self._masker = masker or Masker()
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="privgate-detect")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def screen(self, request: ChatRequest, seed: int,
                     timeout_s: float | None = None) -> Screened:
        """Mask every string of ``request``; raises PolicyViolation or ScreeningUnavailable."""
        if self.detector is None:
            raise ScreeningUnavailable("not_ready")
        texts = request.strings()
        budget = self.timeout_s if timeout_s is None else timeout_s
        t0 = time.perf_counter()
        try:
            raw = await self._detect(texts, budget)
            spans = [validate_spans(t, s) for t, s in zip(texts, raw, strict=True)]
            detect_ms = (time.perf_counter() - t0) * 1e3
            blocked = self.policy.blocked(s.label for group in spans for s in group)
            if blocked:
                raise PolicyViolation([lab.value for lab in blocked])
            t1 = time.perf_counter()
            masked_request, mapping, counts = self._mask(request, texts, spans, seed)
        except (ScreeningUnavailable, PolicyViolation):
            raise
        except Exception as exc:    # noqa: BLE001 - fail closed: anything unexpected blocks
            raise ScreeningUnavailable("internal_error", type(exc).__name__) from None
        return Screened(masked_request, mapping, counts, detect_ms,
                        (time.perf_counter() - t1) * 1e3)

    async def _detect(self, texts: list[str], budget: float) -> list[list[Span]]:
        detector, cancel = self.detector, Event()

        def work() -> list:
            out = []
            for text in texts:
                if cancel.is_set():
                    raise TimeoutError("screening abandoned")
                out.append(detector.detect(text) if text else [])
            return out

        future = asyncio.get_running_loop().run_in_executor(self._executor, work)
        try:
            return await asyncio.wait_for(future, budget)
        except asyncio.TimeoutError:
            cancel.set()
            raise ScreeningUnavailable("detector_timeout") from None
        except Exception as exc:    # noqa: BLE001 - a detector may raise anything
            raise ScreeningUnavailable("detector_error", type(exc).__name__) from None

    def _mask(self, request: ChatRequest, texts: list[str], spans: list[list[Span]],
              seed: int) -> tuple[ChatRequest, Mapping[str, str], Counter]:
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t) + len(SEPARATOR)
        combined = SEPARATOR.join(texts)
        shifted = [Span(s.start + off, s.end + off, s.label, s.score, s.source)
                   for off, group in zip(starts, spans, strict=True) for s in group]
        try:
            masked = self._masker.mask(combined, shifted, "typed", seed)
        except MaskingError:
            raise ScreeningUnavailable("mapping_error", "MaskingError") from None
        if len(masked.edits) != len(shifted):
            raise ScreeningUnavailable("mapping_error", "EditCountMismatch")
        if len({e.replacement for e in masked.edits}) != len(masked.mapping):
            raise ScreeningUnavailable("mapping_error", "AmbiguousMapping")

        # original offset -> masked offset (offsets never fall inside an edit)
        edits = sorted(masked.edits, key=lambda e: e.start)

        def out_pos(p: int) -> int:
            return p + sum(e.out_end - e.out_start - (e.end - e.start)
                           for e in edits if e.end <= p)

        pieces = []
        for text, off in zip(texts, starts, strict=True):
            piece = masked.text[out_pos(off):out_pos(off + len(text))]
            if unmask(piece, masked.mapping) != text:     # the round trip must be exact
                raise ScreeningUnavailable("mapping_error", "RoundTripMismatch")
            pieces.append(piece)
        counts = Counter(e.label.value for e in masked.edits)
        return request.with_strings(pieces), masked.mapping, counts
