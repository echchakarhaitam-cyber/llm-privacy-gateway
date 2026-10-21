"""Annotated documents and the normalisation every gold span goes through.

Every dataset loader turns its own row format into :class:`RawSpan` records and hands them
to :func:`build_document`, so all datasets pass the SAME checks in the same order:

1. map the raw label onto :class:`~privgate.spans.EntityType` (unknown labels are an error
   in strict mode, ``OTHER`` otherwise, and always counted);
2. drop spans whose offsets fall outside the text;
3. verify offsets: when the dataset stores the covered text, ``text[start:end]`` must equal
   it (``offset_check="exact"``), or equal it up to letter case (``"casefold"``, default --
   see docs/DATA.md for why); mismatching spans are dropped and counted;
4. trim surrounding whitespace, because detectors never report it;
5. find overlapping gold spans and, by default, resolve them with
   :func:`~privgate.spans.resolve_overlaps` (the longer span wins);
6. optionally merge same-label neighbours separated only by spaces or tabs on one line
   (``"John" + "Smith"`` -> one PERSON), because detectors report a full name as one entity.

Nothing is dropped silently: every removed span is counted by reason in a
:class:`LoadReport`, and ``spans_read == spans_kept + sum(spans_dropped) + spans_merged``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from privgate.data.mapping import map_label
from privgate.spans import EntityType, Span, resolve_overlaps

OFFSET_CHECKS = ("exact", "casefold")
OVERLAP_POLICIES = ("resolve", "keep")

# Drop reasons, in the order the checks run.
OUT_OF_BOUNDS = "out_of_bounds"
OFFSET_MISMATCH = "offset_mismatch"
EMPTY_AFTER_TRIM = "empty_after_trim"
OVERLAP = "overlap"
DROP_REASONS = (OUT_OF_BOUNDS, OFFSET_MISMATCH, EMPTY_AFTER_TRIM, OVERLAP)

# Horizontal whitespace only: a line break between two names usually separates two
# different entities (a surname at the end of one line, a first name starting the next).
_MERGEABLE_GAP = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Document:
    """One annotated text. Immutable, so an evaluation set cannot be edited by accident.

    ``spans`` are sorted by position. ``meta`` holds string facts about the document
    (domain, document_type, locale, ...); it is copied into a read-only mapping.
    """

    doc_id: str
    text: str
    spans: tuple[Span, ...]
    source: str
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("Document.doc_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("Document.text must be a string")
        spans = tuple(self.spans)
        for s in spans:
            if not isinstance(s, Span):
                raise TypeError(f"Document.spans must contain Span objects, got {s!r}")
            if s.end > len(self.text):
                raise ValueError(f"{self.doc_id}: span [{s.start}, {s.end}) exceeds text")
        object.__setattr__(self, "spans", tuple(sorted(spans, key=_position)))
        meta = dict(self.meta)
        for k, v in meta.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError(f"Document.meta must map str to str, got {k!r}: {v!r}")
        object.__setattr__(self, "meta", MappingProxyType(meta))

    __hash__ = None  # type: ignore[assignment]  # meta is a mapping; compare, don't hash

    def span_text(self, span: Span) -> str:
        return span.text(self.text)


@dataclass(frozen=True)
class RawSpan:
    """A gold span as the dataset stores it, before any check. ``text`` may be absent."""

    start: object
    end: object
    label: str
    text: str | None = None


@dataclass(frozen=True)
class LoadOptions:
    """How strictly to treat a dataset. The defaults are the documented project policy."""

    merge_adjacent: bool = True
    strict_labels: bool = False
    offset_check: str = "casefold"
    overlap_policy: str = "resolve"

    def __post_init__(self) -> None:
        if self.offset_check not in OFFSET_CHECKS:
            raise ValueError(f"offset_check must be one of {OFFSET_CHECKS}")
        if self.overlap_policy not in OVERLAP_POLICIES:
            raise ValueError(f"overlap_policy must be one of {OVERLAP_POLICIES}")


@dataclass(frozen=True)
class LoadReport:
    """What a loader read, kept, dropped and changed.

    ``case_only_mismatches`` are spans kept under ``offset_check="casefold"`` whose stored
    text differs from ``text[start:end]`` only in letter case. ``overlapping_pairs`` counts
    pairs of gold spans sharing at least one character, before resolution.
    """

    dataset: str
    options: LoadOptions
    documents: int
    spans_read: int
    spans_kept: int
    spans_dropped: Mapping[str, int]
    spans_merged: int
    spans_trimmed: int
    case_only_mismatches: int
    overlapping_pairs: int
    raw_label_counts: Mapping[str, int]
    unknown_labels: Mapping[str, int]
    kept_label_counts: Mapping[str, int]

    def is_consistent(self) -> bool:
        """The bookkeeping identity every report must satisfy."""
        dropped = sum(self.spans_dropped.values())
        return self.spans_read == self.spans_kept + dropped + self.spans_merged

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "options": {
                "merge_adjacent": self.options.merge_adjacent,
                "strict_labels": self.options.strict_labels,
                "offset_check": self.options.offset_check,
                "overlap_policy": self.options.overlap_policy,
            },
            "documents": self.documents,
            "spans_read": self.spans_read,
            "spans_kept": self.spans_kept,
            "spans_dropped": dict(self.spans_dropped),
            "spans_merged": self.spans_merged,
            "spans_trimmed": self.spans_trimmed,
            "case_only_mismatches": self.case_only_mismatches,
            "overlapping_pairs": self.overlapping_pairs,
            "raw_label_counts": dict(sorted(self.raw_label_counts.items())),
            "unknown_labels": dict(sorted(self.unknown_labels.items())),
            "kept_label_counts": dict(sorted(self.kept_label_counts.items())),
        }


@dataclass
class LoadTally:
    """Mutable counters filled while a loader streams; :meth:`freeze` gives the report."""

    dataset: str
    options: LoadOptions
    documents: int = 0
    spans_read: int = 0
    spans_kept: int = 0
    spans_merged: int = 0
    spans_trimmed: int = 0
    case_only_mismatches: int = 0
    overlapping_pairs: int = 0
    spans_dropped: Counter = field(default_factory=Counter)
    raw_label_counts: Counter = field(default_factory=Counter)
    unknown_labels: Counter = field(default_factory=Counter)
    kept_label_counts: Counter = field(default_factory=Counter)

    def freeze(self) -> LoadReport:
        return LoadReport(
            dataset=self.dataset,
            options=self.options,
            documents=self.documents,
            spans_read=self.spans_read,
            spans_kept=self.spans_kept,
            spans_dropped=MappingProxyType({r: self.spans_dropped[r] for r in DROP_REASONS}),
            spans_merged=self.spans_merged,
            spans_trimmed=self.spans_trimmed,
            case_only_mismatches=self.case_only_mismatches,
            overlapping_pairs=self.overlapping_pairs,
            raw_label_counts=MappingProxyType(dict(self.raw_label_counts)),
            unknown_labels=MappingProxyType(dict(self.unknown_labels)),
            kept_label_counts=MappingProxyType(dict(self.kept_label_counts)),
        )


def _position(s: Span) -> tuple[int, int, str]:
    return (s.start, s.end, s.label.value)


def count_overlapping_pairs(spans: Iterable[Span]) -> int:
    """Number of span pairs that share at least one character."""
    ordered = sorted(spans, key=_position)
    pairs = 0
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start >= a.end:
                break
            pairs += 1
    return pairs


def merge_adjacent(spans: Iterable[Span], text: str) -> list[Span]:
    """Merge same-label neighbours separated only by spaces/tabs on one line.

    ``"John" + " " + "Smith"`` becomes one span; a comma, any other character, a line
    break or touching spans (empty gap) keep them apart. Chains merge fully
    (``"Mary" "Ann" "Lee"`` -> one span). Overlapping input spans are never merged.
    """
    merged: list[Span] = []
    for span in sorted(spans, key=_position):
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.label == span.label
            and span.start > prev.end
            and _MERGEABLE_GAP.fullmatch(text, prev.end, span.start)
        ):
            merged[-1] = Span(prev.start, span.end, prev.label, prev.score, prev.source)
        else:
            merged.append(span)
    return merged


def build_document(
    doc_id: str,
    text: str,
    raw_spans: Iterable[RawSpan],
    tally: LoadTally,
    meta: Mapping[str, str] | None = None,
) -> Document:
    """Run every gold span through the checks in the module docstring; update ``tally``."""
    opts = tally.options
    checked: list[Span] = []
    for raw in raw_spans:
        tally.spans_read += 1
        tally.raw_label_counts[raw.label] += 1
        label = map_label(tally.dataset, raw.label, strict=opts.strict_labels)
        if label is None:
            tally.unknown_labels[raw.label] += 1
            label = EntityType.OTHER
        span = _checked_span(raw, label, text, tally)
        if span is not None:
            checked.append(span)

    tally.overlapping_pairs += count_overlapping_pairs(checked)
    if opts.overlap_policy == "resolve":
        resolved = resolve_overlaps(checked)
        tally.spans_dropped[OVERLAP] += len(checked) - len(resolved)
        checked = resolved
    if opts.merge_adjacent:
        merged = merge_adjacent(checked, text)
        tally.spans_merged += len(checked) - len(merged)
        checked = merged

    tally.documents += 1
    tally.spans_kept += len(checked)
    tally.kept_label_counts.update(s.label.value for s in checked)
    return Document(doc_id, text, tuple(checked), tally.dataset, meta or {})


def _checked_span(raw: RawSpan, label: EntityType, text: str, tally: LoadTally) -> Span | None:
    start, end = raw.start, raw.end
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        tally.spans_dropped[OUT_OF_BOUNDS] += 1
        return None
    covered = text[start:end]
    if raw.text is not None and covered != raw.text:
        case_only = covered.casefold() == raw.text.casefold()
        if not (case_only and tally.options.offset_check == "casefold"):
            tally.spans_dropped[OFFSET_MISMATCH] += 1
            return None
        tally.case_only_mismatches += 1
    stripped = covered.strip()
    if not stripped:
        tally.spans_dropped[EMPTY_AFTER_TRIM] += 1
        return None
    if stripped != covered:
        tally.spans_trimmed += 1
        start += len(covered) - len(covered.lstrip())
        end = start + len(stripped)
    return Span(start, end, label, source=tally.dataset)
