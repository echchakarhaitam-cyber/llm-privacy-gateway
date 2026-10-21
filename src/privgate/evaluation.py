"""Span-level and character-level evaluation of PII detectors.

Three span matching modes answer three different questions:

* ``strict``: did the detector find exactly this entity, with exact boundaries and the
  right type? (classic NER evaluation)
* ``partial``: did it find this entity with the right type, even if the boundaries are
  off? A prediction matches a gold span of the same label when they share at least one
  character; matching is one-to-one.
* ``type_agnostic``: did it hide this entity at all, whatever type it gave it? (the
  question a privacy gateway cares about most)

The character-level view goes further: a missed character can leak data, while an
extra masked character only costs utility. ``leakage_rate`` is the share of gold
(sensitive) characters left uncovered by any prediction.

Conventions:
* a ratio with a zero denominator is 0.0 (e.g. precision of a label never predicted);
* macro averages are taken over the UNION of labels present in gold or predictions;
* in ``type_agnostic`` mode, true positives and false negatives are attributed to the
  gold label and false positives to the predicted label, so per-label recall still
  says which entity types leak;
* corpus scores are micro-averaged: counts are summed over documents first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from privgate.spans import EntityType, Span

MODES = ("strict", "partial", "type_agnostic")


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass
class Counts:
    """True positives, false positives and false negatives, with derived scores."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        """Accumulate ``other`` into these counts (in place)."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class ModeReport:
    """Scores of one matching mode."""

    per_label: dict[EntityType, Counts] = field(default_factory=dict)

    @property
    def micro(self) -> Counts:
        total = Counts()
        for c in self.per_label.values():
            total.add(c)
        return total

    def _macro(self, attr: str) -> float:
        if not self.per_label:
            return 0.0
        return sum(getattr(c, attr) for c in self.per_label.values()) / len(self.per_label)

    @property
    def macro_precision(self) -> float:
        return self._macro("precision")

    @property
    def macro_recall(self) -> float:
        return self._macro("recall")

    @property
    def macro_f1(self) -> float:
        return self._macro("f1")


@dataclass
class CharReport:
    """Character-level coverage (the privacy view)."""

    gold_chars: int = 0
    covered_gold_chars: int = 0
    pred_chars: int = 0
    pred_chars_in_gold: int = 0

    @property
    def char_recall(self) -> float:
        return _ratio(self.covered_gold_chars, self.gold_chars)

    @property
    def leakage_rate(self) -> float:
        """Share of sensitive characters left visible; 0.0 when there is nothing to hide."""
        return 1.0 - self.char_recall if self.gold_chars else 0.0

    @property
    def char_precision(self) -> float:
        return _ratio(self.pred_chars_in_gold, self.pred_chars)


@dataclass
class EvaluationReport:
    strict: ModeReport
    partial: ModeReport
    type_agnostic: ModeReport
    chars: CharReport
    n_documents: int

    def mode(self, name: str) -> ModeReport:
        if name not in MODES:
            raise KeyError(f"unknown mode {name!r}; expected one of {MODES}")
        return getattr(self, name)


def _match(
    gold: Sequence[Span], pred: Sequence[Span], mode: str
) -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    """One-to-one greedy matching. Returns (pairs, unmatched gold, unmatched pred)."""
    candidates: list[tuple[int, int, int, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if mode == "strict":
                ok = (g.start, g.end, g.label) == (p.start, p.end, p.label)
            elif mode == "partial":
                ok = g.label == p.label and g.overlaps(p)
            elif mode == "type_agnostic":
                ok = g.overlaps(p)
            else:
                raise KeyError(f"unknown mode {mode!r}; expected one of {MODES}")
            if ok:
                # Largest overlap first; ties: smaller gold start, then smaller pred start.
                candidates.append((-g.overlap_length(p), g.start, p.start, gi, pi))
    candidates.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[Span, Span]] = []
    for _, _, _, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            pairs.append((gold[gi], pred[pi]))
    unmatched_gold = [g for i, g in enumerate(gold) if i not in used_g]
    unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    return pairs, unmatched_gold, unmatched_pred


def match_spans(
    gold: Sequence[Span], pred: Sequence[Span], mode: str
) -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    """The one-to-one matching behind every score: (pairs, missed gold, spurious predictions).

    Public so that error analysis inspects exactly the false negatives and false positives
    the metrics count.
    """
    return _match(gold, pred, mode)


def evaluate_mode(gold: Sequence[Span], pred: Sequence[Span], mode: str) -> ModeReport:
    """Per-label counts for one document and one matching mode."""
    pairs, missed, spurious = _match(gold, pred, mode)
    report = ModeReport()

    def bump(label: EntityType) -> Counts:
        return report.per_label.setdefault(label, Counts())

    for g, _ in pairs:
        bump(g.label).tp += 1
    for g in missed:
        bump(g.label).fn += 1
    for p in spurious:
        bump(p.label).fp += 1
    return report


def _union(spans: Iterable[Span]) -> list[tuple[int, int]]:
    intervals = sorted((s.start, s.end) for s in spans)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Characters of interval union ``a`` that also lie in interval union ``b``."""
    total = i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def evaluate_chars(gold: Sequence[Span], pred: Sequence[Span]) -> CharReport:
    g, p = _union(gold), _union(pred)
    shared = _covered(g, p)
    return CharReport(
        gold_chars=sum(e - s for s, e in g),
        covered_gold_chars=shared,
        pred_chars=sum(e - s for s, e in p),
        pred_chars_in_gold=shared,
    )


def evaluate_document(gold: Sequence[Span], pred: Sequence[Span]) -> EvaluationReport:
    return evaluate_corpus([(gold, pred)])


def evaluate_corpus(docs: Iterable[tuple[Sequence[Span], Sequence[Span]]]) -> EvaluationReport:
    """Micro-averaged evaluation over documents: counts are summed, then ratios computed."""
    modes = {m: ModeReport() for m in MODES}
    chars = CharReport()
    n = 0
    for gold, pred in docs:
        n += 1
        for m in MODES:
            for label, counts in evaluate_mode(gold, pred, m).per_label.items():
                modes[m].per_label.setdefault(label, Counts())
                modes[m].per_label[label].add(counts)
        c = evaluate_chars(gold, pred)
        chars.gold_chars += c.gold_chars
        chars.covered_gold_chars += c.covered_gold_chars
        chars.pred_chars += c.pred_chars
        chars.pred_chars_in_gold += c.pred_chars_in_gold
    return EvaluationReport(
        strict=modes["strict"],
        partial=modes["partial"],
        type_agnostic=modes["type_agnostic"],
        chars=chars,
        n_documents=n,
    )
