"""Presidio score-threshold selection, on the development set only.

The rule below was fixed in code before any test-set result existed (see git history):

    On nemotron_dev_2k, against all gold spans, compute for every threshold in THRESHOLDS
    the character leakage_rate and the type-agnostic micro F1. Keep the thresholds whose
    leakage_rate is at most LEAKAGE_TOLERANCE (absolute) above the lowest leakage_rate of
    the sweep; among them choose the highest type-agnostic micro F1; a tie goes to the
    lower threshold.

Why this rule: for a privacy gateway, leaking less matters more than flagging less, so the
leakage criterion comes first; within a near-equal leakage band the threshold that wastes
the fewest masks (higher F1) is preferred. The chosen threshold is then used, unchanged, on
the test sets. Test sets are never used for tuning.

The sweep re-filters Presidio results obtained once at threshold 0.0; this equals running
Presidio at each threshold (see PresidioDetector.spans_from_results).
"""

from __future__ import annotations

from collections.abc import Sequence

from privgate.data.documents import Document
from privgate.evaluation import evaluate_corpus

THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85)
LEAKAGE_TOLERANCE = 0.01
SELECTION_RULE = (
    "On nemotron_dev_2k (all gold spans): keep thresholds whose leakage_rate is within "
    f"{LEAKAGE_TOLERANCE} of the minimum leakage_rate over {list(THRESHOLDS)}; choose the "
    "highest type-agnostic micro F1 among them; ties go to the lower threshold."
)


def sweep_rows(detector, raw_results: Sequence[Sequence], docs: Sequence[Document],
               thresholds: Sequence[float] = THRESHOLDS) -> list[dict]:
    """One row of dev metrics per threshold, from results obtained at threshold 0.0."""
    golds = [list(d.spans) for d in docs]
    rows = []
    for t in thresholds:
        preds = [detector.spans_from_results(raw, t) for raw in raw_results]
        report = evaluate_corpus(zip(golds, preds))
        ta = report.type_agnostic.micro
        rows.append({
            "threshold": t,
            "pred_spans": sum(len(p) for p in preds),
            "strict_f1": report.strict.micro.f1,
            "partial_f1": report.partial.micro.f1,
            "type_agnostic_precision": ta.precision,
            "type_agnostic_recall": ta.recall,
            "type_agnostic_f1": ta.f1,
            "leakage_rate": report.chars.leakage_rate,
            "char_precision": report.chars.char_precision,
        })
    return rows


def select_threshold(rows: Sequence[dict], tolerance: float = LEAKAGE_TOLERANCE) -> float:
    """Apply SELECTION_RULE to sweep rows."""
    if not rows:
        raise ValueError("empty sweep")
    floor = min(r["leakage_rate"] for r in rows)
    eligible = [r for r in rows if r["leakage_rate"] <= floor + tolerance + 1e-12]
    best = max(eligible, key=lambda r: (r["type_agnostic_f1"], -r["threshold"]))
    return best["threshold"]
