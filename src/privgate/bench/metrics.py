"""Benchmark metrics of one detector on one evaluation set, in two views.

* ``all``: every gold span counts. This is the privacy reality: whatever is not detected
  leaks, whether or not the tool claims to handle it.
* ``supported``: gold restricted to the labels the detector supports
  (``supported_labels``); predictions are unchanged, so a prediction landing on an
  unsupported gold span counts as a false positive here. This asks whether the tool is good
  at the job it claims.

Per view: strict / partial / type-agnostic micro P/R/F1 with TP/FP/FN, macro F1 and
per-label scores (privgate.evaluation); character-level recall, leakage and precision,
leakage per gold label (and, in ``all``, per raw dataset label when the raw-label index is
available); 95% bootstrap CIs (documents resampled) for each mode's micro F1 and for the
leakage rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from privgate.bench.bootstrap import (
    N_RESAMPLES,
    SEED,
    bootstrap_cis,
    f1_from_counts,
    leakage_from_counts,
)
from privgate.data.documents import Document
from privgate.evaluation import (
    MODES,
    Counts,
    ModeReport,
    evaluate_chars,
    evaluate_corpus,
    evaluate_document,
)
from privgate.spans import EntityType, Span

VIEWS = ("all", "supported")


def _counts(c: Counts) -> dict:
    return {"tp": c.tp, "fp": c.fp, "fn": c.fn,
            "precision": c.precision, "recall": c.recall, "f1": c.f1}


def _mode(report: ModeReport) -> dict:
    return {
        "micro": _counts(report.micro),
        "macro_f1": report.macro_f1,
        "per_label": {label.value: _counts(report.per_label[label])
                      for label in sorted(report.per_label, key=lambda e: e.value)},
    }


def _leakage_table(acc: dict[str, list[int]]) -> dict:
    return {
        key: {"gold_chars": g, "leaked_chars": g - c, "leakage_rate": leakage_from_counts(g, c)}
        for key, (g, c) in sorted(acc.items())
    }


def evaluate_view(
    golds: Sequence[Sequence[Span]],
    preds: Sequence[Sequence[Span]],
    raw_labels: Sequence[Sequence[str]] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict:
    """Metrics of predictions against one gold view; ``raw_labels`` aligned with ``golds``."""
    if len(golds) != len(preds):
        raise ValueError("one prediction list per document is required")
    corpus = evaluate_corpus(zip(golds, preds))
    columns: dict[str, list[int]] = {f"{m}_{k}": [] for m in MODES for k in ("tp", "fp", "fn")}
    columns["gold_chars"], columns["covered"] = [], []
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_raw: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, (gold, pred) in enumerate(zip(golds, preds)):
        doc = evaluate_document(gold, pred)
        for m in MODES:
            micro = doc.mode(m).micro
            columns[f"{m}_tp"].append(micro.tp)
            columns[f"{m}_fp"].append(micro.fp)
            columns[f"{m}_fn"].append(micro.fn)
        columns["gold_chars"].append(doc.chars.gold_chars)
        columns["covered"].append(doc.chars.covered_gold_chars)
        groups = [(per_label, [g.label.value for g in gold])]
        if raw_labels is not None:
            groups.append((per_raw, list(raw_labels[i])))
        for acc, keys in groups:
            for key in set(keys):
                chars = evaluate_chars([g for g, k in zip(gold, keys) if k == key], pred)
                acc[key][0] += chars.gold_chars
                acc[key][1] += chars.covered_gold_chars

    statistics = {
        f"{m}_micro_f1": (lambda s, m=m: f1_from_counts(s[f"{m}_tp"], s[f"{m}_fp"], s[f"{m}_fn"]))
        for m in MODES
    }
    statistics["leakage_rate"] = lambda s: leakage_from_counts(s["gold_chars"], s["covered"])
    cis = bootstrap_cis(columns, statistics, n_resamples, seed) if golds else {}

    chars = corpus.chars
    result = {
        "documents": len(golds),
        "gold_spans": sum(len(g) for g in golds),
        "pred_spans": sum(len(p) for p in preds),
        **{m: _mode(corpus.mode(m)) for m in MODES},
        "chars": {
            "gold_chars": chars.gold_chars,
            "covered_gold_chars": chars.covered_gold_chars,
            "pred_chars": chars.pred_chars,
            "char_recall": chars.char_recall,
            "leakage_rate": chars.leakage_rate,
            "char_precision": chars.char_precision,
            "per_gold_label": _leakage_table(per_label),
        },
        "ci95": {name: list(ci) for name, ci in cis.items()},
    }
    if raw_labels is not None:
        result["chars"]["per_raw_label"] = _leakage_table(per_raw)
    return result


def restrict(gold: Sequence[Span], supported: Iterable[EntityType]) -> list[Span]:
    allowed = frozenset(supported)
    return [g for g in gold if g.label in allowed]


def evaluate_run(
    docs: Sequence[Document],
    preds: Sequence[Sequence[Span]],
    supported_labels: Iterable[EntityType],
    raw_labels: dict[str, list[str]] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict:
    """Both views for one run. ``raw_labels`` maps doc_id -> raw label per gold span."""
    golds = [list(d.spans) for d in docs]
    raws = [raw_labels[d.doc_id] for d in docs] if raw_labels is not None else None
    supported = frozenset(supported_labels)
    return {
        "all": evaluate_view(golds, preds, raws, n_resamples, seed),
        "supported": evaluate_view([restrict(g, supported) for g in golds], preds, None,
                                   n_resamples, seed),
    }
