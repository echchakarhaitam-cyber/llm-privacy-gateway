"""Paired comparison of two detectors on the same documents.

Both systems are scored on the SAME bootstrap resamples of documents (seed 0, 1000
resamples), and the statistic is the difference A - B of leakage rate and of type-agnostic
micro F1 (all gold spans). Pairing removes the between-document variance shared by both
systems, so the interval of the difference is much narrower than the two separate
intervals would suggest. A difference whose interval excludes 0 is reported as such;
nothing else is claimed.
"""

from __future__ import annotations

from collections.abc import Sequence

from privgate.bench.bootstrap import (
    N_RESAMPLES,
    SEED,
    bootstrap_cis,
    f1_from_counts,
    leakage_from_counts,
)
from privgate.data.documents import Document
from privgate.evaluation import evaluate_document
from privgate.spans import Span


def _columns(docs: Sequence[Document], preds: Sequence[Sequence[Span]], tag: str) -> dict:
    cols: dict[str, list[int]] = {f"{tag}_{k}": [] for k in ("tp", "fp", "fn", "gold", "cov")}
    for doc, pred in zip(docs, preds):
        rep = evaluate_document(list(doc.spans), list(pred))
        micro = rep.type_agnostic.micro
        cols[f"{tag}_tp"].append(micro.tp)
        cols[f"{tag}_fp"].append(micro.fp)
        cols[f"{tag}_fn"].append(micro.fn)
        cols[f"{tag}_gold"].append(rep.chars.gold_chars)
        cols[f"{tag}_cov"].append(rep.chars.covered_gold_chars)
    return cols


def _f1(s, t):
    return f1_from_counts(s[f"{t}_tp"], s[f"{t}_fp"], s[f"{t}_fn"])


def _leak(s, t):
    return leakage_from_counts(s[f"{t}_gold"], s[f"{t}_cov"])


def paired_bootstrap(
    docs: Sequence[Document],
    preds_a: Sequence[Sequence[Span]],
    preds_b: Sequence[Sequence[Span]],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict:
    if not len(docs) == len(preds_a) == len(preds_b):
        raise ValueError("both systems need one prediction list per document")
    columns = {**_columns(docs, preds_a, "a"), **_columns(docs, preds_b, "b")}
    stats = {
        "leakage_delta": lambda s: _leak(s, "a") - _leak(s, "b"),
        "type_agnostic_f1_delta": lambda s: _f1(s, "a") - _f1(s, "b"),
    }
    cis = bootstrap_cis(columns, stats, n_resamples, seed)
    full = {k: sum(v) for k, v in columns.items()}
    out = {"documents": len(docs), "n_resamples": n_resamples, "seed": seed}
    for name, a, b in (("leakage_rate", _leak(full, "a"), _leak(full, "b")),
                       ("type_agnostic_f1", _f1(full, "a"), _f1(full, "b"))):
        key = "leakage_delta" if name == "leakage_rate" else "type_agnostic_f1_delta"
        lo, hi = cis[key]
        out[name] = {"a": a, "b": b, "delta": a - b, "ci95": [lo, hi],
                     "ci_excludes_zero": lo > 0 or hi < 0}
    return out
