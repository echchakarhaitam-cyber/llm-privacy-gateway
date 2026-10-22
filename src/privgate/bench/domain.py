"""Downstream utility proxy: classify a Nemotron document's ``domain`` (30 classes).

Model: TF-IDF over word 1-2 grams (``max_features`` 200 000) + multinomial logistic
regression. ``C`` is chosen by 5-fold cross-validation on the TRAINING documents only, with
folds that keep both renderings of a persona together (StratifiedGroupKFold on
``meta["group_id"]``, shuffled with seed 0), by mean accuracy (ties: smaller C).

Training documents must come from the Nemotron train split and never from an evaluation
set: :func:`train_classifier` calls the Milestone 4 guard
(privgate.ml.training.check_training_pool) and raises otherwise. Evaluation is on
nemotron_dev_2k only: the Nemotron test file shares almost no domain with the train file
(docs/DATA.md), so a domain classifier cannot be evaluated there.

Utility drop = accuracy(original) - accuracy(masked) on the same documents, with a paired
document bootstrap interval (privgate.bench.bootstrap).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from privgate.bench.bootstrap import bootstrap_cis
from privgate.data.documents import Document
from privgate.ml.training import check_training_pool

C_GRID = (1.0, 10.0, 100.0)
FOLDS = 5
SEED = 0
MAX_FEATURES = 200_000
MAX_ITER = 1000


def make_pipeline(C: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)),
        ("clf", LogisticRegression(C=C, max_iter=MAX_ITER, random_state=SEED)),
    ])


def labels_and_groups(docs: Sequence[Document]) -> tuple[list[str], list[str]]:
    return [d.meta["domain"] for d in docs], [d.meta.get("group_id", d.doc_id) for d in docs]


def cross_validate(texts: Sequence[str], labels: Sequence[str], groups: Sequence[str],
                   c_grid: Sequence[float] = C_GRID, folds: int = FOLDS) -> dict:
    """Mean CV accuracy per C on the training texts; the selected C."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(splitter.split(texts, labels, groups))
    rows = []
    for c in c_grid:
        t0 = time.perf_counter()
        accs = []
        for train_idx, test_idx in splits:
            model = make_pipeline(c).fit([texts[i] for i in train_idx],
                                         [labels[i] for i in train_idx])
            pred = model.predict([texts[i] for i in test_idx])
            accs.append(sum(p == labels[i] for p, i in zip(pred, test_idx)) / len(test_idx))
        rows.append({"C": c, "fold_accuracy": accs, "mean_accuracy": sum(accs) / len(accs),
                     "seconds": round(time.perf_counter() - t0, 3)})
    best = max(rows, key=lambda r: (r["mean_accuracy"], -r["C"]))
    return {"folds": folds, "rows": rows, "selected_C": best["C"]}


def train_classifier(docs: Sequence[Document], texts: Sequence[str], forbidden: frozenset[str],
                     c_grid: Sequence[float] = C_GRID, folds: int = FOLDS):
    """Guarded training: CV on ``texts`` (the documents' training view), then a final fit."""
    check_training_pool(docs, forbidden)
    if len(texts) != len(docs):
        raise ValueError("one training text per document is required")
    labels, groups = labels_and_groups(docs)
    cv = cross_validate(texts, labels, groups, c_grid, folds)
    t0 = time.perf_counter()
    model = make_pipeline(cv["selected_C"]).fit(list(texts), labels)
    cv["final_fit_seconds"] = round(time.perf_counter() - t0, 3)
    return model, cv


def accuracy_and_macro_f1(gold: Sequence[str], pred: Sequence[str]) -> dict:
    from sklearn.metrics import f1_score

    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    return {"accuracy": acc, "macro_f1": float(f1_score(gold, pred, average="macro",
                                                         zero_division=0))}


def accuracy_ci(gold: Sequence[str], pred: Sequence[str], n_resamples: int = 1000,
                seed: int = SEED) -> list[float]:
    """95% document bootstrap interval of the accuracy."""
    cols = {"c": [int(g == p) for g, p in zip(gold, pred)], "n": [1] * len(gold)}
    return list(bootstrap_cis(cols, {"a": lambda s: s["c"] / s["n"]}, n_resamples, seed)["a"])


def paired_accuracy_difference(gold: Sequence[str], pred_a: Sequence[str],
                               pred_b: Sequence[str], n_resamples: int = 1000,
                               seed: int = SEED) -> dict:
    """accuracy(A) - accuracy(B) on the same documents, with a paired bootstrap CI."""
    if not len(gold) == len(pred_a) == len(pred_b):
        raise ValueError("one prediction per document for both systems")
    cols = {"a": [int(g == p) for g, p in zip(gold, pred_a)],
            "b": [int(g == p) for g, p in zip(gold, pred_b)], "n": [1] * len(gold)}
    ci = bootstrap_cis(cols, {"d": lambda s: (s["a"] - s["b"]) / s["n"]}, n_resamples,
                       seed)["d"]
    a, b = sum(cols["a"]) / len(gold), sum(cols["b"]) / len(gold)
    return {"a": a, "b": b, "difference": a - b, "ci95": list(ci),
            "ci_excludes_zero": ci[0] > 0 or ci[1] < 0}
