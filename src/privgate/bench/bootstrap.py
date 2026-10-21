"""Percentile bootstrap over documents, standard library only.

Documents, not spans, are resampled: spans in one document are correlated (same template,
same persona), so treating them as independent would give intervals that are too narrow.
Each resample draws ``n`` documents with replacement; a statistic is recomputed from the
summed per-document counts (micro averaging, as in the point estimate). The interval is
the 2.5th and 97.5th percentile of the resampled values (linear interpolation).

The same seed gives the same document draws for every detector on a set of the same size,
so intervals of different detectors are paired (computed on identical resamples).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

N_RESAMPLES = 1000
SEED = 0
LEVEL = 0.95


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """``q``-quantile (0..1) of already sorted values, linear interpolation between ranks."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_cis(
    columns: Mapping[str, Sequence[int]],
    statistics: Mapping[str, Callable[[Mapping[str, int]], float]],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> dict[str, tuple[float, float]]:
    """Confidence interval of each statistic.

    ``columns`` holds one per-document count per document (all columns equally long);
    each statistic maps the column sums of a resample to a number.
    """
    lengths = {len(c) for c in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all columns must have the same length")
    n = lengths.pop()
    if n == 0:
        raise ValueError("bootstrap needs at least one document")
    rng = random.Random(seed)
    population = range(n)
    values: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(n_resamples):
        idx = rng.choices(population, k=n)
        sums = {name: sum(map(col.__getitem__, idx)) for name, col in columns.items()}
        for name, fn in statistics.items():
            values[name].append(fn(sums))
    alpha = (1.0 - level) / 2
    out = {}
    for name, vals in values.items():
        vals.sort()
        out[name] = (percentile(vals, alpha), percentile(vals, 1.0 - alpha))
    return out


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Micro F1 = 2TP / (2TP + FP + FN); 0.0 when there is nothing to score."""
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def leakage_from_counts(gold_chars: int, covered: int) -> float:
    return 1.0 - covered / gold_chars if gold_chars else 0.0
