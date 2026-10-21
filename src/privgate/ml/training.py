"""Milestone 4 training protocol: pool, subsets, guard, hyper-parameter grids, selection.

Everything here was fixed before any model was evaluated on a test set.

* **Pool**: Nemotron TRAIN split only (documents of the train file whose group hashes to
  ``train``, privgate.data.splits). Never dev, never the Nemotron test file, never Gretel.
  :func:`check_training_pool` enforces it and raises on any evaluation document.
* **Subsets**: groups (personas; both renderings of a persona stay together) are sorted and
  drawn with ``random.Random(seed).sample``; a subset of ``n`` documents holds ``n // 2``
  groups. Sizes :data:`SIZES`, seeds :data:`SEEDS`.
* **Selection on nemotron_dev_2k only**: CRF grid :data:`CRF_GRID` and logistic-regression
  grid :data:`LOGREG_C`, both trained on the ``n = GRID_N`` subset with seed 0; choose the
  highest type-agnostic micro F1, ties -> lower leakage, then the smaller regularisation
  values. The CRF size used in the benchmark is the size with the highest mean dev F1 over
  seeds (its seed-0 model).

Deviations from the original plan, decided from measured cost on this laptop (documented
in docs/results-detection.md): n = 20000 is not trained (one model needs several GB of
memory and about an hour), and the grids use ``GRID_N = 1000`` instead of 5000.
"""

from __future__ import annotations

import ctypes
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from privgate.bench.sets import SET_NAMES, load_evaluation_set
from privgate.data.documents import Document
from privgate.data.nemotron import DATASET as NEMOTRON
from privgate.data.splits import assign_split, iter_file, split_key
from privgate.evaluation import evaluate_corpus

SIZES = (1000, 5000)
SEEDS = (0, 1, 2)
GRID_N = 1000
GRID_SEED = 0
CRF_GRID = tuple((c1, c2) for c1 in (0.05, 0.1, 0.3) for c2 in (0.01, 0.05, 0.1))
CRF_MAX_ITERATIONS = 100
LOGREG_C = (0.1, 1.0, 10.0)
LOGREG_MAX_ITER = 1000
LOGREG_N = 5000


class TrainingPoolError(ValueError):
    """A document outside the Nemotron train split was offered for training."""


def evaluation_doc_ids(data_root: Path) -> frozenset[str]:
    ids: set[str] = set()
    for name in SET_NAMES:
        docs, _ = load_evaluation_set(name, data_root)
        ids.update(d.doc_id for d in docs)
    return frozenset(ids)


def check_training_pool(docs: Iterable[Document], forbidden: frozenset[str]) -> None:
    """Raise unless every document is a Nemotron train-split document not used in evaluation."""
    for doc in docs:
        if doc.doc_id in forbidden:
            raise TrainingPoolError(f"{doc.doc_id} belongs to an evaluation set")
        if doc.source != NEMOTRON or assign_split(split_key(doc)) != "train":
            raise TrainingPoolError(f"{doc.doc_id} is not in the Nemotron train split")


def sample_groups(groups: Iterable[str], n_docs: int, seed: int) -> list[str]:
    ordered = sorted(set(groups))
    k = n_docs // 2
    if k > len(ordered):
        raise ValueError(f"cannot draw {k} groups from {len(ordered)}")
    return random.Random(seed).sample(ordered, k)


def load_train_split(data_root: Path, wanted_groups: frozenset[str]) -> dict[str, list[Document]]:
    """Train-split documents of the wanted groups, streamed from the raw train file."""
    pairs, _ = iter_file(data_root, "nemotron_train_file")
    out: dict[str, list[Document]] = {}
    for split, doc in pairs:
        if split == "train" and split_key(doc) in wanted_groups:
            out.setdefault(split_key(doc), []).append(doc)
    return out


def subset(docs_by_group: dict[str, list[Document]], groups: Sequence[str]) -> list[Document]:
    return sorted((d for g in groups for d in docs_by_group[g]), key=lambda d: d.doc_id)


def scores(docs: Sequence[Document], preds: Sequence[Sequence]) -> dict:
    """The selection metrics (all gold spans): type-agnostic micro F1, leakage, strict F1."""
    rep = evaluate_corpus(zip([list(d.spans) for d in docs], preds))
    ta = rep.type_agnostic.micro
    return {"type_agnostic_f1": ta.f1, "type_agnostic_precision": ta.precision,
            "type_agnostic_recall": ta.recall, "strict_f1": rep.strict.micro.f1,
            "partial_f1": rep.partial.micro.f1, "leakage_rate": rep.chars.leakage_rate,
            "char_precision": rep.chars.char_precision}


def select(rows: Sequence[dict], params: Sequence[str]) -> dict:
    """Highest dev type-agnostic F1; ties -> lower leakage, then smaller parameter values."""
    if not rows:
        raise ValueError("empty grid")
    return max(rows, key=lambda r: (r["dev"]["type_agnostic_f1"], -r["dev"]["leakage_rate"],
                                    *(-r[p] for p in params)))


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (None if unavailable)."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                                    counters.cb):
            return counters.PeakWorkingSetSize / 2**20
        return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, OSError):  # pragma: no cover
        return None
