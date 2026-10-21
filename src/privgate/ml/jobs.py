"""One training job = train one model on one saved subset, then score it.

Module-level functions so that a process pool (spawn start method on Windows) can import
and run them. A job reads its training subset from the JSONL file written by
``scripts/train_ml.py subsets`` (checking its SHA-256), trains, scores the model on the
requested evaluation sets (hash-verified, privgate.bench.sets) and optionally saves it.
"""

from __future__ import annotations

import time
from pathlib import Path

from privgate.bench.sets import load_evaluation_set
from privgate.data.splits import file_sha256, read_jsonl
from privgate.ml.training import (
    CRF_MAX_ITERATIONS,
    LOGREG_MAX_ITER,
    check_training_pool,
    evaluation_doc_ids,
    peak_memory_mb,
    scores,
)


def _training_docs(spec: dict):
    path = Path(spec["train_path"])
    if file_sha256(path) != spec["train_sha256"]:
        raise RuntimeError(f"{path} differs from its recorded hash")
    docs = read_jsonl(path)
    check_training_pool(docs, evaluation_doc_ids(Path(spec["data_root"])))
    return docs


def _evaluate(detector, spec: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    out = {}
    for name in spec["eval_sets"]:
        docs, _ = load_evaluation_set(name, Path(spec["data_root"]))
        out[name] = scores(docs, [detector.detect(d.text) for d in docs])
    return out, round(time.perf_counter() - t0, 3)


def _finish(detector, spec: dict) -> dict:
    evaluation, eval_seconds = _evaluate(detector, spec)
    if spec.get("model_dir"):
        detector.save(Path(spec["model_dir"]))
    return {**spec, "training_seconds": detector.meta["training_seconds"],
            "training_set_sha256": detector.meta["training_set_sha256"],
            "eval": evaluation, "eval_seconds": eval_seconds,
            "peak_memory_mb": peak_memory_mb()}


def crf_job(spec: dict) -> dict:
    from privgate.ml.crf import CRFDetector

    docs = _training_docs(spec)
    det = CRFDetector.train(docs, spec["c1"], spec["c2"], CRF_MAX_ITERATIONS, spec["seed"])
    return _finish(det, spec)


def logreg_job(spec: dict) -> dict:
    from privgate.ml.logreg import TokenLogRegDetector

    docs = _training_docs(spec)
    det = TokenLogRegDetector.train(docs, spec["C"], LOGREG_MAX_ITER, spec["seed"])
    result = _finish(det, spec)
    result["n_iter"] = det.meta["hyperparameters"]["n_iter"]
    return result
