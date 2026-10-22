"""Milestone 7, RQ11: does a SMALL amount of target-domain data fix the Gretel leakage?

Usage:
    python scripts/run_domain_adaptation.py prepare     # Gretel train: overlap check, seeded samples
    python scripts/run_domain_adaptation.py train       # the three new CRFs (budget ~60 min)
    python scripts/run_domain_adaptation.py evaluate    # score every model on all three sets

Models (all CRFs with the M4 hyper-parameters c1 = 0.1, c2 = 0.01 and the same features; no
tuning of any kind in this RQ, the protocol was fixed before any result existed):

    crf            (a) Nemotron 5000, the existing M4 model (models/m4/crf, reused, not retrained)
    crf_n5k_g100   (b) the same 5000 Nemotron documents + 100 Gretel-train documents
    crf_n5k_g500   (b) the same 5000 Nemotron documents + 500 Gretel-train documents
    crf_g500       (c) 500 Gretel-train documents only

The Gretel documents are a seeded sample (``random.Random(0).sample``) of the train file after
dropping any document whose text occurs in the Gretel TEST file; the 100-document sample is the
first 100 of the 500-document one (nested). ``train`` skips the k = 500 mixed model if the
models already trained leave too little of the budget for it (rule: elapsed + 1.15 x the time
of the k = 100 model must be <= the budget). ``evaluate`` uses the M3 benchmark
(privgate.bench): bootstrap CIs and a paired bootstrap of every new model against (a) on every
set. Outputs: results/m7/ (committed); data/processed/train/ and models/m7/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from privgate.bench.cli import _load, _score, cmd_compare
from privgate.bench.runner import timed_detect, write_json
from privgate.bench.sets import SET_NAMES
from privgate.data.documents import LoadOptions, LoadTally
from privgate.data.gretel import DATASET as GRETEL
from privgate.data.gretel import TRAIN_ID_PREFIX, iter_gretel
from privgate.data.splits import GRETEL_TRAIN_FILE, file_sha256, read_jsonl, write_jsonl
from privgate.ml.jobs import crf_job
from privgate.ml.training import (
    check_mixed_training_pool,
    evaluation_doc_ids,
    gretel_test_text_hashes,
    split_gretel_overlap,
)

DATA = Path("data")
OUT = Path("results/m7")
MODELS = Path("models/m7")
SUBSETS = DATA / "processed" / "train"
M4_OUT = Path("results/m4")
SEED = 0
K_VALUES = (100, 500)
C1, C2 = 0.1, 0.01                      # the M4 selection (results/m4/crf_grid.json)
BUDGET_MINUTES = 60
BASELINE = "crf"
NEW_MODELS = ("crf_g500", "crf_n5k_g100", "crf_n5k_g500")
LABELS = {
    "crf": "(a) Nemotron 5000", "crf_n5k_g100": "(b) Nemotron 5000 + Gretel 100",
    "crf_n5k_g500": "(b) Nemotron 5000 + Gretel 500", "crf_g500": "(c) Gretel 500 only",
}


def _record() -> dict:
    return json.loads((OUT / "training_sets.json").read_text(encoding="utf-8"))


def cmd_prepare() -> None:
    t0 = time.perf_counter()
    tally = LoadTally(GRETEL, LoadOptions())
    docs = list(iter_gretel(DATA / GRETEL_TRAIN_FILE, tally, prefix=TRAIN_ID_PREFIX))
    hashes = gretel_test_text_hashes(DATA)
    kept, overlapping = split_gretel_overlap(docs, hashes)
    ids = sorted(d.doc_id for d in kept)
    drawn = random.Random(SEED).sample(ids, max(K_VALUES))
    by_id = {d.doc_id: d for d in kept}
    forbidden, record = evaluation_doc_ids(DATA), {}
    m4 = json.loads((M4_OUT / "training_sets.json").read_text(encoding="utf-8"))
    nem = SUBSETS / "n5000_s0.jsonl"
    if file_sha256(nem) != m4["subsets"]["n5000_s0"]["sha256"]:
        raise SystemExit(f"{nem} differs from results/m4/training_sets.json")
    for k in K_VALUES:
        picked = sorted((by_id[i] for i in drawn[:k]), key=lambda d: d.doc_id)
        check_mixed_training_pool(picked, forbidden, hashes)
        path = SUBSETS / f"m7_gretel_k{k}_s{SEED}.jsonl"
        record[f"gretel_k{k}"] = {"k": k, "documents": len(picked), "path": path.as_posix(),
                                  "sha256": write_jsonl(picked, path)}
        # (b): the exact M4 n = 5000 Nemotron subset plus the Gretel sample
        mixed = read_jsonl(nem) + picked
        check_mixed_training_pool(mixed, forbidden, hashes)
        path = SUBSETS / f"m7_n5000_gretel_k{k}_s{SEED}.jsonl"
        record[f"n5000_gretel_k{k}"] = {"k": k, "documents": len(mixed), "path": path.as_posix(),
                                        "sha256": write_jsonl(mixed, path)}
    write_json(OUT / "training_sets.json", {
        "gretel_train_file": GRETEL_TRAIN_FILE.as_posix(),
        "gretel_train_sha256": file_sha256(DATA / GRETEL_TRAIN_FILE),
        "gretel_train_documents": len(docs),
        "gretel_test_hashes": len(hashes),
        "overlap_with_gretel_test": len(overlapping),
        "overlap_action": "dropped before sampling" if overlapping else "none needed",
        "usable_documents": len(kept), "seed": SEED, "sets": record,
        "load_report": tally.freeze().to_dict(), "seconds": round(time.perf_counter() - t0, 2)})
    print(f"gretel train: {len(docs)} documents, overlap with test: {len(overlapping)}, "
          f"usable: {len(kept)}")


def _spec(name: str) -> dict:
    key = {"crf_g500": "gretel_k500", "crf_n5k_g100": "n5000_gretel_k100",
           "crf_n5k_g500": "n5000_gretel_k500"}[name]
    entry = _record()["sets"][key]
    return {"n": entry["documents"], "seed": SEED, "train_path": entry["path"],
            "train_sha256": entry["sha256"], "data_root": str(DATA), "eval_sets": [],
            "model_dir": str(MODELS / name), "c1": C1, "c2": C2, "pool": "mixed"}


def cmd_train(budget_minutes: float) -> None:
    log_path = OUT / "training_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"models": {}}
    for name in NEW_MODELS:
        if name in log["models"]:
            print(f"{name}: already trained")
            continue
        elapsed = sum(m["wall_seconds"] for m in log["models"].values()) / 60
        if name == "crf_n5k_g500" and "crf_n5k_g100" in log["models"]:
            needed = 1.15 * log["models"]["crf_n5k_g100"]["wall_seconds"] / 60
            if elapsed + needed > budget_minutes:
                reason = (f"elapsed {elapsed:.1f} min + estimated {needed:.1f} min exceeds the "
                          f"{budget_minutes:g} min budget")
                log["skipped"] = {name: reason}
                write_json(log_path, log)
                print(f"{name}: skipped, {reason}")
                continue
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=1) as pool:     # one process per model: memory
            result = pool.submit(crf_job, _spec(name)).result()
        wall = time.perf_counter() - t0
        log["models"][name] = {
            "wall_seconds": round(wall, 1), "training_seconds": result["training_seconds"],
            "documents": result["n"], "peak_memory_mb": result["peak_memory_mb"],
            "training_set_sha256": result["training_set_sha256"], "c1": C1, "c2": C2}
        log["total_wall_minutes"] = round(
            sum(m["wall_seconds"] for m in log["models"].values()) / 60, 1)
        write_json(log_path, log)
        print(f"{name}: trained on {result['n']} documents in {wall / 60:.1f} min", flush=True)


def cmd_evaluate() -> None:
    from privgate.ml.crf import CRFDetector

    trained = json.loads((OUT / "training_log.json").read_text(encoding="utf-8"))["models"]
    for name in (n for n in NEW_MODELS if n in trained):
        det = CRFDetector.load(MODELS / name)
        for set_name in SET_NAMES:
            docs, sha, raw = _load(DATA, set_name)
            t0 = time.perf_counter()
            preds, timings = timed_detect(det, docs)
            _score(OUT, set_name, name, [name], {"model_sidecar": det.meta},
                   det.supported_labels, docs, sha, raw, preds, timings,
                   time.perf_counter() - t0)
            cmd_compare(DATA, OUT, M4_OUT, set_name, name, BASELINE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--budget-minutes", type=float, default=BUDGET_MINUTES)
    args = parser.parse_args(argv)
    {"prepare": cmd_prepare, "evaluate": cmd_evaluate,
     "train": lambda: cmd_train(args.budget_minutes)}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
