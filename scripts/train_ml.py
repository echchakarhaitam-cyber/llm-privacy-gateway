"""Milestone 4 training protocol (privgate.ml.training), step by step.

Usage:
    python scripts/train_ml.py subsets            # sample + guard + save training subsets
    python scripts/train_ml.py grid [--workers 3] # CRF and LogReg selection on dev only
    python scripts/train_ml.py curve [--workers 3]# selected CRF for every size and seed
    python scripts/train_ml.py final              # pick the CRF size on dev; train LogReg

Training subsets go to data/processed/train/ (gitignored), models to models/m4/
(gitignored); grids, learning curve, subset hashes and wall times to results/m4/
(committed). No step reads a test-set score to make a choice: grids and the size choice use
nemotron_dev_2k only; test sets are scored for the learning curve, which chooses nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from privgate.bench.runner import write_json
from privgate.bench.sets import DEV_SET, SET_NAMES
from privgate.data.splits import RAW_FILES, assign_split, write_jsonl
from privgate.ml.jobs import crf_job, logreg_job
from privgate.ml.training import (
    CRF_GRID,
    GRID_N,
    GRID_SEED,
    LOGREG_C,
    LOGREG_N,
    SEEDS,
    SIZES,
    check_training_pool,
    evaluation_doc_ids,
    load_train_split,
    sample_groups,
    select,
    subset,
)

DATA = Path("data")
OUT = Path("results/m4")
MODELS = Path("models/m4")
SUBSETS = DATA / "processed" / "train"


def _log(step: str, seconds: float, **extra) -> None:
    path = OUT / "training_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"steps": []}
    log["steps"] = [s for s in log["steps"] if s["step"] != step]
    log["steps"].append({"step": step, "seconds": round(seconds, 3), **extra})
    log["total_seconds"] = round(sum(s["seconds"] for s in log["steps"]), 3)
    write_json(path, log)


def _subsets() -> dict:
    return json.loads((OUT / "training_sets.json").read_text(encoding="utf-8"))


def cmd_subsets() -> None:
    t0 = time.perf_counter()
    uids = pq.read_table(DATA / RAW_FILES["nemotron_train_file"], columns=["uid"])
    pool = sorted({u for u in uids.column(0).to_pylist() if assign_split(u) == "train"})
    draws = {(n, s): sample_groups(pool, n, s) for n in SIZES for s in SEEDS}
    wanted = frozenset(g for groups in draws.values() for g in groups)
    by_group = load_train_split(DATA, wanted)
    forbidden = evaluation_doc_ids(DATA)
    record = {"pool_groups": len(pool), "subsets": {}}
    for (n, s), groups in sorted(draws.items()):
        docs = subset(by_group, groups)
        check_training_pool(docs, forbidden)
        path = SUBSETS / f"n{n}_s{s}.jsonl"
        record["subsets"][f"n{n}_s{s}"] = {"n": n, "seed": s, "documents": len(docs),
                                           "groups": len(groups), "path": path.as_posix(),
                                           "sha256": write_jsonl(docs, path)}
        print(f"subset n={n} seed={s}: {len(docs)} documents")
    write_json(OUT / "training_sets.json", record)
    _log("subsets", time.perf_counter() - t0)


def _spec(n: int, seed: int, eval_sets, model_dir=None, **params) -> dict:
    entry = _subsets()["subsets"][f"n{n}_s{seed}"]
    return {"n": n, "seed": seed, "train_path": entry["path"], "train_sha256": entry["sha256"],
            "data_root": str(DATA), "eval_sets": list(eval_sets),
            "model_dir": str(model_dir) if model_dir else None, **params}


def _run(job, specs, workers: int) -> list[dict]:
    if workers <= 1:
        return [job(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(job, specs))


def _rows(results: list[dict]) -> list[dict]:
    for r in results:
        r["dev"] = r["eval"][DEV_SET]
    return results


def cmd_grid(workers: int) -> None:
    t0 = time.perf_counter()
    specs = [_spec(GRID_N, GRID_SEED, [DEV_SET], c1=c1, c2=c2) for c1, c2 in CRF_GRID]
    rows = _rows(_run(crf_job, specs, workers))
    best = select(rows, ["c1", "c2"])
    write_json(OUT / "crf_grid.json", {
        "set": DEV_SET, "n": GRID_N, "seed": GRID_SEED, "rows": rows,
        "rule": "highest dev type-agnostic micro F1; ties -> lower leakage, smaller c1, c2",
        "selected": {"c1": best["c1"], "c2": best["c2"]}})
    _log("crf_grid", time.perf_counter() - t0, workers=workers)
    print(f"selected CRF c1={best['c1']} c2={best['c2']}")

    t0 = time.perf_counter()
    specs = [_spec(GRID_N, GRID_SEED, [DEV_SET], C=c) for c in LOGREG_C]
    rows = _rows(_run(logreg_job, specs, workers))
    best = select(rows, ["C"])
    write_json(OUT / "logreg_grid.json", {
        "set": DEV_SET, "n": GRID_N, "seed": GRID_SEED, "rows": rows,
        "rule": "highest dev type-agnostic micro F1; ties -> lower leakage, smaller C",
        "selected": {"C": best["C"]}})
    _log("logreg_grid", time.perf_counter() - t0, workers=workers)
    print(f"selected LogReg C={best['C']}")


def _aggregate(models: list[dict]) -> list[dict]:
    rows = []
    for n in sorted({m["n"] for m in models}):
        for name in SET_NAMES:
            vals = [m["eval"][name] for m in models if m["n"] == n]
            row = {"n": n, "set": name, "seeds": len(vals)}
            for metric in ("type_agnostic_f1", "leakage_rate", "strict_f1"):
                xs = [v[metric] for v in vals]
                row[metric] = {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}
            rows.append(row)
    return rows


def cmd_curve(workers: int) -> None:
    grid = json.loads((OUT / "crf_grid.json").read_text(encoding="utf-8"))
    c1, c2 = grid["selected"]["c1"], grid["selected"]["c2"]
    models: list[dict] = []
    for n in SIZES:
        t0 = time.perf_counter()
        specs = [_spec(n, s, SET_NAMES, MODELS / f"crf_n{n}_s{s}", c1=c1, c2=c2) for s in SEEDS]
        # n=5000 runs one at a time: a single model needs about all the free memory
        models += _run(crf_job, specs, workers if n == min(SIZES) else 1)
        _log(f"curve_n{n}", time.perf_counter() - t0,
             workers=workers if n == min(SIZES) else 1)
        write_json(OUT / "learning_curve.json", {
            "c1": c1, "c2": c2, "models": models, "rows": _aggregate(models)})
        print(f"curve n={n} done")
    # determinism check: the grid's selected run and the curve's (GRID_N, GRID_SEED) run
    again = next(m for m in models if m["n"] == GRID_N and m["seed"] == GRID_SEED)
    chosen = next(r for r in grid["rows"] if (r["c1"], r["c2"]) == (c1, c2))
    if again["eval"][DEV_SET] != chosen["dev"]:
        raise RuntimeError("retraining the selected grid model gave different dev scores")


def cmd_final() -> None:
    t0 = time.perf_counter()
    curve = json.loads((OUT / "learning_curve.json").read_text(encoding="utf-8"))
    dev_rows = [r for r in curve["rows"] if r["set"] == DEV_SET]
    best = max(dev_rows, key=lambda r: (r["type_agnostic_f1"]["mean"], r["n"]))
    src = MODELS / f"crf_n{best['n']}_s0"
    if (MODELS / "crf").exists():
        shutil.rmtree(MODELS / "crf")
    shutil.copytree(src, MODELS / "crf")
    c = json.loads((OUT / "logreg_grid.json").read_text(encoding="utf-8"))["selected"]["C"]
    result = logreg_job(_spec(LOGREG_N, 0, [DEV_SET], MODELS / "logreg", C=c))
    write_json(OUT / "final_models.json", {
        "crf": {"n": best["n"], "seed": 0, "source": src.as_posix(), "c1": curve["c1"],
                "c2": curve["c2"], "rule": "size with the highest mean dev type-agnostic F1"},
        "logreg": {"n": LOGREG_N, "seed": 0, "C": c,
                   "training_seconds": result["training_seconds"], "n_iter": result["n_iter"],
                   "peak_memory_mb": result["peak_memory_mb"], "dev": result["eval"][DEV_SET]},
    })
    _log("final", time.perf_counter() - t0)
    print(f"crf size {best['n']}; logreg C={c} trained on n={LOGREG_N}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("subsets", "grid", "curve", "final"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "subsets":
        cmd_subsets()
    elif args.command == "grid":
        cmd_grid(args.workers)
    elif args.command == "curve":
        cmd_curve(args.workers)
    else:
        cmd_final()
    return 0


if __name__ == "__main__":
    sys.exit(main())
