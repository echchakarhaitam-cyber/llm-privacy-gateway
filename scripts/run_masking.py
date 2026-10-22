"""Milestone 5: masking experiments (detector x strategy), deterministic, seed 0.

Usage:
    python scripts/run_masking.py classifier   # domain classifiers (train split only)
    python scripts/run_masking.py grid         # privacy + utility of every detector x strategy
    python scripts/run_masking.py figure       # docs/figures/m5_privacy_utility.png

Detectors: ``oracle`` (the gold spans), ``regex+presidio`` (results/m3), ``crf`` and
``crf+regex+presidio`` (results/m4), reused from their stored predictions after checking
that each run scored the same evaluation-set file (SHA-256). Nothing is retrained.

Outputs: results/m5/domain_classifier.json and results/m5/masking.json (committed);
masked texts in results/m5/masked/ and classifiers in models/m5/ (gitignored). Mappings
(placeholder -> original) are never written to disk: they are as sensitive as the text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from privgate.bench.domain import (
    accuracy_and_macro_f1,
    accuracy_ci,
    paired_accuracy_difference,
    train_classifier,
)
from privgate.bench.privacy import privacy_report
from privgate.bench.runner import PREDICTIONS, read_predictions, write_json
from privgate.bench.sets import DEV_SET, SET_NAMES, load_evaluation_set
from privgate.bench.utility import utility_report
from privgate.data.splits import RAW_FILES, assign_split, file_sha256, write_jsonl
from privgate.masking import STRATEGIES, Masker
from privgate.ml.training import evaluation_doc_ids, load_train_split, sample_groups, subset

DATA = Path("data")
OUT = Path("results/m5")
MODELS = Path("models/m5")
SEED = 0
N_TRAIN = 20_000
DETECTOR_RUNS = {"regex+presidio": Path("results/m3"), "crf": Path("results/m4"),
                 "crf+regex+presidio": Path("results/m4")}
DETECTORS = ("oracle", *DETECTOR_RUNS)
SETTINGS = {"original": "trained on original train text",
            "typed": "trained on oracle-typed-masked train text"}


def _spans(set_name: str, detector: str, docs, sha: str):
    if detector == "oracle":
        return [list(d.spans) for d in docs]
    run = DETECTOR_RUNS[detector] / set_name / detector
    meta = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if meta["set_sha256"] != sha:
        raise SystemExit(f"{run} was scored on another version of {set_name}")
    return read_predictions(run / PREDICTIONS, docs)[0]


def _save_model(model, name: str) -> str:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"domain_{name}.pkl"
    with path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return file_sha256(path)


def _load_model(name: str, sha: str):
    path = MODELS / f"domain_{name}.pkl"
    if file_sha256(path) != sha:
        raise SystemExit(f"{path} differs from its recorded hash")
    with path.open("rb") as fh:
        return pickle.load(fh)  # this project's own file, hash-checked above


def cmd_classifier() -> None:
    t0 = time.perf_counter()
    uids = pq.read_table(DATA / RAW_FILES["nemotron_train_file"], columns=["uid"])
    pool = sorted({u for u in uids.column(0).to_pylist() if assign_split(u) == "train"})
    groups = sample_groups(pool, N_TRAIN, SEED)
    docs = subset(load_train_split(DATA, frozenset(groups)), groups)
    forbidden = evaluation_doc_ids(DATA)
    path = DATA / "processed" / "train" / f"m5_domain_n{N_TRAIN}_s{SEED}.jsonl"
    record = {"n": len(docs), "seed": SEED, "groups": len(groups), "path": path.as_posix(),
              "sha256": write_jsonl(docs, path), "settings": {}}
    dev, dev_sha = load_evaluation_set(DEV_SET, DATA)
    gold = [d.meta["domain"] for d in dev]
    masker = Masker()
    for name, description in SETTINGS.items():
        t1 = time.perf_counter()
        texts = ([d.text for d in docs] if name == "original" else
                 [masker.mask(d.text, d.spans, "typed", SEED).text for d in docs])
        model, cv = train_classifier(docs, texts, forbidden)
        dev_texts = ([d.text for d in dev] if name == "original" else
                     [masker.mask(d.text, d.spans, "typed", SEED).text for d in dev])
        record["settings"][name] = {
            "description": description, "cv": cv, "model_sha256": _save_model(model, name),
            "dev_" + ("original" if name == "original" else "oracle_typed"):
                accuracy_and_macro_f1(gold, list(model.predict(dev_texts))),
            "seconds": round(time.perf_counter() - t1, 3)}
        print(f"{name}: C={cv['selected_C']} {record['settings'][name]}")
    record["dev_set_sha256"] = dev_sha
    record["seconds"] = round(time.perf_counter() - t0, 3)
    write_json(OUT / "domain_classifier.json", record)


def _save_masked(set_name, detector, strategy, docs, masked) -> str:
    path = OUT / "masked" / set_name / detector / f"{strategy}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for d, m in zip(docs, masked):
            line = json.dumps({"doc_id": d.doc_id, "text": m.text}, ensure_ascii=False) + "\n"
            out.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def cmd_grid() -> None:
    t0 = time.perf_counter()
    clf = json.loads((OUT / "domain_classifier.json").read_text(encoding="utf-8"))
    models = {name: _load_model(name, s["model_sha256"]) for name, s in clf["settings"].items()}
    masker = Masker()
    rows, dev_reference = [], {}
    for set_name in SET_NAMES:
        docs, sha = load_evaluation_set(set_name, DATA)
        gold_domain = [d.meta.get("domain", "") for d in docs]
        if set_name == DEV_SET:
            dev_reference["original_pred"] = list(models["original"].predict(
                [d.text for d in docs]))
        for detector in DETECTORS:
            spans = _spans(set_name, detector, docs, sha)
            for strategy in STRATEGIES:
                t1 = time.perf_counter()
                masked = [masker.mask(d.text, s, strategy, SEED) for d, s in zip(docs, spans)]
                row = {"set": set_name, "set_sha256": sha, "detector": detector,
                       "strategy": strategy,
                       "masked_sha256": _save_masked(set_name, detector, strategy, docs, masked),
                       "privacy": privacy_report(docs, masked),
                       "utility": utility_report(docs, masked)}
                if set_name == DEV_SET:
                    texts = [m.text for m in masked]
                    pred = list(models["original"].predict(texts))
                    row["domain_original_classifier"] = {
                        **accuracy_and_macro_f1(gold_domain, pred),
                        "accuracy_ci95": accuracy_ci(gold_domain, pred),
                        "drop_vs_unmasked": paired_accuracy_difference(
                            gold_domain, dev_reference["original_pred"], pred)}
                    row["_pred"] = pred
                    if strategy == "typed":
                        typed_pred = list(models["typed"].predict(texts))
                        row["domain_typed_classifier"] = {
                            **accuracy_and_macro_f1(gold_domain, typed_pred),
                            "accuracy_ci95": accuracy_ci(gold_domain, typed_pred),
                            "drop_vs_unmasked_original_classifier": paired_accuracy_difference(
                                gold_domain, dev_reference["original_pred"], typed_pred)}
                row["seconds"] = round(time.perf_counter() - t1, 3)
                rows.append(row)
                print(f"{set_name} {detector} {strategy}: leakage "
                      f"{row['privacy']['char_leakage']:.4f} verbatim "
                      f"{row['privacy']['verbatim_survival']:.4f}")
    # RQ8: paired accuracy gap to the oracle under the same strategy (dev, original classifier)
    dev_rows = {(r["detector"], r["strategy"]): r for r in rows if r["set"] == DEV_SET}
    gold_dev = [d.meta["domain"] for d in load_evaluation_set(DEV_SET, DATA)[0]]
    for (detector, strategy), r in dev_rows.items():
        if detector != "oracle":
            r["domain_original_classifier"]["gap_vs_oracle"] = paired_accuracy_difference(
                gold_dev, r["_pred"], dev_rows[("oracle", strategy)]["_pred"])
    for r in rows:
        r.pop("_pred", None)
    unmasked = {**accuracy_and_macro_f1(gold_dev, dev_reference["original_pred"]),
                "accuracy_ci95": accuracy_ci(gold_dev, dev_reference["original_pred"])}
    write_json(OUT / "masking.json", {
        "seed": SEED, "detectors": list(DETECTORS), "strategies": list(STRATEGIES),
        "dev_unmasked_original_classifier": unmasked, "rows": rows,
        "seconds": round(time.perf_counter() - t0, 3)})


def cmd_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUT / "masking.json").read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["set"] == DEV_SET]
    markers = {"oracle": "*", "regex+presidio": "o", "crf": "s", "crf+regex+presidio": "D"}
    colors = {"redact": "tab:red", "typed": "tab:blue", "pseudonym": "tab:green",
              "partial": "tab:orange"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        oracle = r["detector"] == "oracle"
        ax.scatter(r["privacy"]["char_leakage"], r["domain_original_classifier"]["accuracy"],
                   marker=markers[r["detector"]], color=colors[r["strategy"]],
                   s=220 if oracle else 70, edgecolors="black" if oracle else "none",
                   linewidths=1.2, zorder=3)
    unmasked = data["dev_unmasked_original_classifier"]["accuracy"]
    ax.scatter(1.0, unmasked, marker="X", color="black", s=120, zorder=3)
    ax.annotate("unmasked original", (1.0, unmasked), textcoords="offset points",
                xytext=(-110, 6))
    for d, m in markers.items():
        ax.scatter([], [], marker=m, color="grey", label=d)
    for s, c in colors.items():
        ax.scatter([], [], marker="o", color=c, label=s)
    ax.set_xlabel("character leakage on nemotron_dev_2k (lower is better)")
    ax.set_ylabel("domain accuracy on nemotron_dev_2k (higher is better)")
    ax.set_title("Privacy-utility trade-off (oracle = gold spans, outlined stars)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    out = Path("docs/figures/m5_privacy_utility.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("classifier", "grid", "figure"))
    args = parser.parse_args(argv)
    {"classifier": cmd_classifier, "grid": cmd_grid, "figure": cmd_figure}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
