"""Command line of the detection benchmark: ``python -m privgate.bench <command>``.

    run --set S --detector D      base detector D (regex, spacy, presidio) on set S
    sweep                         select Presidio's threshold on nemotron_dev_2k
    ensemble --set S --detector E E in {regex+spacy, regex+presidio}, from stored members
    errors --detector D           sampled FN/FP of D on nemotron_dev_2k
    all                           everything above, in the only valid order

Presidio on a test set refuses to run until ``sweep`` has selected a threshold on dev, so
a test-set result can never feed back into the choice. On dev, Presidio is analysed once
at threshold 0.0 (results kept for the sweep) and scored at the selected threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from privgate.bench.errors import collect_errors, render_errors, sample_errors
from privgate.bench.rawlabels import load_raw_labels
from privgate.bench.registry import ALL_DETECTORS, BASE_DETECTORS, ENSEMBLES, make_detector
from privgate.bench.runner import (
    METRICS,
    PREDICTIONS,
    RAW_RESULTS,
    compose_ensemble,
    log_step,
    read_predictions,
    read_raw_results,
    score_run,
    timed_detect,
    timed_presidio_raw,
    write_json,
    write_predictions,
    write_raw_results,
)
from privgate.bench.sets import DEV_SET, SET_NAMES, load_evaluation_set
from privgate.bench.sweep import (
    LEAKAGE_TOLERANCE,
    SELECTION_RULE,
    THRESHOLDS,
    select_threshold,
    sweep_rows,
)
from privgate.spans import EntityType

SWEEP_FILE = "presidio_sweep.json"


def selected_threshold(out_root: Path) -> float | None:
    path = out_root / SWEEP_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["selected_threshold"]


def _load(data_root: Path, set_name: str):
    docs, sha = load_evaluation_set(set_name, data_root)
    return docs, sha, load_raw_labels(data_root, set_name, docs, sha)


def _score(out_root, set_name, detector, members, config, supported, docs, sha, raw, preds,
           timings, runtime):
    out_dir = out_root / set_name / detector
    write_predictions(out_dir / PREDICTIONS, docs, preds, timings)
    t0 = time.perf_counter()
    payload = score_run(out_dir, detector=detector, members=members, config=config,
                        supported_labels=supported, set_name=set_name, set_sha256=sha,
                        docs=docs, preds=preds, timings=timings, raw_labels=raw,
                        runtime_seconds=round(runtime, 3))
    log_step(out_root, "score", set_name, detector, time.perf_counter() - t0)
    ta = payload["views"]["all"]["type_agnostic"]["micro"]["f1"]
    leak = payload["views"]["all"]["chars"]["leakage_rate"]
    print(f"{set_name} / {detector}: type-agnostic F1 {ta:.4f}, leakage {leak:.4f}")
    return payload


def cmd_run(data_root: Path, out_root: Path, set_name: str, detector_name: str) -> None:
    if detector_name not in BASE_DETECTORS:
        raise SystemExit(f"run takes a base detector {BASE_DETECTORS}; use 'ensemble' for "
                         f"{tuple(ENSEMBLES)}")
    docs, sha, raw = _load(data_root, set_name)
    t0 = time.perf_counter()
    chosen = selected_threshold(out_root)
    if detector_name == "presidio" and set_name == DEV_SET:
        det = make_detector("presidio", 0.0)
        raws, _, timings = timed_presidio_raw(det, docs)
        write_raw_results(out_root / set_name / "presidio" / RAW_RESULTS, docs, raws, timings)
        threshold = 0.0 if chosen is None else chosen
        preds = [det.spans_from_results(r, threshold) for r in raws]
        config = {"score_threshold": threshold, "latency_measured_at_threshold": 0.0,
                  "threshold_source": SWEEP_FILE if chosen is not None else "not selected yet"}
    elif detector_name == "presidio":
        if chosen is None:
            raise SystemExit("select Presidio's threshold on dev first: "
                             "python -m privgate.bench sweep")
        det = make_detector("presidio", chosen)
        preds, timings = timed_detect(det, docs)
        config = {"score_threshold": chosen, "threshold_source": f"{SWEEP_FILE} ({DEV_SET})"}
    else:
        det = make_detector(detector_name)
        preds, timings = timed_detect(det, docs)
        config = {}
    if hasattr(det, "model"):
        config["model"] = det.model
    elapsed = time.perf_counter() - t0
    log_step(out_root, "detect", set_name, detector_name, elapsed)
    _score(out_root, set_name, detector_name, [detector_name], config, det.supported_labels,
           docs, sha, raw, preds, timings, elapsed)


def cmd_sweep(data_root: Path, out_root: Path) -> float:
    t0 = time.perf_counter()
    docs, sha, raw = _load(data_root, DEV_SET)
    det = make_detector("presidio", 0.0)
    raws, timings = read_raw_results(out_root / DEV_SET / "presidio" / RAW_RESULTS, docs)
    rows = sweep_rows(det, raws, docs)
    chosen = select_threshold(rows)
    floor = min(r["leakage_rate"] for r in rows)
    write_json(out_root / SWEEP_FILE, {
        "set": DEV_SET, "set_sha256": sha, "thresholds": list(THRESHOLDS),
        "rule": SELECTION_RULE, "leakage_tolerance": LEAKAGE_TOLERANCE,
        "minimum_leakage_rate": floor,
        "eligible_thresholds": [r["threshold"] for r in rows
                                if r["leakage_rate"] <= floor + LEAKAGE_TOLERANCE + 1e-12],
        "rows": rows, "selected_threshold": chosen,
    })
    print(f"selected Presidio threshold on {DEV_SET}: {chosen}")
    preds = [det.spans_from_results(r, chosen) for r in raws]
    config = {"score_threshold": chosen, "latency_measured_at_threshold": 0.0,
              "threshold_source": SWEEP_FILE, "model": det.model}
    elapsed = time.perf_counter() - t0
    log_step(out_root, "sweep", DEV_SET, "presidio", elapsed)
    _score(out_root, DEV_SET, "presidio", ["presidio"], config, det.supported_labels, docs, sha,
           raw, preds, timings, elapsed)
    return chosen


def cmd_ensemble(data_root: Path, out_root: Path, set_name: str, name: str) -> None:
    if name not in ENSEMBLES:
        raise SystemExit(f"unknown ensemble {name!r}; expected one of {tuple(ENSEMBLES)}")
    docs, sha, raw = _load(data_root, set_name)
    t0 = time.perf_counter()
    member_preds, member_timings, configs, supported = [], [], {}, set()
    for m in ENSEMBLES[name]:
        preds, timings = read_predictions(out_root / set_name / m / PREDICTIONS, docs)
        meta = json.loads((out_root / set_name / m / METRICS).read_text(encoding="utf-8"))
        member_preds.append(preds)
        member_timings.append(timings)
        configs[m] = meta["config"]
        supported |= {EntityType(lab) for lab in meta["supported_labels"]}
    preds, timings = compose_ensemble(member_preds, member_timings)
    elapsed = time.perf_counter() - t0
    log_step(out_root, "ensemble", set_name, name, elapsed)
    config = {"members": configs,
              "latency": "sum of member latencies plus measured combination time"}
    _score(out_root, set_name, name, list(ENSEMBLES[name]), config, frozenset(supported), docs,
           sha, raw, preds, timings, elapsed)


def cmd_errors(data_root: Path, out_root: Path, detector: str) -> None:
    t0 = time.perf_counter()
    docs, _, raw = _load(data_root, DEV_SET)
    preds, _ = read_predictions(out_root / DEV_SET / detector / PREDICTIONS, docs)
    fns, fps = collect_errors(docs, preds, raw)
    text = render_errors(detector, DEV_SET, docs, sample_errors(fns), sample_errors(fps),
                         (len(fns), len(fps)))
    path = out_root / f"errors_{detector}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(text)
    log_step(out_root, "errors", DEV_SET, detector, time.perf_counter() - t0)
    print(f"wrote {path}")


def cmd_all(data_root: Path, out_root: Path) -> None:
    for d in BASE_DETECTORS:
        cmd_run(data_root, out_root, DEV_SET, d)
    cmd_sweep(data_root, out_root)
    for s in SET_NAMES:
        if s != DEV_SET:
            for d in BASE_DETECTORS:
                cmd_run(data_root, out_root, s, d)
    for s in SET_NAMES:
        for e in ENSEMBLES:
            cmd_ensemble(data_root, out_root, s, e)
    for d in ALL_DETECTORS:
        cmd_errors(data_root, out_root, d)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m privgate.bench",
                                     description="privgate detection benchmark")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/m3"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--set", required=True, choices=SET_NAMES)
    p.add_argument("--detector", required=True, choices=BASE_DETECTORS)
    sub.add_parser("sweep")
    p = sub.add_parser("ensemble")
    p.add_argument("--set", required=True, choices=SET_NAMES)
    p.add_argument("--detector", required=True, choices=tuple(ENSEMBLES))
    p = sub.add_parser("errors")
    p.add_argument("--detector", required=True, choices=ALL_DETECTORS)
    sub.add_parser("all")
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd_run(args.data_root, args.out, args.set, args.detector)
    elif args.command == "sweep":
        cmd_sweep(args.data_root, args.out)
    elif args.command == "ensemble":
        cmd_ensemble(args.data_root, args.out, args.set, args.detector)
    elif args.command == "errors":
        cmd_errors(args.data_root, args.out, args.detector)
    else:
        cmd_all(args.data_root, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
