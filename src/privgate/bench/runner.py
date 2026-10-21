"""Running a detector over an evaluation set, storing predictions and scoring them.

Layout under the output root (results/m3/):

    <set>/<detector>/predictions.jsonl   one line per document (gitignored)
    <set>/<detector>/raw_results.jsonl   Presidio's unmapped results on dev (gitignored)
    <set>/<detector>/metrics.json        committed
    run_log.json                         wall time of every step (committed)

Detector output is checked against the Detector contract (Span objects inside the text,
sorted, non-overlapping) before it is scored: a detector that breaks the contract fails the
run instead of producing a number.

Ensembles are scored from their members' stored predictions with
:meth:`EnsembleDetector.combine`, which is exactly what ``EnsembleDetector.detect`` returns;
their per-document latency is the sum of the members' latencies plus the measured time of
the combination, i.e. the cost of running the members one after the other.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from privgate.bench.latency import environment, latency_stats
from privgate.bench.metrics import evaluate_run
from privgate.data.documents import Document
from privgate.detectors.base import Detector
from privgate.detectors.ensemble import EnsembleDetector
from privgate.spans import EntityType, Span

PREDICTIONS = "predictions.jsonl"
RAW_RESULTS = "raw_results.jsonl"
METRICS = "metrics.json"
RUN_LOG = "run_log.json"


class ContractError(RuntimeError):
    """A detector returned spans that violate the Detector protocol."""


@dataclass(frozen=True)
class RawResult:
    """A stored Presidio result (same attributes as presidio_analyzer.RecognizerResult)."""

    entity_type: str
    start: int
    end: int
    score: float


def check_spans(spans: Sequence[Span], text: str, detector: str) -> None:
    previous_end = 0
    for s in spans:
        if not isinstance(s, Span):
            raise ContractError(f"{detector}: returned {s!r}, not a Span")
        if s.end > len(text):
            raise ContractError(f"{detector}: span {s} exceeds the text")
        if s.start < previous_end:
            raise ContractError(f"{detector}: spans overlap or are unsorted at {s}")
        previous_end = s.end


def timed_detect(detector: Detector, docs: Sequence[Document]) -> tuple[list[list[Span]], list[int]]:
    """Predictions and per-document nanoseconds spent inside ``detect()``."""
    preds, timings = [], []
    for doc in docs:
        t0 = time.perf_counter_ns()
        spans = detector.detect(doc.text)
        timings.append(time.perf_counter_ns() - t0)
        check_spans(spans, doc.text, detector.name)
        preds.append(list(spans))
    return preds, timings


def timed_presidio_raw(detector, docs: Sequence[Document]):
    """Like timed_detect for Presidio, also keeping the unmapped results for the sweep."""
    raws, preds, timings = [], [], []
    for doc in docs:
        t0 = time.perf_counter_ns()
        raw = detector.analyze_raw(doc.text)
        spans = detector.spans_from_results(raw, detector.score_threshold)
        timings.append(time.perf_counter_ns() - t0)
        check_spans(spans, doc.text, detector.name)
        raws.append([RawResult(r.entity_type, r.start, r.end, float(r.score)) for r in raw])
        preds.append(spans)
    return raws, preds, timings


def _write_lines(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_lines(path: Path, docs: Sequence[Document]) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if [r["doc_id"] for r in records] != [d.doc_id for d in docs]:
        raise ValueError(f"{path} does not match the documents of the evaluation set")
    return records


def write_predictions(path: Path, docs, preds, timings) -> None:
    _write_lines(path, (
        {"doc_id": d.doc_id, "latency_ns": t,
         "spans": [[s.start, s.end, s.label.value, s.score, s.source] for s in p]}
        for d, p, t in zip(docs, preds, timings)
    ))


def read_predictions(path: Path, docs) -> tuple[list[list[Span]], list[int]]:
    records = _read_lines(path, docs)
    preds = [[Span(a, b, EntityType(lab), score, src) for a, b, lab, score, src in r["spans"]]
             for r in records]
    return preds, [r["latency_ns"] for r in records]


def write_raw_results(path: Path, docs, raws, timings) -> None:
    _write_lines(path, (
        {"doc_id": d.doc_id, "latency_ns": t,
         "results": [[r.entity_type, r.start, r.end, r.score] for r in raw]}
        for d, raw, t in zip(docs, raws, timings)
    ))


def read_raw_results(path: Path, docs) -> tuple[list[list[RawResult]], list[int]]:
    records = _read_lines(path, docs)
    return ([[RawResult(*r) for r in rec["results"]] for rec in records],
            [rec["latency_ns"] for rec in records])


def compose_ensemble(member_preds: Sequence[Sequence[Sequence[Span]]],
                     member_timings: Sequence[Sequence[int]]):
    """Ensemble predictions and latencies from each member's stored run."""
    preds, timings = [], []
    for i in range(len(member_preds[0])):
        t0 = time.perf_counter_ns()
        spans = EnsembleDetector.combine(m[i] for m in member_preds)
        timings.append(time.perf_counter_ns() - t0 + sum(t[i] for t in member_timings))
        preds.append(spans)
    return preds, timings


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def score_run(
    out_dir: Path,
    *,
    detector: str,
    members: Sequence[str],
    config: dict,
    supported_labels,
    set_name: str,
    set_sha256: str,
    docs: Sequence[Document],
    preds: Sequence[Sequence[Span]],
    timings: Sequence[int],
    raw_labels: dict | None,
    runtime_seconds: float,
) -> dict:
    """Compute and write metrics.json for one run; returns the payload."""
    payload = {
        "detector": detector,
        "members": list(members),
        "config": config,
        "supported_labels": sorted(label.value for label in supported_labels),
        "set": set_name,
        "set_sha256": set_sha256,
        "views": evaluate_run(docs, preds, supported_labels, raw_labels),
        "latency": latency_stats(timings, [len(d.text) for d in docs]),
        "environment": environment(),
        "runtime_seconds": runtime_seconds,
    }
    write_json(out_dir / METRICS, payload)
    return payload


def log_step(out_root: Path, step: str, set_name: str | None, detector: str | None,
             seconds: float) -> None:
    """Record the wall time of a step; re-running a step replaces its entry."""
    path = Path(out_root) / RUN_LOG
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"steps": []}
    key = (step, set_name, detector)
    steps = [s for s in log["steps"] if (s["step"], s["set"], s["detector"]) != key]
    steps.append({"step": step, "set": set_name, "detector": detector,
                  "seconds": round(seconds, 3)})
    write_json(path, {"steps": steps,
                      "total_seconds": round(sum(s["seconds"] for s in steps), 3)})
