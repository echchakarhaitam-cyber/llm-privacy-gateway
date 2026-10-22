"""Milestone 6, RQ9: does unscreened text reach the upstream when screening fails?

Usage:  python scripts/run_gateway_faults.py [--n 1000] [--verify-sample 25]

1000 requests (one nemotron_dev_2k document -> one user message each), seed 0, through the
real gateway app (in-process ASGI client, MockUpstream that records every payload it gets).
Detector: crf+regex+presidio, replayed from the stored M4 predictions after checking that the
run scored this exact evaluation-set file (SHA-256); ``--verify-sample`` re-runs the REAL
detector on the first documents and records whether it reproduces the stored spans.

Per condition (fault kind x probability q x mode) a fault is injected into a request with
probability q: ``exception`` / ``hang`` / ``malformed`` (detector), ``upstream_5xx``, or
``mixed`` (one of the four, chosen uniformly). The same fault plan is used for both modes.
Modes: ``fail_closed`` (the gateway) and ``fail_open`` (the unsafe experiment-only variant,
enabled below by setting the required environment variable in THIS process only).

Metrics per condition (docs/results-gateway.md): served / blocked / upstream errors; requests
whose upstream payload contained a gold PII value verbatim ("leaked", values of >= 4
characters as in M5's verbatim survival); "fault-induced" leaks = gold values in the payload
that were NOT already there when no fault was injected (the detector's own misses, which
exist at q = 0, are subtracted); gold value characters reaching the upstream.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from privgate.bench.latency import environment
from privgate.bench.privacy import MIN_VALUE_LENGTH
from privgate.bench.runner import PREDICTIONS, read_predictions, write_json
from privgate.bench.sets import DEV_SET, load_evaluation_set
from privgate.gateway.app import ENV_ALLOW_FAIL_OPEN, GatewayConfig, create_app
from privgate.gateway.audit import AuditLog
from privgate.gateway.faults import (
    FAULT_KINDS,
    MIXED,
    FaultInjector,
    FaultyDetector,
    ReplayDetector,
    faulty_upstream,
)
from privgate.gateway.schema import Limits

DATA = Path("data")
OUT = Path("results/m6/faults.json")
DETECTOR = "crf+regex+presidio"
RUN_DIR = Path("results/m4") / DEV_SET / DETECTOR
SEED = 0
QS = (0.0, 0.01, 0.05, 0.1, 0.2)
CONDITIONS = (*FAULT_KINDS, MIXED)
MODES = ("fail_closed", "fail_open")
NO_FAULT = "none"
TIMEOUT_S = 0.05        # the gateway's screening budget in this experiment
HANG_S = 0.25           # an injected hang lasts 5x the budget
WORKERS = 16            # enough that hung detector threads never starve later requests


def fault_plan(kind: str, q: float, n: int) -> list[str | None]:
    """The fault (or None) of each request; identical for both modes."""
    rng = random.Random(f"{SEED}|{kind}|{q}")
    plan: list[str | None] = []
    for _ in range(n):
        hit = rng.random() < q
        pick = rng.choice(FAULT_KINDS)
        plan.append((pick if kind == MIXED else kind) if hit else None)
    return plan


def gold_values(doc) -> list[str]:
    """Distinct gold PII values of >= MIN_VALUE_LENGTH characters (as in M5)."""
    values = {doc.span_text(s) for s in doc.spans}
    return sorted(v for v in values if len(v) >= MIN_VALUE_LENGTH)


def payload_text(payload: dict) -> str:
    parts = []
    for message in payload["messages"]:
        content = message["content"]
        parts.extend([content] if isinstance(content, str) else [p["text"] for p in content])
    return "\n".join(parts)


async def run_condition(mode, kind, q, docs, values, spans_by_text, baseline):
    """Send every request of one condition; returns (row, per-request surviving values)."""
    injector = FaultInjector()
    upstream = faulty_upstream(injector)
    audit = AuditLog(keep=True)
    counter = itertools.count()
    app = create_app(
        GatewayConfig(detector_timeout_s=TIMEOUT_S, detector_workers=WORKERS,
                      fail_open=(mode == "fail_open"), seed=SEED,
                      limits=Limits(max_request_bytes=4_000_000)),
        detector=FaultyDetector(ReplayDetector(spans_by_text), injector, HANG_S),
        upstream=upstream, audit=audit, request_id_factory=lambda: f"r{next(counter):06d}")
    plan = fault_plan(kind, q, len(docs))
    status, survivors = Counter(), []
    leaked = fault_leaked = leaked_chars = fault_leaked_chars = forwarded = 0
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gateway") as client:
        for i, (doc, fault) in enumerate(zip(docs, plan, strict=True)):
            injector.current = fault
            before = len(upstream.received)
            body = {"model": "local", "messages": [{"role": "user", "content": doc.text}]}
            response = await client.post("/v1/chat/completions", json=body)
            status[response.status_code] += 1
            sent = upstream.received[before:]
            found: list[str] = []
            if sent:
                forwarded += 1
                text = "\n".join(payload_text(p) for p in sent)
                found = [v for v in values[i] if v in text]
            survivors.append(found)
            extra = [v for v in found if baseline is not None and v not in baseline[i]]
            leaked += bool(found)
            leaked_chars += sum(map(len, found))
            fault_leaked += bool(extra)
            fault_leaked_chars += sum(map(len, extra))
    injector.current = None
    decisions = Counter(r["decision"] for r in audit.records)
    row = {
        "mode": mode, "fault": kind if q else NO_FAULT, "q": q, "requests": len(docs),
        "faults_injected": sum(f is not None for f in plan),
        "faults_by_kind": dict(sorted(Counter(f for f in plan if f).items())),
        "status": {str(k): v for k, v in sorted(status.items())},
        "served": status[200], "blocked": status[403] + status[503],
        "upstream_errors": status[502],
        "forwarded_to_upstream": forwarded,
        "unscreened_forwarded": decisions["served_unscreened"],
        "decisions": dict(sorted(decisions.items())),
        "leaked_requests": leaked, "fault_leaked_requests": fault_leaked,
        "leaked_value_chars": leaked_chars, "fault_leaked_value_chars": fault_leaked_chars,
        "availability": status[200] / len(docs),
    }
    return row, survivors


def load_replay(n: int):
    full, sha = load_evaluation_set(DEV_SET, DATA)
    meta = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    if meta["set_sha256"] != sha:
        raise SystemExit(f"{RUN_DIR} was scored on another version of {DEV_SET}")
    spans, _ = read_predictions(RUN_DIR / PREDICTIONS, full)
    docs, spans = full[:n], spans[:n]
    by_text: dict[str, list] = {}
    for doc, s in zip(docs, spans, strict=True):
        if by_text.setdefault(doc.text, s) != s:
            raise SystemExit("two documents with the same text have different predictions")
    return docs, by_text, sha


def verify_real_detector(docs, spans_by_text, k: int) -> dict:
    """Does the real detector reproduce the replayed spans on the first k documents?"""
    from privgate.gateway.detectors import build_detector

    detector = build_detector(DETECTOR)
    same = sum(detector.detect(d.text) == spans_by_text[d.text] for d in docs[:k])
    return {"documents": k, "identical": same}


async def main_async(args) -> dict:
    t0 = time.perf_counter()
    docs, spans_by_text, sha = load_replay(args.n)
    values = [gold_values(d) for d in docs]
    total_chars = sum(len(v) for vs in values for v in vs)
    verify = (verify_real_detector(docs, spans_by_text, args.verify_sample)
              if args.verify_sample else None)
    print("real-detector check:", verify, flush=True)
    os.environ[ENV_ALLOW_FAIL_OPEN] = "1"   # this process only: the fail-open arm needs it

    # What reaches the upstream when nothing goes wrong: the detector's own misses.
    _, baseline = await run_condition("fail_closed", MIXED, 0.0, docs, values, spans_by_text,
                                      None)
    rows = []
    for mode in MODES:
        for kind, q in [(MIXED, 0.0)] + [(k, q) for k in CONDITIONS for q in QS if q > 0]:
            row, _ = await run_condition(mode, kind, q, docs, values, spans_by_text, baseline)
            rows.append(row)
            print(f"{mode:11s} {row['fault']:12s} q={q:<5} served={row['served']:4d} "
                  f"blocked={row['blocked']:4d} 502={row['upstream_errors']:3d} "
                  f"unscreened={row['unscreened_forwarded']:4d} "
                  f"leaked={row['leaked_requests']:4d} "
                  f"fault-induced={row['fault_leaked_requests']:4d}", flush=True)
    return {
        "milestone": 6, "experiment": "RQ9", "seed": SEED, "detector": DETECTOR,
        "evaluation_set": DEV_SET, "evaluation_set_sha256": sha, "requests": len(docs),
        "timeout_s": TIMEOUT_S, "hang_s": HANG_S, "detector_workers": WORKERS,
        "qs": list(QS), "gold_value_chars_total": total_chars,
        "baseline_requests_with_misses": sum(bool(b) for b in baseline),
        "real_detector_check": verify, "rows": rows,
        "environment": environment(), "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--verify-sample", type=int, default=25)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    logging.getLogger("privgate.gateway").setLevel(logging.ERROR)   # fail-open warnings
    result = asyncio.run(main_async(args))
    write_json(args.out, result)
    print(f"wrote {args.out} in {result['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
