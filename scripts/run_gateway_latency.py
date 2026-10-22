"""Milestone 6, RQ10: what latency does screening add, and what throughput does one process hold?

Usage:  python scripts/run_gateway_latency.py [--n 500] [--warmup 50] [--wait-minutes 10]

Configurations: ``none`` (a detector that finds nothing: the whole gateway pipeline runs, only
detection is empty -- the passthrough baseline), ``regex``, ``crf``, ``crf+regex``,
``crf+regex+presidio``. MockUpstream with 0 ms latency, so the measured time is the gateway's.
Requests are the first ``warmup + n`` nemotron_dev_2k documents, one user message each.

Two harnesses, per configuration:
  asgi    in-process httpx ASGI client (no sockets): framework + screening only
  server  a real uvicorn process on 127.0.0.1 and an httpx load generator in this process
Each reports p50 / p95 / p99 / mean latency (ms, measured by the client around the request)
and requests per second at concurrency 1 and 4, after ``warmup`` untimed requests.

BEFORE measuring, the machine must be idle: total CPU <= 20 % over 5 s and no other heavy
Python process (>= 10 % CPU or >= 500 MB). If not, the script waits (``--wait-minutes``) and
then stops without measuring. The check is stored with the results.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

DATA = Path("data")
OUT = Path("results/m6/latency.json")
CONFIGS = ("none", "regex", "crf", "crf+regex", "crf+regex+presidio")
CONCURRENCY = (1, 4)
IDLE_CPU_MAX = 20.0
HEAVY_CPU = 10.0
HEAVY_RSS_MB = 500
PACKAGES = ("fastapi", "starlette", "uvicorn", "httpx", "anyio", "h11", "psutil", "pydantic")


class NullDetector:
    """Finds nothing: isolates the cost of everything except detection."""

    name = "none"
    supported_labels = frozenset()

    def detect(self, text: str) -> list:
        return []


def make_app(config: str):
    from privgate.gateway.app import GatewayConfig, create_app
    from privgate.gateway.audit import AuditLog
    from privgate.gateway.detectors import build_detector
    from privgate.gateway.schema import Limits
    from privgate.gateway.upstream import MockUpstream

    detector = NullDetector() if config == "none" else build_detector(config)
    audit = AuditLog(Path(tempfile.gettempdir()) / f"privgate_latency_{os.getpid()}.jsonl",
                     keep=True)
    app = create_app(GatewayConfig(limits=Limits(max_request_bytes=4_000_000)),
                     detector=detector, upstream=MockUpstream(record=False), audit=audit)
    return app, audit


def serve(config: str, port: int) -> None:
    import uvicorn

    logging.getLogger("privgate.gateway").setLevel(logging.ERROR)
    app, _ = make_app(config)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


# --- idle check ------------------------------------------------------------------------------

def idle_check() -> dict:
    import psutil

    me = {os.getpid(), os.getppid()}
    procs = [p for p in psutil.process_iter(["pid", "name"])
             if (p.info["name"] or "").lower().startswith("python") and p.pid not in me]
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    total = psutil.cpu_percent(interval=5.0)
    listing, heavy = [], []
    for p in procs:
        try:
            cpu, rss = p.cpu_percent(None), p.memory_info().rss / 2**20
            cmd = " ".join(p.cmdline())[:100]
        except psutil.Error:
            continue
        listing.append({"pid": p.pid, "cpu_percent": cpu, "rss_mb": round(rss), "cmd": cmd})
        if cpu >= HEAVY_CPU or rss >= HEAVY_RSS_MB:
            heavy.append(p.pid)
    return {"total_cpu_percent_5s": total, "load_ok": total <= IDLE_CPU_MAX,
            "other_python_processes": listing, "heavy_python_pids": heavy,
            "idle": total <= IDLE_CPU_MAX and not heavy,
            "logical_cpus": psutil.cpu_count(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def wait_until_idle(minutes: float) -> dict:
    deadline = time.time() + minutes * 60
    while True:
        check = idle_check()
        print("idle check:", json.dumps({k: v for k, v in check.items()
                                         if k != "other_python_processes"}), flush=True)
        if check["idle"]:
            return check
        if time.time() > deadline:
            print("machine is NOT idle; refusing to measure under load:\n"
                  + json.dumps(check, indent=2), flush=True)
            raise SystemExit(3)
        time.sleep(30)


# --- measurement -----------------------------------------------------------------------------

def stats(latencies_ms: list[float], wall_s: float, errors: int) -> dict:
    from privgate.bench.bootstrap import percentile

    ms = sorted(latencies_ms)
    return {"requests": len(ms), "errors": errors, "p50_ms": percentile(ms, 0.5),
            "p95_ms": percentile(ms, 0.95), "p99_ms": percentile(ms, 0.99),
            "mean_ms": sum(ms) / len(ms), "requests_per_s": len(ms) / wall_s}


async def drive(post, bodies: list[dict], concurrency: int) -> dict:
    latencies: list[float] = []
    errors = 0
    queue = iter(range(len(bodies)))

    async def worker():
        nonlocal errors
        for i in queue:
            t = time.perf_counter()
            status = await post(bodies[i])
            latencies.append((time.perf_counter() - t) * 1e3)
            errors += status != 200

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return stats(latencies, time.perf_counter() - t0, errors)


async def measure_asgi(config: str, warm: list[dict], bodies: list[dict]) -> list[dict]:
    import httpx

    app, audit = make_app(config)
    rows = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gateway", timeout=60) as client:
        async def post(body):
            return (await client.post("/v1/chat/completions", json=body)).status_code

        await drive(post, warm, 1)
        for c in CONCURRENCY:
            start = len(audit.records)
            row = await drive(post, bodies, c)
            recs = audit.records[start:]
            row["gateway_stage_mean_ms"] = {
                k: sum(r["latency_ms"].get(k, 0.0) for r in recs) / max(1, len(recs))
                for k in ("parse", "detect", "mask", "upstream", "unmask", "total")}
            rows.append({"harness": "asgi", "config": config, "concurrency": c, **row})
    return rows


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def measure_server(config: str, warm: list[dict], bodies: list[dict]) -> list[dict]:
    import httpx

    port = free_port()
    proc = await asyncio.to_thread(
        subprocess.Popen, [sys.executable, __file__, "--serve", config, "--port", str(port)],
        env={**os.environ, "PYTHONPATH": "src"})
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):                      # model loading can take a while
            try:
                reply = await asyncio.to_thread(urllib.request.urlopen, base + "/healthz", None, 1)
                if reply.status == 200:
                    break
            except OSError:
                if proc.poll() is not None:
                    raise SystemExit(f"server for {config} exited early") from None
                await asyncio.sleep(0.5)
        else:
            raise SystemExit(f"server for {config} did not become ready")
        rows = []
        limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
        async with httpx.AsyncClient(base_url=base, timeout=60, limits=limits,
                                     trust_env=False) as client:
            async def post(body):
                return (await client.post("/v1/chat/completions", json=body)).status_code

            await drive(post, warm, 1)
            for c in CONCURRENCY:
                rows.append({"harness": "server", "config": config, "concurrency": c,
                             **(await drive(post, bodies, c))})
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


class LoadSampler:
    """Recorded-load protocol: once per second, total CPU % and the top 5 processes by CPU %.

    Runs in a background thread of the measuring process while a run is in progress. Only
    process NAMES and ids are recorded (never command lines). The sampler itself costs some
    CPU (it walks the process table each second); that cost is part of the recorded load.
    """

    def __init__(self) -> None:
        import threading

        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._known: dict = {}

    def _top(self) -> list[dict]:
        import psutil

        rows = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                proc = self._known.setdefault(p.pid, p)
                rows.append({"pid": p.pid, "name": p.info["name"] or "?",
                             "cpu_percent": proc.cpu_percent(None)})
            except psutil.Error:
                continue
        rows.sort(key=lambda r: -r["cpu_percent"])
        return rows[:5]

    def _loop(self) -> None:
        import psutil

        psutil.cpu_percent(None)
        self._top()                       # primes the per-process counters
        t0 = time.time()
        while not self._stop.wait(1.0):
            self.samples.append({"t": round(time.time() - t0, 1),
                                 "total_cpu_percent": psutil.cpu_percent(None),
                                 "top5": self._top()})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        from privgate.bench.bootstrap import percentile

        cpu = sorted(s["total_cpu_percent"] for s in self.samples)
        per_name: dict[str, list[float]] = {}
        for sample in self.samples:
            for row in sample["top5"]:
                per_name.setdefault(row["name"], []).append(row["cpu_percent"])
        heaviest = sorted(per_name.items(), key=lambda kv: -sum(kv[1]))[:8]
        return {"samples": len(cpu), "total_cpu_mean": sum(cpu) / len(cpu) if cpu else None,
                "total_cpu_median": percentile(cpu, 0.5) if cpu else None,
                "total_cpu_p95": percentile(cpu, 0.95) if cpu else None,
                "total_cpu_max": cpu[-1] if cpu else None,
                "share_of_samples_above_idle_limit":
                    sum(c > IDLE_CPU_MAX for c in cpu) / len(cpu) if cpu else None,
                "most_frequent_top5_processes": [
                    {"name": n, "appearances": len(v), "mean_cpu_percent": sum(v) / len(v)}
                    for n, v in heaviest]}


async def measure_all(warm: list[dict], timed: list[dict]) -> list[dict]:
    rows = []
    for measure in (measure_asgi, measure_server):
        for config in CONFIGS:
            t0 = time.perf_counter()
            got = await measure(config, warm, timed)
            rows.extend(got)
            for r in got:
                print(f"{r['harness']:6s} {config:20s} c={r['concurrency']} "
                      f"p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} p99={r['p99_ms']:.2f} "
                      f"rps={r['requests_per_s']:.1f} errors={r['errors']}", flush=True)
            print(f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


def median_rows(runs: list[list[dict]]) -> list[dict]:
    """Per (harness, config, concurrency): the median over runs of p50 / p95 / p99 / req/s."""
    import statistics

    keys = [(r["harness"], r["config"], r["concurrency"]) for r in runs[0]]
    out = []
    for key in keys:
        same = [next(r for r in run if (r["harness"], r["config"], r["concurrency"]) == key)
                for run in runs]
        row = {"harness": key[0], "config": key[1], "concurrency": key[2], "runs": len(same),
               "errors": sum(r["errors"] for r in same)}
        for metric in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "requests_per_s"):
            row[metric] = statistics.median(r[metric] for r in same)
            row[metric + "_min"] = min(r[metric] for r in same)
            row[metric + "_max"] = max(r[metric] for r in same)
        out.append(row)
    return out


async def main_async(args) -> dict:
    from privgate.bench.latency import environment
    from privgate.bench.sets import DEV_SET, load_evaluation_set

    docs, sha = load_evaluation_set(DEV_SET, DATA)
    docs = docs[:args.warmup + args.n]
    bodies = [{"model": "local", "messages": [{"role": "user", "content": d.text}]}
              for d in docs]
    warm, timed = bodies[:args.warmup], bodies[args.warmup:]
    common = {
        "milestone": 7, "experiment": "RQ10", "evaluation_set": DEV_SET,
        "evaluation_set_sha256": sha, "warmup_requests": args.warmup,
        "timed_requests": args.n, "upstream": "MockUpstream, 0 ms latency",
        "environment": environment(), "package_versions": versions(),
    }
    if args.protocol == "idle":
        check = ({"skipped": True} if args.no_idle_check
                 else wait_until_idle(args.wait_minutes))
        rows = await measure_all(warm, timed)
        return {**common, "protocol": "idle", "idle_check_before": check,
                "idle_check_after": {"skipped": True} if args.no_idle_check else idle_check(),
                "rows": rows}
    runs, loads, started = [], [], []
    for number in range(1, args.repeats + 1):
        print(f"--- recorded-load run {number} of {args.repeats}", flush=True)
        started.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        with LoadSampler() as sampler:
            runs.append(await measure_all(warm, timed))
        loads.append({"run": number, "started": started[-1],
                      "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                      **sampler.summary(), "samples_per_second": sampler.samples})
    return {**common, "protocol": "recorded-load", "repeats": args.repeats, "runs": runs,
            "load_per_run": loads, "rows": median_rows(runs)}


def main(argv: list[str] | None = None) -> int:
    from privgate.bench.runner import write_json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--wait-minutes", type=float, default=10)
    parser.add_argument("--protocol", choices=("idle", "recorded-load"), default="idle",
                        help="idle: refuse on a busy machine; recorded-load: measure anyway, "
                             "record the CPU load every second, repeat, report medians")
    parser.add_argument("--repeats", type=int, default=3, help="recorded-load: whole runs")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-idle-check", action="store_true",
                        help="TESTS ONLY: skip the idle check; the output is not a measurement")
    parser.add_argument("--serve", choices=CONFIGS, help="(internal) run the server only")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.serve:
        serve(args.serve, args.port)
        return 0
    logging.getLogger("privgate.gateway").setLevel(logging.ERROR)
    result = asyncio.run(main_async(args))
    write_json(args.out, result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
