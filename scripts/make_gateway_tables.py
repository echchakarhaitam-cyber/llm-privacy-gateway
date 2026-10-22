"""Render docs/results-gateway.md from results/m6/faults.json and results/m6/latency.json.

Usage:  python scripts/make_gateway_tables.py

Every number in the document is read from those two files (written by
scripts/run_gateway_faults.py and scripts/run_gateway_latency.py); the observations are
computed, not typed. A missing file renders its section as "NOT MEASURED".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAULTS = Path("results/m6/faults.json")
LATENCY = Path("results/m6/latency.json")
REFUSAL = Path("results/m6/latency_refusal.json")
OUT = Path("docs/results-gateway.md")
FAULT_ORDER = ("none", "exception", "hang", "malformed", "upstream_5xx", "mixed")
CONFIG_ORDER = ("none", "regex", "crf", "crf+regex", "crf+regex+presidio")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ordered(rows: list[dict], mode: str) -> list[dict]:
    sel = [r for r in rows if r["mode"] == mode]
    return sorted(sel, key=lambda r: (FAULT_ORDER.index(r["fault"]), r["q"]))


def rq9(data) -> str:
    if data is None:
        return "NOT MEASURED (results/m6/faults.json is missing).\n"
    n = data["requests"]
    parts = [
        (f"{n} requests (one `{data['evaluation_set']}` document each, seed {data['seed']}), "
        f"detector `{data['detector']}` replayed from the stored M4 predictions "
        f"(set SHA-256 `{data['evaluation_set_sha256'][:12]}…`); screening budget "
        f"{data['timeout_s'] * 1000:.0f} ms, injected hangs last {data['hang_s'] * 1000:.0f} ms, "
        f"{data['detector_workers']} detector workers. The same fault plan is used for both "
        "modes. Columns: **injected** faults; **served** (200); **blocked** (503 screening "
        "unavailable or 403); **502** upstream failures; **unscreened fwd** requests forwarded "
        "although screening had failed; **leaked** requests whose upstream payload contained a "
        "gold PII value (>= 4 characters) verbatim; **fault-induced** the subset of leaked "
        "values that were not already present at q = 0 (the detector's own misses are "
        "subtracted); **fault chars** gold value characters reaching the upstream because of "
        "faults."),
    ]
    check = data.get("real_detector_check")
    if check:
        parts.append(f"Real-detector check: the real `{data['detector']}` pipeline reproduced "
                     f"the replayed spans on {check['identical']} of {check['documents']} "
                     "documents.")
    parts.append(f"Gold value characters in the {n} requests: {data['gold_value_chars_total']:,}; "
                 f"requests with at least one value the detector misses: "
                 f"{data['baseline_requests_with_misses']}.")
    header = ["fault", "q", "injected", "served", "blocked", "502", "unscreened fwd", "leaked",
              "fault-induced", "fault chars", "availability"]
    for mode, title in (("fail_closed", "Fail-closed (the gateway)"),
                        ("fail_open", "Fail-open (unsafe, experiment only)")):
        rows = [[r["fault"], f"{r['q']:g}", r["faults_injected"], r["served"], r["blocked"],
                 r["upstream_errors"], r["unscreened_forwarded"], r["leaked_requests"],
                 r["fault_leaked_requests"], r["fault_leaked_value_chars"],
                 f"{r['availability']:.3f}"] for r in ordered(data["rows"], mode)]
        parts.append(f"**{title}**\n\n" + table(header, rows))
    return "\n\n".join(parts) + "\n"


def rq9_observations(data) -> list[str]:
    if data is None:
        return []
    closed = [r for r in data["rows"] if r["mode"] == "fail_closed"]
    opened = [r for r in data["rows"] if r["mode"] == "fail_open"]
    out = [(f"Fail-closed: {sum(r['unscreened_forwarded'] for r in closed)} unscreened requests "
           f"were forwarded and {sum(r['fault_leaked_requests'] for r in closed)} requests "
           f"leaked a gold value because of a fault, summed over all {len(closed)} conditions "
           f"({sum(r['faults_injected'] for r in closed)} injected faults).")]
    worst = max(opened, key=lambda r: r["fault_leaked_requests"])
    out.append(
        f"Fail-open: {sum(r['unscreened_forwarded'] for r in opened)} unscreened requests were "
        f"forwarded in total; the worst condition ({worst['fault']}, q = {worst['q']:g}) put "
        f"{worst['fault_leaked_requests']} of {worst['requests']} requests "
        f"({worst['fault_leaked_value_chars']:,} gold value characters) at the upstream because "
        "of faults.")
    base = next(r for r in closed if r["fault"] == "none")
    out.append(f"With no faults, {base['leaked_requests']} of {base['requests']} requests still "
               "carry at least one gold value the detector missed to the upstream, in both "
               "modes: this is the detection gap of RQ7/RQ8, not a gateway failure.")
    for kind in ("exception", "hang", "malformed", "mixed"):
        row = [r for r in closed if r["fault"] == kind and r["q"] == max(data["qs"])]
        if row:
            r = row[0]
            out.append(f"Fail-closed at q = {r['q']:g}, `{kind}`: availability "
                       f"{r['availability']:.3f} ({r['blocked']} blocked of {r['requests']}).")
    row = [r for r in closed if r["fault"] == "upstream_5xx" and r["q"] == max(data["qs"])]
    if row:
        r = row[0]
        out.append(f"Upstream 5xx at q = {r['q']:g} gives {r['upstream_errors']} 502 responses in "
                   "both modes and no fault-induced leak: the payload was already screened.")
    return out


def refusal_note() -> str:
    info = load(REFUSAL)
    if info is None:
        return ""
    rows = [[r["attempt"], r["checks"], f"{r['first']} - {r['last']}", r["cpu_min"],
             r["cpu_median"], r["cpu_max"], r["checks_with_heavy_python"], r["verdict"]]
            for r in info["runs"]]
    return (
        "\n\nThe measurement script refused to run because the machine never met the idle "
        f"criterion (total CPU <= {info['limit_total_cpu_percent']:g} % over 5 s, measured with "
        "psutil, and no other heavy Python process). The checks, from the script's own output "
        "(`results/m6/latency_refusal.json`):\n\n"
        + table(["attempt", "checks", "period", "CPU % min", "median", "max",
                 "checks with a heavy Python process", "verdict"], rows)
        + "\n\nNon-Python background processes seen: "
        f"{info['non_python_background_processes_seen']}. The threshold was not relaxed after "
        "the fact. To measure: close the background load, run "
        "`python scripts/run_gateway_latency.py`, then `python scripts/make_gateway_tables.py`.\n")


def rq10(data) -> str:
    if data is None:
        return "NOT MEASURED (results/m6/latency.json does not exist)." + refusal_note()
    check = data["idle_check_before"]
    if check.get("skipped"):
        return "NOT MEASURED (the result file was produced without the idle check)."             + refusal_note()
    env = data["environment"]
    parts = [
        (f"{data['timed_requests']} timed requests per row after {data['warmup_requests']} "
        f"warm-up requests (`{data['evaluation_set']}`, one document per request); "
        f"{data['upstream']}. Latency is measured by the client around each request (ms). "
        "Configuration `none` runs the whole gateway with a detector that finds nothing. "
        "`asgi` = in-process httpx client without sockets; `server` = a real uvicorn process "
        "on 127.0.0.1 with an httpx load generator. One gateway process, one detector worker."),
    ]
    header = ["config", "c", "p50", "p95", "p99", "mean", "req/s", "errors"]
    for harness, title in (("asgi", "In-process ASGI client"),
                           ("server", "uvicorn server on 127.0.0.1")):
        sel = sorted((r for r in data["rows"] if r["harness"] == harness),
                     key=lambda r: (CONFIG_ORDER.index(r["config"]), r["concurrency"]))
        rows = [[r["config"], r["concurrency"], f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}",
                 f"{r['p99_ms']:.2f}", f"{r['mean_ms']:.2f}", f"{r['requests_per_s']:.1f}",
                 r["errors"]] for r in sel]
        parts.append(f"**{title}** (ms; c = concurrency)\n\n" + table(header, rows))
    sel = [r for r in data["rows"] if r["harness"] == "asgi" and r["concurrency"] == 1]
    rows = [[r["config"]] + [f"{r['gateway_stage_mean_ms'][k]:.2f}"
                             for k in ("parse", "detect", "mask", "upstream", "unmask", "total")]
            for r in sorted(sel, key=lambda r: CONFIG_ORDER.index(r["config"]))]
    parts.append("**Where the time goes** (ASGI, concurrency 1; mean ms per request from the "
                 "gateway's own audit stages)\n\n"
                 + table(["config", "parse", "detect", "mask", "upstream", "unmask", "total"],
                         rows))
    others = check["other_python_processes"]
    parts.append(
        "**Machine** — "
        f"CPU: {env['cpu']} ({env['logical_cpus']} logical CPUs); OS: {env['os']}; Python "
        f"{env['python']}; packages: "
        + ", ".join(f"{k} {v}" for k, v in data["package_versions"].items() if v)
        + f"; privgate commit `{(env.get('privgate_commit') or 'unknown')[:10]}`.")
    parts.append(
        "**Idle check before measuring** — total CPU "
        f"{check['total_cpu_percent_5s']:.1f} % over 5 s (limit 20 %); other Python processes: "
        + ("none" if not others else "; ".join(
            f"pid {p['pid']} {p['cpu_percent']:.1f} % CPU {p['rss_mb']} MB `{p['cmd']}`"
            for p in others))
        + f"; heavy ones: {check['heavy_python_pids'] or 'none'}; verdict: "
        f"{'idle' if check['idle'] else 'NOT idle'} ({check['time']}). After the run: total CPU "
        f"{data['idle_check_after']['total_cpu_percent_5s']:.1f} %.")
    return "\n\n".join(parts) + "\n"


def rq10_observations(data) -> list[str]:
    if data is None or data["idle_check_before"].get("skipped"):
        return []

    def row(harness, config, c):
        return next(r for r in data["rows"] if r["harness"] == harness
                    and r["config"] == config and r["concurrency"] == c)

    out = []
    for harness in ("asgi", "server"):
        base = row(harness, "none", 1)
        text = [f"{harness}, c = 1: baseline p50 {base['p50_ms']:.2f} ms"]
        for config in CONFIG_ORDER[1:]:
            r = row(harness, config, 1)
            delta = r["p50_ms"] - base["p50_ms"]
            text.append(f"`{config}` p50 {r['p50_ms']:.2f} ms ({delta:+.2f}), "
                        f"p95 {r['p95_ms']:.2f} ms")
        out.append("; ".join(text) + ".")
    for config in CONFIG_ORDER:
        a, b = row("server", config, 1), row("server", config, 4)
        out.append(f"`{config}` (server): {a['requests_per_s']:.1f} req/s at c = 1 and "
                   f"{b['requests_per_s']:.1f} req/s at c = 4.")
    out.append("Non-200 responses in all timed requests: "
               f"{sum(r['errors'] for r in data['rows'])}.")
    return out


THREATS = """\
- **Mock upstream.** MockUpstream answers instantly and in-process. A real model adds seconds;
  the numbers here are the gateway's own overhead, not an end-to-end latency. The unmasking
  of real answers (paraphrased placeholders, partial echoes) is not exercised.
- **Replayed detector (RQ9).** The detector output is replayed from stored M4 predictions
  (and checked on a sample against the real pipeline). RQ9 therefore tests the gateway's
  handling of detector outcomes, not the detectors' robustness to odd inputs.
- **Injected, not natural, faults.** The four fault types were chosen by us; a real detector
  can fail in ways not listed (a hang inside native code, memory exhaustion). Faults were
  injected one request at a time, sequentially, and independently across requests.
- **A single process and a thread-based timeout.** An injected hang runs 5x the budget and
  then ends. A detector that hangs forever would hold its worker thread indefinitely
  (docs/GATEWAY.md); this was not measured.
- **Leak metric.** "Leaked" means a gold value of >= 4 characters occurs verbatim in the
  payload. It over-counts values that also occur innocently, under-counts short values and
  anything the annotation missed, and says nothing about quasi-identifiers. Fault-induced
  leaks are measured relative to the same detector's output without faults.
- **Single laptop, one run.** Timings come from one machine and one run per row, with no
  repetition or confidence interval; p99 over 500 requests rests on 5 observations. The
  idle check is a 5-second snapshot taken before the run, not a guarantee for its duration.
  Absolute numbers do not transfer to other hardware; the ranking of configurations should.
- **Synthetic requests.** Requests are single-message documents from the synthetic Nemotron
  dev set, far longer than a typical chat turn, with no multi-turn history, no system prompt
  and no text parts. Concurrency 4 on one detector worker mostly measures queueing.
- **Order effects.** Configurations ran in a fixed order, not interleaved, so thermal state
  or background activity can favour some of them.
"""


def main() -> int:
    faults, latency = load(FAULTS), load(LATENCY)
    observations = rq9_observations(faults) + rq10_observations(latency)
    bullets = "\n".join(f"- {o}" for o in observations) if observations else "NOT MEASURED."
    doc = f"""\
# Gateway results (Milestone 6)

Generated by `scripts/make_gateway_tables.py` from `results/m6/faults.json` and
`results/m6/latency.json`. The gateway is described in [GATEWAY.md](GATEWAY.md).

- **RQ9** Under injected faults (detector exceptions, timeouts, malformed detector output,
  upstream errors), does any unscreened text reach the upstream? Fail-closed compared with a
  deliberately fail-open variant: leaked requests versus availability.
- **RQ10** What latency does screening add per request (p50 / p95) for each detector
  configuration, and what throughput does one laptop process sustain?

## RQ9: fault injection

{rq9(faults)}
## RQ10: latency and throughput

{rq10(latency)}
## Observations

Generated from the tables above; no interpretation beyond the numbers.

{bullets}

## Threats to validity

{THREATS}"""
    OUT.write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
