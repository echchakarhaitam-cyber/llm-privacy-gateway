"""The latency harness: statistics and the in-process path, on a handful of requests.

No timing claim is made here; the real measurement is scripts/run_gateway_latency.py, which
refuses to run on a busy machine.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")


def load_script():
    path = Path(__file__).parent.parent / "scripts" / "run_gateway_latency.py"
    spec = importlib.util.spec_from_file_location("run_gateway_latency", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_gateway_latency"] = module
    spec.loader.exec_module(module)
    return module


def bodies(n):
    return [{"model": "m", "messages": [{"role": "user", "content": f"mail a{i}@b.org"}]}
            for i in range(n)]


def test_stats_percentiles_and_throughput():
    script = load_script()
    out = script.stats([float(i) for i in range(1, 101)], wall_s=2.0, errors=1)
    assert out["requests"] == 100 and out["errors"] == 1
    assert out["p50_ms"] <= out["p95_ms"] <= out["p99_ms"] <= 100
    assert out["requests_per_s"] == 50 and out["mean_ms"] == 50.5


@pytest.mark.parametrize("config", ["none", "regex"])
def test_asgi_harness_runs_both_concurrencies_without_errors(config):
    script = load_script()
    rows = asyncio.run(script.measure_asgi(config, bodies(2), bodies(6)))
    assert [r["concurrency"] for r in rows] == [1, 4]
    assert all(r["errors"] == 0 and r["requests"] == 6 for r in rows)
    assert set(rows[0]["gateway_stage_mean_ms"]) >= {"detect", "mask", "total"}


def test_null_detector_finds_nothing_but_the_pipeline_still_runs():
    script = load_script()
    app, audit = script.make_app("none")
    assert script.NullDetector().detect("a@b.org") == []
    assert app.state.screener.detector.name == "none"
    assert audit.records == []


def test_idle_check_reports_the_documented_fields():
    pytest.importorskip("psutil")
    script = load_script()
    check = script.idle_check()
    assert {"total_cpu_percent_5s", "idle", "heavy_python_pids", "other_python_processes",
            "logical_cpus"} <= set(check)
    assert check["idle"] == (check["total_cpu_percent_5s"] <= script.IDLE_CPU_MAX
                             and not check["heavy_python_pids"])
