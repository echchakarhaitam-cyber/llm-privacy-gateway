"""Fault injection: the wrappers, the experiment's plan, and fail-closed vs fail-open."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from privgate.detectors.regex import RegexDetector
from privgate.gateway.app import ENV_ALLOW_FAIL_OPEN, GatewayConfig, create_app
from privgate.gateway.audit import AuditLog
from privgate.gateway.faults import (
    FAULT_KINDS,
    FaultInjector,
    FaultyDetector,
    ReplayDetector,
    faulty_upstream,
)
from privgate.spans import EntityType, Span

EMAIL = "anna.lee@example.org"


def load_script():
    path = Path(__file__).parent.parent / "scripts" / "run_gateway_faults.py"
    spec = importlib.util.spec_from_file_location("run_gateway_faults", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_gateway_faults"] = module
    spec.loader.exec_module(module)
    return module


def post(app, text):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://g") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": text}]})
    return asyncio.run(go())


def build(fail_open, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_FAIL_OPEN, "1")
    injector = FaultInjector()
    upstream = faulty_upstream(injector)
    detector = FaultyDetector(RegexDetector(), injector, hang_s=0.2)
    app = create_app(GatewayConfig(fail_open=fail_open, detector_timeout_s=0.05,
                                   detector_workers=4),
                     detector=detector, upstream=upstream, audit=AuditLog(keep=True))
    return app, injector, upstream


@pytest.mark.parametrize("fault", ["exception", "hang", "malformed"])
def test_fail_closed_never_forwards_when_the_detector_is_faulty(fault, monkeypatch):
    app, injector, upstream = build(False, monkeypatch)
    injector.current = fault
    for i in range(6):                      # covers every malformed variant
        assert post(app, f"mail {EMAIL} {i}").status_code == 503
    assert upstream.calls == 0


@pytest.mark.parametrize("fault", ["exception", "hang", "malformed"])
def test_fail_open_leaks_exactly_those_requests(fault, monkeypatch):
    app, injector, upstream = build(True, monkeypatch)
    injector.current = fault
    for i in range(3):
        assert post(app, f"mail {EMAIL} {i}").status_code == 200
    assert all(EMAIL in json.dumps(p) for p in upstream.received) and upstream.calls == 3


def test_upstream_fault_is_502_in_both_modes_and_never_leaks(monkeypatch):
    for fail_open in (False, True):
        app, injector, upstream = build(fail_open, monkeypatch)
        injector.current = "upstream_5xx"
        assert post(app, f"mail {EMAIL}").status_code == 502
        assert EMAIL not in json.dumps(upstream.received)        # screening had worked


def test_no_fault_means_normal_masking(monkeypatch):
    app, _, upstream = build(False, monkeypatch)
    assert post(app, f"mail {EMAIL}").status_code == 200
    assert EMAIL not in json.dumps(upstream.received)


def test_replay_detector_replays_and_unknown_text_fails_closed():
    spans = [Span(5, 5 + len(EMAIL), EntityType.EMAIL)]
    replay = ReplayDetector({f"mail {EMAIL}": spans})
    assert replay.detect(f"mail {EMAIL}") == spans
    upstream = faulty_upstream(FaultInjector())
    app = create_app(GatewayConfig(), detector=replay, upstream=upstream,
                     audit=AuditLog(keep=True))
    assert post(app, f"mail {EMAIL}").status_code == 200
    assert post(app, "never seen before").status_code == 503
    assert upstream.calls == 1


def test_fault_plan_is_deterministic_and_matches_q():
    script = load_script()
    a = script.fault_plan("exception", 0.1, 5000)
    assert a == script.fault_plan("exception", 0.1, 5000)
    assert a != script.fault_plan("exception", 0.2, 5000)
    assert 0.08 < sum(f is not None for f in a) / 5000 < 0.12
    assert set(filter(None, a)) == {"exception"}
    assert not any(script.fault_plan("hang", 0.0, 500))
    mixed = script.fault_plan("mixed", 0.5, 4000)
    assert set(filter(None, mixed)) == set(FAULT_KINDS)


@pytest.mark.slow
def test_experiment_end_to_end_fail_closed_has_no_fault_induced_leaks(tmp_path):
    data = Path("data/processed/nemotron_dev_2k.jsonl")
    run_dir = Path("results/m4/nemotron_dev_2k/crf+regex+presidio")
    if not data.exists() or not run_dir.exists():
        pytest.skip("needs the dev set and the M4 predictions")
    script = load_script()
    out = tmp_path / "faults.json"
    assert script.main(["--n", "60", "--verify-sample", "0", "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    closed = [r for r in result["rows"] if r["mode"] == "fail_closed"]
    opened = [r for r in result["rows"] if r["mode"] == "fail_open"]
    assert closed and opened
    assert all(r["fault_leaked_requests"] == 0 and r["unscreened_forwarded"] == 0 for r in closed)
    assert any(r["fault_leaked_requests"] > 0 for r in opened)
