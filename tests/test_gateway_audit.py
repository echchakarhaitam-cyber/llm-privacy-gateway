"""The audit log carries counts and decisions, never text: record validation and a canary test."""

import asyncio
import json
import logging
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from privgate.detectors.regex import RegexDetector
from privgate.gateway import Policy
from privgate.gateway.app import GatewayConfig, create_app
from privgate.gateway.audit import AuditLog, AuditRecord
from privgate.gateway.upstream import MockUpstream
from privgate.spans import EntityType, Span

E = EntityType
CARD = "4111 1111 1111 1111"
ALLOWED_KEYS = {"request_id", "ts", "decision", "status", "counts", "detectors", "latency_ms",
                "error_class", "fail_open"}


def record(**kw):
    base = {"request_id": "abc123", "decision": "served", "status": 200}
    base.update(kw)
    return AuditRecord(**base)


def test_record_serialises_only_allowlisted_fields():
    data = record(counts={"EMAIL": 2, "PERSON": 1}, detectors=("crf", "regex"),
                  latency_ms={"detect": 1.23456, "total": 2.0}).to_dict()
    assert set(data) == ALLOWED_KEYS
    assert data["counts"] == {"EMAIL": 2, "PERSON": 1}
    assert data["latency_ms"]["detect"] == 1.235
    assert data["ts"].endswith("+00:00")


@pytest.mark.parametrize("bad", [
    {"decision": "approved"},
    {"request_id": "has space"}, {"request_id": ""}, {"request_id": "x" * 65},
    {"error_class": "contains a space and text"}, {"error_class": "x" * 81},
    {"error_class": "john@example.org"},
    {"counts": {"john@example.org": 1}}, {"counts": {"EMAIL": "1"}}, {"counts": {"EMAIL": True}},
    {"detectors": ("anna lee",)}, {"detectors": ("a@b.org",)},
], ids=lambda d: next(iter(d)) + "-" + str(next(iter(d.values())))[:12])
def test_record_rejects_anything_that_could_carry_text(bad):
    with pytest.raises(ValueError):
        record(**bad).to_dict()


def test_write_never_raises_and_never_logs_the_record(caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    log = AuditLog(tmp_path / "audit.jsonl", keep=True)
    log.write(record(error_class="secret text here"))          # invalid -> dropped, not raised
    log.write(record())
    log.close()
    log.write(record())                                         # closed file: still no raise
    assert "secret text here" not in caplog.text
    assert any("audit write failed" in r.getMessage() for r in caplog.records)
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["decision"] == "served"
    assert len(log.records) == 2          # the post-close write is still kept in memory


def test_audit_log_appends_json_lines_and_creates_directories(tmp_path):
    path = tmp_path / "deep" / "dir" / "audit.jsonl"
    for _ in range(2):
        log = AuditLog(path)
        log.write(record())
        log.close()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


class Marked:
    """Real regex detection plus controllable failures keyed on markers in the text."""

    name = "marked"
    supported_labels = frozenset(E)

    def __init__(self):
        self.regex = RegexDetector()

    def detect(self, text):
        if "RAISE" in text:
            raise RuntimeError(f"cannot process: {text}")
        if "HANG" in text:
            time.sleep(0.2)
        if "MALFORMED" in text:
            return [Span(0, len(text) + 3, E.PERSON)]
        return self.regex.detect(text)


def canary(i):
    return f"canary{i:03d}@leak-test.example"


def scan_for_leaks(canaries, *haystacks):
    return [c for c in canaries for h in haystacks if c in h]


def test_the_scan_itself_detects_a_leak(caplog):
    caplog.set_level(logging.DEBUG)
    logging.getLogger("positive.control").warning("oops %s", canary(0))
    assert scan_for_leaks([canary(0)], caplog.text) == [canary(0)]


def test_no_canary_reaches_the_audit_log_or_any_log_record(caplog, tmp_path):
    caplog.set_level(logging.DEBUG)                       # capture everything, all loggers
    audit = AuditLog(tmp_path / "audit.jsonl", keep=True)

    def upstream_failure(payload):
        if "UPFAIL" in json.dumps(payload):
            return RuntimeError(f"upstream choked on {payload}")
        return None

    upstream = MockUpstream(failure=upstream_failure)
    app = create_app(GatewayConfig(detector_timeout_s=0.05, detector_workers=4, policy=Policy.blocking(
        [E.CREDIT_CARD])), detector=Marked(), upstream=upstream, audit=audit)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
            statuses = []
            for i in range(50):
                tag = canary(i)
                kind = i % 7
                text = f"mail {tag} please"
                body = {"model": "m", "messages": [{"role": "user", "content": text}]}
                if kind == 1:
                    body["messages"][0]["content"] = f"RAISE {tag}"
                elif kind == 2:
                    body["messages"][0]["content"] = f"MALFORMED {tag}"
                elif kind == 3:
                    body["messages"][0]["content"] = f"HANG {tag}"
                elif kind == 4:
                    body["messages"][0]["content"] = f"{tag} {CARD}"
                elif kind == 5:
                    body["messages"][0]["content"] = f"UPFAIL {tag}"
                elif kind == 6:
                    body[tag] = "unsupported field named after a canary"
                statuses.append((await c.post("/v1/chat/completions", json=body)).status_code)
            return statuses
    statuses = asyncio.run(go())
    assert set(statuses) == {200, 400, 403, 502, 503}

    canaries = [canary(i) for i in range(50)] + [CARD, CARD.replace(" ", "")]
    audit_file = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert scan_for_leaks(canaries, audit_file, json.dumps(audit.records), caplog.text) == []
    assert len(audit.records) == 50 and len(audit_file.splitlines()) == 50
    assert all(set(r) == ALLOWED_KEYS for r in audit.records)
    assert {r["decision"] for r in audit.records} == {
        "served", "screening_unavailable", "blocked_policy", "upstream_error",
        "rejected_invalid"}
    # what IS recorded: counts and error classes
    served = [r for r in audit.records if r["decision"] == "served"]
    assert served and all(r["counts"] == {"EMAIL": 1} for r in served)
    classes = {r["error_class"] for r in audit.records if r["error_class"]}
    assert {"RuntimeError", "detector_timeout", "PolicyViolation", "unknown_field",
            "SpanOutOfBounds"} <= classes


def test_audit_records_stage_latencies_and_detector_names():
    audit = AuditLog(keep=True)
    app = create_app(GatewayConfig(), detector=RegexDetector(), upstream=MockUpstream(),
                     audit=audit)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi a@b.org"}]})
    r = asyncio.run(go())
    (rec,) = audit.records
    assert rec["request_id"] == r.headers["X-Privgate-Request-Id"]
    assert set(rec["latency_ms"]) == {"parse", "detect", "mask", "upstream", "unmask", "total"}
    assert rec["detectors"] == ["regex"] and rec["counts"] == {"EMAIL": 1}
    assert rec["status"] == 200 and rec["error_class"] is None and rec["fail_open"] is False
