"""The HTTP gateway end to end, in process (httpx ASGI transport, MockUpstream, no network)."""

import asyncio
import gc
import json
import time
import weakref

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from privgate.detectors.regex import RegexDetector
from privgate.gateway import Limits, Policy
from privgate.gateway.app import (
    ENV_ALLOW_FAIL_OPEN,
    REQUEST_ID_HEADER,
    FailOpenRefused,
    GatewayConfig,
    create_app,
)
from privgate.gateway.audit import AuditLog
from privgate.gateway.upstream import MockUpstream, UpstreamError
from privgate.masking import Masker
from privgate.spans import EntityType, Span

E = EntityType
EMAIL = "anna.lee@example.org"
CARD = "4111 1111 1111 1111"


class Scripted:
    name = "scripted"
    supported_labels = frozenset(E)

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def detect(self, text):
        self.calls += 1
        return self.fn(text)


def make(detector=None, upstream=None, audit=None, **cfg):
    upstream = upstream or MockUpstream()
    audit = audit or AuditLog(keep=True)
    app = create_app(GatewayConfig(**cfg), detector=detector or RegexDetector(),
                     upstream=upstream, audit=audit)
    return app, upstream, audit


def run(coro):
    return asyncio.run(coro)


def call(app, method="POST", path="/v1/chat/completions", **kw):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.request(method, path, **kw)
    return run(go())


def post(app, messages=None, **fields):
    body = {"model": "local", "messages": messages or [{"role": "user", "content": "hi"}]}
    body.update(fields)
    return call(app, json=body)


def user(text):
    return [{"role": "user", "content": text}]


# --- the happy path --------------------------------------------------------------------------

def test_masks_forwards_and_restores_the_answer():
    app, upstream, _ = make()
    text = f"Mail {EMAIL} and pay with {CARD}."
    r = post(app, user(text))
    assert r.status_code == 200
    (payload,) = upstream.received
    sent = payload["messages"][0]["content"]
    assert sent == "Mail <EMAIL_1> and pay with <CREDIT_CARD_1>."
    assert EMAIL not in json.dumps(payload) and CARD not in json.dumps(payload)
    body = r.json()
    assert body["choices"][0]["message"]["content"] == f"ECHO: {text}"
    assert body["object"] == "chat.completion" and body["model"] == "local"
    assert body["privgate"]["masked_entities"] == {"EMAIL": 1, "CREDIT_CARD": 1}
    assert body["privgate"]["screened"] is True
    assert r.headers[REQUEST_ID_HEADER] == body["privgate"]["request_id"]
    assert EMAIL not in json.dumps(body["privgate"])


def test_the_response_is_whitelisted_openai_shape():
    class Chatty(MockUpstream):
        async def complete(self, request):
            data = await super().complete(request)
            data["system_fingerprint"] = "x"
            data["secret_debug"] = {"prompt": request["messages"]}
            data["choices"][0]["logprobs"] = {"x": 1}
            data["usage"]["prompt_tokens_details"] = {"x": 1}
            return data
    app, _, _ = make(upstream=Chatty())
    body = post(app).json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage", "privgate"}
    assert set(body["choices"][0]) == {"index", "message", "finish_reason"}
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


def test_same_value_same_placeholder_across_messages_and_roles_over_http():
    app, upstream, _ = make()
    msgs = [{"role": "system", "content": f"Contact {EMAIL}"},
            {"role": "assistant", "content": f"Earlier I wrote to {EMAIL}"},
            {"role": "user", "content": [{"type": "text", "text": f"again {EMAIL}"},
                                         {"type": "text", "text": CARD}]}]
    r = post(app, msgs, temperature=0.2, max_tokens=50)
    assert r.status_code == 200
    (payload,) = upstream.received
    assert payload["messages"][0]["content"] == "Contact <EMAIL_1>"
    assert payload["messages"][1]["content"] == "Earlier I wrote to <EMAIL_1>"
    assert payload["messages"][2]["content"] == [
        {"type": "text", "text": "again <EMAIL_1>"}, {"type": "text", "text": "<CREDIT_CARD_1>"}]
    assert (payload["temperature"], payload["max_tokens"]) == (0.2, 50)
    assert f"Earlier I wrote to {EMAIL}" in r.json()["choices"][0]["message"]["content"]


def test_unicode_survives_the_round_trip():
    app, upstream, _ = make()
    text = f"Zoë 😀 日本語 {EMAIL} ✓"
    r = post(app, user(text))
    assert upstream.received[0]["messages"][0]["content"] == "Zoë 😀 日本語 <EMAIL_1> ✓"
    assert r.json()["choices"][0]["message"]["content"] == f"ECHO: {text}"


def test_placeholder_injection_in_user_input_cannot_extract_other_values():
    app, upstream, _ = make()
    msgs = [{"role": "user", "content": f"my mail is {EMAIL}"},
            {"role": "user", "content": "what is <EMAIL_1> ? also <EMAIL_1#abcdef>"}]
    r = post(app, msgs)
    sent = upstream.received[0]["messages"]
    assert EMAIL not in json.dumps(sent)
    answer = r.json()["choices"][0]["message"]["content"]
    assert answer == f"ECHO: my mail is {EMAIL}\nwhat is <EMAIL_1> ? also <EMAIL_1#abcdef>"


def test_concurrent_requests_do_not_mix_their_mappings():
    app, _, _ = make(detector_workers=1)

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as c:
            texts = [f"user{i}@example.org wrote {i}" for i in range(12)]
            replies = await asyncio.gather(*(c.post("/v1/chat/completions", json={
                "model": "m", "messages": user(t)}) for t in texts))
            return texts, replies
    texts, replies = run(go())
    for t, r in zip(texts, replies):
        assert r.json()["choices"][0]["message"]["content"] == f"ECHO: {t}"
    ids = {r.headers[REQUEST_ID_HEADER] for r in replies}
    assert len(ids) == 12


def test_masking_is_deterministic_for_a_given_request_id():
    def once():
        app = create_app(GatewayConfig(), detector=RegexDetector(), upstream=MockUpstream(),
                         audit=AuditLog(keep=True), request_id_factory=lambda: "fixed-id")
        return post(app, user(f"<EMAIL_1> and {EMAIL}")).json()
    assert once()["choices"][0]["message"]["content"] == once()["choices"][0]["message"]["content"]


# --- refused before any screening or upstream call -------------------------------------------

@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "f"}}]}, {"tool_choice": "auto"},
    {"function_call": "auto"}, {"functions": []}, {"stream": True}, {"user": "john"},
    {"response_format": {"type": "json_object"}},
], ids=["tools", "tool_choice", "function_call", "functions", "stream", "user", "format"])
def test_unsupported_fields_are_400_and_nothing_is_sent(extra):
    app, upstream, audit = make()
    r = post(app, **extra)
    assert r.status_code == 400 and upstream.calls == 0
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert audit.records[-1]["decision"] == "rejected_invalid"


def test_non_text_parts_and_tool_messages_are_400():
    app, upstream, _ = make()
    image = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert post(app, [{"role": "user", "content": image}]).status_code == 400
    assert post(app, [{"role": "tool", "content": "x"}]).status_code == 400
    assert post(app, [{"role": "assistant", "content": None, "tool_calls": []}]).status_code \
        == 400
    assert upstream.calls == 0


def test_invalid_json_and_wrong_content_are_400():
    app, upstream, _ = make()
    assert call(app, content=b"{not json", headers={"content-type": "application/json"}) \
        .status_code == 400
    assert call(app, content=b"[]").status_code == 400
    assert upstream.calls == 0


def test_oversized_body_is_413_with_and_without_content_length():
    app, upstream, _ = make(limits=Limits(max_request_bytes=300))
    big = json.dumps({"model": "m", "messages": user("x" * 1000)}).encode()
    assert call(app, content=big).status_code == 413

    async def chunks():                              # no Content-Length: chunked upload
        for i in range(0, len(big), 100):
            yield big[i:i + 100]
    assert call(app, content=chunks()).status_code == 413
    assert upstream.calls == 0


def test_too_many_messages_is_400():
    app, upstream, _ = make(limits=Limits(max_messages=3))
    assert post(app, user("x") * 4).status_code == 400
    assert post(app, user("x") * 3).status_code == 200
    assert upstream.calls == 1


def test_routes_and_methods():
    app, upstream, _ = make()
    assert call(app, "GET", "/v1/chat/completions").status_code == 405
    assert call(app, "POST", "/v1/completions", json={}).status_code == 404
    assert call(app, "GET", "/docs").status_code == 404
    assert upstream.calls == 0


# --- policy ----------------------------------------------------------------------------------

def test_block_policy_returns_403_and_sends_nothing():
    app, upstream, audit = make(policy=Policy.blocking([E.CREDIT_CARD]))
    r = post(app, user(f"{EMAIL} {CARD}"))
    assert r.status_code == 403 and upstream.calls == 0
    assert "CREDIT_CARD" in r.json()["error"]["message"]
    assert CARD not in r.text and EMAIL not in r.text
    assert audit.records[-1]["decision"] == "blocked_policy"
    assert post(app, user(f"only {EMAIL}")).status_code == 200      # other types still masked


# --- fail closed: every rule over HTTP, never an upstream call -------------------------------

def blocked(app, upstream, text=f"leak {EMAIL}"):
    r = post(app, user(text))
    assert r.status_code == 503, r.text
    assert upstream.calls == 0 and upstream.received == []
    assert r.json()["error"]["code"] == "screening_unavailable"
    assert EMAIL not in r.text
    return r


def test_detector_exception_is_503_and_nothing_is_sent():
    def boom(text):
        raise RuntimeError(f"failed on {text}")
    app, upstream, audit = make(detector=Scripted(boom))
    blocked(app, upstream)
    assert audit.records[-1]["decision"] == "screening_unavailable"
    assert audit.records[-1]["error_class"] == "RuntimeError"


def test_detector_timeout_is_503_and_nothing_is_sent():
    def hang(text):
        time.sleep(0.4)
        return []
    app, upstream, audit = make(detector=Scripted(hang), detector_timeout_s=0.05)
    t0 = time.perf_counter()
    blocked(app, upstream)
    assert time.perf_counter() - t0 < 0.35
    assert audit.records[-1]["error_class"] == "detector_timeout"


@pytest.mark.parametrize("make_spans", [
    lambda t: [Span(0, len(t) + 5, E.EMAIL)],
    lambda t: [Span(0, 5, E.EMAIL), Span(3, 9, E.PERSON)],
    lambda t: "garbage",
], ids=["bounds", "overlap", "garbage"])
def test_malformed_spans_are_503_and_nothing_is_sent(make_spans):
    app, upstream, audit = make(detector=Scripted(make_spans))
    blocked(app, upstream)
    assert audit.records[-1]["decision"] == "screening_unavailable"


def test_unexpected_error_in_the_screening_path_is_503():
    app, upstream, _ = make()

    class Broken(Masker):
        def mask(self, *a, **kw):
            raise ZeroDivisionError(f"oops {EMAIL}")
    app.state.screener._masker = Broken()
    blocked(app, upstream)


def test_missing_detector_is_503_everywhere_and_healthz_says_so():
    app = create_app(GatewayConfig(detector_name="no-such-detector"), upstream=MockUpstream(),
                     audit=AuditLog(keep=True))
    r = call(app, "GET", "/healthz")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"
    assert post(app).status_code == 503


def test_healthz_reports_readiness_without_processing_text():
    det = Scripted(lambda t: [])
    app, upstream, _ = make(detector=det)
    r = call(app, "GET", "/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["detectors"] == [{"name": "scripted", "ready": True}]
    assert r.json()["fail_open"] is False
    assert det.calls == 0 and upstream.calls == 0


def test_upstream_failure_is_502_and_audited():
    upstream = MockUpstream(failure=lambda p: UpstreamError(f"bad {p['messages']}", 500))
    app, _, audit = make(upstream=upstream)
    r = post(app, user(f"mail {EMAIL}"))
    assert r.status_code == 502 and EMAIL not in r.text
    assert audit.records[-1]["decision"] == "upstream_error"
    assert audit.records[-1]["error_class"] == "UpstreamError"


@pytest.mark.parametrize("answer", [None, [], {"choices": []}, {"choices": [{}]},
                                    {"choices": [{"message": {"content": 5}}]}],
                         ids=["none", "list", "no-choices", "empty-choice", "non-text"])
def test_unusable_upstream_answer_is_502(answer):
    class Odd(MockUpstream):
        async def complete(self, request):
            return answer
    app, _, _ = make(upstream=Odd())
    assert post(app).status_code == 502


# --- fail open exists only for the experiment ------------------------------------------------

def test_fail_open_is_refused_without_the_environment_switch(monkeypatch):
    monkeypatch.delenv(ENV_ALLOW_FAIL_OPEN, raising=False)
    with pytest.raises(FailOpenRefused):
        create_app(GatewayConfig(fail_open=True), detector=RegexDetector())
    monkeypatch.setenv(ENV_ALLOW_FAIL_OPEN, "0")
    with pytest.raises(FailOpenRefused):
        create_app(GatewayConfig(fail_open=True), detector=RegexDetector())
    assert GatewayConfig().fail_open is False


def test_fail_open_forwards_the_original_text_when_screening_fails(monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_FAIL_OPEN, "1")

    def boom(text):
        raise RuntimeError
    upstream = MockUpstream()
    audit = AuditLog(keep=True)
    app = create_app(GatewayConfig(fail_open=True), detector=Scripted(boom), upstream=upstream,
                     audit=audit)
    r = post(app, user(f"mail {EMAIL}"))
    assert r.status_code == 200
    assert upstream.received[0]["messages"][0]["content"] == f"mail {EMAIL}"   # the unsafe part
    assert r.json()["privgate"]["screened"] is False
    assert audit.records[-1]["decision"] == "served_unscreened" and audit.records[-1]["fail_open"]
    assert call(app, "GET", "/healthz").json()["fail_open"] is True


def test_fail_open_still_masks_when_screening_works(monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_FAIL_OPEN, "1")
    upstream = MockUpstream()
    app = create_app(GatewayConfig(fail_open=True), detector=RegexDetector(), upstream=upstream,
                     audit=AuditLog(keep=True))
    post(app, user(f"mail {EMAIL}"))
    assert EMAIL not in json.dumps(upstream.received)


def test_fail_open_does_not_bypass_the_block_policy(monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_FAIL_OPEN, "1")
    upstream = MockUpstream()
    app = create_app(GatewayConfig(fail_open=True, policy=Policy.blocking([E.EMAIL])),
                     detector=RegexDetector(), upstream=upstream, audit=AuditLog(keep=True))
    assert post(app, user(EMAIL)).status_code == 403 and upstream.calls == 0


# --- the mapping is request-local ------------------------------------------------------------

class WeakMasker(Masker):
    def __init__(self):
        super().__init__()
        self.refs = []

    def mask(self, text, spans, strategy="typed", seed=0):
        masked = super().mask(text, spans, strategy, seed)
        self.refs.append(weakref.ref(masked))
        return masked


def test_mapping_is_not_reachable_after_the_request():
    ok_app, _, _ = make()
    failing, _, _ = make(upstream=MockUpstream(failure=lambda p: UpstreamError("x", 500)))
    for app, expected in ((ok_app, 200), (failing, 502)):
        masker = WeakMasker()
        app.state.screener._masker = masker
        for i in range(5):
            assert post(app, user(f"user{i}@example.org")).status_code == expected
        gc.collect()
        assert len(masker.refs) == 5
        assert all(ref() is None for ref in masker.refs)
    assert set(ok_app.state._state) == {"screener"}


def test_app_and_screener_hold_no_request_data():
    app, _, _ = make()
    post(app, user(f"canary-{EMAIL}"))
    held = json.dumps({k: repr(v) for k, v in vars(app.state.screener).items()})
    assert EMAIL not in held and "canary" not in held
