"""Upstream implementations: the mock, and the local-only OpenAI-compatible client."""

import asyncio
import json
import time

import pytest

httpx = pytest.importorskip("httpx")

from privgate.gateway.upstream import (
    MockUpstream,
    NonLocalUpstreamError,
    OpenAICompatibleUpstream,
    UpstreamError,
    is_local_url,
)

PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hello <PERSON_1>"}]}


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("url", [
    "http://localhost:8000/v1", "http://LOCALHOST/v1", "https://localhost:8443",
    "http://127.0.0.1:11434/v1", "http://127.1.2.3:80", "http://[::1]:8080/v1",
])
def test_local_urls_are_accepted(url):
    assert is_local_url(url)


@pytest.mark.parametrize("url", [
    "http://api.openai.com/v1", "https://example.org", "http://10.0.0.5:8000",
    "http://192.168.1.10", "http://0.0.0.0:8000", "http://localhost.evil.com/v1",
    "http://127.0.0.1.evil.com/v1", "http://evil.com/?h=localhost",
    "http://user:pw@localhost/v1", "ftp://localhost/v1", "file:///etc/passwd",
    "localhost:8000", "//localhost", "", "http://", "http://localhost:99999/v1",
    "http://[::ffff:8.8.8.8]/v1",
])
def test_non_local_urls_are_refused(url):
    assert not is_local_url(url)


def test_constructor_refuses_non_local_base_url_unless_explicitly_allowed():
    with pytest.raises(NonLocalUpstreamError):
        OpenAICompatibleUpstream("https://api.example.com/v1")
    allowed = OpenAICompatibleUpstream("https://api.example.com/v1/", allow_non_local=True)
    assert allowed.base_url == "https://api.example.com/v1"
    assert OpenAICompatibleUpstream("http://127.0.0.1:8000/v1").base_url \
        == "http://127.0.0.1:8000/v1"


def test_mock_echoes_the_masked_prompt_and_records_a_copy():
    mock = MockUpstream()
    payload = {"model": "m", "messages": [{"role": "system", "content": "be nice"},
                                          {"role": "user", "content": [
                                              {"type": "text", "text": "hi <PERSON_1>"}]}]}
    answer = run(mock.complete(payload))
    assert answer["choices"][0]["message"]["content"] == "ECHO: be nice\nhi <PERSON_1>"
    assert answer["model"] == "m" and mock.calls == 1
    payload["messages"][0]["content"] = "mutated afterwards"
    assert mock.received[0]["messages"][0]["content"] == "be nice"


def test_mock_failure_hook_latency_and_no_record():
    mock = MockUpstream(failure=lambda p: UpstreamError("boom", 503), record=False)
    with pytest.raises(UpstreamError):
        run(mock.complete(PAYLOAD))
    assert mock.received == [] and mock.calls == 1
    slow = MockUpstream(latency_s=0.05)
    t0 = time.perf_counter()
    run(slow.complete(PAYLOAD))
    assert time.perf_counter() - t0 >= 0.045
    passing = MockUpstream(failure=lambda p: None)
    run(passing.complete(PAYLOAD))
    assert passing.calls == 1


def client_for(handler, **kw):
    return OpenAICompatibleUpstream("http://localhost:9999/v1",
                                    transport=httpx.MockTransport(handler), **kw)


def test_client_posts_the_payload_to_the_local_server_without_credentials():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["headers"] = {k.lower() for k in request.headers}
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    answer = run(client_for(handler).complete(PAYLOAD))
    assert answer["choices"][0]["message"]["content"] == "ok"
    assert seen["url"] == "http://localhost:9999/v1/chat/completions"
    assert seen["body"] == PAYLOAD
    assert "authorization" not in seen["headers"]


def test_client_maps_failures_to_upstream_error_without_leaking_content():
    secret = "Alice Wonderland"

    def fail(handler):
        with pytest.raises(UpstreamError) as info:
            run(client_for(handler).complete(PAYLOAD))
        assert secret not in str(info.value)
        return info.value

    assert fail(lambda r: httpx.Response(500, text=f"oops {secret}")).status == 500
    fail(lambda r: httpx.Response(200, text=f"<html>{secret}</html>"))
    fail(lambda r: httpx.Response(200, json=[secret]))

    def refused(request):
        raise httpx.ConnectError(f"cannot connect, sent {secret}")

    assert "ConnectError" in str(fail(refused))


def test_client_does_not_follow_redirects():
    hosts = []

    def redirect(request):
        hosts.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://evil.example/steal"})

    with pytest.raises(UpstreamError):
        run(client_for(redirect).complete(PAYLOAD))
    assert hosts == ["localhost"]               # evil.example was never contacted


def test_client_close_is_idempotent_and_the_client_reusable():
    upstream = client_for(lambda r: httpx.Response(200, json={"choices": []}))

    async def go():
        await upstream.aclose()
        await upstream.complete(PAYLOAD)
        await upstream.complete(PAYLOAD)
        await upstream.aclose()
        await upstream.aclose()

    run(go())
