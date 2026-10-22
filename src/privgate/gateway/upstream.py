"""Where a screened request goes: the Upstream protocol and its implementations.

* :class:`MockUpstream` echoes the masked prompt into its answer and records every payload it
  receives. Tests and experiments use it, so no experiment touches a network. Failures and
  latency are injectable.
* :class:`OpenAICompatibleUpstream` talks to a LOCAL OpenAI-compatible server (llama.cpp,
  vLLM, Ollama, ...). A non-local ``base_url`` is refused unless the caller passes
  ``allow_non_local=True``. No API keys are handled in this version: the request carries no
  credentials and the gateway never reads any.

Error messages never contain the response body or the request: they can end up in logs.
"""

from __future__ import annotations

import asyncio
import copy
import ipaddress
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlsplit

COMPLETIONS_PATH = "/chat/completions"


class UpstreamError(Exception):
    """The upstream failed or answered with something unusable (the gateway returns 502)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NonLocalUpstreamError(ValueError):
    """``base_url`` is not a localhost address and ``allow_non_local`` was not set."""


class Upstream(Protocol):
    async def complete(self, request: dict) -> dict:
        """Send an already-screened chat request, return an OpenAI-style response dict."""
        ...


class MockUpstream:
    """An in-process upstream. ``received`` keeps a deep copy of every payload it was sent.

    ``failure(payload)`` may return an exception to raise instead of answering.
    ``record=False`` skips the copy (latency runs: the copy is not part of the gateway).
    """

    def __init__(self, latency_s: float = 0.0,
                 failure: Callable[[dict], Exception | None] | None = None,
                 record: bool = True):
        self.latency_s = latency_s
        self.failure = failure
        self.record = record
        self.received: list[dict] = []
        self.calls = 0

    async def complete(self, request: dict) -> dict:
        self.calls += 1
        if self.record:
            self.received.append(copy.deepcopy(request))
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.failure is not None and (exc := self.failure(request)) is not None:
            raise exc
        parts = []
        for message in request["messages"]:
            content = message["content"]
            parts.extend([content] if isinstance(content, str) else [p["text"] for p in content])
        answer = "ECHO: " + "\n".join(parts)
        prompt_tokens = sum(len(p) for p in parts) // 4
        return {
            "id": f"mock-{self.calls}", "object": "chat.completion", "created": 0,
            "model": request["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(answer) // 4,
                      "total_tokens": prompt_tokens + len(answer) // 4},
        }


def is_local_url(url: str) -> bool:
    """True for http(s) URLs whose host is ``localhost`` or a loopback IP, without userinfo."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if parts.scheme not in ("http", "https") or not host:
            return False
        if parts.username or parts.password:
            return False
        _ = parts.port                  # raises ValueError on a malformed port
    except ValueError:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleUpstream:
    def __init__(self, base_url: str, *, allow_non_local: bool = False,
                 timeout_s: float = 120.0, transport=None):
        if not allow_non_local and not is_local_url(base_url):
            raise NonLocalUpstreamError(
                "upstream base_url must be localhost or a loopback address; pass "
                "allow_non_local=True to override (data leaves this machine!)")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout_s, transport=self._transport,
                                             follow_redirects=False, trust_env=False)
        return self._client

    async def complete(self, request: dict) -> dict:
        import httpx

        try:
            response = await self._http().post(self.base_url + COMPLETIONS_PATH, json=request)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream request failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise UpstreamError(f"upstream answered {response.status_code}",
                                response.status_code)
        try:
            data = response.json()
        except ValueError:
            raise UpstreamError("upstream answer is not JSON") from None
        if not isinstance(data, dict):
            raise UpstreamError("upstream answer is not a JSON object")
        return data

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
