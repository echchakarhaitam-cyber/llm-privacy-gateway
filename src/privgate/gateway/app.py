"""The HTTP gateway: ``POST /v1/chat/completions`` and ``GET /healthz``.

One request, in order: read (size limit) -> validate (whitelist) -> screen every string of
every message -> mask (typed, one mapping per request) -> forward -> unmask the answer ->
respond. Fail closed: the single place that calls the upstream is :func:`_send`, and it is
reached only with a :class:`~privgate.gateway.screening.Screened` request -- or, in the
unsafe experiment-only fail-open mode, with the original request after a screening failure.

Status codes: 400 invalid request, 403 blocked by policy, 413 too large, 502 upstream
failed or answered nonsense, 503 screening unavailable (nothing was sent upstream).

The mapping (placeholder -> original) is a local variable of the request handler. It is
never stored on the app, never logged and never written to the audit log.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from privgate.gateway.audit import AuditLog, AuditRecord
from privgate.gateway.detectors import build_detector, member_names
from privgate.gateway.policy import Policy
from privgate.gateway.schema import Limits, RequestRejected, parse_chat_request
from privgate.gateway.screening import (
    DEFAULT_TIMEOUT_S,
    PolicyViolation,
    Screener,
    ScreeningUnavailable,
)
from privgate.gateway.upstream import MockUpstream, Upstream
from privgate.masking import unmask

logger = logging.getLogger("privgate.gateway")

ENV_ALLOW_FAIL_OPEN = "PRIVGATE_ALLOW_FAIL_OPEN_FOR_EXPERIMENTS"
REQUEST_ID_HEADER = "X-Privgate-Request-Id"
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._:/@+-]{1,100}")


class FailOpenRefused(RuntimeError):
    """``fail_open`` was requested without the explicit experiment-only environment switch."""


@dataclass(frozen=True)
class GatewayConfig:
    policy: Policy = field(default_factory=Policy)
    limits: Limits = field(default_factory=Limits)
    detector_name: str = "crf+regex"        # used only when create_app gets no detector
    detector_timeout_s: float = DEFAULT_TIMEOUT_S   # screening budget of ONE request
    detector_workers: int = 1
    seed: int = 0                            # base of the per-request masking seed
    audit_path: str | None = None
    # UNSAFE: forward the ORIGINAL text when screening fails. Exists only so that RQ9 can
    # measure what fail-closed prevents; create_app refuses it without ENV_ALLOW_FAIL_OPEN=1.
    fail_open: bool = False


def request_seed(base_seed: int, request_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{request_id}".encode()).hexdigest()
    return int(digest[:15], 16)


def _error_body(code: str, message: str, kind: str) -> dict:
    return {"error": {"message": message, "type": kind, "code": code}}


async def _read_body(request: Request, limits: Limits) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > limits.max_request_bytes
        except ValueError:
            raise RequestRejected(400, "invalid_header", "invalid Content-Length") from None
        if too_big:
            raise RequestRejected(413, "request_too_large",
                                  f"request body exceeds {limits.max_request_bytes} bytes")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limits.max_request_bytes:
            raise RequestRejected(413, "request_too_large",
                                  f"request body exceeds {limits.max_request_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def build_response(raw: object, mapping, request_model: str, request_id: str) -> dict:
    """OpenAI-style response from the upstream answer, whitelisted and unmasked.

    Raises on an upstream answer that lacks the expected structure.
    """
    if not isinstance(raw, dict):
        raise TypeError("upstream answer is not an object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("upstream answer has no choices")
    out_choices = []
    for i, choice in enumerate(choices):
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("upstream message content is not text")
        finish = choice.get("finish_reason")
        out_choices.append({
            "index": i,
            "message": {"role": "assistant", "content": unmask(content, mapping)},
            "finish_reason": finish if isinstance(finish, str) and len(finish) < 40 else None,
        })
    answer_id, model = raw.get("id"), raw.get("model")
    created = raw.get("created")
    response = {
        "id": answer_id if isinstance(answer_id, str) and _SAFE_TOKEN.fullmatch(answer_id)
        else f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created if isinstance(created, int) and not isinstance(created, bool) else 0,
        "model": model if isinstance(model, str) and _SAFE_TOKEN.fullmatch(model)
        else request_model,
        "choices": out_choices,
    }
    usage = raw.get("usage")
    if isinstance(usage, dict):
        clean = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                 if isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)}
        if clean:
            response["usage"] = clean
    return response


def create_app(config: GatewayConfig | None = None, *, detector=None,
               upstream: Upstream | None = None, audit: AuditLog | None = None,
               request_id_factory: Callable[[], str] | None = None) -> FastAPI:
    """Build the gateway app.

    ``detector`` defaults to ``build_detector(config.detector_name)``; if that fails the app
    still starts but is NOT READY: ``/healthz`` answers 503 and every chat request is blocked.
    ``upstream`` defaults to an in-process :class:`MockUpstream` (nothing leaves the process).
    """
    cfg = config or GatewayConfig()
    if cfg.fail_open:
        if os.environ.get(ENV_ALLOW_FAIL_OPEN) != "1":
            raise FailOpenRefused(
                f"fail_open forwards UNSCREENED text when screening fails. It is only for the "
                f"RQ9 experiment; set {ENV_ALLOW_FAIL_OPEN}=1 to enable it.")
        logger.warning("UNSAFE fail-open mode is enabled (experiment only)")
    if detector is None:
        try:
            detector = build_detector(cfg.detector_name)
        except Exception as exc:        # noqa: BLE001 - fail closed: start not-ready instead
            logger.error("detector %r failed to load: %s", cfg.detector_name,
                         type(exc).__name__)
    upstream = upstream if upstream is not None else MockUpstream()
    audit = audit if audit is not None else AuditLog(cfg.audit_path)
    new_id = request_id_factory or (lambda: uuid.uuid4().hex)
    screener = Screener(detector, cfg.policy, cfg.detector_timeout_s, cfg.detector_workers)
    detector_names = member_names(detector)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        screener.close()
        if hasattr(upstream, "aclose"):
            await upstream.aclose()
        audit.close()

    app = FastAPI(title="privgate gateway", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.screener = screener          # holds the detector and policy, never request data

    async def _send(payload: dict) -> dict:
        """The only call to the upstream in the whole gateway."""
        return await upstream.complete(payload)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        ready = detector is not None
        members = list(detector_names) or [cfg.detector_name]
        body = {"status": "ok" if ready else "unavailable",
                "detectors": [{"name": n, "ready": ready} for n in members],
                "fail_open": cfg.fail_open}
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        rid, t_start = new_id(), time.perf_counter()
        stages: dict[str, float] = {}
        counts: dict[str, int] = {}

        def finish(status: int, decision: str, body: dict, error_class: str | None = None,
                   served_unscreened: bool = False) -> JSONResponse:
            stages["total"] = (time.perf_counter() - t_start) * 1e3
            audit.write(AuditRecord(rid, decision, status, counts, detector_names, stages,
                                    error_class, fail_open=served_unscreened))
            return JSONResponse(body, status_code=status, headers={REQUEST_ID_HEADER: rid})

        def refuse(status: int, code: str, message: str, decision: str,
                   error_class: str | None = None, kind: str = "invalid_request_error"):
            return finish(status, decision, _error_body(code, message, kind),
                          error_class or code)

        try:
            t = time.perf_counter()
            chat = parse_chat_request(await _read_body(request, cfg.limits), cfg.limits)
            stages["parse"] = (time.perf_counter() - t) * 1e3
        except RequestRejected as exc:
            return refuse(exc.status, exc.code, exc.message, "rejected_invalid")

        # --- screening: any failure here means NOTHING is sent upstream ---------------------
        outbound, mapping, unscreened = None, {}, False
        try:
            screened = await screener.screen(chat, request_seed(cfg.seed, rid))
        except PolicyViolation as exc:
            return refuse(403, "blocked_by_policy",
                          f"request contains blocked entity type(s): {list(exc.labels)}",
                          "blocked_policy", "PolicyViolation", "policy_error")
        except Exception as exc:        # noqa: BLE001 - everything else is "screening failed"
            failure = exc if isinstance(exc, ScreeningUnavailable) else ScreeningUnavailable(
                "internal_error", type(exc).__name__)
            if not cfg.fail_open:
                return refuse(503, "screening_unavailable", "screening unavailable",
                              "screening_unavailable",
                              failure.cause_class or failure.reason, "screening_error")
            logger.warning("fail-open: forwarding unscreened request (%s)", failure.reason)
            outbound, unscreened = chat, True
        else:
            outbound, mapping, counts = screened.request, screened.mapping, dict(screened.counts)
            stages["detect"], stages["mask"] = screened.detect_ms, screened.mask_ms

        # --- upstream and restore ------------------------------------------------------------
        try:
            t = time.perf_counter()
            raw_answer = await _send(outbound.to_payload())
            stages["upstream"] = (time.perf_counter() - t) * 1e3
        except Exception as exc:        # noqa: BLE001
            return refuse(502, "upstream_error", "the upstream request failed", "upstream_error",
                          type(exc).__name__, "upstream_error")
        try:
            t = time.perf_counter()
            answer = build_response(raw_answer, mapping, chat.model, rid)
            stages["unmask"] = (time.perf_counter() - t) * 1e3
        except Exception as exc:        # noqa: BLE001
            return refuse(502, "bad_upstream_response", "the upstream answer was unusable",
                          "upstream_error", type(exc).__name__, "upstream_error")
        answer["privgate"] = {"request_id": rid, "screened": not unscreened,
                              "masked_entities": counts}
        return finish(200, "served_unscreened" if unscreened else "served", answer,
                      served_unscreened=unscreened)

    return app
