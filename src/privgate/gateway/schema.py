"""The accepted subset of the OpenAI chat-completions request, validated strictly.

Why a hand-written validator instead of a permissive model: the request body is the one place
where data can leave the machine in a field the detectors never look at. So the rule is a
whitelist -- exactly ``model``, ``messages``, ``temperature``, ``max_tokens`` (and
``stream: false``); every other key, at any level, is refused with a 400 and nothing is sent
upstream. Content must be text: a string, or a list of ``{"type": "text", "text": ...}``
parts. Images, files, tools and function calls are refused, not "screened" -- the gateway
has no detector for them.

Error messages name fields and limits, never values: they end up in responses and must not
echo a client's text into places where it could be logged.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace

ROLES = ("system", "user", "assistant")
TOP_LEVEL_FIELDS = frozenset({"model", "messages", "temperature", "max_tokens", "stream"})
MESSAGE_FIELDS = frozenset({"role", "content"})
PART_FIELDS = frozenset({"type", "text"})
# The model name is forwarded upstream, so it is a data channel too: restrict its alphabet.
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,99}")


class RequestRejected(Exception):
    """The request is refused before any screening or upstream call."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Limits:
    max_request_bytes: int = 262_144
    max_messages: int = 64
    max_tokens_cap: int = 8192

    def __post_init__(self) -> None:
        for name in ("max_request_bytes", "max_messages", "max_tokens_cap"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class ChatMessage:
    """``content`` is a string, or a tuple of text parts (kept as parts when forwarded)."""

    role: str
    content: str | tuple[str, ...]

    def texts(self) -> tuple[str, ...]:
        return (self.content,) if isinstance(self.content, str) else self.content

    def to_payload(self) -> dict:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role,
                "content": [{"type": "text", "text": t} for t in self.content]}


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_tokens: int | None = None

    def strings(self) -> list[str]:
        """Every string that will be sent upstream, in a fixed order."""
        return [t for m in self.messages for t in m.texts()]

    def with_strings(self, strings: list[str]) -> ChatRequest:
        """The same request with its strings replaced, in :meth:`strings` order."""
        if len(strings) != len(self.strings()):
            raise ValueError("one replacement string per original string is required")
        it = iter(strings)
        messages = []
        for m in self.messages:
            if isinstance(m.content, str):
                messages.append(replace(m, content=next(it)))
            else:
                messages.append(replace(m, content=tuple(next(it) for _ in m.content)))
        return replace(self, messages=tuple(messages))

    def to_payload(self) -> dict:
        payload: dict = {"model": self.model,
                         "messages": [m.to_payload() for m in self.messages]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


def _bad(code: str, message: str, status: int = 400) -> RequestRejected:
    return RequestRejected(status, code, message)


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def _check_text(value: str, where: str) -> str:
    try:
        value.encode("utf-8")           # lone surrogates cannot be sent on
    except UnicodeEncodeError:
        raise _bad("invalid_unicode", f"{where} is not valid Unicode text") from None
    return value


def _parse_content(content: object, where: str):
    if isinstance(content, str):
        return _check_text(content, where)
    if isinstance(content, list):
        if not content:
            raise _bad("invalid_content", f"{where} must not be an empty list")
        parts = []
        for k, part in enumerate(content):
            at = f"{where}[{k}]"
            if not isinstance(part, dict):
                raise _bad("invalid_content", f"{at} must be an object")
            if part.get("type") != "text":
                raise _bad("unsupported_content",
                           f"{at}: only text parts are supported; non-text content is refused")
            extra = sorted(set(part) - PART_FIELDS)
            if extra:
                raise _bad("unknown_field", f"{at}: unsupported field(s) {extra}")
            if not isinstance(part.get("text"), str):
                raise _bad("invalid_content", f"{at}.text must be a string")
            parts.append(_check_text(part["text"], f"{at}.text"))
        return tuple(parts)
    raise _bad("invalid_content", f"{where} must be a string or a list of text parts")


def parse_chat_request(raw: bytes, limits: Limits | None = None) -> ChatRequest:
    """Validate a request body. Raises :class:`RequestRejected` (400 / 413)."""
    limits = limits or Limits()
    if len(raw) > limits.max_request_bytes:
        raise _bad("request_too_large",
                   f"request body exceeds {limits.max_request_bytes} bytes", 413)
    try:
        body = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _bad("invalid_json", "request body is not valid UTF-8 JSON") from None
    if not isinstance(body, dict):
        raise _bad("invalid_json", "request body must be a JSON object")
    extra = sorted(set(body) - TOP_LEVEL_FIELDS)
    if extra:
        raise _bad("unknown_field",
                   f"unsupported field(s) {extra}; accepted: model, messages, temperature, "
                   "max_tokens")
    if body.get("stream", False) is not False:
        raise _bad("unsupported_stream", "stream=true is not supported")

    model = body.get("model")
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
        raise _bad("invalid_model", "model must be a short identifier "
                   "(letters, digits and . _ : / @ + -)")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _bad("invalid_messages", "messages must be a non-empty list")
    if len(messages) > limits.max_messages:
        raise _bad("too_many_messages", f"at most {limits.max_messages} messages are accepted")
    parsed = []
    for i, m in enumerate(messages):
        where = f"messages[{i}]"
        if not isinstance(m, dict):
            raise _bad("invalid_messages", f"{where} must be an object")
        extra = sorted(set(m) - MESSAGE_FIELDS)
        if extra:
            raise _bad("unknown_field", f"{where}: unsupported field(s) {extra}")
        if m.get("role") not in ROLES:
            raise _bad("invalid_role", f"{where}.role must be one of {list(ROLES)}")
        if "content" not in m:
            raise _bad("invalid_content", f"{where}.content is required")
        parsed.append(ChatMessage(m["role"], _parse_content(m["content"], f"{where}.content")))

    temperature = body.get("temperature")
    if temperature is not None:
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise _bad("invalid_temperature", "temperature must be a number in [0, 2]")
        temperature = float(temperature)
    max_tokens = body.get("max_tokens")
    if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= limits.max_tokens_cap):
        raise _bad("invalid_max_tokens",
                   f"max_tokens must be an integer in [1, {limits.max_tokens_cap}]")
    return ChatRequest(model, tuple(parsed), temperature, max_tokens)
