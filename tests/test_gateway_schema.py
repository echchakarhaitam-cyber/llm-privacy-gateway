"""Request validation and policy: every refused field/type, and the accepted subset."""

import json

import pytest

from privgate.gateway.policy import BLOCK, Policy
from privgate.gateway.schema import Limits, RequestRejected, parse_chat_request
from privgate.spans import EntityType

E = EntityType


def body(**override):
    base = {"model": "local-model", "messages": [{"role": "user", "content": "hello"}]}
    base.update(override)
    return json.dumps({k: v for k, v in base.items() if v is not ...}).encode()


def reject(raw, status=400, code=None):
    with pytest.raises(RequestRejected) as info:
        parse_chat_request(raw)
    assert info.value.status == status
    if code:
        assert info.value.code == code
    return info.value


def test_minimal_request_is_accepted():
    req = parse_chat_request(body())
    assert req.model == "local-model" and req.strings() == ["hello"]
    assert req.to_payload() == {"model": "local-model",
                                "messages": [{"role": "user", "content": "hello"}]}


def test_parts_content_roundtrips_as_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    req = parse_chat_request(body(messages=[{"role": "user", "content": parts}],
                                  temperature=0.5, max_tokens=7))
    assert req.strings() == ["a", "b"]
    assert req.with_strings(["x", "y"]).to_payload() == {
        "model": "local-model", "temperature": 0.5, "max_tokens": 7,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"},
                                                  {"type": "text", "text": "y"}]}]}


def test_all_roles_including_assistant_turns_are_listed_for_screening():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
    assert parse_chat_request(body(messages=msgs)).strings() == ["s", "u", "a", "u2"]


@pytest.mark.parametrize("field", ["tools", "tool_choice", "functions", "function_call",
                                   "response_format", "user", "n", "stop", "logit_bias", "extra"])
def test_unknown_top_level_fields_are_refused(field):
    err = reject(body(**{field: "x"}), code="unknown_field")
    assert field in err.message


@pytest.mark.parametrize("key", ["name", "tool_calls", "function_call", "tool_call_id",
                                 "refusal", "audio"])
def test_unknown_message_fields_are_refused(key):
    msg = {"role": "user", "content": "hi", key: "x"}
    reject(body(messages=[msg]), code="unknown_field")


def test_stream_true_is_refused_and_stream_false_allowed():
    reject(body(stream=True), code="unsupported_stream")
    reject(body(stream="yes"), code="unsupported_stream")
    assert parse_chat_request(body(stream=False)).model == "local-model"


@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    {"type": "input_audio", "input_audio": {}},
    {"type": "file", "file": {}},
    {"text": "no type"},
])
def test_non_text_parts_are_refused(part):
    reject(body(messages=[{"role": "user", "content": [part]}]), code="unsupported_content")


def test_text_part_with_extra_field_is_refused():
    part = {"type": "text", "text": "hi", "cache_control": {}}
    reject(body(messages=[{"role": "user", "content": [part]}]), code="unknown_field")


@pytest.mark.parametrize("content", [None, 5, {"a": 1}, [], [5], ["x"], [{"type": "text",
                                                                          "text": 3}]])
def test_bad_content_is_refused(content):
    reject(body(messages=[{"role": "user", "content": content}]))


def test_missing_content_and_bad_roles_are_refused():
    reject(body(messages=[{"role": "user"}]), code="invalid_content")
    for role in ("tool", "function", "developer", "", None, 3):
        reject(body(messages=[{"role": role, "content": "x"}]), code="invalid_role")


@pytest.mark.parametrize("messages", [..., None, [], "hi", {"role": "user"}, ["hi"]])
def test_bad_messages_container_is_refused(messages):
    reject(body(messages=messages))


@pytest.mark.parametrize("model", [..., None, "", 3, "a b", pytest.param("x" * 101, id="long"), "m\n", "ünï",
                                   "john@example.com?q=secret pii"])
def test_model_is_restricted_because_it_is_forwarded(model):
    reject(body(model=model))


@pytest.mark.parametrize("value", [True, "0.5", -0.1, 2.1])
def test_bad_temperature(value):
    reject(body(temperature=value), code="invalid_temperature")


def test_nan_and_infinity_in_json_are_refused():
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        raw = b'{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":' \
            + token + b"}"
        reject(raw, code="invalid_json")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10", 10**9])
def test_bad_max_tokens(value):
    reject(body(max_tokens=value), code="invalid_max_tokens")


@pytest.mark.parametrize("raw", [b"", b"nope", b"[]", b"3", b"null", b"\xff\xfe",
                                 pytest.param(b"[" * 100000, id="deep-nesting")])
def test_malformed_bodies(raw):
    reject(raw, code="invalid_json")


def test_lone_surrogate_is_refused():
    raw = b'{"model":"m","messages":[{"role":"user","content":"a\\ud800b"}]}'
    reject(raw, code="invalid_unicode")


def test_size_limit_is_413_and_message_count_is_400():
    big = body(messages=[{"role": "user", "content": "x" * 500}])
    with pytest.raises(RequestRejected) as info:
        parse_chat_request(big, Limits(max_request_bytes=200, max_messages=2))
    assert info.value.status == 413
    three = body(messages=[{"role": "user", "content": "x"}] * 3)
    with pytest.raises(RequestRejected) as info:
        parse_chat_request(three, Limits(max_messages=2))
    assert (info.value.status, info.value.code) == (400, "too_many_messages")


def test_unicode_content_is_kept_exactly():
    text = "Zoë \U0001F600 \u202eRTL é \u65e5\u672c\u8a9e"
    assert parse_chat_request(body(messages=[{"role": "user", "content": text}])).strings() \
        == [text]


def test_error_messages_never_echo_values():
    secret = "Alice Wonderland alice@example.com"
    for raw in (body(messages=[{"role": "user", "content": 5, "note": secret}]),
                body(model=secret), body(messages=[{"role": secret, "content": "x"}])):
        err = reject(raw)
        assert secret not in err.message and "alice@example.com" not in err.message


def test_limits_reject_nonpositive_values():
    for kw in ({"max_request_bytes": 0}, {"max_messages": 0}, {"max_tokens_cap": 0}):
        with pytest.raises(ValueError):
            Limits(**kw)


def test_policy_defaults_to_mask_and_blocks_listed_labels():
    policy = Policy.blocking(["CREDIT_CARD", E.IBAN])
    assert policy.action(E.PERSON) == "mask"
    assert policy.action(E.CREDIT_CARD) == BLOCK
    assert policy.blocked([E.PERSON, E.IBAN, E.IBAN, E.CREDIT_CARD]) == [E.CREDIT_CARD, E.IBAN]
    assert Policy().blocked([E.PERSON]) == []


def test_policy_rejects_bad_actions_and_labels():
    with pytest.raises(ValueError):
        Policy({E.PERSON: "allow"})
    with pytest.raises(ValueError):
        Policy(default_action="drop")
    with pytest.raises(ValueError):
        Policy({"NOT_A_LABEL": "mask"})
