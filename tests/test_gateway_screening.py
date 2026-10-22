"""The screening pipeline: masking across messages, policy, and every fail-closed rule."""

import asyncio
import dataclasses
import time

import pytest

from privgate.detectors.regex import RegexDetector
from privgate.gateway.policy import Policy
from privgate.gateway.schema import ChatMessage, ChatRequest
from privgate.gateway.screening import (
    PolicyViolation,
    Screener,
    ScreeningUnavailable,
    validate_spans,
)
from privgate.masking import Masker, MaskingError, unmask
from privgate.spans import EntityType, Span

E = EntityType
EMAIL = "anna.lee@example.org"
CARD = "4111 1111 1111 1111"


def run(coro):
    return asyncio.run(coro)


def chat(*texts, roles=None):
    roles = roles or ["user"] * len(texts)
    return ChatRequest("m", tuple(ChatMessage(r, t) for r, t in zip(roles, texts)))


def screen(request, detector=None, seed=0, **kw):
    screener = Screener(detector or RegexDetector(), **kw)
    try:
        return run(screener.screen(request, seed))
    finally:
        screener.close()


def reason(request, detector, **kw):
    with pytest.raises(ScreeningUnavailable) as info:
        screen(request, detector, **kw)
    return info.value


class Scripted:
    """A detector whose behaviour a test controls."""

    name = "scripted"
    supported_labels = frozenset(E)

    def __init__(self, fn):
        self.fn = fn
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        return self.fn(text)


def test_every_message_of_every_role_is_screened_and_masked():
    req = chat(f"system mail {EMAIL}", f"user mail {EMAIL}", f"assistant says {CARD}",
               roles=["system", "user", "assistant"])
    out = screen(req)
    texts = out.request.strings()
    assert all(EMAIL not in t and CARD not in t for t in texts)
    assert [m.role for m in out.request.messages] == ["system", "user", "assistant"]
    assert out.counts == {"EMAIL": 2, "CREDIT_CARD": 1}


def test_same_value_gets_the_same_placeholder_in_every_message():
    out = screen(chat(f"a {EMAIL}", f"b {EMAIL} and again {EMAIL}"))
    (ph,) = out.mapping
    assert out.mapping[ph] == EMAIL
    assert out.request.strings() == [f"a {ph}", f"b {ph} and again {ph}"]


def test_distinct_values_get_distinct_placeholders_numbered_across_messages():
    out = screen(chat("x one@example.org", "y two@example.org"))
    assert out.request.strings() == ["x <EMAIL_1>", "y <EMAIL_2>"]


def test_unmask_restores_every_string_exactly():
    texts = [f"mail {EMAIL}", "nothing here", f"{CARD} then {EMAIL}", ""]
    out = screen(chat(*texts))
    assert [unmask(t, out.mapping) for t in out.request.strings()] == texts


def test_text_parts_are_screened_one_by_one_and_stay_parts():
    req = ChatRequest("m", (ChatMessage("user", (f"a {EMAIL}", "plain", f"{CARD}")),))
    out = screen(req)
    (msg,) = out.request.messages
    assert isinstance(msg.content, tuple) and len(msg.content) == 3
    assert msg.content[1] == "plain" and EMAIL not in msg.content[0]


def test_detector_sees_each_string_separately_with_its_own_offsets():
    det = Scripted(lambda t: [])
    screen(chat("first", "second", ""), det)
    assert det.seen == ["first", "second"]       # empty strings carry no data: not sent


def test_unicode_offsets_survive_masking():
    text = f"Zoë 😀 日本語 — mail {EMAIL} ✓"
    out = screen(chat(text, text))
    assert out.request.strings() == ["Zoë 😀 日本語 — mail <EMAIL_1> ✓"] * 2
    assert [unmask(t, out.mapping) for t in out.request.strings()] == [text, text]


def test_placeholder_like_text_in_the_input_is_not_confused_with_ours():
    hostile = f"I typed <EMAIL_1> myself, and my mail is {EMAIL}; also <EMAIL_1#abcdef>"
    out = screen(chat(hostile, "<EMAIL_1>"), seed=7)
    assert [unmask(t, out.mapping) for t in out.request.strings()] == [hostile, "<EMAIL_1>"]
    assert EMAIL not in out.request.strings()[0]
    # our placeholder differs from the one the user typed, so the answer cannot be forged
    (ph,) = out.mapping
    assert ph != "<EMAIL_1>"


def test_a_placeholder_typed_by_the_user_does_not_restore_another_value():
    out = screen(chat(f"my mail is {EMAIL}", "please repeat <EMAIL_1> back"), seed=3)
    typed_by_user = out.request.strings()[1]
    assert "<EMAIL_1>" in typed_by_user and EMAIL not in typed_by_user
    # echoing the user's own text back must give exactly that text, not the other message's mail
    assert unmask(typed_by_user, out.mapping) == "please repeat <EMAIL_1> back"


def test_policy_block_raises_with_labels_only():
    screener = Screener(RegexDetector(), Policy.blocking([E.CREDIT_CARD]))
    with pytest.raises(PolicyViolation) as info:
        run(screener.screen(chat(f"pay with {CARD}", EMAIL), 0))
    assert info.value.labels == ("CREDIT_CARD",)
    assert CARD not in str(info.value)
    screener.close()


def test_policy_block_ignores_masked_types():
    screener = Screener(RegexDetector(), Policy.blocking([E.IBAN]))
    out = run(screener.screen(chat(f"mail {EMAIL}"), 0))
    assert out.counts == {"EMAIL": 1}
    screener.close()


# --- fail-closed rules: each ends in ScreeningUnavailable, so nothing can be sent --------------

def test_rule_not_ready_without_a_detector():
    screener = Screener(None)
    with pytest.raises(ScreeningUnavailable) as info:
        run(screener.screen(chat("hi"), 0))
    assert info.value.reason == "not_ready"
    screener.close()


def test_rule_detector_exception_blocks_and_names_only_the_class():
    def boom(text):
        raise ValueError(f"cannot parse {text}")
    err = reason(chat(f"secret {EMAIL}"), Scripted(boom))
    assert (err.reason, err.cause_class) == ("detector_error", "ValueError")
    assert EMAIL not in str(err) and EMAIL not in repr(err.cause_class)


def test_rule_detector_exception_in_a_later_message_blocks_the_whole_request():
    def second_fails(text):
        if text == "bad":
            raise RuntimeError
        return []
    assert reason(chat("fine", "bad"), Scripted(second_fails)).reason == "detector_error"


def test_rule_detector_timeout_blocks_promptly():
    def hang(text):
        time.sleep(0.4)
        return []
    t0 = time.perf_counter()
    err = reason(chat("hello"), Scripted(hang), timeout_s=0.05)
    assert err.reason == "detector_timeout"
    assert time.perf_counter() - t0 < 0.3          # did not wait for the detector


def test_timeout_budget_covers_the_whole_request_not_each_string():
    def slow(text):
        time.sleep(0.04)
        return []
    assert reason(chat(*["x"] * 6), Scripted(slow), timeout_s=0.1).reason == "detector_timeout"


def test_per_call_timeout_overrides_the_default():
    def slow(text):
        time.sleep(0.2)
        return []
    screener = Screener(Scripted(slow), timeout_s=5)
    with pytest.raises(ScreeningUnavailable) as info:
        run(screener.screen(chat("x"), 0, timeout_s=0.02))
    assert info.value.reason == "detector_timeout"
    screener.close()


def test_hung_detector_does_not_block_a_later_request_forever():
    def sometimes(text):
        if text == "hang":
            time.sleep(0.15)
        return []
    screener = Screener(Scripted(sometimes), timeout_s=0.03, workers=2)

    async def go():
        with pytest.raises(ScreeningUnavailable):
            await screener.screen(chat("hang"), 0)
        return await screener.screen(chat("ok"), 0)
    assert run(go()).request.strings() == ["ok"]
    screener.close()


@pytest.mark.parametrize("make", [
    lambda t: [Span(0, len(t) + 1, E.PERSON)],                    # out of bounds
    lambda t: [Span(0, 4, E.PERSON), Span(2, 6, E.EMAIL)],         # overlap
    lambda t: [Span(5, 7, E.PERSON), Span(0, 2, E.EMAIL)],         # unsorted
    lambda t: [(0, 4, "PERSON")],                                  # not Span objects
    lambda t: "PERSON",                                            # not a list
    lambda t: None,                                                # no output at all
    lambda t: [Span(0, 3, E.PERSON), None],
], ids=["bounds", "overlap", "unsorted", "tuples", "string", "none", "none-item"])
def test_rule_malformed_detector_output_blocks(make):
    err = reason(chat("Hello world, this is text"), Scripted(make))
    assert err.reason == "malformed_output"


def test_validate_spans_accepts_adjacent_and_empty():
    text = "abcdef"
    assert validate_spans(text, []) == []
    assert len(validate_spans(text, [Span(0, 3, E.PERSON), Span(3, 6, E.EMAIL)])) == 2


def test_validate_spans_checks_bounds_per_string_not_per_request():
    # a span valid for the longer first message must not pass for the short second one
    det = Scripted(lambda t: [Span(0, 10, E.PERSON)])
    assert reason(chat("0123456789", "short"), det).reason == "malformed_output"


class TamperedMasker(Masker):
    """Masks correctly, then breaks one invariant the screener must catch."""

    def __init__(self, tamper):
        super().__init__()
        self.tamper = tamper

    def mask(self, text, spans, strategy="typed", seed=0):
        return self.tamper(super().mask(text, spans, strategy, seed))


def tampered(tamper, request=None):
    screener = Screener(RegexDetector(), masker=TamperedMasker(tamper))
    try:
        with pytest.raises(ScreeningUnavailable) as info:
            run(screener.screen(request or chat(f"a {EMAIL} b {CARD}"), 0))
        return info.value
    finally:
        screener.close()


def test_rule_mapping_collision_blocks():
    # every placeholder mapped to the same original: unmask cannot restore the others
    def collide(m):
        return dataclasses.replace(m, mapping={k: EMAIL for k in m.mapping})
    assert tampered(collide).reason == "mapping_error"


def test_rule_masked_text_that_does_not_round_trip_blocks():
    def corrupt(m):
        return dataclasses.replace(m, text=m.text.replace("<EMAIL_1>", "<EMAIL_9>"))
    assert tampered(corrupt).reason == "mapping_error"


def test_rule_dropped_edit_blocks():
    err = tampered(lambda m: dataclasses.replace(m, edits=m.edits[:-1]))
    assert err.reason == "mapping_error"


def test_rule_masking_error_blocks():
    def fail(m):
        raise MaskingError("no unambiguous placeholders")
    assert tampered(fail).reason == "mapping_error"


def test_rule_unexpected_error_in_masking_blocks_as_internal_error():
    def explode(m):
        raise KeyError("boom " + EMAIL)
    err = tampered(explode)
    assert (err.reason, err.cause_class) == ("internal_error", "KeyError")
    assert EMAIL not in str(err)


def test_constructor_rejects_bad_settings():
    with pytest.raises(ValueError):
        Screener(RegexDetector(), timeout_s=0)
    with pytest.raises(ValueError):
        Screener(RegexDetector(), workers=0)
