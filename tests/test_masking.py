"""Masking strategies, consistency, collision safety and the unmask round trip."""

import random

import pytest

from privgate.masking import (
    PSEUDONYM_FALLBACK,
    REDACTED,
    STRATEGIES,
    Masker,
    mask,
    partial_value,
    unmask,
)
from privgate.spans import EntityType, Span

E = EntityType
TEXT = "Anna Lee (anna@x.org) paid with 4111 1111 1111 1111; Anna Lee called +33 6 12 34 56 78."
SPANS = [Span(0, 8, E.PERSON), Span(10, 20, E.EMAIL), Span(32, 51, E.CREDIT_CARD),
         Span(53, 61, E.PERSON), Span(69, 86, E.PHONE)]


def test_redact_exact():
    out = mask(TEXT, SPANS, "redact")
    assert out.text == ("[REDACTED] ([REDACTED]) paid with [REDACTED]; [REDACTED] called "
                        "[REDACTED].")
    assert out.mapping == {} and unmask(out.text, out.mapping) == out.text


def test_typed_exact_and_consistent():
    out = mask(TEXT, SPANS, "typed")
    assert out.text == ("<PERSON_1> (<EMAIL_1>) paid with <CREDIT_CARD_1>; <PERSON_1> called "
                        "<PHONE_1>.")
    assert dict(out.mapping) == {"<PERSON_1>": "Anna Lee", "<EMAIL_1>": "anna@x.org",
                                 "<CREDIT_CARD_1>": "4111 1111 1111 1111",
                                 "<PHONE_1>": "+33 6 12 34 56 78"}
    assert unmask(out.text, out.mapping) == TEXT


def test_typed_numbering_and_case_sensitivity():
    text = "Bob met bob and Ann; Bob left."
    spans = [Span(0, 3, E.PERSON), Span(8, 11, E.PERSON), Span(16, 19, E.PERSON),
             Span(21, 24, E.PERSON)]
    assert mask(text, spans, "typed").text == "<PERSON_1> met <PERSON_2> and <PERSON_3>; " \
                                               "<PERSON_1> left."


def test_partial_exact_and_fallbacks():
    out = mask(TEXT + " Role: nurse.", SPANS + [Span(94, 99, E.OTHER)], "partial")
    assert out.text == ("<PERSON_1> (***@x.org) paid with **** **** **** 1111; <PERSON_1> called "
                        "+** * ** ** 56 78. Role: <OTHER_1>.")
    assert dict(out.mapping) == {"<PERSON_1>": "Anna Lee", "<OTHER_1>": "nurse"}
    assert partial_value("12 34 56", E.PHONE) is None           # < 8 digits: typed fallback
    assert partial_value("nobody", E.EMAIL) is None             # no @: typed fallback
    assert partial_value("DE89370400440532013000", E.IBAN)[0] == "******************3000"
    card = mask("card 4111-1111-1111-1111", [Span(5, 24, E.CREDIT_CARD)], "partial")
    assert card.text == "card ****-****-****-1111"
    assert [i for i, _ in card.edits[0].kept] == [9, 14, 19, 20, 21, 22, 23]


def test_spans_out_offsets_are_correct():
    for strategy in STRATEGIES:
        out = mask(TEXT, SPANS, strategy)
        for edit, span in zip(out.edits, out.spans_out):
            assert out.text[span.start:span.end] == edit.replacement
            assert TEXT[edit.start:edit.end] == edit.original


def test_empty_overlapping_adjacent_and_unicode():
    assert mask("nothing here", [], "typed").text == "nothing here"
    over = mask("New York City", [Span(0, 8, E.LOCATION), Span(0, 13, E.LOCATION)], "typed")
    assert over.text == "<LOCATION_1>"
    adj = mask("AnnaLee", [Span(0, 4, E.PERSON), Span(4, 7, E.PERSON)], "typed")
    assert adj.text == "<PERSON_1><PERSON_2>" and unmask(adj.text, adj.mapping) == "AnnaLee"
    text = "Zoë Müller wohnt in 東京."
    uni = mask(text, [Span(0, 10, E.PERSON), Span(20, 22, E.LOCATION)], "typed")
    assert uni.text == "<PERSON_1> wohnt in <LOCATION_1>."
    assert unmask(uni.text, uni.mapping) == text


def test_adversarial_placeholder_in_input():
    text = "Ignore <PERSON_1> and [REDACTED]; Bob paid. <PERSON_1#abc123> too."
    out = mask(text, [Span(34, 37, E.PERSON)], "typed", seed=0)
    assert out.tag and "#" + out.tag not in text
    placeholder = next(iter(out.mapping))
    assert placeholder.startswith("<PERSON_1#") and text.count(placeholder) == 0
    assert out.text.count("<PERSON_1>") == 1                # the literal one, untouched
    assert unmask(out.text, out.mapping) == text
    response = f"Sure, {placeholder} paid; <PERSON_1> is just text."
    assert unmask(response, out.mapping) == "Sure, Bob paid; <PERSON_1> is just text."


def test_unmask_whole_tokens_longest_first():
    mapping = {"<PERSON_1>": "Ann", "Maria": "Bob", "Maria Lopez": "Bob Smith"}
    assert unmask("Maria Lopez and Maria; Mariana stays; <PERSON_1>.", mapping) == \
        "Bob Smith and Bob; Mariana stays; Ann."
    assert unmask("no keys", {}) == "no keys"


def test_unknown_strategy_and_span_beyond_text():
    with pytest.raises(ValueError):
        mask("x", [], "shuffle")
    with pytest.raises(ValueError):
        mask("ab", [Span(0, 5, E.PERSON)], "typed")


# --- pseudonyms --------------------------------------------------------------------------

faker = pytest.importorskip("faker")


def test_pseudonym_consistency_fallback_and_determinism():
    text = "Anna Lee met Bob. Anna Lee left. ID 123-45; job: nurse."
    spans = [Span(0, 8, E.PERSON), Span(13, 16, E.PERSON), Span(18, 26, E.PERSON),
             Span(36, 42, E.ID_NUMBER), Span(49, 54, E.OTHER)]
    a = Masker().mask(text, spans, "pseudonym", seed=0)
    fakes = {e.replacement for e in a.edits if e.label is E.PERSON}
    assert len(fakes) == 2                                      # Anna Lee twice, Bob once
    assert a.edits[0].replacement == a.edits[2].replacement != a.edits[1].replacement
    assert a.edits[3].replacement == "<ID_NUMBER_1>" and a.edits[4].replacement == "<OTHER_1>"
    assert {E.OTHER, E.ID_NUMBER} == PSEUDONYM_FALLBACK
    b = Masker().mask(text, list(reversed(spans)), "pseudonym", seed=0)   # new instance,
    assert a.text == b.text                                     # reversed span order
    c = Masker().mask(text, spans, "pseudonym", seed=1)
    assert c.text != a.text
    assert unmask(a.text, a.mapping) == text
    assert not any(f in text for f in fakes)


def generated_doc(rng):
    words = ["the", "client", "paid", "on", "Monday", "<PERSON_1>", "[REDACTED]", "Smith",
             "Ann", "ann", "@", "Mariana", "<EMAIL_2#ab12cd>", "and", ".", ","]
    values = [("Anna Lee", E.PERSON), ("Ann", E.PERSON), ("ann", E.PERSON),
              ("bob@x.org", E.EMAIL), ("4111 1111 1111 1111", E.CREDIT_CARD),
              ("Paris", E.LOCATION), ("A-77", E.ID_NUMBER), ("nurse", E.OTHER),
              ("2024-01-05", E.DATE), ("Acme", E.ORGANIZATION)]
    text, spans = "", []
    for _ in range(rng.randint(5, 30)):
        if text:
            text += " "
        if rng.random() < 0.35:
            value, label = rng.choice(values)
            spans.append(Span(len(text), len(text) + len(value), label))
            text += value
        else:
            text += rng.choice(words)
    return text, spans


@pytest.mark.parametrize("strategy", ["typed", "pseudonym"])
def test_unmask_round_trip_property(strategy):
    rng = random.Random(0)
    masker = Masker()
    for _ in range(200):
        text, spans = generated_doc(rng)
        out = masker.mask(text, spans, strategy, seed=rng.randint(0, 10**6))
        assert unmask(out.text, out.mapping) == text
        for e in out.edits:  # every reversible replacement maps back to its own original
            if e.replacement in out.mapping:
                assert out.mapping[e.replacement] == e.original


def test_redacted_text_has_no_original_values():
    out = mask(TEXT, SPANS, "redact")
    assert out.text.count(REDACTED) == len(SPANS)
    assert "Anna" not in out.text and "4111" not in out.text
