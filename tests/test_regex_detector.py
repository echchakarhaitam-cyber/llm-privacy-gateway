import time

import pytest

from privgate.detectors import Detector, RegexDetector
from privgate.spans import EntityType

T = EntityType
DET = RegexDetector()


def found(text):
    """(label, matched text) pairs, in order."""
    return [(s.label, s.text(text)) for s in DET.detect(text)]


def test_implements_protocol():
    assert isinstance(DET, Detector)
    assert DET.name == "regex"


def test_all_types_with_exact_offsets():
    text = (
        "Mail ana.lopez@example.org, call +33 6 12 34 56 78, card 4111 1111 1111 1111, "
        "IBAN GB82 WEST 1234 5698 7654 32, server 192.168.0.1, see https://example.org/a?b=1."
    )
    spans = DET.detect(text)
    assert [(s.label, s.text(text)) for s in spans] == [
        (T.EMAIL, "ana.lopez@example.org"),
        (T.PHONE, "+33 6 12 34 56 78"),
        (T.CREDIT_CARD, "4111 1111 1111 1111"),
        (T.IBAN, "GB82 WEST 1234 5698 7654 32"),
        (T.IP_ADDRESS, "192.168.0.1"),
        (T.URL, "https://example.org/a?b=1"),
    ]
    first = spans[0]
    assert (first.start, first.end) == (5, 26)
    assert all(s.source == "regex" and s.score == 1.0 for s in spans)
    assert spans == sorted(spans, key=lambda s: s.start)


def test_no_pii():
    assert DET.detect("The meeting is moved to the afternoon, bring the slides.") == []
    assert DET.detect("") == []


def test_luhn_invalid_card_rejected():
    assert found("card 4111 1111 1111 1112 thanks") == []


def test_bad_iban_rejected():
    assert found("IBAN GB82 WEST 1234 5698 7654 33 please") == []


def test_iban_followed_by_uppercase_word():
    # The greedy match also swallows " FR"; the detector must shrink back to the valid IBAN.
    assert found("pay GB82 WEST 1234 5698 7654 32 FR today") == [
        (T.IBAN, "GB82 WEST 1234 5698 7654 32")
    ]


def test_card_followed_by_expiry_digits():
    assert found("4111 1111 1111 1111 12/29") == [(T.CREDIT_CARD, "4111 1111 1111 1111")]


def test_card_with_dashes_and_contiguous():
    assert found("4111-1111-1111-1111") == [(T.CREDIT_CARD, "4111-1111-1111-1111")]
    assert found("4111111111111111") == [(T.CREDIT_CARD, "4111111111111111")]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(ana@example.org)", "ana@example.org"),
        ("<ana@example.org>,", "ana@example.org"),
        ("Contact: ana+news@mail.example.co.uk.", "ana+news@mail.example.co.uk"),
    ],
)
def test_email_in_punctuation(text, expected):
    assert found(text) == [(T.EMAIL, expected)]


@pytest.mark.parametrize(
    "phone",
    ["06 12 34 56 78", "+33 6 12 34 56 78", "+212 6 12 34 56 78", "0612345678", "06.12.34.56.78"],
)
def test_phone_formats(phone):
    assert found(f"Tel: {phone}.") == [(T.PHONE, phone)]


def test_phone_too_short_or_too_long_rejected():
    assert found("+1 23") == []  # fewer than 8 digits
    assert found("call 0012345678") == []  # national numbers start with 0 + non-zero digit


def test_invalid_ip_rejected():
    assert found("version 999.1.1.1 released") == []
    assert found("version 1.2.3 released") == []


def test_url_trailing_punctuation_and_parenthesis():
    assert found("(see www.example.org/page).") == [(T.URL, "www.example.org/page")]
    assert found("docs: https://en.wikipedia.org/wiki/Foo_(bar)") == [
        (T.URL, "https://en.wikipedia.org/wiki/Foo_(bar)")
    ]


def test_email_inside_url_resolved_to_url():
    text = "https://example.org/u/ana@example.org"
    assert found(text) == [(T.URL, text)]


def test_offsets_match_original_text():
    text = "é" * 10 + " ana@example.org"
    (span,) = DET.detect(text)
    assert text[span.start : span.end] == "ana@example.org"


@pytest.mark.parametrize(
    "adversarial",
    [
        "a" * 100_000 + "@" + "a" * 100_000,
        "1" * 200_000,
        "1 " * 100_000,
        "1-" * 100_000,
        "+1" * 100_000,
        "a." * 100_000 + "@",
        "a@" + "a." * 99_999 + "!",
        "06" * 100_000,
        "1." * 100_000,
        "GB82" + " A" * 99_998,
        "http://" + "a" * 199_990,
    ],
    ids=[
        "email-local-domain", "digits", "spaced-digits", "dashed-digits", "plus-ones",
        "dots-then-at", "long-domain", "zero-six", "dotted-digits", "iban-groups", "long-url",
    ],
)
def test_no_catastrophic_backtracking(adversarial):
    assert len(adversarial) >= 199_990
    t0 = time.perf_counter()
    DET.detect(adversarial)
    assert time.perf_counter() - t0 < 2.0
