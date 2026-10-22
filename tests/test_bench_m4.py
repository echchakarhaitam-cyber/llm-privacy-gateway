"""Milestone 4 benchmark additions: Luhn analysis of gold cards, paired bootstrap."""

import pytest

from privgate.bench.cards import luhn_report
from privgate.bench.paired import paired_bootstrap
from privgate.data.documents import Document
from privgate.spans import EntityType, Span

E = EntityType


def test_luhn_report_on_fixed_spans():
    text = "valid 4111 1111 1111 1111 bad 4111-1111-1111-1112 short 4243"
    spans = (Span(6, 25, E.CREDIT_CARD), Span(30, 49, E.CREDIT_CARD),
             Span(56, 60, E.CREDIT_CARD), Span(0, 5, E.OTHER))
    report = luhn_report([Document("d", text, spans, "s")])
    assert report == {"gold_card_spans": 3, "luhn_pass": 1, "luhn_pass_rate": 1 / 3,
                      "length_13_19": 2, "length_13_19_and_luhn_pass": 1}
    assert luhn_report([])["luhn_pass_rate"] == 0.0


def test_paired_bootstrap_is_deterministic_and_signed():
    text = "Anna wrote to bob@x.org"
    gold = (Span(0, 4, E.PERSON), Span(14, 23, E.EMAIL))
    docs = [Document(f"d{i}", text, gold, "s") for i in range(30)]
    good = [[Span(0, 4, E.PERSON), Span(14, 23, E.EMAIL)] for _ in docs]
    half = [[Span(14, 23, E.EMAIL)] if i % 2 else [] for i in range(30)]
    a = paired_bootstrap(docs, good, half, n_resamples=200)
    assert a == paired_bootstrap(docs, good, half, n_resamples=200)
    assert a["leakage_rate"]["a"] == 0.0 and a["leakage_rate"]["delta"] < 0
    _lo, hi = a["leakage_rate"]["ci95"]
    assert hi < 0 and a["leakage_rate"]["ci_excludes_zero"]
    same = paired_bootstrap(docs, good, good, n_resamples=50)
    assert same["type_agnostic_f1"]["ci95"] == [0.0, 0.0]
    assert not same["type_agnostic_f1"]["ci_excludes_zero"]
    with pytest.raises(ValueError):
        paired_bootstrap(docs, good, half[:3])
