import pytest

from privgate.evaluation import (
    Counts,
    evaluate_chars,
    evaluate_corpus,
    evaluate_document,
    evaluate_mode,
)
from privgate.spans import EntityType, Span

P, E, PH = EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE

# Worked example (values computed by hand in docs/theory/01-detection-metrics.md).
GOLD = [Span(0, 5, P), Span(10, 20, E), Span(30, 40, PH)]
PRED = [Span(0, 5, P), Span(10, 18, E), Span(30, 40, E), Span(50, 55, P)]


def approx(x):
    return pytest.approx(x, abs=1e-12)


def counts(c):
    return (c.tp, c.fp, c.fn)


def test_strict_micro():
    m = evaluate_document(GOLD, PRED).strict.micro
    assert counts(m) == (1, 3, 2)
    assert m.precision == approx(0.25)
    assert m.recall == approx(1 / 3)
    assert m.f1 == approx(2 / 7)


def test_strict_per_label_and_macro():
    r = evaluate_document(GOLD, PRED).strict
    assert counts(r.per_label[P]) == (1, 1, 0)
    assert r.per_label[P].precision == approx(0.5)
    assert r.per_label[P].recall == approx(1.0)
    assert r.per_label[P].f1 == approx(2 / 3)
    assert counts(r.per_label[E]) == (0, 2, 1)
    assert counts(r.per_label[PH]) == (0, 0, 1)
    for label in (E, PH):
        c = r.per_label[label]
        assert (c.precision, c.recall, c.f1) == (0.0, 0.0, 0.0)
    assert set(r.per_label) == {P, E, PH}  # union of gold and predicted labels
    assert r.macro_f1 == approx(2 / 9)


def test_partial_micro():
    m = evaluate_document(GOLD, PRED).partial.micro
    assert counts(m) == (2, 2, 1)
    assert m.precision == approx(0.5)
    assert m.recall == approx(2 / 3)
    assert m.f1 == approx(4 / 7)


def test_type_agnostic_micro_and_attribution():
    r = evaluate_document(GOLD, PRED).type_agnostic
    assert counts(r.micro) == (3, 1, 0)
    assert r.micro.precision == approx(0.75)
    assert r.micro.recall == approx(1.0)
    assert r.micro.f1 == approx(6 / 7)
    # TP attributed to the gold label: the PHONE was hidden (as an EMAIL).
    assert counts(r.per_label[PH]) == (1, 0, 0)
    assert counts(r.per_label[P]) == (1, 1, 0)


def test_character_level():
    c = evaluate_document(GOLD, PRED).chars
    assert (c.gold_chars, c.covered_gold_chars, c.pred_chars, c.pred_chars_in_gold) == (
        25,
        23,
        28,
        23,
    )
    assert c.char_recall == approx(0.92)
    assert c.leakage_rate == approx(0.08)
    assert c.char_precision == approx(23 / 28)


def test_overlapping_predictions_count_characters_once():
    c = evaluate_chars([Span(0, 10, P)], [Span(0, 6, P), Span(4, 12, E)])
    assert (c.gold_chars, c.covered_gold_chars, c.pred_chars) == (10, 10, 12)
    assert c.leakage_rate == approx(0.0)


def test_one_to_one_matching():
    # Two predictions overlap the same gold span: only the larger overlap is a TP.
    r = evaluate_mode([Span(0, 10, P)], [Span(0, 3, P), Span(2, 10, P)], "partial")
    assert counts(r.per_label[P]) == (1, 1, 0)
    # Duplicate exact predictions: the second one is a false positive in strict mode.
    r = evaluate_mode([Span(0, 5, P)], [Span(0, 5, P), Span(0, 5, P)], "strict")
    assert counts(r.per_label[P]) == (1, 1, 0)


def test_greedy_tie_break_prefers_smaller_starts():
    gold = [Span(0, 4, P), Span(4, 8, P)]
    pred = [Span(2, 6, P)]  # overlaps both gold spans by 2
    r = evaluate_mode(gold, pred, "partial")
    assert counts(r.per_label[P]) == (1, 0, 1)


def test_empty_inputs():
    r = evaluate_document([], [])
    assert r.strict.per_label == {}
    assert counts(r.strict.micro) == (0, 0, 0)
    assert r.strict.micro.f1 == 0.0 and r.strict.macro_f1 == 0.0
    assert r.chars.leakage_rate == 0.0 and r.chars.char_precision == 0.0
    only_pred = evaluate_document([], [Span(0, 3, E)])
    assert counts(only_pred.partial.micro) == (0, 1, 0)
    only_gold = evaluate_document([Span(0, 3, E)], [])
    assert counts(only_gold.partial.micro) == (0, 0, 1)
    assert only_gold.chars.leakage_rate == 1.0


def test_corpus_is_sum_of_documents():
    doc2_gold = [Span(0, 4, E)]
    doc2_pred = [Span(0, 4, E), Span(6, 9, PH)]
    corpus = evaluate_corpus([(GOLD, PRED), (doc2_gold, doc2_pred)])
    assert corpus.n_documents == 2
    assert counts(corpus.strict.micro) == (2, 4, 2)  # (1,3,2) + (1,1,0)
    assert counts(corpus.strict.per_label[E]) == (1, 2, 1)
    assert corpus.chars.gold_chars == 25 + 4
    assert corpus.chars.covered_gold_chars == 23 + 4
    assert corpus.chars.pred_chars == 28 + 7
    assert corpus.chars.leakage_rate == approx(1 - 27 / 29)


def test_unknown_mode():
    with pytest.raises(KeyError):
        evaluate_mode(GOLD, PRED, "fuzzy")
    with pytest.raises(KeyError):
        evaluate_document(GOLD, PRED).mode("fuzzy")


def test_counts_zero_division():
    assert (Counts().precision, Counts().recall, Counts().f1) == (0.0, 0.0, 0.0)
