"""Privacy and utility metrics on hand-computed cases; domain-classifier helpers and guard."""

import pytest

from privgate.bench.privacy import leaked_chars, privacy_report, surviving_values
from privgate.bench.utility import non_pii_counts, utility_report
from privgate.data.documents import Document
from privgate.data.splits import assign_split
from privgate.masking import mask
from privgate.spans import EntityType, Span

E = EntityType
TEXT = "Bob Stone met Bob Stone at Acme; card 4111 1111 1111 1111."
#       0         1         2         3         4         5
GOLD_CHARS = 41  # 9 + 9 + 4 + 19
GOLD = (Span(0, 9, E.PERSON), Span(14, 23, E.PERSON), Span(27, 31, E.ORGANIZATION),
        Span(38, 57, E.CREDIT_CARD))
DOC = Document("d1", TEXT, GOLD, "synthetic")


def test_detector_missing_one_mention():
    # the detector finds the first "Bob Stone" and the card, misses the second mention
    m = mask(TEXT, [GOLD[0], GOLD[3]], "typed")
    chars = leaked_chars(DOC, m)
    assert chars["PERSON"] == [18, 9]                # only the missed mention leaks
    assert chars["ORGANIZATION"] == [4, 4] and chars["CREDIT_CARD"] == [19, 0]
    values = surviving_values(DOC, m)
    assert values["PERSON"] == [1, 1]                # the value survives verbatim
    assert values["ORGANIZATION"] == [1, 1] and values["CREDIT_CARD"] == [1, 0]
    rep = privacy_report([DOC], [m], n_resamples=20)
    assert rep["char_leakage"] == pytest.approx(13 / 41)
    assert rep["verbatim_survival"] == pytest.approx(2 / 3)
    assert rep["per_label"]["PERSON"]["char_leakage"] == 0.5


def test_oracle_masking_leaks_nothing_and_partial_leaks_exactly_the_kept_chars():
    for strategy in ("redact", "typed"):
        rep = privacy_report([DOC], [mask(TEXT, GOLD, strategy)], n_resamples=20)
        assert rep["char_leakage"] == 0.0 and rep["verbatim_survival"] == 0.0
    part = mask(TEXT, GOLD, "partial")
    # the card keeps 3 spaces + last 4 digits = 7 of 19 characters
    assert leaked_chars(DOC, part)["CREDIT_CARD"] == [19, 7]
    assert privacy_report([DOC], [part], n_resamples=20)["char_leakage"] == pytest.approx(7 / 41)


def test_short_values_are_not_counted_for_verbatim_survival():
    doc = Document("d", "Al met Al.", (Span(0, 2, E.PERSON),), "s")
    assert surviving_values(doc, mask(doc.text, [], "typed")) == {}


def test_preserved_non_pii_and_length_ratio():
    m = mask(TEXT, list(GOLD), "redact")
    non_pii, kept = non_pii_counts(DOC, m)
    assert (non_pii, kept) == (len(TEXT) - 41, len(TEXT) - 41)
    # a false positive covering " met " destroys 5 non-PII characters
    fp = mask(TEXT, [Span(9, 14, E.OTHER)], "typed")
    assert non_pii_counts(DOC, fp) == (len(TEXT) - 41, len(TEXT) - 46)
    rep = utility_report([DOC], [m])
    assert rep["length_ratio"] == pytest.approx(len(m.text) / len(TEXT))
    assert rep["preserved_non_pii"] == 1.0
    assert utility_report([DOC], [mask(TEXT, [], "typed")])["length_ratio"] == 1.0
    with pytest.raises(ValueError):
        utility_report([DOC], [])


def test_paired_accuracy_difference():
    pytest.importorskip("sklearn")
    from privgate.bench.domain import accuracy_and_macro_f1, paired_accuracy_difference

    gold = ["a", "b"] * 50
    perfect, half = list(gold), [g if i % 2 else "x" for i, g in enumerate(gold)]
    d = paired_accuracy_difference(gold, perfect, half, n_resamples=200)
    assert d["difference"] == 0.5 and d["ci_excludes_zero"] and d["ci95"][0] > 0
    assert d == paired_accuracy_difference(gold, perfect, half, n_resamples=200)
    assert accuracy_and_macro_f1(gold, perfect) == {"accuracy": 1.0, "macro_f1": 1.0}


def test_classifier_training_guard():
    pytest.importorskip("sklearn")
    from privgate.bench.domain import train_classifier
    from privgate.ml.training import TrainingPoolError

    dev_group = next(f"g{i}" for i in range(1000) if assign_split(f"g{i}") == "dev")
    train_group = next(f"g{i}" for i in range(1000) if assign_split(f"g{i}") == "train")
    dev_doc = Document(f"nemotron:{dev_group}:us", "x", (), "nemotron-pii",
                       {"group_id": dev_group, "domain": "Finance"})
    with pytest.raises(TrainingPoolError):
        train_classifier([dev_doc], ["x"], frozenset())
    train_doc = Document(f"nemotron:{train_group}:us", "x", (), "nemotron-pii",
                         {"group_id": train_group, "domain": "Finance"})
    with pytest.raises(TrainingPoolError):
        train_classifier([train_doc], ["x"], frozenset({train_doc.doc_id}))


def test_cross_validation_selects_on_train_only():
    pytest.importorskip("sklearn")
    from privgate.bench.domain import cross_validate

    texts = [f"invoice payment bank {i}" for i in range(20)] + \
            [f"patient clinic doctor {i}" for i in range(20)]
    labels = ["Finance"] * 20 + ["Healthcare"] * 20
    groups = [f"g{i // 2}" for i in range(40)]
    cv = cross_validate(texts, labels, groups, c_grid=(1.0, 10.0), folds=4)
    assert cv["selected_C"] == 1.0 and all(r["mean_accuracy"] == 1.0 for r in cv["rows"])
