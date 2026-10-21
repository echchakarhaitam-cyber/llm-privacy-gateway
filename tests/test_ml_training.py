"""Training protocol: pool guard, group sampling, selection rule, fixed constants."""

import pytest

from privgate.data.documents import Document
from privgate.data.splits import assign_split
from privgate.ml import training


def group_with(split):
    return next(f"g{i}" for i in range(1000) if assign_split(f"g{i}") == split)


def nemotron_doc(group, locale="us"):
    return Document(f"nemotron:{group}:{locale}", "x", (), "nemotron-pii", {"group_id": group})


def test_training_pool_guard():
    train_doc = nemotron_doc(group_with("train"))
    training.check_training_pool([train_doc], frozenset())
    with pytest.raises(training.TrainingPoolError, match="evaluation set"):
        training.check_training_pool([train_doc], frozenset({train_doc.doc_id}))
    with pytest.raises(training.TrainingPoolError, match="train split"):
        training.check_training_pool([nemotron_doc(group_with("dev"))], frozenset())
    with pytest.raises(training.TrainingPoolError, match="train split"):
        training.check_training_pool([Document("gretel:1", "x", (), "gretel-finance-en")],
                                     frozenset())


def test_sample_groups_reproducible_and_order_independent():
    groups = [f"g{i}" for i in range(100)]
    a = training.sample_groups(groups, 20, seed=1)
    assert a == training.sample_groups(list(reversed(groups)), 20, seed=1)
    assert len(a) == 10 and a != training.sample_groups(groups, 20, seed=2)
    with pytest.raises(ValueError):
        training.sample_groups(groups, 500, seed=0)


def test_selection_rule():
    rows = [{"c1": 0.1, "c2": 0.1, "dev": {"type_agnostic_f1": 0.8, "leakage_rate": 0.3}},
            {"c1": 0.05, "c2": 0.1, "dev": {"type_agnostic_f1": 0.8, "leakage_rate": 0.2}},
            {"c1": 0.3, "c2": 0.01, "dev": {"type_agnostic_f1": 0.7, "leakage_rate": 0.1}}]
    assert training.select(rows, ["c1", "c2"])["c1"] == 0.05      # tie on F1 -> lower leakage
    tie = [{"C": 1.0, "dev": {"type_agnostic_f1": 0.5, "leakage_rate": 0.2}},
           {"C": 0.1, "dev": {"type_agnostic_f1": 0.5, "leakage_rate": 0.2}}]
    assert training.select(tie, ["C"])["C"] == 0.1                # full tie -> smaller C
    with pytest.raises(ValueError):
        training.select([], ["C"])


def test_protocol_constants():
    assert training.CRF_GRID == tuple((a, b) for a in (0.05, 0.1, 0.3) for b in (0.01, 0.05, 0.1))
    assert training.LOGREG_C == (0.1, 1.0, 10.0) and training.SEEDS == (0, 1, 2)
    assert training.CRF_MAX_ITERATIONS == 100 and training.LOGREG_MAX_ITER == 1000
    mem = training.peak_memory_mb()
    assert mem is None or mem > 0
