"""Real-data training checks (minutes): run with ``pytest -m slow``.

Skipped when the raw Nemotron train file or the evaluation sets are absent.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("sklearn_crfsuite")

from privgate.bench.sets import DEV_SET, load_evaluation_set
from privgate.data.splits import RAW_FILES, assign_split, file_sha256, read_jsonl
from privgate.ml import training
from privgate.ml.crf import CRFDetector

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if not (DATA / RAW_FILES["nemotron_train_file"]).exists():
    pytest.skip("raw Nemotron train file absent", allow_module_level=True)


@pytest.fixture(scope="module")
def forbidden():
    return training.evaluation_doc_ids(DATA)


def test_real_subset_training_is_guarded_and_reproducible(forbidden):
    uids = pq.read_table(DATA / RAW_FILES["nemotron_train_file"], columns=["uid"])
    pool = sorted({u for u in uids.column(0).to_pylist() if assign_split(u) == "train"})
    groups = training.sample_groups(pool, 60, seed=0)
    docs = training.subset(training.load_train_split(DATA, frozenset(groups)), groups)
    assert len(docs) == 60
    training.check_training_pool(docs, forbidden)
    a = CRFDetector.train(docs, c1=0.1, c2=0.05, max_iterations=20, seed=0)
    b = CRFDetector.train(docs, c1=0.1, c2=0.05, max_iterations=20, seed=0)
    assert a.transition_weights() == b.transition_weights()
    dev, _ = load_evaluation_set(DEV_SET, DATA)
    assert [a.detect(d.text) for d in dev[:20]] == [b.detect(d.text) for d in dev[:20]]


def test_recorded_training_subsets_match_and_pass_the_guard(forbidden):
    record = ROOT / "results" / "m4" / "training_sets.json"
    if not record.exists():
        pytest.skip("training subsets not built yet")
    for entry in json.loads(record.read_text(encoding="utf-8"))["subsets"].values():
        path = ROOT / entry["path"]
        if not path.exists():
            continue
        assert file_sha256(path) == entry["sha256"]
        docs = read_jsonl(path)
        assert len(docs) == entry["documents"] == entry["n"]
        training.check_training_pool(docs, forbidden)
