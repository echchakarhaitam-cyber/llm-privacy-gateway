"""RQ11 plumbing: Gretel train/test overlap, the extended pool guard, ids, nested samples."""

import random

import pytest

from privgate.data import gretel
from privgate.data.documents import Document
from privgate.data.splits import GRETEL_TRAIN_FILE, RAW_FILES, assign_split, write_jsonl
from privgate.ml import training

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

TEST_TEXTS = ["Dear Anna, your account is open.", "Wire 1,000 EUR to Bob.", "Hello world"]
TRAIN = gretel.TRAIN_ID_PREFIX


def gretel_train_doc(i, text):
    return Document(f"{TRAIN}:{i}", text, (), gretel.DATASET)


def group_with(split):
    return next(f"g{i}" for i in range(1000) if assign_split(f"g{i}") == split)


def nemotron_doc(group):
    return Document(f"nemotron:{group}:us", "x", (), "nemotron-pii", {"group_id": group})


@pytest.fixture
def data_root(tmp_path):
    path = tmp_path / RAW_FILES["gretel_en_test_file"]
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"generated_text": TEST_TEXTS}), path)
    return tmp_path


def test_test_hashes_cover_every_row_of_the_test_file(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    assert hashes == frozenset(training.text_hash(t) for t in TEST_TEXTS)


def test_overlap_is_detected_and_dropped(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    docs = [gretel_train_doc(1, "a brand new text"), gretel_train_doc(2, TEST_TEXTS[1]),
            gretel_train_doc(3, "another new text"), gretel_train_doc(4, TEST_TEXTS[0])]
    kept, overlapping = training.split_gretel_overlap(docs, hashes)
    assert [d.doc_id for d in kept] == [f"{TRAIN}:1", f"{TRAIN}:3"]
    assert [d.doc_id for d in overlapping] == [f"{TRAIN}:2", f"{TRAIN}:4"]
    assert training.split_gretel_overlap([], hashes) == ([], [])


def test_overlap_is_exact_not_fuzzy(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    near = gretel_train_doc(1, TEST_TEXTS[2] + " ")        # one trailing space: a different text
    assert training.split_gretel_overlap([near], hashes) == ([near], [])


def test_mixed_pool_accepts_nemotron_train_and_clean_gretel_train(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    docs = [nemotron_doc(group_with("train")), gretel_train_doc(7, "never in the test file")]
    training.check_mixed_training_pool(docs, frozenset(), hashes)


def test_mixed_pool_rejects_gretel_test_ids(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    test_doc = Document("gretel:5", "fresh text", (), gretel.DATASET)
    with pytest.raises(training.TrainingPoolError, match="evaluation set"):
        training.check_mixed_training_pool([test_doc], frozenset({"gretel:5"}), hashes)
    with pytest.raises(training.TrainingPoolError, match="train split"):
        training.check_mixed_training_pool([test_doc], frozenset(), hashes)   # not a train id


def test_mixed_pool_rejects_a_gretel_train_doc_with_test_text(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    with pytest.raises(training.TrainingPoolError, match="text of a Gretel test document"):
        training.check_mixed_training_pool([gretel_train_doc(9, TEST_TEXTS[0])], frozenset(),
                                           hashes)


def test_mixed_pool_still_rejects_nemotron_outside_train_and_evaluation_ids(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    with pytest.raises(training.TrainingPoolError, match="train split"):
        training.check_mixed_training_pool([nemotron_doc(group_with("dev"))], frozenset(),
                                           hashes)
    train = nemotron_doc(group_with("train"))
    with pytest.raises(training.TrainingPoolError, match="evaluation set"):
        training.check_mixed_training_pool([train], frozenset({train.doc_id}), hashes)


def test_mixed_pool_rejects_a_train_prefixed_id_from_another_dataset(data_root):
    hashes = training.gretel_test_text_hashes(data_root)
    fake = Document(f"{TRAIN}:1", "x", (), "nemotron-pii")
    with pytest.raises(training.TrainingPoolError, match="not a Gretel document"):
        training.check_mixed_training_pool([fake], frozenset(), hashes)


def test_gretel_train_ids_never_collide_with_test_ids():
    row = {"__index_level_0__": 12}
    assert gretel.doc_id(row) == "gretel:12"
    assert gretel.doc_id(row, TRAIN) == "gretel-train:12"


def test_train_file_is_kept_out_of_the_evaluation_raw_files():
    assert GRETEL_TRAIN_FILE.name == "English_train-00000-of-00001.parquet"
    assert GRETEL_TRAIN_FILE not in RAW_FILES.values()


def test_job_uses_the_mixed_guard_only_when_asked(data_root, tmp_path, monkeypatch):
    from privgate.ml import jobs
    from privgate.ml.jobs import _training_docs

    monkeypatch.setattr(jobs, "evaluation_doc_ids", lambda root: frozenset())
    path = tmp_path / "train.jsonl"
    sha = write_jsonl([gretel_train_doc(1, TEST_TEXTS[0])], path)
    spec = {"train_path": str(path), "train_sha256": sha, "data_root": str(data_root)}
    with pytest.raises(training.TrainingPoolError):
        _training_docs(spec)                         # default guard: Nemotron train only
    with pytest.raises(training.TrainingPoolError, match="text of a Gretel test document"):
        _training_docs({**spec, "pool": "mixed"})


def test_nested_samples_are_prefixes_of_one_draw():
    ids = sorted(f"{TRAIN}:{i}" for i in range(2000))
    drawn = random.Random(0).sample(ids, 500)
    assert drawn == random.Random(0).sample(ids, 500)
    assert len(set(drawn)) == 500 and set(drawn[:100]) < set(drawn)
