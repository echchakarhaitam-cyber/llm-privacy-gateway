"""Deterministic splits, reproducible samples, JSONL round trips and evaluation-set drawing."""

import pytest

from privgate.data import splits
from privgate.data.documents import Document
from privgate.data.splits import (
    EVALUATION_SETS,
    EvaluationSetSpec,
    assign_split,
    read_jsonl,
    sample,
    split_key,
    write_jsonl,
)
from privgate.spans import EntityType, Span


def docs(n, prefix="doc"):
    return [Document(f"{prefix}-{i:05d}", f"text {i}", (), "synthetic") for i in range(n)]


def test_assign_split_is_deterministic_and_known():
    assert assign_split("abc") == assign_split("abc")
    # sha256("abc") ends in ...15ad; as an integer it is 65 mod 100 -> train.
    assert assign_split("abc") == "train"
    assert {assign_split(f"id-{i}") for i in range(200)} == {"train", "dev"}


def test_dev_share_is_about_ten_percent():
    ids = [f"synthetic-{i}" for i in range(10_000)]
    share = sum(assign_split(i) == "dev" for i in ids) / len(ids)
    assert 0.09 <= share <= 0.11


def test_split_key_prefers_group_id():
    twin_us = Document("nemotron:u1:us", "a", (), "s", {"group_id": "u1"})
    twin_intl = Document("nemotron:u1:intl", "b", (), "s", {"group_id": "u1"})
    assert split_key(twin_us) == split_key(twin_intl) == "u1"
    assert split_key(Document("gretel:7", "c", (), "s")) == "gretel:7"


def test_sample_is_reproducible_and_order_independent():
    pool = docs(500)
    a = sample(pool, 50, seed=0)
    b = sample(list(reversed(pool)), 50, seed=0)
    assert [d.doc_id for d in a] == [d.doc_id for d in b]
    assert [d.doc_id for d in sample(pool, 50, seed=1)] != [d.doc_id for d in a]
    assert len({d.doc_id for d in a}) == 50


def test_sample_edge_cases():
    pool = docs(3)
    assert sample(pool, 0, seed=0) == []
    assert sorted(d.doc_id for d in sample(pool, 3, seed=0)) == [d.doc_id for d in pool]
    with pytest.raises(ValueError):
        sample(pool, 4, seed=0)
    with pytest.raises(ValueError):
        sample(pool + pool[:1], 1, seed=0)


def test_named_evaluation_sets():
    by_name = {s.name: s for s in EVALUATION_SETS}
    assert (by_name["nemotron_dev_2k"].size, by_name["nemotron_dev_2k"].seed) == (2000, 0)
    assert (by_name["nemotron_test_5k"].size, by_name["nemotron_test_5k"].split) == (5000, "test")
    assert by_name["gretel_en_test"].size is None


def test_jsonl_round_trip_and_stable_hash(tmp_path):
    original = [
        Document("a", "Zoë Müller, 東京", (Span(0, 10, EntityType.PERSON, source="s"),), "s",
                 {"domain": "x"}),
        Document("b", "line\r\nbreak", (), "s"),
    ]
    h1 = write_jsonl(original, tmp_path / "one.jsonl")
    h2 = write_jsonl(original, tmp_path / "two.jsonl")
    assert h1 == h2 == splits.file_sha256(tmp_path / "one.jsonl")
    assert b"\r\n" not in (tmp_path / "one.jsonl").read_bytes().replace(b"\\r\\n", b"")
    assert read_jsonl(tmp_path / "one.jsonl") == original


def test_build_evaluation_sets_on_fixtures(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import parquet_fixtures as fx

    root = tmp_path / "data"
    rows = [fx.nemotron_row(f"g{i:03d}", loc, f"Person {i} {loc}", [])
            for loc in ("us", "intl") for i in range(60)]
    for key, write, content in (
        ("nemotron_train_file", fx.write_nemotron, rows),
        ("nemotron_test_file", fx.write_nemotron, rows[:10]),
        ("gretel_en_test_file", fx.write_gretel, fx.default_gretel_rows()),
    ):
        path = root / splits.RAW_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, content)
    monkeypatch.setattr(splits, "EVALUATION_SETS", (
        EvaluationSetSpec("dev_small", "nemotron_train_file", "dev", 4, 0, ""),
        EvaluationSetSpec("test_small", "nemotron_test_file", "test", 3, 0, ""),
        EvaluationSetSpec("gretel_all", "gretel_en_test_file", "test", None, None, ""),
    ))
    sets, reports = splits.build_evaluation_sets(root)
    again, _ = splits.build_evaluation_sets(root)
    assert {k: [d.doc_id for d in v] for k, v in sets.items()} == {
        k: [d.doc_id for d in v] for k, v in again.items()}
    assert [len(sets[k]) for k in ("dev_small", "test_small", "gretel_all")] == [4, 3, 2]
    assert all(assign_split(split_key(d)) == "dev" for d in sets["dev_small"])
    assert reports["nemotron_train_file"].documents == 120
    # twins share a group, so every dev group contributes both renderings to the dev pool
    pairs, _ = splits.iter_file(root, "nemotron_train_file")
    dev_ids = [d.doc_id for split, d in pairs if split == "dev"]
    groups = {i.split(":")[1] for i in dev_ids}
    assert len(dev_ids) == 2 * len(groups)
