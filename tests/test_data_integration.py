"""Integration tests on the real pinned files: run with ``pytest -m data``.

Skipped when data/raw/ is absent (run scripts/download_data.py first). Loading both Nemotron
files takes about two minutes, so the sets are built once per module.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data

pq = pytest.importorskip("pyarrow.parquet")

import parquet_fixtures as fx

from privgate.data import splits
from privgate.data.documents import LoadOptions
from privgate.data.mapping import GRETEL_LABEL_MAP, NEMOTRON_LABEL_MAP

DATA = Path(__file__).resolve().parents[1] / "data"
_missing = [k for k, p in splits.RAW_FILES.items() if not (DATA / p).exists()]
if _missing:
    pytest.skip(f"real datasets absent: {_missing}", allow_module_level=True)

# Thresholds justified by the values observed on the pinned files (docs/DATA.md):
# offset mismatches that are not case-only were 0 of 825,456 (train) and 0 of 850,340
# (test) spans; 0.1% leaves room for an upstream fix-up while any systematic offset shift
# would exceed it by orders of magnitude. Case-only mismatches were 0.82% and 0.73%.
MAX_OFFSET_MISMATCH_RATE = 0.001
MAX_CASE_ONLY_RATE = 0.02


@pytest.fixture(scope="module")
def built():
    """All evaluation sets plus the LoadReport of each raw file, labels checked strictly."""
    return splits.build_evaluation_sets(DATA, LoadOptions(strict_labels=True))


def path(key):
    return DATA / splits.RAW_FILES[key]


def test_real_schemas_match_the_unit_test_fixtures():
    for key in ("nemotron_train_file", "nemotron_test_file"):
        assert pq.read_schema(path(key)).remove_metadata() == fx.nemotron_schema()
    assert pq.read_schema(path("gretel_en_test_file")).remove_metadata() == fx.gretel_schema()


def test_every_raw_label_is_mapped(built):
    _, reports = built  # strict_labels=True: an unmapped label would have raised already
    nemotron = set(reports["nemotron_train_file"].raw_label_counts) | set(
        reports["nemotron_test_file"].raw_label_counts)
    assert nemotron == set(NEMOTRON_LABEL_MAP)
    assert set(reports["gretel_en_test_file"].raw_label_counts) == set(GRETEL_LABEL_MAP)
    assert all(not r.unknown_labels for r in reports.values())


def test_offset_mismatch_rate_is_below_threshold(built):
    _, reports = built
    for key in ("nemotron_train_file", "nemotron_test_file"):
        r = reports[key]
        mismatch = r.spans_dropped["offset_mismatch"] / r.spans_read
        case_only = r.case_only_mismatches / r.spans_read
        print(f"{key}: offset mismatch {mismatch:.4%}, case-only {case_only:.4%}")
        assert mismatch <= MAX_OFFSET_MISMATCH_RATE
        assert case_only <= MAX_CASE_ONLY_RATE


def test_load_reports_are_consistent(built):
    _, reports = built
    for r in reports.values():
        assert r.is_consistent()
        assert r.spans_dropped["out_of_bounds"] == 0


def test_evaluation_set_sizes(built):
    sets, _ = built
    gretel_rows = pq.ParquetFile(path("gretel_en_test_file")).metadata.num_rows
    assert len(sets["nemotron_dev_2k"]) == 2000
    assert len(sets["nemotron_test_5k"]) == 5000
    assert len(sets["gretel_en_test"]) == gretel_rows
    for docs in sets.values():
        assert len({d.doc_id for d in docs}) == len(docs)


def test_dev_set_is_disjoint_from_train(built):
    sets, _ = built
    assert all(splits.assign_split(splits.split_key(d)) == "dev"
               for d in sets["nemotron_dev_2k"])
    test_ids = {d.doc_id for d in sets["nemotron_test_5k"]}
    assert not test_ids & {d.doc_id for d in sets["nemotron_dev_2k"]}


def test_dev_share_of_real_groups():
    uids = set(pq.read_table(path("nemotron_train_file"), columns=["uid"]).column(0).to_pylist())
    share = sum(splits.assign_split(u) == "dev" for u in uids) / len(uids)
    assert 0.09 <= share <= 0.11


def test_evaluation_sets_reproduce_manifest_hashes(built, tmp_path):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if "evaluation_sets" not in manifest:
        pytest.skip("manifest has no evaluation_sets yet; run scripts/build_eval_sets.py")
    sets, _ = built
    for name, docs in sets.items():
        entry = manifest["evaluation_sets"][name]
        assert splits.write_jsonl(docs, tmp_path / f"{name}.jsonl") == entry["sha256"]
        assert entry["documents"] == len(docs)
