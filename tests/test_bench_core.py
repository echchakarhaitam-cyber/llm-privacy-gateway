"""Building blocks of the benchmark: bootstrap, latency, metrics views, sets, sweep rule,
raw-label support, raw-label index, error sampling and the runner helpers."""

import json
from dataclasses import dataclass

import pytest

from privgate.bench import errors, rawlabels, runner, sets, support, sweep
from privgate.bench.bootstrap import bootstrap_cis, f1_from_counts, percentile
from privgate.bench.latency import environment, latency_stats
from privgate.bench.metrics import evaluate_run
from privgate.data.documents import Document, RawSpan
from privgate.data.splits import write_jsonl
from privgate.detectors.ensemble import EnsembleDetector
from privgate.detectors.presidio import PresidioDetector
from privgate.evaluation import Counts
from privgate.spans import EntityType, Span

E = EntityType


# --- bootstrap ---------------------------------------------------------------------------

def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([5.0], 0.95) == 5.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_bootstrap_is_deterministic_and_seed_dependent():
    cols = {"x": [1, 0, 3, 2, 5, 0, 1, 4], "n": [1] * 8}
    stats = {"mean": lambda s: s["x"] / s["n"]}
    a = bootstrap_cis(cols, stats, n_resamples=500, seed=0)
    assert a == bootstrap_cis(cols, stats, n_resamples=500, seed=0)
    assert a != bootstrap_cis(cols, stats, n_resamples=500, seed=1)
    lo, hi = a["mean"]
    assert lo <= 2.0 <= hi


def test_bootstrap_of_constant_data_is_a_point():
    ci = bootstrap_cis({"x": [2, 2, 2]}, {"x": lambda s: s["x"] / 3}, n_resamples=50)
    assert ci["x"] == (2.0, 2.0)


def test_bootstrap_input_errors():
    with pytest.raises(ValueError):
        bootstrap_cis({"a": [1], "b": [1, 2]}, {})
    with pytest.raises(ValueError):
        bootstrap_cis({"a": []}, {})


@pytest.mark.parametrize("tp, fp, fn", [(0, 0, 0), (3, 1, 2), (0, 4, 1), (5, 0, 0)])
def test_f1_from_counts_matches_counts(tp, fp, fn):
    assert f1_from_counts(tp, fp, fn) == pytest.approx(Counts(tp, fp, fn).f1)


# --- latency -----------------------------------------------------------------------------

def test_latency_stats_on_fixed_timings():
    timings = [10**9] * 20 + [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    chars = [999] * 20 + [100] * 5
    stats = latency_stats(timings, chars)
    assert stats["documents_timed"] == 5 and stats["warmup_documents_excluded"] == 20
    assert stats["median_ms"] == 3.0
    assert stats["p95_ms"] == pytest.approx(4.8)
    assert stats["mean_ms"] == 3.0
    assert stats["chars_per_second"] == pytest.approx(500 / 0.015)


def test_latency_needs_documents_after_warmup():
    with pytest.raises(ValueError):
        latency_stats([1] * 20, [1] * 20)
    with pytest.raises(ValueError):
        latency_stats([1, 2], [1])


def test_environment_describes_machine():
    env = environment()
    assert {"cpu", "os", "python", "packages", "logical_cpus"} <= set(env)
    assert env["packages"]["privgate"]


# --- metrics views -----------------------------------------------------------------------

TEXT = "Anna Lee, anna@example.org, Paris"
GOLD = (Span(0, 8, E.PERSON), Span(10, 26, E.EMAIL), Span(28, 33, E.LOCATION))


def doc(i=0, spans=GOLD, text=TEXT):
    return Document(f"d{i}", text, spans, "synthetic")


def test_supported_view_restricts_gold_only():
    docs = [doc()]
    preds = [[Span(10, 26, E.EMAIL), Span(0, 4, E.EMAIL)]]  # second hits an unsupported gold
    views = evaluate_run(docs, preds, {E.EMAIL}, n_resamples=20)
    all_ta = views["all"]["type_agnostic"]["micro"]
    sup_ta = views["supported"]["type_agnostic"]["micro"]
    assert (all_ta["tp"], all_ta["fp"], all_ta["fn"]) == (2, 0, 1)
    assert (sup_ta["tp"], sup_ta["fp"], sup_ta["fn"]) == (1, 1, 0)
    assert views["supported"]["gold_spans"] == 1 and views["all"]["gold_spans"] == 3
    assert views["supported"]["chars"]["leakage_rate"] == 0.0
    assert views["all"]["chars"]["leakage_rate"] == pytest.approx(1 - 20 / 29)


def test_metrics_contents_per_label_leakage_and_raw_labels():
    docs = [doc(0), doc(1)]
    preds = [[Span(10, 26, E.EMAIL)], []]
    raw = {"d0": ["first_name", "email", "city"], "d1": ["first_name", "email", "city"]}
    view = evaluate_run(docs, preds, {E.EMAIL}, raw, n_resamples=20)["all"]
    assert set(view) >= {"strict", "partial", "type_agnostic", "chars", "ci95"}
    assert set(view["strict"]) == {"micro", "macro_f1", "per_label"}
    assert view["strict"]["per_label"]["EMAIL"]["tp"] == 1
    assert view["chars"]["per_gold_label"]["EMAIL"] == {
        "gold_chars": 32, "leaked_chars": 16, "leakage_rate": 0.5}
    assert view["chars"]["per_gold_label"]["PERSON"]["leakage_rate"] == 1.0
    assert view["chars"]["per_raw_label"]["city"]["gold_chars"] == 10
    assert set(view["ci95"]) == {"strict_micro_f1", "partial_micro_f1",
                                 "type_agnostic_micro_f1", "leakage_rate"}
    lo, hi = view["ci95"]["leakage_rate"]
    assert lo <= view["chars"]["leakage_rate"] <= hi


# --- evaluation-set integrity ------------------------------------------------------------

def make_root(tmp_path, docs_by_set):
    root = tmp_path / "data"
    entries = {}
    for name, docs in docs_by_set.items():
        entries[name] = {"sha256": write_jsonl(docs, sets.set_path(root, name))}
    (root / "manifest.json").write_text(json.dumps({"evaluation_sets": entries}))
    return root


def test_verified_set_loads(tmp_path):
    root = make_root(tmp_path, {"nemotron_dev_2k": [doc(0), doc(1)]})
    docs, sha = sets.load_evaluation_set("nemotron_dev_2k", root)
    assert [d.doc_id for d in docs] == ["d0", "d1"] and len(sha) == 64


def test_tampered_set_aborts(tmp_path):
    root = make_root(tmp_path, {"nemotron_dev_2k": [doc(0)]})
    path = sets.set_path(root, "nemotron_dev_2k")
    path.write_bytes(path.read_bytes().replace(b"Anna", b"Anne"))
    with pytest.raises(sets.SetIntegrityError, match="refusing"):
        sets.load_evaluation_set("nemotron_dev_2k", root)


def test_missing_set_or_manifest_entry_aborts(tmp_path):
    root = make_root(tmp_path, {"nemotron_dev_2k": [doc(0)]})
    with pytest.raises(sets.SetIntegrityError):
        sets.load_evaluation_set("gretel_en_test", root)
    sets.set_path(root, "nemotron_dev_2k").unlink()
    with pytest.raises(sets.SetIntegrityError, match="missing"):
        sets.load_evaluation_set("nemotron_dev_2k", root)


# --- threshold sweep ---------------------------------------------------------------------

def row(t, leak, f1):
    return {"threshold": t, "leakage_rate": leak, "type_agnostic_f1": f1}


def test_select_threshold_rule():
    rows = [row(0.0, 0.200, 0.50), row(0.3, 0.205, 0.60), row(0.5, 0.209, 0.70),
            row(0.7, 0.300, 0.90)]
    assert sweep.select_threshold(rows) == 0.5          # 0.7 leaks too much
    tie = [row(0.0, 0.2, 0.6), row(0.4, 0.2, 0.6)]
    assert sweep.select_threshold(tie) == 0.0          # tie -> lower threshold
    with pytest.raises(ValueError):
        sweep.select_threshold([])


@dataclass
class Result:
    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def get_supported_entities(self, language):
        return ["PERSON", "EMAIL_ADDRESS"]


def test_sweep_rows_from_one_pass():
    det = PresidioDetector(analyzer=FakeAnalyzer())
    raw = [[Result("PERSON", 0, 8, 0.85), Result("EMAIL_ADDRESS", 10, 26, 1.0),
            Result("PERSON", 28, 33, 0.3)]]
    rows = sweep.sweep_rows(det, raw, [doc()], thresholds=(0.0, 0.5, 1.0))
    assert [r["pred_spans"] for r in rows] == [3, 2, 1]
    assert rows[0]["leakage_rate"] == 0.0 and rows[2]["leakage_rate"] > rows[1]["leakage_rate"]
    assert rows[0]["strict_f1"] < rows[1]["strict_f1"]  # the 0.3 PERSON on Paris is wrong


# --- raw-label support (RQ3) ---------------------------------------------------------------

def test_targeted_raw_labels():
    nem = "nemotron-pii"
    regex = support.targeted_raw_labels(["regex"], {E.EMAIL, E.PHONE}, nem)
    assert {"email", "phone_number", "fax_number"} <= regex and "ssn" not in regex
    spacy = support.targeted_raw_labels(["spacy"], {E.PERSON, E.OTHER}, nem)
    assert {"first_name", "time", "religious_belief"} <= spacy
    assert "occupation" not in spacy and "password" not in spacy   # OTHER is not blanket
    pres = support.targeted_raw_labels(["presidio"], {E.ID_NUMBER, E.OTHER}, nem)
    assert {"ssn", "mac_address", "tax_id"} <= pres and "customer_id" not in pres


def test_untargeted_by_all():
    left = support.untargeted_by_all({
        "regex": (["regex"], {E.EMAIL}),
        "presidio": (["presidio"], {E.PERSON, E.ID_NUMBER, E.OTHER}),
    }, "nemotron-pii")
    assert {"occupation", "password", "customer_id"} <= left
    assert not {"ssn", "email", "first_name", "race_ethnicity"} & left
    with pytest.raises(ValueError):
        support.native_targets("unknown")


# --- raw-label index ---------------------------------------------------------------------

def test_primary_raw_label_of_merged_and_overlapping_spans():
    d = Document("d", "John Smith in New York City", (Span(0, 10, E.PERSON),
                                                       Span(14, 27, E.LOCATION)), "s")
    raw = [RawSpan(0, 4, "first_name"), RawSpan(5, 10, "last_name"),
           RawSpan(14, 22, "state"), RawSpan(14, 27, "city"), RawSpan(0, 10, "company_name")]
    assert rawlabels.primary_raw_labels(d, raw, "nemotron-pii") == ["last_name", "city"]
    assert rawlabels.primary_raw_labels(d, [], "nemotron-pii") == [rawlabels.UNKNOWN] * 2


def test_raw_label_index_from_parquet_and_cache(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import parquet_fixtures as fx

    from privgate.data import splits
    from privgate.data.nemotron import load_nemotron

    root = tmp_path / "data"
    path = root / splits.RAW_FILES["nemotron_train_file"]
    path.parent.mkdir(parents=True)
    fx.write_nemotron(path, fx.default_nemotron_rows())
    docs, _ = load_nemotron(path)
    (root / "processed").mkdir()
    index = rawlabels.load_raw_labels(root, "nemotron_dev_2k", docs[:2], "sha-1")
    assert index["nemotron:u1:us"] == ["last_name", "date_of_birth", "email"]
    assert index["nemotron:u2:us"] == ["first_name", "last_name", "company_name", "city"]
    path.unlink()  # cached now: no raw file needed for the same set hash
    assert rawlabels.load_raw_labels(root, "nemotron_dev_2k", docs[:2], "sha-1") == index
    assert rawlabels.load_raw_labels(root, "nemotron_dev_2k", docs[:2], "sha-2") is None


# --- error analysis ----------------------------------------------------------------------

def test_errors_are_collected_sampled_and_rendered():
    docs = [doc(i) for i in range(20)]
    preds = [[Span(10, 26, E.EMAIL), Span(28, 30, E.PERSON)] for _ in docs]
    fns, fps = errors.collect_errors(docs, preds, {d.doc_id: ["a", "b", "c"] for d in docs})
    assert len(fns) == 20 and len(fps) == 0          # Paris prefix overlaps gold: matched
    assert {c.raw_label for c in fns} == {"a"}
    sample = errors.sample_errors(fns)
    assert len(sample) == 15 and sample == errors.sample_errors(list(reversed(fns)))
    md = errors.render_errors("regex", "nemotron_dev_2k", docs, sample, [], (20, 0))
    assert "CC BY 4.0" in md and "NVIDIA" in md and "⟦Anna Lee⟧" in md


def test_excerpt_marks_span_and_line_breaks():
    text = "x" * 50 + "line\nJohn\r\nend" + "y" * 50
    start = text.index("John")
    shown = errors.excerpt(text, Span(start, start + 4, E.PERSON), width=6)
    assert shown == "xline⏎⟦John⟧⏎endy"  # 6 chars each side; CRLF shown as one ⏎


# --- runner helpers ----------------------------------------------------------------------

@pytest.mark.parametrize("spans", [
    [Span(0, 5, E.PERSON), Span(3, 8, E.PERSON)],       # overlap
    [Span(5, 8, E.PERSON), Span(0, 2, E.PERSON)],       # unsorted
    [Span(0, 99, E.PERSON)],                            # beyond text
    ["not a span"],
])
def test_contract_violations_fail_the_run(spans):
    with pytest.raises(runner.ContractError):
        runner.check_spans(spans, TEXT, "bad")


def test_predictions_round_trip_and_alignment(tmp_path):
    docs = [doc(0), doc(1)]
    preds = [[Span(0, 4, E.PERSON, 0.5, "x")], []]
    path = tmp_path / "p.jsonl"
    runner.write_predictions(path, docs, preds, [10, 20])
    assert runner.read_predictions(path, docs) == (preds, [10, 20])
    with pytest.raises(ValueError):
        runner.read_predictions(path, [doc(1), doc(0)])


def test_compose_ensemble_equals_detect():
    a = [[Span(0, 4, E.PERSON, source="a")], [Span(1, 3, E.EMAIL)]]
    b = [[Span(0, 8, E.PERSON, source="b")], []]
    preds, timings = runner.compose_ensemble([a, b], [[5, 6], [7, 8]])
    assert preds == [EnsembleDetector.combine([a[i], b[i]]) for i in range(2)]
    assert timings[0] >= 12 and timings[1] >= 14


def test_log_step_replaces_and_totals(tmp_path):
    runner.log_step(tmp_path, "detect", "s", "regex", 1.0)
    runner.log_step(tmp_path, "detect", "s", "spacy", 2.0)
    runner.log_step(tmp_path, "detect", "s", "regex", 3.0)
    log = json.loads((tmp_path / runner.RUN_LOG).read_text())
    assert len(log["steps"]) == 2 and log["total_seconds"] == 5.0
