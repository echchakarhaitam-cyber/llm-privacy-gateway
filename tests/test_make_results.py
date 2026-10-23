"""The consolidated-results generator, on fixture JSON (no real data, no matplotlib needed)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SETS = ("nemotron_dev_2k", "nemotron_test_5k", "gretel_en_test")


def load_script():
    path = Path(__file__).parent.parent / "scripts" / "make_results.py"
    spec = importlib.util.spec_from_file_location("make_results", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_results"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def metrics(leak, f1=0.9):
    return {"views": {"all": {
        "chars": {"leakage_rate": leak, "per_gold_label": {
            "PERSON": {"leaked_chars": 30, "leakage_rate": 0.3},
            "OTHER": {"leaked_chars": 70, "leakage_rate": 0.7}}},
        "type_agnostic": {"micro": {"f1": f1}},
        "ci95": {"leakage_rate": [leak - 0.01, leak + 0.01],
                 "type_agnostic_micro_f1": [f1 - 0.01, f1 + 0.01]}}}}


def pair(a, b, delta):
    return {"a": a, "b": b, "delta": delta, "ci95": [delta - 0.01, delta + 0.01]}


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "results"
    leaks = {"regex": 0.7, "presidio": 0.4, "regex+presidio": 0.35, "spacy": 0.6,
             "regex+spacy": 0.4, "crf": 0.07, "crf+regex+presidio": 0.05}
    for s in SETS:
        for det, leak in leaks.items():
            base = "m4" if det.startswith("crf") else "m3"
            gretel = 1.4 if s == "gretel_en_test" else 1.0
            write(r / base / s / det / "metrics.json", metrics(min(0.99, leak * gretel)))
        for model, leak in {"crf_g500": 0.2, "crf_n5k_g100": 0.2, "crf_n5k_g500": 0.15}.items():
            write(r / "m7" / s / model / "metrics.json", metrics(leak))
            write(r / "m7" / s / f"paired_{model}__vs__crf.json",
                  {"leakage_rate": pair(leak, 0.28, leak - 0.28),
                   "type_agnostic_f1": pair(0.9, 0.85, 0.05)})
        write(r / "m4" / s / "paired_crf__vs__regex+presidio.json",
              {"leakage_rate": pair(0.07, 0.35, -0.28)})
    write(r / "m7" / "training_sets.json", {
        "gretel_train_documents": 20000, "gretel_train_sha256": "ab" * 32,
        "overlap_with_gretel_test": 0, "overlap_action": "none needed", "usable_documents": 20000,
        "seed": 0})
    write(r / "m7" / "training_log.json", {"models": {
        "crf_g500": {"documents": 500, "wall_seconds": 60.0, "training_seconds": 50.0,
                     "peak_memory_mb": 800.0}}, "skipped": {"crf_n5k_g500": "no budget"}})
    write(r / "m4" / "learning_curve.json", {"rows": [
        {"set": "nemotron_dev_2k", "n": 1000, "type_agnostic_f1": {"mean": 0.91},
         "leakage_rate": {"mean": 0.09}}]})
    write(r / "m5" / "masking.json", {"rows": [{
        "set": "nemotron_dev_2k", "detector": "crf+regex+presidio", "strategy": "typed",
        "privacy": {"char_leakage": 0.05, "ci95": {"char_leakage": [0.04, 0.06]}},
        "utility": {"preserved_non_pii": 0.98},
        "domain_original_classifier": {"drop_vs_unmasked": {"difference": 0.01,
                                                            "ci95": [0.0, 0.02]}}}]})
    row = {"requests": 1000, "faults_injected": 10, "unscreened_forwarded": 0,
           "fault_leaked_requests": 0, "availability": 0.9, "q": 0.2, "fault": "exception"}
    write(r / "m6" / "faults.json", {"qs": [0.0, 0.2], "rows": [
        {**row, "mode": "fail_closed"},
        {**row, "mode": "fail_open", "unscreened_forwarded": 10, "fault_leaked_requests": 10}]})
    write(r / "m6" / "latency.json", {"protocol": "recorded-load", "repeats": 3,
                                      "idle_check_before": {}, "rows": [
        {"harness": "asgi", "config": "regex", "concurrency": 1, "p50_ms": 5.0,
         "p95_ms": 9.0, "requests_per_s": 150.0}]})
    write(r / "m7" / "error_taxonomy.json", {"sets": {s: {
        "leaked_spans": 10, "gold_spans": 100, "leaked_share_of_gold_spans": 0.1,
        "dimensions": {"mapped_label": [{"category": "PERSON", "count": 5, "share": 0.5}],
                       "raw_label": [{"category": "first_name", "count": 4, "share": 0.4}],
                       "position": [{"category": "prose", "count": 8, "share": 0.8}]}}
        for s in SETS}})
    return r


def test_headline_has_a_row_per_detector_and_the_best_rq11_model(root):
    mod = load_script()
    text = mod.headline(root)
    for det in mod.HEADLINE:
        assert f"`{det}`" in text
    assert "`crf_n5k_g500` — (b) Nemotron 5000 + Gretel 500" in text      # lowest Gretel leakage
    assert "0.700 [0.690, 0.710]" in text                                  # regex on dev, CI shown
    assert mod.best_rq11(root) == "crf_n5k_g500"


def test_headline_marks_missing_files(root):
    (root / "m3" / "gretel_en_test" / "presidio" / "metrics.json").unlink()
    assert "NOT MEASURED" in load_script().headline(root)


def test_rq11_tables_report_overlap_deltas_and_training(root):
    body, notes = load_script().rq11_tables(root)
    assert "Documents whose text also occurs in the Gretel test file: 0" in body
    assert "none needed" in body and "Skipped: `crf_n5k_g500` (no budget)" in body
    assert "-0.080 [-0.090, -0.070]" in body                               # crf_g500 delta
    assert any("`crf_g500` on `gretel_en_test`: leakage 0.280 → 0.200" in n for n in notes)
    assert any("paired CI says lower" in n for n in notes)


def test_rq11_without_results_says_not_measured(tmp_path):
    body, notes = load_script().rq11_tables(tmp_path)
    assert "NOT MEASURED" in body and notes == []


def test_negative_transfer_is_called_out(root):
    write(root / "m7" / "nemotron_dev_2k" / "paired_crf_g500__vs__crf.json",
          {"leakage_rate": pair(0.2, 0.07, 0.13), "type_agnostic_f1": pair(0.8, 0.9, -0.1)})
    notes = load_script().rq11_tables(root)[1]
    assert any("`crf_g500` on `nemotron_dev_2k`" in n and "paired CI says higher" in n
               for n in notes)


def test_consolidated_document_has_every_section_and_computed_numbers(root):
    doc = load_script().build_results(root)
    for heading in ("## Summary", "## Headline", "### RQ1-RQ3", "### RQ4-RQ6", "### RQ7-RQ8",
                    "### RQ9", "### RQ10", "### RQ11", "### Final error analysis",
                    "## Threats to validity", "## What this project does NOT show"):
        assert heading in doc
    assert "Fail-closed: 0 unscreened requests forwarded and 0 fault-induced leaks" in doc
    assert "measured under recorded background load (not an idle machine); median of 3 runs" in doc
    assert "largest raw label `first_name` (40.0%)" in doc
    for link in ("results-detection.md", "results-masking.md", "results-gateway.md",
                 "results-adaptation.md", "error-analysis.md"):
        assert link in doc


def test_document_degrades_to_not_measured_without_results(tmp_path):
    doc = load_script().build_results(tmp_path)
    assert doc.count("NOT MEASURED") >= 5 and "## Summary\n\nNOT MEASURED" in doc


def test_idle_protocol_label(root):
    data = json.loads((root / "m6" / "latency.json").read_text(encoding="utf-8"))
    data["protocol"] = "idle"
    write(root / "m6" / "latency.json", data)
    assert "Idle protocol." in load_script().rq10_table(root)


def test_skipped_idle_check_is_never_presented_as_a_measurement(root):
    data = json.loads((root / "m6" / "latency.json").read_text(encoding="utf-8"))
    data["idle_check_before"] = {"skipped": True}
    write(root / "m6" / "latency.json", data)
    assert load_script().rq10_table(root) == "NOT MEASURED"


def test_adaptation_document(root):
    doc = load_script().build_adaptation(root)
    assert doc.startswith("# Domain adaptation (RQ11)") and "## Observations" in doc


def test_figure_is_written_when_matplotlib_exists(root, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "fig" / "leakage.png"
    assert load_script().figure(root, out) is True
    assert out.exists() and out.stat().st_size > 1000


# --- comparison sentences are generated from the numbers, never typed ---------------------------

@pytest.mark.parametrize("a,b,word", [
    (0.282, 0.318, "less than"), (0.318, 0.282, "more than"), (0.3181, 0.3179, "the same as"),
    (0.0, 0.001, "less than"), (0.9996, 0.9994, "more than"), (0.05, 0.0499, "the same as")])
def test_compare_sentence_direction_follows_the_displayed_numbers(a, b, word):
    mod = load_script()
    sentence = mod.compare_sentence("x", a, "y", b)
    assert f" {word} " in sentence
    assert mod.wrong_comparisons(sentence) == []


def test_the_m7_error_would_have_been_caught():
    mod = load_script()
    wrong = "`crf+regex+presidio` leaks less than `crf_n5k_g500` (0.318 vs 0.282)"
    assert mod.wrong_comparisons(wrong) == [wrong]
    right = "`crf_n5k_g500` leaks less than `crf+regex+presidio` (0.282 vs 0.318)"
    assert mod.wrong_comparisons(right) == []
    swapped_tie = "`a` leaks more than `b` (0.300 vs 0.300)"
    assert mod.wrong_comparisons(swapped_tie) == [swapped_tie]


def test_generated_documents_never_contain_a_wrong_comparison(root):
    mod = load_script()
    for name, leak in (("crf_n5k_g500+regex+presidio", 0.19),):
        for s in SETS:
            write(root / "m7" / s / name / "metrics.json", metrics(leak))
            write(root / "m7" / s / f"paired_{name}__vs__crf.json",
                  {"leakage_rate": pair(leak, 0.28, leak - 0.28),
                   "type_agnostic_f1": pair(0.9, 0.85, 0.05)})
    docs = mod.build_results(root) + mod.build_adaptation(root)
    sentences = mod.COMPARISON.findall(docs)
    assert len(sentences) >= 4                  # the comparisons are really there
    assert mod.wrong_comparisons(docs) == []
    # the fixture's union (0.318 on gretel, scaled) is above the adapted 0.15 / 0.19 models
    assert "`crf_n5k_g500` leaks less than `crf+regex+presidio`" in docs or \
        "`crf_n5k_g500` leaks more than `crf+regex+presidio`" in docs


def test_direction_flips_with_the_data(root):
    mod = load_script()
    for s in SETS:
        write(root / "m7" / s / "crf_n5k_g500" / "metrics.json", metrics(0.9))
    docs = mod.build_results(root)
    assert "`crf_n5k_g500` leaks more than `crf+regex+presidio`" in docs
    for s in SETS:
        write(root / "m7" / s / "crf_n5k_g500" / "metrics.json", metrics(0.01))
    assert "`crf_n5k_g500` leaks less than `crf+regex+presidio`" in mod.build_results(root)


def test_headline_marks_adapted_rows(root):
    mod = load_script()
    for s in SETS:
        write(root / "m7" / s / mod.RQ11_ENSEMBLE / "metrics.json", metrics(0.1))
    text = mod.headline(root)
    assert f"`{mod.RQ11_ENSEMBLE}` — (b) adapted CRF + regex + Presidio" in text
    assert "adapted models, evaluated on sets that include the Gretel test set" in text


def test_committed_docs_have_no_wrong_comparison():
    mod = load_script()
    docs = Path(__file__).parent.parent / "docs"
    files = [*docs.glob("*.md"), Path(__file__).parent.parent / "README.md"]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert mod.wrong_comparisons(text) == [], path.name


# --- README (make_results.py --readme) -----------------------------------------------------------

def load_readme_builder():
    load_script()                      # registers "make_results" for readme_builder's import
    path = Path(__file__).parent.parent / "scripts" / "readme_builder.py"
    spec = importlib.util.spec_from_file_location("readme_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["readme_builder"] = module
    spec.loader.exec_module(module)
    return module


FACTS = {
    "quickstart_detect": {"code": "print('hi')\n", "output": "hi"},
    "quickstart_gateway": {"command": "python -m privgate.gateway --mock-upstream",
                           "curl": "curl -s http://127.0.0.1:8088/v1/chat/completions",
                           "response_headers": ["HTTP/1.1 200 OK", "x-privgate-request-id: ab"],
                           "response_json": {"privgate": {"masked_entities": {"EMAIL": 1}}},
                           "healthz": {"status": "ok"}},
    "tests": {"default": {"command": "pytest -q", "summary": "7 passed in 1.00s"}},
}


def complete(root):
    """Add what the README summary needs beyond the base fixture."""
    masking = json.loads((root / "m5" / "masking.json").read_text(encoding="utf-8"))
    typed = masking["rows"][0]
    for strategy, leak in (("redact", 0.05), ("partial", 0.13)):
        masking["rows"].append({**typed, "strategy": strategy,
                                "privacy": {"char_leakage": leak,
                                            "ci95": {"char_leakage": [leak, leak]}}})
    write(root / "m5" / "masking.json", masking)
    latency = json.loads((root / "m6" / "latency.json").read_text(encoding="utf-8"))
    latency["rows"] = [{"harness": h, "config": c, "concurrency": 1, "p50_ms": 2.0 + i,
                        "p95_ms": 5.0 + i, "requests_per_s": 100.0 - i}
                       for h in ("asgi", "server")
                       for i, c in enumerate(("none", "regex", "crf", "crf+regex",
                                              "crf+regex+presidio"))]
    write(root / "m6" / "latency.json", latency)
    for s in SETS:
        write(root / "m7" / s / "crf_n5k_g500+regex+presidio" / "metrics.json", metrics(0.1))
    return root


def test_readme_has_every_section_in_order(root):
    rb = load_readme_builder()
    complete(root)
    text = rb.build_readme(root, FACTS)
    positions = [text.index(f"\n## {name}\n") for name in rb.SECTIONS]
    assert positions == sorted(positions)
    assert text.startswith("<!-- Generated by")
    assert "actions/workflows/ci.yml/badge.svg" in text and "licence-MIT" in text
    assert text.count("\n") <= 300


def test_readme_numbers_come_from_the_results_and_facts(root):
    rb = load_readme_builder()
    complete(root)
    text = rb.build_readme(root, FACTS)
    assert "0.350 / 0.350 / 0.490" in text            # regex+presidio from the fixture
    assert "`pytest -q`: 7 passed in 1.00s" in text   # captured test summary, not typed
    assert "x-privgate-request-id: ab" in text and '"masked_entities": {"EMAIL": 1}' in text
    assert load_script().wrong_comparisons(text) == []


def test_readme_says_not_measured_without_facts(root):
    text = load_readme_builder().build_readme(root, {})
    assert "NOT MEASURED (run `python scripts/capture_readme_facts.py`)" in text


def test_readme_never_names_a_company_or_private_project(root):
    text = " ".join(load_readme_builder().build_readme(complete(root), FACTS).lower().split())
    assert "internship work on privacy-preserving llm platforms" in text
    for word in ("confidential", "proprietary", "employer", "our company"):
        assert word not in text


def test_committed_readme_is_the_generated_one():
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("<!-- Generated by `python scripts/make_results.py --readme`")
    assert load_script().wrong_comparisons(readme) == []
