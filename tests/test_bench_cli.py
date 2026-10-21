"""The whole benchmark pipeline (``all``) on tiny synthetic sets, with fake models.

RegexDetector is real; spaCy is replaced by a fixed fake and Presidio's analyzer by a fake
returning RecognizerResult-like objects, so no model is needed.
"""

import json
import re
from dataclasses import dataclass

import pytest

from privgate.bench import cli
from privgate.bench.registry import ALL_DETECTORS
from privgate.bench.sets import SET_NAMES, set_path
from privgate.data.documents import Document
from privgate.data.splits import write_jsonl
from privgate.detectors import EnsembleDetector, PresidioDetector, RegexDetector
from privgate.spans import EntityType, Span

E = EntityType
N_DOCS = 25  # more than the 20 warm-up documents


def make_doc(set_name, i):
    text = f"Anna {i} writes to anna{i}@example.org from Lyon."
    email = (text.index("anna"), text.index(" from"))
    lyon = (text.index("Lyon"), text.index("Lyon") + 4)
    spans = (Span(0, 4, E.PERSON), Span(*email, E.EMAIL), Span(*lyon, E.LOCATION))
    return Document(f"{set_name}:{i:03d}", text, spans, "synthetic")


@pytest.fixture
def roots(tmp_path):
    data = tmp_path / "data"
    entries = {}
    for name in SET_NAMES:
        docs = [make_doc(name, i) for i in range(N_DOCS)]
        entries[name] = {"sha256": write_jsonl(docs, set_path(data, name))}
    (data / "manifest.json").write_text(json.dumps({"evaluation_sets": entries}))
    return data, tmp_path / "results"


class FakeSpacy:
    name = "spacy"
    supported_labels = frozenset({E.PERSON, E.LOCATION})

    def detect(self, text):
        return [Span(m.start(), m.end(), E.PERSON, source="spacy")
                for m in re.finditer(r"\bAnna\b", text)]


@dataclass
class Result:
    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def analyze(self, text, language, score_threshold):
        out = [Result("LOCATION", text.index("Lyon"), text.index("Lyon") + 4, 0.85),
               Result("DATE_TIME", 5, 7, 0.4)]               # a wrong, low-score hit
        return [r for r in out if r.score >= score_threshold]

    def get_supported_entities(self, language):
        return ["LOCATION", "DATE_TIME"]


def fake_factory(name, presidio_threshold=0.0):
    if name == "regex":
        return RegexDetector()
    if name == "spacy":
        return FakeSpacy()
    if name == "presidio":
        return PresidioDetector(score_threshold=presidio_threshold, analyzer=FakeAnalyzer())
    raise AssertionError(name)


def run(roots, *args):
    data, out = roots
    return cli.main(["--data-root", str(data), "--out", str(out), *args])


def test_presidio_on_test_set_needs_a_dev_threshold(roots, monkeypatch):
    monkeypatch.setattr(cli, "make_detector", fake_factory)
    with pytest.raises(SystemExit, match="sweep"):
        run(roots, "run", "--set", "nemotron_test_5k", "--detector", "presidio")


def test_full_pipeline(roots, monkeypatch):
    monkeypatch.setattr(cli, "make_detector", fake_factory)
    _, out = roots
    assert run(roots, "all") == 0

    sweep = json.loads((out / "presidio_sweep.json").read_text())
    # 0.5 drops the wrong DATE hit at no leakage cost -> same leakage band, higher F1
    assert sweep["selected_threshold"] == 0.5
    assert sweep["set"] == "nemotron_dev_2k"
    for name in SET_NAMES:
        for det in ALL_DETECTORS:
            m = json.loads((out / name / det / "metrics.json").read_text())
            assert m["set"] == name and m["detector"] == det
            assert set(m["views"]) == {"all", "supported"}
            assert m["latency"]["documents_timed"] == N_DOCS - 20
            assert (out / name / det / "predictions.jsonl").exists()
        presidio = json.loads((out / name / "presidio" / "metrics.json").read_text())
        assert presidio["config"]["score_threshold"] == 0.5
    assert (out / "nemotron_dev_2k" / "presidio" / "raw_results.jsonl").exists()

    # regex finds every email; the regex+spacy ensemble adds the person
    rs = json.loads((out / "gretel_en_test" / "regex+spacy" / "metrics.json").read_text())
    ta = rs["views"]["all"]["type_agnostic"]["micro"]
    assert (ta["tp"], ta["fn"]) == (2 * N_DOCS, N_DOCS)
    assert rs["supported_labels"] == sorted(
        ["CREDIT_CARD", "EMAIL", "IBAN", "IP_ADDRESS", "LOCATION", "PERSON", "PHONE", "URL"])

    # ensemble predictions equal EnsembleDetector.detect on the same members
    doc = make_doc("gretel_en_test", 0)
    first = json.loads((out / "gretel_en_test" / "regex+spacy" / "predictions.jsonl")
                       .read_text(encoding="utf-8").splitlines()[0])
    expected = EnsembleDetector([RegexDetector(), FakeSpacy()], "regex+spacy").detect(doc.text)
    assert [tuple(s[:3]) for s in first["spans"]] == [(s.start, s.end, s.label.value)
                                                      for s in expected]

    for det in ALL_DETECTORS:
        assert "CC BY 4.0" in (out / f"errors_{det}.md").read_text(encoding="utf-8")
    log = json.loads((out / "run_log.json").read_text())
    assert log["total_seconds"] >= 0 and {s["step"] for s in log["steps"]} >= {
        "detect", "score", "sweep", "ensemble", "errors"}


def test_rerun_is_deterministic(roots, monkeypatch):
    monkeypatch.setattr(cli, "make_detector", fake_factory)
    _, out = roots
    run(roots, "run", "--set", "gretel_en_test", "--detector", "regex")
    first = json.loads((out / "gretel_en_test" / "regex" / "metrics.json").read_text())
    run(roots, "run", "--set", "gretel_en_test", "--detector", "regex")
    second = json.loads((out / "gretel_en_test" / "regex" / "metrics.json").read_text())
    assert first["views"] == second["views"]  # scores and CIs identical; only timings vary


def test_tables_are_generated_from_results(roots, monkeypatch, tmp_path):
    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    monkeypatch.setattr(cli, "make_detector", fake_factory)
    data, out = roots
    run(roots, "all")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["models"] = {"en_core_web_lg": {
        "name": "en_core_web_lg", "version": "3.8.0", "license": "MIT", "wheel": "w.whl",
        "url": "https://example.invalid/w.whl", "size": 1, "sha256": "0" * 64}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    repo = Path(__file__).resolve().parents[1]
    docs = {}
    for name in ("results-detection.md", "DETECTORS.md"):
        docs[name] = tmp_path / name
        shutil.copy(repo / "docs" / name, docs[name])
    spec = importlib.util.spec_from_file_location(
        "make_detection_tables", repo / "scripts" / "make_detection_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_detection_tables"] = mod
    spec.loader.exec_module(mod)
    mod.main(["--results", str(out), "--data-root", str(data),
              "--results-doc", str(docs["results-detection.md"]),
              "--detectors-doc", str(docs["DETECTORS.md"])])
    text = docs["results-detection.md"].read_text(encoding="utf-8")
    assert "**selected threshold: 0.50**" in text
    assert "| `gretel_en_test` | `regex+spacy` |" in text
    assert "NOT MEASURED" in text           # no raw files in this fixture -> no raw labels
    assert "spacy==3.8.16" in docs["DETECTORS.md"].read_text(encoding="utf-8")
    # regenerating is idempotent
    before = text
    mod.main(["--results", str(out), "--data-root", str(data),
              "--results-doc", str(docs["results-detection.md"]),
              "--detectors-doc", str(docs["DETECTORS.md"])])
    assert docs["results-detection.md"].read_text(encoding="utf-8") == before
