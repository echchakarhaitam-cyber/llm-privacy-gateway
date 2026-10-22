"""scripts/make_masking_tables.py on a small synthetic results directory."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from privgate.bench.privacy import privacy_report
from privgate.bench.sets import DEV_SET, SET_NAMES
from privgate.bench.utility import utility_report
from privgate.data.documents import Document
from privgate.masking import STRATEGIES, mask
from privgate.spans import EntityType, Span

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("oracle", "regex+presidio", "crf", "crf+regex+presidio")
TEXT = "Bob Stone met Ann at Acme."
GOLD = (Span(0, 9, EntityType.PERSON), Span(14, 17, EntityType.PERSON),
        Span(21, 25, EntityType.ORGANIZATION))


def _module():
    spec = importlib.util.spec_from_file_location("make_masking_tables",
                                                  ROOT / "scripts" / "make_masking_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_masking_tables"] = mod
    spec.loader.exec_module(mod)
    return mod


def _acc(a):
    return {"accuracy": a, "macro_f1": a, "accuracy_ci95": [a - 0.1, a]}


def _diff(d):
    return {"a": 0.9, "b": 0.9 - d, "difference": d, "ci95": [d - 0.01, d + 0.01],
            "ci_excludes_zero": True}


def _results(tmp_path):
    docs = [Document("d", TEXT, GOLD, "synthetic")]
    rows = []
    for set_name in SET_NAMES:
        for det in DETECTORS:
            spans = list(GOLD) if det == "oracle" else list(GOLD[:2])
            for strategy in STRATEGIES:
                masked = [mask(TEXT, spans, strategy)]
                row = {"set": set_name, "detector": det, "strategy": strategy,
                       "privacy": privacy_report(docs, masked, n_resamples=10),
                       "utility": utility_report(docs, masked)}
                if set_name == DEV_SET:
                    row["domain_original_classifier"] = {**_acc(0.8), "drop_vs_unmasked":
                                                         _diff(0.1)}
                    if det != "oracle":
                        row["domain_original_classifier"]["gap_vs_oracle"] = _diff(-0.02)
                    if strategy == "typed":
                        row["domain_typed_classifier"] = {
                            **_acc(0.85), "drop_vs_unmasked_original_classifier": _diff(0.05)}
                rows.append(row)
    out = tmp_path / "m5"
    out.mkdir()
    (out / "masking.json").write_text(json.dumps(
        {"rows": rows, "dev_unmasked_original_classifier": _acc(0.9)}))
    cv = {"rows": [{"C": 1.0, "mean_accuracy": 0.9, "fold_accuracy": [0.9] * 5}],
          "selected_C": 1.0}
    (out / "domain_classifier.json").write_text(json.dumps({
        "n": 20, "groups": 10, "seed": 0, "seconds": 60,
        "settings": {"original": {"cv": cv, "dev_original": _acc(0.9)},
                     "typed": {"cv": cv, "dev_oracle_typed": _acc(0.85)}}}))
    return out


def test_tables_are_generated_and_idempotent(tmp_path):
    mod = _module()
    out = _results(tmp_path)
    doc = tmp_path / "results-masking.md"
    shutil.copy(ROOT / "docs" / "results-masking.md", doc)
    mod.main(["--results", str(out), "--doc", str(doc)])
    text = doc.read_text(encoding="utf-8")
    assert "Fall back to typed**: ID_NUMBER, OTHER" in text
    assert "| `crf+regex+presidio` | partial |" in text
    assert "**selected**" in text and "unmasked dev: accuracy 0.900" in text
    mod.main(["--results", str(out), "--doc", str(doc)])
    assert doc.read_text(encoding="utf-8") == text
