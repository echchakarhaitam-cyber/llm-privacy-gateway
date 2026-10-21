"""Evaluation sets, loaded only after their SHA-256 matches data/manifest.json.

A benchmark number is only meaningful for a known set of documents. A set whose file was
rebuilt differently, edited or truncated is refused (fail closed) instead of being scored.
"""

from __future__ import annotations

import json
from pathlib import Path

from privgate.data.documents import Document
from privgate.data.splits import file_sha256, read_jsonl

SET_NAMES = ("nemotron_dev_2k", "nemotron_test_5k", "gretel_en_test")
DEV_SET = "nemotron_dev_2k"


class SetIntegrityError(RuntimeError):
    """The evaluation-set file is missing, unknown or differs from the manifest."""


def set_path(data_root: Path, name: str) -> Path:
    return Path(data_root) / "processed" / f"{name}.jsonl"


def expected_sha256(data_root: Path, name: str) -> str:
    manifest_path = Path(data_root) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest["evaluation_sets"][name]["sha256"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SetIntegrityError(f"no manifest entry for evaluation set {name!r}") from exc


def load_evaluation_set(name: str, data_root: Path = Path("data")) -> tuple[list[Document], str]:
    """Documents of ``name`` and their verified SHA-256; raises SetIntegrityError."""
    expected = expected_sha256(data_root, name)
    path = set_path(data_root, name)
    if not path.exists():
        raise SetIntegrityError(f"{path} is missing; run scripts/build_eval_sets.py")
    actual = file_sha256(path)
    if actual != expected:
        raise SetIntegrityError(
            f"{path}: SHA-256 {actual} differs from manifest {expected}; refusing to score it")
    return read_jsonl(path), actual
