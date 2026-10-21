"""Offline tests for scripts/install_model.py (file:// URL instead of GitHub, no pip)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # install_model imports its sibling download_data
_spec = importlib.util.spec_from_file_location("install_model", _SCRIPTS / "install_model.py")
im = importlib.util.module_from_spec(_spec)
sys.modules["install_model"] = im
_spec.loader.exec_module(im)

SPEC = im.ModelSpec("toy_model", "1.2.3", "MIT")


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "upstream.whl"
    src.write_bytes(b"not really a wheel\n" * 50)
    return src


def test_canonical_url_and_wheel_name():
    assert im.MODEL.url == (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
    )
    assert SPEC.wheel == "toy_model-1.2.3-py3-none-any.whl"


def test_record_model_writes_manifest_and_keeps_other_keys(tmp_path, source):
    root, models = tmp_path / "data", tmp_path / "models"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"files": [1]}))
    entry = im.record_model(root, models, SPEC, url=source.as_uri())
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["files"] == [1]
    assert manifest["models"]["toy_model"] == entry
    assert entry["sha256"] == im.sha256_of(source)
    assert entry["url"].startswith("https://github.com/explosion/")
    assert (models / SPEC.wheel).read_bytes() == source.read_bytes()


def test_second_run_reuses_wheel_and_date(tmp_path, source):
    root, models = tmp_path / "data", tmp_path / "models"
    first = im.record_model(root, models, SPEC, url=source.as_uri())
    second = im.record_model(root, models, SPEC, url=(tmp_path / "missing").as_uri())
    assert second == first


def test_tampered_wheel_is_rejected(tmp_path, source):
    root, models = tmp_path / "data", tmp_path / "models"
    im.record_model(root, models, SPEC, url=source.as_uri())
    (models / SPEC.wheel).write_bytes(b"tampered")
    with pytest.raises(OSError, match="differs from manifest"):
        im.record_model(root, models, SPEC, url=source.as_uri())
