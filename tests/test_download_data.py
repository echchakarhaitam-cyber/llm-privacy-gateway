"""Offline tests for scripts/download_data.py: file:// URLs stand in for the network."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
_spec = importlib.util.spec_from_file_location("download_data", _SCRIPT)
dd = importlib.util.module_from_spec(_spec)
sys.modules["download_data"] = dd  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(dd)

PAYLOAD = b"synthetic parquet stand-in\n" * 100


def make_spec():
    return dd.FileSpec("org/name", "abc123", "data/file.parquet", len(PAYLOAD), "CC-BY-4.0")


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "upstream" / "file.parquet"
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    return src


def test_local_path_and_url():
    spec = make_spec()
    assert spec.local_path(Path("data")) == Path("data/raw/org__name/file.parquet")
    assert spec.url == (
        "https://huggingface.co/datasets/org/name/resolve/abc123/data/file.parquet"
    )


def test_pinned_files_match_milestone_spec():
    sizes = {(s.repo, s.path): s.size for s in dd.FILES}
    assert sizes[("nvidia/Nemotron-PII", "data/train-00000-of-00001.parquet")] == 156_334_329
    assert sizes[("nvidia/Nemotron-PII", "data/test-00000-of-00001.parquet")] == 150_968_398
    gretel = ("gretelai/synthetic_pii_finance_multilingual",
              "data/English_test-00000-of-00001.parquet")
    assert sizes[gretel] == 2_720_729


def test_fetch_streams_and_renames(tmp_path, source):
    dest = tmp_path / "out" / "file.parquet"
    dd.fetch(source.as_uri(), dest, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name("file.parquet.part").exists()


def test_fetch_wrong_size_leaves_nothing(tmp_path, source):
    dest = tmp_path / "out" / "file.parquet"
    with pytest.raises(OSError, match="expected"):
        dd.fetch(source.as_uri(), dest, len(PAYLOAD) + 1)
    assert not dest.exists()
    assert not dest.with_name("file.parquet.part").exists()


def test_ensure_file_downloads_and_records_entry(tmp_path, source):
    entry = dd.ensure_file(make_spec(), tmp_path, None, url=source.as_uri())
    assert entry["sha256"] == dd.sha256_of(source)
    assert entry["size"] == len(PAYLOAD)
    assert entry["url"].startswith("https://huggingface.co/")  # canonical, not the override
    assert set(entry) == {
        "repo", "revision", "path", "size", "sha256", "license", "url", "downloaded_at",
    }


def test_ensure_file_skips_intact_file(tmp_path, source):
    spec = make_spec()
    first = dd.ensure_file(spec, tmp_path, None, url=source.as_uri())
    # A URL that cannot be fetched proves no download is attempted.
    missing = (tmp_path / "does-not-exist").as_uri()
    second = dd.ensure_file(spec, tmp_path, first, url=missing)
    assert second == first


def test_ensure_file_redownloads_corrupt_file(tmp_path, source):
    spec = make_spec()
    first = dd.ensure_file(spec, tmp_path, None, url=source.as_uri())
    local = spec.local_path(tmp_path)
    local.write_bytes(b"X" * len(PAYLOAD))  # same size, wrong content
    second = dd.ensure_file(spec, tmp_path, first, url=source.as_uri())
    assert local.read_bytes() == PAYLOAD
    assert second["sha256"] == first["sha256"]


def test_ensure_file_rejects_upstream_hash_change(tmp_path, source):
    previous = {"sha256": "0" * 64, "downloaded_at": "2026-01-01T00:00:00+00:00"}
    with pytest.raises(OSError, match="differs from manifest"):
        dd.ensure_file(make_spec(), tmp_path, previous, url=source.as_uri())


def test_download_all_preserves_other_manifest_keys(tmp_path, source, monkeypatch):
    spec = make_spec()
    (tmp_path / "manifest.json").write_text(json.dumps({"evaluation_sets": {"x": 1}}))
    real = dd.ensure_file
    monkeypatch.setattr(
        dd, "ensure_file", lambda s, root, prev: real(s, root, prev, url=source.as_uri())
    )
    manifest = dd.download_all(tmp_path, (spec,))
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["evaluation_sets"] == {"x": 1}
    assert [e["path"] for e in manifest["files"]] == ["data/file.parquet"]


def test_gretel_train_file_is_pinned():
    key = ("gretelai/synthetic_pii_finance_multilingual",
           "data/English_train-00000-of-00001.parquet")
    spec = next(s for s in dd.FILES if (s.repo, s.path) == key)
    assert spec.size == 24_403_359 and spec.license == "Apache-2.0"
    assert spec.revision == "7b844d16738527a04264f50214cb426a4cea0897"


class _FlakyResponse:
    """A response that delivers ``data`` and then stalls (raises) unless it is the last one."""

    def __init__(self, data, status, fail):
        self._data, self.status, self._fail, self._pos = data, status, fail, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        if self._pos >= len(self._data):
            if self._fail:
                raise TimeoutError("stalled")
            return b""
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


def test_fetch_resumes_a_stalled_download_with_range(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        start = int(request.get_header("Range", "bytes=0-").split("=")[1].rstrip("-"))
        calls.append(start)
        if start == 0:
            return _FlakyResponse(PAYLOAD[:700], 200, fail=True)       # stalls after 700 bytes
        return _FlakyResponse(PAYLOAD[start:], 206, fail=False)

    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "f.parquet"
    dd.fetch("https://example.invalid/f", dest, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD and calls == [0, 700]


def test_fetch_restarts_when_the_server_ignores_range(tmp_path, monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout=None):
        attempts.append(request.get_header("Range"))
        if len(attempts) == 1:
            return _FlakyResponse(PAYLOAD[:500], 200, fail=True)
        return _FlakyResponse(PAYLOAD, 200, fail=False)               # 200 despite Range

    monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "f.parquet"
    dd.fetch("https://example.invalid/f", dest, len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD                                # not 500 + full payload


def test_fetch_gives_up_after_the_attempt_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(dd.urllib.request, "urlopen",
                        lambda request, timeout=None: _FlakyResponse(b"", 200, fail=True))
    with pytest.raises(TimeoutError):
        dd.fetch("https://example.invalid/f", tmp_path / "f.parquet", len(PAYLOAD))
    assert not (tmp_path / "f.parquet").exists()
