"""Download the pinned public datasets into data/raw/ and record them in data/manifest.json.

Why pinned revisions and hashes: benchmark numbers are only reproducible if everyone scores
the same bytes. The revision fixes the upstream commit; the size and SHA-256 recorded in
the manifest let anyone check that their local copy is identical.

Usage:  python scripts/download_data.py [--root DIR]

Only the standard library is used, so the script runs before any extra is installed.
Files are streamed to ``<name>.part`` and renamed only after the size check passes, so an
interrupted download never leaves a file that looks complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

URL_PATTERN = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
CHUNK = 1 << 20


@dataclass(frozen=True)
class FileSpec:
    """One pinned upstream file: where it comes from and what it must look like."""

    repo: str
    revision: str
    path: str
    size: int
    license: str

    @property
    def url(self) -> str:
        return URL_PATTERN.format(repo=self.repo, revision=self.revision, path=self.path)

    def local_path(self, root: Path) -> Path:
        """data/raw/<repo with '/' -> '__'>/<filename>."""
        return root / "raw" / self.repo.replace("/", "__") / Path(self.path).name


NEMOTRON_REV = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
GRETEL_REV = "7b844d16738527a04264f50214cb426a4cea0897"

FILES: tuple[FileSpec, ...] = (
    FileSpec(
        "nvidia/Nemotron-PII", NEMOTRON_REV,
        "data/train-00000-of-00001.parquet", 156_334_329, "CC-BY-4.0",
    ),
    FileSpec(
        "nvidia/Nemotron-PII", NEMOTRON_REV,
        "data/test-00000-of-00001.parquet", 150_968_398, "CC-BY-4.0",
    ),
    FileSpec(
        "gretelai/synthetic_pii_finance_multilingual", GRETEL_REV,
        "data/English_test-00000-of-00001.parquet", 2_720_729, "Apache-2.0",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected_size: int) -> None:
    """Stream ``url`` to ``dest`` via a .part file; raise if the byte count is wrong."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    written = 0
    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:
        for block in iter(lambda: resp.read(CHUNK), b""):
            out.write(block)
            written += len(block)
    if written != expected_size:
        part.unlink()
        raise OSError(f"{url}: expected {expected_size} bytes, got {written}")
    os.replace(part, dest)


def ensure_file(spec: FileSpec, root: Path, previous: dict | None, url: str | None = None) -> dict:
    """Make sure the file is present and intact; return its manifest entry.

    A file already on disk is kept only when its size AND hash match the previous manifest
    entry (or, with no previous entry, when its size matches). Anything else is downloaded
    again. ``url`` overrides the download URL (used by tests to avoid the network); the
    manifest always records the canonical URL.
    """
    dest = spec.local_path(root)
    known_hash = (previous or {}).get("sha256")
    if dest.exists() and dest.stat().st_size == spec.size:
        digest = sha256_of(dest)
        if known_hash in (None, digest):
            downloaded_at = (previous or {}).get("downloaded_at") or _now()
            return _entry(spec, digest, downloaded_at)
    fetch(url or spec.url, dest, spec.size)
    digest = sha256_of(dest)
    if known_hash is not None and digest != known_hash:
        raise OSError(f"{spec.path}: sha256 {digest} differs from manifest {known_hash}")
    return _entry(spec, digest, _now())


def _entry(spec: FileSpec, digest: str, downloaded_at: str) -> dict:
    return {
        "repo": spec.repo,
        "revision": spec.revision,
        "path": spec.path,
        "size": spec.size,
        "sha256": digest,
        "license": spec.license,
        "url": spec.url,
        "downloaded_at": downloaded_at,
    }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_all(root: Path, specs: tuple[FileSpec, ...] = FILES) -> dict:
    """Ensure every file; update ``files`` in the manifest, keeping other keys intact."""
    manifest_path = root / "manifest.json"
    manifest = load_manifest(manifest_path)
    previous = {(e["repo"], e["path"]): e for e in manifest.get("files", [])}
    entries = [ensure_file(s, root, previous.get((s.repo, s.path))) for s in specs]
    updated = {**manifest, "files": entries}
    write_manifest(manifest_path, updated)
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data"), help="data directory")
    args = parser.parse_args(argv)
    manifest = download_all(args.root)
    for e in manifest["files"]:
        print(f"{e['repo']}/{e['path']}  {e['size']} bytes  sha256={e['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
