"""Download the pinned spaCy model wheel, record it in data/manifest.json, pip-install it.

Usage:  python scripts/install_model.py [--root DIR] [--no-install]

The model comes from its official GitHub release (explosion/spacy-models). The wheel is kept
in models/ (gitignored). Its SHA-256 is recorded under "models" in the manifest on first
download; later runs verify the local wheel against that hash before installing, so every
benchmark run uses the same model bytes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from download_data import fetch, load_manifest, sha256_of, write_manifest

URL_PATTERN = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "{name}-{version}/{name}-{version}-py3-none-any.whl"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    license: str

    @property
    def wheel(self) -> str:
        return f"{self.name}-{self.version}-py3-none-any.whl"

    @property
    def url(self) -> str:
        return URL_PATTERN.format(name=self.name, version=self.version)


MODEL = ModelSpec("en_core_web_lg", "3.8.0", "MIT")


def ensure_wheel(spec: ModelSpec, models_dir: Path, previous: dict | None,
                 url: str | None = None) -> dict:
    """Download the wheel if absent and return its manifest entry.

    A wheel already on disk must match the recorded hash; a mismatch is an error, never a
    silent re-download, because it means the pinned model changed. ``url`` overrides the
    download URL (tests); the manifest always records the canonical URL.
    """
    dest = models_dir / spec.wheel
    known = (previous or {}).get("sha256")
    if not dest.exists():
        fetch(url or spec.url, dest, None)
    digest = sha256_of(dest)
    if known is not None and digest != known:
        raise OSError(f"{dest}: sha256 {digest} differs from manifest {known}")
    return {
        "name": spec.name,
        "version": spec.version,
        "license": spec.license,
        "url": spec.url,
        "wheel": spec.wheel,
        "size": dest.stat().st_size,
        "sha256": digest,
        "downloaded_at": (previous or {}).get("downloaded_at")
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def record_model(root: Path, models_dir: Path, spec: ModelSpec = MODEL,
                 url: str | None = None) -> dict:
    """Ensure the wheel and update ``models`` in the manifest, keeping other keys intact."""
    manifest_path = root / "manifest.json"
    manifest = load_manifest(manifest_path)
    models = dict(manifest.get("models", {}))
    models[spec.name] = ensure_wheel(spec, models_dir, models.get(spec.name), url)
    write_manifest(manifest_path, {**manifest, "models": models})
    return models[spec.name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data"), help="data directory")
    parser.add_argument("--models", type=Path, default=Path("models"), help="wheel directory")
    parser.add_argument("--no-install", action="store_true", help="only download and record")
    args = parser.parse_args(argv)
    entry = record_model(args.root, args.models)
    print(f"{entry['wheel']}  {entry['size']} bytes  sha256={entry['sha256']}")
    if not args.no_install:
        wheel = args.models / entry["wheel"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)],
                       check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
