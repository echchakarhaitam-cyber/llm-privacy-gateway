"""Saved models: a directory with the model file and a JSON sidecar describing it.

The sidecar records what is needed to trust and reproduce a model: feature version,
SHA-256 of the training set, hyper-parameters, training time, number of documents, seed,
library versions and the SHA-256 of the model file. Loading refuses a model whose file
hash or feature version does not match, so a model is never silently paired with
different features or a corrupted file (the logistic-regression file is a pickle: checking
its hash before unpickling also guards against a swapped file).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Iterable
from pathlib import Path

from privgate.data.documents import Document
from privgate.data.splits import document_to_record, file_sha256
from privgate.ml.features import FEATURE_VERSION

SIDECAR = "model.json"


class ModelFileError(RuntimeError):
    """A saved model is missing, corrupted or built with other features."""


def training_set_sha256(docs: Iterable[Document]) -> str:
    """Hash of the training documents (canonical records, sorted by doc_id)."""
    digest = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        digest.update(json.dumps(document_to_record(doc), sort_keys=True,
                                 ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _versions(packages: Iterable[str]) -> dict[str, str | None]:
    out = {}
    for p in packages:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_sidecar(model_dir: Path, model_file: str, meta: dict, packages: Iterable[str]) -> dict:
    payload = {
        **meta,
        "feature_version": FEATURE_VERSION,
        "model_file": model_file,
        "model_sha256": file_sha256(model_dir / model_file),
        "library_versions": _versions(packages),
    }
    (model_dir / SIDECAR).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return payload


def read_sidecar(model_dir: Path, kind: str) -> dict:
    path = Path(model_dir) / SIDECAR
    if not path.exists():
        raise ModelFileError(f"{path} is missing")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("kind") != kind:
        raise ModelFileError(f"{model_dir} holds a {meta.get('kind')!r} model, not {kind!r}")
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ModelFileError(f"{model_dir}: feature version {meta.get('feature_version')!r} "
                             f"!= {FEATURE_VERSION!r}")
    model_path = Path(model_dir) / meta["model_file"]
    if not model_path.exists() or file_sha256(model_path) != meta["model_sha256"]:
        raise ModelFileError(f"{model_path} is missing or differs from its recorded hash")
    return meta
