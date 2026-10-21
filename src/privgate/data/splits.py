"""Deterministic splits, reproducible samples and the named evaluation sets.

Split rule: a document goes to ``dev`` iff ``int(sha256(key), 16) % 100 < 10``, else to
``train`` (Nemotron train file only; the Nemotron test file is ``test``). A hash, not a
random draw, so the split of a document never depends on file order, on the other
documents or on a seed.

The key is ``meta["group_id"]`` when present, else ``doc_id``. Nemotron stores every
persona twice (a ``us`` and an ``intl`` rendering sharing a uid, dates of birth and
document type); hashing the uid keeps both renderings in the same split, so dev cannot
contain near-copies of train documents.

Evaluation sets are fixed by name (see :data:`EVALUATION_SETS`) and written as JSONL with
a stable byte layout, so their SHA-256 in data/manifest.json identifies them exactly.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from privgate.data.documents import Document, LoadOptions, LoadReport, LoadTally
from privgate.data.gretel import DATASET as GRETEL
from privgate.data.gretel import iter_gretel
from privgate.data.nemotron import DATASET as NEMOTRON
from privgate.data.nemotron import iter_nemotron
from privgate.spans import EntityType, Span

DEV_PERCENT = 10
DEFAULT_OPTIONS = LoadOptions()

# Raw files, relative to the data root (data/).
RAW_FILES = {
    "nemotron_train_file": Path("raw/nvidia__Nemotron-PII/train-00000-of-00001.parquet"),
    "nemotron_test_file": Path("raw/nvidia__Nemotron-PII/test-00000-of-00001.parquet"),
    "gretel_en_test_file": Path(
        "raw/gretelai__synthetic_pii_finance_multilingual/English_test-00000-of-00001.parquet"
    ),
}
_READERS = {
    "nemotron_train_file": (NEMOTRON, iter_nemotron),
    "nemotron_test_file": (NEMOTRON, iter_nemotron),
    "gretel_en_test_file": (GRETEL, iter_gretel),
}


def assign_split(key: str) -> str:
    """``"dev"`` for about 10% of keys, ``"train"`` for the rest; pure function of ``key``."""
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    return "dev" if bucket < DEV_PERCENT else "train"


def split_key(doc: Document) -> str:
    return doc.meta.get("group_id") or doc.doc_id


def sample(docs: Iterable[Document], n: int, seed: int) -> list[Document]:
    """A reproducible subset of ``n`` documents, independent of the input order.

    Documents are sorted by ``doc_id`` first, then drawn with ``random.Random(seed)``;
    duplicate ids are rejected because they would make the sort order input-dependent.
    """
    ordered = sorted(docs, key=lambda d: d.doc_id)
    ids = [d.doc_id for d in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("sample() needs unique doc_ids")
    if not 0 <= n <= len(ordered):
        raise ValueError(f"cannot sample {n} documents from {len(ordered)}")
    return random.Random(seed).sample(ordered, n)


@dataclass(frozen=True)
class EvaluationSetSpec:
    """How a named evaluation set is drawn. ``size=None`` means every document."""

    name: str
    source_file: str
    split: str
    size: int | None
    seed: int | None
    description: str


EVALUATION_SETS: tuple[EvaluationSetSpec, ...] = (
    EvaluationSetSpec(
        "nemotron_dev_2k", "nemotron_train_file", "dev", 2000, 0,
        "sample(dev, 2000, seed=0): in-distribution development set",
    ),
    EvaluationSetSpec(
        "nemotron_test_5k", "nemotron_test_file", "test", 5000, 0,
        "sample(test, 5000, seed=0): held-out test set, same generator, mostly unseen domains",
    ),
    EvaluationSetSpec(
        "gretel_en_test", "gretel_en_test_file", "test", None, None,
        "every row of the Gretel English test file: out-of-distribution set",
    ),
)


def iter_file(
    data_root: Path, file_key: str, options: LoadOptions = DEFAULT_OPTIONS
) -> tuple[Iterator[tuple[str, Document]], LoadTally]:
    """Stream ``(split, document)`` pairs from one raw file, with the tally being filled."""
    dataset, reader = _READERS[file_key]
    tally = LoadTally(dataset, options)
    path = Path(data_root) / RAW_FILES[file_key]

    def pairs() -> Iterator[tuple[str, Document]]:
        for doc in reader(path, tally):
            if file_key == "nemotron_train_file":
                yield assign_split(split_key(doc)), doc
            else:
                yield "test", doc

    return pairs(), tally


def build_evaluation_sets(
    data_root: Path, options: LoadOptions = DEFAULT_OPTIONS
) -> tuple[dict[str, list[Document]], dict[str, LoadReport]]:
    """Draw every set in :data:`EVALUATION_SETS` from the raw files.

    Each raw file is streamed once and only the documents of the requested split are kept.
    Returns the sets and the LoadReport of each raw file read.
    """
    sets: dict[str, list[Document]] = {}
    reports: dict[str, LoadReport] = {}
    for spec in EVALUATION_SETS:
        pairs, tally = iter_file(data_root, spec.source_file, options)
        pool = [doc for split, doc in pairs if split == spec.split]
        sets[spec.name] = pool if spec.size is None else sample(pool, spec.size, spec.seed)
        reports[spec.source_file] = tally.freeze()
    return sets, reports


def document_to_record(doc: Document) -> dict:
    return {
        "doc_id": doc.doc_id,
        "text": doc.text,
        "spans": [{"start": s.start, "end": s.end, "label": s.label.value} for s in doc.spans],
        "source": doc.source,
        "meta": dict(doc.meta),
    }


def document_from_record(record: dict) -> Document:
    spans = tuple(
        Span(s["start"], s["end"], EntityType(s["label"]), source=record["source"])
        for s in record["spans"]
    )
    return Document(record["doc_id"], record["text"], spans, record["source"], record["meta"])


def write_jsonl(docs: Sequence[Document], path: Path) -> str:
    """Write one JSON object per line (UTF-8, LF, sorted keys); return the file's SHA-256.

    The fixed layout makes the hash a fingerprint of the content on every platform.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for doc in docs:
            line = json.dumps(document_to_record(doc), ensure_ascii=False, sort_keys=True)
            out.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as fh:
        return [document_from_record(json.loads(line)) for line in fh if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
