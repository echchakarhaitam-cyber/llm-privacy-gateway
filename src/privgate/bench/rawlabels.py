"""The raw dataset label behind each gold span of an evaluation set (needed for RQ3).

Evaluation sets store mapped types only (docs/DATA.md). To ask which *raw* labels leak,
each gold span is linked back to the raw spans of its row in the pinned parquet file: its
primary raw label is the raw label, among raw spans with the same mapped type, that shares
the most characters with it (ties: earlier start, then label name). A merged span such as
first_name + last_name gets the label of its longer part.

The index is cached next to the set (data/processed/<set>.raw_labels.json, gitignored) and
tied to the set's SHA-256, so a rebuilt set never reuses a stale index. Building it needs
the raw files; without them the benchmark skips the per-raw-label view.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from privgate.data import gretel, nemotron
from privgate.data._parquet import iter_rows
from privgate.data.documents import Document, RawSpan
from privgate.data.mapping import map_label
from privgate.data.splits import EVALUATION_SETS, RAW_FILES
from privgate.spans import EntityType

UNKNOWN = "(unknown)"

# source file -> (dataset, columns, doc_id, parse_spans, span column)
_SOURCES = {
    "nemotron_train_file": (nemotron.DATASET, nemotron.COLUMNS, nemotron.doc_id,
                            nemotron.parse_spans, "spans"),
    "nemotron_test_file": (nemotron.DATASET, nemotron.COLUMNS, nemotron.doc_id,
                           nemotron.parse_spans, "spans"),
    "gretel_en_test_file": (gretel.DATASET, gretel.COLUMNS, gretel.doc_id,
                            gretel.parse_spans, "pii_spans"),
}


def primary_raw_labels(doc: Document, raw_spans: Sequence[RawSpan], dataset: str) -> list[str]:
    """One raw label per gold span of ``doc``, in span order."""
    labels = []
    for gold in doc.spans:
        best: tuple[tuple[int, int, str], str] | None = None
        for raw in raw_spans:
            if not (isinstance(raw.start, int) and isinstance(raw.end, int)):
                continue
            if (map_label(dataset, raw.label) or EntityType.OTHER) is not gold.label:
                continue
            overlap = min(gold.end, raw.end) - max(gold.start, raw.start)
            if overlap <= 0:
                continue
            key = (-overlap, raw.start, raw.label)
            if best is None or key < best[0]:
                best = (key, raw.label)
        labels.append(best[1] if best else UNKNOWN)
    return labels


def build_raw_label_index(
    data_root: Path, set_name: str, docs: Sequence[Document]
) -> dict[str, list[str]]:
    spec = {s.name: s for s in EVALUATION_SETS}[set_name]
    dataset, columns, doc_id, parse, span_column = _SOURCES[spec.source_file]
    wanted = {d.doc_id: d for d in docs}
    index: dict[str, list[str]] = {}
    for row in iter_rows(Path(data_root) / RAW_FILES[spec.source_file], columns):
        key = doc_id(row)
        if key in wanted:
            index[key] = primary_raw_labels(wanted[key], parse(row[span_column]), dataset)
    missing = set(wanted) - set(index)
    if missing:
        raise ValueError(f"{len(missing)} documents of {set_name} not found in the raw file")
    return index


def load_raw_labels(
    data_root: Path, set_name: str, docs: Sequence[Document], set_sha256: str
) -> dict[str, list[str]] | None:
    """Cached index for a verified set; built when absent or stale; None without raw data."""
    cache = Path(data_root) / "processed" / f"{set_name}.raw_labels.json"
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("set_sha256") == set_sha256:
            return stored["labels"]
    spec = {s.name: s for s in EVALUATION_SETS}[set_name]
    if not (Path(data_root) / RAW_FILES[spec.source_file]).exists():
        return None
    index = build_raw_label_index(data_root, set_name, docs)
    cache.write_text(json.dumps({"set_sha256": set_sha256, "labels": index}, sort_keys=True),
                     encoding="utf-8")
    return index
