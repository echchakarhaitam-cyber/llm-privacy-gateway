"""Loader for gretelai/synthetic_pii_finance_multilingual (Apache-2.0), English test file.

Schema of the pinned file (inspected, see docs/DATA.md): ``generated_text`` (string),
``pii_spans`` (a JSON STRING: list of ``{"start", "end", "label"}``), ``pii_types``
(list<string>, the same labels in order), ``pii_type_labels`` (list<int64>), descriptive
string columns, five int64 judge scores and the pandas index ``__index_level_0__``, which
is unique and used as the document id.

The dataset does not store the covered text of a span, so offset verification can only
check that spans lie inside the text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

from privgate.data._parquet import iter_rows
from privgate.data.documents import (
    Document,
    LoadOptions,
    LoadReport,
    LoadTally,
    RawSpan,
    build_document,
)

DATASET = "gretel-finance-en"
META_COLUMNS = ("domain", "document_type", "expanded_type", "language")
COLUMNS = ("__index_level_0__", "generated_text", "pii_spans", *META_COLUMNS)


def parse_spans(field: str) -> list[RawSpan]:
    """Parse the ``pii_spans`` JSON string; anything but a list of span dicts raises."""
    try:
        items = json.loads(field)
    except json.JSONDecodeError as exc:
        raise ValueError(f"unparseable pii_spans field: {field[:80]!r}") from exc
    if not isinstance(items, list):
        raise TypeError(f"pii_spans field is not a list: {field[:80]!r}")
    spans = []
    for item in items:
        if not isinstance(item, dict) or not {"start", "end", "label"} <= item.keys():
            raise TypeError(f"malformed span entry: {item!r}")
        spans.append(RawSpan(item["start"], item["end"], str(item["label"])))
    return spans


def doc_id(row: Mapping) -> str:
    """privgate id of a raw row, from the unique pandas index column."""
    return f"gretel:{row['__index_level_0__']}"


def iter_gretel(
    path: str | Path, tally: LoadTally, limit: int | None = None
) -> Iterator[Document]:
    """Stream Documents from the Gretel parquet file, updating ``tally``."""
    for n, row in enumerate(iter_rows(path, COLUMNS)):
        if limit is not None and n >= limit:
            return
        index = row["__index_level_0__"]
        try:
            raw = parse_spans(row["pii_spans"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: index {index}: {exc}") from exc
        meta = {c: row[c] for c in META_COLUMNS if row[c] is not None}
        yield build_document(doc_id(row), row["generated_text"], raw, tally, meta)


def load_gretel(
    path: str | Path,
    merge_adjacent: bool = True,
    limit: int | None = None,
    *,
    strict_labels: bool = False,
    offset_check: str = "casefold",
    overlap_policy: str = "resolve",
) -> tuple[list[Document], LoadReport]:
    """Load the Gretel English parquet file into Documents plus a LoadReport."""
    options = LoadOptions(merge_adjacent, strict_labels, offset_check, overlap_policy)
    tally = LoadTally(DATASET, options)
    docs = list(iter_gretel(path, tally, limit))
    return docs, tally.freeze()
