"""Loader for nvidia/Nemotron-PII (CC BY 4.0), revision b70ffaf5.

Schema of the pinned parquet files (inspected, see docs/DATA.md): string columns ``uid,
domain, document_type, document_description, document_format, locale, text, spans,
text_tagged``. ``spans`` is a STRING holding a list of ``{"start", "end", "text", "label"}``
dicts, written as a Python literal (single quotes) in almost every row and as JSON in a
handful; the stored ``text`` is occasionally an int (ages, CVVs).

Each ``uid`` occurs twice per file, once with ``locale="us"`` and once with ``"intl"``:
two renderings of the same generated persona and document type. ``doc_id`` therefore
includes the locale, and ``meta["group_id"]`` keeps the uid so that both renderings land
in the same split (see :mod:`privgate.data.splits`).
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
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

DATASET = "nemotron-pii"
META_COLUMNS = ("domain", "document_type", "document_format", "locale")
COLUMNS = ("uid", "text", "spans", *META_COLUMNS)


def parse_spans(field: str) -> list[RawSpan]:
    """Parse the ``spans`` string (JSON or Python literal) into raw spans.

    ``ast.literal_eval`` only evaluates literals, so a crafted string cannot run code. A
    field that is not a list of span dicts raises: a document whose annotations cannot be
    read must not enter an evaluation set looking PII-free.
    """
    try:
        items = json.loads(field)
    except json.JSONDecodeError:
        try:
            items = ast.literal_eval(field)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"unparseable spans field: {field[:80]!r}") from exc
    if not isinstance(items, list):
        raise TypeError(f"spans field is not a list: {field[:80]!r}")
    spans = []
    for item in items:
        if not isinstance(item, dict) or not {"start", "end", "label"} <= item.keys():
            raise TypeError(f"malformed span entry: {item!r}")
        stored = item.get("text")
        spans.append(
            RawSpan(item["start"], item["end"], str(item["label"]),
                    None if stored is None else str(stored))
        )
    return spans


def iter_nemotron(
    path: str | Path, tally: LoadTally, limit: int | None = None
) -> Iterator[Document]:
    """Stream Documents from one Nemotron-PII parquet file, updating ``tally``."""
    for n, row in enumerate(iter_rows(path, COLUMNS)):
        if limit is not None and n >= limit:
            return
        uid = row["uid"]
        try:
            raw = parse_spans(row["spans"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}: uid {uid} ({row['locale']}): {exc}") from exc
        meta = {c: row[c] for c in META_COLUMNS if row[c] is not None}
        meta["group_id"] = uid
        yield build_document(f"nemotron:{uid}:{row['locale']}", row["text"], raw, tally, meta)


def load_nemotron(
    path: str | Path,
    merge_adjacent: bool = True,
    limit: int | None = None,
    *,
    strict_labels: bool = False,
    offset_check: str = "casefold",
    overlap_policy: str = "resolve",
) -> tuple[list[Document], LoadReport]:
    """Load one Nemotron-PII parquet file into Documents plus a LoadReport."""
    options = LoadOptions(merge_adjacent, strict_labels, offset_check, overlap_policy)
    tally = LoadTally(DATASET, options)
    docs = list(iter_nemotron(path, tally, limit))
    return docs, tally.freeze()
