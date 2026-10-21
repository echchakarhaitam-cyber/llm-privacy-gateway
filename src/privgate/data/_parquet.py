"""Batched parquet reading shared by the loaders; pyarrow is imported only here, lazily.

Why lazily: the core library (spans, detectors, metrics, gateway) must stay importable
without the optional ``data`` extra. Why batched: a 150 MB parquet file expands to several
hundred MB of Python strings; reading one batch at a time keeps a single copy in memory
(the Documents being built), not the Arrow table plus a Python list plus the Documents.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

BATCH_ROWS = 1000


def iter_rows(path: str | Path, columns: Sequence[str]) -> Iterator[dict]:
    """Yield the requested columns of each row as a dict, one batch in memory at a time."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading datasets needs pyarrow: pip install -e '.[data]'"
        ) from exc
    pf = pq.ParquetFile(path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {pf.schema_arrow.names}")
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=list(columns)):
        values = [batch.column(c).to_pylist() for c in columns]
        for row in zip(*values):
            yield dict(zip(columns, row))
