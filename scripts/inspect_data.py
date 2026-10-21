"""Print the schema findings of the raw files: what docs/DATA.md "Schema findings" is based on.

Usage:  python scripts/inspect_data.py [--root DIR]

For each raw parquet file: row count, Arrow schema, two example rows (long values
truncated), how the span column is encoded, the keys of the span dicts, how often the
stored span text differs from ``text[start:end]`` and the distinct raw labels with counts.
Read-only; nothing is written.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from privgate.data.splits import RAW_FILES

TRUNCATE = 160


def _short(value: object) -> str:
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= TRUNCATE else s[:TRUNCATE] + f"... [{len(s)} chars]"


def _decode(field: str) -> tuple[list, str]:
    try:
        return json.loads(field), "json"
    except json.JSONDecodeError:
        return ast.literal_eval(field), "python-literal"


def inspect(path: Path) -> None:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    text_col = "text" if "text" in names else "generated_text"
    span_col = "spans" if "spans" in names else "pii_spans"
    print("=" * 80, f"\n{path}\nrows: {pf.metadata.num_rows}\nschema:")
    for field in pf.schema_arrow:
        print(f"  {field.name}: {field.type}")
    for i, row in enumerate(pf.read_row_group(0).slice(0, 2).to_pylist()):
        print(f"example row {i}:")
        for k, v in row.items():
            print(f"  {k}: {_short(v)}")

    encodings, keys, labels, stored = Counter(), Counter(), Counter(), Counter()
    for batch in pf.iter_batches(columns=[text_col, span_col], batch_size=5000):
        for text, field in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            spans, encoding = _decode(field)
            encodings[encoding] += 1
            for s in spans:
                keys[tuple(sorted(s))] += 1
                labels[s["label"]] += 1
                if "text" in s:
                    covered = text[s["start"]:s["end"]]
                    value = str(s["text"])
                    stored["equal" if covered == value else
                           "case-only" if covered.casefold() == value.casefold() else "other"] += 1
    print(f"span column encoding (rows): {dict(encodings)}")
    print(f"span dict keys: {dict(keys)}")
    print(f"stored text vs text[start:end]: {dict(stored) or 'no stored text'}")
    print(f"distinct raw labels: {len(labels)}")
    for label, n in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {label}: {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    for rel in RAW_FILES.values():
        inspect(args.root / rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
