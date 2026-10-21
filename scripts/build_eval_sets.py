"""Build the named evaluation sets from data/raw/ and record them in data/manifest.json.

Usage:  python scripts/build_eval_sets.py [--root DIR]

Writes data/processed/<name>.jsonl for every set in privgate.data.splits.EVALUATION_SETS and
stores, under "evaluation_sets" in the manifest, each file's SHA-256, size and the exact
recipe that produced it. Re-running on the same raw files gives byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from privgate.data.splits import (
    DEFAULT_OPTIONS,
    DEV_PERCENT,
    EVALUATION_SETS,
    RAW_FILES,
    build_evaluation_sets,
    write_jsonl,
)

SPLIT_RULE = (
    f"dev iff int(sha256(meta['group_id'] or doc_id), 16) % 100 < {DEV_PERCENT}, else train "
    "(Nemotron train file only); the Nemotron test file is test"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data"), help="data directory")
    args = parser.parse_args(argv)
    root: Path = args.root

    sets, reports = build_evaluation_sets(root, DEFAULT_OPTIONS)
    entries = {}
    for spec in EVALUATION_SETS:
        docs = sets[spec.name]
        path = root / "processed" / f"{spec.name}.jsonl"
        entries[spec.name] = {
            "path": path.as_posix(),
            "sha256": write_jsonl(docs, path),
            "documents": len(docs),
            "spans": sum(len(d.spans) for d in docs),
            "source_file": (root / RAW_FILES[spec.source_file]).as_posix(),
            "split": spec.split,
            "size": spec.size,
            "seed": spec.seed,
            "definition": spec.description,
            "load_options": reports[spec.source_file].to_dict()["options"],
        }
        print(f"{spec.name}: {len(docs)} documents -> {path} sha256={entries[spec.name]['sha256']}")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {**manifest, "evaluation_sets": entries, "split_rule": SPLIT_RULE}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
