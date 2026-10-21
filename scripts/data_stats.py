"""Dataset statistics -> results/data_stats.json and the generated blocks of docs/DATA.md.

Usage:  python scripts/data_stats.py [--root DIR] [--docs FILE] [--out FILE]

Every number in docs/DATA.md comes from this script: it streams the raw files once (load
reports, full-split statistics, file facts), re-reads the evaluation sets after checking
their SHA-256 against data/manifest.json, and rewrites the text between the
``<!-- name:begin -->`` / ``<!-- name:end -->`` markers. Text outside the markers is never
touched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from privgate.data.mapping import LABEL_MAPS, mapping_table_markdown
from privgate.data.splits import (
    EVALUATION_SETS,
    RAW_FILES,
    file_sha256,
    iter_file,
    read_jsonl,
)
from privgate.data.stats import SetStats
from privgate.spans import EntityType

ID_COLUMNS = {
    "nemotron_train_file": "uid",
    "nemotron_test_file": "uid",
    "gretel_en_test_file": "__index_level_0__",
}
SPLIT_NAMES = {
    "nemotron_train_file": {"train": "nemotron_train", "dev": "nemotron_dev"},
    "nemotron_test_file": {"test": "nemotron_test"},
    "gretel_en_test_file": {"test": "gretel_en_test"},
}


def file_facts(path: Path, id_column: str) -> dict:
    table = pq.read_table(path, columns=[id_column, "domain"])
    ids = table.column(0).to_pylist()
    per_id = Counter(Counter(ids).values())
    return {"rows": len(ids), "distinct_ids": len(set(ids)),
            "rows_per_id": {str(k): v for k, v in sorted(per_id.items())},
            "domains": sorted(set(table.column(1).to_pylist()))}


def collect(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    splits: dict[str, SetStats] = {}
    reports, facts = {}, {}
    for key in RAW_FILES:
        pairs, tally = iter_file(root, key)
        for split, doc in pairs:
            splits.setdefault(SPLIT_NAMES[key][split], SetStats()).add(doc)
        reports[key] = tally.freeze().to_dict()
        facts[key] = file_facts(root / RAW_FILES[key], ID_COLUMNS[key])

    eval_sets = {}
    for spec in EVALUATION_SETS:
        entry = manifest["evaluation_sets"][spec.name]
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise SystemExit(f"{path}: SHA-256 differs from manifest; rebuild the sets")
        stats = SetStats()
        for doc in read_jsonl(path):
            stats.add(doc)
        eval_sets[spec.name] = {**stats.to_dict(), "sha256": entry["sha256"]}

    return {
        "generated_by": "scripts/data_stats.py",
        "evaluation_sets": eval_sets,
        "splits": {name: s.to_dict() for name, s in sorted(splits.items())},
        "load_reports": reports,
        "file_facts": facts,
    }


# --- markdown rendering -----------------------------------------------------------------

def _n(x: int) -> str:
    return f"{x:,}"


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render_sets(title: str, sets: dict) -> list[str]:
    names = list(sets)
    lines = [f"**{title}**", "", "| | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for label, key in (("documents", "documents"), ("characters", "characters"),
                       ("gold spans", "spans")):
        lines.append(f"| {label} | " + " | ".join(_n(sets[s][key]) for s in names) + " |")
    lines.append("| PII share of characters | "
                 + " | ".join(_pct(sets[s]["pii_char_share"]) for s in names) + " |")
    for e in EntityType:
        lines.append(f"| spans: {e.value} | "
                     + " | ".join(_n(sets[s]["spans_per_label"][e.value]) for s in names) + " |")
    lines += ["", "Top domains (documents):", ""]
    for s in names:
        top = ", ".join(f"{d} ({_n(c)})" for d, c in sets[s]["top_domains"])
        lines.append(f"- `{s}`: {top}")
    return lines + [""]


def render_stats(data: dict) -> str:
    lines = render_sets("Evaluation sets", data["evaluation_sets"])
    lines += ["SHA-256 of the evaluation-set files:", ""]
    lines += [f"- `{s}`: `{v['sha256']}`" for s, v in data["evaluation_sets"].items()]
    lines += [""] + render_sets("Full splits", data["splits"])
    return "\n".join(lines).rstrip()


def render_reports(data: dict) -> str:
    reports = data["load_reports"]
    keys = list(reports)
    lines = ["| | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    rows = [("documents", lambda r: r["documents"]),
            ("spans read", lambda r: r["spans_read"])]
    rows += [(f"dropped: {reason}", lambda r, reason=reason: r["spans_dropped"][reason])
             for reason in next(iter(reports.values()))["spans_dropped"]]
    rows += [("merged away (merge_adjacent)", lambda r: r["spans_merged"]),
             ("spans kept", lambda r: r["spans_kept"]),
             ("trimmed (surrounding whitespace)", lambda r: r["spans_trimmed"]),
             ("case-only text mismatches (kept)", lambda r: r["case_only_mismatches"]),
             ("overlapping gold pairs found", lambda r: r["overlapping_pairs"]),
             ("unknown raw labels", lambda r: sum(r["unknown_labels"].values()))]
    for name, get in rows:
        lines.append(f"| {name} | " + " | ".join(_n(get(reports[k])) for k in keys) + " |")
    lines.append("| offset mismatch rate (dropped / read) | " + " | ".join(
        _pct(reports[k]["spans_dropped"]["offset_mismatch"] / reports[k]["spans_read"])
        for k in keys) + " |")
    lines.append("| case-only mismatch rate | " + " | ".join(
        _pct(reports[k]["case_only_mismatches"] / reports[k]["spans_read"]) for k in keys) + " |")
    return "\n".join(lines)


def render_facts(data: dict) -> str:
    facts = data["file_facts"]
    lines = ["| file | rows | distinct ids | rows per id | distinct domains |",
             "|---|---|---|---|---|"]
    for key, f in facts.items():
        per = ", ".join(f"{k}: {_n(v)} ids" for k, v in f["rows_per_id"].items())
        lines.append(f"| {key} | {_n(f['rows'])} | {_n(f['distinct_ids'])} | {per} "
                     f"| {_n(len(f['domains']))} |")
    shared = sorted(set(facts["nemotron_train_file"]["domains"])
                    & set(facts["nemotron_test_file"]["domains"]))
    summary = f"Domains present in both Nemotron files: {_n(len(shared))} "
    lines += ["", summary + f"({', '.join(shared) or 'none'})."]
    return "\n".join(lines)


def render_labels(data: dict) -> str:
    reports = data["load_reports"]
    lines = []
    for dataset, table in LABEL_MAPS.items():
        keys = [k for k, r in reports.items() if r["dataset"] == dataset]
        lines += [f"**{dataset}**", "", "| raw label | mapped to | " + " | ".join(keys) + " |",
                  "|---" * (len(keys) + 2) + "|"]
        totals = Counter()
        for k in keys:
            totals.update(reports[k]["raw_label_counts"])
        for label, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
            mapped = table[label].value if label in table else "(unmapped)"
            counts = " | ".join(_n(reports[k]["raw_label_counts"].get(label, 0)) for k in keys)
            lines.append(f"| `{label}` | {mapped} | {counts} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def replace_block(document: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:begin -->)\n.*?(<!-- {name}:end -->)", re.DOTALL)
    if not pattern.search(document):
        raise SystemExit(f"marker block {name!r} not found")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--docs", type=Path, default=Path("docs/DATA.md"))
    parser.add_argument("--out", type=Path, default=Path("results/data_stats.json"))
    args = parser.parse_args(argv)

    data = collect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    doc = args.docs.read_text(encoding="utf-8")
    for name, content in (("mapping", mapping_table_markdown()), ("facts", render_facts(data)),
                          ("labels", render_labels(data)), ("reports", render_reports(data)),
                          ("stats", render_stats(data))):
        doc = replace_block(doc, name, content)
    with args.docs.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    print(f"wrote {args.out} and updated {args.docs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
