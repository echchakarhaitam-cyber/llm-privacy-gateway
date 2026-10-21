"""Generate the tables of docs/results-detection.md and docs/DETECTORS.md from results/m3/.

Usage:  python scripts/make_detection_tables.py [--results DIR] [--data-root DIR]

Inputs are only committed artifacts: results/m3/<set>/<detector>/metrics.json,
results/m3/presidio_sweep.json, results/m3/run_log.json and the "models" entry of
data/manifest.json, plus label maps and support tables read from the code. Every number in
the two documents is written here, between ``<!-- name:begin -->`` / ``<!-- name:end -->``
markers; text outside the markers is never touched.

"Best detector" (RQ3) is fixed in advance as the detector with the lowest character
leakage on nemotron_dev_2k (all gold spans), ties broken by type-agnostic F1; like the
Presidio threshold it is chosen on dev, never on a test set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from privgate.bench.registry import ALL_DETECTORS, members
from privgate.bench.sets import DEV_SET, SET_NAMES
from privgate.bench.support import NATIVE_RAW_TARGETS, targeted_raw_labels, untargeted_by_all
from privgate.data.mapping import LABEL_MAPS
from privgate.detectors.presidio import PRESIDIO_LABEL_MAP
from privgate.detectors.regex import RegexDetector
from privgate.detectors.spacy_ner import NON_NER_PIPES, SPACY_LABEL_MAP
from privgate.spans import EntityType

SET_DATASET = {"nemotron_dev_2k": "nemotron-pii", "nemotron_test_5k": "nemotron-pii",
               "gretel_en_test": "gretel-finance-en"}


# --- loading -----------------------------------------------------------------------------

def load_results(root: Path) -> dict:
    metrics = {}
    for s in SET_NAMES:
        for d in ALL_DETECTORS:
            path = root / s / d / "metrics.json"
            if not path.exists():
                raise SystemExit(f"missing {path}; run python -m privgate.bench all")
            metrics[(s, d)] = json.loads(path.read_text(encoding="utf-8"))
    sweep = json.loads((root / "presidio_sweep.json").read_text(encoding="utf-8"))
    log = json.loads((root / "run_log.json").read_text(encoding="utf-8"))
    return {"metrics": metrics, "sweep": sweep, "log": log}


def view(m: dict, name: str = "all") -> dict:
    return m["views"][name]


def f1(m: dict, mode: str, v: str = "all") -> float:
    return view(m, v)[mode]["micro"]["f1"]


def ci(m: dict, key: str, v: str = "all") -> tuple[float, float]:
    lo, hi = view(m, v)["ci95"][key]
    return lo, hi


def leak(m: dict, v: str = "all") -> float:
    return view(m, v)["chars"]["leakage_rate"]


def best_detector(metrics: dict) -> str:
    return min(ALL_DETECTORS, key=lambda d: (leak(metrics[(DEV_SET, d)]),
                                             -f1(metrics[(DEV_SET, d)], "type_agnostic")))


# --- formatting --------------------------------------------------------------------------

def n3(x: float) -> str:
    return f"{x:.3f}"


def with_ci(x: float, bounds: tuple[float, float]) -> str:
    return f"{x:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ms(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows)]


# --- results-detection.md blocks ---------------------------------------------------------

def block_rq1(r: dict) -> str:
    out = []
    for v, title in (("all", "All gold spans (privacy view)"),
                     ("supported", "Gold restricted to each detector's supported labels")):
        rows = []
        for s in SET_NAMES:
            for d in ALL_DETECTORS:
                m = r["metrics"][(s, d)]
                rows.append([f"`{s}`", f"`{d}`",
                             with_ci(f1(m, "strict", v), ci(m, "strict_micro_f1", v)),
                             with_ci(f1(m, "partial", v), ci(m, "partial_micro_f1", v)),
                             with_ci(f1(m, "type_agnostic", v), ci(m, "type_agnostic_micro_f1", v)),
                             with_ci(leak(m, v), ci(m, "leakage_rate", v)),
                             n3(view(m, v)["chars"]["char_precision"]),
                             ms(m["latency"]["median_ms"])])
        out += [(f"**{title}** (micro F1 and leakage with 95% bootstrap CI; latency = median "
                "ms per document)"), ""]
        out += table(["set", "detector", "strict F1", "partial F1", "type-agnostic F1",
                      "leakage rate", "char precision", "median ms"], rows)
        out.append("")
    return "\n".join(out).rstrip()


def block_rq2(r: dict) -> str:
    rows = []
    for d in ALL_DETECTORS:
        ms_ = [r["metrics"][(s, d)] for s in SET_NAMES]
        rows.append([f"`{d}`", *(n3(f1(m, "strict")) for m in ms_),
                     *(with_ci(f1(m, "type_agnostic"), ci(m, "type_agnostic_micro_f1"))
                       for m in ms_),
                     *(with_ci(leak(m), ci(m, "leakage_rate")) for m in ms_)])
    short = ["dev", "test", "gretel"]
    header = (["detector"] + [f"strict F1 {s}" for s in short]
              + [f"type-agnostic F1 {s}" for s in short] + [f"leakage {s}" for s in short])
    return "\n".join([("All gold spans; dev = `nemotron_dev_2k`, test = `nemotron_test_5k`, "
                      "gretel = `gretel_en_test`."), "", *table(header, rows)])


def untargeted(dataset: str, r: dict) -> frozenset[str]:
    dets = {}
    for d in ALL_DETECTORS:
        m = r["metrics"][(DEV_SET, d)]
        dets[d] = (members(d), [EntityType(x) for x in m["supported_labels"]])
    return untargeted_by_all(dets, dataset)


NOT_MEASURED = "NOT MEASURED"


def unsupported_share(m: dict, dataset: str, r: dict) -> tuple[float, int, int] | None:
    """None when the run had no raw-label index (raw files absent)."""
    per_raw = view(m)["chars"].get("per_raw_label")
    if per_raw is None:
        return None
    left = untargeted(dataset, r)
    leaked = sum(v["leaked_chars"] for v in per_raw.values())
    from_left = sum(v["leaked_chars"] for k, v in per_raw.items() if k in left)
    return (from_left / leaked if leaked else 0.0), from_left, leaked


def block_rq3(r: dict) -> str:
    best = best_detector(r["metrics"])
    out = [f"Best detector (lowest leakage on `{DEV_SET}`, chosen on dev): `{best}`.", ""]
    rows = []
    for e in EntityType:
        cells = []
        for s in SET_NAMES:
            per = view(r["metrics"][(s, best)])["chars"]["per_gold_label"].get(e.value)
            cells.append("--" if per is None else
                         f"{pct(per['leakage_rate'])} of {per['gold_chars']:,}")
        rows.append([e.value, *cells])
    out += [("Leakage rate per gold label (share of that label's characters left visible, of "
            "N gold characters):"), ""]
    out += table(["gold label", *SET_NAMES], rows)
    out += ["", ("Share of all leaked gold characters that belong to raw labels NO tested "
            "detector targets:"), ""]
    rows = []
    for d in ALL_DETECTORS:
        cells = []
        for s in SET_NAMES:
            res = unsupported_share(r["metrics"][(s, d)], SET_DATASET[s], r)
            cells.append(NOT_MEASURED if res is None else
                         f"{pct(res[0])} ({res[1]:,} of {res[2]:,})")
        rows.append([f"`{d}`", *cells])
    out += table(["detector", *SET_NAMES], rows)
    out += ["", (f"Raw labels leaking the most characters for `{best}` on `{DEV_SET}` "
            "(top 12):"), ""]
    per_raw = view(r["metrics"][(DEV_SET, best)])["chars"].get("per_raw_label")
    if per_raw is None:
        return "\n".join([*out, NOT_MEASURED])
    total = sum(v["leaked_chars"] for v in per_raw.values())
    left = untargeted("nemotron-pii", r)
    rows = []
    for k, v in sorted(per_raw.items(), key=lambda kv: (-kv[1]["leaked_chars"], kv[0]))[:12]:
        mapped = LABEL_MAPS["nemotron-pii"].get(k)
        rows.append([f"`{k}`", mapped.value if mapped else "?", "no" if k in left else "yes",
                     f"{v['leaked_chars']:,}", pct(v["leaked_chars"] / total if total else 0.0),
                     pct(v["leakage_rate"])])
    out += table(["raw label", "mapped", "targeted by some detector", "leaked chars",
                  "share of leaked", "leakage rate"], rows)
    return "\n".join(out)


def block_support(r: dict) -> str:
    out = [("Raw labels targeted by NO tested detector (derived from `supported_labels` and "
           "`privgate.bench.support.NATIVE_RAW_TARGETS`):"), ""]
    for dataset in LABEL_MAPS:
        labels = ", ".join(f"`{x}`" for x in sorted(untargeted(dataset, r)))
        out.append(f"- {dataset}: {labels}")
    out += ["", "Explicit raw-label targets of native types inside OTHER / ID_NUMBER:", ""]
    for det, table_ in NATIVE_RAW_TARGETS.items():
        pairs = "; ".join(f"{k} -> {', '.join(sorted(v))}" for k, v in table_.items()) or "none"
        out.append(f"- `{det}`: {pairs}")
    return "\n".join(out)


def block_sweep(r: dict) -> str:
    sw = r["sweep"]
    rows = [[f"{x['threshold']:.2f}", f"{x['pred_spans']:,}",
             n3(x["strict_f1"]), n3(x["partial_f1"]), n3(x["type_agnostic_precision"]),
             n3(x["type_agnostic_recall"]), n3(x["type_agnostic_f1"]), n3(x["leakage_rate"]),
             n3(x["char_precision"]),
             "yes" if x["threshold"] in sw["eligible_thresholds"] else "no"] for x in sw["rows"]]
    out = [f"Set: `{sw['set']}` (all gold spans). Rule: {sw['rule']}", ""]
    out += table(["threshold", "pred spans", "strict F1", "partial F1", "TA precision",
                  "TA recall", "TA F1", "leakage", "char precision", "eligible"], rows)
    out += ["", (f"Minimum leakage {n3(sw['minimum_leakage_rate'])}; eligible thresholds "
            f"{', '.join(f'{t:.2f}' for t in sw['eligible_thresholds'])}; "
            f"**selected threshold: {sw['selected_threshold']:.2f}**, used unchanged on "
            "`nemotron_test_5k` and `gretel_en_test`.")]
    return "\n".join(out)


def block_latency(r: dict) -> str:
    env = r["metrics"][(DEV_SET, "regex")]["environment"]
    out = [(f"Machine: {env['cpu']} ({env['logical_cpus']} logical CPUs), {env['os']}, "
           f"Python {env['python']}. Latency depends on this laptop and its load; compare "
           "detectors within this table only."), ""]
    rows = []
    for s in SET_NAMES:
        for d in ALL_DETECTORS:
            lat = r["metrics"][(s, d)]["latency"]
            rows.append([f"`{s}`", f"`{d}`", ms(lat["median_ms"]), ms(lat["p95_ms"]),
                         ms(lat["mean_ms"]), f"{lat['chars_per_second']:,.0f}"])
    out += table(["set", "detector", "median ms", "p95 ms", "mean ms", "chars/s"], rows)
    return "\n".join(out)


def block_runtime(r: dict) -> str:
    by_step: dict[str, float] = {}
    for s in r["log"]["steps"]:
        by_step[s["step"]] = by_step.get(s["step"], 0.0) + s["seconds"]
    total = sum(by_step.values())
    parts = ", ".join(f"{k} {v / 60:.1f} min" for k, v in sorted(by_step.items()))
    return (f"Total wall time of all benchmark steps: {total / 60:.1f} min ({parts}); from "
            "`results/m3/run_log.json`. Model loading is included in `detect`.")


def block_observations(r: dict) -> str:
    met = r["metrics"]
    lines = []
    for s in SET_NAMES:
        by_leak = sorted(ALL_DETECTORS, key=lambda d: leak(met[(s, d)]))
        by_f1 = sorted(ALL_DETECTORS, key=lambda d: -f1(met[(s, d)], "type_agnostic"))
        a, b = by_leak[0], by_leak[1]
        lo_a, hi_a = ci(met[(s, a)], "leakage_rate")
        lo_b, hi_b = ci(met[(s, b)], "leakage_rate")
        overlap = "overlap" if lo_b <= hi_a and lo_a <= hi_b else "do not overlap"
        lines.append(
            f"- `{s}`: lowest leakage `{a}` {with_ci(leak(met[(s, a)]), (lo_a, hi_a))}, next "
            f"`{b}` {with_ci(leak(met[(s, b)]), (lo_b, hi_b))} (intervals {overlap}); highest "
            f"type-agnostic F1 `{by_f1[0]}` {n3(f1(met[(s, by_f1[0])], 'type_agnostic'))}.")
    for d in ALL_DETECTORS:
        seq = " -> ".join(n3(f1(met[(s, d)], "type_agnostic")) for s in SET_NAMES)
        lk = " -> ".join(n3(leak(met[(s, d)])) for s in SET_NAMES)
        sup = n3(leak(met[(DEV_SET, d)], "supported"))
        lines.append(f"- `{d}`: type-agnostic F1 dev -> test -> gretel {seq}; leakage {lk}; "
                     f"dev leakage on its supported labels only {sup}.")
    best = best_detector(met)
    for s in SET_NAMES:
        res = unsupported_share(met[(s, best)], SET_DATASET[s], r)
        share = NOT_MEASURED if res is None else pct(res[0])
        lines.append(f"- `{best}` on `{s}`: {share} of the leaked gold characters belong "
                     "to raw labels no tested detector targets.")
    lines.append(f"- Presidio threshold selected on dev: {r['sweep']['selected_threshold']:.2f}.")
    return "\n".join(lines)


# --- DETECTORS.md blocks -----------------------------------------------------------------

def block_versions(r: dict, data_root: Path) -> str:
    env = r["metrics"][(DEV_SET, "spacy")]["environment"]
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["models"]["en_core_web_lg"]
    rows = [[f"`{p}`", f"`{v}`" if v else "not installed"] for p, v in env["packages"].items()]
    out = table(["package", "installed version (as recorded in metrics.json)"], rows)
    out += ["", (f"spaCy model: `{model['name']}` {model['version']} ({model['license']}), wheel "
            f"`{model['wheel']}` from {model['url']}, {model['size']:,} bytes, SHA-256 "
            f"`{model['sha256']}`."), "",
            f"Recorded on: {env['cpu']}, {env['os']}, Python {env['python']}."]
    return "\n".join(out)


def block_maps() -> str:
    out = ["**spaCy** (`SpacyDetector`, name `spacy`): pipes excluded at load: "
           + ", ".join(f"`{p}`" for p in NON_NER_PIPES) + ".", ""]
    out += table(["spaCy label", "privgate type"],
                 [[f"`{k}`", v.value] for k, v in SPACY_LABEL_MAP.items()])
    out += ["", ("All other spaCy labels (CARDINAL, EVENT, LANGUAGE, LAW, MONEY, ORDINAL, "
            "PERCENT, PRODUCT, QUANTITY, WORK_OF_ART) are ignored."), "",
            "**Presidio** (`PresidioDetector`, name `presidio`):", ""]
    out += table(["Presidio entity type", "privgate type"],
                 [[f"`{k}`", v.value] for k, v in PRESIDIO_LABEL_MAP.items()])
    out += ["", "**Regex baseline** (`RegexDetector`, name `regex`): "
            + ", ".join(sorted(e.value for e in RegexDetector.supported_labels)) + "."]
    return "\n".join(out)


def block_supported(r: dict) -> str:
    rows = []
    for d in ALL_DETECTORS:
        labels = r["metrics"][(DEV_SET, d)]["supported_labels"]
        targeted = targeted_raw_labels(members(d), [EntityType(x) for x in labels], "nemotron-pii")
        rows.append([f"`{d}`", ", ".join(labels), str(len(targeted))])
    return "\n".join(table(["detector", "supported_labels", "Nemotron raw labels targeted"],
                           rows))


# --- writing -----------------------------------------------------------------------------

def replace_block(document: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:begin -->)\n.*?(<!-- {name}:end -->)", re.DOTALL)
    if not pattern.search(document):
        raise SystemExit(f"marker block {name!r} not found")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", document)


def rewrite(path: Path, blocks: dict[str, str]) -> None:
    doc = path.read_text(encoding="utf-8")
    for name, content in blocks.items():
        doc = replace_block(doc, name, content)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/m3"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-doc", type=Path, default=Path("docs/results-detection.md"))
    parser.add_argument("--detectors-doc", type=Path, default=Path("docs/DETECTORS.md"))
    args = parser.parse_args(argv)
    r = load_results(args.results)
    rewrite(args.results_doc, {
        "rq1": block_rq1(r), "rq2": block_rq2(r), "rq3": block_rq3(r),
        "support": block_support(r), "sweep": block_sweep(r), "latency": block_latency(r),
        "runtime": block_runtime(r), "observations": block_observations(r),
    })
    rewrite(args.detectors_doc, {
        "versions": block_versions(r, args.data_root), "maps": block_maps(),
        "supported": block_supported(r),
    })
    print(f"updated {args.results_doc} and {args.detectors_doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
