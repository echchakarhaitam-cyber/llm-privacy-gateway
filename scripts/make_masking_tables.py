"""Generate the tables of docs/results-masking.md from results/m5/ (Milestone 5).

Usage:  python scripts/make_masking_tables.py [--results DIR] [--doc FILE]

Inputs: results/m5/masking.json, results/m5/domain_classifier.json and the strategy
constants of privgate.masking. Every number in the document is written here, between
``<!-- name:begin -->`` / ``<!-- name:end -->`` markers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from privgate.bench.sets import DEV_SET, SET_NAMES
from privgate.masking import (
    PARTIAL_KEEP_DIGITS,
    PARTIAL_MIN_DIGITS,
    PARTIAL_NUMBER_LABELS,
    PSEUDONYM_FALLBACK,
    PSEUDONYM_PROVIDERS,
    STRATEGIES,
)
from privgate.spans import EntityType


def n3(x: float) -> str:
    return f"{x:.3f}"


def ci(v: float, bounds) -> str:
    return f"{v:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"


def sci(v: float, bounds) -> str:
    return f"{v:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"


def table(header, rows) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(map(str, r)) + " |" for r in rows)]


def rows_of(data, set_name):
    return [r for r in data["rows"] if r["set"] == set_name]


def find(data, set_name, detector, strategy):
    return next(r for r in data["rows"] if (r["set"], r["detector"], r["strategy"]) ==
                (set_name, detector, strategy))


def block_fallbacks() -> str:
    fakes = ", ".join(f"{e.value} -> Faker `{p}()`" for e, p in PSEUDONYM_PROVIDERS.items())
    typed_pseudo = ", ".join(sorted(e.value for e in PSEUDONYM_FALLBACK))
    partial_kept = ", ".join(sorted(e.value for e in PARTIAL_NUMBER_LABELS))
    partial_typed = ", ".join(sorted(e.value for e in EntityType
                                     if e not in PARTIAL_NUMBER_LABELS and e is not
                                     EntityType.EMAIL))
    return "\n".join([
        (f"- `pseudonym`: {fakes}. **Fall back to typed**: {typed_pseudo}; also any value for "
        "which no fake could be found that is absent from the text and stays a whole token."),
        (f"- `partial`: EMAIL keeps `@domain`; {partial_kept} keep their last "
        f"{PARTIAL_KEEP_DIGITS} digits and separators when they have at least "
        f"{PARTIAL_MIN_DIGITS} digits. **Fall back to typed**: {partial_typed}; EMAIL without "
        f"`@`; {partial_kept} with fewer than {PARTIAL_MIN_DIGITS} digits."),
        "- `redact` and `typed` have no fallback."])


def block_classifier(clf) -> str:
    out = [(f"Training sample: {clf['n']:,} Nemotron train-split documents ({clf['groups']:,} "
           f"persona groups, seed {clf['seed']}); training wall time "
           f"{clf['seconds'] / 60:.1f} min."), ""]
    rows = []
    for name, s in clf["settings"].items():
        for r in s["cv"]["rows"]:
            rows.append([name, r["C"], n3(r["mean_accuracy"]),
                         ", ".join(n3(a) for a in r["fold_accuracy"]),
                         "**selected**" if r["C"] == s["cv"]["selected_C"] else ""])
    out += table(["training text", "C", "mean CV accuracy", "fold accuracies", ""], rows)
    a = clf["settings"]["original"]["dev_original"]
    b = clf["settings"]["typed"]["dev_oracle_typed"]
    out += ["", (f"Dev accuracy of the original-text classifier on unmasked dev: "
            f"**{n3(a['accuracy'])}** (macro-F1 {n3(a['macro_f1'])}); of the typed-text "
            f"classifier on oracle-typed dev: {n3(b['accuracy'])} (macro-F1 "
            f"{n3(b['macro_f1'])}).")]
    return "\n".join(out)


def block_rq7(data) -> str:
    out = []
    un = data["dev_unmasked_original_classifier"]
    for set_name in SET_NAMES:
        dev = set_name == DEV_SET
        header = ["detector", "strategy", "char leakage", "verbatim survival",
                  "preserved non-PII", "length ratio"]
        if dev:
            header += ["domain accuracy", "macro-F1", "utility drop"]
        rows = []
        for r in rows_of(data, set_name):
            p, u = r["privacy"], r["utility"]
            row = [f"`{r['detector']}`", r["strategy"], ci(p["char_leakage"], p["ci95"]["char_leakage"]),
                   ci(p["verbatim_survival"], p["ci95"]["verbatim_survival"]),
                   n3(u["preserved_non_pii"]), n3(u["length_ratio"])]
            if dev:
                d = r["domain_original_classifier"]
                row += [ci(d["accuracy"], d["accuracy_ci95"]), n3(d["macro_f1"]),
                        sci(d["drop_vs_unmasked"]["difference"], d["drop_vs_unmasked"]["ci95"])]
            rows.append(row)
        title = f"**`{set_name}`**"
        if dev:
            title += (f" (unmasked dev: accuracy {ci(un['accuracy'], un['accuracy_ci95'])}, "
                      f"macro-F1 {n3(un['macro_f1'])}; utility drop = unmasked - masked)")
        out += [title, "", *table(header, rows), ""]
    return "\n".join(out).rstrip()


def block_setting_b(data) -> str:
    rows = []
    for r in rows_of(data, DEV_SET):
        if r["strategy"] != "typed":
            continue
        a, b = r["domain_original_classifier"], r["domain_typed_classifier"]
        drop = b["drop_vs_unmasked_original_classifier"]
        rows.append([f"`{r['detector']}`", ci(a["accuracy"], a["accuracy_ci95"]),
                     ci(b["accuracy"], b["accuracy_ci95"]), n3(b["macro_f1"]),
                     sci(drop["difference"], drop["ci95"])])
    return "\n".join(table(["detector (typed masking)", "original-text classifier",
                            "typed-text classifier", "typed-text macro-F1",
                            "drop vs unmasked original classifier"], rows))


def block_perlabel(data) -> str:
    dets = ("oracle", "regex+presidio", "crf+regex+presidio")
    rows = []
    labels = sorted({lab for d in dets
                     for lab in find(data, DEV_SET, d, "typed")["privacy"]["per_label"]})
    for lab in labels:
        cells = []
        for d in dets:
            per = find(data, DEV_SET, d, "typed")["privacy"]["per_label"].get(lab)
            cells.append("--" if per is None else
                         f"{n3(per['char_leakage'])} / {n3(per['verbatim_survival'])}")
        rows.append([lab, *cells])
    return "\n".join(["char leakage / verbatim survival:", "",
                      *table(["gold label", *[f"`{d}`" for d in dets]], rows)])


def block_rq8(data) -> str:
    out = [("Detector minus oracle under the same strategy (positive leakage gap = the "
           "detector leaks more; negative accuracy gap = the detector costs accuracy)."), ""]
    for set_name in SET_NAMES:
        dev = set_name == DEV_SET
        header = ["detector", "strategy", "char leakage gap", "verbatim survival gap",
                  "preserved non-PII gap"]
        if dev:
            header.append("accuracy gap [CI]")
        rows = []
        for r in rows_of(data, set_name):
            if r["detector"] == "oracle":
                continue
            o = find(data, set_name, "oracle", r["strategy"])
            row = [f"`{r['detector']}`", r["strategy"],
                   f"{r['privacy']['char_leakage'] - o['privacy']['char_leakage']:+.3f}",
                   f"{r['privacy']['verbatim_survival'] - o['privacy']['verbatim_survival']:+.3f}",
                   f"{r['utility']['preserved_non_pii'] - o['utility']['preserved_non_pii']:+.3f}"]
            if dev:
                g = r["domain_original_classifier"]["gap_vs_oracle"]
                row.append(sci(g["difference"], g["ci95"]))
            rows.append(row)
        out += [f"**`{set_name}`**", "", *table(header, rows), ""]
    return "\n".join(out).rstrip()


def block_observations(data, clf) -> str:
    lines = []
    dev_rows = rows_of(data, DEV_SET)
    for strategy in STRATEGIES:
        o = find(data, DEV_SET, "oracle", strategy)
        lines.append(f"- `{strategy}`, oracle on dev: char leakage "
                     f"{n3(o['privacy']['char_leakage'])}, verbatim survival "
                     f"{n3(o['privacy']['verbatim_survival'])}, accuracy "
                     f"{n3(o['domain_original_classifier']['accuracy'])}.")
    best = min((r for r in dev_rows if r["detector"] != "oracle"),
               key=lambda r: (r["privacy"]["char_leakage"], r["strategy"]))
    lines.append(f"- Lowest non-oracle char leakage on dev: `{best['detector']}` "
                 f"{best['strategy']} {n3(best['privacy']['char_leakage'])}.")
    worst_drop = max(dev_rows, key=lambda r: r["domain_original_classifier"]
                     ["drop_vs_unmasked"]["difference"])
    d = worst_drop["domain_original_classifier"]["drop_vs_unmasked"]
    lines.append(f"- Largest utility drop on dev: `{worst_drop['detector']}` "
                 f"{worst_drop['strategy']} {sci(d['difference'], d['ci95'])}.")
    for set_name in SET_NAMES:
        r = find(data, set_name, "crf+regex+presidio", "typed")
        o = find(data, set_name, "oracle", "typed")
        lines.append(f"- `{set_name}`, typed: `crf+regex+presidio` char leakage "
                     f"{n3(r['privacy']['char_leakage'])} (oracle {n3(o['privacy']['char_leakage'])}),"
                     f" verbatim survival {n3(r['privacy']['verbatim_survival'])} (oracle "
                     f"{n3(o['privacy']['verbatim_survival'])}).")
    sel = {k: v["cv"]["selected_C"] for k, v in clf["settings"].items()}
    lines.append(f"- Selected classifier C: original text {sel['original']}, typed text "
                 f"{sel['typed']}.")
    return "\n".join(lines)


def replace_block(document: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:begin -->)\n.*?(<!-- {name}:end -->)", re.DOTALL)
    if not pattern.search(document):
        raise SystemExit(f"marker block {name!r} not found")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/m5"))
    parser.add_argument("--doc", type=Path, default=Path("docs/results-masking.md"))
    args = parser.parse_args(argv)
    data = json.loads((args.results / "masking.json").read_text(encoding="utf-8"))
    clf = json.loads((args.results / "domain_classifier.json").read_text(encoding="utf-8"))
    doc = args.doc.read_text(encoding="utf-8")
    for name, content in (("m5_fallbacks", block_fallbacks()),
                          ("m5_classifier", block_classifier(clf)),
                          ("m5_rq7", block_rq7(data)), ("m5_setting_b", block_setting_b(data)),
                          ("m5_perlabel", block_perlabel(data)), ("m5_rq8", block_rq8(data)),
                          ("m5_observations", block_observations(data, clf))):
        doc = replace_block(doc, name, content)
    with args.doc.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    print(f"updated {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
