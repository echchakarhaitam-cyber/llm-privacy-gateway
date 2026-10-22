"""Milestone 4 analyses written to results/m4/ (inputs: verified evaluation sets, models).

Usage:  python scripts/m4_analysis.py {alignment,cards,weights,curve} [--data-root DIR]

alignment  tokenisation ceiling: gold -> BIO -> spans scored against gold, per set
cards      Luhn check of the gold CREDIT_CARD spans, per set
weights    top CRF transition weights and state features of the selected model
curve      learning-curve figure docs/figures/m4_learning_curve.png from learning_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from privgate.bench.cards import luhn_report
from privgate.bench.runner import write_json
from privgate.bench.sets import SET_NAMES, load_evaluation_set
from privgate.evaluation import evaluate_corpus
from privgate.ml.bio import round_trip

OUT = Path("results/m4")


def alignment(data_root: Path) -> dict:
    out = {}
    for name in SET_NAMES:
        docs, sha = load_evaluation_set(name, data_root)
        golds = [list(d.spans) for d in docs]
        trips = [round_trip(d.text, d.spans) for d in docs]
        rep = evaluate_corpus(zip(golds, trips))
        def key(spans):
            return [(s.start, s.end, s.label) for s in spans]

        changed = sum(key(g) != key(t) for g, t in zip(golds, trips))
        out[name] = {
            "set_sha256": sha,
            "documents": len(docs),
            "documents_with_changed_spans": changed,
            "gold_spans": sum(map(len, golds)),
            "round_trip_spans": sum(map(len, trips)),
            "strict_f1": rep.strict.micro.f1,
            "partial_f1": rep.partial.micro.f1,
            "type_agnostic_f1": rep.type_agnostic.micro.f1,
            "leakage_rate": rep.chars.leakage_rate,
            "char_precision": rep.chars.char_precision,
        }
    return out


def cards(data_root: Path) -> dict:
    return {name: luhn_report(load_evaluation_set(name, data_root)[0]) for name in SET_NAMES}


def weights(model_dir: Path, top_transitions: int = 20, top_states: int = 10) -> dict:
    from privgate.ml.crf import CRFDetector

    det = CRFDetector.load(model_dir)
    trans = sorted(det.transition_weights().items(), key=lambda kv: -abs(kv[1]))
    states: dict[str, list] = {}
    for (feature, tag), w in det.state_weights().items():
        if tag == "O":
            continue
        states.setdefault(tag.split("-", 1)[1], []).append([feature, tag, w])
    return {
        "model": str(model_dir),
        "top_transitions": [[a, b, w] for (a, b), w in trans[:top_transitions]],
        "top_state_features": {label: sorted(rows, key=lambda r: -r[2])[:top_states]
                               for label, rows in sorted(states.items())},
    }


def curve(path: Path, figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "type_agnostic_f1", "type-agnostic micro F1"),
                              (axes[1], "leakage_rate", "character leakage rate")):
        for set_name in SET_NAMES:
            pts = [r for r in rows if r["set"] == set_name]
            ns = [r["n"] for r in pts]
            mean = [r[metric]["mean"] for r in pts]
            err = [[m - r[metric]["min"] for m, r in zip(mean, pts)],
                   [r[metric]["max"] - m for m, r in zip(mean, pts)]]
            ax.errorbar(ns, mean, yerr=err, marker="o", capsize=3, label=set_name)
        ax.set_xscale("log")
        sizes = sorted({r["n"] for r in rows})
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("training documents (log scale)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle("CRF learning curve (error bars: min-max over training seeds)")
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("alignment", "cards", "weights", "curve"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/m4/crf"))
    args = parser.parse_args(argv)
    if args.command == "alignment":
        write_json(OUT / "alignment.json", alignment(args.data_root))
    elif args.command == "cards":
        write_json(OUT / "cards.json", cards(args.data_root))
    elif args.command == "weights":
        write_json(OUT / "crf_weights.json", weights(args.model))
    else:
        curve(OUT / "learning_curve.json", Path("docs/figures/m4_learning_curve.png"))
    print(f"done: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
