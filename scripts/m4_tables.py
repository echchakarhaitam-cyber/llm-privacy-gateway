"""Milestone 4 blocks of docs/results-detection.md and docs/DETECTORS.md.

Used by scripts/make_detection_tables.py when results/m4/ exists. Inputs are committed
JSON files only: results/m4/{alignment,cards,crf_grid,logreg_grid,learning_curve,
final_models,training_log,training_sets,crf_weights}.json, results/m4/<set>/<detector>/
metrics.json, results/m4/<set>/paired_<a>__vs__<b>.json and, for the baseline,
results/m3/<set>/regex+presidio/metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from privgate.bench.sets import DEV_SET, SET_NAMES
from privgate.spans import EntityType

BASELINE = "regex+presidio"
M4_DETECTORS = ("crf", "logreg", "crf+regex+presidio")
COMPARISONS = (("crf", BASELINE), ("crf+regex+presidio", BASELINE), ("crf", "logreg"))
SHORT = {"nemotron_dev_2k": "dev", "nemotron_test_5k": "test", "gretel_en_test": "gretel"}


def _read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load(m4: Path, m3: Path) -> dict:
    r = {name: _read(m4 / f"{name}.json") for name in (
        "alignment", "cards", "crf_grid", "logreg_grid", "learning_curve", "final_models",
        "training_log", "training_sets", "crf_weights")}
    r["metrics"] = {(s, d): _read(m4 / s / d / "metrics.json")
                    for s in SET_NAMES for d in M4_DETECTORS}
    r["metrics"].update({(s, BASELINE): _read(m3 / s / BASELINE / "metrics.json")
                         for s in SET_NAMES})
    r["paired"] = {(s, a, b): _read(m4 / s / f"paired_{a}__vs__{b}.json")
                   for s in SET_NAMES for a, b in COMPARISONS}
    return r


def n3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    return f"{x:+.3f}"


def _mb(x: float | None) -> str:
    """Peak memory in MB; the grid runs predate the working memory probe."""
    return "NOT MEASURED" if x is None else f"{x:.0f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(header, rows) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(map(str, row)) + " |" for row in rows)]


def _view(m):
    return m["views"]["all"]


def _ci(m, key):
    lo, hi = _view(m)["ci95"][key]
    return f"[{lo:.3f}, {hi:.3f}]"


def block_alignment(r) -> str:
    rows = [[f"`{s}`", f"{a['gold_spans']:,}", f"{a['round_trip_spans']:,}",
             f"{a['documents_with_changed_spans']:,} of {a['documents']:,}", n3(a["strict_f1"]),
             n3(a["partial_f1"]), n3(a["type_agnostic_f1"]), n3(a["leakage_rate"]),
             n3(a["char_precision"])]
            for s, a in ((s, r["alignment"][s]) for s in SET_NAMES)]
    return "\n".join([("Gold spans -> BIO tags -> spans, scored against the gold spans (the best "
                      "any token tagger can do with this tokeniser):"), "",
                      *table(["set", "gold spans", "spans after round trip",
                              "documents changed", "strict F1", "partial F1",
                              "type-agnostic F1", "leakage", "char precision"], rows)])


def block_cards(r) -> str:
    rows = [[f"`{s}`", c["gold_card_spans"], c["luhn_pass"], pct(c["luhn_pass_rate"]),
             c["length_13_19"], c["length_13_19_and_luhn_pass"]]
            for s, c in ((s, r["cards"][s]) for s in SET_NAMES)]
    out = ["Gold CREDIT_CARD spans, digits only:", ""]
    out += table(["set", "gold card spans", "pass Luhn", "pass rate", "13-19 digits",
                  "13-19 digits and Luhn"], rows)
    out += ["", f"For comparison, `{BASELINE}` leaks this share of gold card characters:", ""]
    out += table(["set", "CREDIT_CARD leakage"],
                 [[f"`{s}`", pct(_view(r["metrics"][(s, BASELINE)])["chars"]["per_gold_label"]
                                 ["CREDIT_CARD"]["leakage_rate"])] for s in SET_NAMES])
    return "\n".join(out)


def block_training(r) -> str:
    grid, lr = r["crf_grid"], r["logreg_grid"]
    out = [(f"CRF grid on `{grid['set']}` (n = {grid['n']}, seed {grid['seed']}); rule: "
           f"{grid['rule']}."), ""]
    rows = [[row["c1"], row["c2"], n3(row["dev"]["type_agnostic_f1"]),
             n3(row["dev"]["leakage_rate"]), n3(row["dev"]["strict_f1"]),
             f"{row['training_seconds']:.0f}", _mb(row["peak_memory_mb"]),
             "**selected**" if (row["c1"], row["c2"]) == (grid["selected"]["c1"],
                                                          grid["selected"]["c2"]) else ""]
            for row in grid["rows"]]
    out += table(["c1", "c2", "dev TA F1", "dev leakage", "dev strict F1", "train s",
                  "peak MB", ""], rows)
    out += ["", (f"LogReg grid on `{lr['set']}` (n = {lr['n']}, seed {lr['seed']}); rule: "
            f"{lr['rule']}."), ""]
    rows = [[row["C"], n3(row["dev"]["type_agnostic_f1"]), n3(row["dev"]["leakage_rate"]),
             n3(row["dev"]["strict_f1"]), row["n_iter"], f"{row['training_seconds']:.0f}",
             _mb(row["peak_memory_mb"]),
             "**selected**" if row["C"] == lr["selected"]["C"] else ""]
            for row in lr["rows"]]
    out += table(["C", "dev TA F1", "dev leakage", "dev strict F1", "iterations", "train s",
                  "peak MB", ""], rows)
    return "\n".join(out)


def block_cost(r) -> str:
    models = r["learning_curve"]["models"]
    rows = [[m["n"], m["seed"], f"{m['training_seconds']:.0f}", _mb(m["peak_memory_mb"]),
             f"{m['eval_seconds']:.0f}"] for m in models]
    lr = r["final_models"]["logreg"]
    rows.append([f"{lr['n']} (LogReg, C={lr['C']})", lr["seed"], f"{lr['training_seconds']:.0f}",
                 _mb(lr["peak_memory_mb"]), "--"])
    out = [("Training cost of every model on this laptop (peak MB = peak working set of the "
           "training process):"), ""]
    out += table(["n (documents)", "seed", "train s", "peak MB", "eval s (3 sets)"], rows)
    big = [m for m in models if m["n"] == max(x["n"] for x in models)]
    t = sum(m["training_seconds"] for m in big) / len(big)
    mem = sum(m["peak_memory_mb"] for m in big) / len(big)
    scale = 20000 / big[0]["n"]
    log = r["training_log"]
    out += ["", (f"**Largest feasible n: {big[0]['n']}.** n = 20000 was not trained: scaling "
            f"the measured n = {big[0]['n']} cost linearly (an extrapolation, NOT a "
            f"measurement) gives about {t * scale / 60:.0f} min and {mem * scale / 1024:.1f} GB "
            f"for one model. Total wall time of all training steps: "
            f"{log['total_seconds'] / 60:.1f} min (`results/m4/training_log.json`).")]
    return "\n".join(out)


def block_curve(r) -> str:
    lc = r["learning_curve"]
    rows = []
    for row in lc["rows"]:
        ta, lk = row["type_agnostic_f1"], row["leakage_rate"]
        rows.append([row["n"], f"`{row['set']}`", row["seeds"],
                     f"{ta['mean']:.3f} [{ta['min']:.3f}, {ta['max']:.3f}]",
                     f"{lk['mean']:.3f} [{lk['min']:.3f}, {lk['max']:.3f}]",
                     n3(row["strict_f1"]["mean"])])
    out = [(f"CRF with c1 = {lc['c1']}, c2 = {lc['c2']} (selected on dev); mean over training "
           "seeds [min, max]. Figure: `docs/figures/m4_learning_curve.png`."), ""]
    out += table(["n", "set", "seeds", "type-agnostic F1", "leakage", "strict F1"], rows)
    fm = r["final_models"]["crf"]
    out += ["", (f"Size used in the benchmark (rule: {fm['rule']}): n = {fm['n']}, seed "
            f"{fm['seed']}.")]
    return "\n".join(out)


def block_rq4(r) -> str:
    rows = []
    for s in SET_NAMES:
        for d in (BASELINE, *M4_DETECTORS):
            m = r["metrics"][(s, d)]
            v = _view(m)
            rows.append([f"`{s}`", f"`{d}`",
                         f"{n3(v['strict']['micro']['f1'])} {_ci(m, 'strict_micro_f1')}",
                         (f"{n3(v['type_agnostic']['micro']['f1'])} "
                         f"{_ci(m, 'type_agnostic_micro_f1')}"),
                         f"{n3(v['chars']['leakage_rate'])} {_ci(m, 'leakage_rate')}",
                         n3(v["chars"]["char_precision"]),
                         f"{m['latency']['median_ms']:.1f}"])
    out = ["All gold spans; 95% bootstrap CI; latency = median ms per document.", ""]
    out += table(["set", "detector", "strict F1", "type-agnostic F1", "leakage",
                  "char precision", "median ms"], rows)
    out += ["", (f"Paired differences against `{BASELINE}` (A - B; same document resamples, "
            "1000, seed 0; negative leakage delta = A leaks less):"), ""]
    rows = []
    for a, b in COMPARISONS[:2]:
        for s in SET_NAMES:
            p = r["paired"][(s, a, b)]
            lk, f = p["leakage_rate"], p["type_agnostic_f1"]
            rows.append([f"`{a}` - `{b}`", f"`{s}`",
                         f"{signed(lk['delta'])} [{lk['ci95'][0]:+.3f}, {lk['ci95'][1]:+.3f}]",
                         "yes" if lk["ci_excludes_zero"] else "no",
                         f"{signed(f['delta'])} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}]",
                         "yes" if f["ci_excludes_zero"] else "no"])
    out += table(["comparison", "set", "leakage delta [CI]", "CI excludes 0",
                  "type-agnostic F1 delta [CI]", "CI excludes 0"], rows)
    return "\n".join(out)


def block_rq6(r) -> str:
    rows = []
    for s in SET_NAMES:
        crf, lr = r["metrics"][(s, "crf")], r["metrics"][(s, "logreg")]
        p = r["paired"][(s, "crf", "logreg")]
        lk, f = p["leakage_rate"], p["type_agnostic_f1"]
        rows.append([f"`{s}`",
                     (f"{n3(_view(crf)['strict']['micro']['f1'])} / "
                     f"{n3(_view(lr)['strict']['micro']['f1'])}"),
                     (f"{n3(_view(crf)['partial']['micro']['f1'])} / "
                     f"{n3(_view(lr)['partial']['micro']['f1'])}"),
                     f"{n3(f['a'])} / {n3(f['b'])}", f"{n3(lk['a'])} / {n3(lk['b'])}",
                     f"{signed(f['delta'])} [{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}]",
                     f"{signed(lk['delta'])} [{lk['ci95'][0]:+.3f}, {lk['ci95'][1]:+.3f}]"])
    fm = r["final_models"]
    out = [(f"Same features; CRF n = {fm['crf']['n']} (c1 = {fm['crf']['c1']}, c2 = "
           f"{fm['crf']['c2']}), LogReg n = {fm['logreg']['n']} (C = {fm['logreg']['C']}); "
           "values CRF / LogReg; paired deltas CRF - LogReg with 95% CI."), ""]
    out += table(["set", "strict F1", "partial F1", "type-agnostic F1", "leakage",
                  "TA F1 delta [CI]", "leakage delta [CI]"], rows)
    return "\n".join(out)


def block_perlabel(r) -> str:
    rows = []
    for e in EntityType:
        cells = []
        for s in SET_NAMES:
            vals = []
            for d in ("crf", BASELINE):
                per = _view(r["metrics"][(s, d)])["chars"]["per_gold_label"].get(e.value)
                vals.append("--" if per is None else pct(per["leakage_rate"]))
            cells.append(" / ".join(vals))
        rows.append([e.value, *cells])
    return "\n".join([f"Leakage rate per gold label, `crf` / `{BASELINE}`:", "",
                      *table(["gold label", *SET_NAMES], rows)])


def block_weights(r) -> str:
    w = r["crf_weights"]
    out = ["Top 20 transition weights (by absolute value):", ""]
    out += table(["from", "to", "weight"],
                 [[f"`{a}`", f"`{b}`", f"{x:+.2f}"] for a, b, x in w["top_transitions"]])
    out += ["", "Top 10 positive state features per entity type (feature, tag, weight):", ""]
    for label, feats in w["top_state_features"].items():
        items = "; ".join(f"`{f}` {tag} {x:+.2f}" for f, tag, x in feats)
        out.append(f"- **{label}**: {items}")
    return "\n".join(out)


def block_observations(r) -> str:
    lines = []
    for s in SET_NAMES:
        p = r["paired"][(s, "crf", BASELINE)]
        q = r["paired"][(s, "crf+regex+presidio", BASELINE)]
        lines.append(
            f"- `{s}`: leakage `crf` {n3(p['leakage_rate']['a'])} vs `{BASELINE}` "
            f"{n3(p['leakage_rate']['b'])} (delta {signed(p['leakage_rate']['delta'])}, CI "
            f"{'excludes' if p['leakage_rate']['ci_excludes_zero'] else 'includes'} 0); "
            f"`crf+regex+presidio` delta {signed(q['leakage_rate']['delta'])} (CI "
            f"{'excludes' if q['leakage_rate']['ci_excludes_zero'] else 'includes'} 0).")
    for s in SET_NAMES:
        p = r["paired"][(s, "crf", "logreg")]
        lines.append(f"- `{s}`: CRF - LogReg type-agnostic F1 {signed(p['type_agnostic_f1']['delta'])}"
                     f", leakage {signed(p['leakage_rate']['delta'])}.")
    rows = [row for row in r["learning_curve"]["rows"] if row["set"] == DEV_SET]
    if len(rows) >= 2:
        lines.append("- Learning curve on dev: type-agnostic F1 "
                     + " -> ".join(f"{row['type_agnostic_f1']['mean']:.3f} (n={row['n']})"
                                   for row in rows) + ".")
    for s in SET_NAMES:
        c = r["cards"][s]
        lines.append(f"- `{s}`: {pct(c['luhn_pass_rate'])} of the gold card numbers pass Luhn.")
    a = r["alignment"]
    lines.append("- Tokenisation ceiling (round trip) strict F1: "
                 + ", ".join(f"{SHORT[s]} {n3(a[s]['strict_f1'])}" for s in SET_NAMES)
                 + "; leakage: " + ", ".join(f"{SHORT[s]} {n3(a[s]['leakage_rate'])}"
                                             for s in SET_NAMES) + ".")
    return "\n".join(lines)


def results_blocks(m4: Path, m3: Path) -> dict[str, str]:
    r = load(m4, m3)
    return {"m4_alignment": block_alignment(r), "m4_cards": block_cards(r),
            "m4_training": block_training(r), "m4_cost": block_cost(r),
            "m4_curve": block_curve(r), "m4_rq4": block_rq4(r), "m4_rq6": block_rq6(r),
            "m4_perlabel": block_perlabel(r), "m4_weights": block_weights(r),
            "m4_observations": block_observations(r)}


def detectors_blocks(m4: Path) -> dict[str, str]:
    rows = []
    for d in ("crf", "logreg"):
        meta = _read(m4 / DEV_SET / d / "metrics.json")["config"]["model_sidecar"]
        versions = ", ".join(f"`{k}` {v}" for k, v in sorted(meta["library_versions"].items()))
        rows.append([f"`{d}`", versions, f"`{meta['model_sha256'][:16]}...`",
                     f"`{meta['training_set_sha256'][:16]}...`", meta["n_docs"],
                     json.dumps(meta["hyperparameters"], sort_keys=True)])
    return {"m4_versions": "\n".join(table(
        ["model", "library versions", "model SHA-256", "training-set SHA-256", "n_docs",
         "hyper-parameters"], rows))}
