"""Render docs/results.md (consolidated) and docs/results-adaptation.md (RQ11) from results/*.

Usage:  python scripts/make_results.py

Every number comes from committed JSON under results/ (m3 detection, m4 CRF, m5 masking, m6
gateway, m7 adaptation and error taxonomy). The prose states no measured value except through
f-strings filled from those files. A missing file renders its part as "NOT MEASURED". The
figure docs/figures/leakage_by_set.png is drawn from the same numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SETS = ("nemotron_dev_2k", "nemotron_test_5k", "gretel_en_test")
SHORT = {"nemotron_dev_2k": "dev", "nemotron_test_5k": "test", "gretel_en_test": "gretel"}
HEADLINE = ("regex", "presidio", "regex+presidio", "crf", "crf+regex+presidio")
M3 = ("regex", "presidio", "regex+presidio", "spacy", "regex+spacy")
RQ11_MODELS = ("crf_g500", "crf_n5k_g100", "crf_n5k_g500")
RQ11_LABELS = {
    "crf": "(a) Nemotron 5000", "crf_n5k_g100": "(b) Nemotron 5000 + Gretel 100",
    "crf_n5k_g500": "(b) Nemotron 5000 + Gretel 500", "crf_g500": "(c) Gretel 500 only"}
NOT_MEASURED = "NOT MEASURED"
LOAD_LABEL = "measured under recorded background load (not an idle machine)"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_dir(root: Path, set_name: str, detector: str) -> Path:
    """Where the stored run of ``detector`` lives (m7 for the RQ11 models, m4 for CRFs)."""
    if detector in RQ11_MODELS:
        return root / "m7" / set_name / detector
    if detector in ("crf", "crf+regex+presidio", "logreg"):
        return root / "m4" / set_name / detector
    return root / "m3" / set_name / detector


def metrics(root: Path, set_name: str, detector: str):
    return load(run_dir(root, set_name, detector) / "metrics.json")


def leakage(m: dict) -> tuple[float, float, float]:
    v = m["views"]["all"]
    lo, hi = v["ci95"]["leakage_rate"]
    return v["chars"]["leakage_rate"], lo, hi


def f1(m: dict) -> tuple[float, float, float]:
    v = m["views"]["all"]
    lo, hi = v["ci95"]["type_agnostic_micro_f1"]
    return v["type_agnostic"]["micro"]["f1"], lo, hi


def ci(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def delta(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def paired(root: Path, set_name: str, a: str, b: str):
    for base in ("m7", "m4", "m3"):
        data = load(root / base / set_name / f"paired_{a}__vs__{b}.json")
        if data:
            return data
    return None


def best_rq11(root: Path) -> str | None:
    """The RQ11 model with the lowest Gretel leakage (descriptive, chosen on the test set)."""
    found = [(leakage(m)[0], name) for name in RQ11_MODELS
             if (m := metrics(root, "gretel_en_test", name))]
    return min(found)[1] if found else None


# --- headline ---------------------------------------------------------------------------------

def headline_models(root: Path) -> list[str]:
    best = best_rq11(root)
    return [*HEADLINE, *([best] if best else [])]


def headline(root: Path) -> str:
    rows = []
    for det in headline_models(root):
        cells = []
        for s in SETS:
            m = metrics(root, s, det)
            cells.append(ci(leakage(m)) if m else NOT_MEASURED)
        label = f"`{det}`" if det in HEADLINE else f"`{det}` — {RQ11_LABELS[det]}"
        rows.append([label, *cells])
    note = ("Character leakage = share of gold PII characters left uncovered by the detected "
            "spans (lower is better), with 95 % document-bootstrap intervals. ")
    best = best_rq11(root)
    if best:
        note += (f"The last row is the RQ11 model with the lowest Gretel leakage (`{best}`); "
                 "that choice is descriptive and was made on the Gretel test set itself.")
    return note + "\n\n" + table(["detector", *(f"{SHORT[s]} (`{s}`)" for s in SETS)], rows)


def figure(root: Path, out: Path) -> bool:
    """Grouped bars with CI error bars, one group per set. False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    models = headline_models(root)
    palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(models)
    for i, det in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for g, s in enumerate(SETS):
            m = metrics(root, s, det)
            if not m:
                continue
            v, a, b = leakage(m)
            xs.append(g + (i - (len(models) - 1) / 2) * width)
            ys.append(v)
            lo.append(v - a)
            hi.append(b - v)
        ax.bar(xs, ys, width * 0.92, yerr=[lo, hi], capsize=2, color=palette[i % len(palette)],
               label=det, error_kw={"elinewidth": 1})
    ax.set_xticks(range(len(SETS)), [f"{SHORT[s]}\n{s}" for s in SETS])
    ax.set_ylabel("character leakage (lower is better)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    ax.set_title("Share of gold PII characters left uncovered, by detector and set")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# --- RQ11 -------------------------------------------------------------------------------------

def rq11_tables(root: Path) -> tuple[str, list[str]]:
    sets_info = load(root / "m7" / "training_sets.json")
    log = load(root / "m7" / "training_log.json")
    present = [n for n in ("crf", *RQ11_MODELS) if all(metrics(root, s, n) for s in SETS)]
    if sets_info is None or len(present) < 2:
        return f"{NOT_MEASURED} (results/m7 is incomplete).\n", []
    parts = [
        (f"Gretel train file: {sets_info['gretel_train_documents']:,} documents, "
        f"SHA-256 `{sets_info['gretel_train_sha256'][:12]}…`. **Documents whose text also occurs "
        f"in the Gretel test file: {sets_info['overlap_with_gretel_test']}** "
        f"({sets_info['overlap_action']}); {sets_info['usable_documents']:,} usable. The Gretel "
        f"documents are a seeded sample (seed {sets_info['seed']}); the 100-document sample is "
        "the first 100 of the 500-document one. Every CRF uses the M4 hyper-parameters (c1 = "
        "0.1, c2 = 0.01) and features; nothing was tuned in this RQ."),
    ]
    header = ["model", *(f"leakage {SHORT[s]}" for s in SETS), *(f"F1 {SHORT[s]}" for s in SETS)]
    rows = [[f"`{n}` {RQ11_LABELS[n]}",
             *(ci(leakage(metrics(root, s, n))) for s in SETS),
             *(ci(f1(metrics(root, s, n))) for s in SETS)] for n in present]
    parts.append("**Leakage and type-agnostic F1 per model and set** (95 % bootstrap CIs)\n\n"
                 + table(header, rows))
    header = ["model vs (a)", *(f"Δ leakage {SHORT[s]}" for s in SETS),
              *(f"Δ F1 {SHORT[s]}" for s in SETS)]
    rows, notes = [], []
    for n in present[1:]:
        deltas = {s: paired(root, s, n, "crf") for s in SETS}
        row = [f"`{n}`"]
        row += [delta(deltas[s]["leakage_rate"]) if deltas[s] else NOT_MEASURED for s in SETS]
        row += [delta(deltas[s]["type_agnostic_f1"]) if deltas[s] else NOT_MEASURED
                for s in SETS]
        rows.append(row)
        for s in SETS:
            d = deltas[s]
            if not d:
                continue
            lk = d["leakage_rate"]
            reduction = -lk["delta"] / lk["b"] if lk["b"] else 0.0
            verdict = ("lower" if lk["ci95"][1] < 0 else "higher" if lk["ci95"][0] > 0
                       else "not distinguishable")
            notes.append(f"`{n}` on `{s}`: leakage {lk['b']:.3f} → {lk['a']:.3f} (relative "
                         f"reduction {reduction:.1%}; the paired CI says {verdict}); "
                         f"type-agnostic F1 {d['type_agnostic_f1']['delta']:+.3f}.")
    parts.append("**Paired differences against (a)** (new model minus (a); a negative leakage "
                 "difference is an improvement; paired bootstrap, same document resamples)\n\n"
                 + table(header, rows))
    if log:
        rows = [[f"`{n}`", m["documents"], f"{m['wall_seconds'] / 60:.1f}",
                 f"{m['training_seconds'] / 60:.1f}", f"{m['peak_memory_mb']:.0f}"]
                for n, m in log["models"].items()]
        text = ("**Training cost** (this laptop)\n\n"
                + table(["model", "documents", "wall min", "CRF training min", "peak MB"], rows))
        if log.get("skipped"):
            text += "\n\nSkipped: " + "; ".join(f"`{k}` ({v})" for k, v in log["skipped"].items())
        parts.append(text)
    return "\n\n".join(parts) + "\n", notes


# --- other RQs --------------------------------------------------------------------------------

def rq_detection(root: Path) -> str:
    rows = []
    for det in M3:
        m = metrics(root, "nemotron_dev_2k", det)
        if m:
            rows.append([f"`{det}`", ci(f1(m)), ci(leakage(m))])
    return table(["detector (dev)", "type-agnostic F1", "leakage"], rows) if rows else NOT_MEASURED


def rq3_text(root: Path) -> str:
    m = metrics(root, "nemotron_dev_2k", "regex+presidio")
    if not m:
        return NOT_MEASURED
    per = m["views"]["all"]["chars"]["per_gold_label"]
    total = sum(v["leaked_chars"] for v in per.values())
    top = sorted(per.items(), key=lambda kv: -kv[1]["leaked_chars"])[:4]
    return "; ".join(f"{k} {v['leaked_chars'] / total:.1%} of leaked characters "
                     f"(type leakage {v['leakage_rate']:.3f})" for k, v in top)


def rq4_table(root: Path) -> str:
    rows = []
    for a, b, label in (("crf", "regex+presidio", "`crf` − `regex+presidio` (RQ4)"),
                        ("crf+regex+presidio", "regex+presidio",
                         "`crf+regex+presidio` − `regex+presidio` (RQ4)"),
                        ("crf", "logreg", "`crf` − `logreg` (RQ6)")):
        cells = []
        for s in SETS:
            d = paired(root, s, a, b)
            cells.append(delta(d["leakage_rate"]) if d else NOT_MEASURED)
        rows.append([label, *cells])
    return table(["paired leakage difference", *(SHORT[s] for s in SETS)], rows)


def rq5_text(root: Path) -> str:
    curve = load(root / "m4" / "learning_curve.json")
    if not curve:
        return NOT_MEASURED
    rows = [r for r in curve["rows"] if r["set"] == "nemotron_dev_2k"]
    return "; ".join(f"n = {r['n']}: F1 {r['type_agnostic_f1']['mean']:.3f}, leakage "
                     f"{r['leakage_rate']['mean']:.3f}" for r in sorted(rows, key=lambda r: r["n"]))


def rq7_table(root: Path) -> str:
    m = load(root / "m5" / "masking.json")
    if not m:
        return NOT_MEASURED
    rows = []
    for r in m["rows"]:
        if r["set"] == "nemotron_dev_2k" and r["detector"] == "crf+regex+presidio":
            p, drop = r["privacy"], r["domain_original_classifier"]["drop_vs_unmasked"]
            rows.append([r["strategy"], ci((p["char_leakage"], *p["ci95"]["char_leakage"])),
                         f"{r['utility']['preserved_non_pii']:.3f}",
                         (f"{drop['difference']:+.3f} [{drop['ci95'][0]:+.3f}, "
                         f"{drop['ci95'][1]:+.3f}]")])
    return table(["strategy (`crf+regex+presidio`, dev)", "char leakage",
                  "preserved non-PII", "domain-accuracy drop"], rows)


def rq9_text(root: Path) -> str:
    d = load(root / "m6" / "faults.json")
    if not d:
        return NOT_MEASURED
    closed = [r for r in d["rows"] if r["mode"] == "fail_closed"]
    opened = [r for r in d["rows"] if r["mode"] == "fail_open"]
    worst = max(opened, key=lambda r: r["fault_leaked_requests"])
    detector_faults = [r["availability"] for r in closed
                       if r["q"] == max(d["qs"]) and r["fault"] in ("exception", "hang",
                                                                    "malformed")]
    return (f"Fail-closed: {sum(r['unscreened_forwarded'] for r in closed)} unscreened requests "
            f"forwarded and {sum(r['fault_leaked_requests'] for r in closed)} fault-induced "
            f"leaks over {sum(r['faults_injected'] for r in closed)} injected faults. "
            f"Fail-open: the worst condition ({worst['fault']}, q = {worst['q']:g}) leaked "
            f"{worst['fault_leaked_requests']} of {worst['requests']} requests. Fail-closed "
            f"availability at q = {max(d['qs']):g} with a detector fault was at least "
            f"{min(detector_faults):.3f}.")


def rq10_table(root: Path) -> str:
    d = load(root / "m6" / "latency.json")
    if not d or d.get("idle_check_before", {}).get("skipped"):
        return NOT_MEASURED
    label = "Idle protocol" if d.get("protocol") == "idle" else LOAD_LABEL
    rows = []
    for harness in ("asgi", "server"):
        sel = [r for r in d["rows"] if r["harness"] == harness and r["concurrency"] == 1]
        for r in sorted(sel, key=lambda r: r["p50_ms"]):
            rows.append([harness, f"`{r['config']}`", f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}",
                         f"{r['requests_per_s']:.1f}"])
    recorded = d.get("protocol") == "recorded-load"
    suffix = f"; median of {d['repeats']} runs" if recorded else ""
    return (f"*{label}{suffix}.*\n\n"
            + table(["harness (c = 1)", "config", "p50 ms", "p95 ms", "req/s"], rows))


def taxonomy_text(root: Path) -> str:
    d = load(root / "m7" / "error_taxonomy.json")
    if not d:
        return NOT_MEASURED
    lines = []
    for s in SETS:
        r = d["sets"][s]
        top, raw = r["dimensions"]["mapped_label"][0], r["dimensions"]["raw_label"][0]
        pos = r["dimensions"]["position"][0]
        lines.append(f"- `{s}`: {r['leaked_spans']} of {r['gold_spans']} gold spans leak "
                     f"({r['leaked_share_of_gold_spans']:.3f}); largest mapped label "
                     f"{top['category']} ({top['share']:.1%}), largest raw label "
                     f"`{raw['category']}` ({raw['share']:.1%}), largest position "
                     f"{pos['category']} ({pos['share']:.1%}).")
    return "\n".join(lines)


# --- documents --------------------------------------------------------------------------------

THREATS = """\
- **Synthetic data only.** Nemotron-PII and the Gretel finance set are generated. Their text is
  more regular than real documents, and their labels come from the generator or an NER+LLM
  pass; a detector's "false positive" can be real, unlabelled PII.
- **One generator for train and two of three evaluation sets.** Nemotron train, dev and test
  share templates and label set; in-domain gains partly measure how well a model learned that
  generator. Gretel is the only independent set, and it is itself a single generator.
- **Label mapping and heterogeneous OTHER.** Raw labels are mapped onto 12 types by hand.
  Strict and partial scores depend on the mapping; character leakage and type-agnostic scores
  do not. OTHER mixes very different attributes.
- **Detectors.** Presidio and spaCy ran with library defaults (one Presidio threshold chosen on
  dev); the CRF and logistic regression were trained on at most 5,000 documents with
  hyper-parameters chosen on n = 1,000; the union ensembles maximise coverage, not precision.
- **Masking and utility.** Utility is the accuracy of one bag-of-words domain classifier on one
  dataset; it says nothing about the quality of an LLM's answer to a masked prompt.
- **Gateway experiments.** MockUpstream, a replayed detector (checked on 25 documents), four
  injected fault types applied independently one request at a time, and single-message
  synthetic requests. The leak metric is verbatim presence of gold values of at least four
  characters.
- **Latency.** One Windows laptop, one process, one detector worker, a mock upstream with 0 ms
  latency, configurations run in a fixed order. The idle criterion was never met, so the
  numbers are medians of three runs measured under recorded background load
  (results-gateway.md).
- **Domain adaptation (RQ11).** One seed, two sample sizes, one set of hyper-parameters; the
  Gretel train and test files come from the same generator and may share templates even when no
  text is identical.
- **Error taxonomy.** Categories are computed by simple rules (format, line position); they
  describe leaked spans, they do not explain why the model missed them.
- **Selection on evaluation data.** The "best RQ11 model" in the headline table is the lowest
  Gretel leakage among three models, chosen on the Gretel test set; it is a description, not a
  validated choice.
"""

NOT_SHOWN = """\
- Performance on **real-world data**: every document is synthetic.
- Detection of PII in **languages other than English**, in **images, audio or files**, or in
  code and structured logs beyond what the synthetic documents contain.
- That any configuration is **safe to use with real personal data**: a substantial share of gold
  PII characters still leaks on every set, and the gateway cannot do better than its detector.
- The effect of masking on the **quality of an LLM's answer**: no model was queried; utility is
  measured by a bag-of-words domain classifier and by text preservation.
- **Re-identification risk** from quasi-identifiers or context that is not labelled as PII.
- **End-to-end latency** with a real model, or latency on other hardware; the numbers are a
  mock-upstream, single-laptop measurement under background load.
- Robustness of the gateway to a **detector that hangs forever**, to load beyond concurrency 4,
  to malicious clients (no authentication, TLS or rate limiting), or to natural (not injected)
  faults.
- That a **small amount of target-domain data** generalises beyond the one synthetic target
  domain, one seed and two sample sizes tested.
"""


def build_results(root: Path) -> str:
    notes = rq11_tables(root)[1]
    best = best_rq11(root)
    crf = [metrics(root, s, "crf") for s in SETS]
    union = [metrics(root, s, "crf+regex+presidio") for s in SETS]
    base = [metrics(root, s, "regex+presidio") for s in SETS]
    faults = rq9_text(root)
    gretel_best = metrics(root, "gretel_en_test", best) if best else None
    if all(crf) and all(union) and all(base) and gretel_best:
        def trio(ms):
            return ", ".join(f"{leakage(m)[0]:.3f}" for m in ms)

        summary = (
            "On three synthetic evaluation sets (dev, Nemotron test, Gretel), the best "
            f"off-the-shelf configuration (`regex+presidio`) leaves {trio(base)} of the gold PII "
            f"characters uncovered. A CRF trained on 5,000 Nemotron documents gives {trio(crf)}, "
            f"and the union `crf+regex+presidio` {trio(union)}: the gain is large in-domain "
            "and smaller out of distribution. Adding Gretel training documents moves the Gretel "
            f"leakage of the CRF from {leakage(crf[2])[0]:.3f} to {leakage(gretel_best)[0]:.3f} "
            f"(`{best}`). Under every masking strategy the privacy of the output is set by the "
            f"detector, not by the strategy (RQ7). {faults}")
    else:
        summary = NOT_MEASURED
    items = "\n".join("- " + n for n in notes) if notes else NOT_MEASURED
    return f"""\
# Results (consolidated)

Generated by `scripts/make_results.py` from the committed JSON under `results/`. Detailed
files: [results-detection.md](results-detection.md) (RQ1-RQ6),
[results-masking.md](results-masking.md) (RQ7-RQ8), [results-gateway.md](results-gateway.md)
(RQ9-RQ10), [results-adaptation.md](results-adaptation.md) (RQ11),
[error-analysis.md](error-analysis.md) and [error-examples.md](error-examples.md); the
gateway is described in [GATEWAY.md](GATEWAY.md), the data in [DATA.md](DATA.md).

## Summary

{summary}

## Headline: leakage by detector and set

{headline(root)}

![Leakage by set](figures/leakage_by_set.png)

## Research questions

### RQ1-RQ3: off-the-shelf detection
How well do a rule-based baseline and off-the-shelf NER/PII tools detect PII; how does this
change across dev, Nemotron test and Gretel; which types leak? Details:
[results-detection.md](results-detection.md).

{rq_detection(root)}

Largest contributors to the leaked characters of `regex+presidio` on dev (RQ3): {rq3_text(root)}.

### RQ4-RQ6: a trained sequence model
Does a CRF trained on Nemotron reduce leakage (RQ4), how does it scale with data (RQ5, dev:
{rq5_text(root)}), and what do transitions add (RQ6)? Paired differences:

{rq4_table(root)}

### RQ7-RQ8: masking strategies and the cost of detector errors
Privacy and utility of every detector x strategy, and the gap to oracle masking. Details:
[results-masking.md](results-masking.md).

{rq7_table(root)}

### RQ9: fail-closed under injected faults
{faults}

### RQ10: gateway latency
{rq10_table(root)}

Protocol and load records: [results-gateway.md](results-gateway.md).

### RQ11: domain adaptation
Does a small amount of target-domain data fix the Gretel leakage? Full tables:
[results-adaptation.md](results-adaptation.md).

{items}

### Final error analysis
{taxonomy_text(root)}

Tables: [error-analysis.md](error-analysis.md); examples: [error-examples.md](error-examples.md).

## Threats to validity

{THREATS}
## What this project does NOT show

{NOT_SHOWN}"""


def build_adaptation(root: Path) -> str:
    body, notes = rq11_tables(root)
    items = "\n".join("- " + n for n in notes) if notes else NOT_MEASURED
    return f"""\
# Domain adaptation (RQ11)

Generated by `scripts/make_results.py` from `results/m7/`. **RQ11**: the CRF trained on
Nemotron leaks a large share of the Gretel gold characters. Does adding a SMALL amount of
target-domain (Gretel train) data fix most of this, and does it hurt the Nemotron sets
(negative transfer)? Protocol fixed before any model was trained: models (a)-(c) as below, no
hyper-parameter tuning, all models scored on all three sets with the M3 benchmark, paired
bootstrap against (a). Reproduce: `python scripts/download_data.py`, then
`python scripts/run_domain_adaptation.py prepare | train | evaluate`.

{body}
## Observations

Generated from the tables above.

{items}
"""


def main() -> int:
    root, docs = Path("results"), Path("docs")
    ok = figure(root, docs / "figures" / "leakage_by_set.png")
    print("figure:", "written" if ok else "skipped (matplotlib missing)")
    for name, text in (("results.md", build_results(root)),
                       ("results-adaptation.md", build_adaptation(root))):
        (docs / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote docs/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
