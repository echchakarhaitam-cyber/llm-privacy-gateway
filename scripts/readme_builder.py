"""README.md template, filled from committed results (``python scripts/make_results.py --readme``).

Every number in the README comes from results/*.json through the helpers of make_results.py;
the quick-start outputs and test counts come from results/readme_facts.json, written by
scripts/capture_readme_facts.py, which really runs them. Nothing measured is typed here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from make_results import (
    ADAPTED_NOTE,
    LOAD_LABEL,
    NOT_MEASURED,
    RQ11_ENSEMBLE,
    RQ11_LABELS,
    SETS,
    compare_sentence,
    headline,
    leakage,
    load,
    metrics,
    paired,
    rq4_table,
    rq5_text,
    rq7_table,
    rq9_text,
    rq_detection,
    table,
)

REPO = "https://github.com/echchakarhaitam-cyber/llm-privacy-gateway"
CONFIGS = ("none", "regex", "crf", "crf+regex", "crf+regex+presidio")
SECTIONS = (
    "Executive summary", "Problem statement", "Motivation", "Architecture", "Methods",
    "Data and attribution", "Installation", "Quick start", "Reproducing the results", "Results",
    "Testing and validation", "Limitations and what this project does not show", "Future work",
    "References", "Licence and citation")


def trio(root: Path, det: str) -> str:
    ms = [metrics(root, s, det) for s in SETS]
    return " / ".join(f"{leakage(m)[0]:.3f}" for m in ms) if all(ms) else NOT_MEASURED


def masking_row(root: Path, det: str, strategy: str, set_name: str = "nemotron_dev_2k"):
    m = load(root / "m5" / "masking.json") or {"rows": []}
    return next((r for r in m["rows"] if r["detector"] == det and r["strategy"] == strategy
                 and r["set"] == set_name), None)


def latency(root: Path, harness: str, config: str, c: int = 1):
    d = load(root / "m6" / "latency.json")
    if not d:
        return None
    return next((r for r in d["rows"] if r["harness"] == harness and r["config"] == config
                 and r["concurrency"] == c), None)


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def env_cpu(root: Path) -> str:
    d = load(root / "m6" / "latency.json") or load(root / "m6" / "faults.json") or {}
    env = d.get("environment") or {}
    return f"{env.get('cpu', 'unknown CPU')}, {env.get('os', 'unknown OS')}"


def wall_times(root: Path) -> list[list[str]]:
    """Measured wall time of each reproduction step, from the committed logs."""
    def stored(path: str, key: str = "total_seconds"):
        d = load(root / path)
        return d.get(key) if d else None

    lat = load(root / "m6" / "latency.json")
    lat_s = None
    if lat and lat.get("load_per_run"):
        fmt = "%Y-%m-%d %H:%M:%S"
        lat_s = sum((datetime.strptime(r["ended"], fmt) - datetime.strptime(r["started"], fmt))
                    .total_seconds() for r in lat["load_per_run"])
    m7 = load(root / "m7" / "training_log.json")
    rows = [
        ("Download and build the evaluation sets",
         "python scripts/download_data.py && python scripts/build_eval_sets.py", None),
        ("Detection benchmark (regex, spaCy, Presidio, ensembles)",
         "python -m privgate.bench all", stored("m3/run_log.json")),
        ("CRF and logistic regression training",
         "python scripts/train_ml.py {subsets,grid,curve,final}", stored("m4/training_log.json")),
        ("Domain classifier for the utility metric", "python scripts/run_masking.py classifier",
         stored("m5/domain_classifier.json", "seconds")),
        ("Masking grid", "python scripts/run_masking.py grid", stored("m5/masking.json", "seconds")),
        ("Fault injection (RQ9)", "python scripts/run_gateway_faults.py",
         stored("m6/faults.json", "seconds")),
        ("Latency, 3 runs (RQ10)",
         "python scripts/run_gateway_latency.py --protocol recorded-load", lat_s),
        ("Domain-adaptation training (RQ11)", "python scripts/run_domain_adaptation.py train",
         (m7 or {}).get("total_wall_minutes", 0) * 60 or None),
    ]
    return [[name, f"`{cmd}`", minutes(s) if s else NOT_MEASURED] for name, cmd, s in rows]


def quickstart(facts: dict) -> str:
    detect, gateway = facts.get("quickstart_detect"), facts.get("quickstart_gateway")
    if not detect or not gateway:
        return f"{NOT_MEASURED} (run `python scripts/capture_readme_facts.py`)."
    body = json.dumps(gateway["response_json"])          # complete body, one line
    return f"""\
**(a) Detect and mask in Python** (the regex detector needs no model):

```python
{detect['code'].rstrip()}
```

Output, captured by actually running it (`scripts/capture_readme_facts.py`):

```text
{detect['output']}
```

The regex baseline finds the structured identifiers and misses the name: the gap the CRF and
Presidio address (RQ1-RQ6).

**(b) Run the gateway with the mock upstream and send one request** (bash; in PowerShell call
`curl.exe` and put the JSON body in a file, `-d "@request.json"`):

```bash
{gateway['command']}   # terminal 1
{gateway['curl']}   # terminal 2
```

Real response (status line, request-id header, body). The mock upstream echoes the *masked*
prompt it received (`<EMAIL_1>`, `<PHONE_1>`); the gateway restored the originals:

```text
{chr(10).join(gateway['response_headers'])}

{body}
```"""


def tests_line(facts: dict) -> str:
    t = facts.get("tests")
    if not t:
        return NOT_MEASURED
    order = ("default", "data_or_models", "slow")
    return "; ".join(f"`{t[k]['command']}`: {t[k]['summary']}" for k in order if k in t)


def summary(root: Path) -> str:
    m_crf, m_g500, m_ens, m_union = (metrics(root, "gretel_en_test", n) for n in
                                     ("crf", "crf_n5k_g500", RQ11_ENSEMBLE, "crf+regex+presidio"))
    d_dev = paired(root, "nemotron_dev_2k", "crf_n5k_g500", "crf")
    typed, partial, redact = (masking_row(root, "crf+regex+presidio", s)
                              for s in ("typed", "partial", "redact"))
    lat = {c: latency(root, "asgi", c) for c in ("none", "regex", "crf", "crf+regex+presidio")}
    if not all([m_crf, m_g500, m_ens, m_union, d_dev, typed, partial, redact, *lat.values()]):
        return f"- {NOT_MEASURED}: the committed results are incomplete."
    drop = typed["domain_original_classifier"]["drop_vs_unmasked"]["difference"]
    return "\n".join([
        ("- **Off-the-shelf detection leaves much PII uncovered.** The best off-the-shelf "
        f"configuration (`regex+presidio`) leaves {trio(root, 'regex+presidio')} of the gold PII "
        "characters uncovered (dev / Nemotron test / Gretel)."),
        (f"- **A CRF trained on 5,000 synthetic documents** reaches {trio(root, 'crf')}; its union "
        f"with regex and Presidio {trio(root, 'crf+regex+presidio')}. The gain shrinks sharply "
        "out of distribution (Gretel)."),
        "- **Domain adaptation (RQ11):** 500 Gretel training documents move the CRF's Gretel "
        f"leakage from {leakage(m_crf)[0]:.3f} to {leakage(m_g500)[0]:.3f}; on Gretel, "
        + compare_sentence(RQ11_ENSEMBLE, leakage(m_ens)[0], "crf+regex+presidio",
                           leakage(m_union)[0])
        + f". The dev leakage of the adapted CRF changes by {d_dev['leakage_rate']['delta']:+.3f}.",
        ("- **The detector, not the masking strategy, sets privacy:** with `crf+regex+presidio` "
        f"on dev, redact and typed masking leak {redact['privacy']['char_leakage']:.3f} and "
        f"{typed['privacy']['char_leakage']:.3f}, partial masking "
        f"{partial['privacy']['char_leakage']:.3f}; typed masking changes domain-classification "
        f"accuracy by {-drop:+.3f}."),
        f"- **Gateway under injected faults (RQ9).** {rq9_text(root)}",
        f"- **Latency (RQ10)**, {LOAD_LABEL}: median in-process p50 "
        + ", ".join(f"`{c}` {r['p50_ms']:.1f} ms" for c, r in lat.items()) + ".",
    ])


def latency_table(root: Path) -> str:
    rows = []
    for c in CONFIGS:
        a, s = latency(root, "asgi", c), latency(root, "server", c)
        if a and s:
            rows.append([f"`{c}`", f"{a['p50_ms']:.2f}", f"{s['p50_ms']:.2f}",
                         f"{s['p95_ms']:.2f}", f"{s['requests_per_s']:.1f}"])
    if not rows:
        return NOT_MEASURED
    return table(["config", "in-process p50 ms", "server p50 ms", "server p95 ms",
                  "server req/s"], rows)


def rq11_table(root: Path) -> str:
    rows = []
    for n in ("crf", "crf_n5k_g100", "crf_n5k_g500", "crf_g500", RQ11_ENSEMBLE,
              "crf+regex+presidio"):
        ms = [metrics(root, s, n) for s in SETS]
        if all(ms):
            rows.append([f"`{n}`", RQ11_LABELS.get(n, "M4 union, not adapted"),
                         *(f"{leakage(m)[0]:.3f}" for m in ms)])
    if not rows:
        return NOT_MEASURED
    return table(["model", "training data", "dev", "test", "gretel"], rows)


def build_readme(root: Path, facts: dict) -> str:
    ci = f"[![CI]({REPO}/actions/workflows/ci.yml/badge.svg)]({REPO}/actions/workflows/ci.yml)"
    lic = "[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)"
    return f"""\
<!-- Generated by `python scripts/make_results.py --readme` from results/. Do not edit by hand. -->
# llm-privacy-gateway

A fail-closed privacy gateway for LLM requests, with a reproducible benchmark of how much personal
data (PII) regex, spaCy, Presidio and trained CRF detectors leave behind.

{ci} {lic}

## Executive summary

{summary(root)}

Numbers are generated from committed results; all data are synthetic. Details: [docs/results.md](docs/results.md), [docs/README.md](docs/README.md).

## Problem statement

A text $x$ carries gold spans $G(x)=\\{{(s_i,e_i,t_i)\\}}$: half-open character intervals with an
entity type. A detector returns spans $D(x)$; a masking function $M$ maps $(x, D(x))$ to a masked
text $\\tilde x$ and a mapping $\\pi$ (placeholder to original) that restores the model's answer.

- **Character leakage** $L=\\lvert C_G\\setminus C_D\\rvert / \\lvert C_G\\rvert$, where $C_G$ and
  $C_D$ are the characters covered by gold and by detected spans, pooled over a corpus, with 95 %
  document-bootstrap intervals. For redact, typed and pseudonym masking the leaked characters are
  exactly the undetected ones; partial masking also keeps some characters by design.
- **Utility**: the share of non-PII characters preserved, the length ratio, and the accuracy
  change of a domain classifier on masked versus original text.
- **Fail-closed invariant**: for every request $r$ and every failure in the screening path
  (detector exception, timeout, malformed output, mapping error), the upstream receives nothing;
  it only ever receives $M(r, D(r))$, after the masked text was verified to unmask back to $r$.

## Motivation

Text sent to an externally hosted language model leaves its owner's control. A gateway can remove
personal data first, but only if its detectors are measured rather than assumed and a screening
failure can never degrade into sending raw text. The project was motivated by hands-on internship
work on privacy-preserving LLM platforms; it is an independent research project built only from
public datasets, open-source libraries and textbook methods.

## Architecture

```mermaid
flowchart LR
  N[Nemotron-PII] & G[Gretel finance] --> L[loaders, label mapping] --> S[hash-verified sets]
  R[regex + validators] & P[spaCy / Presidio] & C[CRF] --> E[union ensembles]
  S & E --> B[benchmark: bootstrap CIs, paired tests]
  E --> GW[gateway: validate, screen, mask, verify] --> U[upstream: local model or mock]
  U --> UM[unmask answer]
  B --> X[experiments: masking, faults, latency, adaptation] --> T[generated tables, README]
```

| module | role |
|---|---|
| `privgate.spans`, `privgate.evaluation` | span model; strict, partial and type-agnostic matching; character leakage |
| `privgate.data` | pinned loaders, label mapping, normalisation, deterministic splits |
| `privgate.detectors` | regex with checksum validators, spaCy, Presidio, union ensemble |
| `privgate.ml` | tokenisation, BIO tags, CRF and per-token logistic regression, training-pool guard |
| `privgate.masking` | redact, typed, pseudonym and partial masking; collision-safe reversible mapping |
| `privgate.bench` | benchmark runner, bootstrap, privacy and utility metrics, error taxonomy |
| `privgate.gateway` | FastAPI gateway: request whitelist, policy, fail-closed screening, audit log, upstreams |
| `scripts/` | downloads, training, experiments and every table generator |

## Methods

- **Regex baseline** with validators (Luhn for card numbers, ISO 13616 mod-97 for IBANs),
  **spaCy** `en_core_web_lg` NER and **Presidio** (score threshold chosen on dev only). **Union
  ensembles** resolve overlaps deterministically (longer span first, then higher score).
- **Linear-chain CRF** (CRFsuite, L-BFGS, L1 + L2 regularisation) on hand-crafted token features,
  trained on the Nemotron train split with dev-only model selection
  ([docs/theory/02-crf.md](docs/theory/02-crf.md)).
- **Masking**: redact, typed placeholders, Faker pseudonyms and partial masking; placeholders are
  tagged when the input already contains placeholder-like text, and every mapping is verified to
  round-trip ([docs/theory/03-masking.md](docs/theory/03-masking.md)).
- **Gateway**: strict request whitelist, per-type mask or block policy, detection in a worker
  thread under a time budget, one mapping per request, a single upstream call site, and an audit
  log that records counts and error classes only ([docs/GATEWAY.md](docs/GATEWAY.md)).

## Data and attribution

- **Nemotron-PII** by NVIDIA Corporation, <https://huggingface.co/datasets/nvidia/Nemotron-PII>,
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made: labels
  mapped onto 12 types, adjacent same-label spans merged, spans verified and trimmed, documents
  split into train, dev and test sets.
- **Gretel synthetic PII finance (multilingual)**, English files, by Gretel.ai,
  <https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual>, licensed under
  [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Both datasets are synthetic. **No dataset file and no trained model is redistributed**: the
  pinned files are downloaded and hash-checked locally ([docs/DATA.md](docs/DATA.md)).

## Installation

Python 3.10 to 3.12. Extras: `dev` (pytest, ruff), `data` (pyarrow), `detectors` (spaCy,
Presidio), `ml` (CRFsuite, scikit-learn), `masking` (Faker), `gateway` (FastAPI, uvicorn, httpx,
psutil), `plots` (matplotlib).

```powershell
# Windows PowerShell
python -m venv .venv; .venv\\Scripts\\Activate.ps1
pip install -e ".[dev,data,detectors,ml,masking,gateway,plots]"
python scripts/install_model.py   # optional: spaCy en_core_web_lg, hash-checked
```

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,detectors,ml,masking,gateway,plots]"
python scripts/install_model.py   # optional: spaCy en_core_web_lg, hash-checked
```

## Quick start

{quickstart(facts)}

## Reproducing the results

Wall times measured on one laptop ({env_cpu(root)}); steps without a stored log are marked.

{table(["step", "command", "wall time"], wall_times(root))}

Then `run_domain_adaptation.py evaluate`, `error_taxonomy.py` and the `make_*_tables.py` / `make_results.py --readme` generators.

## Results

{headline(root)}

![Leakage by set](docs/figures/leakage_by_set.png)

**RQ1-RQ3, off-the-shelf detection** (dev; [results-detection.md](docs/results-detection.md)):

{rq_detection(root)}

**RQ4-RQ6, a trained CRF.** Paired leakage differences with 95 % CIs; learning curve on dev:
{rq5_text(root)}.

{rq4_table(root)}

**RQ7-RQ8, masking** ([results-masking.md](docs/results-masking.md)):

{rq7_table(root)}

**RQ9-RQ10, gateway** ([results-gateway.md](docs/results-gateway.md)). {rq9_text(root)}
Latency, medians of 3 runs, {LOAD_LABEL}:

{latency_table(root)}

**RQ11, domain adaptation**, character leakage ([results-adaptation.md](docs/results-adaptation.md)).
{ADAPTED_NOTE}

{rq11_table(root)}

## Testing and validation

{tests_line(facts)}. The default suite needs no model and no data; CI runs it on Ubuntu and
Windows with Python 3.10 to 3.12. Each fail-closed rule has a unit test and an HTTP test; a canary
test sends 50 requests through success and every failure path and scans the audit file and every
log record; a weak-reference test shows that the mapping does not outlive its request. Evaluation
sets are refused unless their SHA-256 matches `data/manifest.json`, and the training-pool guard
refuses every evaluation and Gretel-test document. What the best configuration still leaks:
[docs/error-analysis.md](docs/error-analysis.md).

## Limitations and what this project does not show

- All data are synthetic; nothing is measured on real documents or in languages other than English.
- A substantial share of gold PII still leaks on every set, dominated by organisation names and
  occupations; the gateway is never better than its detector.
- Utility is a bag-of-words classifier and text preservation, not the quality of an LLM's answer.
- Latency is a single-laptop, mock-upstream measurement under background load.
- No authentication, TLS or rate limiting; the operator machine is trusted; images, files and
  tools are refused, not screened. Full list: [docs/results.md](docs/results.md).

## Future work

Organisation and occupation detection; multilingual data; evaluation on real, consented documents;
transformer-based NER; utility measured as LLM answer quality; authentication, TLS and rate
limiting for the gateway.

## References

1. J. Lafferty, A. McCallum, F. Pereira. *Conditional Random Fields: Probabilistic Models for
   Segmenting and Labeling Sequence Data.* Proc. ICML 2001 (page numbers: to verify).
2. N. Okazaki. *CRFsuite: a fast implementation of Conditional Random Fields*, 2007,
   <https://www.chokkan.org/software/crfsuite/> (to verify).
3. Microsoft Presidio documentation, <https://microsoft.github.io/presidio/>.
4. spaCy documentation, <https://spacy.io/>; model `en_core_web_lg` 3.8.0.
5. Dataset cards: <https://huggingface.co/datasets/nvidia/Nemotron-PII>,
   <https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual>.
6. ISO/IEC 7812-1 (Luhn check digit) and ISO 13616 (IBAN, mod-97) (editions: to verify).

## Licence and citation

Code and documentation: [MIT](LICENSE), (c) 2026 Haitam Echchakar. The datasets keep their own
licences. Cite with [CITATION.cff](CITATION.cff).
"""
