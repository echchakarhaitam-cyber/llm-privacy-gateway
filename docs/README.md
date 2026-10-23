# Documentation index

Start with the project [README](../README.md) and the consolidated [results](results.md).

## Results

| document | content |
|---|---|
| [results.md](results.md) | consolidated results: summary, headline leakage table, RQ1-RQ11, threats to validity, what the project does not show |
| [results-detection.md](results-detection.md) | RQ1-RQ3 (off-the-shelf detectors) and RQ4-RQ6 (CRF, learning curve, transitions) |
| [results-masking.md](results-masking.md) | RQ7-RQ8: privacy and utility of every detector x masking strategy, gap to oracle masking |
| [results-gateway.md](results-gateway.md) | RQ9 fault injection (fail-closed vs fail-open) and RQ10 latency, with the measurement protocol |
| [results-adaptation.md](results-adaptation.md) | RQ11: domain adaptation with a small amount of target-domain data |
| [error-analysis.md](error-analysis.md) | automatic taxonomy of the spans the best configuration still leaks |
| [error-examples.md](error-examples.md) | seeded, attributed excerpts for the top error categories |

## Methods and system

| document | content |
|---|---|
| [theory/01-detection-metrics.md](theory/01-detection-metrics.md) | span matching modes, character leakage, bootstrap intervals, checksum validators |
| [theory/02-crf.md](theory/02-crf.md) | linear-chain CRF: model, training, features, decoding |
| [theory/03-masking.md](theory/03-masking.md) | masking strategies, reversible mapping, collision safety |
| [DETECTORS.md](DETECTORS.md) | detector configurations, label mappings, versions |
| [GATEWAY.md](GATEWAY.md) | the gateway: API, policy, fail-closed rules, audit log, sequence diagram, threat model |
| [DATA.md](DATA.md) | datasets, licences and attribution, schemas, label mapping, splits, statistics |

## Figures

- [figures/leakage_by_set.png](figures/leakage_by_set.png): leakage by detector and set (headline).
- [figures/m4_learning_curve.png](figures/m4_learning_curve.png): CRF learning curve (RQ5).
- [figures/m5_privacy_utility.png](figures/m5_privacy_utility.png): privacy-utility trade-off (RQ7).

## Error samples of earlier milestones

`results/m3/errors_<detector>.md` and `results/m4/errors_crf.md`: 15 false negatives and 15 false
positives per detector on `nemotron_dev_2k` (synthetic, attributed excerpts).

## Project files

[LICENSE](../LICENSE) (MIT, code and documentation only), [CITATION.cff](../CITATION.cff),
[AGENTS.md](../AGENTS.md) (rules for contributors and coding agents),
[CI workflow](../.github/workflows/ci.yml).
