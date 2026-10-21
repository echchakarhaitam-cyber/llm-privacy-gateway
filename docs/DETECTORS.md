# Detectors

Every detector implements `privgate.detectors.Detector`: `detect(text)` returns
non-overlapping `Span`s sorted by start (overlaps resolved with `resolve_overlaps`), and
`supported_labels` lists the privgate types it can emit. spaCy and Presidio are optional
(`pip install -e ".[detectors]"`) and imported lazily: the core library imports without them.

## Installed versions

Exact pins in `pyproject.toml` (extra `detectors`): `spacy==3.8.16`,
`presidio-analyzer==2.2.364`. The spaCy model is not on PyPI; `scripts/install_model.py`
downloads the official GitHub release wheel, records its SHA-256 under `"models"` in
`data/manifest.json` and installs it. The model declares compatibility with spaCy
`>=3.8.0,<3.9.0`. Versions below are read from the benchmark's recorded environment.

<!-- versions:begin -->
| package | installed version (as recorded in metrics.json) |
|---|---|
| `en-core-web-lg` | `3.8.0` |
| `presidio-analyzer` | `2.2.364` |
| `privgate` | `0.1.0` |
| `pyarrow` | `25.0.1` |
| `spacy` | `3.8.16` |

spaCy model: `en_core_web_lg` 3.8.0 (MIT), wheel `en_core_web_lg-3.8.0-py3-none-any.whl` from https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl, 400,658,291 bytes, SHA-256 `293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63`.

Recorded on: AMD Ryzen 5 7540U w/ Radeon 740M Graphics, Windows-10-10.0.26200-SP0, Python 3.10.11.
<!-- versions:end -->

## Label maps

Generated from the code. Mapping choices follow the gold mapping of docs/DATA.md so that a
correct detection is not counted as a type error:

- spaCy `TIME` and `NORP` and Presidio `NRP` map to OTHER, because the gold time-of-day and
  ethnic/religious/political attributes are OTHER.
- Presidio identifier types issued to a person or account map to ID_NUMBER.
- Presidio `MAC_ADDRESS` and `CRYPTO` map to OTHER (technical identifiers are OTHER in the
  gold mapping).
- Presidio `DATE_TIME` maps to DATE.
- An entity type missing from the Presidio map is emitted as OTHER and logged once.

<!-- maps:begin -->
**spaCy** (`SpacyDetector`, name `spacy`): pipes excluded at load: `tok2vec`, `tagger`, `parser`, `attribute_ruler`, `lemmatizer`, `senter`.

| spaCy label | privgate type |
|---|---|
| `PERSON` | PERSON |
| `ORG` | ORGANIZATION |
| `GPE` | LOCATION |
| `LOC` | LOCATION |
| `FAC` | LOCATION |
| `DATE` | DATE |
| `TIME` | OTHER |
| `NORP` | OTHER |

All other spaCy labels (CARDINAL, EVENT, LANGUAGE, LAW, MONEY, ORDINAL, PERCENT, PRODUCT, QUANTITY, WORK_OF_ART) are ignored.

**Presidio** (`PresidioDetector`, name `presidio`):

| Presidio entity type | privgate type |
|---|---|
| `PERSON` | PERSON |
| `EMAIL_ADDRESS` | EMAIL |
| `PHONE_NUMBER` | PHONE |
| `CREDIT_CARD` | CREDIT_CARD |
| `IBAN_CODE` | IBAN |
| `IP_ADDRESS` | IP_ADDRESS |
| `URL` | URL |
| `DATE_TIME` | DATE |
| `LOCATION` | LOCATION |
| `ORGANIZATION` | ORGANIZATION |
| `NRP` | OTHER |
| `CRYPTO` | OTHER |
| `MAC_ADDRESS` | OTHER |
| `US_SSN` | ID_NUMBER |
| `US_PASSPORT` | ID_NUMBER |
| `US_DRIVER_LICENSE` | ID_NUMBER |
| `US_BANK_NUMBER` | ID_NUMBER |
| `US_ITIN` | ID_NUMBER |
| `UK_NHS` | ID_NUMBER |
| `MEDICAL_LICENSE` | ID_NUMBER |

**Regex baseline** (`RegexDetector`, name `regex`): CREDIT_CARD, EMAIL, IBAN, IP_ADDRESS, PHONE, URL.
<!-- maps:end -->

## Configuration

- **`SpacyDetector(model="en_core_web_lg", batch_size=64)`** loads only the `ner`
  component. In en_core_web_lg 3.8 NER has its own embedding layer, and excluding the
  other pipes gave identical entities on development documents. `detect_many` batches
  through `nlp.pipe`. The benchmark times `detect()` per document instead, because a
  per-document latency is what a gateway pays.
- **`PresidioDetector(score_threshold=0.0, language="en")`** builds `AnalyzerEngine` from
  Presidio's shipped `conf/default.yaml`: spaCy engine on en_core_web_lg, Presidio's
  spaCy-label mapping, ORGANIZATION and numeric labels ignored, and the default English
  recognizer registry. Building it from a bare model name instead would silently use a
  different NER mapping with no ignore list. Presidio scores are kept in `Span.score`. The
  analyzer is a constructor argument, so tests inject a fake.
- **`EnsembleDetector(detectors, name)`** returns `resolve_overlaps` over the union of its
  members' spans. `supported_labels` is the union of the members'.
- The benchmark's Presidio threshold is selected on dev (docs/results-detection.md).

## Supported labels

<!-- supported:begin -->
| detector | supported_labels | Nemotron raw labels targeted |
|---|---|---|
| `regex` | CREDIT_CARD, EMAIL, IBAN, IP_ADDRESS, PHONE, URL | 7 |
| `spacy` | DATE, LOCATION, ORGANIZATION, OTHER, PERSON | 17 |
| `presidio` | CREDIT_CARD, DATE, EMAIL, IBAN, ID_NUMBER, IP_ADDRESS, LOCATION, OTHER, PERSON, PHONE, URL | 27 |
| `regex+spacy` | CREDIT_CARD, DATE, EMAIL, IBAN, IP_ADDRESS, LOCATION, ORGANIZATION, OTHER, PERSON, PHONE, URL | 24 |
| `regex+presidio` | CREDIT_CARD, DATE, EMAIL, IBAN, ID_NUMBER, IP_ADDRESS, LOCATION, OTHER, PERSON, PHONE, URL | 27 |
<!-- supported:end -->
