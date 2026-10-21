"""Microsoft Presidio Analyzer used as a PII detector.

The default analyzer is Presidio's ``AnalyzerEngine`` with its spaCy NLP engine on
``en_core_web_lg`` (the same model as :class:`~privgate.detectors.spacy_ner.SpacyDetector`)
and Presidio's default English recognizer registry. With presidio-analyzer 2.2.364 that
registry can return the 19 entity types listed in :data:`PRESIDIO_LABEL_MAP` (except
ORGANIZATION, which Presidio's spaCy recognizer ignores by default; it is mapped in case a
custom configuration returns it).

Mapping choices: identifier types issued to a person or account (US_SSN, US_PASSPORT, ...)
-> ID_NUMBER. MAC_ADDRESS -> OTHER, CRYPTO -> OTHER and NRP -> OTHER, because the gold
mapping (docs/DATA.md) puts MAC addresses, technical identifiers and ethnic/religious/
political attributes in OTHER; mapping them elsewhere would make every correct hit a type
error. DATE_TIME -> DATE. An entity type not in the map is emitted as OTHER and logged once.

Presidio's confidence is kept in ``Span.score``. The analyzer is injectable so that unit
tests can use a fake returning RecognizerResult-like objects (``entity_type``, ``start``,
``end``, ``score``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from privgate.spans import EntityType, Span, resolve_overlaps

T = EntityType
logger = logging.getLogger(__name__)

PRESIDIO_LABEL_MAP: Mapping[str, EntityType] = MappingProxyType({
    "PERSON": T.PERSON,
    "EMAIL_ADDRESS": T.EMAIL,
    "PHONE_NUMBER": T.PHONE,
    "CREDIT_CARD": T.CREDIT_CARD,
    "IBAN_CODE": T.IBAN,
    "IP_ADDRESS": T.IP_ADDRESS,
    "URL": T.URL,
    "DATE_TIME": T.DATE,
    "LOCATION": T.LOCATION,
    "ORGANIZATION": T.ORGANIZATION,
    "NRP": T.OTHER,
    "CRYPTO": T.OTHER,
    "MAC_ADDRESS": T.OTHER,
    "US_SSN": T.ID_NUMBER,
    "US_PASSPORT": T.ID_NUMBER,
    "US_DRIVER_LICENSE": T.ID_NUMBER,
    "US_BANK_NUMBER": T.ID_NUMBER,
    "US_ITIN": T.ID_NUMBER,
    "UK_NHS": T.ID_NUMBER,
    "MEDICAL_LICENSE": T.ID_NUMBER,
})

DEFAULT_MODEL = "en_core_web_lg"


def build_default_analyzer(language: str = "en", model: str = DEFAULT_MODEL) -> Any:
    """Presidio AnalyzerEngine with Presidio's shipped default NLP configuration.

    The configuration is read from presidio_analyzer/conf/default.yaml (spaCy engine on
    en_core_web_lg, its spaCy-label mapping and ``labels_to_ignore``, which drops
    ORGANIZATION); only the model name and language are overridden. Passing just a model
    name instead would silently fall back to a different NER mapping with no ignore list.
    """
    try:
        import presidio_analyzer
        import yaml
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PresidioDetector needs presidio-analyzer: pip install -e '.[detectors]'"
        ) from exc
    conf_file = Path(presidio_analyzer.__file__).parent / "conf" / "default.yaml"
    config = yaml.safe_load(conf_file.read_text(encoding="utf-8"))
    config["models"] = [{"lang_code": language, "model_name": model}]
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])


class PresidioDetector:
    """PII spans from Presidio results scoring at least ``score_threshold``."""

    name = "presidio"

    def __init__(
        self,
        score_threshold: float = 0.0,
        language: str = "en",
        analyzer: Any = None,
        model: str = DEFAULT_MODEL,
    ):
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        self.score_threshold = float(score_threshold)
        self.language = language
        self.model = model
        self._analyzer = analyzer if analyzer is not None else build_default_analyzer(
            language, model)
        self._warned: set[str] = set()
        self.supported_labels: frozenset[EntityType] = frozenset(
            self._map(e) for e in self._analyzer.get_supported_entities(language)
        )

    def detect(self, text: str) -> list[Span]:
        return self.spans_from_results(self.analyze_raw(text), self.score_threshold)

    def analyze_raw(self, text: str) -> list[Any]:
        """Presidio's own results at this detector's threshold, before mapping."""
        return self._analyzer.analyze(
            text=text, language=self.language, score_threshold=self.score_threshold)

    def spans_from_results(self, results: Iterable[Any], threshold: float) -> list[Span]:
        """Keep results with ``score >= threshold``, map them, resolve overlaps.

        Applying this to results obtained at a lower threshold gives the same spans as
        analysing at ``threshold`` directly: Presidio filters by ``score >= threshold`` and
        then only removes duplicates contained in a same-type result with a higher or
        equal score, which a higher threshold cannot bring back. The threshold sweep in
        :mod:`privgate.bench.sweep` relies on this.
        """
        spans = [
            Span(r.start, r.end, self._map(r.entity_type),
                 score=min(1.0, max(0.0, float(r.score))), source=self.name)
            for r in results
            if r.score >= threshold and r.end > r.start
        ]
        return resolve_overlaps(spans)

    def _map(self, entity_type: str) -> EntityType:
        label = PRESIDIO_LABEL_MAP.get(entity_type)
        if label is None:
            if entity_type not in self._warned:
                self._warned.add(entity_type)
                logger.warning("unmapped Presidio entity type %r emitted as OTHER", entity_type)
            return T.OTHER
        return label
