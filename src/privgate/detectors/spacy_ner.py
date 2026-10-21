"""spaCy statistical NER (``en_core_web_lg``) used as a PII detector.

spaCy's OntoNotes labels are mapped explicitly (:data:`SPACY_LABEL_MAP`); every other label
(CARDINAL, MONEY, PRODUCT, ...) is ignored, i.e. never emitted, because it is not personal
data in the project's taxonomy. ``TIME`` and ``NORP`` (nationalities, religious and
political groups) map to ``OTHER``, mirroring the gold mapping of time-of-day and of
religious/political/ethnic attributes.

Only the ``ner`` component is loaded: in ``en_core_web_lg`` 3.8 the NER component embeds its
own token-to-vector layer, so excluding the shared ``tok2vec``, tagger, parser, attribute
ruler and lemmatizer gives the same entities (checked on development documents) at a
fraction of the cost.

spaCy is imported lazily so that the core library does not depend on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from privgate.spans import EntityType, Span, resolve_overlaps

T = EntityType

SPACY_LABEL_MAP: Mapping[str, EntityType] = MappingProxyType({
    "PERSON": T.PERSON,
    "ORG": T.ORGANIZATION,
    "GPE": T.LOCATION,
    "LOC": T.LOCATION,
    "FAC": T.LOCATION,
    "DATE": T.DATE,
    "TIME": T.OTHER,
    "NORP": T.OTHER,
})

#: Components not needed for NER; excluded at load time.
NON_NER_PIPES = ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "senter")


def load_ner_pipeline(model: str) -> Any:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("SpacyDetector needs spaCy: pip install -e '.[detectors]'") from exc
    return spacy.load(model, exclude=list(NON_NER_PIPES))


class SpacyDetector:
    """PII spans from spaCy entities. ``nlp`` can be injected (tests use a blank pipeline)."""

    name = "spacy"
    supported_labels: frozenset[EntityType] = frozenset(SPACY_LABEL_MAP.values())

    def __init__(self, model: str = "en_core_web_lg", batch_size: int = 64, nlp: Any = None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.batch_size = batch_size
        self._nlp = nlp if nlp is not None else load_ner_pipeline(model)

    def detect(self, text: str) -> list[Span]:
        return self._spans(self._nlp(text))

    def detect_many(self, texts: Iterable[str]) -> list[list[Span]]:
        """Same result as ``[detect(t) for t in texts]``, batched through ``nlp.pipe``."""
        return [self._spans(doc) for doc in self._nlp.pipe(texts, batch_size=self.batch_size)]

    def _spans(self, doc: Any) -> list[Span]:
        spans = [
            Span(ent.start_char, ent.end_char, SPACY_LABEL_MAP[ent.label_], source=self.name)
            for ent in doc.ents
            if ent.label_ in SPACY_LABEL_MAP and ent.end_char > ent.start_char
        ]
        return resolve_overlaps(spans)
