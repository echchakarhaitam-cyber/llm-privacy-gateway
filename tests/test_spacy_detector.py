"""SpacyDetector on a blank English pipeline with a fixed EntityRuler (no model needed)."""

import pytest

spacy = pytest.importorskip("spacy")

from privgate.detectors.spacy_ner import SPACY_LABEL_MAP, SpacyDetector
from privgate.spans import EntityType

E = EntityType
TEXT = "Ada Lovelace met Acme Corp in London on Monday at noon; Swiss MONEY 5 dollars."


@pytest.fixture(scope="module")
def detector():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "PERSON", "pattern": "Ada Lovelace"},
        {"label": "ORG", "pattern": "Acme Corp"},
        {"label": "GPE", "pattern": "London"},
        {"label": "DATE", "pattern": "Monday"},
        {"label": "TIME", "pattern": "noon"},
        {"label": "NORP", "pattern": "Swiss"},
        {"label": "MONEY", "pattern": "5 dollars"},   # ignored label
        {"label": "CARDINAL", "pattern": "5"},
    ])
    return SpacyDetector(nlp=nlp)


def found(det, text):
    return [(text[s.start:s.end], s.label) for s in det.detect(text)]


def test_label_mapping_and_offsets(detector):
    assert found(detector, TEXT) == [
        ("Ada Lovelace", E.PERSON), ("Acme Corp", E.ORGANIZATION), ("London", E.LOCATION),
        ("Monday", E.DATE), ("noon", E.OTHER), ("Swiss", E.OTHER),
    ]
    spans = detector.detect(TEXT)
    assert all(s.source == "spacy" and s.score == 1.0 for s in spans)
    assert [s.start for s in spans] == sorted(s.start for s in spans)


def test_ignored_labels_are_not_emitted(detector):
    assert detector.detect("5 dollars") == []
    assert detector.detect("") == []


def test_detect_many_matches_detect(detector):
    texts = [TEXT, "London", "nothing here", ""]
    assert detector.detect_many(texts) == [detector.detect(t) for t in texts]


def test_supported_labels_and_name(detector):
    assert detector.name == "spacy"
    assert detector.supported_labels == frozenset(SPACY_LABEL_MAP.values())
    assert detector.supported_labels == {E.PERSON, E.ORGANIZATION, E.LOCATION, E.DATE, E.OTHER}


def test_map_covers_only_documented_labels():
    assert set(SPACY_LABEL_MAP) == {"PERSON", "ORG", "GPE", "LOC", "FAC", "DATE", "TIME", "NORP"}


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        SpacyDetector(nlp=spacy.blank("en"), batch_size=0)
