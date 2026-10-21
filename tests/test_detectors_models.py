"""Integration tests with the real en_core_web_lg model: run with ``pytest -m models``."""

import importlib.util

import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(importlib.util.find_spec("en_core_web_lg") is None,
                       reason="spaCy model en_core_web_lg is not installed"),
]

from privgate.spans import EntityType


@pytest.fixture(scope="module")
def spacy_detector():
    from privgate.detectors import SpacyDetector

    return SpacyDetector()


def test_real_spacy_finds_person_with_offsets(spacy_detector):
    text = "Barack Obama visited Paris."
    spans = spacy_detector.detect(text)
    persons = [s for s in spans if s.label is EntityType.PERSON]
    assert [(s.start, s.end) for s in persons] == [(0, 12)]
    assert text[persons[0].start:persons[0].end] == "Barack Obama"
    assert any(text[s.start:s.end] == "Paris" and s.label is EntityType.LOCATION for s in spans)
    assert spacy_detector.detect_many([text]) == [spans]


@pytest.fixture(scope="module")
def presidio_detector():
    pytest.importorskip("presidio_analyzer")
    from privgate.detectors import PresidioDetector

    return PresidioDetector()


def test_real_presidio_finds_email_and_phone(presidio_detector):
    text = "Contact Maria at maria.lopez@example.com or call (212) 555-0187 tomorrow."
    found = {(text[s.start:s.end], s.label) for s in presidio_detector.detect(text)}
    assert ("maria.lopez@example.com", EntityType.EMAIL) in found
    assert ("(212) 555-0187", EntityType.PHONE) in found


def test_real_presidio_registry_is_fully_mapped(presidio_detector):
    from privgate.detectors.presidio import PRESIDIO_LABEL_MAP

    registered = presidio_detector._analyzer.get_supported_entities("en")
    assert set(registered) <= set(PRESIDIO_LABEL_MAP)
    assert EntityType.ORGANIZATION not in presidio_detector.supported_labels
