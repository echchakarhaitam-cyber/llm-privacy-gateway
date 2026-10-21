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
