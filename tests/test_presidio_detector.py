"""PresidioDetector with a fake analyzer (no Presidio or model needed)."""

import logging
from dataclasses import dataclass

import pytest

from privgate.detectors.presidio import PRESIDIO_LABEL_MAP, PresidioDetector
from privgate.spans import EntityType

E = EntityType


@dataclass
class Result:
    """Shape of presidio_analyzer.RecognizerResult used by the detector."""

    entity_type: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def __init__(self, results, supported=("PERSON", "EMAIL_ADDRESS", "US_SSN")):
        self.results = results
        self.supported = supported
        self.calls = []

    def analyze(self, text, language, score_threshold):
        self.calls.append((text, language, score_threshold))
        return [r for r in self.results if r.score >= score_threshold]

    def get_supported_entities(self, language):
        return list(self.supported)


TEXT = "Jane Roe, jane@example.org, SSN 123-45-6789, visited Lyon."
def at(sub, entity_type, score, offset=0):
    start = TEXT.index(sub) + offset
    return Result(entity_type, start, TEXT.index(sub) + len(sub), score)


RESULTS = [
    at("Jane Roe", "PERSON", 0.85),
    at("jane@example.org", "EMAIL_ADDRESS", 1.0),
    at("jane@example.org", "URL", 0.5, offset=5),   # inside the email: overlap
    at("123-45-6789", "US_SSN", 0.5),
    at("Lyon", "LOCATION", 0.3),
]


def labels(det, text=TEXT):
    return [(text[s.start:s.end], s.label, s.score) for s in det.detect(text)]


def test_mapping_scores_and_overlap_resolution():
    det = PresidioDetector(analyzer=FakeAnalyzer(RESULTS))
    assert labels(det) == [
        ("Jane Roe", E.PERSON, 0.85), ("jane@example.org", E.EMAIL, 1.0),
        ("123-45-6789", E.ID_NUMBER, 0.5), ("Lyon", E.LOCATION, 0.3),
    ]
    assert all(s.source == "presidio" for s in det.detect(TEXT))


def test_threshold_filters_and_is_passed_to_analyzer():
    fake = FakeAnalyzer(RESULTS)
    det = PresidioDetector(score_threshold=0.5, analyzer=fake)
    assert [lab for _, lab, _ in labels(det)] == [E.PERSON, E.EMAIL, E.ID_NUMBER]
    assert fake.calls[-1] == (TEXT, "en", 0.5)


def test_spans_from_results_equals_detect_at_that_threshold():
    fake = FakeAnalyzer(RESULTS)
    base = PresidioDetector(score_threshold=0.0, analyzer=fake)
    raw = base.analyze_raw(TEXT)
    for t in (0.0, 0.3, 0.5, 0.85, 1.0):
        at_t = PresidioDetector(score_threshold=t, analyzer=FakeAnalyzer(RESULTS))
        assert base.spans_from_results(raw, t) == at_t.detect(TEXT)


def test_unknown_type_becomes_other_with_one_warning(caplog):
    fake = FakeAnalyzer([Result("AU_TFN", 0, 4, 0.9), Result("AU_TFN", 5, 8, 0.9)],
                        supported=("AU_TFN",))
    with caplog.at_level(logging.WARNING, logger="privgate.detectors.presidio"):
        det = PresidioDetector(analyzer=fake)
        spans = det.detect(TEXT)
    assert [s.label for s in spans] == [E.OTHER, E.OTHER]
    assert det.supported_labels == {E.OTHER}
    warnings = [r for r in caplog.records if "AU_TFN" in r.getMessage()]
    assert len(warnings) == 1


def test_supported_labels_follow_the_registry():
    det = PresidioDetector(analyzer=FakeAnalyzer([]))
    assert det.supported_labels == {E.PERSON, E.EMAIL, E.ID_NUMBER}


def test_scores_are_clamped_and_empty_results_dropped():
    fake = FakeAnalyzer([Result("PERSON", 0, 4, 1.2), Result("PERSON", 6, 6, 0.9)])
    assert [(s.start, s.end, s.score) for s in PresidioDetector(analyzer=fake).detect(TEXT)] == [
        (0, 4, 1.0)]


def test_invalid_threshold():
    with pytest.raises(ValueError):
        PresidioDetector(score_threshold=1.5, analyzer=FakeAnalyzer([]))


def test_identifier_types_map_to_id_number():
    ids = {"US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER", "US_ITIN", "UK_NHS",
           "MEDICAL_LICENSE"}
    assert {k for k, v in PRESIDIO_LABEL_MAP.items() if v is E.ID_NUMBER} == ids
    assert PRESIDIO_LABEL_MAP["DATE_TIME"] is E.DATE
    assert PRESIDIO_LABEL_MAP["NRP"] is PRESIDIO_LABEL_MAP["CRYPTO"] is E.OTHER
