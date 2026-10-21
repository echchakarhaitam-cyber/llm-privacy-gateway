"""EnsembleDetector union/overlap behaviour and supported_labels of every detector."""

import pytest

from privgate.detectors import Detector, EnsembleDetector, RegexDetector
from privgate.spans import EntityType, Span

E = EntityType


class Fixed:
    """A detector returning fixed spans."""

    def __init__(self, name, spans, labels):
        self.name = name
        self._spans = spans
        self.supported_labels = frozenset(labels)

    def detect(self, text):
        return list(self._spans)


A = Fixed("a", [Span(0, 4, E.PERSON, source="a"), Span(10, 20, E.EMAIL, source="a")],
          {E.PERSON, E.EMAIL})
B = Fixed("b", [Span(0, 9, E.PERSON, source="b"), Span(12, 15, E.URL, source="b"),
                Span(30, 35, E.DATE, source="b")], {E.PERSON, E.URL, E.DATE})


def test_union_then_overlap_resolution():
    ens = EnsembleDetector([A, B], name="a+b")
    assert ens.detect("x" * 40) == [
        Span(0, 9, E.PERSON, source="b"),      # longer of the two person spans
        Span(10, 20, E.EMAIL, source="a"),     # contains the URL span
        Span(30, 35, E.DATE, source="b"),
    ]
    assert ens.name == "a+b"
    assert isinstance(ens, Detector)


def test_combine_equals_detect_and_is_order_independent():
    text = "x" * 40
    ens = EnsembleDetector([A, B], name="a+b")
    assert EnsembleDetector.combine([A.detect(text), B.detect(text)]) == ens.detect(text)
    assert EnsembleDetector([B, A], name="b+a").detect(text) == ens.detect(text)


def test_supported_labels_is_union():
    assert EnsembleDetector([A, B], "a+b").supported_labels == {E.PERSON, E.EMAIL, E.URL, E.DATE}


def test_regex_supported_labels():
    det = RegexDetector()
    assert det.supported_labels == {E.EMAIL, E.URL, E.IBAN, E.CREDIT_CARD, E.IP_ADDRESS, E.PHONE}
    assert {s.label for s in det.detect("a@b.org +33 6 12 34 56 78")} <= det.supported_labels


def test_invalid_ensembles():
    with pytest.raises(ValueError):
        EnsembleDetector([], "empty")
    with pytest.raises(ValueError):
        EnsembleDetector([A], "")
