import dataclasses

import pytest

from privgate.spans import EntityType, Span, resolve_overlaps

P = EntityType.PERSON
E = EntityType.EMAIL
U = EntityType.URL


def test_basic_properties():
    s = Span(2, 7, P)
    assert s.length == 5
    assert s.text("0123456789") == "23456"
    assert s.score == 1.0


def test_label_from_string_is_converted():
    assert Span(0, 1, "EMAIL").label is E


@pytest.mark.parametrize(
    "args, exc",
    [
        ((-1, 2, P), ValueError),
        ((3, 3, P), ValueError),
        ((4, 2, P), ValueError),
        ((0.0, 2, P), TypeError),
        ((True, 2, P), TypeError),
        ((0, 2, "NOT_A_LABEL"), ValueError),
    ],
)
def test_invalid_spans(args, exc):
    with pytest.raises(exc):
        Span(*args)


@pytest.mark.parametrize("score", [-0.1, 1.5, float("nan"), float("inf")])
def test_invalid_scores(score):
    with pytest.raises(ValueError):
        Span(0, 1, P, score=score)


def test_score_bool_rejected():
    with pytest.raises(TypeError):
        Span(0, 1, P, score=True)


def test_text_outside_raises():
    with pytest.raises(ValueError):
        Span(0, 20, P).text("short")


def test_overlap():
    a, b, c = Span(0, 5, P), Span(4, 9, P), Span(5, 9, P)
    assert a.overlap_length(b) == 1 and a.overlaps(b)
    assert a.overlap_length(c) == 0 and not a.overlaps(c)  # half-open: touching != overlapping


def test_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Span(0, 1, P).start = 3


def test_resolve_longer_wins():
    url = Span(0, 30, U)
    email = Span(10, 20, E)
    assert resolve_overlaps([email, url]) == [url]


def test_resolve_score_then_priority_then_position():
    a = Span(0, 10, P, score=0.9)
    b = Span(0, 10, E, score=0.8)
    assert resolve_overlaps([b, a]) == [a]  # same length -> higher score
    c = Span(0, 10, P)
    d = Span(0, 10, E)
    assert resolve_overlaps([c, d], priority=[E, P]) == [d]  # same length+score -> priority
    assert resolve_overlaps([c, d]) == [d] == resolve_overlaps([d, c])  # label name: EMAIL < PERSON
    e = Span(0, 10, P)
    f = Span(5, 15, P)
    assert resolve_overlaps([f, e]) == [e]  # earlier start wins


def test_resolve_keeps_disjoint_and_sorts():
    spans = [Span(20, 25, P), Span(0, 5, E), Span(10, 15, U)]
    assert resolve_overlaps(spans) == [Span(0, 5, E), Span(10, 15, U), Span(20, 25, P)]


def test_resolve_empty():
    assert resolve_overlaps([]) == []
