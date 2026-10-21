"""Tokenisation and BIO conversion, including the repair rule and partial-token overlap."""

import pytest

from privgate.ml.bio import TAGS, bio_to_spans, round_trip, spans_to_bio
from privgate.ml.tokens import tokenize
from privgate.spans import EntityType, Span

E = EntityType


def test_tokenize_offsets_are_exact():
    text = "Dr. Ana-María O'Neil, tel +33 6 12; e-mail: a.b@x.fr!"
    tokens = tokenize(text)
    assert all(text[t.start:t.end] == t.text for t in tokens)
    assert [t.text for t in tokens][:8] == ["Dr", ".", "Ana", "-", "María", "O", "'", "Neil"]
    assert not any(t.text.isspace() for t in tokens)


def test_tokenize_unicode_and_edge_cases():
    text = "Zoë\tMüller  東京 ١٢٣  x"
    assert [t.text for t in tokenize(text)] == ["Zoë", "Müller", "東京", "١٢٣", "x"]
    assert tokenize("") == [] and tokenize("   \n ") == []
    combining = "été"  # e + combining acute: the mark is not a word character
    assert "".join(t.text for t in tokenize(combining)) == combining


def test_tag_inventory():
    assert len(TAGS) == 25 and TAGS[0] == "O" and "B-PERSON" in TAGS and "I-OTHER" in TAGS


def test_spans_to_bio_and_back():
    text = "Call John Smith at john@x.org now"
    spans = [Span(5, 15, E.PERSON), Span(19, 29, E.EMAIL)]
    tokens = tokenize(text)
    tags = spans_to_bio(tokens, spans)
    assert tags == ["O", "B-PERSON", "I-PERSON", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL",
                    "I-EMAIL", "I-EMAIL", "O"]
    assert bio_to_spans(tokens, tags) == spans


def test_adjacent_same_label_spans_stay_separate():
    text = "Anna, Bob"
    spans = [Span(0, 4, E.PERSON), Span(6, 9, E.PERSON)]
    assert round_trip(text, spans) == spans
    touching = [Span(0, 2, E.DATE), Span(2, 4, E.DATE)]   # two spans inside one token
    assert round_trip("2024", touching) == [Span(0, 4, E.DATE)]


def test_partial_token_overlap_expands_to_token():
    text = "ID: AB12345 end"
    spans = [Span(6, 11, E.ID_NUMBER)]  # covers only "12345" of the token "AB12345"
    assert round_trip(text, spans) == [Span(4, 11, E.ID_NUMBER)]


def test_token_overlapping_two_spans_takes_the_larger_overlap():
    tokens = tokenize("abcdef")
    assert spans_to_bio(tokens, [Span(0, 2, E.PERSON), Span(2, 6, E.EMAIL)]) == ["B-EMAIL"]


def test_repair_rule_i_without_b_starts_a_span():
    tokens = tokenize("a b c d e")
    tags = ["I-PERSON", "I-PERSON", "O", "B-DATE", "I-EMAIL"]
    assert bio_to_spans(tokens, tags) == [
        Span(0, 3, E.PERSON), Span(6, 7, E.DATE), Span(8, 9, E.EMAIL)]


def test_bio_scores_and_errors():
    tokens = tokenize("a b")
    spans = bio_to_spans(tokens, ["B-PERSON", "I-PERSON"], source="m", scores=[0.9, 0.6])
    assert spans == [Span(0, 3, E.PERSON, 0.6, "m")]
    with pytest.raises(ValueError):
        bio_to_spans(tokens, ["O"])
    with pytest.raises(ValueError):
        bio_to_spans(tokens, ["X-PERSON", "O"])
