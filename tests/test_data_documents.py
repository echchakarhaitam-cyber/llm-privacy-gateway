"""Document model, gold-span normalisation, LoadReport bookkeeping and label maps."""

import dataclasses

import pytest

from privgate.data.documents import (
    Document,
    LoadOptions,
    LoadTally,
    RawSpan,
    build_document,
    count_overlapping_pairs,
    merge_adjacent,
)
from privgate.data.mapping import (
    GRETEL_LABEL_MAP,
    NEMOTRON_LABEL_MAP,
    UnknownLabelError,
    map_label,
)
from privgate.spans import EntityType, Span

P, LOC, ORG, O = EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION, EntityType.OTHER
NEM = "nemotron-pii"


def raw(text, sub, label, stored=None, nth=0):
    start = -1
    for _ in range(nth + 1):
        start = text.index(sub, start + 1)
    return RawSpan(start, start + len(sub), label, sub if stored is None else stored)


def build(text, raws, **opts):
    tally = LoadTally(NEM, LoadOptions(**opts))
    doc = build_document("d1", text, raws, tally)
    return doc, tally.freeze()


def texts(doc):
    return [(doc.span_text(s), s.label) for s in doc.spans]


# --- Document ---------------------------------------------------------------------------

def test_document_is_immutable():
    doc = Document("d", "John", (Span(0, 4, P),), "x", {"domain": "a"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.text = "other"
    with pytest.raises(TypeError):
        doc.meta["domain"] = "b"
    assert isinstance(doc.spans, tuple)


def test_document_copies_meta_and_sorts_spans():
    meta = {"domain": "a"}
    doc = Document("d", "ab cd", [Span(3, 5, P), Span(0, 2, P)], "x", meta)
    meta["domain"] = "changed"
    assert doc.meta["domain"] == "a"
    assert [s.start for s in doc.spans] == [0, 3]


@pytest.mark.parametrize("kwargs, error", [
    ({"doc_id": ""}, ValueError),
    ({"text": 5}, TypeError),
    ({"spans": ("not a span",)}, TypeError),
    ({"spans": (Span(0, 10, P),)}, ValueError),     # beyond text
    ({"meta": {"n": 1}}, TypeError),                 # non-str value
])
def test_document_validation(kwargs, error):
    args = {"doc_id": "d", "text": "John", "spans": (), "source": "x", "meta": {}} | kwargs
    with pytest.raises(error):
        Document(**args)


# --- merge_adjacent ---------------------------------------------------------------------

def test_first_and_last_name_merge_into_one_person():
    doc, report = build("Dr John Smith called.",
                        [raw("Dr John Smith called.", "John", "first_name"),
                         raw("Dr John Smith called.", "Smith", "last_name")])
    assert texts(doc) == [("John Smith", P)]
    assert report.spans_merged == 1


def test_comma_blocks_merge():
    text = "John, Smith"
    doc, _ = build(text, [raw(text, "John", "first_name"), raw(text, "Smith", "last_name")])
    assert texts(doc) == [("John", P), ("Smith", P)]


def test_different_labels_do_not_merge():
    text = "Smith Acme"
    doc, _ = build(text, [raw(text, "Smith", "last_name"), raw(text, "Acme", "company_name")])
    assert texts(doc) == [("Smith", P), ("Acme", ORG)]


def test_line_break_and_touching_spans_do_not_merge():
    text = "Dorkin\nPaul\r\nJohnSmith"
    spans = [Span(0, 6, P), Span(7, 11, P), Span(13, 17, P), Span(17, 22, P)]
    assert merge_adjacent(spans, text) == spans


def test_merge_chains_and_tabs():
    text = "Mary  Ann\tLee"
    spans = [Span(0, 4, P), Span(6, 9, P), Span(10, 13, P)]
    assert merge_adjacent(spans, text) == [Span(0, 13, P)]


def test_merge_can_be_switched_off():
    text = "John Smith"
    doc, report = build(text, [raw(text, "John", "first_name"), raw(text, "Smith", "last_name")],
                        merge_adjacent=False)
    assert len(doc.spans) == 2 and report.spans_merged == 0


# --- offset verification, trimming, overlaps ---------------------------------------------

def test_offset_mismatch_is_counted_and_dropped():
    text = "Call Anna now"
    doc, report = build(text, [raw(text, "Anna", "first_name", stored="Bob"),
                               raw(text, "now", "time")])
    assert texts(doc) == [("now", O)]
    assert report.spans_dropped["offset_mismatch"] == 1
    assert report.is_consistent()


def test_case_only_mismatch_kept_by_default_dropped_when_exact():
    text = "Ethnicity: White"
    spans = [raw(text, "White", "race_ethnicity", stored="white")]
    doc, report = build(text, spans)
    assert texts(doc) == [("White", O)] and report.case_only_mismatches == 1
    doc, report = build(text, spans, offset_check="exact")
    assert doc.spans == () and report.spans_dropped["offset_mismatch"] == 1


@pytest.mark.parametrize("start, end", [(-1, 3), (2, 2), (5, 3), (0, 99), ("0", 3), (True, 3)])
def test_out_of_bounds_or_malformed_offsets_dropped(start, end):
    _, report = build("abcdef", [RawSpan(start, end, "city")])
    assert report.spans_dropped["out_of_bounds"] == 1 and report.spans_kept == 0


def test_whitespace_trimmed_and_blank_span_dropped():
    text = "at  Paris  and   ."
    doc, report = build(text, [RawSpan(2, 11, "city"), RawSpan(14, 17, "city")])
    assert texts(doc) == [("Paris", LOC)]
    assert report.spans_trimmed == 1 and report.spans_dropped["empty_after_trim"] == 1


def test_overlapping_gold_spans_reported_and_resolved():
    text = "in New York City today"
    spans = [raw(text, "New York", "state"), raw(text, "New York City", "city")]
    doc, report = build(text, spans)
    assert texts(doc) == [("New York City", LOC)]
    assert report.overlapping_pairs == 1 and report.spans_dropped["overlap"] == 1
    doc, report = build(text, spans, overlap_policy="keep")
    assert len(doc.spans) == 2 and report.overlapping_pairs == 1
    assert report.spans_dropped["overlap"] == 0


def test_count_overlapping_pairs():
    assert count_overlapping_pairs([Span(0, 10, P), Span(2, 4, P), Span(3, 12, P),
                                    Span(12, 14, P)]) == 3
    assert count_overlapping_pairs([]) == 0


def test_load_options_validation():
    with pytest.raises(ValueError):
        LoadOptions(offset_check="fuzzy")
    with pytest.raises(ValueError):
        LoadOptions(overlap_policy="drop")


def test_load_report_counts_add_up_across_documents():
    tally = LoadTally(NEM, LoadOptions())
    t1 = "John Smith in New York City"
    build_document("a", t1, [raw(t1, "John", "first_name"), raw(t1, "Smith", "last_name"),
                             raw(t1, "New York", "state"), raw(t1, "New York City", "city")],
                   tally)
    t2 = "Anna at 99"
    build_document("b", t2, [raw(t2, "Anna", "first_name", stored="Bob"),
                             RawSpan(8, 50, "age"), raw(t2, "99", "mystery_label")], tally)
    r = tally.freeze()
    assert r.is_consistent()
    assert (r.documents, r.spans_read, r.spans_kept, r.spans_merged) == (2, 7, 3, 1)
    assert sum(r.spans_dropped.values()) == 3
    assert sum(r.kept_label_counts.values()) == r.spans_kept
    assert sum(r.raw_label_counts.values()) == r.spans_read
    assert r.unknown_labels == {"mystery_label": 1}
    assert r.to_dict()["spans_dropped"] == {"out_of_bounds": 1, "offset_mismatch": 1,
                                             "empty_after_trim": 0, "overlap": 1}


# --- mapping ----------------------------------------------------------------------------

def test_unknown_label_strict_raises_default_is_other():
    with pytest.raises(UnknownLabelError):
        map_label(NEM, "favourite_colour", strict=True)
    assert map_label(NEM, "favourite_colour") is None
    doc, report = build("blue", [RawSpan(0, 4, "favourite_colour")])
    assert doc.spans[0].label is O and report.unknown_labels == {"favourite_colour": 1}
    with pytest.raises(UnknownLabelError):
        build("blue", [RawSpan(0, 4, "favourite_colour")], strict_labels=True)


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError):
        map_label("no-such-dataset", "email")


def test_maps_cover_the_observed_label_inventories():
    # Sizes found when the pinned files were inspected (docs/DATA.md).
    assert len(NEMOTRON_LABEL_MAP) == 55 and len(GRETEL_LABEL_MAP) == 29


@pytest.mark.parametrize("dataset, raw_label, expected", [
    (NEM, "first_name", P), (NEM, "fax_number", EntityType.PHONE),
    (NEM, "credit_debit_card", EntityType.CREDIT_CARD), (NEM, "ipv6", EntityType.IP_ADDRESS),
    (NEM, "date_of_birth", EntityType.DATE), (NEM, "coordinate", LOC),
    (NEM, "company_name", ORG), (NEM, "medical_record_number", EntityType.ID_NUMBER),
    (NEM, "occupation", O), (NEM, "cvv", O),
    ("gretel-finance-en", "name", P), ("gretel-finance-en", "iban", EntityType.IBAN),
    ("gretel-finance-en", "local_latlng", LOC), ("gretel-finance-en", "bban",
                                                  EntityType.ID_NUMBER),
])
def test_rule_examples(dataset, raw_label, expected):
    assert map_label(dataset, raw_label, strict=True) is expected
