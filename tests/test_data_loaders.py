"""Nemotron and Gretel loaders on tiny parquet fixtures with the real schemas."""

import pytest

pytest.importorskip("pyarrow")

import parquet_fixtures as fx

from privgate.data import load_gretel, load_nemotron
from privgate.data.mapping import GRETEL_LABEL_MAP, NEMOTRON_LABEL_MAP
from privgate.spans import EntityType

E = EntityType


@pytest.fixture
def nemotron_file(tmp_path):
    path = tmp_path / "nemotron.parquet"
    fx.write_nemotron(path, fx.default_nemotron_rows())
    return path


@pytest.fixture
def gretel_file(tmp_path):
    path = tmp_path / "gretel.parquet"
    fx.write_gretel(path, fx.default_gretel_rows())
    return path


def texts(doc):
    return [(doc.span_text(s), s.label) for s in doc.spans]


def test_fixtures_use_the_real_schemas(nemotron_file, gretel_file):
    import pyarrow.parquet as pq

    assert pq.read_schema(nemotron_file).remove_metadata() == fx.nemotron_schema()
    assert pq.read_schema(gretel_file).remove_metadata() == fx.gretel_schema()


def test_every_raw_label_in_fixtures_is_mapped(nemotron_file, gretel_file):
    _, nem = load_nemotron(nemotron_file, strict_labels=True)
    _, gre = load_gretel(gretel_file, strict_labels=True)
    assert set(nem.raw_label_counts) <= set(NEMOTRON_LABEL_MAP)
    assert set(gre.raw_label_counts) <= set(GRETEL_LABEL_MAP)
    assert not nem.unknown_labels and not gre.unknown_labels


def test_nemotron_documents(nemotron_file):
    docs, report = load_nemotron(nemotron_file)
    assert [d.doc_id for d in docs] == [
        "nemotron:u1:us", "nemotron:u2:us", "nemotron:u1:intl", "nemotron:u2:intl",
    ]
    first, second, third, empty = docs
    assert texts(first) == [("John Smith", E.PERSON), ("1980-02-03", E.DATE),
                            ("john.smith@example.org", E.EMAIL)]
    # comma blocks the merge; case-only mismatch kept; overlap resolved to the longer span
    assert texts(second) == [("Anna", E.PERSON), ("Lee", E.PERSON),
                             ("Acme Corp", E.ORGANIZATION), ("New York City", E.LOCATION)]
    # JSON-encoded row; int stored text accepted; the wrong coordinate dropped
    assert texts(third) == [("44", E.OTHER), ("555-0100", E.PHONE),
                            ("4111 1111 1111 1111", E.CREDIT_CARD)]
    assert empty.spans == ()
    assert first.meta == {"domain": "Healthcare", "document_type": "intake form",
                          "document_format": "unstructured", "locale": "us", "group_id": "u1"}
    assert first.source == "nemotron-pii"

    assert report.is_consistent()
    assert (report.documents, report.spans_read, report.spans_kept) == (4, 13, 10)
    assert report.spans_dropped == {"out_of_bounds": 0, "offset_mismatch": 1,
                                    "empty_after_trim": 0, "overlap": 1}
    assert (report.spans_merged, report.case_only_mismatches, report.overlapping_pairs) == (
        1, 1, 1)


def test_nemotron_exact_offset_check_drops_case_mismatch(nemotron_file):
    _, report = load_nemotron(nemotron_file, offset_check="exact")
    assert report.spans_dropped["offset_mismatch"] == 2 and report.case_only_mismatches == 0


def test_nemotron_limit_and_no_merge(nemotron_file):
    docs, report = load_nemotron(nemotron_file, merge_adjacent=False, limit=1)
    assert len(docs) == 1 and report.documents == 1
    assert texts(docs[0])[:2] == [("John", E.PERSON), ("Smith", E.PERSON)]


def test_nemotron_unknown_label_strict_and_lenient(tmp_path):
    text = "Favourite colour: teal"
    path = tmp_path / "n.parquet"
    fx.write_nemotron(path, [fx.nemotron_row("u9", "us", text,
                                             [fx.at(text, "teal", "favourite_colour")])])
    docs, report = load_nemotron(path)
    assert docs[0].spans[0].label is E.OTHER and report.unknown_labels == {"favourite_colour": 1}
    with pytest.raises(KeyError, match="favourite_colour"):
        load_nemotron(path, strict_labels=True)


@pytest.mark.parametrize("spans_field", ["[{'start': 0", "{'start': 0}", "[{'start': 0}]"])
def test_nemotron_malformed_spans_raise(tmp_path, spans_field):
    row = fx.nemotron_row("u9", "us", "text", [])
    row["spans"] = spans_field
    path = tmp_path / "n.parquet"
    fx.write_nemotron(path, [row])
    with pytest.raises(ValueError, match="u9"):
        load_nemotron(path)


def test_missing_column_is_an_error(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"uid": ["a"], "text": ["t"]}), path)
    with pytest.raises(ValueError, match="missing columns"):
        load_nemotron(path)


def test_gretel_documents(gretel_file):
    docs, report = load_gretel(gretel_file)
    assert [d.doc_id for d in docs] == ["gretel:35255", "gretel:35135"]
    assert texts(docs[0]) == [("Maria Garcia", E.PERSON),
                              ("DE89370400440532013000", E.IBAN),
                              ("Blue Harbor Bank", E.ORGANIZATION)]
    assert texts(docs[1]) == [("Golden Anchor Marine Insurance Company", E.ORGANIZATION),
                              ("192.0.2.1", E.IP_ADDRESS), ("2024-01-05", E.DATE)]
    assert docs[0].meta == {"domain": "finance", "document_type": "Email",
                            "expanded_type": "test", "language": "English"}
    assert report.is_consistent()
    assert (report.spans_read, report.spans_kept, report.overlapping_pairs) == (7, 6, 1)
    assert report.case_only_mismatches == 0  # Gretel stores no span text to compare


def test_gretel_out_of_bounds_and_bad_json(tmp_path):
    path = tmp_path / "g.parquet"
    row = fx.gretel_row(1, "short", [{"start": 0, "end": 50, "label": "name"}])
    fx.write_gretel(path, [row])
    docs, report = load_gretel(path)
    assert docs[0].spans == () and report.spans_dropped["out_of_bounds"] == 1
    row["pii_spans"] = "[{'start': 0}]"  # Python literal is not valid for Gretel
    fx.write_gretel(path, [row])
    with pytest.raises(ValueError, match="index 1"):
        load_gretel(path)
