"""Error taxonomy: fixed examples for every rule, partition property, seeded examples."""

import pytest

from privgate.bench.errors import ErrorCase
from privgate.bench.taxonomy import (
    DIMENSIONS,
    POSITIONS,
    SHAPES,
    TARGETED,
    UNTARGETED,
    categorize,
    categorize_all,
    count_categories,
    length_bucket,
    position_of,
    ranked,
    seeded_examples,
    shape_of,
)
from privgate.spans import EntityType, Span


@pytest.mark.parametrize("value,shape", [
    ("123456", SHAPES[0]), ("12 34 56", SHAPES[0]), ("9", SHAPES[0]),
    ("john@example.org", SHAPES[1]), ("12/05/2023", SHAPES[1]), ("O'Brien", SHAPES[1]),
    ("ID-00123", SHAPES[1]), ("AB123", SHAPES[2]), ("4e5f", SHAPES[2]),
    ("Anna", SHAPES[3]), ("New York", SHAPES[3]),
    ("teacher", SHAPES[4]), ("senior engineer", SHAPES[4]),
    ("USA", SHAPES[5]), ("iPhone", SHAPES[5]), ("McDonald", SHAPES[5]),
    ("", SHAPES[5]),
])
def test_shape_rules(value, shape):
    assert shape_of(value) == shape


@pytest.mark.parametrize("n,bucket", [(1, "1-3"), (3, "1-3"), (4, "4-7"), (7, "4-7"),
                                      (8, "8-15"), (15, "8-15"), (16, "16-31"), (31, "16-31"),
                                      (32, "32+"), (500, "32+"), (0, "0")])
def test_length_buckets(n, bucket):
    assert length_bucket(n) == bucket


def locate(text, value):
    start = text.index(value)
    return start, start + len(value)


@pytest.mark.parametrize("text,value,expected", [
    ("Dear Anna, welcome back.", "Anna", POSITIONS[2]),
    ("Name: Anna Lee\nCity: Paris", "Anna Lee", POSITIONS[1]),
    ("Name: Anna Lee\nCity: Paris", "Paris", POSITIONS[1]),
    ("- employee id = 4412", "4412", POSITIONS[1]),
    ("| Anna | 4412 | Paris |", "4412", POSITIONS[0]),
    ("Anna\t4412\tParis", "4412", POSITIONS[0]),
    ("Anna Lee: said hello to me", "Anna Lee", POSITIONS[2]),      # span is BEFORE the colon
    ("This is a very long sentence that has a colon at the end of it: Anna", "Anna",
     POSITIONS[2]),                                                   # label longer than 40 chars
    ("Prose first.\nKey: value", "Prose", POSITIONS[2]),
    ("One | pipe only: Anna", "Anna", POSITIONS[2]),                  # a single pipe is not a table
])
def test_position_heuristic(text, value, expected):
    start, end = locate(text, value)
    assert position_of(text, start, end) == expected


def case(doc_id, start, end, label=EntityType.PERSON, raw="first_name"):
    return ErrorCase(doc_id, Span(start, end, label), raw)


TEXTS = {"d1": "Name: Anna Lee\nCall 5551234 or mail a@b.org please.",
         "d2": "The teacher Bob wrote | x | y |"}


def test_categorize_one_case():
    c = case("d1", 6, 14, EntityType.PERSON, "first_name")
    cats = categorize(c, TEXTS["d1"], frozenset({"occupation"}))
    assert cats == {"mapped_label": "PERSON", "raw_label": "first_name", "untargeted": TARGETED,
                    "shape": SHAPES[3], "length": "8-15", "position": POSITIONS[1]}
    cats = categorize(case("d1", 6, 14, raw="occupation"), TEXTS["d1"], frozenset({"occupation"}))
    assert cats["untargeted"] == UNTARGETED
    assert categorize(case("d1", 6, 14, raw=None), TEXTS["d1"], frozenset())["raw_label"] \
        == "(unknown)"


def items():
    cases = [case("d1", 6, 14), case("d1", 20, 27, EntityType.PHONE, "phone_number"),
             case("d1", 31, 38, EntityType.EMAIL, "email"),
             case("d2", 12, 15, EntityType.PERSON, "first_name"),
             case("d2", 4, 11, EntityType.OTHER, "occupation")]
    return categorize_all(cases, TEXTS, frozenset({"occupation"}))


def test_every_dimension_partitions_the_leaked_spans():
    its = items()
    counts = count_categories(its)
    assert set(counts) == set(DIMENSIONS)
    assert all(sum(c.values()) == len(its) for c in counts.values())
    assert counts["mapped_label"]["PERSON"] == 2
    assert counts["untargeted"][UNTARGETED] == 1


def test_ranked_orders_by_count_then_name_and_gives_shares():
    its = items()
    rows = ranked(count_categories(its)["mapped_label"], len(its))
    assert [r["category"] for r in rows] == ["PERSON", "EMAIL", "OTHER", "PHONE"]
    assert rows[0] == {"category": "PERSON", "count": 2, "share": 0.4}
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
    assert len(ranked(count_categories(its)["mapped_label"], len(its), limit=2)) == 2
    assert ranked(count_categories([])["shape"], 0) == []


def test_seeded_examples_are_reproducible_and_scoped():
    cases = [case(f"d{i}", 0, 4) for i in range(30)]
    its = categorize_all(cases, {c.doc_id: "Anna was here" for c in cases}, frozenset())
    a = seeded_examples(its, "mapped_label", "PERSON", "set1", n=10)
    assert a == seeded_examples(its, "mapped_label", "PERSON", "set1", n=10)
    assert len(a) == 10 and len({c.doc_id for c in a}) == 10
    assert a != seeded_examples(its, "mapped_label", "PERSON", "set2", n=10)
    assert len(seeded_examples(its, "mapped_label", "PERSON", "set1", n=100)) == 30
    assert seeded_examples(its, "mapped_label", "EMAIL", "set1") == []
    shuffled = seeded_examples(list(reversed(its)), "mapped_label", "PERSON", "set1", n=10)
    assert shuffled == a                       # independent of input order
