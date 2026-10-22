"""Automatic error taxonomy of leaked gold spans (Milestone 7, scripts/error_taxonomy.py).

A *leaked gold span* is a gold span the detector did not find (a type-agnostic false negative,
exactly the errors of privgate.bench.errors): its characters stay in the text after masking.
Each leaked span is described along independent dimensions, all computed from the span, its
text and the dataset labels -- no manual labelling:

``mapped_label``  privgate's EntityType of the gold span
``raw_label``     the dataset's own label (privgate.bench.rawlabels)
``untargeted``    the raw label is targeted by none of the M3 detectors (privgate.bench.support)
``shape``         the format of the value, first matching rule of :func:`shape_of`
``length``        number of characters, bucketed (:data:`LENGTH_BUCKETS`)
``position``      where the value stands, :func:`position_of`

Each dimension assigns exactly one category per span, so counts of a dimension sum to the
number of leaked spans.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from privgate.bench.errors import ErrorCase
from privgate.bench.rawlabels import UNKNOWN

DIMENSIONS = ("mapped_label", "raw_label", "untargeted", "shape", "length", "position")
LENGTH_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 10**9))
UNTARGETED = "untargeted raw label"
TARGETED = "targeted by an M3 detector"
SHAPES = ("all digits", "contains punctuation", "mixed alphanumeric", "title-case word(s)",
          "lowercase word(s)", "other letters")
POSITIONS = ("table / delimited line", "key: value line", "prose")

_KEY_VALUE = re.compile(r"^[ \t]*(?:[-*•][ \t]*)?[A-Za-z][\w .#/()-]{0,40}?[ \t]*[:=][ \t]+")


def shape_of(value: str) -> str:
    """Format of a value; the first rule that matches wins.

    1. all digits: every non-space character is a digit
    2. contains punctuation: any character that is neither alphanumeric nor whitespace
    3. mixed alphanumeric: letters and digits, no punctuation
    4. title-case word(s): ``str.istitle()`` (``Anna``, ``New York``)
    5. lowercase word(s): ``str.islower()``
    6. other letters: upper case, camelCase, anything else
    """
    chars = [c for c in value if not c.isspace()]
    if chars and all(c.isdigit() for c in chars):
        return SHAPES[0]
    if any(not c.isalnum() for c in chars):
        return SHAPES[1]
    if any(c.isdigit() for c in chars) and any(c.isalpha() for c in chars):
        return SHAPES[2]
    if value.istitle():
        return SHAPES[3]
    if value.islower():
        return SHAPES[4]
    return SHAPES[5]


def length_bucket(n: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    return "0"


def position_of(text: str, start: int, end: int) -> str:
    """Where the span stands, from the line that contains its start.

    * ``table / delimited line``: the line has at least two ``|`` or a tab (pipe tables, TSV);
    * ``key: value line``: the line starts like ``Label: `` / ``- label = `` (a short label of
      at most ~40 characters followed by ``:`` or ``=``) and the span begins after that
      delimiter;
    * ``prose``: everything else.
    """
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end != -1 else len(text)]
    if line.count("|") >= 2 or "\t" in line:
        return POSITIONS[0]
    m = _KEY_VALUE.match(line)
    if m and start - line_start >= m.end():
        return POSITIONS[1]
    return POSITIONS[2]


def categorize(case: ErrorCase, text: str, untargeted: frozenset[str]) -> dict[str, str]:
    s = case.span
    raw = case.raw_label or UNKNOWN
    return {
        "mapped_label": s.label.value,
        "raw_label": raw,
        "untargeted": UNTARGETED if raw in untargeted else TARGETED,
        "shape": shape_of(text[s.start:s.end]),
        "length": length_bucket(s.end - s.start),
        "position": position_of(text, s.start, s.end),
    }


@dataclass(frozen=True)
class Categorized:
    case: ErrorCase
    categories: Mapping[str, str]


def categorize_all(cases: Iterable[ErrorCase], texts: Mapping[str, str],
                   untargeted: frozenset[str]) -> list[Categorized]:
    return [Categorized(c, categorize(c, texts[c.doc_id], untargeted)) for c in cases]


def count_categories(items: Sequence[Categorized]) -> dict[str, Counter]:
    return {dim: Counter(i.categories[dim] for i in items) for dim in DIMENSIONS}


def ranked(counter: Counter, total: int, limit: int | None = None) -> list[dict]:
    """Categories by count (ties: name), with their share of ``total``."""
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"category": k, "count": v, "share": v / total if total else 0.0}
            for k, v in rows[:limit]]


def seeded_examples(items: Sequence[Categorized], dimension: str, category: str,
                    set_name: str, n: int = 10, seed: int = 0) -> list[ErrorCase]:
    """``n`` cases of one category, drawn with a seed that depends on (set, dimension, category)
    so each draw is reproducible and independent of the others."""
    cases = sorted((i.case for i in items if i.categories[dimension] == category),
                   key=lambda c: (c.doc_id, c.span.start, c.span.end, c.span.label.value))
    rng = random.Random(f"{seed}|{set_name}|{dimension}|{category}")
    return rng.sample(cases, min(n, len(cases)))
