"""Tokenisation for sequence models: words and single punctuation marks, exact offsets.

``\\w+|[^\\w\\s]`` splits text into maximal runs of word characters (Unicode letters,
digits, underscore) and single non-space, non-word characters. Whitespace is never part of
a token, so ``text[t.start:t.end] == t.text`` always holds and token boundaries are a
subset of character boundaries. The price: an entity boundary that falls inside a token
(e.g. a gold span covering half of a word) cannot be represented; the alignment loss of
this choice is measured in docs/results-detection.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    text: str


def tokenize(text: str) -> list[Token]:
    return [Token(m.start(), m.end(), m.group()) for m in TOKEN_PATTERN.finditer(text)]
