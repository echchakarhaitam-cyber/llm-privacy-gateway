"""Token features shared by the CRF and the per-token logistic regression.

Per token (each feature also exists for the neighbours at offsets -2, -1, +1, +2, with the
offset as key prefix, e.g. ``-1:lower``; the token itself uses ``0:``):

==============  =========================================================================
``lower``       lowercased token text
``shape``       character classes, ``X`` upper, ``x`` lower, ``d`` digit, other kept as is
                (``John`` -> ``Xxxx``, ``06-12`` -> ``dd-dd``); capped at 12 characters
``sshape``      short shape: runs of the same class collapsed (``Xxxx`` -> ``Xx``)
``p1..p3``      prefixes of length 1-3 (lowercased)
``s1..s3``      suffixes of length 1-3 (lowercased)
``len``         length bucket: ``1``, ``2-3``, ``4-6``, ``7-10``, ``11+``
``digit``       1.0 if all characters are digits
``upper``       1.0 if all cased characters are upper case
``title``       1.0 if title case
``hasdigit``    1.0 if any character is a digit
``hasat``       1.0 if the token is ``@`` or contains it
``punct``       1.0 if the token is a single non-word character
==============  =========================================================================

Plus, for the token itself only: ``bias`` (1.0), ``BOS`` / ``EOS`` (1.0 at the first / last
token), and the stacked feature ``regex``: the label the regex baseline gives the token
(``EMAIL``, ``CREDIT_CARD``, ... or ``O``), so the model can trust the validated
structured detectors and spend its capacity elsewhere.

Boolean features are present with value 1.0 only when true (absent otherwise), which both
crfsuite and scikit-learn's DictVectorizer read the same way. Features depend only on the
text: extraction is deterministic. Bump :data:`FEATURE_VERSION` when anything changes;
saved models record it and refuse to load with another version.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from privgate.detectors.regex import RegexDetector
from privgate.ml.bio import OUTSIDE, spans_to_bio
from privgate.ml.tokens import Token

FEATURE_VERSION = "m4-v1"
WINDOW = (-2, -1, 1, 2)
MAX_SHAPE = 12
_REGEX = RegexDetector()


def _char_class(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isupper():
        return "X"
    if ch.islower():
        return "x"
    return ch


def _length_bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


@lru_cache(maxsize=200_000)
def word_features(text: str) -> tuple[tuple[str, object], ...]:
    """Features of one token text (cached: vocabularies repeat heavily)."""
    lower = text.lower()
    shape = "".join(_char_class(c) for c in text)[:MAX_SHAPE]
    short = "".join(c for i, c in enumerate(shape) if i == 0 or c != shape[i - 1])
    feats: list[tuple[str, object]] = [
        ("lower", lower), ("shape", shape), ("sshape", short), ("len", _length_bucket(len(text))),
    ]
    feats += [(f"p{k}", lower[:k]) for k in (1, 2, 3) if len(lower) >= k]
    feats += [(f"s{k}", lower[-k:]) for k in (1, 2, 3) if len(lower) >= k]
    flags = {
        "digit": text.isdigit(),
        "upper": text.isupper(),
        "title": text.istitle(),
        "hasdigit": any(c.isdigit() for c in text),
        "hasat": "@" in text,
        "punct": len(text) == 1 and not (text.isalnum() or text == "_"),
    }
    feats += [(name, 1.0) for name, on in flags.items() if on]
    return tuple(feats)


@lru_cache(maxsize=400_000)
def _prefixed(text: str, offset: int) -> tuple[tuple[str, object], ...]:
    tag = "0" if offset == 0 else f"{offset:+d}"   # 0, -2, -1, +1, +2
    return tuple((f"{tag}:{k}", v) for k, v in word_features(text))


def regex_labels(text: str, tokens: Sequence[Token]) -> list[str]:
    """Stacked feature: the regex baseline's label of each token, or ``O``."""
    tags = spans_to_bio(tokens, _REGEX.detect(text))
    return [OUTSIDE if t == OUTSIDE else t.split("-", 1)[1] for t in tags]


def doc_features(text: str, tokens: Sequence[Token]) -> list[dict[str, object]]:
    """One feature dict per token."""
    words = [t.text for t in tokens]
    stacked = regex_labels(text, tokens)
    out = []
    last = len(words) - 1
    for i, word in enumerate(words):
        feats: dict[str, object] = {"bias": 1.0, "regex": stacked[i]}
        feats.update(_prefixed(word, 0))
        for off in WINDOW:
            j = i + off
            if 0 <= j <= last:
                feats.update(_prefixed(words[j], off))
        if i == 0:
            feats["BOS"] = 1.0
        if i == last:
            feats["EOS"] = 1.0
        out.append(feats)
    return out
