"""Do the gold CREDIT_CARD spans hold valid card numbers? (the Milestone 3 puzzle)

The regex baseline only reports digit runs of plausible length that pass the Luhn check
(docs/theory and privgate.detectors.regex); Presidio's credit-card recognizer validates the
checksum too. If many gold card numbers fail Luhn, a large card leakage of validated
detectors is a property of the synthetic data, not a detector bug. Only the digits of each
span are checked (separators removed).
"""

from __future__ import annotations

from collections.abc import Iterable

from privgate.data.documents import Document
from privgate.detectors.validators import luhn_valid
from privgate.spans import EntityType

CARD_LENGTHS = range(13, 20)  # ISO/IEC 7812 primary account numbers: 13-19 digits


def card_digits(text: str) -> str:
    return "".join(ch for ch in text if ch in "0123456789")


def luhn_report(docs: Iterable[Document]) -> dict:
    total = passing = plausible = plausible_passing = 0
    for doc in docs:
        for span in doc.spans:
            if span.label is not EntityType.CREDIT_CARD:
                continue
            digits = card_digits(doc.span_text(span))
            ok = luhn_valid(digits)
            length_ok = len(digits) in CARD_LENGTHS
            total += 1
            passing += ok
            plausible += length_ok
            plausible_passing += ok and length_ok
    return {
        "gold_card_spans": total,
        "luhn_pass": passing,
        "luhn_pass_rate": passing / total if total else 0.0,
        "length_13_19": plausible,
        "length_13_19_and_luhn_pass": plausible_passing,
    }
