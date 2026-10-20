"""Checksum and range validators that turn regex candidates into confident detections.

A regular expression alone cannot tell a card number from any other 16-digit string.
Many identifiers carry a check digit precisely so that typing errors can be caught;
we reuse those checks to cut false positives. Each function is pure and total: it
returns False for malformed input instead of raising.
"""

from __future__ import annotations

import re

_IBAN_SHAPE = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}")


def luhn_valid(digits: str) -> bool:
    """Luhn (mod 10) check used by payment card numbers.

    From the rightmost digit, double every second digit; if doubling gives a two-digit
    number, subtract 9 (= add its digits). The number is valid iff the total is a
    multiple of 10. Example: 4111 1111 1111 1111 -> 30 -> valid.
    """
    if len(digits) < 2 or not digits.isascii() or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(s: str) -> bool:
    """ISO 13616 IBAN check (ISO 7064 mod 97-10).

    Remove spaces, move the first four characters to the end, replace each letter by
    two digits (A=10 ... Z=35) and read the result as an integer: the IBAN is valid iff
    that integer mod 97 equals 1. The country-specific length table is NOT checked;
    the generic shape is 2 letters + 2 digits + 11 to 30 alphanumerics.
    """
    compact = s.replace(" ", "").upper()
    if not compact.isascii() or _IBAN_SHAPE.fullmatch(compact) is None:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def ipv4_valid(s: str) -> bool:
    """Dotted-quad IPv4: exactly four decimal octets in 0..255.

    Leading zeros (``"01"``) are rejected because some parsers read them as octal; a
    baseline should prefer a missed detection over an ambiguous one.
    """
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part or len(part) > 3 or not part.isascii() or not part.isdigit():
            return False
        if len(part) > 1 and part[0] == "0":
            return False
        if int(part) > 255:
            return False
    return True
