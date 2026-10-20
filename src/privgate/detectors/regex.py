"""Rule-based PII baseline: regular expressions plus checksum validation.

This detector is a *baseline*, not a product. It covers structured identifiers whose
shape is regular (emails, phone numbers, payment cards, IBANs, IPv4 addresses, URLs)
and deliberately ignores free-text entities such as names, places and organisations,
which need statistical models.

Coverage of phone numbers (anything else is NOT detected):
  * international numbers written with a leading ``+`` and a 1-3 digit country code,
    in up to six digit groups separated by single spaces, dots or dashes, with 8 to 15
    digits in total (the E.164 maximum);
  * French and Moroccan national numbers: ``0`` + a non-zero digit + 8 digits, either
    contiguous (``0612345678``) or in pairs separated by one space, dot or dash
    (``06 12 34 56 78``).

Safety against catastrophic backtracking (ReDoS): every repeated group either has a
fixed upper bound or is forced to end on a delimiter it cannot itself contain, and the
look-behinds stop a match from starting in the middle of a run of the same characters.
A gateway runs this code on untrusted input, so this property is tested.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from privgate.detectors.validators import iban_valid, ipv4_valid, luhn_valid
from privgate.spans import EntityType, Span, resolve_overlaps

_EMAIL = re.compile(
    r"(?<![\w.+%-])[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24}(?![\w-])"
)
_URL = re.compile(r"(?<![\w@/.])(?:https?://|www\.)[^\s<>\"'`]{1,2048}", re.IGNORECASE)
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)")
_CARD = re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])")
_IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]){11,30}(?![A-Za-z0-9])")
_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .-]?\d{1,4}){2,6}(?!\w)")
_PHONE_NATIONAL = re.compile(r"(?<![\w+])0[1-9](?:[ .-]?\d{2}){4}(?!\w)")

_URL_TRAILING = ".,;:!?"

#: Used by resolve_overlaps when two candidates have the same length and score.
PRIORITY: tuple[EntityType, ...] = (
    EntityType.EMAIL,
    EntityType.URL,
    EntityType.IBAN,
    EntityType.CREDIT_CARD,
    EntityType.IP_ADDRESS,
    EntityType.PHONE,
)


def _group_ends(candidate: str) -> list[int]:
    """Offsets (relative to the candidate) where a separator-delimited group ends.

    Returned longest first. Used to try shorter prefixes when a greedy match swallowed
    a following word or number group (e.g. an IBAN followed by an upper-case word).
    """
    ends = [i for i in range(1, len(candidate)) if candidate[i] in " -" and candidate[i - 1] != " "]
    ends.append(len(candidate))
    return sorted(set(ends), reverse=True)


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


class RegexDetector:
    """Regex + validator baseline for structured identifiers."""

    name = "regex"

    def detect(self, text: str) -> list[Span]:
        candidates: list[Span] = []
        for finder in (
            self._emails,
            self._urls,
            self._ibans,
            self._cards,
            self._ipv4s,
            self._phones,
        ):
            candidates.extend(finder(text))
        return resolve_overlaps(candidates, PRIORITY)

    # Each finder yields validated spans of one entity type.

    @staticmethod
    def _span(start: int, end: int, label: EntityType) -> Span:
        return Span(start, end, label, score=1.0, source="regex")

    def _emails(self, text: str) -> Iterator[Span]:
        for m in _EMAIL.finditer(text):
            yield self._span(m.start(), m.end(), EntityType.EMAIL)

    def _urls(self, text: str) -> Iterator[Span]:
        for m in _URL.finditer(text):
            value = m.group()
            value = value.rstrip(_URL_TRAILING)
            # Drop a closing parenthesis that closes surrounding prose, not the URL.
            while value.endswith(")") and value.count(")") > value.count("("):
                value = value[:-1].rstrip(_URL_TRAILING)
            if len(value) > len("www."):
                yield self._span(m.start(), m.start() + len(value), EntityType.URL)

    def _shrinking(
        self, pattern: re.Pattern[str], text: str, accept: Callable[[str], bool], label: EntityType
    ) -> Iterator[Span]:
        for m in pattern.finditer(text):
            candidate = m.group()
            for end in _group_ends(candidate):
                if accept(candidate[:end]):
                    yield self._span(m.start(), m.start() + end, label)
                    break

    def _ibans(self, text: str) -> Iterator[Span]:
        yield from self._shrinking(_IBAN, text, iban_valid, EntityType.IBAN)

    def _cards(self, text: str) -> Iterator[Span]:
        def accept(s: str) -> bool:
            digits = _digits(s)
            return 13 <= len(digits) <= 19 and luhn_valid(digits)

        yield from self._shrinking(_CARD, text, accept, EntityType.CREDIT_CARD)

    def _ipv4s(self, text: str) -> Iterator[Span]:
        for m in _IPV4.finditer(text):
            if ipv4_valid(m.group()):
                yield self._span(m.start(), m.end(), EntityType.IP_ADDRESS)

    def _phones(self, text: str) -> Iterator[Span]:
        for m in _PHONE_INTL.finditer(text):
            if 8 <= len(_digits(m.group())) <= 15:
                yield self._span(m.start(), m.end(), EntityType.PHONE)
        for m in _PHONE_NATIONAL.finditer(text):
            yield self._span(m.start(), m.end(), EntityType.PHONE)
