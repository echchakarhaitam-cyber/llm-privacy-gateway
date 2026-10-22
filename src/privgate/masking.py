"""Masking: turn detected spans into a safe text, and restore originals in a model's answer.

Strategies (docs/theory/03-masking.md):

* ``redact``    every span -> ``[REDACTED]``. Not reversible.
* ``typed``     span -> ``<LABEL_n>``, n numbered per label in order of first appearance; the
                same original string (exact, case-sensitive) gets the same placeholder
                within a document. Reversible.
* ``pseudonym`` span -> a realistic fake value of the same type (Faker, optional extra
                ``masking``), deterministic per (seed, label, original value) and consistent
                within a document. Reversible. Types without a meaningful fake
                (:data:`PSEUDONYM_FALLBACK`: OTHER, ID_NUMBER) fall back to ``typed``.
* ``partial``   keeps a minimal part (:func:`partial_value`): EMAIL keeps ``@domain``;
                CREDIT_CARD / IBAN / PHONE with at least 8 digits keep their last 4 digits
                and their separators; everything else falls back to ``typed``. Partial
                masking LEAKS BY DESIGN (the kept characters are still the person's data):
                it is a policy choice, measured in docs/results-masking.md.

Collision safety: a placeholder must never be confused with text that was already there.
When the original text contains something placeholder-like, placeholders carry a short
per-document tag derived from the seed (``<PERSON_1#3fa9c2>``); in every case the result is
verified (each reversible replacement occurs exactly as often as it was inserted, as a
whole token) and regenerated deterministically with the next attempt number otherwise.
Pseudonyms are also chosen so that they occur nowhere in the original text and stay
whole tokens at their positions.

Overlapping input spans are resolved first with :func:`privgate.spans.resolve_overlaps`.
Each :class:`Edit` records which original characters it keeps, so privacy and utility
metrics can align original and masked text exactly, without a fuzzy diff.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from privgate.spans import EntityType, Span, resolve_overlaps

T = EntityType
STRATEGIES = ("redact", "typed", "pseudonym", "partial")
REDACTED = "[REDACTED]"
REVERSIBLE = frozenset({"typed", "pseudonym"})

PSEUDONYM_PROVIDERS: Mapping[EntityType, str] = MappingProxyType({
    T.PERSON: "name", T.EMAIL: "email", T.PHONE: "phone_number",
    T.CREDIT_CARD: "credit_card_number", T.IBAN: "iban", T.IP_ADDRESS: "ipv4",
    T.URL: "url", T.DATE: "date", T.LOCATION: "city", T.ORGANIZATION: "company",
})
PSEUDONYM_FALLBACK = frozenset(set(EntityType) - set(PSEUDONYM_PROVIDERS))
PARTIAL_NUMBER_LABELS = frozenset({T.CREDIT_CARD, T.IBAN, T.PHONE})
PARTIAL_MIN_DIGITS = 8
PARTIAL_KEEP_DIGITS = 4
PLACEHOLDER_LIKE = re.compile(r"<[A-Z][A-Z_]*_\d+(?:#[0-9a-f]+)?>")
MAX_ATTEMPTS = 50
FAKER_LOCALE = "en_US"


class MaskingError(RuntimeError):
    """No unambiguous masking could be produced."""


@dataclass(frozen=True)
class Edit:
    """One replaced span: original ``[start, end)`` -> output ``[out_start, out_end)``.

    ``kept`` lists ``(original_index, output_index)`` of original characters copied into the
    output unchanged (partial masking only).
    """

    start: int
    end: int
    out_start: int
    out_end: int
    label: EntityType
    original: str
    replacement: str
    kept: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class MaskedText:
    text: str
    mapping: Mapping[str, str]      # reversible replacement -> original value
    spans_out: tuple[Span, ...]     # replaced spans, offsets in the masked text
    edits: tuple[Edit, ...]
    strategy: str
    tag: str = ""


def _digest(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest(), 16)


def _placeholder(label: EntityType, n: int, tag: str) -> str:
    return f"<{label.value}_{n}#{tag}>" if tag else f"<{label.value}_{n}>"


def partial_value(value: str, label: EntityType) -> tuple[str, tuple[int, ...]] | None:
    """Partial replacement of ``value`` and the indices (in value) it keeps, or None.

    None means "no partial form": the caller falls back to a typed placeholder.
    """
    if label is T.EMAIL and "@" in value:
        at = value.rindex("@")
        if at == 0 or at == len(value) - 1:
            return None
        return "***" + value[at:], tuple(range(at, len(value)))
    if label in PARTIAL_NUMBER_LABELS:
        digits = [i for i, c in enumerate(value) if c.isdigit()]
        if len(digits) < PARTIAL_MIN_DIGITS:
            return None
        keep_digits = set(digits[-PARTIAL_KEEP_DIGITS:])
        out, kept = [], []
        for i, c in enumerate(value):
            if i in keep_digits or not c.isalnum():
                out.append(c)
                kept.append(i)
            else:
                out.append("*")
        return "".join(out), tuple(kept)
    return None


def _token_pattern(keys: Iterable[str]) -> re.Pattern | None:
    """Whole-token alternation of ``keys``, longest first (privgate.masking.unmask)."""
    parts = []
    for key in sorted(set(keys), key=lambda k: (-len(k), k)):
        left = r"(?<!\w)" if re.match(r"\w", key[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", key[-1]) else ""
        parts.append(left + re.escape(key) + right)
    return re.compile("|".join(parts)) if parts else None


def unmask(response: str, mapping: Mapping[str, str]) -> str:
    """Replace every reversible placeholder or pseudonym in ``response`` by its original.

    Matching is on whole tokens, longest key first, in one pass (a restored value is never
    rescanned). With an empty mapping (redact) the response is returned unchanged.
    """
    pattern = _token_pattern(mapping)
    if pattern is None:
        return response
    return pattern.sub(lambda m: mapping[m.group(0)], response)


class Masker:
    """Applies a masking strategy; pseudonyms need Faker (imported lazily)."""

    def __init__(self, faker_locale: str = FAKER_LOCALE):
        self.faker_locale = faker_locale
        self._faker = None

    def mask(self, text: str, spans: Iterable[Span], strategy: str = "typed",
             seed: int = 0) -> MaskedText:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        ordered = resolve_overlaps(list(spans))
        for s in ordered:
            if s.end > len(text):
                raise ValueError(f"span {s} exceeds the text")
        needs_tag = bool(PLACEHOLDER_LIKE.search(text))
        for attempt in range(MAX_ATTEMPTS):
            tag = self._tag(text, seed, attempt) if (needs_tag or attempt) else ""
            result = self._build(text, ordered, strategy, seed, attempt, tag)
            if strategy == "redact" or self._unambiguous(result):
                return result
        raise MaskingError("could not build unambiguous placeholders")

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _tag(text: str, seed: int, attempt: int) -> str:
        for k in range(MAX_ATTEMPTS):
            tag = f"{_digest('tag', seed, attempt, k, text):x}"[:6]
            if f"#{tag}" not in text:
                return tag
        raise MaskingError("no free placeholder tag")

    def _replacements(self, text: str, spans: list[Span], strategy: str, seed: int,
                      attempt: int, tag: str) -> dict[tuple[EntityType, str], str]:
        """One replacement per distinct (label, original value)."""
        keys: list[tuple[EntityType, str]] = []
        for s in spans:
            key = (s.label, text[s.start:s.end])
            if key not in keys:
                keys.append(key)       # order of first appearance
        counters: dict[EntityType, int] = {}
        typed: dict[tuple[EntityType, str], str] = {}
        for label, value in keys:
            counters[label] = counters.get(label, 0) + 1
            typed[(label, value)] = _placeholder(label, counters[label], tag)
        if strategy == "typed":
            return typed
        if strategy == "pseudonym":
            out = dict(typed)
            taken: set[str] = set()
            # sorted, so the choice does not depend on the order of the input spans
            for label, value in sorted(keys, key=lambda k: (k[0].value, k[1])):
                if label in PSEUDONYM_FALLBACK:
                    continue
                fake = self._fake(text, spans, label, value, seed, attempt, taken)
                if fake is not None:
                    out[(label, value)] = fake
                    taken.add(fake)
            return out
        return typed  # partial: typed is the fallback; partial forms are applied per edit

    def _fake(self, text, spans, label, value, seed, attempt, taken) -> str | None:
        """A fake value that never occurs in the text and stays a whole token in place."""
        faker = self._faker_instance()
        provider = PSEUDONYM_PROVIDERS[label]
        for k in range(MAX_ATTEMPTS):
            faker.seed_instance(_digest("fake", seed, label.value, value, attempt, k))
            fake = str(getattr(faker, provider)())
            if not fake or fake in taken or fake in text or PLACEHOLDER_LIKE.search(fake):
                continue
            if all(self._stays_token(text, s, fake) for s in spans
                   if s.label is label and text[s.start:s.end] == value):
                return fake
        return None  # falls back to the typed placeholder

    @staticmethod
    def _stays_token(text: str, span: Span, fake: str) -> bool:
        before = text[span.start - 1] if span.start else ""
        after = text[span.end] if span.end < len(text) else ""
        glued_left = bool(before) and (before.isalnum() or before == "_") and \
            (fake[0].isalnum() or fake[0] == "_")
        glued_right = bool(after) and (after.isalnum() or after == "_") and \
            (fake[-1].isalnum() or fake[-1] == "_")
        return not (glued_left or glued_right)

    def _faker_instance(self):
        if self._faker is None:
            try:
                from faker import Faker
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError("pseudonym masking needs Faker: pip install -e '.[masking]'") \
                    from exc
            self._faker = Faker(self.faker_locale)
        return self._faker

    def _build(self, text: str, spans: list[Span], strategy: str, seed: int, attempt: int,
               tag: str) -> MaskedText:
        replacements = ({} if strategy == "redact"
                        else self._replacements(text, spans, strategy, seed, attempt, tag))
        pieces: list[str] = []
        edits: list[Edit] = []
        mapping: dict[str, str] = {}
        pos = out_len = 0
        for s in spans:
            pieces.append(text[pos:s.start])
            out_len += s.start - pos
            value = text[s.start:s.end]
            kept: tuple[tuple[int, int], ...] = ()
            if strategy == "redact":
                replacement = REDACTED
            elif strategy == "partial" and (part := partial_value(value, s.label)):
                replacement, kept_idx = part
                kept = tuple((s.start + i, out_len + i) for i in kept_idx)
            else:
                replacement = replacements[(s.label, value)]
                mapping[replacement] = value
            pieces.append(replacement)
            edits.append(Edit(s.start, s.end, out_len, out_len + len(replacement), s.label,
                              value, replacement, kept))
            out_len += len(replacement)
            pos = s.end
        pieces.append(text[pos:])
        masked = "".join(pieces)
        spans_out = tuple(Span(e.out_start, e.out_end, e.label, source="mask") for e in edits)
        return MaskedText(masked, MappingProxyType(mapping), spans_out, tuple(edits), strategy,
                          tag)

    @staticmethod
    def _unambiguous(result: MaskedText) -> bool:
        """Each reversible key occurs exactly where it was inserted, nowhere else."""
        pattern = _token_pattern(result.mapping)
        if pattern is None:
            return True
        found = [(m.start(), m.group(0)) for m in pattern.finditer(result.text)]
        inserted = [(e.out_start, e.replacement) for e in result.edits
                    if e.replacement in result.mapping]
        return found == inserted


_DEFAULT = Masker()


def mask(text: str, spans: Iterable[Span], strategy: str = "typed", seed: int = 0) -> MaskedText:
    """Module-level shortcut for ``Masker().mask``."""
    return _DEFAULT.mask(text, spans, strategy, seed)
