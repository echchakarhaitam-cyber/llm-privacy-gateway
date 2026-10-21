"""Descriptive statistics of a document collection, computed in one streaming pass."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from privgate.data.documents import Document
from privgate.spans import EntityType

TOP_DOMAINS = 10


def covered_characters(doc: Document) -> int:
    """Characters inside at least one gold span (overlaps counted once)."""
    covered = 0
    reach = 0
    for s in doc.spans:  # sorted by start
        if s.end > reach:
            covered += s.end - max(s.start, reach)
            reach = s.end
    return covered


@dataclass
class SetStats:
    """Documents, characters, gold spans per label, PII character share and domains."""

    documents: int = 0
    characters: int = 0
    pii_characters: int = 0
    spans_per_label: Counter = field(default_factory=Counter)
    domains: Counter = field(default_factory=Counter)

    def add(self, doc: Document) -> None:
        self.documents += 1
        self.characters += len(doc.text)
        self.pii_characters += covered_characters(doc)
        self.spans_per_label.update(s.label.value for s in doc.spans)
        self.domains[doc.meta.get("domain", "(none)")] += 1

    @property
    def pii_char_share(self) -> float:
        return self.pii_characters / self.characters if self.characters else 0.0

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "characters": self.characters,
            "pii_characters": self.pii_characters,
            "pii_char_share": round(self.pii_char_share, 6),
            "spans": sum(self.spans_per_label.values()),
            "spans_per_label": {e.value: self.spans_per_label[e.value] for e in EntityType},
            "top_domains": [
                [name, n]
                for name, n in sorted(self.domains.items(), key=lambda kv: (-kv[1], kv[0]))
            ][:TOP_DOMAINS],
        }
