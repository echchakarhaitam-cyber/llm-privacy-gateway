"""Utility of a masked text: what of the original, non-private content is still there.

* ``preserved_non_pii``: share of the original's non-PII characters (outside every gold
  span) that are unchanged in the masked text. Alignment comes from the edits themselves:
  a character outside every edit is copied unchanged, a character inside an edit is gone
  unless the edit keeps it (partial). Non-PII characters are lost only where a detector
  span covers them (a false positive, or a span wider than the gold one).
* ``length_ratio``: total masked length / total original length.

The downstream utility task (domain classification) lives in privgate.bench.domain.
"""

from __future__ import annotations

from collections.abc import Sequence

from privgate.data.documents import Document
from privgate.masking import MaskedText


def _in_gold(i: int, doc: Document) -> bool:
    return any(g.start <= i < g.end for g in doc.spans)


def non_pii_counts(doc: Document, masked: MaskedText) -> tuple[int, int]:
    """(non-PII characters of the original, of which preserved) for one document."""
    gold = sum(g.length for g in doc.spans)  # gold spans do not overlap
    non_pii = len(doc.text) - gold
    kept = {orig for e in masked.edits for orig, _ in e.kept}
    lost = 0
    for e in masked.edits:
        for i in range(e.start, e.end):
            if i not in kept and not _in_gold(i, doc):
                lost += 1
    return non_pii, non_pii - lost


def utility_report(docs: Sequence[Document], masked: Sequence[MaskedText]) -> dict:
    if len(docs) != len(masked):
        raise ValueError("one masked text per document is required")
    non_pii = preserved = original_len = masked_len = 0
    for doc, m in zip(docs, masked):
        n, p = non_pii_counts(doc, m)
        non_pii += n
        preserved += p
        original_len += len(doc.text)
        masked_len += len(m.text)
    return {
        "non_pii_chars": non_pii, "preserved_non_pii_chars": preserved,
        "preserved_non_pii": preserved / non_pii if non_pii else 1.0,
        "original_chars": original_len, "masked_chars": masked_len,
        "length_ratio": masked_len / original_len if original_len else 1.0,
    }
