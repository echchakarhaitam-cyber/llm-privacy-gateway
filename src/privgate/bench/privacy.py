"""Privacy of a masked text, measured against the gold spans of the original.

* ``char_leakage``: share of gold PII characters whose original character is still present
  in the masked text at the position the transformation maps it to. Characters outside every
  edit are copied unchanged; characters inside an edit are gone, except those an edit keeps
  explicitly (partial masking). For redact / typed / pseudonym this equals the detector's
  character leakage; for partial it is higher, by exactly the kept characters.
* ``verbatim_survival``: share of gold PII values (distinct strings of a document, length
  >= :data:`MIN_VALUE_LENGTH`) that still occur verbatim anywhere in the masked text. It
  catches what the character view misses: a value that is masked where the detector found
  it but survives in another, missed mention (or in text the annotators did not label).

Both are summed over documents (micro) and reported per gold label, with 95% document
bootstrap intervals (privgate.bench.bootstrap).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from privgate.bench.bootstrap import bootstrap_cis
from privgate.data.documents import Document
from privgate.masking import MaskedText

MIN_VALUE_LENGTH = 4


def _overlap(a_start: int, a_end: int, edits) -> int:
    return sum(max(0, min(a_end, e.end) - max(a_start, e.start)) for e in edits)


def leaked_chars(doc: Document, masked: MaskedText) -> dict[str, list[int]]:
    """Per gold label: [gold characters, leaked characters] of one document."""
    kept = {orig for e in masked.edits for orig, _ in e.kept}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g in doc.spans:
        removed = _overlap(g.start, g.end, masked.edits)
        kept_here = sum(1 for i in range(g.start, g.end) if i in kept)
        acc = out[g.label.value]
        acc[0] += g.length
        acc[1] += g.length - removed + kept_here
    return out


def surviving_values(doc: Document, masked: MaskedText) -> dict[str, list[int]]:
    """Per gold label: [distinct values, values still present verbatim] of one document."""
    first_label: dict[str, str] = {}
    for g in doc.spans:
        value = doc.span_text(g)
        if len(value) >= MIN_VALUE_LENGTH:
            first_label.setdefault(value, g.label.value)
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for value, label in first_label.items():
        out[label][0] += 1
        out[label][1] += value in masked.text
    return out


def privacy_report(docs: Sequence[Document], masked: Sequence[MaskedText],
                   n_resamples: int = 1000, seed: int = 0) -> dict:
    if len(docs) != len(masked):
        raise ValueError("one masked text per document is required")
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    cols: dict[str, list[int]] = {"gold": [], "leaked": [], "values": [], "surviving": []}
    for doc, m in zip(docs, masked):
        chars, values = leaked_chars(doc, m), surviving_values(doc, m)
        cols["gold"].append(sum(v[0] for v in chars.values()))
        cols["leaked"].append(sum(v[1] for v in chars.values()))
        cols["values"].append(sum(v[0] for v in values.values()))
        cols["surviving"].append(sum(v[1] for v in values.values()))
        for label, (g, k) in chars.items():
            per_label[label][0] += g
            per_label[label][1] += k
        for label, (n, s) in values.items():
            per_label[label][2] += n
            per_label[label][3] += s
    totals = {k: sum(v) for k, v in cols.items()}

    def ratio(a, b):
        return a / b if b else 0.0

    cis = bootstrap_cis(cols, {
        "char_leakage": lambda s: ratio(s["leaked"], s["gold"]),
        "verbatim_survival": lambda s: ratio(s["surviving"], s["values"]),
    }, n_resamples, seed) if docs else {}
    return {
        "documents": len(docs),
        "gold_chars": totals["gold"], "leaked_chars": totals["leaked"],
        "char_leakage": ratio(totals["leaked"], totals["gold"]),
        "values": totals["values"], "values_surviving": totals["surviving"],
        "verbatim_survival": ratio(totals["surviving"], totals["values"]),
        "per_label": {label: {"gold_chars": g, "leaked_chars": k,
                              "char_leakage": ratio(k, g), "values": n,
                              "values_surviving": s, "verbatim_survival": ratio(s, n)}
                      for label, (g, k, n, s) in sorted(per_label.items())},
        "ci95": {k: list(v) for k, v in cis.items()},
    }
