"""BIO tags over privgate's 12 entity types (25 tags), and conversions to and from spans.

spans -> tags (:func:`spans_to_bio`): a token takes the label of the span it overlaps (if
it overlaps several, the one sharing the most characters, ties: earlier span). The first
token of a span gets ``B-``, the following tokens of the same span ``I-``; tokens outside
every span get ``O``. A token only partly covered by a span is tagged as a whole.

tags -> spans (:func:`bio_to_spans`): a run of ``B-X I-X I-X ...`` becomes one span from
the start of its first token to the end of its last token. **Repair rule**: an ``I-X``
that does not continue an open ``X`` run (it follows ``O``, another label, or nothing)
starts a new span, exactly as if it were ``B-X``. Models trained per token (logistic
regression) produce such sequences; treating them as entities rather than dropping them is
the fail-safe choice for a privacy gateway.

The round trip gold -> tags -> spans is not the identity: boundaries inside tokens move to
token boundaries, and two same-label spans in one token merge. :func:`round_trip`
measures this tokenisation ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence

from privgate.ml.tokens import Token, tokenize
from privgate.spans import EntityType, Span

OUTSIDE = "O"
TAGS: tuple[str, ...] = (OUTSIDE,) + tuple(
    f"{prefix}-{e.value}" for e in EntityType for prefix in ("B", "I"))


def spans_to_bio(tokens: Sequence[Token], spans: Sequence[Span]) -> list[str]:
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    tags: list[str] = []
    previous: Span | None = None
    j = 0
    for tok in tokens:
        while j < len(ordered) and ordered[j].end <= tok.start:
            j += 1
        best: Span | None = None
        best_overlap = 0
        k = j
        while k < len(ordered) and ordered[k].start < tok.end:
            overlap = min(tok.end, ordered[k].end) - max(tok.start, ordered[k].start)
            if overlap > best_overlap:
                best, best_overlap = ordered[k], overlap
            k += 1
        if best is None:
            tags.append(OUTSIDE)
        else:
            prefix = "I" if best is previous else "B"
            tags.append(f"{prefix}-{best.label.value}")
        previous = best
    return tags


def bio_to_spans(tokens: Sequence[Token], tags: Sequence[str], source: str = "",
                 scores: Sequence[float] | None = None) -> list[Span]:
    """Spans of B/I runs, with the repair rule of the module docstring.

    ``scores`` (one per token, optional) gives each span the minimum score of its tokens.
    """
    if len(tokens) != len(tags):
        raise ValueError("one tag per token is required")
    spans: list[Span] = []
    run: tuple[int, int, str, float] | None = None  # (start, end, label, score)

    def close() -> None:
        if run is not None:
            spans.append(Span(run[0], run[1], EntityType(run[2]), run[3], source))

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        score = 1.0 if scores is None else scores[i]
        if tag == OUTSIDE:
            close()
            run = None
            continue
        prefix, _, label = tag.partition("-")
        if prefix not in ("B", "I") or not label:
            raise ValueError(f"invalid BIO tag {tag!r}")
        if prefix == "I" and run is not None and run[2] == label:
            run = (run[0], tok.end, label, min(run[3], score))
        else:  # B-, or an I- that repairs into a new span
            close()
            run = (tok.start, tok.end, label, score)
    close()
    return spans


def round_trip(text: str, spans: Sequence[Span]) -> list[Span]:
    """Gold spans after tokenisation and BIO encoding/decoding (the model's best case)."""
    tokens = tokenize(text)
    return bio_to_spans(tokens, spans_to_bio(tokens, spans))
