# Masking: from detected spans to a text that is safe to send

A detector only says where the personal data is. The gateway must then rewrite the text
before it leaves, and often restore the originals in the model's answer.
`privgate.masking` implements four strategies; all of them replace exactly the detected
spans and copy everything else unchanged.

## The four strategies

| strategy | `Anna Lee paid 4111 1111 1111 1111` becomes | reversible |
|---|---|---|
| redact | `[REDACTED] paid [REDACTED]` | no |
| typed | `<PERSON_1> paid <CREDIT_CARD_1>` | yes |
| pseudonym | `Alison Chavez paid 4210814414984413674` (realistic fakes) | yes |
| partial | `<PERSON_1> paid **** **** **** 1111` | only the typed parts |

- **redact** removes the most: the model no longer knows what kind of thing was there.
- **typed** keeps the type and the identity relation; the model can still say "<PERSON_1>
  should call <PERSON_2>".
- **pseudonym** keeps the text natural, which helps models that react badly to tags. Types
  with no meaningful fake value (OTHER, ID_NUMBER) fall back to typed placeholders.
- **partial** keeps a small part (an email's domain, the last 4 digits of a card, IBAN or
  phone number); every other type falls back to typed.

## Consistency

Within a document, the same original string always gets the same replacement, and
different strings get different ones. Without that, "Anna met Bob. Anna paid." could become
"<PERSON_1> met <PERSON_2>. <PERSON_3> paid." The model would lose the fact that the
same person met and paid, and answers about "who paid" would become wrong or
unrestorable. Matching is exact and case-sensitive ("Anna" and "ANNA" are different
values). That is a conservative choice: merging them would need a normalisation that could
also merge different people. Pseudonyms are deterministic per (seed, label, value), so a
rerun gives the same text and the choice does not depend on the order of the spans.

## Reversibility and its risk

For typed and pseudonym masking, `mask()` returns a mapping replacement -> original, and
`unmask(answer, mapping)` restores the originals in the model's answer. Matching is on
whole tokens, longest replacement first, in a single pass, so a restored value is never
rewritten again. The mapping is exactly as sensitive as the original text, since it IS
the personal data. It must stay in the gateway's memory for the duration of one request,
must never be logged, sent or persisted, and must be discarded afterwards. The benchmark
writes masked texts to disk but never the mappings.

## Placeholder injection

Placeholders are plain text, so an input can already contain one: a document quoting
`<PERSON_1>`, or an attacker who plants it so that `unmask` writes a real name into the
answer at that place. `mask()` therefore checks the input. If anything placeholder-like
is present, all placeholders of that document carry a short tag derived from the seed
(`<PERSON_1#3fa9c2>`) that does not occur in the text. Every result is then verified:
each reversible replacement must occur exactly where it was inserted and nowhere else,
otherwise it is regenerated deterministically. Pseudonyms are likewise chosen so that they
appear nowhere in the original text. A literal `<PERSON_1>` in the input stays untouched
and is never "restored".

## Why partial masking is a policy choice

Partial masking leaks by design: the kept characters are still the person's data, and a
domain plus the last 4 digits of a card can re-identify someone together with other
facts. It exists because some workflows need it (a support agent confirming "the card
ending in 1111"). Whether that residual disclosure is acceptable is a policy decision, not
a technical one. privgate measures it exactly: in the privacy metric, kept characters
count as leaked.

## Measuring the trade-off

Privacy is measured on the masked text against the gold spans: character leakage (gold
characters still present) and verbatim survival (gold values still present anywhere, e.g.
a second mention the detector missed). Utility is the share of non-personal characters
left unchanged, the length ratio, and a downstream proxy task (domain classification).
Comparing each detector with the oracle (masking the gold spans) separates the cost of
the strategy from the cost of detector errors.
