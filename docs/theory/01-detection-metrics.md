# Detection metrics and checksum validators

## Spans

A detection is a span `[start, end)` over the characters of a text, with a label
(`EMAIL`, `PHONE`, `PERSON`, ...). Gold annotations and predictions use the same
representation, so they can be compared directly.

## Three ways to match a prediction to a gold span

| Mode | A prediction is a true positive when ... | Question answered |
|---|---|---|
| strict | it has exactly the gold boundaries and label | Is this a perfect NER system? |
| partial | it has the gold label and shares >= 1 character | Did it find the entity, roughly? |
| type_agnostic | it shares >= 1 character, whatever the label | Was the entity hidden at all? |

Matching is one-to-one and greedy: candidate pairs are taken by decreasing overlap,
ties by smaller gold start, then smaller prediction start. Unmatched gold spans are
false negatives (FN), unmatched predictions false positives (FP).
Precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R); a zero
denominator gives 0. Macro scores average over the union of gold and predicted labels.

## Worked example

Gold: `PERSON [0,5)`, `EMAIL [10,20)`, `PHONE [30,40)`.
Predictions: `PERSON [0,5)`, `EMAIL [10,18)`, `EMAIL [30,40)`, `PERSON [50,55)`.

| Mode | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| strict | 1 | 3 | 2 | 0.25 | 1/3 | 2/7 |
| partial | 2 | 2 | 1 | 0.50 | 2/3 | 4/7 |
| type_agnostic | 3 | 1 | 0 | 0.75 | 1 | 6/7 |

The same predictions look poor as NER (strict F1 0.29) but hide every sensitive entity
(type-agnostic recall 1). Both views are reported because they answer different questions.

## Character level: the privacy view

For a gateway, the costs are asymmetric. A sensitive character left visible may leak
data; an extra masked character only reduces the usefulness of the text. We therefore
also measure, over the union of spans (overlapping predictions count each character once):

- `char_recall` = gold characters covered by any prediction / all gold characters;
- `leakage_rate` = 1 - `char_recall` (0 when there is nothing to hide);
- `char_precision` = predicted characters inside a gold span / all predicted characters.

In the worked example 23 of the 25 gold characters are covered (the email loses its last
two), so `leakage_rate` = 2/25 = 0.08, and `char_precision` = 23/28.
Corpus scores sum counts over documents before dividing (micro-averaging).

## Luhn check (payment cards)

From the rightmost digit, double every second digit and subtract 9 from results above 9;
the number is valid iff the sum is a multiple of 10.
Example `79927398713`: digits from the right are 3 1 7 8 9 3 7 2 9 9 7; the doubled ones
become 2, 7 (16 - 9), 6, 4, 9 (18 - 9); the sum is 70, so the number is valid. Changing the
last digit to 0 gives 67: invalid. The check catches every single-digit error.

## ISO 13616 IBAN check (mod 97-10)

Remove spaces, move the first four characters to the end, replace letters by numbers
(A = 10, ..., Z = 35) and compute the remainder modulo 97; the IBAN is valid iff it is 1.
Example `GB82 WEST 1234 5698 7654 32` becomes `WEST12345698765432GB82`, then
`3214282912345698765432161182`, whose remainder modulo 97 is 1: valid.
The country-specific length table is not checked by this baseline.

## What the regex baseline does not cover

Names, places, organisations and dates are free text: no regular expression finds them
reliably. Phone coverage is limited to the formats listed in `privgate/detectors/regex.py`.
These limits are why statistical detectors are evaluated next.
