"""Tiny synthetic parquet files with exactly the column names and types of the real files.

The schemas below were read from the pinned files (docs/DATA.md, "Schema findings");
tests/test_data_integration.py checks them against the real files, so a fixture can not
silently drift from the data it stands in for. All text here is invented.
"""

from __future__ import annotations

import json

NEMOTRON_COLUMNS = (
    "uid", "domain", "document_type", "document_description", "document_format", "locale",
    "text", "spans", "text_tagged",
)
GRETEL_STRING_COLUMNS = (
    "document_type", "document_description", "expanded_type", "expanded_description",
    "language", "language_description", "domain", "generated_text", "pii_spans",
)
GRETEL_SCORE_COLUMNS = (
    "conformance_score", "quality_score", "toxicity_score", "bias_score", "groundedness_score",
)


def nemotron_schema():
    import pyarrow as pa

    return pa.schema([(c, pa.string()) for c in NEMOTRON_COLUMNS])


def gretel_schema():
    import pyarrow as pa

    return pa.schema(
        [(c, pa.string()) for c in GRETEL_STRING_COLUMNS]
        + [(c, pa.int64()) for c in GRETEL_SCORE_COLUMNS]
        + [
            ("pii_types", pa.list_(pa.string())),
            ("pii_type_labels", pa.list_(pa.int64())),
            ("__index_level_0__", pa.int64()),
        ]
    )


def at(text: str, sub: str, label: str, nth: int = 0, stored=None) -> dict:
    """A span dict covering the ``nth`` occurrence of ``sub``; ``stored`` overrides 'text'."""
    start = -1
    for _ in range(nth + 1):
        start = text.index(sub, start + 1)
    return {"start": start, "end": start + len(sub),
            "text": sub if stored is None else stored, "label": label}


def nemotron_row(uid, locale, text, spans, *, as_json=False, domain="Healthcare") -> dict:
    """One row; ``spans`` is serialised like the real file: a Python literal, or JSON."""
    return {
        "uid": uid, "domain": domain, "document_type": "intake form",
        "document_description": "synthetic test document", "document_format": "unstructured",
        "locale": locale, "text": text,
        "spans": json.dumps(spans) if as_json else repr(spans),
        "text_tagged": text,
    }


def write_nemotron(path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows, schema=nemotron_schema()), path)


def gretel_row(index, text, spans, *, document_type="Email") -> dict:
    spans = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in spans]
    return {
        "document_type": document_type, "document_description": "synthetic",
        "expanded_type": "test", "expanded_description": "synthetic", "language": "English",
        "language_description": "English", "domain": "finance", "generated_text": text,
        "pii_spans": json.dumps(spans),
        "conformance_score": 90, "quality_score": 90, "toxicity_score": 0, "bias_score": 0,
        "groundedness_score": 90,
        "pii_types": [s["label"] for s in spans], "pii_type_labels": [0 for _ in spans],
        "__index_level_0__": index,
    }


def write_gretel(path, rows) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows, schema=gretel_schema()), path)


# --- default fixture content ------------------------------------------------------------

T1 = "Patient John Smith, born 1980-02-03, email john.smith@example.org."
T2 = "Contact: Anna, Lee works at Acme Corp in New York City."
T3 = "Age 44, phone 555-0100, card 4111 1111 1111 1111, coordinates 40.7, -74.0."


def default_nemotron_rows() -> list[dict]:
    return [
        nemotron_row("u1", "us", T1, [
            at(T1, "John", "first_name"), at(T1, "Smith", "last_name"),
            at(T1, "1980-02-03", "date_of_birth"), at(T1, "john.smith@example.org", "email"),
        ]),
        nemotron_row("u2", "us", T2, [
            at(T2, "Anna", "first_name"), at(T2, "Lee", "last_name"),
            at(T2, "Acme Corp", "company_name", stored="acme corp"),   # case-only mismatch
            at(T2, "New York", "state"), at(T2, "New York City", "city"),  # overlap
        ]),
        nemotron_row("u1", "intl", T3, [
            at(T3, "44", "age", stored=44),                              # int stored text
            at(T3, "555-0100", "phone_number"),
            at(T3, "4111 1111 1111 1111", "credit_debit_card"),
            at(T3, "40.7, -74.0", "coordinate", stored="41.0, -73.0"),  # real mismatch
        ], as_json=True),
        nemotron_row("u2", "intl", "No personal data here.", []),
    ]


G1 = "Dear Maria Garcia, your IBAN DE89370400440532013000 is linked to Blue Harbor Bank."
G2 = "Wire from Golden Anchor Marine Insurance Company, IP 192.0.2.1, on 2024-01-05."


def default_gretel_rows() -> list[dict]:
    return [
        gretel_row(35255, G1, [
            at(G1, "Maria Garcia", "name"), at(G1, "DE89370400440532013000", "iban"),
            at(G1, "Blue Harbor Bank", "company"),
        ]),
        gretel_row(35135, G2, [
            at(G2, "Golden Anchor", "company"),                           # overlap, inner
            at(G2, "Golden Anchor Marine Insurance Company", "company"),
            at(G2, "192.0.2.1", "ipv4"), at(G2, "2024-01-05", "date"),
        ]),
    ]
