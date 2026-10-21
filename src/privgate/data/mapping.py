"""Explicit raw-label -> EntityType maps, one per dataset.

Every raw label observed in the pinned files is listed (55 for Nemotron-PII, 29 for the
Gretel finance set). Listing them, instead of falling back silently, means a new upstream
label shows up as an error in strict mode and as a counted ``unknown_labels`` entry
otherwise, never as a quiet ``OTHER``.

Mapping rules (docs/DATA.md explains the borderline cases):

* PERSON: personal-name parts (first_name, last_name, full name).
* ID_NUMBER: identifiers issued to one person or to their account or record (government,
  tax, medical, insurance, customer, employee, bank account, licence, vehicle).
* OTHER: everything else, notably attributes (age, gender, occupation, ...), credentials
  (password, PIN, CVV, API key), identifiers of institutions rather than people (bank
  routing number, SWIFT/BIC), technical identifiers (MAC address, device id, cookie),
  user names and times of day.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from privgate.spans import EntityType

T = EntityType


class UnknownLabelError(KeyError):
    """A raw label that the dataset's map does not list (raised in strict mode)."""


def _flatten(groups: dict[EntityType, tuple[str, ...]]) -> Mapping[str, EntityType]:
    flat: dict[str, EntityType] = {}
    for entity, labels in groups.items():
        for label in labels:
            if label in flat:
                raise ValueError(f"raw label {label!r} mapped twice")
            flat[label] = entity
    return MappingProxyType(dict(sorted(flat.items())))


NEMOTRON_LABEL_MAP = _flatten({
    T.PERSON: ("first_name", "last_name"),
    T.EMAIL: ("email",),
    T.PHONE: ("phone_number", "fax_number"),
    T.CREDIT_CARD: ("credit_debit_card",),
    T.IP_ADDRESS: ("ipv4", "ipv6"),
    T.URL: ("url",),
    T.DATE: ("date", "date_time", "date_of_birth"),
    T.LOCATION: (
        "street_address", "city", "county", "state", "country", "postcode", "coordinate",
    ),
    T.ORGANIZATION: ("company_name",),
    T.ID_NUMBER: (
        "ssn", "national_id", "tax_id", "account_number", "customer_id", "employee_id",
        "medical_record_number", "health_plan_beneficiary_number",
        "certificate_license_number", "license_plate", "vehicle_identifier",
        "biometric_identifier", "unique_id",
    ),
    T.OTHER: (
        "age", "gender", "sexuality", "race_ethnicity", "religious_belief", "political_view",
        "blood_type", "language", "occupation", "education_level", "employment_status",
        "user_name", "password", "pin", "cvv", "api_key", "http_cookie", "mac_address",
        "device_identifier", "bank_routing_number", "swift_bic", "time",
    ),
})

GRETEL_LABEL_MAP = _flatten({
    T.PERSON: ("name", "first_name", "last_name"),
    T.EMAIL: ("email",),
    T.PHONE: ("phone_number",),
    T.CREDIT_CARD: ("credit_card_number",),
    T.IBAN: ("iban",),
    T.IP_ADDRESS: ("ipv4", "ipv6"),
    T.DATE: ("date", "date_time", "date_of_birth"),
    T.LOCATION: ("street_address", "local_latlng"),
    T.ORGANIZATION: ("company",),
    T.ID_NUMBER: (
        "ssn", "passport_number", "driver_license_number", "customer_id", "employee_id",
        "bban",
    ),
    T.OTHER: (
        "user_name", "password", "account_pin", "credit_card_security_code", "api_key",
        "bank_routing_number", "swift_bic_code", "time",
    ),
})

LABEL_MAPS: Mapping[str, Mapping[str, EntityType]] = MappingProxyType({
    "nemotron-pii": NEMOTRON_LABEL_MAP,
    "gretel-finance-en": GRETEL_LABEL_MAP,
})


def map_label(dataset: str, raw: str, *, strict: bool = False) -> EntityType | None:
    """Mapped type of ``raw`` in ``dataset``.

    Unknown labels raise :class:`UnknownLabelError` when ``strict``; otherwise ``None`` is
    returned so the caller can count the label before treating it as ``OTHER``.
    """
    try:
        table = LABEL_MAPS[dataset]
    except KeyError:
        raise ValueError(f"no label map for dataset {dataset!r}") from None
    if raw in table:
        return table[raw]
    if strict:
        raise UnknownLabelError(f"{dataset}: raw label {raw!r} is not in the label map")
    return None


def mapping_table_markdown() -> str:
    """The label-mapping table of docs/DATA.md, generated from the maps above."""
    lines = []
    for entity in EntityType:
        cells = []
        for table in LABEL_MAPS.values():
            raws = [f"`{r}`" for r, e in table.items() if e is entity]
            cells.append(", ".join(raws) if raws else "--")
        lines.append(f"| {entity.value} | " + " | ".join(cells) + " |")
    header = "| privgate type | " + " | ".join(LABEL_MAPS) + " |"
    rule = "|---" * (len(LABEL_MAPS) + 1) + "|"
    return "\n".join([header, rule, *lines])
