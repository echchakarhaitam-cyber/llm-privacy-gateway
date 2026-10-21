"""Which raw gold labels a detector targets: the basis of RQ3.

``supported_labels`` works on privgate's mapped types, which is too coarse for two of them:
OTHER and ID_NUMBER each group many raw labels (occupation, password, religious belief,
...; SSN, customer id, licence plate, ...), and a detector that emits OTHER for
nationalities is not a detector of passwords. So a raw label counts as targeted by a
detector when

* its mapped type is in the detector's ``supported_labels`` and is not one of the
  heterogeneous types (OTHER, ID_NUMBER), or
* one of the detector's native entity types is designed for it (:data:`NATIVE_RAW_TARGETS`,
  an explicit correspondence between native types and raw labels).

An ensemble targets what any member targets. The table is a documented judgement, listed in
docs/results-detection.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from privgate.data.mapping import LABEL_MAPS
from privgate.spans import EntityType

HETEROGENEOUS_TYPES = frozenset({EntityType.OTHER, EntityType.ID_NUMBER})

NATIVE_RAW_TARGETS: Mapping[str, Mapping[str, frozenset[str]]] = MappingProxyType({
    "regex": MappingProxyType({}),
    "spacy": MappingProxyType({
        "TIME": frozenset({"time"}),
        "NORP": frozenset({"race_ethnicity", "religious_belief", "political_view"}),
    }),
    "presidio": MappingProxyType({
        "DATE_TIME": frozenset({"time"}),
        "NRP": frozenset({"race_ethnicity", "religious_belief", "political_view"}),
        "MAC_ADDRESS": frozenset({"mac_address"}),
        "US_SSN": frozenset({"ssn"}),
        "US_PASSPORT": frozenset({"passport_number"}),
        "US_DRIVER_LICENSE": frozenset({"driver_license_number"}),
        "US_BANK_NUMBER": frozenset({"account_number", "bban"}),
        "US_ITIN": frozenset({"tax_id"}),
    }),
})


def native_targets(member: str) -> frozenset[str]:
    try:
        table = NATIVE_RAW_TARGETS[member]
    except KeyError:
        raise ValueError(f"no raw-label support table for detector {member!r}") from None
    return frozenset().union(*table.values())


def targeted_raw_labels(
    members: Iterable[str], supported_labels: Iterable[EntityType], dataset: str
) -> frozenset[str]:
    """Raw labels of ``dataset`` targeted by a detector made of base detectors ``members``."""
    supported = frozenset(supported_labels)
    explicit = frozenset().union(*(native_targets(m) for m in members))
    return frozenset(
        raw for raw, mapped in LABEL_MAPS[dataset].items()
        if (mapped in supported and mapped not in HETEROGENEOUS_TYPES) or raw in explicit
    )


def untargeted_by_all(
    detectors: Mapping[str, tuple[Iterable[str], Iterable[EntityType]]], dataset: str
) -> frozenset[str]:
    """Raw labels that NO detector targets; ``detectors`` maps name -> (members, labels)."""
    targeted = frozenset().union(*(
        targeted_raw_labels(members, labels, dataset) for members, labels in detectors.values()
    ))
    return frozenset(LABEL_MAPS[dataset]) - targeted
