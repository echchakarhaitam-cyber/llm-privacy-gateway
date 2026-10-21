"""Public PII datasets loaded into the project's span model.

Importing this package does not need pyarrow; only reading parquet does (``.[data]``).
"""

from privgate.data.documents import (
    Document,
    LoadOptions,
    LoadReport,
    RawSpan,
    build_document,
    merge_adjacent,
)
from privgate.data.gretel import load_gretel
from privgate.data.mapping import LABEL_MAPS, UnknownLabelError, map_label
from privgate.data.nemotron import load_nemotron

__all__ = [
    "LABEL_MAPS",
    "Document",
    "LoadOptions",
    "LoadReport",
    "RawSpan",
    "UnknownLabelError",
    "build_document",
    "load_gretel",
    "load_nemotron",
    "map_label",
    "merge_adjacent",
]
