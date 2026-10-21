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
from privgate.data.splits import (
    EVALUATION_SETS,
    assign_split,
    build_evaluation_sets,
    read_jsonl,
    sample,
    write_jsonl,
)
from privgate.data.stats import SetStats

__all__ = [
    "EVALUATION_SETS",
    "LABEL_MAPS",
    "Document",
    "LoadOptions",
    "LoadReport",
    "RawSpan",
    "SetStats",
    "UnknownLabelError",
    "assign_split",
    "build_document",
    "build_evaluation_sets",
    "load_gretel",
    "load_nemotron",
    "map_label",
    "merge_adjacent",
    "read_jsonl",
    "sample",
    "write_jsonl",
]
