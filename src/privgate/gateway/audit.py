"""Privacy-safe audit log: one JSON line per request, built only from an allowlist of fields.

What is recorded: request id, UTC timestamp, decision, HTTP status, per-label COUNTS of masked
entities, the detector names, latency per stage, and the CLASS NAME of the error that caused a
refusal. What is never recorded: message text, entity values, placeholders, the mapping, or
exception messages (a detector's exception message may quote the text it choked on).

The record type enforces this instead of relying on discipline at the call sites: counts
keys must be entity-type names, the error class must look like an identifier, everything
else is a number or a fixed vocabulary word.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from privgate.spans import EntityType

logger = logging.getLogger("privgate.gateway.audit")

DECISIONS = ("served", "served_unscreened", "blocked_policy", "rejected_invalid",
             "screening_unavailable", "upstream_error", "internal_error")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,79}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_LABELS = frozenset(e.value for e in EntityType)


@dataclass(frozen=True)
class AuditRecord:
    request_id: str
    decision: str
    status: int
    counts: Mapping[str, int] = field(default_factory=dict)
    detectors: tuple[str, ...] = ()
    latency_ms: Mapping[str, float] = field(default_factory=dict)
    error_class: str | None = None
    fail_open: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")
        if not _REQUEST_ID.fullmatch(self.request_id):
            raise ValueError("request_id must be a short token")
        if self.error_class is not None and not _IDENTIFIER.fullmatch(self.error_class):
            raise ValueError("error_class must be an identifier")
        for label, n in self.counts.items():
            if label not in _LABELS or isinstance(n, bool) or not isinstance(n, int):
                raise ValueError("counts must map entity-type names to integers")
        for name in self.detectors:
            if not _IDENTIFIER.fullmatch(name.replace("+", "_").replace("-", "_")):
                raise ValueError("detector names must be identifiers (joined by '+')")
        return {
            "request_id": self.request_id,
            "ts": self.timestamp or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "decision": self.decision,
            "status": int(self.status),
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
            "detectors": list(self.detectors),
            "latency_ms": {k: round(float(v), 3) for k, v in self.latency_ms.items()},
            "error_class": self.error_class,
            "fail_open": bool(self.fail_open),
        }


class AuditLog:
    """Appends records to ``path`` (JSON lines); ``keep=True`` also keeps them in memory."""

    def __init__(self, path: str | Path | None = None, keep: bool = False):
        self.path = Path(path) if path else None
        self.records: list[dict] | None = [] if keep else None
        self._lock = threading.Lock()
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: AuditRecord) -> None:
        """Never raises: a full disk must not turn a screened request into an error."""
        try:
            data = record.to_dict()
            line = json.dumps(data, sort_keys=True, ensure_ascii=True)
            with self._lock:
                if self._fh:
                    self._fh.write(line + "\n")
                    self._fh.flush()
                if self.records is not None:
                    self.records.append(data)
        except Exception as exc:           # noqa: BLE001 - class name only, never the record
            logger.warning("audit write failed: %s", type(exc).__name__)

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
