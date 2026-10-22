"""Per-entity-type policy: what the gateway does when a detector finds a given type.

``mask`` replaces the value by a typed placeholder and lets the request through; ``block``
refuses the whole request (403) because the operator decided that this type of data must
never be sent upstream, not even masked. The default for every type is ``mask``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from privgate.spans import EntityType

MASK = "mask"
BLOCK = "block"
ACTIONS = (MASK, BLOCK)


@dataclass(frozen=True)
class Policy:
    actions: Mapping[EntityType, str] = field(default_factory=dict)
    default_action: str = MASK

    def __post_init__(self) -> None:
        if self.default_action not in ACTIONS:
            raise ValueError(f"default_action must be one of {ACTIONS}, got "
                             f"{self.default_action!r}")
        checked = {}
        for label, action in dict(self.actions).items():
            if action not in ACTIONS:
                raise ValueError(f"action for {label!r} must be one of {ACTIONS}, got {action!r}")
            checked[EntityType(label)] = action
        object.__setattr__(self, "actions", MappingProxyType(checked))

    def action(self, label: EntityType) -> str:
        return self.actions.get(label, self.default_action)

    def blocked(self, labels: Iterable[EntityType]) -> list[EntityType]:
        """The distinct labels among ``labels`` whose action is ``block``, sorted by name."""
        return sorted({lab for lab in labels if self.action(lab) == BLOCK},
                      key=lambda lab: lab.value)

    @classmethod
    def blocking(cls, labels: Iterable[EntityType | str]) -> Policy:
        """Mask everything except ``labels``, which block the request."""
        return cls({EntityType(lab): BLOCK for lab in labels})
