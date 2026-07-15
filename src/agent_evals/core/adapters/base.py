"""Trace adapter layer (note 05 §1 — 'the most underestimated piece').

Each agent framework logs a different trace shape into Langfuse; an
adapter maps that raw shape into the canonical Trace so evaluators never
contain agent-specific parsing. Onboarding agent N = YAML + adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_evals.core.schemas import Trace

_ADAPTERS: dict[str, type["TraceAdapter"]] = {}


def register_adapter(cls: type["TraceAdapter"]) -> type["TraceAdapter"]:
    _ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> "TraceAdapter":
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown trace adapter '{name}'. Available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]()


class TraceAdapter(ABC):
    name: str = ""

    @abstractmethod
    def to_trace(self, raw: dict[str, Any]) -> Trace:
        """Map one raw trace payload (trace + observations) to the canonical Trace."""
