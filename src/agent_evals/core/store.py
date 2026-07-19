"""Trace-store abstraction — the platform counterpart of the adapter layer.

Adapters normalize trace *shape*; a TraceStore normalizes the *platform*
(Langfuse, LangSmith, ...): golden datasets, score write-back, prompt/rubric
management, trace fetching, and annotation queues. Every module that used to
import LangfuseClient by name goes through get_store(cfg.trace_store) instead,
so switching platforms is a config edit plus (if the trace shape differs) a
trace adapter.

Store classes register here but lazy-import their SDK inside __init__, so the
harness stays runnable with zero platform dependencies installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_evals.core.schemas import Case, Score

_STORES: dict[str, type["TraceStore"]] = {}


def register_store(cls: type["TraceStore"]) -> type["TraceStore"]:
    _STORES[cls.name] = cls
    return cls


def get_store(name: str) -> "TraceStore":
    if name not in _STORES:
        raise KeyError(f"Unknown trace store '{name}'. Available: {sorted(_STORES)}")
    return _STORES[name]()


def available_stores() -> list[str]:
    return sorted(_STORES)


class TraceStore(ABC):
    """The platform contract. A conforming store makes the whole harness —
    dataset loading, score posting, rubric management, online scoring, and
    the annotation flywheel — work against its platform."""

    name: str = ""

    @abstractmethod
    def load_dataset(self, name: str) -> list[Case]:
        """Fetch a golden dataset by name as canonical Cases."""

    @abstractmethod
    def post_score(self, score: Score) -> None:
        """Attach one Score to its trace (score.metadata.source_trace_id
        wins over score.trace_id — the agent's real platform trace)."""

    @abstractmethod
    def seed_dataset(self, name: str, cases: list[Case]) -> int:
        """Create/extend a dataset from cases. Callers redact PII first."""

    @abstractmethod
    def get_prompt(self, name: str) -> tuple[str, int, dict]:
        """(text, version, config) from the platform's prompt management.
        config carries the pinned rubric_version when present."""

    @abstractmethod
    def push_prompt(self, name: str, text: str, rubric_version: str) -> None:
        """Create/update a rubric, pinning rubric_version so cache keys
        survive the storage migration (note 09 §8)."""

    @abstractmethod
    def fetch_trace_raw(self, trace_id: str) -> dict:
        """Fetch one trace in the raw dict shape its paired trace adapter
        consumes (see core/adapters). Used by the online pipeline."""

    @abstractmethod
    def enqueue_annotation(self, trace_id: str, queue_name: str, reason: str) -> bool:
        """Push a trace to human review. Best-effort: on failure, degrade to
        a `needs_annotation` score so the signal is never dropped."""

    def flush(self) -> None:
        """Flush buffered writes; default no-op for synchronous clients."""


# self-registering store implementations (SDK imports stay lazy inside them)
import agent_evals.core.langfuse_client  # noqa: E402,F401
import agent_evals.core.langsmith_client  # noqa: E402,F401
