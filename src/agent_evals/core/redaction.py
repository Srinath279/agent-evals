"""PII redaction (note 05 §2) — applied before anything leaves the trace
store: judge payloads now; dataset promotion in Phase 1.

Baseline regex pass. Swap in Cloud DLP behind the same functions when
compliance requires it; callers must not care which engine runs.
"""

from __future__ import annotations

import re
from typing import Any

# order matters: more specific patterns run first so the greedy phone
# pattern can't consume pieces of IPs/SSNs/cards
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP_ADDRESS>"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "<CARD_NUMBER>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    (re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]\d{3,4}[ .-]?\d{0,4}\b"), "<PHONE>"),
]


def redact_text(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(obj: Any) -> Any:
    """Recursively redact strings inside dicts/lists; other types pass through."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj
