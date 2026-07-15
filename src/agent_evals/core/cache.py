"""Score cache — the idempotency layer (notes 05 §A3, 08 rule 3).

Keyed by (trace_id, evaluator, rubric_version, judge_provider, judge_model)
so retries — local reruns or at-least-once Temporal activities — never
double-spend judge calls or double-write scores.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

from agent_evals.core.schemas import Score


def score_cache_key(
    trace_id: str,
    evaluator: str,
    rubric_version: str = "n/a",
    judge_provider: str = "none",
    judge_model: str = "none",
) -> str:
    raw = "|".join([trace_id, evaluator, rubric_version, judge_provider, judge_model])
    return hashlib.sha256(raw.encode()).hexdigest()


class ScoreCache:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score_json TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Score]:
        row = self._conn.execute("SELECT score_json FROM scores WHERE key = ?", (key,)).fetchone()
        return Score(**json.loads(row[0])) if row else None

    def put(self, key: str, score: Score) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO scores (key, score_json) VALUES (?, ?)",
            (key, score.model_dump_json()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
