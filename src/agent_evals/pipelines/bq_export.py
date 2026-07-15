"""Scores -> BigQuery (master plan §6): Langfuse for per-trace drill-down,
BigQuery for long-horizon trends and per-agent/per-version slicing.
Run nightly (Temporal Schedule) or ad hoc after a run."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = [
    ("name", "STRING"), ("value", "FLOAT64"), ("comment", "STRING"),
    ("level", "STRING"), ("trace_id", "STRING"), ("case_id", "STRING"),
    ("repeat_index", "INT64"), ("metadata", "JSON"),
]


def export_scores_to_bq(scores_jsonl: str | Path, table: str, project: str | None = None) -> int:
    """Load a run's scores.jsonl into `dataset.table` (created if missing).
    Requires: pip install google-cloud-bigquery, plus ADC credentials."""
    try:
        from google.cloud import bigquery
    except ImportError as e:  # pragma: no cover
        raise ImportError("BigQuery export requires: pip install google-cloud-bigquery") from e

    client = bigquery.Client(project=project)
    rows = []
    with open(scores_jsonl) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["metadata"] = json.dumps(row.get("metadata") or {})
                rows.append(row)

    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(n, t) for n, t in SCHEMA],
        write_disposition="WRITE_APPEND",
    )
    job = client.load_table_from_json(rows, table, job_config=job_config)
    job.result()
    return len(rows)
