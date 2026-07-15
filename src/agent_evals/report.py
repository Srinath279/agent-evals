"""Markdown run report — the artifact linked in PRs / stored in GCS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_evals.core.config import AgentConfig
    from agent_evals.runner import RunResult


def render_report(result: "RunResult", cfg: "AgentConfig") -> str:
    lines = [
        f"# Eval run — {result.agent}",
        "",
        f"- **Run ID**: `{result.run_id}`",
        f"- **Cases**: {result.n_cases} × k={result.k} repeats = {len(result.case_results)} executions",
        f"- **Pass rate** (per execution): {result.pass_rate:.1%}",
        f"- **pass^{result.k}** (per case, all repeats): {result.pass_k_rate:.1%}",
        f"- **Gate**: {'✅ PASSED' if result.gate_passed else '❌ FAILED'}",
        "",
    ]
    if result.gate_failures:
        lines += ["## Gate failures", ""]
        lines += [f"- {failure}" for failure in result.gate_failures]
        lines.append("")

    lines += ["## Metrics", "", "| metric | mean | threshold | status |", "|---|---|---|---|"]
    for name, mean in result.metric_means.items():
        threshold = cfg.score_thresholds.get(name)
        if threshold is None:
            lines.append(f"| {name} | {mean:.3f} | — | tracked |")
        else:
            status = "✅" if mean >= threshold else "❌"
            lines.append(f"| {name} | {mean:.3f} | {threshold} | {status} |")
    lines.append("")

    failing = [cr for cr in result.case_results if not cr.passed]
    if failing:
        lines += ["## Failing executions", ""]
        for cr in failing:
            lines.append(f"### `{cr.case_id}` (repeat {cr.repeat_index})")
            lines += [f"- {failure}" for failure in cr.failures]
            lines.append("")
    return "\n".join(lines)
