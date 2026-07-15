"""Multi-provider JudgeClient (master plan §7).

Evaluators never import a provider SDK; they call
`judge.verdict(rubric, payload) -> Verdict` and get a structured
reasoning-before-score result on every provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from pydantic import BaseModel, Field

from agent_evals.core.config import JudgeConfig

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator of AI agent behavior. "
    "Follow the rubric exactly. Always give your reasoning first, then the score. "
    "Judge only what the evidence supports; an agent claiming success without "
    "tool evidence must be scored as if it failed."
)


class Verdict(BaseModel):
    reasoning: str
    score: float = Field(ge=0.0, le=1.0)


class BaseJudge(ABC):
    provider: str = ""
    model: str = ""

    def __init__(self) -> None:
        self.calls = 0  # judge invocations actually made (cache hits don't count)

    @abstractmethod
    def _verdict(self, rubric: str, payload: str) -> Verdict: ...

    def verdict(self, rubric: str, payload: str) -> Verdict:
        self.calls += 1
        return self._verdict(rubric, payload)


class MockJudge(BaseJudge):
    """Deterministic judge for tests and local dry-runs — never spends money."""

    provider = "mock"
    model = "mock-judge"

    def __init__(self, verdict_fn: Optional[Callable[[str, str], Verdict]] = None) -> None:
        super().__init__()
        self._fn = verdict_fn or (lambda rubric, payload: Verdict(reasoning="mock verdict", score=1.0))

    def _verdict(self, rubric: str, payload: str) -> Verdict:
        return self._fn(rubric, payload)


_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the evaluation verdict. Reasoning first, then score.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Step-by-step evaluation reasoning."},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["reasoning", "score"],
    },
}


def _anthropic_tool_verdict(client, model: str, rubric: str, payload: str) -> Verdict:  # pragma: no cover - network
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{rubric}\n\n{payload}"}],
        tools=[_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return Verdict(**block.input)


class AnthropicJudge(BaseJudge):
    """Judge via the Anthropic API. Forces a structured verdict through a
    required tool call so parsing never depends on prose style."""

    provider = "anthropic"

    def __init__(self, model: str) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Anthropic judge requires the 'anthropic' extra: pip install 'agent-evals[anthropic]'"
            ) from e
        self.model = model
        self._client = anthropic.Anthropic()

    def _verdict(self, rubric: str, payload: str) -> Verdict:  # pragma: no cover - network
        return _anthropic_tool_verdict(self._client, self.model, rubric, payload)


class VertexJudge(BaseJudge):
    """Claude on Vertex AI (GCP ADC auth, no API key). Same bare model IDs
    as the Anthropic API; Anthropic's Batches API is NOT available here —
    batch scoring must use Vertex batch prediction (master plan §7)."""

    provider = "vertex"

    def __init__(self, model: str, project_id: str | None = None, region: str | None = None) -> None:
        super().__init__()
        try:
            from anthropic import AnthropicVertex
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Vertex judge requires the 'anthropic' extra: pip install 'agent-evals[anthropic]'"
            ) from e
        self.model = model
        kwargs = {}
        if project_id:
            kwargs["project_id"] = project_id
        if region:
            kwargs["region"] = region
        self._client = AnthropicVertex(**kwargs)  # falls back to env/ADC

    def _verdict(self, rubric: str, payload: str) -> Verdict:  # pragma: no cover - network
        return _anthropic_tool_verdict(self._client, self.model, rubric, payload)


class OpenAIJudge(BaseJudge):
    """Judge via the OpenAI API, structured verdict through a forced
    function call."""

    provider = "openai"

    def __init__(self, model: str) -> None:
        super().__init__()
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenAI judge requires: pip install openai") from e
        self.model = model
        self._client = openai.OpenAI()

    def _verdict(self, rubric: str, payload: str) -> Verdict:  # pragma: no cover - network
        import json
        import re

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": JUDGE_SYSTEM_PROMPT
                 + ' Respond by calling submit_verdict, or with ONLY a JSON object '
                   '{"reasoning": "...", "score": 0.0-1.0}.'},
                {"role": "user", "content": f"{rubric}\n\n{payload}"},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "submit_verdict",
                    "description": _VERDICT_TOOL["description"],
                    "parameters": _VERDICT_TOOL["input_schema"],
                },
            }],
            tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
        )
        message = resp.choices[0].message
        if message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            return Verdict(**args)
        # OpenAI-compatible local servers (Ollama, vLLM) don't always honor a
        # forced tool_choice — fall back to parsing a JSON object from content
        match = re.search(r"\{.*\}", message.content or "", re.DOTALL)
        if not match:
            raise ValueError(f"judge returned neither tool call nor JSON: {message.content!r:.200}")
        data = json.loads(match.group(0))
        return Verdict(reasoning=str(data.get("reasoning", "")),
                       score=max(0.0, min(1.0, float(data["score"]))))


class FallbackJudge(BaseJudge):
    """Degrade to a second provider on primary outage/rate-limit (master
    plan §7). provider/model always reflect whichever judge actually
    produced the verdict, so score stamps stay truthful. Caveat (note 09):
    a failover mid-run changes cache keys for subsequent retries."""

    def __init__(self, primary: BaseJudge, fallback: BaseJudge) -> None:
        super().__init__()
        self._primary = primary
        self._fallback = fallback
        self._last = primary

    @property
    def provider(self) -> str:  # type: ignore[override]
        return self._last.provider

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._last.model

    def _verdict(self, rubric: str, payload: str) -> Verdict:
        try:
            self._last = self._primary
            return self._primary.verdict(rubric, payload)
        except Exception:
            self._last = self._fallback
            return self._fallback.verdict(rubric, payload)


class BudgetExceededError(RuntimeError):
    """Raised when the judge's daily budget is exhausted — a visible kill
    switch, never silent degradation (note 08 rule 6)."""


class BudgetedJudge(BaseJudge):
    """Enforces daily_budget_usd per (day, provider, model) in a sqlite
    counter. Accounting unit is est_cost_per_call_usd until real token
    costing lands; the cap errs conservative."""

    def __init__(self, inner: BaseJudge, daily_budget_usd: float,
                 est_cost_per_call_usd: float = 0.01,
                 db_path: str = "runs/judge_budget.sqlite3") -> None:
        super().__init__()
        import sqlite3
        from pathlib import Path

        self._inner = inner
        self._budget = daily_budget_usd
        self._est = est_cost_per_call_usd
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS spend (day TEXT, provider TEXT, model TEXT, usd REAL, "
            "PRIMARY KEY (day, provider, model))"
        )
        self._conn.commit()

    @property
    def provider(self) -> str:  # type: ignore[override]
        return self._inner.provider

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._inner.model

    def _key(self) -> tuple[str, str, str]:
        import datetime

        return (datetime.date.today().isoformat(), self._inner.provider, self._inner.model)

    def _spent(self) -> float:
        row = self._conn.execute(
            "SELECT usd FROM spend WHERE day=? AND provider=? AND model=?", self._key()
        ).fetchone()
        return row[0] if row else 0.0

    def _verdict(self, rubric: str, payload: str) -> Verdict:
        spent = self._spent()
        if spent + self._est > self._budget:
            raise BudgetExceededError(
                f"daily judge budget exhausted: spent ${spent:.2f} of ${self._budget:.2f} "
                f"({self.provider}/{self.model})"
            )
        verdict = self._inner.verdict(rubric, payload)
        day, provider, model = self._key()
        self._conn.execute(
            "INSERT INTO spend (day, provider, model, usd) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(day, provider, model) DO UPDATE SET usd = usd + ?",
            (day, provider, model, self._est, self._est),
        )
        self._conn.commit()
        return verdict


def _provider_judge(cfg: JudgeConfig) -> BaseJudge:
    if cfg.provider == "mock":
        return MockJudge()
    if cfg.provider == "anthropic":
        return AnthropicJudge(model=cfg.model)
    if cfg.provider == "vertex":
        return VertexJudge(model=cfg.model, project_id=cfg.project_id, region=cfg.region)
    if cfg.provider == "openai":
        return OpenAIJudge(model=cfg.model)
    raise ValueError(f"Unknown judge provider: {cfg.provider}")


def make_judge(cfg: JudgeConfig) -> BaseJudge:
    judge = _provider_judge(cfg)
    if cfg.fallback:
        judge = FallbackJudge(judge, _provider_judge(cfg.fallback))
    if cfg.daily_budget_usd and cfg.provider != "mock":
        judge = BudgetedJudge(
            judge,
            daily_budget_usd=cfg.daily_budget_usd,
            est_cost_per_call_usd=cfg.est_cost_per_call_usd,
            db_path=cfg.budget_db or "runs/judge_budget.sqlite3",
        )
    return judge
