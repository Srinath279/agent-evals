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


class AnthropicJudge(BaseJudge):
    """Judge via the Anthropic API. Forces a structured verdict through a
    required tool call so parsing never depends on prose style."""

    provider = "anthropic"

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
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{rubric}\n\n{payload}"}],
            tools=[self._VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "submit_verdict"},
        )
        block = next(b for b in msg.content if b.type == "tool_use")
        return Verdict(**block.input)


def make_judge(cfg: JudgeConfig) -> BaseJudge:
    if cfg.provider == "mock":
        return MockJudge()
    if cfg.provider == "anthropic":
        return AnthropicJudge(model=cfg.model)
    if cfg.provider in ("vertex", "openai"):
        raise NotImplementedError(
            f"Judge provider '{cfg.provider}' lands in Phase 2 (see master plan §7); "
            "use 'anthropic' or 'mock' in the Phase 0 slice."
        )
    raise ValueError(f"Unknown judge provider: {cfg.provider}")
