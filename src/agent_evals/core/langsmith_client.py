"""LangSmith trace store — registered as "langsmith".

Pair with `trace_adapter: langsmith-generic` (runs have a different raw
shape than Langfuse observations). Credentials via the standard env vars
(LANGSMITH_API_KEY / LANGSMITH_ENDPOINT).

Platform mapping:
- golden dataset      -> LangSmith dataset examples
- Score               -> feedback (key=name, score=value) on the run
- prompt management   -> LangSmith prompts; rubric_version pinned in tags
  as "rubric_version=<v>" (note 09 §8: same version string, new storage)
- annotation queue    -> annotation queue matched by name, with the same
  needs_annotation degradation as Langfuse when queues are unavailable
"""

from __future__ import annotations

from agent_evals.core.schemas import Case, Score
from agent_evals.core.store import TraceStore, register_store

_VERSION_TAG = "rubric_version="


@register_store
class LangSmithClient(TraceStore):
    name = "langsmith"

    def __init__(self) -> None:
        try:
            from langsmith import Client
        except ImportError as e:
            raise ImportError(
                "LangSmith features require the 'langsmith' extra: "
                "pip install 'agent-evals[langsmith]'"
            ) from e
        self._ls = Client()

    def load_dataset(self, name: str) -> list[Case]:
        cases = []
        for example in self._ls.list_examples(dataset_name=name):
            meta = dict(example.metadata or {})
            outputs = example.outputs or {}
            cases.append(
                Case(
                    case_id=str(example.id),
                    input=example.inputs,
                    expected_output=outputs.get("expected_output", outputs or None),
                    expected_labels=meta.get("expected_labels", {}),
                    expected_tools=meta.get("expected_tools", []),
                    metadata=meta,
                )
            )
        return cases

    def post_score(self, score: Score) -> None:
        self._ls.create_feedback(
            run_id=score.metadata.get("source_trace_id") or score.trace_id,
            key=score.name,
            score=score.value,
            comment=score.comment or None,
            extra=score.metadata or None,
        )

    def seed_dataset(self, name: str, cases: list[Case]) -> int:
        try:
            dataset = self._ls.create_dataset(dataset_name=name)
        except Exception:
            dataset = self._ls.read_dataset(dataset_name=name)
        self._ls.create_examples(
            dataset_id=dataset.id,
            inputs=[c.input if isinstance(c.input, dict) else {"input": c.input}
                    for c in cases],
            outputs=[{"expected_output": c.expected_output} for c in cases],
            metadata=[
                {
                    "expected_labels": c.expected_labels,
                    "expected_tools": c.expected_tools,
                    "source_case_id": c.case_id,
                    **c.metadata,
                }
                for c in cases
            ],
        )
        return len(cases)

    def get_prompt(self, name: str) -> tuple[str, int, dict]:
        commit = self._ls.pull_prompt_commit(name)
        text = _extract_prompt_text(commit.manifest)
        prompt = self._ls.get_prompt(name)
        config = {}
        for tag in prompt.tags or []:
            if tag.startswith(_VERSION_TAG):
                config["rubric_version"] = tag[len(_VERSION_TAG):]
        # numeric version: LangSmith identifies commits by hash, not counter
        return text, commit.commit_hash[:8], config

    def push_prompt(self, name: str, text: str, rubric_version: str) -> None:
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as e:
            raise ImportError(
                "LangSmith prompt management requires langchain-core "
                "(pip install langchain-core), or keep rubrics as code "
                "constants with rubric_from_store: false"
            ) from e
        # template_format="mustache": rubric text uses {trace}/{case} braces
        # that must not be parsed as f-string variables
        prompt = ChatPromptTemplate.from_messages(
            [("user", text)], template_format="mustache"
        )
        self._ls.push_prompt(name, object=prompt, tags=[f"{_VERSION_TAG}{rubric_version}"])

    def fetch_trace_raw(self, trace_id: str) -> dict:
        run = self._ls.read_run(trace_id, load_child_runs=True)
        as_dict = run.dict() if hasattr(run, "dict") else dict(run)
        children = as_dict.pop("child_runs", None) or []
        children = [c.dict() if hasattr(c, "dict") else dict(c) for c in children]
        return {"run": as_dict, "child_runs": children}

    def enqueue_annotation(self, trace_id: str, queue_name: str, reason: str) -> bool:
        try:
            queue = next(self._ls.list_annotation_queues(name=queue_name))
            self._ls.add_runs_to_annotation_queue(queue.id, run_ids=[trace_id])
            return True
        except Exception:
            self.post_score(Score(
                name="needs_annotation", value=1.0, trace_id=trace_id,
                comment=f"[{queue_name}] {reason}", level="session",
            ))
            return False


def _extract_prompt_text(manifest: dict) -> str:
    """Pull the raw template string out of a LangSmith prompt-commit
    manifest (a serialized langchain prompt)."""
    kwargs = (manifest or {}).get("kwargs", {})
    if "template" in kwargs:
        return kwargs["template"]
    for message in kwargs.get("messages", []):
        inner = message.get("kwargs", {}).get("prompt", {}).get("kwargs", {})
        if "template" in inner:
            return inner["template"]
    raise ValueError("Could not extract template text from LangSmith prompt manifest")
