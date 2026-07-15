# Importing these modules registers the built-in evaluators.
import agent_evals.evaluators.deterministic.exact_match  # noqa: F401
import agent_evals.evaluators.deterministic.json_schema_valid  # noqa: F401
import agent_evals.evaluators.deterministic.label_match  # noqa: F401
import agent_evals.evaluators.deterministic.thresholds  # noqa: F401
import agent_evals.evaluators.deterministic.tool_called  # noqa: F401
import agent_evals.evaluators.trajectory.efficiency  # noqa: F401
import agent_evals.evaluators.trajectory.tool_selection  # noqa: F401
import agent_evals.evaluators.llm_judge.goal_success  # noqa: F401
