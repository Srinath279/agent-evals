"""CLI — must always work locally without Temporal or Langfuse.

    evals run --config configs/support_agent.yaml
    evals list-evaluators
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description="agent-evals harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run an offline eval experiment")
    run_parser.add_argument("--config", required=True, help="path to the agent's YAML config")
    run_parser.add_argument("--k", type=int, default=None, help="override repeats (pass^k)")
    run_parser.add_argument("--out", default="runs", help="output directory for run artifacts")
    run_parser.add_argument(
        "--post-scores", action="store_true", help="also post scores to Langfuse"
    )

    sub.add_parser("list-evaluators", help="list registered evaluators")

    args = parser.parse_args(argv)

    if args.command == "list-evaluators":
        import agent_evals.evaluators  # noqa: F401
        from agent_evals.core.evaluator import available_evaluators

        for name in available_evaluators():
            print(name)
        return 0

    from agent_evals.core.config import load_config
    from agent_evals.runner import run_offline

    cfg = load_config(args.config)
    result = run_offline(cfg, k=args.k, out_dir=args.out, post_to_langfuse=args.post_scores)

    print(f"run:       {result.run_id}")
    print(f"agent:     {result.agent}")
    print(f"cases:     {result.n_cases} x k={result.k}")
    for name, mean in result.metric_means.items():
        threshold = cfg.score_thresholds.get(name)
        suffix = f" (threshold {threshold})" if threshold is not None else ""
        print(f"  {name}: {mean:.3f}{suffix}")
    print(f"pass rate: {result.pass_rate:.1%}   pass^{result.k}: {result.pass_k_rate:.1%}")
    print(f"artifacts: {result.out_dir}")
    if result.gate_passed:
        print("gate:      PASSED")
        return 0
    print("gate:      FAILED")
    for failure in result.gate_failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
