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

    run_parser = sub.add_parser("run", help="run an offline eval (experiment or trace-replay)")
    run_parser.add_argument("--config", required=True, help="path to the agent's YAML config")
    run_parser.add_argument("--mode", choices=["experiment", "replay"], default="experiment")
    run_parser.add_argument("--traces", default=None,
                            help="replay mode: raw traces JSONL (adapter-shaped lines)")
    run_parser.add_argument("--k", type=int, default=None, help="override repeats (pass^k)")
    run_parser.add_argument("--out", default="runs", help="output directory for run artifacts")
    run_parser.add_argument("--baselines", default=None,
                            help="baseline registry dir; enables regression gating (e.g. baselines/)")
    run_parser.add_argument(
        "--post-scores", action="store_true", help="also post scores to Langfuse"
    )

    promote_parser = sub.add_parser("promote-baseline",
                                    help="promote a completed run to be the agent's baseline")
    promote_parser.add_argument("--config", required=True)
    promote_parser.add_argument("--run", required=True, help="run directory (contains scores.jsonl)")
    promote_parser.add_argument("--baselines", default="baselines")

    push_parser = sub.add_parser(
        "push-rubrics",
        help="seed/update judge rubrics in Langfuse Prompt Management (version-continuous)",
    )
    push_parser.add_argument("--config", required=True)

    cal_parser = sub.add_parser("calibrate",
                                help="judge-vs-human agreement (kappa, pearson) for one metric")
    cal_parser.add_argument("--scores", required=True, help="scores.jsonl from a run")
    cal_parser.add_argument("--labels", required=True,
                            help="JSONL of human labels: {trace_id, name, value}")
    cal_parser.add_argument("--metric", required=True)
    cal_parser.add_argument("--pass-threshold", type=float, default=0.5,
                            help="binarization cutoff for kappa")

    sub.add_parser("list-evaluators", help="list registered evaluators")

    seed_parser = sub.add_parser(
        "seed-dataset", help="upload a local JSONL golden set to a Langfuse dataset (PII-redacted)"
    )
    seed_parser.add_argument("--config", required=True)
    seed_parser.add_argument("--from", dest="source", default=None,
                             help="JSONL of cases (default: the config's local_dataset)")
    seed_parser.add_argument("--name", default=None,
                             help="dataset name (default: the config's langfuse_dataset)")

    args = parser.parse_args(argv)

    if args.command == "list-evaluators":
        import agent_evals.evaluators  # noqa: F401
        from agent_evals.core.evaluator import available_evaluators

        for name in available_evaluators():
            print(name)
        return 0

    if args.command == "calibrate":
        return _calibrate(args)

    from agent_evals.core.config import load_config
    from agent_evals.runner import run_offline

    cfg = load_config(args.config)

    if args.command == "push-rubrics":
        import agent_evals.evaluators  # noqa: F401
        from agent_evals.core.evaluator import get_evaluator_class
        from agent_evals.core.langfuse_client import LangfuseClient

        client = LangfuseClient()
        pushed = 0
        for spec in cfg.evaluators:
            cls = get_evaluator_class(spec.name)
            rubric_text = getattr(cls, "rubric_text", None)
            if not (cls and cls.requires_judge and rubric_text):
                continue
            rubric_name = spec.params.get("rubric_name", cls.rubric_name)
            client.push_prompt(rubric_name, rubric_text, cls.rubric_version)
            print(f"pushed rubric '{rubric_name}' (version pinned: {cls.rubric_version})")
            pushed += 1
        client.flush()
        if not pushed:
            print("no judge rubrics found in this config")
        return 0

    if args.command == "promote-baseline":
        from agent_evals.baselines import promote_baseline

        baseline = promote_baseline(args.baselines, cfg.agent, args.run)
        print(f"promoted {baseline['run_id']} as baseline for {cfg.agent} "
              f"({len(baseline['metrics'])} metrics) -> {args.baselines}/{cfg.agent}.json")
        return 0

    if args.command == "seed-dataset":
        import json

        from agent_evals.core.langfuse_client import LangfuseClient
        from agent_evals.core.redaction import redact
        from agent_evals.core.schemas import Case

        name = args.name or cfg.langfuse_dataset
        if not name:
            print("error: no dataset name (--name or config langfuse_dataset)")
            return 2
        source = args.source or cfg.local_dataset
        if not source:
            print("error: no source JSONL (--from or config local_dataset)")
            return 2
        with open(cfg.resolve_path(source) if args.source is None else source) as f:
            cases = [Case(**json.loads(line)) for line in f if line.strip()]
        for case in cases:  # redaction before anything leaves the machine (note 05 §2)
            case.input = redact(case.input)
            case.expected_output = redact(case.expected_output)
        client = LangfuseClient()
        n = client.seed_dataset(name, cases)
        client.flush()
        print(f"seeded {n} redacted cases into Langfuse dataset '{name}'")
        return 0
    result = run_offline(
        cfg,
        k=args.k,
        out_dir=args.out,
        post_to_langfuse=args.post_scores,
        mode=args.mode,
        traces_path=args.traces,
        baselines_dir=args.baselines,
    )

    print(f"run:       {result.run_id} ({args.mode})")
    print(f"agent:     {result.agent}")
    print(f"cases:     {result.n_cases} x k={result.k}")
    for metric, cmp in result.baseline_comparison.items():
        lo, hi = cmp["ci"]
        tag = "REGRESSION" if cmp["regression"] else "ok"
        print(f"  vs baseline {cmp['baseline_run']} — {metric}: "
              f"{cmp['mean_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] {tag}")
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


def _calibrate(args) -> int:
    """Join judge scores to human labels by trace_id and report agreement.
    Target: kappa > 0.8 before trusting the judge (master plan §7)."""
    import json

    from agent_evals.core.stats import cohens_kappa, pearson

    def load(path):
        rows = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("name") == args.metric:
                        rows[row["trace_id"]] = float(row["value"])
        return rows

    judge_scores, human_labels = load(args.scores), load(args.labels)
    common = sorted(set(judge_scores) & set(human_labels))
    if len(common) < 2:
        print(f"error: only {len(common)} overlapping trace_ids for metric '{args.metric}'")
        return 2

    judge_vals = [judge_scores[t] for t in common]
    human_vals = [human_labels[t] for t in common]
    cutoff = args.pass_threshold
    kappa = cohens_kappa([int(v >= cutoff) for v in judge_vals],
                         [int(v >= cutoff) for v in human_vals])
    r = pearson(judge_vals, human_vals)

    print(f"metric:   {args.metric}   n={len(common)}")
    print(f"kappa:    {kappa:.3f} (binarized at {cutoff}; target > 0.8)")
    print(f"pearson:  {r:.3f}")
    verdict = "CALIBRATED" if kappa > 0.8 else "NOT calibrated — do not gate on this judge yet"
    print(f"verdict:  {verdict}")
    return 0 if kappa > 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
