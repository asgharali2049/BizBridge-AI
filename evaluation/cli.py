"""Explicit live evaluation CLI; importing it never initializes an LLM provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from agent.config import AgentConfig
from agent.errors import AgentError
from evaluation.dataset import load_evaluation_cases
from evaluation.runner import run_evaluations, write_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_target(value: str, base_config: AgentConfig) -> AgentConfig:
    provider, separator, model = value.partition(":")
    if not separator or not provider.strip() or not model.strip():
        raise argparse.ArgumentTypeError(
            "targets must use PROVIDER:MODEL, for example ollama:beam:cloud"
        )
    environment = dict(os.environ)
    environment["LLM_PROVIDER"] = provider.strip()
    environment["LLM_MODEL"] = model.strip()
    selected = AgentConfig.from_env(environ=environment)
    return replace(
        selected,
        max_iterations=base_config.max_iterations,
        temperature=base_config.temperature,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the explicit, live BizBridge AI model evaluation."
    )
    parser.add_argument(
        "--target",
        action="append",
        metavar="PROVIDER:MODEL",
        help=(
            "provider/model target; repeat to compare models. Without this option, "
            "uses LLM_PROVIDER and LLM_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="evaluate only this case ID; may be supplied multiple times",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation_report.json",
        help="JSON report output path (default: project evaluation_report.json)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list database-grounded cases without loading credentials or calling providers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cases = load_evaluation_cases()
        if args.list_cases:
            print(
                json.dumps(
                    [
                        {
                            "case_id": case.case_id,
                            "question": case.question,
                            "expected_tools": [
                                call.name for call in case.expected_tool_calls
                            ],
                        }
                        for case in cases
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        base_config = AgentConfig.from_env()
        targets = (
            [_parse_target(value, base_config) for value in args.target]
            if args.target
            else [base_config]
        )
        if args.case_ids:
            requested = set(args.case_ids)
            known = {case.case_id for case in cases}
            unknown = requested - known
            if unknown:
                parser.error(f"Unknown evaluation case ID(s): {', '.join(sorted(unknown))}")
            cases = [case for case in cases if case.case_id in requested]
        report = asyncio.run(run_evaluations(targets, cases))
        output_path = write_report(report, args.output)
    except (AgentError, ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Evaluation report saved to {output_path}")
    print(f"Cases per model: {report['case_count']}")
    for model in report["models"]:
        summary = model["summary"]
        print(
            f"{model['provider']}:{model['model']} "
            f"accuracy={summary['accuracy']:.1%} "
            f"tool_selection={summary['tool_selection_accuracy']:.1%} "
            f"numerical={summary['numerical_accuracy']:.1%} "
            f"latency={summary['mean_latency_seconds']:.2f}s "
            f"failure_rate={summary['failure_rate']:.1%}"
        )
    for comparison in report["comparisons"]:
        deltas = comparison["delta_comparison_minus_baseline"]
        print(
            f"{comparison['comparison']} vs {comparison['baseline']}: "
            f"accuracy {deltas['accuracy']:+.1%}, "
            f"tool selection {deltas['tool_selection_accuracy']:+.1%}, "
            f"numerical {deltas['numerical_accuracy']:+.1%}, "
            f"latency {deltas['latency_seconds']:+.2f}s, "
            f"failure rate {deltas['failure_rate']:+.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
