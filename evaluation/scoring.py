"""Deterministic evaluation metrics; no LLM judge is used."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Sequence

from evaluation.dataset import EvaluationCase


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])-?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?!\w)")
_CLARIFICATION_MARKERS = (
    "which",
    "please specify",
    "please clarify",
    "could you clarify",
    "customer id",
    "customer ID",
    "what date",
    "valid date",
    "which one",
    "did you mean",
)
_REFUSAL_MARKERS = (
    "not available",
    "not recorded",
    "not in the database",
    "cannot determine",
    "don't record",
    "do not record",
    "does not contain",
    "can't provide",
    "cannot provide",
    "not tracked",
)


def _normalize_number(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace(",", "").rstrip("%")


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric claims while ignoring ISO date components."""
    without_dates = _ISO_DATE.sub(" ", text)
    return {
        _normalize_number(match.group(0))
        for match in _NUMBER.finditer(without_dates)
    }


def _arguments_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    return name, json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fact_coverage(answer: str, groups: Sequence[Sequence[str]]) -> float:
    if not groups:
        return 1.0
    normalized = " ".join(answer.casefold().split())
    covered = sum(
        any(" ".join(alternative.casefold().split()) in normalized for alternative in group)
        for group in groups
    )
    return covered / len(groups)


def _behavior_correct(case: EvaluationCase, answer: str, calls: Sequence[dict[str, Any]]) -> bool:
    lowered = answer.casefold()
    if case.behavior == "clarify":
        return not calls and any(marker.casefold() in lowered for marker in _CLARIFICATION_MARKERS)
    if case.behavior == "refuse":
        return not calls and any(marker in lowered for marker in _REFUSAL_MARKERS)
    if case.behavior == "answer_without_tools":
        return not calls and bool(answer.strip())
    return bool(answer.strip())


def score_case(
    case: EvaluationCase,
    actual_tool_calls: Sequence[dict[str, Any]],
    answer: str,
    *,
    execution_success: bool,
    error: str | None,
    latency_seconds: float,
    token_usage: dict[str, int] | None,
    tool_results: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare one run against its deterministic tool, fact, and numeric expectations."""
    expected_names = Counter(call.name for call in case.expected_tool_calls)
    actual_names = Counter(call.get("name", "") for call in actual_tool_calls)
    expected_arguments = Counter(
        _arguments_key(call.name, call.arguments) for call in case.expected_tool_calls
    )
    actual_arguments = Counter(
        _arguments_key(call.get("name", ""), call.get("arguments", {}))
        for call in actual_tool_calls
    )

    expected_call_count = sum(expected_names.values())
    actual_call_count = sum(actual_names.values())
    selected_correct = sum((expected_names & actual_names).values())
    arguments_correct = sum((expected_arguments & actual_arguments).values())
    tool_precision = selected_correct / actual_call_count if actual_call_count else (
        1.0 if expected_call_count == 0 else 0.0
    )
    tool_recall = selected_correct / expected_call_count if expected_call_count else 1.0
    exact_selection = expected_names == actual_names
    exact_arguments = expected_arguments == actual_arguments
    argument_denominator = max(expected_call_count, actual_call_count)

    answer_numbers = extract_numbers(answer)
    expected_numbers = {_normalize_number(value) for value in case.expected_numbers}
    numerical_accuracy = (
        sum(value in answer_numbers for value in expected_numbers) / len(expected_numbers)
        if expected_numbers
        else 1.0
    )
    evidence_text = " ".join(
        [
            case.question,
            *(json.dumps(call, ensure_ascii=False, sort_keys=True) for call in actual_tool_calls),
            *tool_results,
        ]
    )
    evidence_numbers = extract_numbers(evidence_text) | expected_numbers
    unsupported_numbers = answer_numbers - evidence_numbers
    unsupported_claim_rate = (
        len(unsupported_numbers) / len(answer_numbers) if answer_numbers else 0.0
    )
    fact_coverage = _fact_coverage(answer, case.required_fact_groups)
    behavior_correct = _behavior_correct(case, answer, actual_tool_calls)
    completed = (
        execution_success
        and error is None
        and exact_selection
        and exact_arguments
        and numerical_accuracy == 1.0
        and fact_coverage == 1.0
        and behavior_correct
    )
    return {
        "tool_selection_accuracy": float(exact_selection),
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "tool_argument_accuracy": (
            arguments_correct / argument_denominator
            if argument_denominator
            else 1.0
        ),
        "exact_tool_arguments": exact_arguments,
        "numerical_accuracy": numerical_accuracy,
        "required_fact_coverage": fact_coverage,
        "unsupported_claim_rate": unsupported_claim_rate,
        "unsupported_numeric_claims": sorted(unsupported_numbers),
        "behavior_correct": behavior_correct,
        "successful_task_completion": completed,
        "execution_success": execution_success,
        "latency_seconds": round(latency_seconds, 6),
        "token_usage": token_usage,
    }


def compare_model_summaries(models: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return pairwise metric differences for model summaries in one report."""
    comparisons: list[dict[str, Any]] = []
    metrics = (
        ("accuracy", "accuracy"),
        ("tool_selection_accuracy", "tool_selection_accuracy"),
        ("numerical_accuracy", "numerical_accuracy"),
        ("latency_seconds", "mean_latency_seconds"),
        ("failure_rate", "failure_rate"),
    )
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            differences = {
                report_metric: round(
                    float(right["summary"][summary_metric])
                    - float(left["summary"][summary_metric]),
                    6,
                )
                for report_metric, summary_metric in metrics
            }
            comparisons.append(
                {
                    "baseline": f"{left['provider']}:{left['model']}",
                    "comparison": f"{right['provider']}:{right['model']}",
                    "delta_comparison_minus_baseline": differences,
                }
            )
    return comparisons


def summarize_model_results(
    case_results: Sequence[dict[str, Any]],
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    count = len(case_results)
    attempted_results = [result for result in case_results if not result.get("skipped", False)]

    def mean(metric: str, values: Sequence[dict[str, Any]] | None = None) -> float:
        selected_results = case_results if values is None else values
        return round(
            sum(float(result["scores"][metric]) for result in selected_results)
            / len(selected_results),
            6,
        ) if selected_results else 0.0

    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    available_usage = [
        result["scores"]["token_usage"]
        for result in case_results
        if result["scores"]["token_usage"] is not None
    ]
    token_usage = (
        {
            key: sum(usage.get(key, 0) for usage in available_usage)
            for key in token_keys
            if any(key in usage for usage in available_usage)
        }
        if available_usage
        else None
    )
    completed = sum(
        bool(result["scores"]["successful_task_completion"])
        for result in case_results
    )
    failures = sum(
        not bool(result["scores"]["execution_success"])
        for result in attempted_results
    )
    return {
        "provider": provider,
        "model": model,
        "case_count": count,
        "accuracy": round(completed / count, 6) if count else 0.0,
        "tool_selection_accuracy": mean("tool_selection_accuracy"),
        "tool_precision": mean("tool_precision"),
        "tool_recall": mean("tool_recall"),
        "tool_argument_accuracy": mean("tool_argument_accuracy"),
        "numerical_accuracy": mean("numerical_accuracy"),
        "required_fact_coverage": mean("required_fact_coverage"),
        "unsupported_claim_rate": mean("unsupported_claim_rate"),
        "behavior_accuracy": mean("behavior_correct"),
        "successful_task_completion_rate": round(completed / count, 6) if count else 0.0,
        "attempted_case_count": len(attempted_results),
        "skipped_case_count": count - len(attempted_results),
        "failure_rate": (
            round(failures / len(attempted_results), 6)
            if attempted_results
            else 0.0
        ),
        "mean_latency_seconds": mean("latency_seconds", attempted_results),
        "token_usage": token_usage,
    }
