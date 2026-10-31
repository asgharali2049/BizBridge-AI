"""Explicitly invoked live evaluation orchestration and JSON reporting."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from agent.config import AgentConfig
from agent.errors import ProviderRequestError
from agent.mcp_client import StdioMCPToolClient
from agent.orchestrator import Agent
from agent.providers.factory import create_provider
from agent.types import AssistantTurn, ToolCall, ToolDefinition, ToolResult
from evaluation.dataset import EvaluationCase
from evaluation.scoring import compare_model_summaries, score_case, summarize_model_results


@dataclass
class _ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


class _RecordingToolClient:
    def __init__(self, client: Any, calls: list[_ToolExecution]) -> None:
        self._client = client
        self._calls = calls

    async def list_tools(self) -> list[ToolDefinition]:
        return await self._client.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            result = await self._client.call_tool(name, arguments)
        except Exception:
            result = ToolResult('{"error":"tool execution failed"}', is_error=True)
        self._calls.append(
            _ToolExecution(
                name=name,
                arguments=dict(arguments),
                result=result.content,
                is_error=result.is_error,
            )
        )
        return result


class _RecordingToolContext:
    def __init__(self, factory: Callable[[], Any], calls: list[_ToolExecution]) -> None:
        self._context = factory()
        self._calls = calls
        self._client: Any | None = None

    async def __aenter__(self) -> _RecordingToolClient:
        client = await self._context.__aenter__()
        self._client = _RecordingToolClient(client, self._calls)
        return self._client

    async def __aexit__(self, *exc_info: object) -> None:
        await self._context.__aexit__(*exc_info)
        self._client = None


class _RecordingProvider:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.provider_name = provider.provider_name
        self.model = provider.model
        self.calls: list[ToolCall] = []
        self._usage_totals: dict[str, int] = {}
        self._has_usage = False

    async def complete(
        self,
        messages: Sequence[Any],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        turn = await self._provider.complete(messages, tools, temperature)
        self.calls.extend(turn.tool_calls)
        if turn.token_usage:
            self._has_usage = True
            for name, value in turn.token_usage.items():
                self._usage_totals[name] = self._usage_totals.get(name, 0) + value
        return turn

    def token_usage(self, baseline: dict[str, int] | None = None) -> dict[str, int] | None:
        if not self._has_usage:
            return None
        if baseline is None:
            return dict(self._usage_totals)
        delta = {
            key: value - baseline.get(key, 0)
            for key, value in self._usage_totals.items()
            if value - baseline.get(key, 0) >= 0
        }
        return delta or None


def _safe_failure(error: Exception) -> str:
    if isinstance(error, ProviderRequestError):
        return "Provider request failed; check the provider status and retry later."
    if error.__class__.__name__ in {"UnsupportedModelError", "ConfigurationError"}:
        return str(error)
    return f"Evaluation failed safely ({error.__class__.__name__})."


def _case_result(
    case: EvaluationCase,
    *,
    answer: str,
    tool_calls: Sequence[ToolCall],
    tool_executions: Sequence[_ToolExecution],
    elapsed: float,
    token_usage: dict[str, int] | None,
    error: str | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    actual_calls = [
        {"name": call.name, "arguments": call.arguments}
        for call in tool_calls
    ]
    success = (
        error is None
        and not skipped
        and all(not execution.is_error for execution in tool_executions)
    )
    scores = score_case(
        case,
        actual_calls,
        answer,
        execution_success=success,
        error=error,
        latency_seconds=elapsed,
        token_usage=token_usage,
        tool_results=[execution.result for execution in tool_executions],
    )
    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_tool_calls": [
            {"name": expected.name, "arguments": expected.arguments}
            for expected in case.expected_tool_calls
        ],
        "tool_calls": actual_calls,
        "tool_executions": [
            {
                "name": execution.name,
                "arguments": execution.arguments,
                "result": execution.result,
                "is_error": execution.is_error,
            }
            for execution in tool_executions
        ],
        "final_answer": answer,
        "execution_success": success,
        "skipped": skipped,
        "error": error,
        "scores": scores,
    }


async def run_evaluations(
    targets: Sequence[AgentConfig],
    cases: Sequence[EvaluationCase],
    *,
    provider_factory: Callable[[AgentConfig], Any] = create_provider,
    tool_client_factory: Callable[[], Any] = StdioMCPToolClient,
) -> dict[str, Any]:
    """Run explicit live/mock targets against cases and return a secret-free report."""
    if not targets:
        raise ValueError("At least one provider/model target is required.")
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    model_reports: list[dict[str, Any]] = []
    for config in targets:
        provider_name = config.provider
        model_name = config.model
        try:
            provider = _RecordingProvider(provider_factory(config))
            provider_name = provider.provider_name
            model_name = provider.model
        except Exception as error:
            safe_error = _safe_failure(error)
            failed_cases = [
                _case_result(
                    case,
                    answer="",
                    tool_calls=(),
                    tool_executions=(),
                    elapsed=0.0,
                    token_usage=None,
                    error=safe_error,
                )
                for case in cases
            ]
            model_reports.append(
                {
                    "summary": summarize_model_results(
                        failed_cases, provider=provider_name, model=model_name
                    ),
                    "results": failed_cases,
                    "provider": provider_name,
                    "model": model_name,
                }
            )
            continue

        results: list[dict[str, Any]] = []
        halt_reason: str | None = None
        for case in cases:
            if halt_reason:
                results.append(
                    _case_result(
                        case,
                        answer="",
                        tool_calls=(),
                        tool_executions=(),
                        elapsed=0.0,
                        token_usage=None,
                        error=halt_reason,
                        skipped=True,
                    )
                )
                continue

            calls_before = len(provider.calls)
            usage_before = provider.token_usage()
            executions: list[_ToolExecution] = []
            started = time.perf_counter()
            answer = ""
            error_message: str | None = None
            try:
                agent = Agent(
                    provider=provider,
                    tool_client_factory=lambda: _RecordingToolContext(
                        tool_client_factory,
                        executions,
                    ),
                    max_iterations=config.max_iterations,
                    temperature=config.temperature,
                )
                answer = (await agent.run(case.question)).answer
            except Exception as error:
                error_message = _safe_failure(error)
                if isinstance(error, ProviderRequestError):
                    halt_reason = (
                        "Evaluation stopped after a provider failure; no further "
                        "requests were sent to this target."
                    )
            elapsed = time.perf_counter() - started
            results.append(
                _case_result(
                    case,
                    answer=answer,
                    tool_calls=provider.calls[calls_before:],
                    tool_executions=executions,
                    elapsed=elapsed,
                    token_usage=provider.token_usage(usage_before),
                    error=error_message,
                )
            )

        model_reports.append(
            {
                "summary": summarize_model_results(
                    results,
                    provider=provider_name,
                    model=model_name,
                ),
                "results": results,
                "provider": provider_name,
                "model": model_name,
            }
        )

    return {
        "report_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "models": model_reports,
        "comparisons": compare_model_summaries(model_reports),
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON report, refusing paths whose parent does not already exist."""
    path = Path(output_path).expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Report directory does not exist: {path.parent}")
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
