from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import seed
from agent.config import AgentConfig
from agent.errors import ProviderRequestError
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition, ToolResult
from evaluation.dataset import EvaluationCase, ExpectedToolCall, load_evaluation_cases
from evaluation.runner import run_evaluations, write_report
from evaluation.scoring import score_case


class _FakeToolClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeToolClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("sales_summary", "Summarize sales", {"type": "object"}),
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult('{"order_count":2,"revenue_pkr":1200}')


class _FakeProvider:
    provider_name = "mock"

    def __init__(self, turns: Sequence[AssistantTurn] | Exception, model: str) -> None:
        self.turns = list(turns) if not isinstance(turns, Exception) else turns
        self.model = model

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        if isinstance(self.turns, Exception):
            raise self.turns
        return self.turns.pop(0)


class EvaluationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "evaluation-fixture.db"
        seed.generate_database(
            self.database_path,
            seed=418,
            reference_date="2025-12-31",
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_loads_at_least_twenty_cases_grounded_in_fixture_database(self) -> None:
        before = self.database_path.read_bytes()
        cases = load_evaluation_cases(self.database_path)
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        by_id = {case.case_id: case for case in cases}

        full = by_id["sales-full-period"]
        result = seed.connect_database(self.database_path)
        try:
            expected = result.execute(
                "SELECT COUNT(*), SUM(total) FROM orders WHERE status='completed'"
            ).fetchone()
        finally:
            result.close()
        self.assertEqual(full.expected_numbers[:2], tuple(expected))
        self.assertEqual(full.expected_tool_calls[0].name, "sales_summary")

        weekly = by_id["weekly-recurring-orders"]
        self.assertEqual(weekly.expected_tool_calls[0].arguments["customer_id"], 1)
        self.assertEqual(
            by_id["clarify-invalid-calendar-date"].expected_tool_calls,
            (),
        )
        self.assertEqual(before, self.database_path.read_bytes())


class EvaluationScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = EvaluationCase(
            case_id="fixture-case",
            question="Summarize from 2025-01-01 through 2025-01-02.",
            expected_tool_calls=(
                ExpectedToolCall(
                    "sales_summary",
                    {"start_date": "2025-01-01", "end_date": "2025-01-02"},
                ),
            ),
            required_fact_groups=(("2 completed orders",), ("PKR 1,200", "1,200 PKR")),
            expected_numbers=(2, 1200),
        )

    def test_expected_tool_calls_and_arguments_are_compared(self) -> None:
        correct = score_case(
            self.case,
            [
                {
                    "name": "sales_summary",
                    "arguments": {
                        "start_date": "2025-01-01",
                        "end_date": "2025-01-02",
                    },
                }
            ],
            "There were 2 completed orders and revenue was PKR 1,200.",
            execution_success=True,
            error=None,
            latency_seconds=0.1,
            token_usage=None,
        )
        self.assertEqual(correct["tool_selection_accuracy"], 1)
        self.assertEqual(correct["tool_argument_accuracy"], 1)
        self.assertEqual(correct["required_fact_coverage"], 1)

    def test_missing_and_extra_calls_reduce_tool_selection_and_argument_scores(self) -> None:
        missing = score_case(
            self.case,
            [],
            "There were 2 completed orders and revenue was PKR 1,200.",
            execution_success=True,
            error=None,
            latency_seconds=0,
            token_usage=None,
        )
        self.assertEqual(missing["tool_selection_accuracy"], 0)
        self.assertEqual(missing["tool_recall"], 0)
        self.assertEqual(missing["tool_argument_accuracy"], 0)

        extra = score_case(
            self.case,
            [
                {
                    "name": "sales_summary",
                    "arguments": {
                        "start_date": "2025-01-01",
                        "end_date": "2025-01-02",
                    },
                },
                {"name": "run_sql", "arguments": {"query": "SELECT 1"}},
            ],
            "There were 2 completed orders and revenue was PKR 1,200.",
            execution_success=True,
            error=None,
            latency_seconds=0,
            token_usage=None,
        )
        self.assertEqual(extra["tool_selection_accuracy"], 0)
        self.assertEqual(extra["tool_precision"], 0.5)

    def test_numeric_claims_and_unsupported_claim_rate_are_deterministic(self) -> None:
        result = score_case(
            self.case,
            [
                {
                    "name": "sales_summary",
                    "arguments": {
                        "start_date": "2025-01-01",
                        "end_date": "2025-01-02",
                    },
                }
            ],
            "There were 2 completed orders and revenue was PKR 1,500.",
            execution_success=True,
            error=None,
            latency_seconds=0,
            token_usage=None,
            tool_results=['{"order_count":2,"revenue_pkr":1200}'],
        )
        self.assertEqual(result["numerical_accuracy"], 0.5)
        self.assertEqual(result["unsupported_numeric_claims"], ["1500"])
        self.assertEqual(result["unsupported_claim_rate"], 0.5)

    def test_clarification_behavior_requires_no_tool_call(self) -> None:
        clarification_case = EvaluationCase(
            "ask-clarifying-question",
            "Which customer do you mean?",
            (),
            (("which customer",),),
            behavior="clarify",
        )
        result = score_case(
            clarification_case,
            [],
            "Which customer ID should I use?",
            execution_success=True,
            error=None,
            latency_seconds=0,
            token_usage=None,
        )
        self.assertTrue(result["behavior_correct"])


class EvaluationRunnerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _no_tool_case() -> EvaluationCase:
        return EvaluationCase(
            "no-tool",
            "Say the word ready.",
            (),
            (("ready",),),
            behavior="answer_without_tools",
        )

    @staticmethod
    def _tool_case() -> EvaluationCase:
        return EvaluationCase(
            "one-tool",
            "Get the summary.",
            (ExpectedToolCall("sales_summary", {"start_date": "2025-01-01", "end_date": "2025-01-02"}),),
            (("2 completed orders",),),
            (2,),
        )

    async def test_runs_multiple_targets_and_serializes_secret_free_report(self) -> None:
        configs = [
            AgentConfig("ollama", "beam:cloud", api_key="must-not-appear"),
            AgentConfig("ollama", "nemotron-3-nano:30b-cloud", api_key="another-secret"),
        ]

        def fake_provider_factory(config: AgentConfig) -> _FakeProvider:
            answer = "ready" if config.model == "beam:cloud" else "unprepared"
            return _FakeProvider([AssistantTurn(content=answer)], config.model)

        report = await run_evaluations(
            configs,
            [self._no_tool_case()],
            provider_factory=fake_provider_factory,
            tool_client_factory=_FakeToolClient,
        )
        serialized = json.dumps(report)
        self.assertNotIn("must-not-appear", serialized)
        self.assertNotIn("another-secret", serialized)
        self.assertEqual(len(report["models"]), 2)
        self.assertEqual(report["models"][0]["summary"]["accuracy"], 1)
        self.assertEqual(report["models"][1]["summary"]["accuracy"], 0)
        self.assertEqual(len(report["comparisons"]), 1)
        self.assertEqual(
            report["comparisons"][0]["delta_comparison_minus_baseline"]["accuracy"],
            -1,
        )
        self.assertIsInstance(
            report["comparisons"][0]["delta_comparison_minus_baseline"]["latency_seconds"],
            float,
        )
        self.assertIsNone(report["models"][0]["summary"]["token_usage"])

    async def test_report_is_written_as_machine_readable_json(self) -> None:
        report = await run_evaluations(
            [AgentConfig("ollama", "beam:cloud", api_key="private-key")],
            [self._no_tool_case()],
            provider_factory=lambda _: _FakeProvider(
                [AssistantTurn(content="ready")], "beam:cloud"
            ),
            tool_client_factory=_FakeToolClient,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = write_report(report, Path(directory) / "report.json")
            serialized = output.read_text(encoding="utf-8")
            loaded = json.loads(serialized)
        self.assertEqual(loaded["case_count"], 1)
        self.assertNotIn("private-key", serialized)

    async def test_records_tool_calls_latency_and_token_usage(self) -> None:
        config = AgentConfig("ollama", "beam:cloud", api_key="placeholder")
        fake_provider = _FakeProvider(
            [
                AssistantTurn(
                    content="",
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "sales_summary",
                            {"start_date": "2025-01-01", "end_date": "2025-01-02"},
                        ),
                    ),
                    token_usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                ),
                AssistantTurn(
                    "2 completed orders.",
                    token_usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                ),
            ],
            "beam:cloud",
        )
        report = await run_evaluations(
            [config],
            [self._tool_case()],
            provider_factory=lambda _: fake_provider,
            tool_client_factory=_FakeToolClient,
        )
        result = report["models"][0]["results"][0]
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "sales_summary")
        self.assertEqual(result["scores"]["token_usage"]["total_tokens"], 28)
        self.assertGreaterEqual(result["scores"]["latency_seconds"], 0)

    async def test_provider_failure_is_reported_and_stops_repeated_requests(self) -> None:
        config = AgentConfig("ollama", "beam:cloud", api_key="secret-key")
        cases = [self._no_tool_case(), self._no_tool_case()]
        report = await run_evaluations(
            [config],
            cases,
            provider_factory=lambda _: _FakeProvider(
                ProviderRequestError("safe provider error"), "beam:cloud"
            ),
            tool_client_factory=_FakeToolClient,
        )
        results = report["models"][0]["results"]
        self.assertFalse(results[0]["execution_success"])
        self.assertIn("Provider request failed", results[0]["error"])
        self.assertTrue(results[1]["skipped"])
        self.assertNotIn("secret-key", json.dumps(report))
        self.assertEqual(report["models"][0]["summary"]["attempted_case_count"], 1)
        self.assertEqual(report["models"][0]["summary"]["skipped_case_count"], 1)
        self.assertEqual(report["models"][0]["summary"]["failure_rate"], 1)

    async def test_production_database_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "isolated.db"
            seed.generate_database(database, seed=418, reference_date="2025-12-31")
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            report = await run_evaluations(
                [AgentConfig("ollama", "mock-model", api_key="test-key")],
                [self._no_tool_case()],
                provider_factory=lambda _: _FakeProvider(
                    [AssistantTurn(content="ready")], "mock-model"
                ),
                tool_client_factory=_FakeToolClient,
            )
            after = hashlib.sha256(database.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(report["models"][0]["results"][0]["execution_success"], True)


if __name__ == "__main__":
    unittest.main()
