from __future__ import annotations

import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from agent.config import AgentConfig, ConfigurationError
from agent.errors import (
    AgentIterationLimitError,
    ProviderRequestError,
    UnsupportedModelError,
)
from agent.orchestrator import Agent
from agent.providers.factory import create_provider
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition, ToolResult


class AgentConfigurationTests(unittest.TestCase):
    def test_selects_each_supported_provider_and_hides_api_key_from_repr(self) -> None:
        cases = (
            ("ollama", "OLLAMA_API_KEY", "nemotron-3-nano:30b-cloud"),
            ("gemini", "GEMINI_API_KEY", "gemini-2.5-flash"),
            ("openai", "OPENAI_API_KEY", "gpt-6-astra"),
        )
        for provider, key_name, model in cases:
            with self.subTest(provider=provider):
                config = AgentConfig.from_env(
                    {
                        "LLM_PROVIDER": provider,
                        "LLM_MODEL": model,
                        key_name: "test-secret",
                    }
                )
                self.assertEqual(config.provider, provider)
                self.assertEqual(config.model, model)
                self.assertNotIn("test-secret", repr(config))

    def test_rejects_missing_provider_model_and_credentials(self) -> None:
        for values in (
            {},
            {"LLM_PROVIDER": "openai"},
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-6-astra"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ConfigurationError):
                    AgentConfig.from_env(values)

    def test_rejects_unknown_provider_and_invalid_limits_or_temperature(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Unsupported LLM_PROVIDER"):
            AgentConfig.from_env({"LLM_PROVIDER": "custom", "LLM_MODEL": "m"})
        base = {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-6-astra",
            "OPENAI_API_KEY": "test-key",
        }
        for name, value in (
            ("LLM_MAX_ITERATIONS", "0"),
            ("LLM_MAX_ITERATIONS", "many"),
            ("LLM_TEMPERATURE", "nan"),
            ("LLM_TEMPERATURE", "3"),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ConfigurationError):
                    AgentConfig.from_env({**base, name: value})

    def test_loads_dotenv_from_an_explicit_file(self) -> None:
        with TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / "agent.env"
            dotenv_path.write_text(
                "LLM_PROVIDER=gemini\n"
                "LLM_MODEL=gemini-2.5-flash\n"
                "GEMINI_API_KEY=dotenv-test-key\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                config = AgentConfig.from_env(dotenv_path=dotenv_path)
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.api_key, "dotenv-test-key")

    def test_provider_factory_does_not_fallback_for_unknown_model(self) -> None:
        config = AgentConfig(
            provider="openai",
            model="unsupported-model",
            api_key="test-key",
        )
        with self.assertRaises(UnsupportedModelError):
            create_provider(config)


class _FakeProvider:
    provider_name = "test"
    model = "scripted"

    def __init__(self, turns: Sequence[AssistantTurn] | Exception) -> None:
        self.turns = list(turns) if not isinstance(turns, Exception) else turns
        self.calls: list[tuple[list[ConversationMessage], list[ToolDefinition], float]] = []

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        self.calls.append((list(messages), list(tools), temperature))
        if isinstance(self.turns, Exception):
            raise self.turns
        return self.turns.pop(0)


class _FakeToolClient:
    def __init__(self) -> None:
        self.tools = [ToolDefinition("sales_summary", "Summarize sales", {"type": "object"})]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeToolClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def list_tools(self) -> list[ToolDefinition]:
        return self.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult('{"order_count":2}')


class _FakeToolClientFactory:
    def __init__(self, client: _FakeToolClient) -> None:
        self.client = client

    def __call__(self) -> _FakeToolClient:
        return self.client


class AgentOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_multiple_tool_calls_and_returns_final_answer(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider(
            [
                AssistantTurn(
                    content="",
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "sales_summary",
                            {"start_date": "2025-01-01", "end_date": "2025-01-02"},
                        ),
                        ToolCall(
                            "call-2",
                            "sales_summary",
                            {"start_date": "2025-02-01", "end_date": "2025-02-02"},
                        ),
                    ),
                ),
                AssistantTurn(content="There were two orders in the first range."),
            ]
        )
        result = await Agent(
            provider,
            _FakeToolClientFactory(client),
            max_iterations=3,
            temperature=0.7,
        ).run("Summarize recent sales.")

        self.assertEqual(result.answer, "There were two orders in the first range.")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.tools_executed, 2)
        self.assertEqual(
            [activity.name for activity in result.tool_activities],
            ["sales_summary", "sales_summary"],
        )
        self.assertTrue(all(activity.executed for activity in result.tool_activities))
        self.assertTrue(all(not activity.is_error for activity in result.tool_activities))
        self.assertEqual(result.tool_activities[0].result, '{"order_count":2}')
        self.assertEqual(len(client.calls), 2)
        second_request = provider.calls[1][0]
        self.assertEqual([message.role for message in second_request], [
            "user",
            "assistant",
            "tool",
            "tool",
        ])
        self.assertEqual(provider.calls[0][2], 0.7)

    async def test_unregistered_tool_is_never_executed(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider(
            [
                AssistantTurn(
                    content="",
                    tool_calls=(
                        ToolCall("bad-call", "run_sql", {"query": "DELETE FROM orders"}),
                    ),
                ),
                AssistantTurn(content="I cannot run that tool."),
            ]
        )
        result = await Agent(provider, _FakeToolClientFactory(client)).run(
            "Delete records."
        )
        self.assertEqual(result.answer, "I cannot run that tool.")
        self.assertEqual(client.calls, [])
        self.assertEqual(len(result.tool_activities), 1)
        self.assertFalse(result.tool_activities[0].executed)
        self.assertTrue(result.tool_activities[0].is_error)
        self.assertIn("Requested tool is not registered", provider.calls[1][0][-1].content)

    async def test_iteration_limit_is_reported_instead_of_falling_back(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider(
            [
                AssistantTurn(
                    content="",
                    tool_calls=(ToolCall("call-1", "sales_summary", {}),),
                )
            ]
            * 2
        )
        with self.assertRaisesRegex(AgentIterationLimitError, "2-iteration limit"):
            await Agent(
                provider,
                _FakeToolClientFactory(client),
                max_iterations=2,
            ).run("Keep querying.")

    async def test_provider_errors_are_propagated(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider(ProviderRequestError("OpenAI request failed (HTTP 429)."))
        with self.assertRaisesRegex(ProviderRequestError, "HTTP 429"):
            await Agent(provider, _FakeToolClientFactory(client)).run("Question?")

    async def test_empty_question_is_rejected(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider([AssistantTurn(content="answer")])
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await Agent(provider, _FakeToolClientFactory(client)).run("  ")

    async def test_conversation_history_is_sent_to_the_provider(self) -> None:
        client = _FakeToolClient()
        provider = _FakeProvider([AssistantTurn(content="The follow-up answer.")])
        history = [
            ConversationMessage(role="user", content="What were sales?"),
            ConversationMessage(role="assistant", content="Revenue was PKR 500."),
        ]

        result = await Agent(provider, _FakeToolClientFactory(client)).run(
            "How many orders made that revenue?",
            conversation_history=history,
        )

        self.assertEqual(result.answer, "The follow-up answer.")
        self.assertEqual(
            [(message.role, message.content) for message in provider.calls[0][0]],
            [
                ("user", "What were sales?"),
                ("assistant", "Revenue was PKR 500."),
                ("user", "How many orders made that revenue?"),
            ],
        )

    async def test_conversation_history_rejects_tool_messages(self) -> None:
        history = [
            ConversationMessage(
                role="tool",
                content='{"result": "not trusted"}',
                tool_call_id="call-1",
                tool_name="sales_summary",
            )
        ]
        with self.assertRaisesRegex(ValueError, "only non-empty user and assistant"):
            await Agent(
                _FakeProvider([AssistantTurn(content="unused")]),
                _FakeToolClientFactory(_FakeToolClient()),
            ).run("Follow up", conversation_history=history)
