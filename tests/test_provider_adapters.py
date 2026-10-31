from __future__ import annotations

import json
import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.genai import types

from agent.errors import ProviderRequestError, UnsupportedModelError
from agent.providers.gemini import GeminiProvider
from agent.providers.gemini import _message_to_gemini
from agent.providers.ollama_cloud import OllamaCloudProvider
from agent.providers.openai_responses import OpenAIProvider
from agent.types import ConversationMessage, ToolCall, ToolDefinition


TOOLS = [
    ToolDefinition(
        name="sales_summary",
        description="Summarize sales",
        input_schema={
            "type": "object",
            "properties": {"start_date": {"type": "string"}},
            "required": ["start_date"],
        },
    )
]
USER_MESSAGE = [ConversationMessage(role="user", content="Summarize sales.")]


class OllamaAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_checks_tool_capability_and_normalizes_tool_calls(self) -> None:
        response_message = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="sales_summary",
                        arguments={"start_date": "2025-01-01"},
                    )
                )
            ],
        )
        client = SimpleNamespace(
            show=AsyncMock(return_value=SimpleNamespace(capabilities=["completion", "tools"])),
            chat=AsyncMock(return_value=SimpleNamespace(message=response_message)),
        )
        provider = OllamaCloudProvider(
            "nemotron-3-nano:30b-cloud",
            "test-key",
            client=client,
        )
        turn = await provider.complete(USER_MESSAGE, TOOLS, temperature=0.4)

        self.assertEqual(turn.tool_calls[0].name, "sales_summary")
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"start_date": "2025-01-01"},
        )
        chat_request = client.chat.await_args.kwargs
        self.assertEqual(chat_request["options"], {"temperature": 0.4})
        self.assertEqual(chat_request["tools"][0]["function"]["name"], "sales_summary")
        self.assertEqual(client.show.await_count, 1)

    async def test_rejects_ollama_models_without_tools_capability(self) -> None:
        client = SimpleNamespace(
            show=AsyncMock(return_value=SimpleNamespace(capabilities=["completion"])),
            chat=AsyncMock(),
        )
        provider = OllamaCloudProvider("text-only", "test-key", client=client)
        with self.assertRaises(UnsupportedModelError):
            await provider.complete(USER_MESSAGE, TOOLS, temperature=0.2)
        client.chat.assert_not_awaited()

    async def test_redacts_ollama_provider_errors(self) -> None:
        client = SimpleNamespace(
            show=AsyncMock(side_effect=RuntimeError("test-key leaked")),
            chat=AsyncMock(),
        )
        provider = OllamaCloudProvider("model", "test-key", client=client)
        with self.assertRaises(ProviderRequestError) as error:
            await provider.complete(USER_MESSAGE, TOOLS, temperature=0.2)
        self.assertNotIn("test-key", str(error.exception))


class GeminiAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalizes_gemini_function_call_with_provider_tool_schema(self) -> None:
        function_call = types.FunctionCall(
            name="sales_summary",
            args={"start_date": "2025-01-01"},
            id="gemini-call-1",
        )
        content = types.Content(
            role="model",
            parts=[types.Part(function_call=function_call)],
        )
        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(
                        return_value=SimpleNamespace(
                            candidates=[SimpleNamespace(content=content)],
                            text=None,
                        )
                    )
                )
            )
        )
        provider = GeminiProvider("gemini-2.5-flash", "test-key", client=client)
        turn = await provider.complete(USER_MESSAGE, TOOLS, temperature=0.3)

        self.assertEqual(
            turn.tool_calls,
            (
                ToolCall(
                    "gemini-call-1",
                    "sales_summary",
                    {"start_date": "2025-01-01"},
                ),
            ),
        )
        config = client.aio.models.generate_content.await_args.kwargs["config"]
        declaration = config.tools[0].function_declarations[0]
        self.assertEqual(declaration.name, "sales_summary")
        self.assertEqual(declaration.parameters_json_schema, TOOLS[0].input_schema)
        self.assertTrue(config.automatic_function_calling.disable)
        tool_response = _message_to_gemini(
            ConversationMessage(
                role="tool",
                content='{"order_count":2}',
                tool_call_id=turn.tool_calls[0].call_id,
                tool_name="sales_summary",
            )
        )
        function_response = tool_response.parts[0].function_response
        self.assertEqual(function_response.id, "gemini-call-1")
        self.assertEqual(function_response.name, "sales_summary")
        self.assertEqual(function_response.response, {"order_count": 2})

    async def test_rejects_unverified_gemini_model_and_redacts_errors(self) -> None:
        with self.assertRaises(UnsupportedModelError):
            GeminiProvider("gemini-audio-model", "test-key", client=object())

        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(side_effect=RuntimeError("test-key leaked"))
                )
            )
        )
        provider = GeminiProvider("gemini-2.5-flash", "test-key", client=client)
        with self.assertRaises(ProviderRequestError) as error:
            await provider.complete(USER_MESSAGE, TOOLS, temperature=0.3)
        self.assertNotIn("test-key", str(error.exception))


class OpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalizes_openai_response_tool_call_and_keeps_continuation_items(self) -> None:
        call_item = SimpleNamespace(
            type="function_call",
            name="sales_summary",
            call_id="openai-call-1",
            arguments=json.dumps({"start_date": "2025-01-01"}),
            model_dump=lambda **kwargs: {
                "type": "function_call",
                "name": "sales_summary",
                "call_id": "openai-call-1",
                "arguments": '{"start_date":"2025-01-01"}',
            },
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        output=[call_item],
                        output_text="",
                    )
                )
            )
        )
        provider = OpenAIProvider("gpt-6-astra", "test-key", client=client)
        turn = await provider.complete(USER_MESSAGE, TOOLS, temperature=0.1)

        self.assertEqual(
            turn.tool_calls,
            (
                ToolCall(
                    "openai-call-1",
                    "sales_summary",
                    {"start_date": "2025-01-01"},
                ),
            ),
        )
        self.assertEqual(turn.provider_payload, [call_item.model_dump()])
        request = client.responses.create.await_args.kwargs
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(request["tools"][0]["parameters"], TOOLS[0].input_schema)
        self.assertNotIn("temperature", request)

    async def test_openai_responses_continuation_keeps_function_call_id(self) -> None:
        first_output = SimpleNamespace(
            type="function_call",
            name="sales_summary",
            call_id="openai-call-2",
            arguments='{"start_date":"2025-01-01"}',
            model_dump=lambda **kwargs: {
                "type": "function_call",
                "name": "sales_summary",
                "call_id": "openai-call-2",
                "arguments": '{"start_date":"2025-01-01"}',
            },
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(
                    side_effect=[
                        SimpleNamespace(output=[first_output], output_text=""),
                        SimpleNamespace(output=[], output_text="Revenue was PKR 300."),
                    ]
                )
            )
        )
        provider = OpenAIProvider("gpt-6-astra", "test-key", client=client)
        first_turn = await provider.complete(USER_MESSAGE, TOOLS, temperature=0.1)
        second_turn = await provider.complete(
            [
                *USER_MESSAGE,
                ConversationMessage(
                    role="assistant",
                    content=first_turn.content,
                    tool_calls=first_turn.tool_calls,
                    provider_payload=first_turn.provider_payload,
                ),
                ConversationMessage(
                    role="tool",
                    content='{"order_count":1}',
                    tool_call_id=first_turn.tool_calls[0].call_id,
                    tool_name="sales_summary",
                ),
            ],
            TOOLS,
            temperature=0.1,
        )
        self.assertEqual(second_turn.content, "Revenue was PKR 300.")
        second_input = client.responses.create.await_args_list[1].kwargs["input"]
        self.assertEqual(second_input[-1]["type"], "function_call_output")
        self.assertEqual(second_input[-1]["call_id"], "openai-call-2")

    async def test_rejects_unverified_openai_model_and_redacts_provider_error(self) -> None:
        with self.assertRaises(UnsupportedModelError):
            OpenAIProvider("gpt-unknown", "test-key", client=object())

        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("test-key leaked"))
            )
        )
        provider = OpenAIProvider("gpt-6-astra", "test-key", client=client)
        with self.assertRaises(ProviderRequestError) as error:
            await provider.complete(USER_MESSAGE, TOOLS, temperature=0.1)
        self.assertNotIn("test-key", str(error.exception))


if __name__ == "__main__":
    unittest.main()
