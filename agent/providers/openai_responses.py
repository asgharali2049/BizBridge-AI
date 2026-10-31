"""OpenAI adapter using the official Responses API function tools."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI

from agent.errors import ProviderProtocolError, ProviderRequestError, UnsupportedModelError
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition


SUPPORTED_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})


class OpenAIProvider:
    provider_name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if model not in SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"OpenAI model '{model}' is not in the verified function-calling model set."
            )
        self.model = model
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        request_tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": False,
            }
            for tool in tools
        ]
        request_input = _messages_to_openai(messages)
        try:
            response = await self._client.responses.create(
                model=self.model,
                tools=request_tools,
                input=request_input,
                parallel_tool_calls=True,
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise ProviderRequestError(f"OpenAI request failed{suffix}.") from None

        output = response.output or []
        calls = []
        for item in output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, ValueError):
                raise ProviderProtocolError(
                    "OpenAI returned malformed function arguments."
                ) from None
            if not isinstance(arguments, Mapping):
                raise ProviderProtocolError(
                    "OpenAI function arguments must be a JSON object."
                )
            calls.append(
                ToolCall(
                    call_id=item.call_id or str(uuid.uuid4()),
                    name=item.name,
                    arguments=dict(arguments),
                )
            )

        provider_payload = [
            item.model_dump(mode="json", exclude_none=True) for item in output
        ]
        return AssistantTurn(
            content=response.output_text or "",
            tool_calls=tuple(calls),
            provider_payload=provider_payload,
            token_usage=_normalize_usage(getattr(response, "usage", None)),
        )


def _messages_to_openai(
    messages: Sequence[ConversationMessage],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            result.append({"role": "user", "content": message.content})
        elif message.role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": message.content,
                }
            )
        elif message.provider_payload:
            result.extend(message.provider_payload)
        else:
            result.append({"role": "assistant", "content": message.content})
    return result


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    names = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
    }
    return {
        target: value
        for source, target in names.items()
        if isinstance((value := getattr(usage, source, None)), int)
    } or None
