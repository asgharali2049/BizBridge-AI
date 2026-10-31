"""Ollama Cloud adapter using the official Ollama Python SDK."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from ollama import AsyncClient

from agent.errors import ProviderProtocolError, ProviderRequestError, UnsupportedModelError
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition


SUPPORTED_MODELS = (
    "nemotron-3-nano:30b-cloud",
    "nemotron-3-super:cloud",
    "beam:cloud",
)


class OllamaCloudProvider:
    provider_name = "ollama"

    def __init__(
        self,
        model: str,
        api_key: str,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required.")
        self.model = model
        self._client = client or AsyncClient(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model_verified = False

    async def _verify_model(self) -> None:
        if self._model_verified:
            return
        try:
            model_info = await self._client.show(self.model)
        except Exception as error:
            status = getattr(error, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise ProviderRequestError(
                f"Ollama Cloud could not verify model tool support{suffix}."
            ) from None
        capabilities = getattr(model_info, "capabilities", None) or []
        if "tools" not in capabilities:
            raise UnsupportedModelError(
                "The selected Ollama model does not advertise the 'tools' capability."
            )
        self._model_verified = True

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        await self._verify_model()
        ollama_messages = [_message_to_ollama(message) for message in messages]
        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]
        try:
            response = await self._client.chat(
                model=self.model,
                messages=ollama_messages,
                tools=ollama_tools,
                options={"temperature": temperature},
                stream=False,
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise ProviderRequestError(f"Ollama Cloud request failed{suffix}.") from None

        message = response.message
        calls = []
        for requested_call in message.tool_calls or []:
            function = requested_call.function
            arguments = function.arguments
            if not isinstance(arguments, Mapping):
                raise ProviderProtocolError("Ollama returned invalid tool arguments.")
            calls.append(
                ToolCall(
                    call_id=str(uuid.uuid4()),
                    name=function.name,
                    arguments=dict(arguments),
                )
            )
        return AssistantTurn(
            content=message.content or "",
            tool_calls=tuple(calls),
            token_usage=_normalize_usage(
                getattr(response, "prompt_eval_count", None),
                getattr(response, "eval_count", None),
            ),
        )


def _message_to_ollama(message: ConversationMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_name": message.tool_name or "",
            "content": message.content,
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content}


def _normalize_usage(prompt_tokens: Any, completion_tokens: Any) -> dict[str, int] | None:
    if not isinstance(prompt_tokens, int) and not isinstance(completion_tokens, int):
        return None
    usage: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens
    if len(usage) == 2:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage
