"""Google Gemini adapter using the official google-genai SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types

from agent.errors import ProviderProtocolError, ProviderRequestError, UnsupportedModelError
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition


SUPPORTED_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"})


class GeminiProvider:
    provider_name = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        if model not in SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"Gemini model '{model}' is not in the verified function-calling model set."
            )
        self.model = model
        self._client = client or genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn:
        contents = [_message_to_gemini(message) for message in messages]
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.input_schema,
            )
            for tool in tools
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as error:
            status = getattr(error, "code", None)
            suffix = f" (HTTP {status})" if isinstance(status, int) else ""
            raise ProviderRequestError(f"Gemini request failed{suffix}.") from None

        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            return AssistantTurn(
                content=response.text or "",
                token_usage=_normalize_usage(getattr(response, "usage_metadata", None)),
            )

        parts = candidates[0].content.parts or []
        text = "".join(part.text for part in parts if part.text)
        calls = []
        for part in parts:
            function_call = part.function_call
            if function_call and function_call.name:
                arguments = function_call.args or {}
                if not isinstance(arguments, Mapping):
                    raise ProviderProtocolError("Gemini returned invalid function arguments.")
                calls.append(
                    ToolCall(
                        call_id=function_call.id,
                        name=function_call.name,
                        arguments=dict(arguments),
                    )
                )
        return AssistantTurn(
            content=text,
            tool_calls=tuple(calls),
            token_usage=_normalize_usage(getattr(response, "usage_metadata", None)),
        )


def _message_to_gemini(message: ConversationMessage) -> types.Content:
    if message.role == "user":
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=message.content)],
        )

    if message.role == "assistant":
        parts = []
        if message.content:
            parts.append(types.Part.from_text(text=message.content))
        parts.extend(
            types.Part(
                function_call=types.FunctionCall(
                    id=call.call_id,
                    name=call.name,
                    args=call.arguments,
                )
            )
            for call in message.tool_calls
        )
        return types.Content(role="model", parts=parts)

    try:
        result: Any = json.loads(message.content)
    except ValueError:
        result = {"result": message.content}
    if not isinstance(result, dict):
        result = {"result": result}
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=message.tool_call_id,
                    name=message.tool_name or "",
                    response=result,
                )
            )
        ],
    )


def _normalize_usage(metadata: Any) -> dict[str, int] | None:
    if metadata is None:
        return None
    names = {
        "prompt_token_count": "prompt_tokens",
        "candidates_token_count": "completion_tokens",
        "total_token_count": "total_tokens",
    }
    return {
        target: value
        for source, target in names.items()
        if isinstance((value := getattr(metadata, source, None)), int)
    } or None
