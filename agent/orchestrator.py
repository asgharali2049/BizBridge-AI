"""Provider-independent multi-step agent orchestration."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from agent.errors import AgentIterationLimitError, ProviderProtocolError
from agent.providers.base import LLMProvider
from agent.types import AssistantTurn, ConversationMessage, ToolCall, ToolDefinition, ToolResult


class MCPToolClient(Protocol):
    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult: ...


class MCPToolClientContext(Protocol):
    async def __aenter__(self) -> MCPToolClient: ...

    async def __aexit__(self, *exc_info: object) -> None: ...


ToolClientFactory = Callable[[], MCPToolClientContext]


@dataclass(frozen=True)
class ToolActivity:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    executed: bool


@dataclass(frozen=True)
class AgentResult:
    answer: str
    iterations: int
    tools_executed: int
    tool_activities: tuple[ToolActivity, ...] = ()


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tool_client_factory: ToolClientFactory,
        max_iterations: int = 6,
        temperature: float = 0.2,
    ) -> None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer.")
        if not 1 <= max_iterations <= 50:
            raise ValueError("max_iterations must be between 1 and 50.")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        self.provider = provider
        self._tool_client_factory = tool_client_factory
        self.max_iterations = max_iterations
        self.temperature = temperature

    async def run(
        self,
        question: str,
        conversation_history: Sequence[ConversationMessage] = (),
    ) -> AgentResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The business question cannot be empty.")
        if not isinstance(conversation_history, Sequence):
            raise TypeError("conversation_history must be a sequence of messages.")
        for message in conversation_history:
            if (
                not isinstance(message, ConversationMessage)
                or message.role not in ("user", "assistant")
                or message.tool_calls
                or not message.content.strip()
            ):
                raise ValueError(
                    "conversation_history may contain only non-empty user and "
                    "assistant messages without tool calls."
                )
        messages = [
            *conversation_history,
            ConversationMessage(role="user", content=question.strip()),
        ]
        total_tool_calls = 0
        tool_activities: list[ToolActivity] = []

        async with self._tool_client_factory() as tool_client:
            tools = await tool_client.list_tools()
            registered_tools = {tool.name for tool in tools}
            for iteration in range(1, self.max_iterations + 1):
                turn = await self.provider.complete(
                    messages,
                    tools,
                    self.temperature,
                )
                if not turn.tool_calls:
                    if not turn.content.strip():
                        raise ProviderProtocolError(
                            "The model returned neither an answer nor a tool call."
                        )
                    return AgentResult(
                        answer=turn.content.strip(),
                        iterations=iteration,
                        tools_executed=total_tool_calls,
                        tool_activities=tuple(tool_activities),
                    )
                if iteration == self.max_iterations:
                    raise AgentIterationLimitError(
                        f"The model reached the {self.max_iterations}-iteration limit "
                        "before providing a final answer."
                    )

                messages.append(
                    ConversationMessage(
                        role="assistant",
                        content=turn.content,
                        tool_calls=turn.tool_calls,
                        provider_payload=turn.provider_payload,
                    )
                )
                for call in turn.tool_calls:
                    executed = (
                        call.name in registered_tools
                        and isinstance(call.arguments, dict)
                    )
                    result = await self._execute_registered_tool(
                        tool_client,
                        call,
                        registered_tools,
                    )
                    if not result.is_error:
                        total_tool_calls += 1
                    tool_activities.append(
                        ToolActivity(
                            name=call.name,
                            arguments=call.arguments,
                            result=result.content,
                            is_error=result.is_error,
                            executed=executed,
                        )
                    )
                    messages.append(
                        ConversationMessage(
                            role="tool",
                            content=result.content,
                            tool_call_id=call.call_id,
                            tool_name=call.name,
                        )
                    )

        raise AssertionError("The iteration loop exited unexpectedly.")

    @staticmethod
    async def _execute_registered_tool(
        tool_client: MCPToolClient,
        call: ToolCall,
        registered_tools: set[str],
    ) -> ToolResult:
        if call.name not in registered_tools:
            return ToolResult(
                content=json.dumps({"error": "Requested tool is not registered."}),
                is_error=True,
            )
        if not isinstance(call.arguments, dict):
            return ToolResult(
                content=json.dumps({"error": "Tool arguments must be a JSON object."}),
                is_error=True,
            )
        try:
            return await tool_client.call_tool(call.name, call.arguments)
        except Exception:
            return ToolResult(
                content=json.dumps({"error": "Tool execution failed safely."}),
                is_error=True,
            )
