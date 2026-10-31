"""Provider-neutral messages and tool types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    call_id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    provider_payload: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    provider_payload: Any = field(default=None, repr=False, compare=False)
    token_usage: dict[str, int] | None = None


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


ToolSchema = Mapping[str, Any]
