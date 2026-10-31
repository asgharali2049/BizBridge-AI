"""Shared protocol for model providers."""

from __future__ import annotations

from typing import Protocol, Sequence

from agent.types import AssistantTurn, ConversationMessage, ToolDefinition


class LLMProvider(Protocol):
    provider_name: str
    model: str

    async def complete(
        self,
        messages: Sequence[ConversationMessage],
        tools: Sequence[ToolDefinition],
        temperature: float,
    ) -> AssistantTurn: ...
