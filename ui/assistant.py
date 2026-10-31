"""Streamlit chat interface for the existing provider-independent agent."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import streamlit as st

from ui.styles import render_empty_state, render_page_heading
from agent.config import AgentConfig, ConfigurationError
from agent.errors import (
    AgentError,
    AgentIterationLimitError,
    ProviderProtocolError,
    ProviderRequestError,
    UnsupportedModelError,
)
from agent.mcp_client import StdioMCPToolClient
from agent.orchestrator import Agent, AgentResult
from agent.providers.factory import create_provider
from agent.types import ConversationMessage
from ui.settings import get_session_agent_config


CHAT_HISTORY_KEY = "bizbridge_chat_history"
_BUSY_KEY = "bizbridge_chat_busy"
_UNEXPECTED_ERROR = (
    "An unexpected error prevented the assistant from completing the request."
)


def _run_agent_question(
    question: str,
    previous_history: Sequence[dict[str, Any]] = (),
) -> AgentResult:
    """Run one user turn through the configured production agent."""
    config = get_session_agent_config() or AgentConfig.from_env()
    provider = create_provider(config)
    agent = Agent(
        provider=provider,
        tool_client_factory=StdioMCPToolClient,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
    )
    conversation_history = [
        ConversationMessage(role=message["role"], content=message["content"])
        for message in previous_history
        if message.get("role") in ("user", "assistant") and not message.get("error")
    ]
    return asyncio.run(
        agent.run(question, conversation_history=conversation_history)
    )


def _friendly_error(error: Exception) -> str:
    if isinstance(error, ConfigurationError):
        return (
            "The AI assistant is not configured. Check LLM_PROVIDER, LLM_MODEL, "
            "and the matching provider API key in your local .env file."
        )
    if isinstance(error, UnsupportedModelError):
        return (
            "The configured model is not supported for tool calling. Select a "
            "verified tool-capable model in your local configuration."
        )
    if isinstance(error, ProviderRequestError):
        return (
            "The AI provider could not complete the request. Check your network "
            "connection and provider status, then try again."
        )
    if isinstance(error, AgentIterationLimitError):
        return (
            "The assistant reached its tool-call limit before finishing. Try a "
            "more focused question."
        )
    if isinstance(error, ProviderProtocolError):
        return (
            "The model returned an empty or unusable response. Please try again."
        )
    if isinstance(error, RuntimeError):
        return (
            "The analytics service could not be reached. Check that the local "
            "MCP server and database are available."
        )
    if isinstance(error, AgentError):
        return "The assistant could not complete that request. Please try again."
    return _UNEXPECTED_ERROR


def _normalize_tool_activities(result: AgentResult) -> list[dict[str, Any]]:
    activities: list[dict[str, Any]] = []
    for activity in result.tool_activities:
        if not isinstance(activity.arguments, dict):
            continue
        activities.append(asdict(activity))
    return activities


def _render_tool_activities(activities: list[dict[str, Any]]) -> None:
    if not activities:
        return
    with st.expander(f"Tool activity · {len(activities)} call(s)", expanded=False):
        for index, activity in enumerate(activities, start=1):
            status = "Executed" if activity["executed"] else "Not executed"
            if activity["is_error"]:
                status += " · error returned"
            with st.container(border=True):
                st.markdown(f"**{index}. `{activity['name']}`** · {status}")
                with st.expander("Request and result", expanded=False):
                    st.caption("Arguments")
                    st.json(activity["arguments"])
                    st.caption("Tool result")
                    st.code(activity["result"] or "(empty result)", language="json")


def _render_history(history: list[dict[str, Any]]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.warning(message["content"])
            else:
                st.markdown(message["content"])
            _render_tool_activities(message.get("tool_activities", []))


def _append_question(history: list[dict[str, Any]], question: str) -> None:
    """Append one submitted turn and its result, guarding active duplicate work."""
    if st.session_state.get(_BUSY_KEY, False):
        return
    submission = question.strip()
    if not submission:
        return
    st.session_state[_BUSY_KEY] = True
    history.append({"role": "user", "content": submission})
    try:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question…"):
                result = _run_agent_question(submission, history[:-1])
            if not isinstance(result, AgentResult) or not result.answer.strip():
                raise ProviderProtocolError("The model returned an empty response.")
            assistant_message = {
                "role": "assistant",
                "content": result.answer.strip(),
                "tool_activities": _normalize_tool_activities(result),
            }
            history.append(assistant_message)
            _render_tool_activities(assistant_message["tool_activities"])
            st.markdown(assistant_message["content"])
    except (AgentError, ConfigurationError, OSError, RuntimeError) as error:
        assistant_message = {
            "role": "assistant",
            "content": _friendly_error(error),
            "error": True,
            "tool_activities": [],
        }
        history.append(assistant_message)
        with st.chat_message("assistant"):
            st.warning(assistant_message["content"])
    except Exception:
        assistant_message = {
            "role": "assistant",
            "content": _UNEXPECTED_ERROR,
            "error": True,
            "tool_activities": [],
        }
        history.append(assistant_message)
        with st.chat_message("assistant"):
            st.warning(assistant_message["content"])
    finally:
        st.session_state[_BUSY_KEY] = False


def render_assistant_page() -> None:
    """Render chat history and submit questions through the configured agent."""
    render_page_heading(
        "AI Assistant",
        "Ask about sales, products, customers, and inventory. "
        "Answers use the read-only analytics tools connected to your business data.",
        "ASK BIZBRIDGE",
    )

    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []
    history = st.session_state[CHAT_HISTORY_KEY]

    heading_column, action_column = st.columns((5, 2))
    with heading_column:
        st.markdown("#### Conversation")
        st.caption(
            f"{len(history) // 2} completed turn(s)"
            if history
            else "Your analysis conversation will appear here."
        )
    with action_column:
        new_conversation = st.button(
            "New chat",
            key="bizbridge_new_conversation",
            use_container_width=True,
            help="Start a new conversation",
        )
    if new_conversation:
        st.session_state[CHAT_HISTORY_KEY] = []
        st.rerun()

    if history:
        _render_history(history)
    else:
        render_empty_state(
            "Ask a business question below to explore sales, customers, products, or stock."
        )

    question = st.chat_input(
        "Ask a question about your business…",
        disabled=st.session_state.get(_BUSY_KEY, False),
        key="bizbridge_chat_input",
    )
    if question is not None:
        _append_question(history, question)
