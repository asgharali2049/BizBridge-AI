"""Interactive command-line entry point for BizBridge AI's analytics agent."""

from __future__ import annotations

import asyncio
import sys

from agent.config import AgentConfig
from agent.errors import AgentError
from agent.mcp_client import StdioMCPToolClient
from agent.orchestrator import Agent
from agent.providers.factory import create_provider


async def _run(question: str) -> str:
    config = AgentConfig.from_env()
    provider = create_provider(config)
    agent = Agent(
        provider=provider,
        tool_client_factory=StdioMCPToolClient,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
    )
    return (await agent.run(question)).answer


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Business question: ").strip()
    if not question:
        print("Error: the business question cannot be empty.", file=sys.stderr)
        return 2
    try:
        print(asyncio.run(_run(question)))
    except AgentError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
