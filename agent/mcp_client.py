"""MCP stdio client and safe tool-result normalization."""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.types import ToolDefinition, ToolResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "server_stubs.py"


class StdioMCPToolClient:
    """Lifecycle-managed client for the project's existing stdio MCP server."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> StdioMCPToolClient:
        try:
            read_stream, write_stream = await self._stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(SERVER_SCRIPT)],
                        cwd=str(PROJECT_ROOT),
                    )
                )
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self._stack.aclose()
            raise RuntimeError("Unable to connect to the BizBridge MCP server.") from None
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[ToolDefinition]:
        if self._session is None:
            raise RuntimeError("MCP tool client is not connected.")
        try:
            response = await self._session.list_tools()
        except Exception:
            raise RuntimeError("Unable to discover BizBridge MCP tools.") from None
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in response.tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if self._session is None:
            raise RuntimeError("MCP tool client is not connected.")
        try:
            response = await self._session.call_tool(name, arguments)
        except Exception:
            return ToolResult(
                content=json.dumps({"error": "Tool execution failed safely."}),
                is_error=True,
            )
        if response.isError:
            return ToolResult(
                content=json.dumps({"error": "Tool rejected the request."}),
                is_error=True,
            )
        structured = response.structuredContent
        if structured is not None:
            return ToolResult(content=json.dumps(structured, ensure_ascii=False))
        text = "\n".join(
            item.text
            for item in response.content
            if getattr(item, "type", None) == "text"
        )
        return ToolResult(content=text or "null")
