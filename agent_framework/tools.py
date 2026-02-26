from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ToolConfig


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    output: dict[str, Any]


class MCPToolAdapter:
    """Placeholder MCP tool adapter.

    Replace `execute` with the MCP transport/client library you prefer.
    This class intentionally stays generic so runtime YAML controls behavior.
    """

    def __init__(self, config: ToolConfig) -> None:
        self.config = config

    async def execute(self, payload: dict[str, Any]) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=self.config.name,
            output={
                "status": "example",
                "settings": self.config.settings,
                "received": payload,
            },
        )


class ToolRegistry:
    def __init__(self, tools: list[ToolConfig]) -> None:
        self._tools = {tool.name: MCPToolAdapter(tool) for tool in tools}

    def list_names(self) -> list[str]:
        return list(self._tools)

    async def call(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        tool = self._tools[tool_name]
        return await tool.execute(payload)
