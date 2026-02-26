from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import ToolConfig


@dataclass(slots=True)
class ToolExecutionResult:
    tool_name: str
    output: dict[str, Any]


async def web_request_tool(payload: dict[str, Any]) -> dict[str, Any]:
    method = str(payload.get("method", "GET")).upper()
    url = payload.get("url")
    headers = payload.get("headers") or {}
    request_payload = payload.get("payload")

    if not url:
        return {
            "status": "error",
            "error": "Missing required field: url",
        }

    if not isinstance(headers, dict):
        return {
            "status": "error",
            "error": "headers must be an object/dictionary",
        }

    try:
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(
                method=method,
                url=str(url),
                headers={str(k): str(v) for k, v in headers.items()},
                json=request_payload if isinstance(request_payload, (dict, list)) else None,
                data=request_payload if isinstance(request_payload, (str, bytes)) else None,
                content=request_payload if isinstance(request_payload, bytes) else None,
            )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Request failed: {exc}",
        }

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    return {
        "status": "ok",
        "request": {
            "method": method,
            "url": str(url),
        },
        "response": {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        },
    }


class ToolRegistry:
    def __init__(self, tools: list[ToolConfig]) -> None:
        self._tool_names = [tool.name for tool in tools if tool.name == "WebRequest"]

    def list_names(self) -> list[str]:
        return self._tool_names

    async def call(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        if tool_name != "WebRequest" or tool_name not in self._tool_names:
            raise KeyError(f"Unknown tool: {tool_name}")

        output = await web_request_tool(payload)
        return ToolExecutionResult(tool_name=tool_name, output=output)
