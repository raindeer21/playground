from __future__ import annotations

from typing import Any

import httpx


class LLMClient:
    """Simple OpenAI-compatible client targeting local/offline-hosted API."""

    def __init__(self, base_url: str, api_key: str = "not-required") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(trust_env=False, timeout=60.0) as client:
            print(payload)
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
