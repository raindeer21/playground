from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class LLMClient:
    """LangChain-powered OpenAI-compatible chat client."""

    def __init__(self, base_url: str, api_key: str = "not-required") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    @staticmethod
    def _to_langchain_messages(messages: list[dict[str, Any]]) -> list[BaseMessage]:
        mapped: list[BaseMessage] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                mapped.append(SystemMessage(content=content))
            elif role == "assistant":
                mapped.append(AIMessage(content=content))
            else:
                mapped.append(HumanMessage(content=content))
        return mapped

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        if not model:
            raise ValueError("chat_completion payload must include 'model'.")

        client = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=payload.get("temperature", 0),
            max_retries=2,
        )
        ai_message = await client.ainvoke(self._to_langchain_messages(payload.get("messages", [])))

        usage = getattr(ai_message, "usage_metadata", None) or {}
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": ai_message.content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
