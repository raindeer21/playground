import pytest

from agent_framework.llm_client import LLMClient


class StubAIMessage:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}


class StubChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        self.messages = messages
        return StubAIMessage(
            content="stubbed response",
            usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )


@pytest.mark.asyncio
async def test_chat_completion_uses_langchain(monkeypatch):
    monkeypatch.setattr("agent_framework.llm_client.ChatOpenAI", StubChatOpenAI)
    client = LLMClient(base_url="http://localhost:11434/v1", api_key="test-key")

    payload = {
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.1,
    }

    response = await client.chat_completion(payload)

    assert response["choices"][0]["message"]["content"] == "stubbed response"
    assert response["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
