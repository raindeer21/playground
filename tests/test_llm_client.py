import pytest

from agent_framework.llm_client import LLMClient


class StubAIMessage:
    def __init__(self, content, usage_metadata=None, tool_calls=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}
        self.tool_calls = tool_calls or []


class StubChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.bound_tools = None
        self.bound_tool_choice = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bound_tool_choice = kwargs.get("tool_choice")
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        if self.bound_tools:
            return StubAIMessage(
                content="",
                usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "next_action",
                        "args": {"decision": "final_response"},
                    }
                ],
            )
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
    assert response["choices"][0]["message"]["tool_calls"] is None
    assert response["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


@pytest.mark.asyncio
async def test_chat_completion_returns_tool_calls(monkeypatch):
    monkeypatch.setattr("agent_framework.llm_client.ChatOpenAI", StubChatOpenAI)
    client = LLMClient(base_url="http://localhost:11434/v1", api_key="test-key")

    response = await client.chat_completion(
        {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Plan next step"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "next_action",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "next_action"}},
        }
    )

    tool_calls = response["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "next_action"
    assert '"decision": "final_response"' in tool_calls[0]["function"]["arguments"]
