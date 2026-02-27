from __future__ import annotations

import json
from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
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
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")

        if tools:
            bind_kwargs: dict[str, Any] = {}
            if tool_choice is not None:
                bind_kwargs["tool_choice"] = tool_choice
            client = client.bind_tools(tools, **bind_kwargs)

        ai_message = await client.ainvoke(self._to_langchain_messages(payload.get("messages", [])))

        usage = getattr(ai_message, "usage_metadata", None) or {}
        tool_calls = []
        for call in getattr(ai_message, "tool_calls", []) or []:
            arguments = call.get("args", {})
            if isinstance(arguments, str):
                arguments = arguments
            else:
                arguments = json.dumps(arguments)
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": ai_message.content,
                        "tool_calls": tool_calls or None,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    async def plan_with_agent_executor(
        self,
        *,
        model: str,
        system_prompt: str,
        user_input: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Use LangChain AgentExecutor to return a single tool call decision."""

        llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_retries=2,
        )

        async def _capture_tool_call(**kwargs: Any) -> str:
            return json.dumps(kwargs)

        lc_tools: list[StructuredTool] = []
        for tool in tools:
            tool_name = tool.get("name") or tool.get("function", {}).get("name")
            if not tool_name:
                continue
            description = tool.get("description") or tool.get("function", {}).get("description") or ""
            lc_tools.append(
                StructuredTool.from_function(
                    coroutine=_capture_tool_call,
                    func=None,
                    name=tool_name,
                    description=description,
                )
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, lc_tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=lc_tools,
            verbose=False,
            max_iterations=1,
            return_intermediate_steps=True,
            early_stopping_method="force",
        )

        result = await executor.ainvoke({"input": user_input})
        intermediate_steps = result.get("intermediate_steps") or []
        if not intermediate_steps:
            raise ValueError("AgentExecutor did not produce any tool call")

        first_action = intermediate_steps[0][0]
        tool_name = getattr(first_action, "tool", "")
        tool_input = getattr(first_action, "tool_input", {})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_agent_executor",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_input),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {},
        }
