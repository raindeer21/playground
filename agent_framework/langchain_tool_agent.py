from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from .tools import ToolRegistry

_JSON_TYPE_TO_PYTHON: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class LangChainToolCallingAgent:
    """Tool-calling agent built directly on create_tool_calling_agent + AgentExecutor."""

    def __init__(
        self,
        *,
        model: str,
        tool_registry: ToolRegistry,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0,
        system_prompt: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or (
            "You are a practical execution agent. Use tools whenever they help answer the user request. "
            "Return a concise final response."
        )
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        self.langchain_tools = self._build_langchain_tools()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.langchain_tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            verbose=verbose,
            return_intermediate_steps=True,
        )

    async def ainvoke(self, user_input: str, chat_history: list[Any] | None = None) -> dict[str, Any]:
        """Run one end-to-end tool-calling cycle through AgentExecutor."""
        payload: dict[str, Any] = {"input": user_input}
        if chat_history:
            payload["chat_history"] = chat_history
        return await self.executor.ainvoke(payload)

    def _build_langchain_tools(self) -> list[StructuredTool]:
        tools: list[StructuredTool] = []
        for spec in self.tool_registry.list_specs():
            tool_name = spec["name"]
            args_schema = _pydantic_from_json_schema(tool_name, spec.get("args_schema") or {})

            async def _tool_runner(_tool_name: str = tool_name, **kwargs: Any) -> dict[str, Any]:
                result = await self.tool_registry.call(_tool_name, kwargs)
                return result.output

            tools.append(
                StructuredTool.from_function(
                    coroutine=_tool_runner,
                    name=tool_name,
                    description=spec.get("description", ""),
                    args_schema=args_schema,
                )
            )
        return tools


def _pydantic_from_json_schema(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type")
        python_type = _JSON_TYPE_TO_PYTHON.get(json_type, Any)
        description = field_schema.get("description", "")

        if field_name in required:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            default = field_schema.get("default")
            fields[field_name] = (python_type | None, Field(default=default, description=description))

    if not fields:
        fields["input"] = (dict[str, Any], Field(default_factory=dict, description="Tool input payload"))

    return create_model(f"{tool_name}Args", **fields)
