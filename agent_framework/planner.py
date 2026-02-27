from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm_client import LLMClient


class GatewayAction(BaseModel):
    step_id: str
    title: str
    objective: str
    required_skills: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_payload: dict[str, Any] = Field(default_factory=dict)


class GatewayNextAction(BaseModel):
    summary: str
    decision: Literal["run_tool", "ask_for_skill", "final_response"] = "final_response"
    action: GatewayAction | None = None
    final_response: str | None = None

    @property
    def is_done(self) -> bool:
        return self.decision == "final_response"


class PlanningGatewayAgent:
    """LLM-based planning/gateway layer with skill-header-first routing."""

    ASK_FOR_SKILL_TOOL = {
        "type": "function",
        "function": {
            "name": "ask_for_skill",
            "description": "Request one or more skills to be loaded before continuing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "step_id": {"type": "string"},
                    "title": {"type": "string"},
                    "objective": {"type": "string"},
                    "required_skills": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "step_id", "title", "objective", "required_skills"],
                "additionalProperties": False,
            },
        },
    }

    FINAL_RESPONSE_TOOL = {
        "type": "function",
        "function": {
            "name": "final_response",
            "description": "Finish the workflow and return the final response to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["summary", "response"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            return json.loads(raw_arguments or "{}")
        return {}

    def _extract_action_from_tool_call(self, response: dict[str, Any], available_tool_names: set[str]) -> GatewayNextAction:
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raise ValueError("Model did not return any tool call")

        call = tool_calls[0]
        function = call.get("function", {})
        name = function.get("name")
        arguments = self._parse_arguments(function.get("arguments"))

        if name == "ask_for_skill":
            return GatewayNextAction(
                summary=arguments.get("summary", "Requesting skills."),
                decision="ask_for_skill",
                action=GatewayAction(
                    step_id=arguments.get("step_id", "step-skill"),
                    title=arguments.get("title", "Load skill"),
                    objective=arguments.get("objective", "Load required skill content"),
                    required_skills=arguments.get("required_skills", []),
                    tool_name=None,
                    tool_payload={},
                ),
                final_response=None,
            )

        if name == "final_response":
            return GatewayNextAction(
                summary=arguments.get("summary", "Completed."),
                decision="final_response",
                action=None,
                final_response=arguments.get("response", ""),
            )

        if name in available_tool_names:
            return GatewayNextAction(
                summary=f"Running tool {name}.",
                decision="run_tool",
                action=GatewayAction(
                    step_id=f"step-{name.lower()}",
                    title=f"Run {name}",
                    objective=f"Execute {name} requested by planner",
                    required_skills=[],
                    tool_name=name,
                    tool_payload=arguments,
                ),
                final_response=None,
            )

        raise ValueError(f"Unknown function call: {name}")

    async def decide_next_action(
        self,
        model: str,
        user_request: str,
        selected_skills: list[dict],
        available_skills: list[dict],
        tool_specs: list[dict],
        execution_history: list[dict],
    ) -> GatewayNextAction:
        previous_step = execution_history[-1] if execution_history else {}
        previous_step_context = {
            "previous_action": {
                "step_id": previous_step.get("step_id"),
                "title": previous_step.get("title"),
                "objective": previous_step.get("objective"),
                "tool_name": previous_step.get("tool_name"),
                "tool_payload": previous_step.get("tool_payload"),
            },
            "previous_result": previous_step.get("tool_result"),
        }

        action_prompt = (
            "You are an execution coordinator. Always respond with exactly one function call and no text. "
            "Allowed function calls are ask_for_skill, final_response, and any provided external tool function. "
            "Use ask_for_skill when you need full skill bodies. "
            "Use external tool calls directly when execution is needed. "
            "Use final_response only when ready to answer the user."
        )
        request_blob = {
            "request": user_request,
            "selected_skills": selected_skills,
            "available_skills": available_skills,
            "tool_specs": tool_specs,
            "execution_history": execution_history,
            "previous_step_context": previous_step_context,
            "instruction": "Return exactly one function call only.",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": action_prompt},
                {"role": "user", "content": json.dumps(self._drop_empty_values(request_blob))},
            ],
            "tools": [self.ASK_FOR_SKILL_TOOL, self.FINAL_RESPONSE_TOOL, *tool_specs],
            "tool_choice": "required",
            "temperature": 0.1,
            "max_tokens": 400,
            "stream": False,
        }

        try:
            result = await self.llm_client.chat_completion(payload)
            available_tool_names = {spec.get("name") for spec in tool_specs if spec.get("name")}
            return self._extract_action_from_tool_call(result, available_tool_names)
        except Exception:
            return GatewayNextAction(
                summary="Fallback next action due to missing/invalid function call output.",
                decision="final_response",
                action=None,
                final_response="I could not produce a structured next action, so I am returning a safe fallback response.",
            )

    @staticmethod
    def _drop_empty_values(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {key: PlanningGatewayAgent._drop_empty_values(item) for key, item in value.items()}
            return {key: item for key, item in cleaned.items() if item is not None and item != "" and item != [] and item != {}}
        if isinstance(value, list):
            cleaned = [PlanningGatewayAgent._drop_empty_values(item) for item in value]
            return [item for item in cleaned if item is not None and item != "" and item != [] and item != {}]
        return value
