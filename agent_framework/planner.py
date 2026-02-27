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

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def decide_next_action(
        self,
        model: str,
        user_request: str,
        selected_skills: list[dict],
        tool_names: list[str],
        execution_history: list[dict],
    ) -> GatewayNextAction:
        action_prompt = (
            "You are an execution coordinator. "
            "Given the user request, selected skill headers, available tools, and execution history, choose exactly one option. "
            "Allowed options are: run_tool, ask_for_skill, final_response. "
            "Return JSON only with keys: summary, decision, action, final_response. "
            "For run_tool: set decision=run_tool and include action with step_id, title, objective, required_skills, tool_name, tool_payload. "
            "For ask_for_skill: set decision=ask_for_skill and include action with required_skills listing skills to load; tool_name must be null. "
            "For final_response: set decision=final_response, action=null, and include final_response."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": action_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": user_request,
                            "selected_skills": selected_skills,
                            "tool_names": tool_names,
                            "execution_history": execution_history,
                        }
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 700,
            "stream": False,
        }
        result = await self.llm_client.chat_completion(payload)
        raw = result["choices"][0]["message"]["content"]

        try:
            data = json.loads(raw)
            return GatewayNextAction.model_validate(data)
        except Exception:
            return GatewayNextAction(
                summary="Fallback next action due to non-JSON action output.",
                decision="final_response",
                action=None,
                final_response="I could not produce a structured next action, so I am returning a safe fallback response.",
            )
