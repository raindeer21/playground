from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .llm_client import LLMClient


class GatewaySkillPlan(BaseModel):
    summary: str
    required_skills: list[str] = Field(default_factory=list)


class GatewayAction(BaseModel):
    step_id: str
    title: str
    objective: str
    required_skills: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    tool_payload: dict[str, str] = Field(default_factory=dict)


class GatewayNextAction(BaseModel):
    summary: str
    is_done: bool = False
    action: GatewayAction | None = None
    final_response: str | None = None


class PlanningGatewayAgent:
    """LLM-based planning/gateway layer with skill-header-first routing."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        model: str,
        user_request: str,
        skill_headers: list[dict],
    ) -> GatewaySkillPlan:
        planner_prompt = (
            "You are a planning gateway for an agent framework. "
            "Given a user request and available skill headers, choose required skills for the next round. "
            "Only use skill names from the provided headers. Return JSON only with keys: summary, required_skills."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": planner_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": user_request,
                            "skill_headers": skill_headers,
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
            return GatewaySkillPlan.model_validate(data)
        except Exception:
            return GatewaySkillPlan(
                summary="Fallback plan due to non-JSON planner output.",
                required_skills=[],
            )

    async def decide_next_action(
        self,
        model: str,
        user_request: str,
        selected_skills: list[dict],
        execution_history: list[dict],
    ) -> GatewayNextAction:
        action_prompt = (
            "You are an execution coordinator. "
            "Given the user request, selected skills, and execution history, decide the next action. "
            "Return JSON only with keys: summary, is_done, action, final_response. "
            "If is_done=true, set action=null and include final_response. "
            "If is_done=false, set action with: step_id, title, objective, required_skills, tool_name, tool_payload."
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
                is_done=True,
                action=None,
                final_response="I could not produce a structured next action, so I am returning a safe fallback response.",
            )
