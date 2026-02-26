from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .llm_client import LLMClient


class PlanStep(BaseModel):
    step_id: str
    title: str
    objective: str
    required_skills: list[str] = Field(default_factory=list)


class GatewayPlan(BaseModel):
    summary: str
    steps: list[PlanStep]


class PlanningGatewayAgent:
    """LLM-based planning/gateway layer with skill-header-first routing."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def build_plan(self, model: str, user_request: str, skill_headers: list[dict]) -> GatewayPlan:
        planner_prompt = (
            "You are a planning gateway for an agent framework. "
            "Given a user request and available skill headers, generate a concrete plan and map required skills per step. "
            "Only use skill names from the provided headers. Return JSON only with keys: summary, steps. "
            "Each step must include: step_id, title, objective, required_skills."
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
                            "constraints": {
                                "steps_min": 3,
                                "steps_max": 7,
                                "skill_header_only": True,
                            },
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
            return GatewayPlan.model_validate(data)
        except Exception:
            return GatewayPlan(
                summary="Fallback plan due to non-JSON planner output.",
                steps=[
                    PlanStep(
                        step_id="step-1",
                        title="Understand request",
                        objective="Parse user goal and constraints.",
                        required_skills=[],
                    ),
                    PlanStep(
                        step_id="step-2",
                        title="Match candidate skills",
                        objective="Use skill headers to select relevant skills.",
                        required_skills=[],
                    ),
                    PlanStep(
                        step_id="step-3",
                        title="Execute and respond",
                        objective="Run workflow and return final answer.",
                        required_skills=[],
                    ),
                ],
            )
