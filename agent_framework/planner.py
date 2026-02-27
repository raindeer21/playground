from __future__ import annotations

import json
import re
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

    @staticmethod
    def _extract_structured_json(raw: str) -> dict[str, Any]:
        """Extract first valid JSON object from model output."""

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\\s*", "", text)
            text = re.sub(r"\\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[idx:])
                if isinstance(candidate, dict):
                    return candidate
            except json.JSONDecodeError:
                continue

        raise ValueError("No JSON object found in model output")

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
            "You are an execution coordinator. "
            "Given the user request, selected_skills (full content only for skills previously requested via ask_for_skill), available_skills (headers for all skills), available tool specs, and execution history, choose exactly one option. "
            "Allowed options are: run_tool, ask_for_skill, final_response. "
            "Return JSON only with keys: summary, decision, action, final_response. "
            "Do not include markdown fences, commentary, or any extra fields. "
            "Never include reasoning_content or chain-of-thought. "
            "For run_tool: set decision=run_tool and include action with step_id, title, objective, required_skills, tool_name, tool_payload. "
            "For ask_for_skill: set decision=ask_for_skill and include action with required_skills listing skills to load; tool_name must be null. "
            "For final_response: set decision=final_response, action=null, and include final_response. "
            "If selected_skills is empty, ask_for_skill is allowed when a skill is needed; otherwise continue with tools or final_response. "
            "Example run_tool output: {\"summary\":\"Need fetch\",\"decision\":\"run_tool\",\"action\":{\"step_id\":\"step-1\",\"title\":\"Fetch listings\",\"objective\":\"Query listings API\",\"required_skills\":[\"housing-search\"],\"tool_name\":\"WebRequest\",\"tool_payload\":{\"method\":\"GET\",\"url\":\"https://example.com/listings\"}},\"final_response\":null}. "
            "Example ask_for_skill output: {\"summary\":\"Need policy details\",\"decision\":\"ask_for_skill\",\"action\":{\"step_id\":\"step-2\",\"title\":\"Load skill\",\"objective\":\"Load repo-assistant full content\",\"required_skills\":[\"repo-assistant\"],\"tool_name\":null,\"tool_payload\":{}},\"final_response\":null}. "
            "Example final_response output: {\"summary\":\"Need user input\",\"decision\":\"final_response\",\"action\":null,\"final_response\":\"Please share budget and preferred district.\"}. "
            "For every turn, use previous_action and previous_result when present, then decide and return the explicit next step. "
            "Always answer the question: what is the single best next step now?"
        )
        request_blob = {
            "request": user_request,
            "selected_skills": selected_skills,
            "available_skills": available_skills,
            "tool_specs": tool_specs,
            "execution_history": execution_history,
            "previous_step_context": previous_step_context,
            "instruction": "Return the next step only as JSON with summary/decision/action/final_response.",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": action_prompt},
                {
                    "role": "user",
                    "content": json.dumps(self._drop_empty_values(request_blob)),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "stream": False,
        }
        result = await self.llm_client.chat_completion(payload)
        raw = result["choices"][0]["message"]["content"]

        try:
            data = self._extract_structured_json(raw)
            return GatewayNextAction.model_validate(data)
        except Exception:
            return GatewayNextAction(
                summary="Fallback next action due to non-JSON action output.",
                decision="final_response",
                action=None,
                final_response="I could not produce a structured next action, so I am returning a safe fallback response.",
            )

    @staticmethod
    def _drop_empty_values(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {
                key: PlanningGatewayAgent._drop_empty_values(item)
                for key, item in value.items()
            }
            return {
                key: item
                for key, item in cleaned.items()
                if item is not None and item != "" and item != [] and item != {}
            }
        if isinstance(value, list):
            cleaned = [PlanningGatewayAgent._drop_empty_values(item) for item in value]
            return [item for item in cleaned if item is not None and item != "" and item != [] and item != {}]
        return value
