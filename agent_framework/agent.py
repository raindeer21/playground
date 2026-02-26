from __future__ import annotations

import time
import uuid

from .config import ConfigStore
from .llm_client import LLMClient
from .models import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from .planner import GatewayNextAction, PlanningGatewayAgent
from .prompts import PromptRegistry
from .skills import SkillStore
from .tools import ToolRegistry


class LiteAgentRuntime:
    """Config-driven sample runtime with LLM planning/gateway behavior."""

    def __init__(self, config_store: ConfigStore, llm_client: LLMClient) -> None:
        self.config_store = config_store
        self.llm_client = llm_client
        self.prompt_registry = PromptRegistry(config_store.list_prompts())
        self.tool_registry = ToolRegistry(config_store.list_tools())
        self.skill_store = SkillStore(config_store.settings().skills_dir)
        self.gateway_agent = PlanningGatewayAgent(llm_client)

    async def handle_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        user_text = "\n".join(msg.content for msg in request.messages if msg.role == "user")
        skill_headers = self.skill_store.headers()
        skill_plan = await self.gateway_agent.build_plan(request.model, user_text, skill_headers)

        selected_skill_headers = []
        selected_skills: list[str] = []
        for skill_name in skill_plan.required_skills:
            skill_manifest = self.skill_store.get(skill_name)
            if skill_manifest and skill_name not in selected_skills:
                selected_skills.append(skill_name)
                selected_skill_headers.append(skill_manifest.header)

        execution_history: list[dict] = []
        final_action: GatewayNextAction | None = None

        for _ in range(5):
            next_action = await self.gateway_agent.decide_next_action(
                request.model,
                user_text,
                selected_skill_headers,
                execution_history,
            )
            final_action = next_action

            if next_action.action:
                for skill in next_action.action.required_skills:
                    if skill not in selected_skills:
                        selected_skills.append(skill)
                        skill_manifest = self.skill_store.get(skill)
                        if skill_manifest:
                            selected_skill_headers.append(skill_manifest.header)

            if next_action.is_done:
                break

            if not next_action.action:
                execution_history.append(
                    {
                        "status": "error",
                        "message": "Planner did not provide an action while is_done is false.",
                    }
                )
                break

            tool_result = {"status": "skipped", "reason": "No tool requested"}
            if next_action.action.tool_name and next_action.action.tool_name in self.tool_registry.list_names():
                result = await self.tool_registry.call(
                    next_action.action.tool_name,
                    next_action.action.tool_payload or {"objective": next_action.action.objective},
                )
                tool_result = result.output

            execution_history.append(
                {
                    "step_id": next_action.action.step_id,
                    "title": next_action.action.title,
                    "objective": next_action.action.objective,
                    "required_skills": next_action.action.required_skills,
                    "tool_name": next_action.action.tool_name,
                    "tool_payload": next_action.action.tool_payload,
                    "tool_result": tool_result,
                }
            )

        requested_full_skills = bool(request.metadata.get("include_full_skills", False))

        system_messages = []
        for skill_name in selected_skills:
            skill_manifest = self.skill_store.get(skill_name)
            if not skill_manifest:
                continue
            if requested_full_skills:
                content = skill_manifest.body
            else:
                content = (
                    "Skill header only:\n"
                    f"{skill_manifest.header}\n"
                    "Set metadata.include_full_skills=true to request the full skill body."
                )
            system_messages.append(ChatMessage(role="system", content=content, name=skill_name))

        if system_messages:
            request.messages = [*system_messages, *request.messages]

        if execution_history:
            request.messages = [
                *request.messages,
                ChatMessage(
                    role="system",
                    content=f"Execution history: {execution_history}",
                    name="gateway-execution-history",
                ),
            ]

        if final_action and final_action.is_done and final_action.final_response:
            assistant_message = ChatMessage(role="assistant", content=final_action.final_response)
            upstream_usage = {}
        else:
            upstream = await self.llm_client.chat_completion(request.model_dump())
            assistant_message = ChatMessage.model_validate(upstream["choices"][0]["message"])
            upstream_usage = upstream.get("usage", {})

        full_skills = None
        if requested_full_skills:
            full_skills = {
                name: self.skill_store.get(name).body
                for name in selected_skills
                if self.skill_store.get(name)
            }

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(message=assistant_message)],
            usage=upstream_usage,
            gateway_plan={
                "summary": skill_plan.summary,
                "selected_skills": selected_skills,
                "execution_summary": final_action.summary if final_action else "No execution output.",
                "is_done": final_action.is_done if final_action else False,
                "last_action": final_action.action.model_dump() if final_action and final_action.action else None,
                "execution_history": execution_history,
            },
            skill_headers=[self.skill_store.get(name).header for name in selected_skills if self.skill_store.get(name)],
            full_skills=full_skills,
        )
