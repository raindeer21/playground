from __future__ import annotations

import time
import uuid

from .config import ConfigStore
from .llm_client import LLMClient
from .models import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from .planner import PlanningGatewayAgent
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
        plan = await self.gateway_agent.build_plan(request.model, user_text, skill_headers)

        requested_full_skills = bool(request.metadata.get("include_full_skills", False))
        selected_skills = plan.steps[1].required_skills if len(plan.steps) > 1 else []

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

        upstream = await self.llm_client.chat_completion(request.model_dump())
        assistant_message = ChatMessage.model_validate(upstream["choices"][0]["message"])

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
            usage=upstream.get("usage", {}),
            gateway_plan=plan.model_dump(),
            skill_headers=[self.skill_store.get(name).header for name in selected_skills if self.skill_store.get(name)],
            full_skills=full_skills,
        )
