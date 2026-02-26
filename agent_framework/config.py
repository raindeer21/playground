from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    name: str
    kind: str = "mcp"
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class PromptConfig(BaseModel):
    name: str
    template: str
    variables: list[str] = Field(default_factory=list)


class RuntimeSettings(BaseModel):
    skills_dir: str = "examples/skills"
    skills_docs_index: str = "https://agentskills.io/llms.txt"


class AgentRuntimeConfig(BaseModel):
    tools: list[ToolConfig] = Field(default_factory=list)
    prompts: list[PromptConfig] = Field(default_factory=list)
    settings: RuntimeSettings = Field(default_factory=RuntimeSettings)


class ConfigStore:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.runtime_config = self._load_config(self.config_path)

    @staticmethod
    def _load_config(path: Path) -> AgentRuntimeConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return AgentRuntimeConfig.model_validate(payload or {})

    def list_tools(self) -> list[ToolConfig]:
        return self.runtime_config.tools

    def list_prompts(self) -> list[PromptConfig]:
        return self.runtime_config.prompts

    def settings(self) -> RuntimeSettings:
        return self.runtime_config.settings
