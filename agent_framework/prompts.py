from __future__ import annotations

from string import Template

from .config import PromptConfig


class PromptRegistry:
    def __init__(self, prompts: list[PromptConfig]) -> None:
        self._prompts = {prompt.name: prompt for prompt in prompts}

    def render(self, prompt_name: str, values: dict[str, str]) -> str:
        prompt = self._prompts[prompt_name]
        templated = Template(prompt.template)
        return templated.safe_substitute(values)

    def list_names(self) -> list[str]:
        return list(self._prompts)
