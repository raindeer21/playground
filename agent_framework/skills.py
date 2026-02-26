from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NAME_PATTERN = re.compile(r"^(?!-)(?!.*--)[a-z0-9-]{1,64}(?<!-)$")


@dataclass(slots=True)
class SkillManifest:
    name: str
    description: str
    header: dict[str, Any]
    body: str
    path: Path


class SkillSpecError(ValueError):
    pass


class SkillStore:
    """Loads spec-compliant Agent Skills from a local directory."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills = self._load_skills()

    def _load_skills(self) -> dict[str, SkillManifest]:
        manifests: dict[str, SkillManifest] = {}
        if not self.skills_dir.exists():
            return manifests

        for child in self.skills_dir.iterdir():
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.exists():
                continue
            manifest = self._parse_skill(child.name, skill_file)
            manifests[manifest.name] = manifest
        return manifests

    def _parse_skill(self, folder_name: str, skill_file: Path) -> SkillManifest:
        text = skill_file.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise SkillSpecError(f"{skill_file}: missing YAML frontmatter")
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise SkillSpecError(f"{skill_file}: malformed frontmatter")
        _, frontmatter, body = parts
        data = yaml.safe_load(frontmatter) or {}

        name = data.get("name", "")
        description = data.get("description", "")

        if not NAME_PATTERN.match(name):
            raise SkillSpecError(f"{skill_file}: invalid skill name '{name}'")
        if name != folder_name:
            raise SkillSpecError(f"{skill_file}: name must match directory '{folder_name}'")
        if not isinstance(description, str) or not (1 <= len(description) <= 1024):
            raise SkillSpecError(f"{skill_file}: invalid description")

        return SkillManifest(
            name=name,
            description=description,
            header=data,
            body=body.strip(),
            path=skill_file,
        )

    def list_skills(self) -> list[SkillManifest]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillManifest | None:
        return self._skills.get(name)

    def headers(self) -> list[dict[str, Any]]:
        return [skill.header for skill in self._skills.values()]
