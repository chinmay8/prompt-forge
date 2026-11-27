"""Dynamic YAML-frontmatter and Markdown skill parser and registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.config import SKILLS_DIR

_SKILL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class SkillNotFoundError(KeyError):
    """Raised when a requested skill ID does not exist in the registry."""


class SkillValidationError(ValueError):
    """Raised when skill metadata or file creation parameters are invalid."""


@dataclass(slots=True)
class Skill:
    """Represents a prompting framework skill loaded from a Markdown file."""

    id: str
    name: str
    category: str
    description: str
    best_for: str
    trigger_keywords: list[str]
    body: str
    filepath: Path

    def to_summary_dict(self) -> dict[str, Any]:
        """Return compact metadata without the Markdown body for low-token routing."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "best_for": self.best_for,
            "trigger_keywords": list(self.trigger_keywords),
            "filepath": str(self.filepath),
        }


@dataclass(slots=True)
class SkillLoadError:
    """Represents a malformed or invalid skill file encountered during scanning."""

    filepath: Path
    error: str


def _coerce_keywords(raw_keywords: Any) -> list[str]:
    """Normalize trigger_keywords from YAML list or comma-separated string."""
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, str):
        return [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
    if isinstance(raw_keywords, list):
        return [str(kw).strip() for kw in raw_keywords if str(kw).strip()]
    return [str(raw_keywords).strip()]


def _read_frontmatter_only(filepath: Path) -> dict[str, Any]:
    """Read and parse only the YAML frontmatter block without loading the full body."""
    frontmatter_lines: list[str] = []
    with filepath.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
        if first_line.strip() != "---":
            raise SkillValidationError(
                "Missing opening YAML frontmatter delimiter '---' on line 1."
            )

        closed = False
        for line in handle:
            if line.strip() == "---":
                closed = True
                break
            frontmatter_lines.append(line)

        if not closed:
            raise SkillValidationError(
                "Missing closing YAML frontmatter delimiter '---'."
            )

    parsed = yaml.safe_load("".join(frontmatter_lines))
    if not isinstance(parsed, dict):
        raise SkillValidationError("YAML frontmatter must parse to a mapping/object.")
    return parsed


def _split_frontmatter_and_body(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split complete Markdown text into YAML frontmatter dict and Markdown body."""
    lines = raw_text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillValidationError(
            "Missing opening YAML frontmatter delimiter '---' on line 1."
        )

    closing_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break

    if closing_idx is None:
        raise SkillValidationError("Missing closing YAML frontmatter delimiter '---'.")

    yaml_text = "".join(lines[1:closing_idx])
    parsed = yaml.safe_load(yaml_text)
    if not isinstance(parsed, dict):
        raise SkillValidationError("YAML frontmatter must parse to a mapping/object.")

    body = "".join(lines[closing_idx + 1 :]).strip()
    if not body:
        raise SkillValidationError("Skill Markdown body cannot be empty.")

    return parsed, body


def _validate_frontmatter_fields(meta: dict[str, Any], filepath: Path) -> Skill:
    """Validate required frontmatter keys and construct a metadata-only Skill."""
    skill_id = str(meta.get("id", "")).strip()
    name = str(meta.get("name", "")).strip()
    category = str(meta.get("category", "General")).strip() or "General"
    description = str(meta.get("description", "")).strip()
    best_for = str(meta.get("best_for", "")).strip()
    trigger_keywords = _coerce_keywords(meta.get("trigger_keywords"))

    if not skill_id:
        raise SkillValidationError("Frontmatter is missing required field 'id'.")
    if not _SKILL_ID_PATTERN.match(skill_id):
        raise SkillValidationError(
            f"Invalid skill id '{skill_id}'. Use 2-64 lowercase alphanumeric, '_' or '-' characters."
        )
    if not name:
        raise SkillValidationError("Frontmatter is missing required field 'name'.")
    if not description:
        raise SkillValidationError("Frontmatter is missing required field 'description'.")

    return Skill(
        id=skill_id,
        name=name,
        category=category,
        description=description,
        best_for=best_for,
        trigger_keywords=trigger_keywords,
        body="",
        filepath=filepath,
    )


class SkillRegistry:
    """Discovers skills/*.md, indexes frontmatter, and lazy-loads Markdown bodies."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        self.skills_dir: Path = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
        self._metadata_by_id: dict[str, Skill] = {}
        self._load_errors: list[SkillLoadError] = []
        self.refresh()

    @property
    def load_errors(self) -> list[SkillLoadError]:
        """Return any errors recorded during the latest directory scan."""
        return list(self._load_errors)

    def refresh(self) -> None:
        """Scan skills/*.md and index only YAML frontmatter metadata."""
        self._metadata_by_id.clear()
        self._load_errors.clear()

        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            return

        for filepath in sorted(self.skills_dir.glob("*.md")):
            try:
                meta = _read_frontmatter_only(filepath)
                skill_meta = _validate_frontmatter_fields(meta, filepath)
                if skill_meta.id in self._metadata_by_id:
                    existing_path = self._metadata_by_id[skill_meta.id].filepath
                    raise SkillValidationError(
                        f"Duplicate skill id '{skill_meta.id}' (already defined in {existing_path.name})."
                    )
                self._metadata_by_id[skill_meta.id] = skill_meta
            except Exception as exc:
                self._load_errors.append(SkillLoadError(filepath=filepath, error=str(exc)))

    def list_skill_ids(self) -> list[str]:
        """Return all indexed skill IDs."""
        return list(self._metadata_by_id.keys())

    def get_catalog_summary(self) -> list[dict[str, Any]]:
        """Return compact metadata for all skills without loading bodies."""
        return [skill.to_summary_dict() for skill in self._metadata_by_id.values()]

    def get_skill_metadata(self, skill_id: str) -> Skill:
        """Return metadata-only Skill record without reading the Markdown body."""
        if skill_id not in self._metadata_by_id:
            raise SkillNotFoundError(
                f"Skill '{skill_id}' not found. Available skills: {list(self._metadata_by_id.keys())}"
            )
        meta = self._metadata_by_id[skill_id]
        return Skill(
            id=meta.id,
            name=meta.name,
            category=meta.category,
            description=meta.description,
            best_for=meta.best_for,
            trigger_keywords=list(meta.trigger_keywords),
            body="",
            filepath=meta.filepath,
        )

    def get_skill_by_id(self, skill_id: str) -> Skill:
        """Lazy-load the full Markdown body of the chosen skill from disk."""
        if skill_id not in self._metadata_by_id:
            raise SkillNotFoundError(
                f"Skill '{skill_id}' not found. Available skills: {list(self._metadata_by_id.keys())}"
            )
        meta_entry = self._metadata_by_id[skill_id]
        raw_text = meta_entry.filepath.read_text(encoding="utf-8-sig")
        parsed_meta, body = _split_frontmatter_and_body(raw_text)
        validated = _validate_frontmatter_fields(parsed_meta, meta_entry.filepath)
        validated.body = body
        return validated

    def get_raw_markdown(self, skill_id: str) -> str:
        """Return the complete raw Markdown file content for inspection."""
        if skill_id not in self._metadata_by_id:
            raise SkillNotFoundError(f"Skill '{skill_id}' not found.")
        return self._metadata_by_id[skill_id].filepath.read_text(encoding="utf-8-sig")

    def save_new_skill(
        self,
        skill_id: str,
        name: str,
        category: str,
        description: str,
        best_for: str,
        trigger_keywords: list[str] | str,
        body: str,
        overwrite: bool = False,
    ) -> Skill:
        """Validate, write a new skills/<skill_id>.md file, and refresh the registry."""
        clean_id = skill_id.strip().lower().replace(" ", "_")
        if not _SKILL_ID_PATTERN.match(clean_id):
            raise SkillValidationError(
                "Skill ID must be 2-64 characters using lowercase letters, numbers, '_' or '-'."
            )

        clean_name = name.strip()
        clean_category = category.strip() or "General"
        clean_description = description.strip()
        clean_best_for = best_for.strip()
        clean_keywords = _coerce_keywords(trigger_keywords)
        clean_body = body.strip()

        if not clean_name:
            raise SkillValidationError("Skill name is required.")
        if not clean_description:
            raise SkillValidationError("Skill description is required.")
        if not clean_body:
            raise SkillValidationError("Skill Markdown instructions body cannot be empty.")

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        target_path = (self.skills_dir / f"{clean_id}.md").resolve()
        if target_path.parent != self.skills_dir.resolve():
            raise SkillValidationError("Invalid skill file path.")

        if target_path.exists() and not overwrite:
            raise SkillValidationError(
                f"Skill file '{target_path.name}' already exists. Enable overwrite or choose a new ID."
            )

        frontmatter_payload: dict[str, Any] = {
            "id": clean_id,
            "name": clean_name,
            "category": clean_category,
            "description": clean_description,
            "best_for": clean_best_for or clean_description,
            "trigger_keywords": clean_keywords,
        }
        yaml_block = yaml.safe_dump(
            frontmatter_payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        content = f"---\n{yaml_block}\n---\n\n{clean_body}\n"
        target_path.write_text(content, encoding="utf-8")

        self.refresh()
        return self.get_skill_by_id(clean_id)
