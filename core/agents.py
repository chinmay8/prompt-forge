"""Multi-agent prompt engineering pipeline and orchestration logic."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from core.config import (
    AUTO_MODE_ID,
    CLASSIFY_TEMP,
    GENERATE_TEMP,
    MAX_RETRIES,
    MIN_PASS_SCORE,
    VALIDATE_TEMP,
)
from core.openai_client import OpenAIClient, OpenAIClientError
from core.skill_loader import Skill, SkillRegistry

_PREAMBLE_LINE_RE = re.compile(
    r"^(?:sure[!,.]?|certainly[!,.]?|of course[!,.]?|here(?:'s| is)\s+"
    r"(?:the|your)\s+(?:engineered|final|updated|revised|formatted)?\s*prompt:?)\s*$",
    re.IGNORECASE,
)
_OUTER_FENCE_RE = re.compile(
    r"^```(?:markdown|md|text|prompt)?\s*\n(.*?)\n```$",
    re.DOTALL | re.IGNORECASE,
)
_ROOT_SECTION_HEADERS: tuple[str, ...] = (
    "ROLE & RESPONSIBILITY:",
    "### Context",
    "# **Role:**",
    "**Role:**",
)
_TRAILING_PROMPT_ECHO_RE = re.compile(
    r"\n-{3,}\s*\n+\s*(?:\*\*)?(?:Final\s+)?Prompt:?(?:\*\*)?\s*\n.*$",
    re.IGNORECASE | re.DOTALL,
)


class InferredParameters(BaseModel):
    """Domain and target modality inferred from the user's meta-prompt."""

    domain: str = "General"
    output_modality: str = "Text / System Prompt"


class ClassificationResult(BaseModel):
    """Structured output returned by ClassifierAgent (or synthesized on manual override)."""

    selected_skill_id: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    intent_summary: str = ""
    rationale: str = ""
    inferred_parameters: InferredParameters = Field(default_factory=InferredParameters)
    is_manual_override: bool = False


class ValidationResult(BaseModel):
    """Structured quality audit returned by ValidatorAgent."""

    is_valid: bool
    score: int = Field(ge=1, le=10)
    framework_compliance: Literal["PASS", "FAIL"] = "PASS"
    intent_alignment: Literal["PASS", "FAIL"] = "PASS"
    compliance_notes: str = ""
    critique: str = ""
    improvement_instructions: str = ""


class PipelineAttempt(BaseModel):
    """Record of one Generator + Validator iteration."""

    attempt_number: int
    generated_prompt: str
    validation: ValidationResult
    duration_seconds: float
    critique_used: str = ""


class PipelineTrace(BaseModel):
    """Complete execution trace across Classify -> Generate -> Validate."""

    user_meta_prompt: str
    model: str
    selection_mode: Literal["Auto", "Manual"]
    classification: ClassificationResult
    skill_id: str
    skill_name: str
    skill_filepath: str
    attempts: list[PipelineAttempt]
    final_prompt: str
    final_validation: ValidationResult
    passed: bool
    retries_used: int
    total_duration_seconds: float


ProgressCallback = Callable[[str, dict[str, Any]], None]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce model-returned booleans safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "yes", "1", "pass", "passed", "valid"}:
            return True
        if norm in {"false", "no", "0", "fail", "failed", "invalid"}:
            return False
    return default


def _coerce_score(value: Any, default: int = 5) -> int:
    """Coerce and clamp score into 1..10."""
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = default
    return max(1, min(10, numeric))


def _coerce_confidence(value: Any, default: float = 0.85) -> float:
    """Coerce confidence into 0.0..1.0 (handling percentage inputs like 92)."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num > 1.0 and num <= 100.0:
        num = num / 100.0
    return max(0.0, min(1.0, round(num, 3)))


def _coerce_pass_fail(value: Any, fallback_bool: bool) -> Literal["PASS", "FAIL"]:
    """Normalize PASS/FAIL literals."""
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper in {"PASS", "PASSED", "TRUE", "OK"}:
            return "PASS"
        if upper in {"FAIL", "FAILED", "FALSE"}:
            return "FAIL"
    return "PASS" if fallback_bool else "FAIL"


def _deduplicate_repeated_prompt(text: str) -> str:
    """Remove duplicated prompt blocks sometimes emitted by small 1B-1.5B models."""
    stripped = text.strip()
    if not stripped:
        return ""

    # 1. Remove trailing `---\n**Prompt:** ...` echo when structured sections already exist
    has_structured_sections = any(
        marker in stripped
        for marker in ("OBJECTIVE:", "### Objective", "# **Objective:**", "**Task:**")
    )
    if has_structured_sections:
        stripped = _TRAILING_PROMPT_ECHO_RE.sub("", stripped).strip()

    # 2. If a root framework header repeats outside code fences, keep the first complete block
    lines = stripped.splitlines()
    for root_header in _ROOT_SECTION_HEADERS:
        in_fence = False
        occurrences: list[int] = []
        for idx, line in enumerate(lines):
            sline = line.strip()
            if sline.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and sline == root_header:
                occurrences.append(idx)
        if len(occurrences) >= 2 and occurrences[0] <= 2:
            cutoff = occurrences[1]
            candidate_lines = lines[:cutoff]
            while candidate_lines and (
                not candidate_lines[-1].strip()
                or set(candidate_lines[-1].strip()) <= {"-"}
            ):
                candidate_lines.pop()
            if candidate_lines:
                stripped = "\n".join(candidate_lines).strip()
                lines = stripped.splitlines()

    # 3. Exact repeated halves (with or without a `---` divider or blank lines between them)
    divider_parts = re.split(r"\n\s*-{3,}\s*\n", stripped)
    if len(divider_parts) == 2 and divider_parts[0].strip() == divider_parts[1].strip():
        return divider_parts[0].strip()

    all_lines = stripped.splitlines()
    non_empty_indices = [idx for idx, line in enumerate(all_lines) if line.strip()]
    if len(non_empty_indices) >= 2 and len(non_empty_indices) % 2 == 0:
        half_count = len(non_empty_indices) // 2
        first_norm = [all_lines[i].strip() for i in non_empty_indices[:half_count]]
        second_norm = [all_lines[i].strip() for i in non_empty_indices[half_count:]]
        if first_norm == second_norm:
            last_idx = non_empty_indices[half_count - 1]
            return "\n".join(all_lines[: last_idx + 1]).strip()

    return stripped


def _clean_generated_output(text: str) -> str:
    """Strip accidental outer wrapper fences, LLM preambles, and duplicate prompt blocks."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    # Unwrap only if the entire output is enclosed in a single outer code fence
    fence_match = _OUTER_FENCE_RE.match(cleaned)
    if fence_match:
        inner = fence_match.group(1)
        if "```" not in inner:
            cleaned = inner.strip()

    lines = cleaned.splitlines()
    if lines and _PREAMBLE_LINE_RE.match(lines[0].strip()):
        cleaned = "\n".join(lines[1:]).strip()

    return _deduplicate_repeated_prompt(cleaned)


_VISUAL_SIGNAL_TERMS: tuple[str, ...] = (
    "image",
    "video",
    "photo",
    "photograph",
    "picture",
    "scene",
    "camera",
    "lens",
    "lighting",
    "cinematic",
    "movie",
    "shot",
    "render",
    "midjourney",
    "sora",
    "runway",
    "veo",
    "diffusion",
    "visual",
    "illustration",
    "concept art",
    "artwork",
    "poster",
    "logo",
    "draw",
    "paint",
    "portrait",
    "anime",
    "3d",
    "cgi",
    "film",
    "cyberpunk",
    "aesthetic",
    "vibe",
    "mood",
    "mashup",
)


def _keyword_matches_prompt(keyword: str, prompt_lower: str) -> bool:
    """Match a trigger keyword/phrase on word boundaries inside prompt_lower."""
    kw = keyword.strip().lower()
    if not kw:
        return False
    if " " in kw:
        pattern = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
        return bool(re.search(pattern, prompt_lower))
    # For single tokens, avoid matching hyphenated compound modifiers like 'multi-agent'
    pattern = rf"(?<![a-z0-9-]){re.escape(kw)}(?![a-z0-9-])"
    return bool(re.search(pattern, prompt_lower))


def _score_catalog_keywords(
    user_meta_prompt: str,
    catalog: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Return matched trigger_keywords per skill ID for the given user meta-prompt."""
    prompt_lower = user_meta_prompt.lower()
    matches: dict[str, list[str]] = {}
    for entry in catalog:
        matched = [
            str(kw).strip()
            for kw in entry.get("trigger_keywords", [])
            if _keyword_matches_prompt(str(kw), prompt_lower)
        ]
        matches[entry["id"]] = matched
    return matches


def _has_visual_intent(user_meta_prompt: str, catalog: list[dict[str, Any]]) -> bool:
    """Return True if the meta-prompt contains any image/video visual trigger terms."""
    prompt_lower = user_meta_prompt.lower()
    visual_terms = set(_VISUAL_SIGNAL_TERMS)
    for entry in catalog:
        if "image" in str(entry.get("category", "")).lower() or "video" in str(entry.get("category", "")).lower():
            for kw in entry.get("trigger_keywords", []):
                visual_terms.add(str(kw).strip().lower())
    return any(_keyword_matches_prompt(term, prompt_lower) for term in visual_terms)


_AGENT_ACTION_RE = re.compile(
    r"\b(?:build|create|design|develop|configure|implement|architect|make|need)\b.{0,30}\b(?:agent|bot)\b",
    re.IGNORECASE,
)
_AGENT_TECH_TERMS: tuple[str, ...] = (
    "autonomous",
    "research agent",
    "bot",
    "tool",
    "tools",
    "tool-calling",
    "workflow",
    "pipeline",
    "guardrails",
    "langgraph",
    "crewai",
    "autogen",
)


def _has_agent_design_intent(user_meta_prompt: str) -> bool:
    """Return True if the user is asking to design/build an AI agent rather than just mentioning one."""
    lower = user_meta_prompt.lower()
    if _AGENT_ACTION_RE.search(lower):
        return True
    return any(_keyword_matches_prompt(term, lower) for term in _AGENT_TECH_TERMS)


def _resolve_skill_id(
    candidate_id: str,
    user_meta_prompt: str,
    catalog: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Ensure selected_skill_id maps to a valid skill in the catalog."""
    if not catalog:
        raise ValueError("No framework skills are loaded in the registry.")

    valid_ids = [entry["id"] for entry in catalog]
    norm_candidate = candidate_id.strip().lower().replace("-", "_").replace(" ", "_")

    if norm_candidate in valid_ids:
        return norm_candidate, None

    # Exact or substring match against id or name
    for entry in catalog:
        eid = entry["id"]
        ename = entry["name"].lower().replace(" ", "_")
        if eid in norm_candidate or norm_candidate in eid or norm_candidate in ename:
            return eid, f"Normalized classifier skill '{candidate_id}' -> '{eid}'."

    # Keyword overlap fallback against user meta-prompt
    keyword_matches = _score_catalog_keywords(user_meta_prompt, catalog)
    best_id = max(valid_ids, key=lambda sid: len(keyword_matches.get(sid, [])))
    best_hits = len(keyword_matches.get(best_id, []))

    if best_hits <= 0 and "costar" in valid_ids:
        best_id = "costar"

    return (
        best_id,
        f"Classifier returned unrecognized skill '{candidate_id}'; routed to '{best_id}' via keyword fallback.",
    )


class ClassifierAgent:
    """Agent 1: Analyzes the user's meta-prompt and selects the best framework skill."""

    SYSTEM_PROMPT = """You are Agent 1 (Classify & Analyze) in an enterprise Prompt Engineering Studio.
Inspect the user's raw meta-prompt and select the SINGLE best-matching skill from the catalog below by comparing its `category`, `description`, `best_for`, and `trigger_keywords`.

FRAMEWORK CATALOG:
{catalog_block}

STRICT ROUTING RULES:
1. Classify by the **DELIVERABLE ACTION** (what the user asks to produce: an email, strategy plan, agent system prompt, CustomGPT config, or visual scene)—NEVER by nouns inside the product or topic being written about.
2. `master_director` and `master_prompt` are ONLY for **Image & Video Generation** (photos, videos, artwork, visual scenes). NEVER choose them for text tasks (emails, copywriting, coding, strategy, or assistants).
3. Choose `magic_prompt` for copywriting, launch/marketing emails, blog posts, coding help, tutoring, summarization, and general single-turn text prompts.
4. Choose `costar` for complex business/technical analysis, strategy, multi-step reasoning, evaluations, and plans needing self-reflection.
5. Choose `agent_prompting` ONLY when building an autonomous tool-using AI agent, bot, or operational workflow.
6. Choose `customgpt_markdown` for persistent assistants, CustomGPTs, Gems, or Claude Projects.

DISAMBIGUATION EXAMPLES:
- "Write a launch email for our multi-agent studio" -> `magic_prompt` (deliverable is an email; "multi-agent" is only the product topic)
- "Analyze our Q4 strategy for launching an AI copilot" -> `costar` (deliverable is strategic analysis)
- "Build an autonomous research agent with web search tools" -> `agent_prompting` (deliverable is an agent prompt)
- "Create a dark cyberpunk scene inspired by Blade Runner" -> `master_prompt` (deliverable is a visual prompt)

OUTPUT CONTRACT:
Think left-to-right by filling `user_intent_summary`, `inferred_parameters`, and `routing_rationale` BEFORE choosing `selected_skill_id`. Return ONLY valid JSON (no markdown fences, no commentary):
{{
  "user_intent_summary": "<concise 1-sentence summary of the deliverable the user wants>",
  "inferred_parameters": {{
    "domain": "<primary domain>",
    "output_modality": "<e.g. Marketing Email / Copywriting, Strategic Analysis, Autonomous AI Agent, CustomGPT Config, Image / Video Prompt>"
  }},
  "routing_rationale": "<why the deliverable action matches that skill's best_for and trigger_keywords>",
  "selected_skill_id": "<exact id from catalog>",
  "confidence": 0.92
}}"""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    @staticmethod
    def _format_catalog_for_prompt(catalog_summary: list[dict[str, Any]]) -> str:
        """Render all skills with category, description, best_for, and trigger_keywords concisely."""
        blocks: list[str] = []
        for item in catalog_summary:
            keywords = ", ".join(str(kw) for kw in item.get("trigger_keywords", []))
            blocks.append(
                f"- id: {item['id']}\n"
                f"  name: {item['name']}\n"
                f"  category: {item['category']}\n"
                f"  description: {item.get('description', '')}\n"
                f"  best_for: {item.get('best_for', '')}\n"
                f"  trigger_keywords: [{keywords}]"
            )
        return "\n".join(blocks)

    def run(
        self,
        user_meta_prompt: str,
        catalog_summary: list[dict[str, Any]],
        model: str,
        temperature: float = CLASSIFY_TEMP,
    ) -> ClassificationResult:
        """Classify user intent and return a validated ClassificationResult."""
        if not catalog_summary:
            raise ValueError("Skill catalog is empty; cannot classify meta-prompt.")

        keyword_matches = _score_catalog_keywords(user_meta_prompt, catalog_summary)
        ranked_matches = sorted(
            ((sid, kws) for sid, kws in keyword_matches.items() if kws),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        matched_summary = [
            f"- {sid} ({len(kws)} hit{'s' if len(kws) != 1 else ''}): {', '.join(kws)}"
            for sid, kws in ranked_matches
        ]
        hints_block = (
            "\n".join(matched_summary)
            if matched_summary
            else "- (no direct trigger_keyword matches)"
        )

        system_prompt = self.SYSTEM_PROMPT.format(
            catalog_block=self._format_catalog_for_prompt(catalog_summary)
        )
        user_prompt = (
            f"USER META-PROMPT:\n{user_meta_prompt.strip()}\n\n"
            f"MATCHED TRIGGER KEYWORDS PER SKILL (ranked by hit count):\n{hints_block}"
        )

        raw_json = self.client.chat_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True,
        )

        raw_skill_id = str(raw_json.get("selected_skill_id") or "").strip()
        resolved_id, fallback_note = _resolve_skill_id(
            raw_skill_id, user_meta_prompt, catalog_summary
        )

        # Modality & deliverable sanity guards for small 1B-1.5B classifiers
        by_id = {entry["id"]: entry for entry in catalog_summary}
        chosen_cat = str(by_id[resolved_id].get("category", "")).lower()
        is_visual_skill = "image" in chosen_cat or "video" in chosen_cat
        if is_visual_skill and not _has_visual_intent(user_meta_prompt, catalog_summary):
            non_visual_candidates = [
                entry["id"]
                for entry in catalog_summary
                if "image" not in str(entry.get("category", "")).lower()
                and "video" not in str(entry.get("category", "")).lower()
            ]
            best_non_visual = max(
                non_visual_candidates,
                key=lambda sid: len(keyword_matches.get(sid, [])),
                default=None,
            )
            if best_non_visual and len(keyword_matches.get(best_non_visual, [])) > 0:
                prev_id = resolved_id
                resolved_id = best_non_visual
                guard_note = (
                    f"Modality guard rerouted non-visual task from '{prev_id}' to "
                    f"'{resolved_id}' (matched keywords: {', '.join(keyword_matches[resolved_id])})."
                )
                fallback_note = f"{fallback_note} {guard_note}".strip() if fallback_note else guard_note

        # Prevent agent_prompting from hijacking copywriting/analysis when prompt lacks agent-design intent
        chosen_cat = str(by_id[resolved_id].get("category", "")).lower()
        if "agent" in chosen_cat and not _has_agent_design_intent(user_meta_prompt):
            text_candidates = [
                entry["id"]
                for entry in catalog_summary
                if "image" not in str(entry.get("category", "")).lower()
                and "video" not in str(entry.get("category", "")).lower()
                and "agent" not in str(entry.get("category", "")).lower()
            ]
            best_text_skill = max(
                text_candidates,
                key=lambda sid: len(keyword_matches.get(sid, [])),
                default=None,
            )
            if (
                best_text_skill
                and len(keyword_matches.get(best_text_skill, []))
                > len(keyword_matches.get(resolved_id, []))
            ):
                prev_id = resolved_id
                resolved_id = best_text_skill
                guard_note = (
                    f"Deliverable guard rerouted non-agent task from '{prev_id}' to "
                    f"'{resolved_id}' (matched keywords: {', '.join(keyword_matches[resolved_id])})."
                )
                fallback_note = f"{fallback_note} {guard_note}".strip() if fallback_note else guard_note

        intent_summary = str(
            raw_json.get("intent_summary")
            or raw_json.get("user_intent_summary")
            or "Engineer a structured prompt matching the user's request."
        ).strip()
        rationale = str(
            raw_json.get("rationale")
            or raw_json.get("routing_rationale")
            or ""
        ).strip()
        if fallback_note:
            rationale = f"{rationale} ({fallback_note})".strip()

        raw_params = raw_json.get("inferred_parameters")
        if not isinstance(raw_params, dict):
            raw_params = {}
        inferred = InferredParameters(
            domain=str(raw_params.get("domain") or "General").strip(),
            output_modality=str(
                raw_params.get("output_modality") or "Text / System Prompt"
            ).strip(),
        )

        return ClassificationResult(
            selected_skill_id=resolved_id,
            confidence=_coerce_confidence(raw_json.get("confidence"), default=0.85),
            intent_summary=intent_summary,
            rationale=rationale or f"Selected '{resolved_id}' based on prompt intent.",
            inferred_parameters=inferred,
            is_manual_override=False,
        )


class GeneratorAgent:
    """Agent 2: Generates an engineered prompt strictly following the loaded skill file."""

    SYSTEM_PROMPT_TEMPLATE = """You are Agent 2 (Prompt Generator) in an enterprise Prompt Engineering Studio.
Transform the user's rough meta-prompt into a complete, production-ready prompt that strictly obeys the Active Framework Skill Specification below.

CRITICAL OUTPUT RULES:
1. Output ONLY the final prompt formatted in the framework.
2. Do NOT include any conversational greeting, explanation, commentary, or outer markdown code fence around the entire response.
3. Follow every required section header, ordering, vocabulary rule, and formatting guideline in the skill specification.
4. Replace every template placeholder with concrete, domain-specific instructions tailored to the user's request.
5. Output the engineered prompt ONCE ONLY—never repeat sections, append a summary `Prompt:` block, or echo a previous draft.

=== ACTIVE FRAMEWORK SKILL SPECIFICATION ({skill_name} | id: {skill_id}) ===
{skill_body}
=== END SKILL SPECIFICATION ==="""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def run(
        self,
        user_meta_prompt: str,
        classification: ClassificationResult,
        skill: Skill,
        model: str,
        temperature: float = GENERATE_TEMP,
        critique: str | None = None,
        improvement_instructions: str | None = None,
        previous_draft: str | None = None,
    ) -> str:
        """Generate (or refine) the framework-compliant prompt."""
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            skill_name=skill.name,
            skill_id=skill.id,
            skill_body=skill.body,
        )

        user_parts = [
            f"USER META-PROMPT:\n{user_meta_prompt.strip()}",
            (
                "ANALYSIS CONTEXT:\n"
                f"- Intent Summary: {classification.intent_summary}\n"
                f"- Domain: {classification.inferred_parameters.domain}\n"
                f"- Target Modality: {classification.inferred_parameters.output_modality}"
            ),
        ]

        if previous_draft and (critique or improvement_instructions):
            user_parts.append(
                "PREVIOUS DRAFT THAT FAILED VALIDATION:\n"
                f"{previous_draft.strip()}\n\n"
                "VALIDATOR CRITIQUE:\n"
                f"{(critique or '').strip()}\n\n"
                "REQUIRED FIXES:\n"
                f"{(improvement_instructions or critique or '').strip()}\n\n"
                "Rewrite the complete prompt fixing every issue above while keeping ONLY the final prompt text."
            )
        else:
            user_parts.append(
                "Generate the complete, framework-compliant prompt now. Output ONLY the engineered prompt."
            )

        raw_output = self.client.chat_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt="\n\n".join(user_parts),
            temperature=temperature,
            json_mode=False,
        )
        return _clean_generated_output(raw_output)


class ValidatorAgent:
    """Agent 3: Audits structural compliance, intent fidelity, and output cleanliness."""

    SYSTEM_PROMPT_TEMPLATE = """You are Agent 3 (Prompt Quality Validator) in an enterprise Prompt Engineering Studio.
Audit the generated prompt against BOTH the user's original meta-prompt and the Active Framework Skill Specification.

EVALUATION CRITERIA:
1. Structural Compliance: Are all required sections, sequence rules, and formatting conventions of the framework present and followed?
2. Intent Alignment: Does the generated prompt accurately capture every goal and constraint of the user's meta-prompt without hallucinating conflicting goals?
3. Clean Output: Is the prompt free of conversational preamble ("Here is your prompt:") or explanatory postscript?
4. Actionability & Depth: Are placeholders replaced with concrete, domain-specific details?

=== ACTIVE FRAMEWORK SKILL SPECIFICATION ({skill_name} | id: {skill_id}) ===
{skill_body}
=== END SKILL SPECIFICATION ===

OUTPUT CONTRACT:
Return ONLY valid JSON matching this exact schema:
{{
  "is_valid": true,
  "score": 9,
  "framework_compliance": "PASS",
  "intent_alignment": "PASS",
  "compliance_notes": "<concise audit of section compliance and clean formatting>",
  "critique": "<specific strengths or remaining defects>",
  "improvement_instructions": "<actionable fixes required if failing, or empty string if passing>"
}}"""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def run(
        self,
        user_meta_prompt: str,
        classification: ClassificationResult,
        skill: Skill,
        generated_prompt: str,
        model: str,
        temperature: float = VALIDATE_TEMP,
    ) -> ValidationResult:
        """Audit the generated prompt and return a ValidationResult."""
        if not generated_prompt or len(generated_prompt.strip()) < 20:
            return ValidationResult(
                is_valid=False,
                score=1,
                framework_compliance="FAIL",
                intent_alignment="FAIL",
                compliance_notes="Generated output was empty or too short.",
                critique="Generator produced insufficient output.",
                improvement_instructions="Generate the complete prompt following every section of the framework.",
            )

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            skill_name=skill.name,
            skill_id=skill.id,
            skill_body=skill.body,
        )

        user_prompt = (
            f"ORIGINAL USER META-PROMPT:\n{user_meta_prompt.strip()}\n\n"
            f"ROUTING CONTEXT:\n"
            f"- Framework: {skill.name} ({skill.id})\n"
            f"- Intent Summary: {classification.intent_summary}\n"
            f"- Domain: {classification.inferred_parameters.domain}\n\n"
            f"GENERATED PROMPT TO AUDIT:\n{generated_prompt.strip()}"
        )

        raw_json = self.client.chat_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True,
        )

        score = _coerce_score(raw_json.get("score"), default=6)
        raw_valid = _coerce_bool(raw_json.get("is_valid"), default=(score >= MIN_PASS_SCORE))
        framework_compliance = _coerce_pass_fail(
            raw_json.get("framework_compliance"), fallback_bool=raw_valid
        )
        intent_alignment = _coerce_pass_fail(
            raw_json.get("intent_alignment"), fallback_bool=raw_valid
        )

        # Enforce quality threshold deterministically
        effective_valid = (
            raw_valid
            and score >= MIN_PASS_SCORE
            and framework_compliance == "PASS"
            and intent_alignment == "PASS"
        )

        critique = str(raw_json.get("critique") or "").strip()
        improvement_instructions = str(raw_json.get("improvement_instructions") or "").strip()
        compliance_notes = str(
            raw_json.get("compliance_notes")
            or f"Framework: {framework_compliance} | Intent: {intent_alignment} | Score: {score}/10"
        ).strip()

        if not effective_valid and not improvement_instructions:
            improvement_instructions = (
                critique
                or f"Ensure full compliance with {skill.name} structure and address all aspects of the user meta-prompt."
            )

        return ValidationResult(
            is_valid=effective_valid,
            score=score,
            framework_compliance=framework_compliance,
            intent_alignment=intent_alignment,
            compliance_notes=compliance_notes,
            critique=critique or compliance_notes,
            improvement_instructions=improvement_instructions,
        )


class PromptEngineeringOrchestrator:
    """Coordinates ClassifierAgent, SkillRegistry, GeneratorAgent, and ValidatorAgent."""

    def __init__(
        self,
        skill_registry: SkillRegistry,
        llm_client: OpenAIClient,
    ) -> None:
        self.skill_registry = skill_registry
        self.llm_client = llm_client
        self.classifier = ClassifierAgent(llm_client)
        self.generator = GeneratorAgent(llm_client)
        self.validator = ValidatorAgent(llm_client)

    def run_pipeline(
        self,
        user_meta_prompt: str,
        model: str,
        selected_option: str = AUTO_MODE_ID,
        max_retries: int = MAX_RETRIES,
        generate_temperature: float = GENERATE_TEMP,
        on_stage_update: ProgressCallback | None = None,
    ) -> PipelineTrace:
        """Execute Classify/Load -> Generate -> Validate (+ self-correction retries)."""
        cleaned_input = user_meta_prompt.strip()
        if not cleaned_input:
            raise ValueError("User meta-prompt cannot be empty.")

        pipeline_start = time.perf_counter()
        self.skill_registry.refresh()
        catalog = self.skill_registry.get_catalog_summary()
        if not catalog:
            raise OpenAIClientError(
                f"No valid skill files found in '{self.skill_registry.skills_dir}'."
            )

        is_auto = selected_option.strip().lower() == AUTO_MODE_ID.lower()
        selection_mode: Literal["Auto", "Manual"] = "Auto" if is_auto else "Manual"

        # Stage 1: Classification or Manual Override
        if is_auto:
            self._emit(on_stage_update, "classify_start", {"model": model})
            classification = self.classifier.run(
                user_meta_prompt=cleaned_input,
                catalog_summary=catalog,
                model=model,
            )
            self._emit(
                on_stage_update,
                "classify_done",
                {"classification": classification.model_dump()},
            )
        else:
            meta = self.skill_registry.get_skill_metadata(selected_option)
            classification = ClassificationResult(
                selected_skill_id=meta.id,
                confidence=1.0,
                intent_summary=f"Manual framework selection: {meta.name}",
                rationale="User manually locked this prompting framework; ClassifierAgent skipped.",
                inferred_parameters=InferredParameters(
                    domain=meta.category,
                    output_modality=meta.category,
                ),
                is_manual_override=True,
            )
            self._emit(
                on_stage_update,
                "classify_skipped",
                {"classification": classification.model_dump()},
            )

        # Stage 2 prep: Lazy-load ONLY the chosen skill's full Markdown instructions
        skill = self.skill_registry.get_skill_by_id(classification.selected_skill_id)
        self._emit(
            on_stage_update,
            "skill_loaded",
            {"skill_id": skill.id, "skill_name": skill.name, "filepath": str(skill.filepath)},
        )

        # Stage 2 & 3: Generate + Validate loop (max 1 + max_retries attempts)
        attempts: list[PipelineAttempt] = []
        critique_arg: str | None = None
        instructions_arg: str | None = None
        previous_draft_arg: str | None = None

        total_allowed = max(0, int(max_retries)) + 1
        for attempt_idx in range(1, total_allowed + 1):
            attempt_start = time.perf_counter()

            self._emit(
                on_stage_update,
                "generate_start",
                {
                    "attempt": attempt_idx,
                    "max_attempts": total_allowed,
                    "is_retry": attempt_idx > 1,
                    "skill_id": skill.id,
                },
            )
            generated_prompt = self.generator.run(
                user_meta_prompt=cleaned_input,
                classification=classification,
                skill=skill,
                model=model,
                temperature=generate_temperature,
                critique=critique_arg,
                improvement_instructions=instructions_arg,
                previous_draft=previous_draft_arg,
            )
            self._emit(
                on_stage_update,
                "generate_done",
                {"attempt": attempt_idx, "chars": len(generated_prompt)},
            )

            self._emit(
                on_stage_update,
                "validate_start",
                {"attempt": attempt_idx, "max_attempts": total_allowed},
            )
            validation = self.validator.run(
                user_meta_prompt=cleaned_input,
                classification=classification,
                skill=skill,
                generated_prompt=generated_prompt,
                model=model,
            )
            attempt_duration = round(time.perf_counter() - attempt_start, 2)

            attempt_record = PipelineAttempt(
                attempt_number=attempt_idx,
                generated_prompt=generated_prompt,
                validation=validation,
                duration_seconds=attempt_duration,
                critique_used=instructions_arg or critique_arg or "",
            )
            attempts.append(attempt_record)

            self._emit(
                on_stage_update,
                "validate_done",
                {
                    "attempt": attempt_idx,
                    "validation": validation.model_dump(),
                    "duration_seconds": attempt_duration,
                },
            )

            if validation.is_valid and validation.score >= MIN_PASS_SCORE:
                break

            previous_draft_arg = generated_prompt
            critique_arg = validation.critique
            instructions_arg = validation.improvement_instructions

        # Select passing attempt, or highest-scoring attempt if retries exhausted
        passing_attempts = [
            att
            for att in attempts
            if att.validation.is_valid and att.validation.score >= MIN_PASS_SCORE
        ]
        if passing_attempts:
            chosen_attempt = passing_attempts[-1]
            passed = True
        else:
            chosen_attempt = max(attempts, key=lambda att: att.validation.score)
            passed = False

        total_duration = round(time.perf_counter() - pipeline_start, 2)
        return PipelineTrace(
            user_meta_prompt=cleaned_input,
            model=model,
            selection_mode=selection_mode,
            classification=classification,
            skill_id=skill.id,
            skill_name=skill.name,
            skill_filepath=str(skill.filepath),
            attempts=attempts,
            final_prompt=chosen_attempt.generated_prompt,
            final_validation=chosen_attempt.validation,
            passed=passed,
            retries_used=max(0, len(attempts) - 1),
            total_duration_seconds=total_duration,
        )

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if callback is not None:
            callback(stage, payload)
