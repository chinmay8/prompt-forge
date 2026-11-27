"""Application constants, OpenAI-compatible API defaults, and UI presets."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SKILLS_DIR: Path = PROJECT_ROOT / "skills"

# Load .env before reading environment-backed API defaults
load_dotenv(PROJECT_ROOT / ".env")

# OpenAI-compatible Chat Completions endpoint & credentials (configured via .env)
DEFAULT_API_BASE_URL: str = (
    os.environ.get("API_BASE_URL") or "http://localhost:11434/v1"
).strip()

DEFAULT_API_KEY: str = (os.environ.get("API_KEY") or "").strip()

DEFAULT_MODEL: str = (os.environ.get("DEFAULT_MODEL") or "qwen2.5:1.5b").strip()

# Curated fallback/priority models optimized for low-memory local or hosted runs
RECOMMENDED_MODELS: list[str] = [
    DEFAULT_MODEL,
    "qwen2.5:1.5b",
    "gemma3:1b",
]
# Deduplicate preserving order
RECOMMENDED_MODELS = list(dict.fromkeys(m for m in RECOMMENDED_MODELS if m))

# Framework routing constants
AUTO_MODE_ID: str = "Auto"
AUTO_MODE_LABEL: str = "Auto (AI Classifies & Selects Framework)"

# Agent sampling, memory-guard, & validation defaults
CLASSIFY_TEMP: float = 0.1
GENERATE_TEMP: float = 0.45
VALIDATE_TEMP: float = 0.1
DEFAULT_MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS") or 2048)
MAX_LOCAL_NUM_CTX: int = 2048
LOCAL_NUM_CTX: int = min(
    max(1, int(os.environ.get("LOCAL_NUM_CTX") or MAX_LOCAL_NUM_CTX)),
    MAX_LOCAL_NUM_CTX,
)
MAX_RETRIES: int = 2
MIN_PASS_SCORE: int = 7
DEFAULT_REQUEST_TIMEOUT: float = 120.0


def reload_env_settings() -> tuple[str, str, str]:
    """Reload .env with override=True and return (api_base_url, api_key, default_model)."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_base_url = (
        os.environ.get("API_BASE_URL") or "http://localhost:11434/v1"
    ).strip()
    api_key = (os.environ.get("API_KEY") or "").strip()
    default_model = (os.environ.get("DEFAULT_MODEL") or "qwen2.5:1.5b").strip()
    return api_base_url, api_key, default_model

# Quick-start example meta-prompts displayed in Tab 1
QUICK_START_EXAMPLES: list[dict[str, str]] = [
    {
        "label": "Research Agent for Market Intelligence",
        "prompt": (
            "Build an autonomous research agent that monitors fintech and AI market "
            "intelligence daily, cross-references SEC filings and reputable news, flags "
            "conflicting claims, and produces structured executive briefings."
        ),
    },
    {
        "label": "Cinematic Sci-Fi Street Scene",
        "prompt": (
            "Create a dark, suspenseful sci-fi visual prompt of a customized Mumbai "
            "auto-rickshaw speeding through a rain-slicked neon cyberpunk street at night, "
            "inspired by Blade Runner 2049 and Dune."
        ),
    },
    {
        "label": "Enterprise CustomGPT Support Bot",
        "prompt": (
            "Configure a persistent CustomGPT customer support specialist for an enterprise "
            "B2B SaaS platform that triages billing and API integration issues, asks clarifying "
            "questions, and escalates critical outages safely."
        ),
    },
    {
        "label": "Executive Product Launch Strategy",
        "prompt": (
            "Analyze our Q4 enterprise AI copilot launch strategy across target buyer "
            "segments, pricing tiers, developer adoption risks, and a step-by-step go-to-market plan."
        ),
    },
    {
        "label": "High-Converting Launch Email",
        "prompt": (
            "Write a persuasive product launch email announcing our multi-agent prompt "
            "engineering studio to engineering managers and AI product leads."
        ),
    },
]
