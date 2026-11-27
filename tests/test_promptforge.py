"""Comprehensive unit and integration tests for PromptForge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import BadRequestError, NotFoundError
from streamlit.testing.v1 import AppTest

from core.agents import (
    ClassifierAgent,
    PromptEngineeringOrchestrator,
    _clean_generated_output,
)
from core.config import (
    AUTO_MODE_ID,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    LOCAL_NUM_CTX,
    RECOMMENDED_MODELS,
    SKILLS_DIR,
)
from core.openai_client import OpenAIClient, OpenAIClientError, extract_json_payload
from core.skill_loader import SkillNotFoundError, SkillRegistry, SkillValidationError


# ---------------------------------------------------------------------------
# SkillRegistry tests
# ---------------------------------------------------------------------------


def test_default_skills_load_and_lazy_body() -> None:
    """All 6 shipped skills load metadata cleanly and lazy-load full Markdown bodies."""
    registry = SkillRegistry(SKILLS_DIR)
    assert registry.load_errors == []

    catalog = registry.get_catalog_summary()
    ids = {item["id"] for item in catalog}
    assert ids == {
        "master_director",
        "master_prompt",
        "agent_prompting",
        "costar",
        "magic_prompt",
        "customgpt_markdown",
    }

    # Metadata lookup keeps body empty until lazy-loaded
    meta_only = registry.get_skill_metadata("costar")
    assert meta_only.body == ""
    assert meta_only.name == "CoSTAR Framework"

    # Full load reads Markdown body
    full_skill = registry.get_skill_by_id("costar")
    assert "### Context" in full_skill.body
    assert "### Reflection" in full_skill.body


def test_malformed_skill_does_not_crash_registry(tmp_path: Path) -> None:
    """Malformed YAML frontmatter is recorded in load_errors without breaking valid skills."""
    (tmp_path / "good.md").write_text(
        "---\nid: good_skill\nname: Good Skill\ncategory: General\n"
        "description: Valid test skill.\n---\n\n# Body\nDo good things.\n",
        encoding="utf-8",
    )
    (tmp_path / "bad_no_fence.md").write_text(
        "# Missing frontmatter\nJust text.",
        encoding="utf-8",
    )
    (tmp_path / "bad_yaml.md").write_text(
        "---\nid: [unclosed\n---\n\n# Body\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)
    assert registry.list_skill_ids() == ["good_skill"]
    assert len(registry.load_errors) == 2


def test_save_new_skill_and_hot_reload(tmp_path: Path) -> None:
    """Creating a new skill writes valid YAML+Markdown and refreshes registry immediately."""
    registry = SkillRegistry(tmp_path)
    created = registry.save_new_skill(
        skill_id="chain_of_verification",
        name="Chain of Verification",
        category="General & System Prompts",
        description="Draft, generate verification questions, verify, and revise.",
        best_for="High-accuracy fact checking",
        trigger_keywords="verify, audit, fact check",
        body="# Skill: CoVe\n\nVerify claims step by step.",
    )
    assert created.id == "chain_of_verification"
    assert (tmp_path / "chain_of_verification.md").exists()
    assert "chain_of_verification" in registry.list_skill_ids()
    assert created.trigger_keywords == ["verify", "audit", "fact check"]

    with pytest.raises(SkillValidationError):
        registry.save_new_skill(
            skill_id="chain_of_verification",
            name="Dup",
            category="General",
            description="Dup",
            best_for="",
            trigger_keywords=[],
            body="Dup body",
            overwrite=False,
        )

    with pytest.raises(SkillNotFoundError):
        registry.get_skill_by_id("nonexistent")


# ---------------------------------------------------------------------------
# OpenAIClient tests
# ---------------------------------------------------------------------------


def test_extract_json_payload_variants() -> None:
    """extract_json_payload handles direct JSON, think blocks, fences, and extra prose."""
    assert extract_json_payload('{"a": 1}') == {"a": 1}
    assert extract_json_payload('<think>reasoning</think>\n```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json_payload('Sure! Here is JSON:\n{"score": 8}\nHope that helps!') == {"score": 8}

    with pytest.raises(OpenAIClientError):
        extract_json_payload("Not JSON at all")


def test_openai_client_env_config_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAIClient reads API_BASE_URL and API_KEY from env and flags missing remote keys."""
    monkeypatch.setenv("API_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("API_KEY", "")

    missing_key_client = OpenAIClient()
    assert missing_key_client.base_url == "https://ollama.com/v1"
    assert missing_key_client.has_api_key is False
    assert missing_key_client.check_connection() is False
    assert "API_KEY" in (missing_key_client.last_error or "")

    with pytest.raises(OpenAIClientError):
        missing_key_client.chat_completion(
            model="qwen2.5:1.5b",
            system_prompt="sys",
            user_prompt="usr",
        )

    # Configured key via env works
    monkeypatch.setenv("API_KEY", "test-key-123")
    configured = OpenAIClient()
    assert configured.has_api_key is True
    assert configured.api_key == "test-key-123"


def test_openai_client_models_chat_and_json_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAIClient lists models, invokes chat.completions.create, and falls back gracefully."""
    monkeypatch.setenv("API_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("API_KEY", "test-key-123")

    client = OpenAIClient()
    mock_sdk = MagicMock()
    mock_sdk.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="qwen2.5:1.5b"),
            SimpleNamespace(id="gemma3:1b"),
        ]
    )
    client._client = mock_sdk

    assert client.check_connection() is True
    assert client.is_local_endpoint is False
    assert client.get_available_models() == ["qwen2.5:1.5b", "gemma3:1b"]

    # JSON mode uses response_format={"type": "json_object"} and max_tokens
    mock_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"selected_skill_id": "costar"}')
            )
        ]
    )
    res = client.chat_completion(
        model="qwen2.5:1.5b",
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.1,
        json_mode=True,
    )
    assert res == {"selected_skill_id": "costar"}
    call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "qwen2.5:1.5b"
    assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert "extra_body" not in call_kwargs

    # Text mode strips <think> blocks cleanly
    mock_sdk.chat.completions.create.reset_mock()
    mock_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<think>internal reasoning</think>Cleaned prompt")
            )
        ]
    )
    text_res = client.chat_completion(
        model="qwen2.5:1.5b",
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.45,
        json_mode=False,
    )
    assert text_res == "Cleaned prompt"

    # BadRequestError rejecting response_format falls back cleanly without response_format
    req = httpx.Request("POST", "https://ollama.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    mock_sdk.chat.completions.create.reset_mock()
    mock_sdk.chat.completions.create.side_effect = [
        BadRequestError("response_format not supported", response=resp, body=None),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        ),
    ]
    json_fallback = client.chat_completion(
        model="qwen2.5:1.5b",
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.1,
        json_mode=True,
    )
    assert json_fallback == {"ok": True}
    assert mock_sdk.chat.completions.create.call_count == 2
    assert "response_format" not in mock_sdk.chat.completions.create.call_args_list[1].kwargs

    # NotFoundError on remote /v1/models falls back to [DEFAULT_MODEL]
    req_models = httpx.Request("GET", "https://ollama.com/v1/models")
    resp_404 = httpx.Response(404, request=req_models)
    mock_sdk.models.list.side_effect = NotFoundError("not found", response=resp_404, body=None)
    assert client.check_connection() is True
    assert client.get_available_models() == [client.default_model]


def test_openai_client_local_num_ctx_and_json_repair_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local endpoint caps num_ctx at 2048, drops extra_body if rejected, and repairs bad JSON once."""
    monkeypatch.setenv("API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("API_KEY", "ollama")
    monkeypatch.setenv("LOCAL_NUM_CTX", "8192")

    assert LOCAL_NUM_CTX == 2048
    client = OpenAIClient()
    assert client.is_local_endpoint is True

    mock_sdk = MagicMock()
    req = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    mock_sdk.chat.completions.create.side_effect = [
        # 1st attempt: server rejects extra_body -> dropped automatically
        BadRequestError("unrecognized extra_body options", response=resp, body=None),
        # Retry without extra_body returns malformed JSON from 1B model
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not valid json"))]
        ),
        # 1x automatic JSON repair retry succeeds
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"is_valid": true, "score": 8}'))]
        ),
    ]
    client._client = mock_sdk

    parsed = client.chat_completion(
        model="gemma3:1b",
        system_prompt="sys",
        user_prompt="usr",
        temperature=0.1,
        json_mode=True,
    )
    assert parsed == {"is_valid": True, "score": 8}
    assert mock_sdk.chat.completions.create.call_count == 3
    first_kwargs = mock_sdk.chat.completions.create.call_args_list[0].kwargs
    assert first_kwargs["extra_body"] == {"options": {"num_ctx": 2048}}
    assert "extra_body" not in mock_sdk.chat.completions.create.call_args_list[1].kwargs


# ---------------------------------------------------------------------------
# Multi-Agent & Orchestrator tests
# ---------------------------------------------------------------------------


def _make_fake_client(responses: list[dict | str]) -> OpenAIClient:
    """Create an OpenAIClient whose chat_completion yields scripted responses."""
    client = OpenAIClient(base_url="https://ollama.com/v1", api_key="test-key")
    iterator = iter(responses)

    def fake_chat(**kwargs):
        return next(iterator)

    client.chat_completion = MagicMock(side_effect=fake_chat)
    return client


def test_orchestrator_auto_mode_with_self_correction_loop() -> None:
    """Auto mode classifies, generates, fails score<7, retries with critique, and passes."""
    registry = SkillRegistry(SKILLS_DIR)
    fake_client = _make_fake_client(
        [
            {
                "selected_skill_id": "costar",
                "confidence": 0.94,
                "user_intent_summary": "Develop a Q4 SaaS go-to-market strategy",
                "inferred_parameters": {
                    "domain": "Enterprise SaaS",
                    "output_modality": "Strategic Plan",
                },
                "routing_rationale": "Requires multi-step reasoning and reflection.",
            },
            "Here is your prompt:\n### Context\nSaaS launch\n### Objective\nLaunch Q4",
            {
                "is_valid": True,
                "score": 5,
                "framework_compliance": "FAIL",
                "intent_alignment": "PASS",
                "compliance_notes": "Missing Steps, Tools, Actions, and Reflection sections.",
                "critique": "Incomplete CoSTAR structure.",
                "improvement_instructions": "Add all 6 CoSTAR sections including Reflection.",
            },
            (
                "```markdown\n"
                "### Context\nLaunching an AI tool in Q4.\n\n"
                "### Objective\nCreate a 90-day GTM strategy.\n\n"
                "### Steps\n1. Segment ICP.\n2. Define pricing.\n\n"
                "### Tools\nMarket sizing benchmarks.\n\n"
                "### Actions\nExecutive Markdown report.\n\n"
                "### Reflection\nVerify assumptions against budget constraints.\n"
                "```"
            ),
            {
                "is_valid": True,
                "score": 9,
                "framework_compliance": "PASS",
                "intent_alignment": "PASS",
                "compliance_notes": "All 6 CoSTAR sections complete.",
                "critique": "Strong alignment and clean structure.",
                "improvement_instructions": "",
            },
        ]
    )

    orchestrator = PromptEngineeringOrchestrator(registry, fake_client)
    events: list[str] = []
    trace = orchestrator.run_pipeline(
        user_meta_prompt="Plan our Q4 SaaS launch strategy",
        model="gpt-oss:120b",
        selected_option=AUTO_MODE_ID,
        max_retries=2,
        on_stage_update=lambda stage, _payload: events.append(stage),
    )

    assert trace.selection_mode == "Auto"
    assert trace.skill_id == "costar"
    assert trace.passed is True
    assert trace.retries_used == 1
    assert len(trace.attempts) == 2
    assert trace.attempts[0].validation.is_valid is False
    assert trace.final_validation.score == 9
    assert trace.final_prompt.startswith("### Context")
    assert "```" not in trace.final_prompt
    assert "classify_done" in events
    assert events.count("validate_done") == 2


def test_orchestrator_manual_override_skips_classifier() -> None:
    """Selecting a specific skill skips ClassifierAgent and loads that skill directly."""
    registry = SkillRegistry(SKILLS_DIR)
    fake_client = _make_fake_client(
        [
            "Intense, dark suspense visual inspired by Blade Runner 2049, featuring a neon Mumbai auto-rickshaw...",
            {
                "is_valid": True,
                "score": 9,
                "framework_compliance": "PASS",
                "intent_alignment": "PASS",
                "compliance_notes": "Follows Master Prompt narrative flow.",
                "critique": "Evocative and clean.",
                "improvement_instructions": "",
            },
        ]
    )

    orchestrator = PromptEngineeringOrchestrator(registry, fake_client)
    trace = orchestrator.run_pipeline(
        user_meta_prompt="Mumbai auto-rickshaw in cyberpunk alley",
        model="gpt-oss:120b",
        selected_option="master_prompt",
        max_retries=2,
    )

    assert trace.selection_mode == "Manual"
    assert trace.classification.is_manual_override is True
    assert trace.skill_id == "master_prompt"
    assert fake_client.chat_completion.call_count == 2
    assert trace.passed is True


def test_classifier_unknown_skill_falls_back_to_catalog() -> None:
    """If a model returns an unknown skill ID, ClassifierAgent recovers cleanly."""
    registry = SkillRegistry(SKILLS_DIR)
    fake_client = _make_fake_client(
        [
            {
                "selected_skill_id": "hallucinated_framework_xyz",
                "confidence": 88,
                "user_intent_summary": "Build an autonomous research agent with search tools",
                "routing_rationale": "Agent workflow needed.",
            }
        ]
    )
    classifier = ClassifierAgent(fake_client)
    result = classifier.run(
        user_meta_prompt="Build an autonomous research agent with web search tools and guardrails",
        catalog_summary=registry.get_catalog_summary(),
        model="gpt-oss:120b",
    )

    assert result.selected_skill_id == "agent_prompting"
    assert result.confidence == 0.88


def test_classifier_passes_all_skill_context_and_guards_text_vs_visual() -> None:
    """Classifier passes best_for/trigger_keywords for every skill and prevents visual misrouting."""
    registry = SkillRegistry(SKILLS_DIR)
    catalog = registry.get_catalog_summary()

    # Even if a small model misroutes a launch email prompt to master_prompt, modality guard fixes it
    fake_client = _make_fake_client(
        [
            {
                "selected_skill_id": "master_prompt",
                "confidence": 0.78,
                "user_intent_summary": "Write a persuasive product launch email",
                "routing_rationale": "Misrouted to master_prompt.",
            }
        ]
    )
    classifier = ClassifierAgent(fake_client)
    email_prompt = (
        "Write a persuasive product launch email announcing our multi-agent "
        "prompt engineering studio to engineering managers and AI product leads."
    )
    res = classifier.run(
        user_meta_prompt=email_prompt,
        catalog_summary=catalog,
        model="qwen2.5:1.5b",
    )

    assert res.selected_skill_id == "magic_prompt"
    sent_kwargs = fake_client.chat_completion.call_args.kwargs
    # Verify all skills' best_for, description, and trigger_keywords were passed in system_prompt
    for entry in catalog:
        assert entry["best_for"] in sent_kwargs["system_prompt"]
        for kw in entry["trigger_keywords"]:
            assert kw in sent_kwargs["system_prompt"]
    assert "magic_prompt (2 hits): write, email" in sent_kwargs["user_prompt"]
    assert "agent_prompting" not in sent_kwargs["user_prompt"].split("MATCHED TRIGGER KEYWORDS")[1]

    # If a small model misroutes the launch email to agent_prompting due to 'multi-agent' in topic
    agent_misroute_client = _make_fake_client(
        [
            {
                "user_intent_summary": "Write a product launch email",
                "routing_rationale": "Mentions multi-agent",
                "selected_skill_id": "agent_prompting",
                "confidence": 0.72,
            }
        ]
    )
    rerouted_res = ClassifierAgent(agent_misroute_client).run(
        user_meta_prompt=email_prompt,
        catalog_summary=catalog,
        model="qwen2.5:1.5b",
    )
    assert rerouted_res.selected_skill_id == "magic_prompt"

    # Real visual prompt keeps master_prompt intact
    visual_client = _make_fake_client(
        [
            {
                "selected_skill_id": "master_prompt",
                "confidence": 0.93,
                "user_intent_summary": "Cinematic sci-fi scene",
                "routing_rationale": "Mood and movie reference visual prompt.",
            }
        ]
    )
    visual_res = ClassifierAgent(visual_client).run(
        user_meta_prompt="Create a dark cyberpunk scene inspired by Blade Runner 2049",
        catalog_summary=catalog,
        model="qwen2.5:1.5b",
    )
    assert visual_res.selected_skill_id == "master_prompt"


def test_clean_generated_output_deduplicates_repeated_prompts() -> None:
    """_clean_generated_output removes repeated halves, duplicate headers, and trailing Prompt echoes."""
    # 1. Trailing --- **Prompt:** echo
    with_echo = (
        "ROLE & RESPONSIBILITY:\nFintech analyst.\n\n"
        "OBJECTIVE:\nDaily briefing.\n\n"
        "---\n\n**Prompt:**\nCreate a daily briefing."
    )
    assert _clean_generated_output(with_echo) == (
        "ROLE & RESPONSIBILITY:\nFintech analyst.\n\nOBJECTIVE:\nDaily briefing."
    )

    # 2. Repeated root framework header
    repeated_costar = (
        "### Context\nFirst context.\n### Objective\nFirst objective.\n\n"
        "---\n\n"
        "### Context\nSecond context.\n### Objective\nSecond objective."
    )
    assert _clean_generated_output(repeated_costar) == (
        "### Context\nFirst context.\n### Objective\nFirst objective."
    )

    # 3. Exact repeated single-paragraph prompts (e.g. master_director)
    single_para = "A neon Mumbai auto-rickshaw, wide shot, 35mm lens, Cinestill 800T."
    assert _clean_generated_output(f"{single_para}\n\n---\n\n{single_para}") == single_para
    assert _clean_generated_output(f"{single_para}\n{single_para}") == single_para


# ---------------------------------------------------------------------------
# Streamlit AppTest smoke & interactive tests
# ---------------------------------------------------------------------------

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_streamlit_app_renders_missing_key_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit app boots cleanly when API key is missing and guides user to .env."""
    monkeypatch.setenv("API_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("API_KEY", "")

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()

    assert not at.exception
    assert any("PromptForge Studio" in title.value for title in at.title)
    framework_selectboxes = [
        sb for sb in at.sidebar.selectbox if sb.label == "Framework Mode"
    ]
    assert len(framework_selectboxes) == 1
    assert len(framework_selectboxes[0].options) == 7

    sidebar_errors = [err.value for err in at.sidebar.error]
    assert any("API_KEY" in msg for msg in sidebar_errors)
    assert any(".env" in i.value for i in at.info)


def test_streamlit_end_to_end_pipeline_and_regenerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full Studio flow: default qwen2.5:1.5b -> chip -> Engineer & Validate -> Regenerate."""
    scripted_responses = iter(
        [
            {
                "selected_skill_id": "agent_prompting",
                "confidence": 0.95,
                "user_intent_summary": "Build a fintech news research agent",
                "inferred_parameters": {
                    "domain": "Fintech",
                    "output_modality": "Agent System Prompt",
                },
                "routing_rationale": "Tool-using research agent.",
            },
            "ROLE & RESPONSIBILITY:\nSenior Fintech Research Agent.\n\nOBJECTIVE:\nSynthesize daily filings.",
            {
                "is_valid": True,
                "score": 9,
                "framework_compliance": "PASS",
                "intent_alignment": "PASS",
                "compliance_notes": "All sections complete.",
                "critique": "High quality.",
                "improvement_instructions": "",
            },
            {
                "selected_skill_id": "agent_prompting",
                "confidence": 0.96,
                "user_intent_summary": "Build a fintech news research agent",
                "inferred_parameters": {
                    "domain": "Fintech",
                    "output_modality": "Agent System Prompt",
                },
                "routing_rationale": "Tool-using research agent.",
            },
            "ROLE & RESPONSIBILITY:\nPrincipal Fintech Intelligence Agent.\n\nOBJECTIVE:\nDaily verified synthesis.",
            {
                "is_valid": True,
                "score": 10,
                "framework_compliance": "PASS",
                "intent_alignment": "PASS",
                "compliance_notes": "Exemplary.",
                "critique": "Complete.",
                "improvement_instructions": "",
            },
        ]
    )

    monkeypatch.setenv("API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("API_KEY", "ollama")
    monkeypatch.setenv("DEFAULT_MODEL", "qwen2.5:1.5b")
    monkeypatch.setattr(OpenAIClient, "check_connection", lambda self: True)
    monkeypatch.setattr(
        OpenAIClient,
        "get_available_models",
        lambda self: ["gemma3:1b", "qwen2.5:1.5b"],
    )
    monkeypatch.setattr(
        OpenAIClient,
        "chat_completion",
        lambda self, **kwargs: next(scripted_responses),
    )

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception

    # Default model should prioritize qwen2.5:1.5b over gemma3:1b
    model_sb = [sb for sb in at.sidebar.selectbox if sb.label == "Active Model"][0]
    assert model_sb.value == "qwen2.5:1.5b"

    # Click quick-start chip to populate meta-prompt text area
    chip_btn = [b for b in at.button if b.label == "Research Agent for Market Intelligence"][0]
    chip_btn.click().run()
    assert not at.exception
    assert "fintech" in at.text_area[0].value.lower()

    # Click primary CTA
    run_btn = [b for b in at.button if b.label == "Engineer & Validate Prompt"][0]
    run_btn.click().run()
    assert not at.exception
    assert any("Senior Fintech Research Agent" in c.value for c in at.code)

    # Click Regenerate button
    regen_btn = [b for b in at.button if b.label == "Regenerate"][0]
    regen_btn.click().run()
    assert not at.exception
    assert any("Principal Fintech Intelligence Agent" in c.value for c in at.code)


def test_streamlit_local_endpoint_does_not_inject_unpulled_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When local Ollama lacks DEFAULT_MODEL, UI selects an actually installed model."""
    monkeypatch.setenv("API_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("API_KEY", "ollama")
    monkeypatch.setenv("DEFAULT_MODEL", "qwen2.5:1.5b")
    monkeypatch.setattr(OpenAIClient, "check_connection", lambda self: True)
    monkeypatch.setattr(
        OpenAIClient,
        "get_available_models",
        lambda self: ["gemma3:1b"],
    )

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    assert "qwen2.5:1.5b" in RECOMMENDED_MODELS
    assert "gemma3:1b" in RECOMMENDED_MODELS
    model_sb = [sb for sb in at.sidebar.selectbox if sb.label == "Active Model"][0]
    assert model_sb.options == ["gemma3:1b"]
    assert model_sb.value == "gemma3:1b"


def test_streamlit_skill_builder_creates_and_hot_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tab 2 Skill Builder creates a new .md skill and hot-reloads the sidebar selector."""
    import core.skill_loader as skill_loader_mod

    (tmp_path / "costar.md").write_text(
        (SKILLS_DIR / "costar.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_loader_mod, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(OpenAIClient, "check_connection", lambda self: False)
    monkeypatch.setattr(OpenAIClient, "get_available_models", lambda self: [])

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    tab2 = at.tabs[1]
    by_label_ti = {ti.label: ti for ti in tab2.text_input}
    by_label_ta = {ta.label: ta for ta in tab2.text_area}

    by_label_ti["Skill ID (filename slug)"].input("cove_test")
    by_label_ti["Display Name"].input("CoVe Verification")
    by_label_ta["Short Description"].input("Fact verification framework.")
    by_label_ti["Best For"].input("High-precision fact checking")
    by_label_ti["Trigger Keywords (comma-separated)"].input("verify, audit")

    submit_btn = [
        b for b in tab2.button if "Save Skill to skills/*.md" in b.label
    ][0]
    submit_btn.click().run()

    assert not at.exception
    assert (tmp_path / "cove_test.md").exists()
    framework_sb = [sb for sb in at.sidebar.selectbox if sb.label == "Framework Mode"][0]
    assert any("CoVe Verification" in opt for opt in framework_sb.options)
