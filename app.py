"""PromptForge — Enterprise Multi-Agent Prompt Engineering Studio."""

from __future__ import annotations

import difflib
import os
from typing import Any

import streamlit as st

from core.agents import PipelineTrace, PromptEngineeringOrchestrator
from core.config import (
    AUTO_MODE_ID,
    AUTO_MODE_LABEL,
    DEFAULT_API_BASE_URL,
    DEFAULT_MODEL,
    GENERATE_TEMP,
    MAX_RETRIES,
    MIN_PASS_SCORE,
    QUICK_START_EXAMPLES,
    RECOMMENDED_MODELS,
    reload_env_settings,
)
from core.openai_client import OpenAIClient, OpenAIClientError
from core.skill_loader import SkillNotFoundError, SkillRegistry, SkillValidationError

st.set_page_config(
    page_title="PromptForge — Multi-Agent Prompt Studio",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
.pf-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.35rem;
}
.pf-badge-green {
    background-color: rgba(16, 185, 129, 0.16);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}
.pf-badge-red {
    background-color: rgba(239, 68, 68, 0.16);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.pf-badge-amber {
    background-color: rgba(245, 158, 11, 0.16);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.pf-badge-blue {
    background-color: rgba(59, 130, 246, 0.16);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.4);
}
.pf-card {
    padding: 0.9rem 1.1rem;
    border-radius: 0.65rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    background-color: rgba(128, 128, 128, 0.05);
    margin-bottom: 0.75rem;
}
</style>
"""

_NEW_SKILL_STARTER_BODY = """# Skill: Custom Framework

## Core Principle
Describe the guiding philosophy of this prompting framework.

## Mandatory Output Template
Structure the engineered prompt using these exact sections:

```markdown
### Context
[Relevant background and constraints]

### Objective
[Exact deliverable]

### Instructions
1. [Step 1]
2. [Step 2]
```

## Strict Execution Rules
1. Populate every section with concrete details from the user's meta-prompt.
2. Output **ONLY** the engineered prompt—no conversational preamble or postscript.
"""


def _init_session_state() -> None:
    """Initialize required Streamlit session state keys safely."""
    defaults: dict[str, Any] = {
        "meta_prompt_input": "",
        "last_trace": None,
        "last_pipeline_error": None,
        "api_base_url": (os.environ.get("API_BASE_URL") or DEFAULT_API_BASE_URL).strip(),
        "custom_model_override": "",
        "trigger_regenerate": False,
        "skill_saved_msg": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_quick_example(prompt_text: str) -> None:
    """Callback to populate the meta-prompt editor before widget rendering."""
    st.session_state["meta_prompt_input"] = prompt_text
    st.session_state["last_pipeline_error"] = None
    st.session_state["last_trace"] = None


def _trigger_regenerate(prompt_text: str) -> None:
    """Callback to queue pipeline regeneration before text_area instantiation."""
    st.session_state["meta_prompt_input"] = prompt_text
    st.session_state["trigger_regenerate"] = True
    st.session_state["last_pipeline_error"] = None
    st.session_state["last_trace"] = None


def _rank_available_models(
    available_models: list[str],
    default_model: str | None = None,
    include_default_if_missing: bool = True,
) -> list[str]:
    """Rank discovered models with active default_model and RECOMMENDED_MODELS first."""
    prioritized: list[str] = []
    available_set = set(available_models)
    active_default = (default_model or DEFAULT_MODEL).strip()

    preferred_order = list(
        dict.fromkeys(
            m for m in [active_default, *RECOMMENDED_MODELS] if m
        )
    )
    for rec in preferred_order:
        if rec in available_set and rec not in prioritized:
            prioritized.append(rec)

    for tag in available_models:
        if tag not in prioritized:
            prioritized.append(tag)

    # For remote gateways whose /v1/models may omit a valid tag, keep default selectable
    if include_default_if_missing and active_default and active_default not in prioritized:
        prioritized.insert(0, active_default)

    return prioritized


def _execute_pipeline(
    orchestrator: PromptEngineeringOrchestrator,
    meta_prompt: str,
    model: str,
    selected_option: str,
    max_retries: int,
    temperature: float,
) -> None:
    """Run the 3-agent pipeline with live st.status updates."""
    st.session_state["last_pipeline_error"] = None
    st.session_state["last_trace"] = None
    with st.status("Running multi-agent prompt engineering pipeline...", expanded=True) as status_box:
        stage_log = st.empty()
        lines: list[str] = []

        def on_stage(stage: str, payload: dict[str, Any]) -> None:
            if stage == "classify_start":
                lines.append(
                    f"**Stage 1 — Classifier Agent:** Analyzing intent using `{payload['model']}`..."
                )
                status_box.update(label="Stage 1/3: Classifying meta-prompt & routing...")
            elif stage == "classify_done":
                c = payload["classification"]
                conf_pct = int(round(float(c["confidence"]) * 100))
                lines.append(
                    f"**Stage 1 Complete:** Selected `{c['selected_skill_id']}` "
                    f"(confidence `{conf_pct}%`) — {c['intent_summary']}"
                )
            elif stage == "classify_skipped":
                c = payload["classification"]
                lines.append(
                    f"**Stage 1 — Manual Framework Override:** Locked `{c['selected_skill_id']}` "
                    f"(Classifier bypassed)."
                )
            elif stage == "skill_loaded":
                lines.append(
                    f"**Skill Injection:** Loaded `{payload['skill_name']}` from `skills/{payload['skill_id']}.md`."
                )
            elif stage == "generate_start":
                attempt = payload["attempt"]
                total = payload["max_attempts"]
                action = (
                    "Refining prompt with validator critique"
                    if payload["is_retry"]
                    else "Generating initial engineered prompt"
                )
                lines.append(
                    f"**Stage 2 — Generator Agent (Attempt {attempt}/{total}):** {action}..."
                )
                status_box.update(
                    label=f"Stage 2/3: Generating prompt (attempt {attempt}/{total})..."
                )
            elif stage == "generate_done":
                lines.append(
                    f"**Generator Output:** Produced {payload['chars']} characters."
                )
            elif stage == "validate_start":
                attempt = payload["attempt"]
                total = payload["max_attempts"]
                lines.append(
                    f"**Stage 3 — Validator Agent (Attempt {attempt}/{total}):** Auditing structure & intent fidelity..."
                )
                status_box.update(
                    label=f"Stage 3/3: Validating output (attempt {attempt}/{total})..."
                )
            elif stage == "validate_done":
                v = payload["validation"]
                verdict = "PASS" if (v["is_valid"] and v["score"] >= MIN_PASS_SCORE) else "FAIL"
                lines.append(
                    f"**Validation Result (Attempt {payload['attempt']}):** `{verdict}` "
                    f"| Score: `{v['score']}/10` | Framework: `{v['framework_compliance']}` "
                    f"| Intent: `{v['intent_alignment']}` ({payload['duration_seconds']}s)"
                )

            stage_log.markdown("\n\n".join(lines))

        try:
            trace = orchestrator.run_pipeline(
                user_meta_prompt=meta_prompt,
                model=model,
                selected_option=selected_option,
                max_retries=max_retries,
                generate_temperature=temperature,
                on_stage_update=on_stage,
            )
            st.session_state["last_trace"] = trace.model_dump()
            final_label = (
                f"Pipeline complete — {trace.skill_name} "
                f"(Score: {trace.final_validation.score}/10, {trace.total_duration_seconds}s)"
            )
            status_box.update(
                label=final_label,
                state="complete" if trace.passed else "error",
                expanded=False,
            )
        except (OpenAIClientError, SkillNotFoundError, ValueError) as exc:
            st.session_state["last_pipeline_error"] = str(exc)
            status_box.update(label="Pipeline failed", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            st.session_state["last_pipeline_error"] = f"Unexpected error: {exc}"
            status_box.update(label="Pipeline encountered an unexpected error", state="error", expanded=True)
            st.error(f"Unexpected error: {exc}")


def _render_sidebar(
    registry: SkillRegistry,
) -> tuple[OpenAIClient, str | None, str, int, float, bool]:
    """Render sidebar controls and return active configuration."""
    st.sidebar.title("Configuration & API Engine")

    # Connection & credential check
    client = OpenAIClient(base_url=st.session_state["api_base_url"])
    is_connected = client.check_connection()

    col_status, col_refresh = st.sidebar.columns([3, 1])
    with col_status:
        if is_connected:
            st.markdown(
                f'<span class="pf-badge pf-badge-green">● Connected</span>'
                f'<span style="font-size:0.78rem;opacity:0.75;">{client.base_url}</span>',
                unsafe_allow_html=True,
            )
        elif not client.has_api_key:
            st.markdown(
                '<span class="pf-badge pf-badge-amber">● API Key Missing</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="pf-badge pf-badge-red">● Endpoint Offline</span>',
                unsafe_allow_html=True,
            )
    with col_refresh:
        if st.button("↻", help="Reload .env, API connection, and skill registry", use_container_width=True):
            reloaded_url, _, _ = reload_env_settings()
            st.session_state["api_base_url"] = reloaded_url
            registry.refresh()
            st.rerun()

    if not is_connected:
        st.sidebar.error(
            client.last_error
            or "API endpoint is unreachable. Check `API_BASE_URL` and `API_KEY` in `.env`."
        )

    # Model picker
    st.sidebar.subheader("Model Selection")
    available_models = client.get_available_models() if is_connected else []
    active_default_model = (os.environ.get("DEFAULT_MODEL") or DEFAULT_MODEL).strip()
    ranked_models = _rank_available_models(
        available_models,
        default_model=active_default_model,
        include_default_if_missing=not client.is_local_endpoint,
    )

    selected_model: str | None = None
    if is_connected and ranked_models:
        selected_model = st.sidebar.selectbox(
            "Active Model",
            options=ranked_models,
            index=0,
            help="Model passed to OpenAI Chat Completions API across all 3 agents.",
        )
        if (
            client.is_local_endpoint
            and active_default_model
            and active_default_model not in available_models
        ):
            st.sidebar.caption(
                f"Configured `DEFAULT_MODEL` (`{active_default_model}`) is not installed on this local endpoint; using `{selected_model}`."
            )
        override = st.session_state.get("custom_model_override", "").strip()
        if override:
            selected_model = override
            st.sidebar.caption(f"Using custom model override: `{selected_model}`")
    elif is_connected:
        st.sidebar.warning(
            "Connected, but no models are installed on this local endpoint. "
            "Pull a lightweight model (e.g. `qwen2.5:1.5b` or `gemma3:1b`) or set a Custom Model ID below."
        )
        override = st.session_state.get("custom_model_override", "").strip()
        if override:
            selected_model = override
            st.sidebar.caption(f"Using custom model override: `{selected_model}`")
    else:
        st.sidebar.caption("Configure API credentials in `.env` to load available models.")

    st.sidebar.divider()

    # Dynamic Framework Selector
    st.sidebar.subheader("Prompt Framework Selector")
    catalog = registry.get_catalog_summary()
    catalog_map = {item["id"]: item for item in catalog}
    framework_options = [AUTO_MODE_ID] + [item["id"] for item in catalog]

    def format_framework(opt: str) -> str:
        if opt == AUTO_MODE_ID:
            return AUTO_MODE_LABEL
        entry = catalog_map.get(opt)
        if not entry:
            return opt
        return f"[{entry['category']}] {entry['name']}"

    selected_framework = st.sidebar.selectbox(
        "Framework Mode",
        options=framework_options,
        index=0,
        format_func=format_framework,
    )

    if selected_framework == AUTO_MODE_ID:
        st.sidebar.caption(
            f"ClassifierAgent analyzes your meta-prompt and selects from **{len(catalog)}** loaded skills in `skills/*.md`."
        )
    else:
        entry = catalog_map[selected_framework]
        st.sidebar.caption(
            f"**Manual Override:** Classifier bypassed. {entry['description']}"
        )

    if st.sidebar.button("Reload `skills/*.md`", use_container_width=True):
        registry.refresh()
        st.rerun()

    st.sidebar.divider()

    # Advanced Pipeline & Endpoint Settings
    with st.sidebar.expander("Pipeline & API Settings", expanded=False):
        max_retries = st.slider(
            "Max Validation Retries",
            min_value=0,
            max_value=3,
            value=MAX_RETRIES,
            step=1,
            help="Maximum self-correction passes if ValidatorAgent scores < 7 or fails compliance.",
        )
        temperature = st.slider(
            "Generation Temperature",
            min_value=0.1,
            max_value=0.9,
            value=GENERATE_TEMP,
            step=0.05,
            help="Classifier & Validator stay pinned at 0.1 for deterministic JSON.",
        )
        st.text_input(
            "Custom Model ID Override",
            key="custom_model_override",
            placeholder="e.g., qwen2.5:1.5b or gemma3:1b",
            help="Optional: overrides the Active Model dropdown when set.",
        )
        new_url = st.text_input(
            "API Base URL",
            value=st.session_state["api_base_url"],
            help="Defaults to API_BASE_URL from .env.",
        ).strip()
        if new_url and new_url != st.session_state["api_base_url"]:
            st.session_state["api_base_url"] = new_url
            st.rerun()
        key_state = "✓ API key loaded from `.env`" if client.api_key else "✗ No API key found in `.env`"
        st.caption(key_state)

    return (
        client,
        selected_model,
        selected_framework,
        max_retries,
        float(temperature),
        is_connected,
    )


def _render_trace_details(trace: PipelineTrace) -> None:
    """Render the 3-stage agent execution audit and final prompt deliverable."""
    st.subheader("Agent Execution Audit")

    col_c, col_g, col_v = st.columns(3)
    with col_c:
        c = trace.classification
        mode_badge = (
            '<span class="pf-badge pf-badge-amber">Manual Framework Override</span>'
            if c.is_manual_override
            else f'<span class="pf-badge pf-badge-blue">Auto Routed • {int(round(c.confidence * 100))}%</span>'
        )
        st.markdown(
            f"""<div class="pf-card">
            <div><strong>1. Classifier / Router</strong></div>
            <div style="margin:0.35rem 0;">{mode_badge}</div>
            <div><strong>Skill:</strong> <code>{c.selected_skill_id}</code></div>
            <div><strong>Domain:</strong> {c.inferred_parameters.domain}</div>
            <div style="font-size:0.85rem;margin-top:0.35rem;">{c.intent_summary}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_g:
        st.markdown(
            f"""<div class="pf-card">
            <div><strong>2. Prompt Generator</strong></div>
            <div style="margin:0.35rem 0;"><span class="pf-badge pf-badge-blue">{trace.skill_name}</span></div>
            <div><strong>Skill File:</strong> <code>skills/{trace.skill_id}.md</code></div>
            <div><strong>Attempts:</strong> {len(trace.attempts)} ({trace.retries_used} retries)</div>
            <div style="font-size:0.85rem;margin-top:0.35rem;">Lazy-loaded only the active skill instructions for lean context.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_v:
        v = trace.final_validation
        score_badge_cls = "pf-badge-green" if trace.passed else "pf-badge-red"
        verdict_txt = "PASS" if trace.passed else "NEEDS REVIEW"
        st.markdown(
            f"""<div class="pf-card">
            <div><strong>3. Quality Validator</strong></div>
            <div style="margin:0.35rem 0;"><span class="pf-badge {score_badge_cls}">{verdict_txt} • {v.score}/10</span></div>
            <div><strong>Framework:</strong> {v.framework_compliance} &nbsp;|&nbsp; <strong>Intent:</strong> {v.intent_alignment}</div>
            <div><strong>Total Runtime:</strong> {trace.total_duration_seconds:.2f}s</div>
            <div style="font-size:0.85rem;margin-top:0.35rem;">{v.critique}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Detailed Validation Trace & Self-Correction Diffs", expanded=False):
        st.markdown(f"**Routing Rationale:** {trace.classification.rationale}")
        for idx, attempt in enumerate(trace.attempts):
            val = attempt.validation
            badge = "PASS" if (val.is_valid and val.score >= MIN_PASS_SCORE) else "FAIL"
            st.markdown(
                f"#### Attempt {attempt.attempt_number} — `{badge}` ({val.score}/10, {attempt.duration_seconds:.2f}s)"
            )
            st.markdown(
                f"- **Framework Compliance:** `{val.framework_compliance}` | "
                f"**Intent Alignment:** `{val.intent_alignment}`\n"
                f"- **Compliance Notes:** {val.compliance_notes}\n"
                f"- **Critique:** {val.critique}"
            )
            if attempt.critique_used:
                st.info(f"Critique fed into this attempt: {attempt.critique_used}")

            if idx > 0:
                prev_lines = trace.attempts[idx - 1].generated_prompt.splitlines()
                curr_lines = attempt.generated_prompt.splitlines()
                diff_text = "\n".join(
                    difflib.unified_diff(
                        prev_lines,
                        curr_lines,
                        fromfile=f"attempt_{idx}",
                        tofile=f"attempt_{idx + 1}",
                        lineterm="",
                    )
                )
                if diff_text:
                    st.markdown("**Refinement Diff vs Previous Attempt:**")
                    st.code(diff_text, language="diff")

    st.divider()

    # Final Prompt Deliverable Card
    st.subheader("Final Engineered Prompt")
    if not trace.passed:
        st.warning(
            f"Validation exhausted {trace.retries_used} retry(s) with best score {trace.final_validation.score}/10. "
            "Review the critique above or click **Regenerate**."
        )

    st.code(trace.final_prompt, language="markdown", wrap_lines=True)

    btn_dl, btn_regen, _spacer = st.columns([1.2, 1.2, 4])
    with btn_dl:
        st.download_button(
            label="Download .md",
            data=trace.final_prompt,
            file_name=f"{trace.skill_id}_prompt.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with btn_regen:
        st.button(
            "Regenerate",
            on_click=_trigger_regenerate,
            args=(trace.user_meta_prompt,),
            use_container_width=True,
        )


def _render_studio_tab(
    registry: SkillRegistry,
    client: OpenAIClient,
    selected_model: str | None,
    selected_framework: str,
    max_retries: int,
    temperature: float,
    is_connected: bool,
) -> None:
    """Render Tab 1: Prompt Engineering Studio."""
    st.markdown("**Quick-Start Examples:**")
    chip_cols = st.columns(len(QUICK_START_EXAMPLES))
    for col, example in zip(chip_cols, QUICK_START_EXAMPLES):
        with col:
            st.button(
                example["label"],
                key=f"chip_{example['label']}",
                on_click=_apply_quick_example,
                args=(example["prompt"],),
                use_container_width=True,
            )

    meta_prompt = st.text_area(
        "Raw Meta-Prompt",
        key="meta_prompt_input",
        height=150,
        placeholder=(
            "Describe what you want to build or generate — e.g., "
            "'Build a research agent for fintech news' or "
            "'Cinematic sci-fi Mumbai auto-rickshaw scene'..."
        ),
    )

    can_run = bool(is_connected and selected_model)
    run_clicked = st.button(
        "Engineer & Validate Prompt",
        type="primary",
        disabled=not can_run,
    )

    if not can_run:
        st.info(
            "Configure `API_BASE_URL` and `API_KEY` in `.env` and ensure an active model is selected (see `README.md`)."
        )

    should_execute = run_clicked or st.session_state.pop("trigger_regenerate", False)
    if should_execute:
        if not meta_prompt.strip():
            st.warning("Please enter a meta-prompt or select a quick-start example above.")
        elif selected_model:
            orchestrator = PromptEngineeringOrchestrator(
                skill_registry=registry,
                llm_client=client,
            )
            _execute_pipeline(
                orchestrator=orchestrator,
                meta_prompt=meta_prompt,
                model=selected_model,
                selected_option=selected_framework,
                max_retries=max_retries,
                temperature=temperature,
            )

    if st.session_state.get("last_pipeline_error"):
        st.error(st.session_state["last_pipeline_error"])

    raw_trace = st.session_state.get("last_trace")
    if raw_trace:
        trace = PipelineTrace.model_validate(raw_trace)
        _render_trace_details(trace)


def _render_skill_library_tab(registry: SkillRegistry) -> None:
    """Render Tab 2: Skill Library Inspector & Custom Framework Builder."""
    if registry.load_errors:
        with st.expander(
            f"Malformed Skill File Warnings ({len(registry.load_errors)})",
            expanded=True,
        ):
            for err in registry.load_errors:
                st.warning(f"`{err.filepath.name}` skipped: {err.error}")

    col_inspect, col_create = st.columns([1.15, 1.0], gap="large")

    with col_inspect:
        st.subheader("Skill Library Inspector")
        catalog = registry.get_catalog_summary()
        if not catalog:
            st.info("No skills found in `skills/`. Create one on the right.")
        else:
            id_to_meta = {item["id"]: item for item in catalog}
            chosen_id = st.selectbox(
                "Select Framework Skill",
                options=list(id_to_meta.keys()),
                format_func=lambda sid: f"{id_to_meta[sid]['name']} ({sid}.md)",
            )
            skill = registry.get_skill_by_id(chosen_id)
            raw_md = registry.get_raw_markdown(chosen_id)

            st.markdown(
                f"**Category:** `{skill.category}` &nbsp;|&nbsp; **ID:** `{skill.id}`  \n"
                f"**File:** `{skill.filepath.name}`  \n"
                f"**Description:** {skill.description}  \n"
                f"**Best For:** {skill.best_for or '—'}  \n"
                f"**Trigger Keywords:** `{', '.join(skill.trigger_keywords) or '—'}`"
            )

            view_mode = st.radio(
                "View Format",
                options=["Rendered Instructions", "Raw Markdown (.md)"],
                horizontal=True,
            )
            if view_mode == "Rendered Instructions":
                st.markdown(skill.body)
            else:
                st.code(raw_md, language="markdown")

    with col_create:
        st.subheader("Create New Framework Skill")
        st.caption(
            "Saves a standalone `.md` file directly into `skills/` and hot-reloads the Framework Selector."
        )
        saved_msg = st.session_state.pop("skill_saved_msg", None)
        if saved_msg:
            st.success(saved_msg)
        with st.form("create_skill_form", clear_on_submit=False):
            new_id = st.text_input(
                "Skill ID (filename slug)",
                placeholder="e.g., chain_of_verification",
                help="Lowercase letters, digits, '_' or '-' only.",
            )
            new_name = st.text_input(
                "Display Name",
                placeholder="e.g., Chain of Verification (CoVe)",
            )
            new_category = st.text_input(
                "Category",
                value="General & System Prompts",
            )
            new_description = st.text_area(
                "Short Description",
                height=70,
                placeholder="Concise summary of what this framework does.",
            )
            new_best_for = st.text_input(
                "Best For",
                placeholder="Fact-checking, multi-step verification, high-stakes analysis",
            )
            new_keywords = st.text_input(
                "Trigger Keywords (comma-separated)",
                placeholder="verify, fact check, audit, hallucination, accuracy",
            )
            new_body = st.text_area(
                "Markdown Skill Instructions & Template",
                value=_NEW_SKILL_STARTER_BODY,
                height=260,
            )
            overwrite = st.checkbox("Overwrite existing file if ID matches", value=False)
            submitted = st.form_submit_button("Save Skill to skills/*.md", type="primary")

            if submitted:
                try:
                    created = registry.save_new_skill(
                        skill_id=new_id,
                        name=new_name,
                        category=new_category,
                        description=new_description,
                        best_for=new_best_for,
                        trigger_keywords=new_keywords,
                        body=new_body,
                        overwrite=overwrite,
                    )
                    st.session_state["skill_saved_msg"] = (
                        f"Saved `{created.filepath.name}` and registered **{created.name}**!"
                    )
                    st.rerun()
                except SkillValidationError as exc:
                    st.error(str(exc))


def main() -> None:
    """Application entrypoint."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _init_session_state()

    registry = SkillRegistry()
    (
        client,
        selected_model,
        selected_framework,
        max_retries,
        temperature,
        is_connected,
    ) = _render_sidebar(registry)

    st.title("⚒️ PromptForge Studio")
    st.caption(
        "Enterprise Multi-Agent Prompt Engineering Studio · Dynamic Markdown Skills · "
        "OpenAI-Compatible Chat Completions Client (.env Configured)"
    )

    tab_studio, tab_library = st.tabs(
        ["Prompt Engineering Studio", "Skill Library & Builder"]
    )
    with tab_studio:
        _render_studio_tab(
            registry=registry,
            client=client,
            selected_model=selected_model,
            selected_framework=selected_framework,
            max_retries=max_retries,
            temperature=temperature,
            is_connected=is_connected,
        )
    with tab_library:
        _render_skill_library_tab(registry)


if __name__ == "__main__":
    main()
