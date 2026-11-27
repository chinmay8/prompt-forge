# PromptForge — Enterprise Multi-Agent Prompt Engineering Studio

Transform rough, ambiguous user meta-prompts into production-grade, framework-compliant prompts using a **multi-agent pipeline** powered by Python, Streamlit, and any **OpenAI-compatible Chat Completions API** (such as [Ollama Cloud](https://ollama.com/), OpenAI, OpenRouter, or a local `/v1` server).

---

## Architecture Overview

```text
[User Meta-Prompt] + [Framework Selection: "Auto" or Specific Skill]
         │
         ├─── If "Auto" ──────────► [1. ClassifierAgent] (JSON routing)
         │                                │
         └─── If Specific Skill ──► [SkillRegistry] (lazy-loads skills/<id>.md)
                                          ▼
                                   [2. GeneratorAgent] (strict framework output)
                                          ▼
                                   [3. ValidatorAgent] (structure & intent audit)
                                          ├─── PASS (score >= 7) ──► Final Prompt + Audit Log
                                          └─── FAIL ──► Self-Correction Loop (up to 2 retries)
```

### Repository Structure

```text
├── app.py                      # Streamlit web studio (Studio + Skill Library/Builder)
├── .env                        # API Base URL, API Key, and default model configuration
├── requirements.txt            # streamlit, openai, pydantic, pyyaml, python-dotenv
├── README.md                   # Setup & API configuration guide
├── core/
│   ├── __init__.py
│   ├── config.py               # .env-backed API defaults, agent parameters & quick-start examples
│   ├── skill_loader.py         # YAML-frontmatter indexer + lazy Markdown body loader
│   ├── openai_client.py        # OpenAI Chat Completions wrapper (JSON mode + resilient fallback)
│   └── agents.py               # ClassifierAgent, GeneratorAgent, ValidatorAgent, Orchestrator
├── skills/                     # Zero-code extensible Markdown skill files
│   ├── master_director.md
│   ├── master_prompt.md
│   ├── agent_prompting.md
│   ├── costar.md
│   ├── magic_prompt.md
│   └── customgpt_markdown.md
└── tests/
    └── test_promptforge.py     # Unit & integration test suite
```

---

## API Configuration (`.env`)

[`core/openai_client.py`](core/openai_client.py) uses the official `openai` Python SDK (`OpenAI(base_url=..., api_key=...)`) and reads your endpoint, credentials, and default model directly from `.env`:

```dotenv
# Local low-RAM Ollama endpoint (default)
API_BASE_URL=http://localhost:11434/v1
API_KEY=ollama
DEFAULT_MODEL=qwen2.5:1.5b
LOCAL_NUM_CTX=2048
MAX_TOKENS=2048
```

Recommended lightweight models for constrained local RAM (configured in `RECOMMENDED_MODELS` in [`core/config.py`](core/config.py)):
- `qwen2.5:1.5b` (default)
- `gemma3:1b`

When targeting a local loopback endpoint (`localhost` / `127.0.0.1`), `core/openai_client.py` automatically caps `num_ctx` at `2048` (strictly capped at `2048`; lower values can be set via `LOCAL_NUM_CTX`) and limits completions via `max_tokens=2048` (overrideable via `MAX_TOKENS`) to prevent memory spikes.

You can switch to **any** remote OpenAI-compatible Chat Completions provider simply by updating `.env`:

- **Local Ollama (`/v1`):** `API_BASE_URL=http://localhost:11434/v1`, `API_KEY=ollama`, `DEFAULT_MODEL=qwen2.5:1.5b`
- **Ollama Cloud:** `API_BASE_URL=https://ollama.com/v1`, `API_KEY=...`, `DEFAULT_MODEL=gpt-oss:20b-cloud`
- **OpenAI:** `API_BASE_URL=https://api.openai.com/v1`, `API_KEY=...`, `DEFAULT_MODEL=gpt-4o-mini`
- **OpenRouter:** `API_BASE_URL=https://openrouter.ai/api/v1`, `API_KEY=...`, `DEFAULT_MODEL=...`

---

## Quick Start

### 1. Configure `.env`

Update [`.env`](.env) in the project root with your API base URL, API key, and preferred default model.

### 2. Install Python Dependencies

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Studio

```bash
streamlit run app.py
```

---

## Adding Custom Prompting Frameworks (Zero Python Changes)

Add any `.md` file to `skills/` with YAML frontmatter (`id`, `name`, `category`, `description`, `best_for`, `trigger_keywords`) and Markdown instructions—or create one directly in **Tab 2: Skill Library & Builder** inside the UI. Click **Reload `skills/*.md`** in the sidebar to include it immediately in both the dropdown and `ClassifierAgent` routing.

## Running Tests

```bash
pytest -v
```
