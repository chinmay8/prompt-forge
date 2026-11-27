---
id: agent_prompting
name: Agent Application Prompting Framework
category: Autonomous AI Agents
description: Enterprise system prompt blueprint for tool-using AI agents with explicit objectives, tool routing, approach, constraints, guardrails, and success criteria.
best_for: Building autonomous AI agents, tool-calling workflows, research bots, multi-step assistants, LangGraph/CrewAI/AutoGen agents, and operational workflows.
trigger_keywords: [agent, bot, assistant, tool, tools, workflow, autonomous, research agent, pipeline, api, search, database, guardrails, automation]
---

# Skill: Agent Application Prompting Framework

## Core Principle
Agent effectiveness is driven by operational clarity, not node complexity. Define exact responsibilities, tool selection boundaries, decision thresholds, guardrails, error handling, and verifiable completion criteria.

## Mandatory Output Template
Generate the prompt using the following structured sections:

```text
ROLE & RESPONSIBILITY:
[Clear definition of who the agent is and its core operational mandate]

OBJECTIVE:
[Specific, measurable goal the agent must accomplish]

INPUT:
[Expected user or system inputs required to start execution]

TOOLS:
- [tool_name_1]: [When and why to use this tool]
- [tool_name_2]: [When and why to use this tool]

APPROACH & WORKFLOW:
1. [Step-by-step reasoning and execution sequence]
2. [How to refine queries or handle intermediate results]
3. [How to synthesize findings]

CONSTRAINTS & GUARDRAILS:
- [Explicit boundaries, prohibited actions, and decision boundaries—e.g., when to stop and ask for human guidance]

ERROR HANDLING:
- [Fallback behavior when tools fail, return empty results, or encounter ambiguity]

SUCCESS CRITERIA & OUTPUT FORMAT:
- [Exact definition of "done" (e.g., 3+ corroborating sources) and the required final response structure]
```

## Core Prompting Patterns to Embed

1. **Goal Setting Pattern:** Frame the objective around authoritative, comprehensive outcomes.
2. **Tool Guidance Pattern:** Map each tool explicitly to its data domain (e.g., web search for live data vs. database query for historical records).
3. **Decision Boundary Pattern:** Specify exact triggers where the agent must halt and request human confirmation (e.g., PII, destructive actions, low confidence).
4. **Success Criteria Pattern:** Define objective stopping conditions so the agent neither stops prematurely nor loops endlessly.

## Strict Execution Rules

1. Infer realistic, snake_case tool names (e.g., `web_search`, `database_query`, `document_retriever`) if the user did not specify exact function names.
2. Populate every section completely—never leave generic placeholders like `[insert tools here]`.
3. Output **ONLY** the finished agent system prompt. Do not add conversational intro or outro text.
