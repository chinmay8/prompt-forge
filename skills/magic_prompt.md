---
id: magic_prompt
name: Magic Prompt Framework
category: General & System Prompts
description: Clean, versatile persona-driven prompt framework (Role, Context, Task, Constraints & Format, Examples) ideal for copywriting, coding, tutoring, and everyday high-quality tasks.
best_for: Copywriting, emails, marketing content, coding assistance, tutoring, summarization, and general-purpose single-turn LLM prompts.
trigger_keywords: [write, draft, email, blog, copy, explain, tutor, coach, summarize, generate text, code, script, marketing, post]
---

# Skill: Magic Prompt Framework

## Core Principle
Establish an authoritative persona, ground it in situational context, assign an unambiguous task, enforce strict formatting guardrails, and anchor quality with a concrete example.

## Mandatory Output Template
Structure the engineered prompt using these exact sections:

```markdown
**Role:**
[Define the exact expert persona the AI should adopt, including domain expertise and perspective.]

**Context:**
[Provide essential background details, target audience, situational constraints, and source inputs.]

**Task:**
[State the exact action or deliverable to produce in clear, imperative language.]

**Constraints & Format:**
- [Rule 1: Tone and style guidelines]
- [Rule 2: Structural layout—e.g., bullet points, Markdown table, code block, word/length limits]
- [Rule 3: Negative constraints—what to strictly avoid]

**Examples:**
[Provide a concise sample input/output or stylistic snippet demonstrating what "good" looks like for this task.]
```

## Strict Execution Rules

1. Choose a highly specific persona in **Role** (e.g., "Senior B2B SaaS Product Marketer" rather than "Writer").
2. Include concrete, actionable rules in **Constraints & Format** (word counts, tone, formatting).
3. Always provide a relevant, high-signal **Examples** section to calibrate the model's output style.
4. Output **ONLY** the engineered prompt—no introductory or concluding commentary.
