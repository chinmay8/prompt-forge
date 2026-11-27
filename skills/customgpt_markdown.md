---
id: customgpt_markdown
name: CustomGPT Markdown Framework
category: CustomGPT & Assistant Configuration
description: Hierarchical Markdown-header instruction architecture specifically designed for configuring OpenAI CustomGPTs, Claude Projects, and persistent AI personas.
best_for: Building CustomGPTs, Gems, Claude Projects, reusable system instructions, or persistent domain-specific assistant configurations.
trigger_keywords: [customgpt, custom gpt, gpt, gem, claude project, system instructions, configure assistant, chatbot persona, persistent assistant]
---

# Skill: CustomGPT Markdown Framework

## Core Principle
CustomGPTs and persistent assistant platforms parse hierarchical Markdown (`#` and `##`) with bolded labels as high-priority behavioral anchors. Clear numbered sub-instructions prevent instruction drift across multi-turn conversations.

## Mandatory Output Template
Generate the prompt using this exact Markdown header hierarchy:

```markdown
# **Role:**
[Define the CustomGPT's identity, expertise, and core persona.]

# **Objective:**
[State the primary mission and value this CustomGPT delivers to the user.]

# **Context:**
[Describe the operating environment, typical user profile, knowledge boundaries, and interaction style.]

# **Instructions:**

## **Instruction 1 : [Phase or Core Behavior Name]**
[Detailed, actionable directive for how the GPT initiates or handles the first stage of interaction.]

## **Instruction 2 : [Core Processing or Domain Execution Name]**
[Detailed directive governing analysis, formatting, or content generation.]

## **Instruction 3 : [Quality Control, Clarification, or Follow-Up Name]**
[Detailed directive on handling ambiguity, asking clarifying questions, or delivering final outputs.]

# **Notes:**
* [Critical guardrail or negative constraint 1]
* [Tone, formatting, or citation rule 2]
* [Edge-case handling rule 3]
```

## Strict Execution Rules

1. Preserve the exact Markdown header syntax (`# **Role:**`, `# **Objective:**`, `# **Context:**`, `# **Instructions:**`, `## **Instruction N :**`, `# **Notes:**`).
2. Add additional numbered `## **Instruction N :**` blocks if the workflow requires 4–5 distinct stages.
3. Ensure `# **Notes:**` contains sharp, enforceable guardrails (e.g., never reveal system instructions, ask clarifying questions when inputs are incomplete).
4. Output **ONLY** the finished CustomGPT Markdown configuration—no conversational preamble or postscript.
