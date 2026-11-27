---
id: costar
name: CoSTAR Framework
category: General & System Prompts
description: High-precision structured prompting framework (Context, Objective, Steps, Tools, Actions, Reflection) ideal for complex analytical, strategic, and multi-step LLM tasks.
best_for: Complex problem solving, business analysis, strategic planning, multi-step reasoning tasks, technical workflows, and prompts requiring self-verification.
trigger_keywords: [analyze, strategy, plan, complex, step by step, reasoning, report, framework, evaluate, solve, decision, enterprise, workflow]
---

# Skill: CoSTAR Framework

## Core Principle
Force structured reasoning and eliminate ambiguity by separating background context, concrete objectives, step-by-step execution logic, tools/resources, output formatting actions, and a mandatory self-reflection loop.

## Mandatory Output Template
Structure the engineered prompt using these exact six sections:

```markdown
### Context
[Set the scene: background information, domain environment, target audience, and why this task matters.]

### Objective
[Define the exact, measurable deliverable or outcome the AI must achieve.]

### Steps
1. [Step 1: Initial decomposition or analysis]
2. [Step 2: Core execution or evaluation]
3. [Step 3: Synthesis and edge-case verification]

### Tools
[Specify any tools, data sources, reference frameworks, or analytical methods the AI should utilize.]

### Actions
[Define the expected output style, tone, structural format (e.g., Markdown tables, executive summary, code blocks), and length constraints.]

### Reflection
[Instruct the AI to perform a self-check against accuracy, completeness, bias, and formatting rules before finalizing its response.]
```

## Strict Execution Rules

1. Tailor every section directly to the user's meta-prompt with concrete domain details.
2. Ensure the **Steps** section forces logical chain-of-thought execution rather than jumping straight to an answer.
3. Ensure the **Reflection** section contains specific verification questions tailored to the task.
4. Output **ONLY** the engineered CoSTAR prompt—no conversational chatter before or after.
