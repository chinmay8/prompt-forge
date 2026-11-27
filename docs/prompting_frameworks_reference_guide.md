# Comprehensive Prompting Frameworks & Reference Guide

This document consolidates the prompting frameworks, visual cheat sheets, agent patterns, and practical examples extracted from your course notes.

---

## 1. Image and Video Generation Prompts

### A. The Master Director Framework
* **Core Concept:** Diffusion models understand photography and cinematography, not logic.
* **Formula:**
  ```text
  [Subject], [Action], [Environment], [Atmosphere], [Camera], [Lens], [Lighting], [Style Reference]
  ```

### B. The Master Prompt Framework
* **Formula:**
  ```text
  [Emotional tone] + [Visual reference] + [Subject] + [Composition] + [Lighting] + [Camera settings]
  ```
* **Sample Example:**
  > *"Intense, mystery, dark, suspense visual like The dark knight, dune, star wars and star trek where a Mumbai rikshaw in interstellar galaxy riding with a female passenger, Steam Punk, Cosmic, Imax film"*

---

### C. Vocabulary Cheat Sheet (Photographer, Cinematographer, or Director)

| Category | Keywords & Options |
| :--- | :--- |
| **Angles** | Dutch, Low (Power), Closeup, Cowboy Shot |
| **Lenses** | Anamorphic, Macro, 50mm, 35mm |
| **Lighting** | Volumetric, Chiaroscuro, Rim, Bioluminescent, Natural |
| **Film Stock** | Kodak Portra, Fujifilm Velvia, Cinestill 800T |

---

### D. Prompt-Builder Cheat Sheet (Mix & Match Matrix)

Combine one element from each column below to construct detailed image or video generation prompts:

| [Subject + Action] | [Environment + Atmosphere] | [Camera + Lens] | [Lighting] | [Style] |
| :--- | :--- | :--- | :--- | :--- |
| A person holding and admiring a product | On a clean white studio background | Wide Shot (shows the whole scene) | Golden Hour (warm, soft light at sunrise/sunset) | Photorealistic (looks like a real photo) |
| A close-up of hands typing on a laptop | In a modern, sun-drenched open-plan office | Medium Shot (from the waist up) | Studio Lighting (clean, professional, controlled) | Cinematic (looks like a movie) |
| A family laughing together on a sofa | In a cozy, warm living room with a fireplace | Close-Up Shot (focuses on the face/product) | Soft Window Light (gentle, natural light) | Minimalist (clean, simple, uncluttered) |
| An athlete running in slow motion | On a misty forest trail at dawn | Extreme Close-Up (focuses on a single detail) | Rim Lighting (creates a glowing outline) | Clean Commercial Aesthetic |
| A chef meticulously plating a dish | In a professional, stainless steel kitchen | Cowboy Shot (mid-thigh up) | Volumetric Lighting (visible light rays, "god rays") | Vibrant Color Grading (rich, saturated colors) |
| A product rotating 360 degrees | In a futuristic, neon-lit cityscape at night | Low Angle Shot (makes the subject look powerful) | Neon Reflections (light bouncing off wet pavement) | Luxury Brand Aesthetic (elegant, premium feel) |
| A woman practicing yoga at sunrise | On a serene mountaintop overlooking clouds | High Angle Shot (looks down on the subject) | Backlighting (light source is behind the subject) | Documentary Style (realistic, unstaged feel) |
| A coffee mug with steam gently rising | On a rustic wooden table in a cafe | Macro Shot (extremely close, shows tiny details) | Chiaroscuro (dramatic high-contrast shadows) | Shot on Kodak Portra 400 (warm, cinematic film look) |
| Two people shaking hands in a meeting | In a sleek, modern boardroom | Overhead Shot (directly from above, "flat lay") | Lens Flare (bright light hits the lens) | Hyperrealistic (more detailed than a photo) |
| A car driving on a winding coastal road | On a scenic highway during a sunny day | Dutch Angle (camera is tilted for a dynamic feel) | Harsh Midday Sun (creates strong, sharp shadows) | 1980s Retro Film Look |
| A scientist looking into a microscope | In a clean, high-tech laboratory | Point-of-View (POV) Shot (see through subject's eyes) | Bioluminescent Glow (magical, natural light) | Unreal Engine 5 Render (video game quality) |
| A dog happily chasing a ball | In a lush, green suburban backyard | Handheld Shot (feels immediate and energetic) | Softbox Lighting (diffused studio light, flattering) | Organic and Natural Feel (earthy tones, authentic) |
| A person's eyes widening in surprise | Against a simple, colored background | Dolly Zoom (background changes, subject is static) | Three-Point Lighting (standard professional setup) | Black and White Noir (dramatic, high contrast) |
| A drop of liquid splashing in slow motion | In a minimalist, abstract space | Slow Motion (120fps) (dramatic, highlights detail) | Warm Tungsten Light (cozy, yellowish indoor light) | Futuristic / Sci-Fi Aesthetic |
| A child building with colorful blocks | In a bright, sunlit playroom | Shallow Depth of Field (blurry background, sharp subject) | Blue Hour (cool, deep blue light after sunset) | Apple Commercial Style (clean, minimalist, product-focused) |
| A person swiping on a smartphone | In a bustling urban environment | Drone Shot / Aerial View | Ambient Light (the natural light of the location) | Motion Graphics Overlay (animated text/shapes) |
| A product being unboxed by elegant hands | On a luxurious velvet surface | 85mm Lens (flattering for portraits/products) | Split Lighting (half the face/object is in shadow) | Gopro / Action Cam Style |
| A couple dancing in the rain | On a city street at night with neon signs | Anamorphic Lens (cinematic widescreen look) | Candlelight (very warm, intimate, soft light) | Vintage / Sepia Tone |
| A designer sketching on a tablet | In a creative studio filled with art | 35mm Lens (classic, natural field of view) | Overcast Day Lighting (soft, even, no harsh shadows) | Flat Design / 2D Animation Style |
| A group of friends toasting glasses | At a lively rooftop bar at sunset | Crane Shot (camera swoops up or down) | Spotlight (a focused beam of light on the subject) | Surreal / Dreamlike Quality |
| A person meditating peacefully | In a tranquil Japanese zen garden | Static Shot (no camera movement) | Hard Light (creates defined, sharp edges) | Pastel Color Palette |
| Animated data streams flowing | In a dark, abstract digital cyberspace | Rack Focus (focus shifts from foreground to background) | Prismatic / Rainbow Light Effects | Corporate / Business Aesthetic |
| A cat curiously looking at the camera | On a plush, comfortable armchair | Fish-eye Lens (distorted, ultra-wide view) | Silhouette (subject is dark against a bright background) | Watercolor or Painted Style |
| A person assembling a piece of furniture | In a well-lit, clean workshop | Time-lapse (shows a long process sped up) | Reflective/Caustic Light (light on water or metal) | Grunge / Urban Style |
| A beverage being poured into a glass | Against a dark, dramatic background | Gimbal Shot (smooth, flowing movement) | Flickering Light (like a fire or old projector) | Shot on Fujifilm Velvia (vivid, saturated colors) |
| A single flower blooming in time-lapse | In a serene, dew-covered garden | Tilt Shot (camera moves up or down vertically) | Moonlight (cool, low-light, mysterious) | Clean and Crisp Product Shot |
| A person putting on headphones and smiling | In a soundproof recording studio | Pan Shot (camera moves left or right horizontally) | Dramatic Downlighting (light from directly above) | Tech Ad Style (sleek, modern, often dark) |
| A close-up of shoes walking on pavement | On a rain-slicked city sidewalk | Tracking Shot (camera moves alongside the subject) | Edge Lighting (another term for Rim Lighting) | Warm and Inviting Feel |
| A product emerging from water | In a dark, mysterious void | Underwater Shot | Natural Morning Light (bright, clean, optimistic) | Bold and Graphic Style |
| A developer coding in a dark room | With multiple monitors displaying code | Zoom In/Out (dramatic reveal or focus) | Screen Glow (light emitted from a monitor or phone) | Epic / Blockbuster Movie Feel |

---

## 2. Agent Application Prompts

### A. Agent Prompting Framework
```text
Objective: Accomplish this goal...
Tools: You have access to...
Approach: Explore and determine...
Constraints: Avoid these actions...
Success: You'll know you're done when...
```

### B. Core Prompting Patterns (The Building Blocks)
1. **Goal Setting Pattern:**
   > *"Your objective is to gather comprehensive information about [topic] from multiple authoritative sources"*
2. **Tool Guidance Pattern:**
   > *"Use web_search for current information, use database_query for historical data"*
3. **Decision Boundary Pattern:**
   > *"If you encounter personal information, stop and request guidance"*
4. **Success Criteria Pattern:**
   > *"Consider the task complete when you have 3+ corroborating sources"*

### C. Practical Prompt Example: Research Agent Prompt
```text
OBJECTIVE: Research and summarize recent developments in sustainable packaging

APPROACH:
- Start by understanding the scope of the topic
- Search for information from the last 6 months
- Prioritize industry reports and academic sources
- Look for contrasting viewpoints

TOOLS:
- Use web_search for recent news and developments
- Use academic_search for research papers
- Use industry_database for market reports

GUIDELINES:
- Aim for 5–10 high-quality sources
- If initial searches are too broad, refine your query
- Stop searching once you have comprehensive coverage
- Focus on practical implementations over theory

SUCCESS: A balanced summary covering technology, adoption, challenges, and future outlook
```

### D. Essential Elements of an Effective Agent
*Note: Effectiveness is not about complexity or number of nodes—it's about clarity. The more details you define, the better the agent quality.*

* **Role & Responsibility:** What exactly does this agent do?
* **Tools:** What tools or external data can it access?
* **Input:** What's the required input from the user?
* **Workflow Details:** How should it behave? (Makes behavior more reliable)
* **Guardrails:** What are the boundaries and must-follow rules?
* **Output Format:** What does success look like?
* **Error Handling:** How does it handle unexpected situations?

---

## 3. General & System Prompt Frameworks

### A. CoSTAR Framework
* **Context:** Set the scene, background info
* **Objective:** What exactly should the AI achieve
* **Steps:** Break down the task (force reasoning)
* **Tools:** Any tools/data/formats AI should use
* **Actions:** Expected output style or format
* **Reflection:** Self-check or iteration loop

### B. Magic Prompt Framework
* **Role:** Who the AI should act as (e.g., senior copywriter, Python tutor, fitness coach).
* **Context:** Background details, audience, situation, or reference data the AI needs.
* **Task:** The exact action or deliverable you want.
* **Constraints & Format:** Rules like word count, tone, things to avoid, and layout (e.g., bullet points, table, and code block).
* **Examples (Optional):** Sample inputs or style examples to show the AI what "good" looks like.

### C. Special Prompting Structure for CustomGPTs (Markdown Prompting)
Use structured Markdown headers and bolding to clearly delineate system instructions for CustomGPTs:

```markdown
# **Role:**

# **Objective:**

# **Context:**

# **Instructions:**

## **Instruction 1 :**
## **Instruction 2 :**
## **Instruction 3 :**

# **Notes:**
* Note 1
* Note 2
* Note 3
```