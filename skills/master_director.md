---
id: master_director
name: Master Director Framework (Image & Video)
category: Image & Video Generation
description: Cinematography-first prompt structure for diffusion and video models using camera, lens, lighting, and film stock direction.
best_for: Photorealistic images, AI video generation (Sora, Runway, Veo, Midjourney), cinematic scenes, commercial product shots, and director-style visual framing.
trigger_keywords: [image, video, photo, scene, camera, lens, lighting, cinematic, movie, shot, render, midjourney, sora, diffusion, visual]
---

# Skill: Master Director Framework

## Core Principle
Diffusion and video generation models understand photography, cinematography, and visual staging—not abstract logic. Translate the user's meta-prompt into concrete visual, spatial, and optical terms.

## Mandatory Output Formula
Generate a single, rich, comma-separated cinematic prompt following this exact sequence:
`[Subject], [Action], [Environment], [Atmosphere], [Camera], [Lens], [Lighting], [Style Reference]`

## Cinematographer & Director Vocabulary Reference
Use precise industry terminology from these categories where appropriate:
- **Camera Angles & Framing:** Wide Shot, Medium Shot, Close-Up Shot, Extreme Close-Up, Cowboy Shot (mid-thigh up), Low Angle (Power), High Angle, Dutch Angle (tilted), Overhead / Flat Lay, Point-of-View (POV), Handheld, Dolly Zoom, Slow Motion (120fps), Drone / Aerial View, Gimbal Tracking Shot, Rack Focus, Underwater Shot.
- **Lenses:** Anamorphic Lens (cinematic widescreen), Macro Lens, 85mm Lens (portraits/products), 50mm Lens, 35mm Lens (natural field of view), Fish-eye Lens, Shallow Depth of Field.
- **Lighting:** Golden Hour, Blue Hour, Volumetric Lighting ("god rays"), Chiaroscuro (high-contrast shadows), Rim / Edge Lighting, Bioluminescent Glow, Studio Softbox Lighting, Three-Point Lighting, Neon Reflections, Backlighting, Candlelight, Hard Light, Ambient Light, Screen Glow.
- **Style & Film Stock References:** Photorealistic, Cinematic, Shot on Kodak Portra 400, Shot on Fujifilm Velvia, Cinestill 800T, IMAX 70mm Film, Black and White Noir, Documentary Style, Luxury Brand Aesthetic, Apple Commercial Style, Unreal Engine 5 Render, 1980s Retro Film Look.

## Strict Execution Rules
1. Expand vague user ideas into vivid visual specifics (textures, materials, weather, lens focal length, lighting direction).
2. Follow the exact 8-part sequence (`Subject` -> `Action` -> `Environment` -> `Atmosphere` -> `Camera` -> `Lens` -> `Lighting` -> `Style Reference`).
3. Output **ONLY** the final prompt text. Do not include labels like `[Subject]:`, introductory text, or explanations.

## Example Output
A weathered deep-sea diver in a brass-and-canvas retro suit, slowly inspecting a glowing bioluminescent shipwreck, inside an abyssal underwater canyon filled with drifting silt and deep indigo currents, eerie and mysterious deep-ocean atmosphere, Low Angle Wide Shot, Anamorphic Lens with shallow depth of field, volumetric shafts of cold moonlight piercing from above mixed with warm amber helmet lamp rim lighting, cinematic IMAX 70mm film look, photorealistic.
