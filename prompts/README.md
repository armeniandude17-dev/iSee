# Prompt Pack

This folder contains example prompts you can load into iSee to turn it
into a domain-aware assistant. Without a custom prompt, Qwen 9B is a
general assistant glancing at your screen. **With** a domain-specific
prompt, the same model becomes a specialist who happens to have eyes
on your workspace.

The difference is genuinely large — see the README at the top of the
repo for a side-by-side example.

## How to load a prompt

1. Launch iSee.
2. In the footer, click the **Prompt:** dropdown → **Manage prompts...**
3. Click **+ New**.
4. Open one of the `.txt` files in this folder, copy the whole thing.
5. Paste into the Content field, give it a name (e.g. "DaVinci"), Save.
6. Pick it from the **Prompt:** dropdown to make it active.

The active prompt gets appended to every Qwen call until you switch
back to Default.

## What's here

- **`davinci_resolve_assistant.txt`** — a full DaVinci Resolve expert
  prompt covering the Edit, Fusion, Color, Fairlight, and Deliver
  pages. Knows node connections, common Fusion patterns (overlays,
  green screen, screen replacement, tracking, masking), export
  presets, troubleshooting checklists, and Resolve-specific
  terminology. Big prompt (~900 lines) — ships as a stress-test
  example of how detailed a domain pack can get.

## Writing your own

A few patterns that work well:

- **Open with a role definition.** "You are a [domain] assistant"
  followed by what the user is trying to accomplish in that domain.
- **List the things the assistant should know about** — UI areas,
  common tools, key terminology. Qwen pattern-matches these heavily.
- **Add a troubleshooting checklist.** "Common problems: ..." gives
  the model a place to start when the user describes a symptom.
- **Tell it what NOT to do.** "Don't pretend to see the screen unless
  a screenshot is attached." "Don't hallucinate menu paths." These
  guardrails matter for small models.
- **End with stylistic preferences.** "Give direct steps first,
  explain why if useful." "Avoid overcomplicating unless asked."

Prompts that work for you are worth sharing. Drop yours into this
folder via PR if you make something good.
