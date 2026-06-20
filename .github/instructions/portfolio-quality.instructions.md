---
description: "Use when editing Python, API, Streamlit, and documentation files in this repository. Enforce portfolio-grade data science project quality and clean architecture."
applyTo: "**/*.{py,md,txt}"
---

# Portfolio Quality Rules

- Keep model inference logic in reusable modules, not duplicated across UI/API layers.
- Avoid hardcoded absolute local paths; use project-relative paths via `pathlib`.
- Ensure all app entrypoints fail gracefully with actionable error messages.
- Prefer concise, professional documentation with setup, run, and test instructions.
- Remove dead files and redundant dependencies whenever they are no longer used.
