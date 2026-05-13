# AGENTS.md

This repository's project-specific agent instructions are maintained in
`CLAUDE.md`.

When working in this repository, Codex agents must read and follow
`CLAUDE.md` before planning or making non-trivial changes. Treat its project
overview, codebase reference, commands, architecture notes, constraints, and
working guidelines as the authoritative local instructions for this repo.

If `CLAUDE.md` conflicts with higher-priority system, developer, tool, or user
instructions, follow the higher-priority instruction and note the conflict when
it affects the requested work.

Keep this file small so `CLAUDE.md` remains the single source of truth. If the
Claude rules change, update `CLAUDE.md`; Codex will inherit them through this
bridge file.
