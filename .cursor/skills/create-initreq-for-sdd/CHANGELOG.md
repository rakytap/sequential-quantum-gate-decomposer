# Changelog — create-initreq-for-sdd

Revision history lives here rather than in `SKILL.md`, where dated notes cost tokens on
every activation and go stale. Revisions are forward-only.

## rev E — imported into the SQUANDER density-matrix repository

Copied from the reference SDD stack unchanged in method; only the repo-specific parts
changed. `make specs-check` became
`bash .cursor/skills/spec-driven-development/scripts/specs_check.sh`; the "Gotchas in
this repo" section now describes this repository (the `qgd` conda lanes, the archived
Phases 1–3.1 under `docs/density_matrix_project/archive/`, the sequential `NoisyCircuit`
baseline) instead of the reference project's infrastructure.

## rev D

Restructured for progressive disclosure and verification.

- Body trimmed to a router under the 250-line budget; the seven-section document structure
  moved to `references/document-structure.md`, and the critique pass plus the update path to
  `references/critique-and-update.md`.
- Description rewritten as a routing key, with an explicit boundary against
  `create-product-roadmap` upstream and `spec-driven-development` downstream — the most
  likely mis-routing in the stack, since "spec the milestone" plausibly matches all three.
- Added completion criteria, a verification step (`make specs-check`), and repo-specific
  gotchas: `uv`-only Python, non-numeric milestone ids, the hermetic versus opt-in test
  lanes, and the milestones that live outside `milestones/`.
- Added the requirement that every `REQ-*` carry at least one negative or error scenario,
  and that each requirement's evidence route be nameable before freeze.
- Replaced the duplicated stack diagram with a position table plus a pointer to the one
  canonical copy in `docs/sdd-skills-guide.md`.

## rev C

Added the Phase 2b adversarial critique pass before the validation checkpoint: attack
assumptions, acceptance, NFR response measures, traceability, and scope, then tighten or
record every finding rather than leaving it implicit.

## rev B

Added an explicit update/evolve path so the skill is bidirectional — run it on an existing
`INITIAL_REQUIREMENTS.md` to evolve it, preserving stable `REQ-*` ids and appending a Change
log entry — rather than greenfield creation only.
