# Changelog — create-product-roadmap

Revision history lives here rather than in `SKILL.md`, where dated notes cost tokens on
every activation and go stale. Revisions are forward-only.

## rev D — imported into the SQUANDER density-matrix repository

Copied from the reference SDD stack unchanged in method; only the repo-specific parts
changed. `make specs-check` became
`bash .cursor/skills/spec-driven-development/scripts/specs_check.sh`; the "Gotchas in
this repo" section now describes this repository (the `qgd` conda lanes, the archived
Phases 1–3.1 under `docs/density_matrix_project/archive/`, the sequential `NoisyCircuit`
baseline) instead of the reference project's infrastructure.

## rev C

Restructured for progressive disclosure and verification.

- Body trimmed to a router under the 250-line budget; the `ROADMAP.md` section structure
  moved to `references/document-structure.md` and the revalidation loop to
  `references/revalidation.md`, loaded only when needed.
- Description rewritten as a routing key with an explicit boundary against
  `create-product-statement` and `create-initreq-for-sdd`.
- Added an adversarial critique pass before publishing a drafted or revalidated roadmap,
  matching the pass `create-initreq-for-sdd` already ran.
- Added completion criteria, repo-specific gotchas (non-numeric milestone ids, the cost of
  renaming a slug, the relocated trees), and `make specs-check` as the verification step.
- Replaced the duplicated stack diagram with a position table plus a pointer to the one
  canonical copy in `docs/sdd-skills-guide.md`.
- Added explicit reading discipline for revalidation: read the milestone closeout, not the
  milestone's slice-level task files.

## rev B

Phase 5 revalidation began consuming `spec-driven-development`'s
`<MILESTONE_ID>_CLOSEOUT.md` as its input, and confirming any `CHANGE_CONTROL.md` sign-off
status before marking a milestone Delivered.
