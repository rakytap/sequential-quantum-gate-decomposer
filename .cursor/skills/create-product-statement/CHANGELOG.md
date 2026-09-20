# Changelog — create-product-statement

Revision history lives here rather than in `SKILL.md`, where dated notes cost tokens on
every activation and go stale. Revisions are forward-only.

## rev C — imported into the SQUANDER density-matrix repository

Copied from the reference SDD stack unchanged in method; only the repo-specific parts
changed. `make specs-check` became
`bash .cursor/skills/spec-driven-development/scripts/specs_check.sh`; the "Gotchas in
this repo" section now describes this repository (the `qgd` conda lanes, the archived
Phases 1–3.1 under `docs/density_matrix_project/archive/`, the sequential `NoisyCircuit`
baseline) instead of the reference project's infrastructure.

## rev B

Restructured for progressive disclosure and verification.

- Body trimmed to a router under the 250-line budget; the nine-section document structure
  and change policy moved to `references/document-structure.md`, and the PR-FAQ, Lean
  Canvas, assumption/risk map, critique pass, and stakeholder checkpoint to
  `references/validation-lenses.md`.
- Description rewritten as a routing key, leading with the distinctive test this artifact
  exists to pass ("decides what not to build") and an explicit boundary against
  `create-product-roadmap` and `create-initreq-for-sdd`.
- Added an explicit adversarial critique pass, matching the pass
  `create-initreq-for-sdd` already ran.
- Added completion criteria, `make specs-check` as the verification step, and repo-specific
  gotchas — most importantly that `CAP-*`/`QA-*` ids are cited across `docs/specs/` and
  `tests/fitness/`, so they may be extended but never renumbered.
- Replaced the duplicated stack diagram with a position table plus a pointer to the one
  canonical copy in `docs/sdd-skills-guide.md`.
- Dropped the persona preamble in favour of a stated output contract.

## rev A

Initial skill: the durable product statement as the stable top of the SDD stack — vision,
customers and problem, value proposition, `CAP-*` capabilities, `QA-*` quality attributes,
ubiquitous-language seed, guardrails, and riskiest assumptions, validated with a PR-FAQ or
Lean Canvas and handed off to `create-product-roadmap`.
