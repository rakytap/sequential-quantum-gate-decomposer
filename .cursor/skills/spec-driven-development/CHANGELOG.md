# Changelog — spec-driven-development

Revision history lives here rather than in `SKILL.md`: dated notes and "effective from"
caveats are time-sensitive content that costs tokens on every activation and goes stale.
Revisions are **forward-only** — a slice keeps the convention it shipped under.

## rev B — adoption of the four-skill SDD stack

Replaced the phase-based, publication-coupled skill (rev A) with the layered milestone
workflow used by the `cursor-stuff/sdd` reference stack, adapted to this repository.

- **Spec root moved** from `docs/density_matrix_project/phases/<phase>/` to `docs/specs/`
  with `milestones/<slug>/` trees. The delivered phase trees and program-level planning
  (`PLANNING.md`, `ADRs.md`, `REFERENCES.md`) were archived, unchanged, under
  `docs/density_matrix_project/archive/` and are read-only history.
- **Upstream layers added.** `create-product-statement` (`CAP-*`, `QA-*`),
  `create-product-roadmap` (`M#`, Now/Next/Later), and `create-initreq-for-sdd` (`REQ-*`)
  now feed Layer 1; the traceability spine
  `CAP-*/QA-* → M# → REQ-* → delivery story → engineering task → evidence` replaces the
  informal "requirement → decision → task → evidence" wording.
- **Layer 3/4 renamed** to `task-<n>/DELIVERY_STORIES.md` and `task-<n>/ENGINEERING_TASKS.md`
  (one file per slice, red-first TDD template). Rev A used `TASK_<n>_STORIES.md` and one
  `STORY_<n>_IMPLEMENTATION_PLAN.md` / `ENGINEERING_TASK_*_IMPLEMENTATION_PLAN.md` per story.
- **Named close and governance artifacts**: `task-<n>/CLOSEOUT.md`,
  `task-<n>/STEP_4A_HANDBACK.md`, `<MILESTONE_ID>_CLOSEOUT.md`, `CHANGE_CONTROL.md`.
  Rev A had an ad-hoc `CLOSURE_PLAN_*` per phase.
- **Publication surfaces decoupled.** `SHORT_PAPER_*`, `SHORT_PAPER_NARRATIVE.md`,
  `ABSTRACT_*`, `PAPER_*` and `PUBLICATIONS.md` are no longer produced or gated by this
  skill. A paper consumes a milestone closeout's evidence matrix; it is not a spec artifact.
- **Progressive disclosure**: the body is a router under 250 lines; templates, practice
  catalogues, rubrics and the artifact map live in `references/`, loaded per step.
- **Mechanical verification**: `scripts/check_artifacts.py`, `scripts/check_traceability.py`
  and the `specs_check.sh` wrapper (conda `qgd` aware) plus `docs/specs/.sdd-lint.json`.
  `.cursor/hooks.json` audits spec edits and re-runs the linters at session stop.
- **Planning / code-generation seam** carried by `.cursor/agents/sdd-planner.md`
  (`readonly`), `sdd-implementer.md`, and `sdd-critic.md` (`readonly`).
- **Size budgets** and the just-in-time reading order added; rev A had none, and its
  Layer 1 files ran to 1,000–1,300 lines.
- Repo gotchas rewritten for this codebase: `qgd` conda lanes, the sequential
  `NoisyCircuit` exact baseline, the Qiskit Aer external reference, the benchmark evidence
  pipelines, and the frozen archive.

## rev A — phase-based skill (historical)

Scoped to `docs/density_matrix_project/` only. Three-step phase workflow (create phase
documents → readiness gap list → close checklist) producing `DETAILED_PLANNING_PHASE_X.md`,
`ADRs_PHASE_X.md`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, task mini-specs, and a
two-paper publication model per phase. Delivered Phases 1, 2, 3 and 3.1 under this
convention; their trees keep it.
