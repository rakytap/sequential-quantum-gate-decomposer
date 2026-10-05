# Changelog — spec-driven-development

Revision history lives here rather than in `SKILL.md`: dated notes and "effective from"
caveats are time-sensitive content that costs tokens on every activation and goes stale.
Revisions are **forward-only** — a slice keeps the convention it shipped under.

## rev E — ADR companion file and skill-rule maintenance (P0b)

- **`ADR_AMENDMENTS_<SLUG>.md`**: optional single continuation beside `ADRS_<SLUG>.md`
  (400-line budget, slug check, context header). Forbidden at milestone root: any other
  `ADR*.md` (including `ADRS_<SLUG>_2`, `ADRS2_*`, `ADR_AMENDMENTS2_*`, `ADRs_*`,
  `ADR_ADDENDUM_*`, bare `ADR_AMENDMENTS.md`, or a second companion).
- **`check_artifacts.py`**: allows only `ADRS_<SLUG>.md` and `ADR_AMENDMENTS_<SLUG>.md`;
  `ADR_FORBIDDEN_CONTINUATION` and `SLUG_MISMATCH` for violations. **`_sddlint.py`**: same
  ADR budget for amendments.
- **`artifact-map.md`**, **`docs/sdd-skills-guide.md`**, **`.cursor/rules/spec-driven-docs-specs.mdc`**, and the SDD skill size-budget section document the companion.
- **Skill bodies**: moved two-commit close detail to `references/two-commit-close.md`
  (step (g) follows slice ADRs/mini-spec, not a hard-wired option (i)); moved reading order,
  repo gotchas, and planning/code seam to `references/`; **removed** milestone-review model
  pins from `SKILL.md` (not relocated into `two-commit-close.md`); removed "(new guidance)"
  markers from the skill tree.
- **`references/repo-gotchas.md`**: density-track gotchas moved verbatim from P0 `SKILL.md`.

**Negative controls** (throwaway `--root`, normal and `--strict`):

| NC | Setup | Result |
|----|-------|--------|
| NC-0 | Unmodified exactness-reconfirmation tree | 0 errors, 1 warn `SLICE_MISSING_CLOSEOUT` (task-2); strict: same finding only; companion accepted |
| NC-1 | Companion padded to 401 lines | `SIZE_BUDGET` |
| NC-2 | Companion exactly 400 lines | no `SIZE_BUDGET` on companion |
| NC-3 | `ADR_AMENDMENTS_WRONG_SLUG.md` | `SLUG_MISMATCH` |
| NC-4a | `ADRS_EXACTNESS_RECONFIRMATION_2.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-4b | `ADRS2_EXACTNESS_RECONFIRMATION.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-4c | Second `ADR_AMENDMENTS_*` (wrong slug) | `ADR_FORBIDDEN_CONTINUATION` (+ `SLUG_MISMATCH` when slug wrong) |
| NC-4d | `ADR_AMENDMENTS2_EXACTNESS_RECONFIRMATION.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-4e | `ADRs_EXACTNESS_RECONFIRMATION.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-4f | bare `ADR_AMENDMENTS.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-4g | `ADR_ADDENDUM_EXACTNESS_RECONFIRMATION.md` | `ADR_FORBIDDEN_CONTINUATION` |
| NC-5 | Companion context header stripped | `MISSING_CONTEXT_HEADER` |
| NC-6 | `ADRS_*` padded to 401 lines | `SIZE_BUDGET` |
| NC-7 | Only `exactness-reconfirmation` under `docs/specs/milestones/` on the real tree | vacuous (no second milestone to scan) |
| NC-7b | Throwaway tree: copy `exactness-reconfirmation` → `synthetic-no-companion`, delete `ADR_AMENDMENTS_*`, rename slugged Layer 1 files/headers, drop `task-2`; lint full `--root` tree | normal/strict: same as NC-0 (`SLICE_MISSING_CLOSEOUT` only); **no** `ADR_FORBIDDEN_CONTINUATION`, `SLUG_MISMATCH`, or `SIZE_BUDGET` on the companion-less milestone |

**KNOWN GAP:** `check_traceability.py` never resolves ADR ids (only REQ/CAP/QA/milestone at
lines 43–46); dangling `ADR-F1A-NNN` references are not detected — deferred by Tech Lead.

## rev D — slice-close practice from the M-F1a q4 tracer

Folded the M-F1a q4 tracer lessons into the skill as default practice, citing that
milestone's ADRs as precedent: the two-commit slice close for clean-start evidence
(ADR-F1A-009), the code-ready versus slice-close gate split (ADR-F1A-008 decision 1),
whole-worktree clean-start evidence with dirty non-counted runs parked outside the checkout
(ADR-F1A-004), planning-header sync before the pre-C1 Reviewer pass, the G-10 shared-kernel
limitation, a full Reviewer milestone review at milestone close, and path-exact commits.
No linter or script changed. Slices already shipped keep the convention they shipped under.

## rev C — requirements-baseline stage in the artifact linter

A milestone directory holding only `INITIAL_REQUIREMENTS.md` used to report three
`L1_MISSING_ARTIFACT` errors, so `create-initreq-for-sdd` could never meet its own
"`specs_check.sh` runs clean" criterion, and a waiver was not an option because the
milestone is in flight by definition. `check_artifacts.py` now reports that stage as one
`L1_NOT_STARTED` info finding. As soon as any Layer 1 file, `task-<n>` slice, milestone
closeout, or `CHANGE_CONTROL.md` exists, each missing Layer 1 file is an error again, so a
partial Layer 1 or a slice without Layer 1 still fails. The rule was exercised against
throwaway trees for each of those states before adoption. No milestone had delivered under
the new convention, so nothing shipped under the old rule.

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
