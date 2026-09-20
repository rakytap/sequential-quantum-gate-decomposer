# SDD artifact map — canonical paths, naming, and size budgets

Read when creating a milestone tree, naming a new artifact, or deciding where
something belongs. `docs/specs/` is the only spec root; never introduce a second one.

## Contents

- Program-level artifacts
- Milestone-level artifacts (Layer 1)
- Per-slice artifacts (Layers 2–4)
- Naming rules
- Relocated and legacy trees
- Size budgets
- Current-state architecture docs
- Source-of-truth reading order

## Program-level artifacts

| Path | Owner skill | Purpose |
|------|-------------|---------|
| `docs/specs/PRODUCT_STATEMENT.md` | `create-product-statement` | North Star: `CAP-*`, `QA-*`, guardrails, riskiest assumptions |
| `docs/specs/ROADMAP.md` | `create-product-roadmap` | Outcome milestones `M#`, Now/Next/Later, revalidation log |
| `docs/specs/ARCHITECTURE_OVERVIEW.md` | `spec-driven-development` | Current-state architecture (descriptive) |
| `docs/specs/TECH_STACK.md` | `spec-driven-development` | Current-state stack and commands (descriptive) |
| `docs/specs/.sdd-lint.json` | `spec-driven-development` | Linter budgets, milestone-slug overrides, waivers |

## Milestone-level artifacts (Layer 1)

All under `docs/specs/milestones/<milestone-slug>/`:

| File | Purpose |
|------|---------|
| `INITIAL_REQUIREMENTS.md` | `REQ-*` baseline from `create-initreq-for-sdd` (upstream input) |
| `DETAILED_PLANNING_<MILESTONE_SLUG>.md` | Scope, goals, frozen contracts, acceptance, traceability |
| `ADRS_<MILESTONE_SLUG>.md` | Decisions affecting more than one work package |
| `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | Gap list and the ready / not-ready verdict |
| `<MILESTONE_ID>_CLOSEOUT.md` | Delivery record; the input to roadmap revalidation |
| `CHANGE_CONTROL.md` | Only when a deviation needs formal governance sign-off |

## Per-slice artifacts (Layers 2–4)

All under `docs/specs/milestones/<milestone-slug>/task-<n>/`:

| File | Layer | Purpose |
|------|-------|---------|
| `TASK_<n>_MINI_SPEC.md` | 2 | Work-package contract: required/unsupported behavior, evidence matrix, interfaces |
| `DELIVERY_STORIES.md` | 3 | Behavioral slices with acceptance signals and traceability |
| `ENGINEERING_TASKS.md` | 4 | Red-first implementation tasks with done criteria |
| `CLOSEOUT.md` | — | Slice verdict: `shipped` or `implementation handback` |
| `STEP_4A_HANDBACK.md` | — | Questions back to planning when Step 4b hits a contract/ADR gap |

## Naming rules

- `<milestone-slug>` is short and filesystem-safe, and is reused verbatim by every
  downstream artifact. Take it from `ROADMAP.md`; do not invent a variant.
- `<MILESTONE_SLUG>` is the uppercase-underscore form of the directory name:
  directory `milestones/db-schema-migrations` gives
  `DETAILED_PLANNING_DB_SCHEMA_MIGRATIONS.md` and `ADRS_DB_SCHEMA_MIGRATIONS.md`.
- `<MILESTONE_ID>` is the roadmap id (`M4`, `M4A`, `M5`, …) with `-` rendered as `_`:
  `M4_CLOSEOUT.md`.
- Sub-milestones get their own directory with the same artifact set and a consistent
  slug, so traceability stays obvious.
- Reference every path from the repo root.

`check_artifacts.py` enforces these rules. When a directory name legitimately differs
from its roadmap slug, declare the mapping in `milestone_slugs` in
`docs/specs/.sdd-lint.json` rather than letting the check drift.

## Relocated and legacy trees

The canonical layout above describes where *new* work goes, not where all history lives.
Phases 1, 2, 3 and 3.1 of the density-matrix track were delivered under an earlier,
phase-based convention and are archived **unchanged** and **read-only**:

| Tree | State |
|------|-------|
| `docs/specs/milestones/<slug>/` | Canonical location for planned and in-flight milestones |
| `docs/density_matrix_project/archive/phases/phase-<n>/` | Delivered Phases 1, 2, 3, 3.1 — Layer 1 contracts, mini-specs, stories, implementation plans, evidence reviews, and per-phase paper surfaces |
| `docs/density_matrix_project/archive/planning/` | Program-level `PLANNING.md`, `ADRs.md` (ADR-001 … ADR-008), `REFERENCES.md` — the long-horizon rationale the product statement and roadmap lift from |

The legacy convention maps onto the current one as follows; do **not** retro-fit the
archive and do not carry the old names into `docs/specs/`:

| Legacy (archive) | Current (`docs/specs/`) |
|------------------|-------------------------|
| `phases/phase-<n>/` | `milestones/<slug>/` |
| `DETAILED_PLANNING_PHASE_<N>.md`, `ADRs_PHASE_<N>.md`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | `DETAILED_PLANNING_<MILESTONE_SLUG>.md`, `ADRS_<MILESTONE_SLUG>.md`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` |
| (none — intent lived in `planning/PLANNING.md`) | `INITIAL_REQUIREMENTS.md` (`REQ-*`) |
| `task-<n>/TASK_<n>_MINI_SPEC.md` | `task-<n>/TASK_<n>_MINI_SPEC.md` (same) |
| `task-<n>/TASK_<n>_STORIES.md`, `<NTH>_VERTICAL_SLICE_..._STORIES_AND_ENGINEERING_TASKS.md` | `task-<n>/DELIVERY_STORIES.md` |
| `task-<n>/STORY_<k>_IMPLEMENTATION_PLAN.md`, `ENGINEERING_TASK_*_IMPLEMENTATION_PLAN.md` | `task-<n>/ENGINEERING_TASKS.md` |
| `CLOSURE_PLAN_PHASE_<N>.md`, `PRE_PUBLICATION_EVIDENCE_REVIEW_*.md` | `task-<n>/CLOSEOUT.md`, `<MILESTONE_ID>_CLOSEOUT.md` |
| `SHORT_PAPER_*`, `SHORT_PAPER_NARRATIVE.md`, `ABSTRACT_*`, `PAPER_*`, `PUBLICATIONS.md` | not spec artifacts — papers consume a closeout's evidence matrix |
| `API_REFERENCE_PHASE_<N>.md` | linked from `ARCHITECTURE_OVERVIEW.md`; refresh outside `docs/specs/` when the API changes |

Revisions are forward-only: plan new slices under the current convention and leave
delivered phases as they are. Paths of the form `docs/density_matrix_project/phases/…` or
`…/planning/…` quoted inside the archived files refer to their pre-archive location.

## Size budgets

An artifact that no longer fits in working memory stops being read and starts being
skimmed. Budgets are in lines and are enforced as warnings by `check_artifacts.py`:

| Artifact | Budget |
|----------|--------|
| `INITIAL_REQUIREMENTS.md` | 300 |
| `DETAILED_PLANNING_*` | 400 |
| `ADRS_*` | 400 |
| `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | 250 |
| `TASK_<n>_MINI_SPEC.md` | 250 |
| `DELIVERY_STORIES.md` | 200 |
| `ENGINEERING_TASKS.md` | 300 |
| `CLOSEOUT.md` (slice) | 200 |
| `<MILESTONE_ID>_CLOSEOUT.md` | 250 |
| `STEP_4A_HANDBACK.md` | 200 |
| `CHANGE_CONTROL.md` | 200 |

When a slice exceeds its budget the slice is too big: split the slice, or move the
long evidence tables into the slice's `ENGINEERING_TASKS.md` and keep the mini-spec
to the contract. Do not append.

Every artifact opens with a context header of at most ten lines — status, milestone
or slice, scope, and what it traces to — so a partial read is still decision-useful.

## Current-state architecture docs

`ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` describe **what is true now**. They are
not product intent and not a proposal.

`ARCHITECTURE_OVERVIEW.md` carries C4 context/container/component views at the level
useful for the repo, bounded contexts and domain ownership, ports/adapters and
anti-corruption layers, domain events, major data flows, external systems and
integration contracts, deployment and runtime topology, current architecture risks and
constraints, and links to the ADRs that made each decision.

`TECH_STACK.md` carries languages, frameworks, runtimes, package managers and build
tools; the build, test, lint, typecheck, run, migration and deployment commands;
databases, queues, storage, auth, observability, feature flags, CI/CD, infrastructure
and version constraints; and local development setup with the tool conventions an
agent must know before changing code.

**Creation:** if missing, create them during the first product-walking-skeleton
milestone, or earlier when the repo already holds enough current-state information.
**Update:** at milestone close, whenever the milestone changed architecture, stack,
commands, runtime topology, integrations, data ownership, operational behavior, or ADR
status. Link to ADRs instead of duplicating rationale.

## Source-of-truth reading order

Top-down, for milestone or program work:

1. `PRODUCT_STATEMENT.md` — `CAP-*`, `QA-*`
2. `ROADMAP.md` — the milestone `M#` and its outcome
3. `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md` — current-state constraints
4. `milestones/<slug>/INITIAL_REQUIREMENTS.md` — `REQ-*`
5. `milestones/<slug>/DETAILED_PLANNING_*`, `ADRS_*`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
6. `milestones/<slug>/task-<n>/` — only the current slice
