---
name: create-product-roadmap
description: Creates or revalidates docs/specs/ROADMAP.md — outcome milestones (M#) on Now/Next/Later horizons, the walking-skeleton first, each with a slug and CAP-*/QA-* traces. Use to sequence, re-scope, or prioritise milestones, and to revalidate the plan after one ships. Not for product vision, requirements, or milestone planning.
---

# Create a product roadmap

Turn the durable product statement into a **living, outcome-based roadmap** of milestones,
and **revalidate** that roadmap after each milestone ships. The roadmap is fluid where the
product statement is stable; every item still traces to the North Star.

**Output:** `docs/specs/ROADMAP.md` — milestones `M#` with measurable outcomes, horizons,
traces, dependencies, and a dated revalidation log.

## Position in the stack

| Layer | Skill | Artifact |
|-------|-------|----------|
| Product | `create-product-statement` | `docs/specs/PRODUCT_STATEMENT.md` (`CAP-*`, `QA-*`) — stable |
| **Roadmap** | **this skill** | `docs/specs/ROADMAP.md` (`M#`) — living, revalidated per milestone |
| Milestone input | `create-initreq-for-sdd` | `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) |
| Layers 1–4 | `spec-driven-development` | `milestones/<slug>/…` → closeout → back here |

**Upstream:** `PRODUCT_STATEMENT.md`. If it is missing or thin, run
`create-product-statement` first. **Downstream:** each milestone is the input to
`create-initreq-for-sdd`, then `spec-driven-development` implements it slice by slice.
**Loop:** when a milestone is delivered, `spec-driven-development` hands control back here
with its `<MILESTONE_ID>_CLOSEOUT.md`.

A milestone advances one or more `CAP-*`/`QA-*`, and keeps those ids in its traceability so
the spine `CAP-*/QA-* → M# → REQ-* → delivery story → engineering task → evidence` stays
unbroken. Full stack diagram and shared glossary: `docs/sdd-skills-guide.md`.

Terms specific to this skill: a **milestone** is a deployable outcome slice of the product
with a measurable target and a reusable **`milestone-slug`**. A **product walking
skeleton** is the first roadmap milestone — a deployable product-thin path that exercises
the whole architecture once. Inside each milestone, `spec-driven-development` still starts
with a smaller **slice tracer**.

## Principles

1. **Outcomes, not outputs.** Each milestone states a measurable result, not "build
   feature F".
2. **Trace to the North Star.** Every milestone cites the `CAP-*`/`QA-*` it advances. No
   orphans.
3. **Now / Next / Later, no false precision.** Avoid date-stamped long-range commitments;
   *Later* items are explicitly hypotheses.
4. **Every milestone is deployable and independently valuable** — and within it, every
   slice ships a working chunk.
5. **First milestone is the product walking skeleton.** Prove the architecture and the
   riskiest assumptions end-to-end before broadening, and create the first useful
   current-state docs if they do not exist.
6. **Sequence by value × risk × dependency.** Pull high-value, high-risk,
   assumption-validating work earlier; respect hard dependencies; express dependencies as
   outcomes.
7. **Limit work in progress.** Keep a small number of active *Now* outcomes (roughly three
   to five at portfolio scale) so focus is real.
8. **Living document.** Review the near term often and the long range at strategic
   checkpoints.

## Workflow

**1. Ingest the product statement.** Read `docs/specs/PRODUCT_STATEMENT.md` and list the
`CAP-*` and `QA-*`. If it is absent or thin, pause and run `create-product-statement`.

**2. Draft the roadmap.** Group capabilities into optional strategic themes, then define
milestones as outcomes. Assign each to Now / Next / Later, give each a `milestone-slug`,
flag the first as the product walking skeleton, set a measurable target, and record
traces, dependencies, and what ships. Structure: `references/document-structure.md`.

**3. Validate and sequence.** Confirm that every *Now* milestone is deployable,
independently valuable, and small enough for a few slices; that dependencies are sane;
that ordering is risk- and value-first; that the first milestone is a walking skeleton; and
that every outcome has a measure. Run the critique pass in
`references/revalidation.md`, then ask the user:

> Is this sequence right? Which milestone outcomes, measures, or dependencies are wrong,
> missing, or mis-prioritized before we open the first milestone?

**4. Hand off the current milestone.** For the *Now* milestone, invoke
`create-initreq-for-sdd` with its outcome, measure, `CAP-*`/`QA-*` traces,
`milestone-slug`, dependencies, what ships, and whether `ARCHITECTURE_OVERVIEW.md` /
`TECH_STACK.md` must be created or updated. Do not author requirements or planning here.

**5. Revalidate after each milestone ships.** Full procedure, critique pass, and checklist:
`references/revalidation.md`. In short: read the closeout, record the outcome actually
achieved, re-sequence what remains, refresh horizons, confirm the current-state docs,
append a dated revalidation log entry, escalate a broken core assumption to
`create-product-statement`, and hand the new *Now* milestone to `create-initreq-for-sdd`.

## Completion criteria

- `docs/specs/ROADMAP.md` exists with all six sections and a **last revalidated** marker.
- Every milestone row has a measurable outcome, a horizon, non-empty `CAP-*`/`QA-*` traces,
  a `milestone-slug`, and a deployable "what ships".
- The first milestone is flagged as the product walking skeleton and includes creating the
  current-state docs when they are missing.
- *Now* and *Next* milestones have per-milestone detail including the riskiest assumption
  each validates.
- After a delivery: the outcome verdict, the revalidation log entry, and the promotion of
  the next milestone are all recorded, and the handoff is stated.
- `specs_check.sh` runs clean.

## Gotchas in this repo

- `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` wraps the spec
  linters.
- The `milestone-slug` becomes a directory name and a filename fragment
  (`DETAILED_PLANNING_<MILESTONE_SLUG>.md`), so renaming it later means renaming a whole
  tree. Pick it once.
- Phases 1, 2, 3 and 3.1 of the density-matrix track were delivered before this roadmap
  existed, under `docs/density_matrix_project/archive/phases/`. Record them in the roadmap
  as **Delivered** milestones with a one-line outcome and a link to the archived phase
  directory — do not re-plan them and do not create `milestones/<slug>/` trees for them.
  The first *Now* milestone is the product walking skeleton for the **new** convention.
- Milestone ids may carry a suffix (`M4`, `M4A`, …). Keep the id in the closeout filename
  as `<MILESTONE_ID>_CLOSEOUT.md` with `-` written as `_`.
- The archived `PLANNING.md` (§3 dependency order, §6 decision gates, §8 scope cuts) and
  `RESEARCH_ALIGNMENT.md` are the sequencing inputs; the PhD plan milestones they cite
  are outcomes to trace, not dates to promise.

## References

- `references/document-structure.md` — required sections, milestone table columns,
  per-milestone detail, size discipline. Read while drafting or re-sequencing.
- `references/revalidation.md` — inputs, steps, critique pass, checklist, escalation. Read
  when a milestone ships.

## Anti-patterns

- **Feature-list roadmap** — "build X" items with no measurable outcome.
- **Dates as promises** — long-range date-stamped commitments that become a plan to defend
  instead of a strategy to adapt.
- **Big-bang first milestone** — a broad first milestone instead of a thin end-to-end
  walking skeleton.
- **Orphan milestones** — work that traces to no `CAP-*`/`QA-*`.
- **Plan once, never revisit** — skipping revalidation, so learnings never reshape the plan.
- **Silent product drift** — changing strategy inside the roadmap instead of escalating a
  core-assumption break to the product statement.
- **Non-deployable milestones** — a milestone or slice that produces nothing shippable.
- **Revalidation as bookkeeping** — recording that a milestone shipped without naming what
  changed in the plan because of it.
