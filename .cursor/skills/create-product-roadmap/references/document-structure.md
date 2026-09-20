# `ROADMAP.md` — required structure

Read while drafting or re-sequencing the roadmap.

## Contents

- Section-by-section structure
- Milestone table columns
- Per-milestone detail
- Size discipline

## Section-by-section structure

**1. Summary & horizon** — a link to the vision, the current Now / Next / Later snapshot,
and a **last revalidated** marker (date plus the milestone that triggered it).

**2. Strategic themes** (optional) — two to four narratives grouping the capabilities they
serve.

**3. Milestone table** — one row per milestone, using the columns below. Flag the first
milestone as the **product walking skeleton**.

**4. Per-milestone detail** — at least for *Now* and *Next*. See below.

**5. Sequencing rationale** — why this order: value, risk, dependencies, and
walking-skeleton-first.

**6. Assumptions, risks & revalidation log** — current assumptions and risks, plus dated
entries recording *milestone delivered → what we learned → what changed in the roadmap*.

## Milestone table columns

| M# | slug | Outcome (measurable) | Horizon | Traces (CAP-*/QA-*) | Depends on | What ships (deployable) | Status |

- **Outcome** is a measurable result — "reduce X to Y", "enable Z for persona P" — never
  "build feature F".
- **slug** is short, filesystem-safe, and reused verbatim by every downstream artifact.
  `create-initreq-for-sdd` and `spec-driven-development` both key off it, so choosing it
  carelessly costs a rename across a whole milestone tree.
- **Traces** must be non-empty: a milestone with no `CAP-*`/`QA-*` is an orphan.
- **Depends on** expresses dependencies as outcomes ("M3's outcome depends on M1's
  outcome"), not as tasks.
- **What ships** names the deployable result. For the first milestone, include
  establishing `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` unless they already exist
  and are current.

## Per-milestone detail

For each *Now* and *Next* milestone:

- outcome, and why now;
- success measure / key result;
- in scope / out of scope;
- the **riskiest assumption it validates**;
- dependencies;
- the handoff `milestone-slug`;
- whether it creates or updates the current-state architecture and tech-stack docs.

## Size discipline

The roadmap is a living document that accumulates revalidation history, which is exactly
how it grows past the point of being read. Keep the milestone table and per-milestone
detail tight; when the revalidation log dominates the file, summarise older entries to one
line each and let the milestone closeouts under `docs/specs/milestones/` carry the detail.

*Later* items are deliberately low-precision. Writing them at *Now* fidelity is false
precision, not diligence.
