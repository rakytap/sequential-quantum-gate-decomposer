# Phase 3.1 - Sixth Vertical Slice (Pre-Publication Evidence Review)

This document is the **implementation planning slice** for the final
pre-publication closure increment after the full counted performance matrix is
emitted. It keeps the Layer 2 mini-specs authoritative and defines the
smallest Task 4/5-adjacent expansion needed to make the publication-closure
decision mechanical rather than interpretive.

**Do not** treat this as Task 5 publication closure. This slice exists to
produce the formal review artifact that decides whether Task 5 may proceed and,
if so, under which closure mode.

## Source mini-specs (Layer 2)

| Task | Document |
|------|----------|
| 4 | [`task-4/TASK_4_MINI_SPEC.md`](task-4/TASK_4_MINI_SPEC.md) |
| 5 | [`task-5/TASK_5_MINI_SPEC.md`](task-5/TASK_5_MINI_SPEC.md) |

## Slice boundary

**In**

- The current Phase 3.1 bounded correctness and external-reference package:
  - 4 strict counted microcases,
  - 2 hybrid counted continuity anchors,
  - required bounded 5-row external slice.
- The full counted performance matrix from `P31-S11`.
- The matrix-wide `break_even_table` / `justification_map`.
- One formal review artifact:
  - `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`.
- One explicit closure-state decision:
  - `positive-methods-ready`,
  - `decision-study-ready`,
  - or `not-ready-yet`.

**Out**

- The actual Task 5 rewrite of the publication surfaces.
- Venue-specific manuscript shaping.
- Program-level doc sync after publication closure.
- Any Task 6 host-acceleration work.
- Any broadening of the frozen support surface or performance inventory.

## Slice dependency

This slice should start **only after** the immediate predecessor slice records:

- the full 26-row counted performance matrix,
- route-aware summaries on the counted rows,
- and a matrix-wide decision artifact.

If those are missing, the correct action is to return to
`FIFTH_VERTICAL_SLICE_FULL_COUNTED_PERFORMANCE_MATRIX_STORIES_AND_ENGINEERING_TASKS.md`,
not to weaken the review standard.

---

### Story P31-S12: A pre-closure evidence summary makes the publication decision mechanical

**User/Research value**

- Prevents ad hoc interpretation after the bundles land.
- Keeps Task 5 publication closure evidence-driven rather than aspiration-driven.
- Forces the phase to say explicitly whether it is:
  - positive-methods-ready,
  - decision-study-ready,
  - or not-ready-yet.

**Given / When / Then**

- **Given** the Task 3 bounded correctness package and the full Task 4
  performance package exist for the frozen counted slice.
- **When** the Phase 3.1 pre-publication evidence review runs.
- **Then** the phase records one of the three allowed closure states, with
  explicit blockers and no manuscript rewriting yet attached.

**Scope**

- **In:** one concise review artifact, closure-state rubric, artifact-indexed
  and case-ID-indexed summaries, and explicit blockers if closure cannot yet be
  upgraded.
- **Out:** technical short-paper rewrites, abstract tightening, narrative
  reframing, or venue selection.

**Acceptance signals**

- One review table maps:
  - correctness coverage,
  - external-reference coverage,
  - performance-matrix coverage,
  - and open blockers.
- The review records exactly one closure state:
  - `positive-methods-ready`,
  - `decision-study-ready`,
  - `not-ready-yet`.
- The review states explicitly that Task 5 remains downstream of this slice.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §6, §8 Task 5, §9, §10, §13
- ADRs: `P31-ADR-005`, `P31-ADR-006`, `P31-ADR-013`

#### Engineering tasks (Story P31-S12)

##### Engineering Task P31-S12-E01: Produce the pre-publication evidence review pack and go/no-go summary

**Implementation plan:** [`task-5/ENGINEERING_TASK_P31_S12_E01_IMPLEMENTATION_PLAN.md`](task-5/ENGINEERING_TASK_P31_S12_E01_IMPLEMENTATION_PLAN.md)

**Implements story**

- Story P31-S12

**Change type**

- docs | validation automation

**Definition of done**

- `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md` exists and is reviewable.
- The review records one of the three allowed closure states.
- The review is evidence-indexed and does not yet rewrite the paper surfaces.

**Execution checklist**

- Summarize correctness-slice coverage from the current bounded package.
- Summarize performance-matrix coverage and matrix-wide decision artifacts.
- Record the current closure state and blockers.
- State explicitly whether Task 5 may proceed.

**Evidence produced**

- One reviewer-facing pre-publication evidence review artifact.
- Explicit go / no-go input for the later Task 5 manuscript update.

**Risks / rollback**

- Risk: the review drifts into argument rather than evidence summary.
- Rollback/mitigation: keep the artifact tied to emitted bundles, stable case
  IDs, and the closure-state rubric.

---

## Story-to-engineering map

| Story   | Engineering tasks |
| ------- | ----------------- |
| P31-S12 | P31-S12-E01       |

## Suggested implementation order

1. `P31-S12-E01`

Parallelism:

- None recommended. The review should be written after the counted performance
  matrix and decision artifact are stable.

---

## Slice implementation status

Update this section when the sixth-slice engineering task lands.

| Field                                 | Value                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Correctness package required          | bounded 6-row counted Phase 3.1 slice plus required 5-row external subset                      |
| Performance package required          | full 26-row counted matrix plus matrix-wide decision artifact                                   |
| Sixth vertical slice implementation   | **Planned**                                                                                     |
| Immediate successor after this slice  | Task 5 publication closure                                                                      |
| Deferred beyond this slice            | Task 5 surface rewrites, top-level program sync, optional Task 6 host acceleration             |
