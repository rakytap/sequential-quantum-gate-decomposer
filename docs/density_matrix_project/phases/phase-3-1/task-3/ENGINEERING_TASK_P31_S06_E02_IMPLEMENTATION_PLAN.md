# Engineering Task P31-S06-E02 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S06-E02: Record second-slice completion and the remaining Task 3 / Task 4 expansion hooks`

This is a Layer 4 file-level implementation plan for the second engineering task
under Story `P31-S06` from
`../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the second-slice Task 3 documentation hook into a concrete plan against
the current status/pointer surfaces in:

- `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
- `../FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md`,
- `../DETAILED_PLANNING_PHASE_3_1.md`,
- `TASK_3_MINI_SPEC.md`,
- `ENGINEERING_TASK_P31_S06_E01_IMPLEMENTATION_PLAN.md`,
- and, as the reviewer-facing evidence pointer only,
  `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`.

## Scope

This engineering task closes the **documentation loop** for the second vertical
slice after `P31-S06-E01` lands:

- record the counted 2-qubit case ID and the larger-workload smoke case in the
  second-slice status surface,
- mark the second slice complete in the story file when the E01 evidence is
  true,
- keep the deferred counted IDs, full Task 3 package work, and Task 4 work
  explicit so the slice does not over-claim,
- add and preserve Layer 4 plan cross-links for the `P31-S06` engineering tasks,
- and maintain the already-established pointers from the first-slice and
  detailed-planning docs to the second-slice story file.

Out of scope for this engineering task:

- code changes in `squander/`,
- test or fixture changes beyond updating reviewer-facing pointers,
- moving implementation progress into
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
- full Task 3 correctness-package activation,
- Aer/external-reference packaging from `P31-ADR-011`,
- schema/bundle updates from `P31-ADR-013`,
- and Task 4 performance packaging.

## Current Documentation Gap To Close

The second-slice story already contains the right **status surface**, but it is
still intentionally incomplete until `P31-S06-E02` finishes.

### `SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`

This file already has:

- the `P31-S06` story and its two engineering tasks,
- a **Slice completion checklist** with explicit `P31-S06-E01` / `P31-S06-E02`
  rows,
- and a **Slice implementation status** table with the counted case ID, the
  recommended smoke ID, the deferred counted IDs, and the deferred Task 3 / Task
  4 work.

But the status still says:

- second vertical slice implementation is **In progress**,
- the slice completion checklist still expects `P31-S06-E02`,
- and, before this task lands, the `P31-S06` Layer 4 plan surface is only
  partially linked.

So this file should be the **single source of truth** for second-slice
implementation progress.

### `TASK_3_MINI_SPEC.md`

The Task 3 mini-spec is the natural place to surface the Layer 4 implementation
plans for Story `P31-S06`, just as Task 2 now links to the `P31-S05` engineering
plans.

This file should expose the plan links, but it should **not** become a second
status ledger for slice implementation progress.

### `FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md` and `DETAILED_PLANNING_PHASE_3_1.md`

These documents already point reviewers to the second-slice story file:

- `FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md` points from the first
  slice completion status to the second-slice document as the deferred 2-qubit
  increment.
- `DETAILED_PLANNING_PHASE_3_1.md` already names the second-slice story file as
  the next bounded Task 1–3 increment.

The preferred plan is therefore **minimal change** here:

- keep those pointers intact,
- and only strengthen them if a small one-line clarification materially improves
  reviewer navigation to the second-slice status surface.

## Dependencies And Assumptions

- `P31-S06-E01` is assumed to land first and provide the concrete evidence that
  the docs will record:
  - counted case ID:
    `phase31_microcase_2q_cnot_local_noise_pair`,
  - non-counted smoke ID:
    `phase31_local_support_q4_spectator_embedding_smoke`,
  - evidence module:
    `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`.
- The source-of-truth contract remains:
  - `TASK_3_MINI_SPEC.md`,
  - `../SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`
    (`P31-ADR-009`, `P31-ADR-011`, `P31-ADR-013`).
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` remains a closed Layer 1 gate and
  should not be reused for slice implementation tracking.
- This task should describe the second slice as **complete for its bounded
  claimed surface**, while still naming the remaining deferred counted IDs and
  the broader Task 3 / Task 4 follow-on work.
- The correct reviewer flow remains:
  first-slice story -> second-slice story -> remaining Task 3 / Task 4 work.

## Target Files And Responsibilities

### Primary status file: `SECOND_VERTICAL_SLICE_2Q_LOCAL_SUPPORT_STORIES_AND_ENGINEERING_TASKS.md`

This file should carry the authoritative second-slice implementation status.

#### Expected updates in this file

- Add or maintain Layer 4 implementation-plan links for `P31-S06`.
- Add or maintain the per-task `Implementation plan:` line under each
  `P31-S06` engineering task heading.
- Update the **Slice completion checklist** once `P31-S06-E01` evidence is in
  place.
- Update the **Slice implementation status** table from “In progress” to a
  completion statement that is honest about the bounded second-slice claim.
- Keep the deferred counted IDs and the deferred Task 3 / Task 4 work visible.

### Task-level contract file: `TASK_3_MINI_SPEC.md`

This file should expose the Layer 4 implementation plans for Story `P31-S06`.

#### Preferred update

- Add a small `Layer 4 implementation plans` section near the top.
- Link to:
  - `ENGINEERING_TASK_P31_S06_E01_IMPLEMENTATION_PLAN.md`
  - `ENGINEERING_TASK_P31_S06_E02_IMPLEMENTATION_PLAN.md`

### Existing pointer docs: `FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md` and `DETAILED_PLANNING_PHASE_3_1.md`

These files already point to the second-slice story surface.

#### Preferred plan

- Re-read them after the main status update.
- Leave them unchanged if the existing pointers remain clear.
- Apply only a tiny wording clarification if the second-slice completion hook is
  otherwise hard to discover.

### Explicit non-targets

The following should remain unchanged in this task unless a documentation
consistency bug is exposed:

- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
- `benchmarks/density_matrix/correctness_evidence/`
- `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`
  (beyond being cited as evidence)

## Implementation Sequence

### Step 1: Expose the `P31-S06` Layer 4 plans in the right documentation surfaces

**Goal**

Make the engineering-task plans for Story `P31-S06` discoverable from the Task 3
mini-spec and the second-slice story file.

**Execution checklist**

- [ ] Add or maintain a `Layer 4 implementation plans` section in
      `TASK_3_MINI_SPEC.md`.
- [ ] Add or maintain a corresponding `Layer 4 implementation plans (Task 3)`
      section in the second-slice story file.
- [ ] Add or maintain `Implementation plan:` lines under both
      `P31-S06-E01` and `P31-S06-E02`.

**Why first**

This task is documentation-centered. Reviewers should be able to navigate from
the task contract to the implementation plans before the completion status is
updated.

### Step 2: Record second-slice completion in the story-owned status surfaces

**Goal**

Close the slice progress loop in the one file that already owns it.

**Execution checklist**

- [ ] Update the `P31-S06-E01` and `P31-S06-E02` rows in the **Slice completion
      checklist** when the underlying evidence/doc work is complete.
- [ ] Update the **Slice implementation status** table so the second vertical
      slice is recorded as complete for the bounded local-support claim.
- [ ] Keep the counted case ID and non-counted smoke ID explicit in that table.
- [ ] Keep the evidence-module pointer explicit:
      `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py`.

**Recommended completion wording**

Prefer wording that is truthful and bounded, for example:

- second vertical slice complete for the counted 2-qubit microcase plus the
  non-counted 4-qubit local-support smoke case,
- while full Task 3 package closure and Task 4 performance work remain deferred.

### Step 3: Keep the deferred follow-on hooks explicit and reviewer-friendly

**Goal**

Ensure the completion record makes clear what the second slice does **not**
claim.

**Execution checklist**

- [ ] Preserve the deferred counted correctness IDs after this slice:
      - `phase31_microcase_2q_multi_noise_entangler_chain`,
      - `phase31_microcase_2q_dense_same_support_motif`,
      - `phase2_xxz_hea_q4_continuity`,
      - `phase2_xxz_hea_q6_continuity`,
      - plus counted performance cases from `P31-ADR-010`.
- [ ] Preserve the deferred full Task 3 package work:
      Aer per `P31-ADR-011`; bundle/schema work per `P31-ADR-013`.
- [ ] Preserve the deferred Task 4 performance families and packaging hooks.
- [ ] Avoid wording that could be read as “full Task 3 closure complete.”

**Why this matters**

`P31-S06-E02` is as much about claim-boundary honesty as about status
bookkeeping.

### Step 4: Re-read first-slice and planning pointers without creating duplicate status ledgers

**Goal**

Keep the cross-document reviewer path coherent without spreading slice progress
state across multiple files.

**Execution checklist**

- [ ] Re-read `FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md`.
- [ ] Re-read `DETAILED_PLANNING_PHASE_3_1.md`.
- [ ] If their current pointers to the second-slice story file are already clear,
      leave them unchanged.
- [ ] If a small clarification helps, add a one-line pointer to the second-slice
      story/status surface only.
- [ ] Do not move implementation status into
      `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.

**Preferred principle**

One file should own second-slice status; the other docs should point to it, not
restate it independently.

### Step 5: Validate document consistency and traceability

**Goal**

Finish with a clean, internally consistent doc surface for the second slice.

**Execution checklist**

- [ ] Re-read the updated task mini-spec and second-slice story file.
- [ ] Re-read any pointer docs that were touched.
- [ ] Verify all newly added relative links resolve within the `phase-3-1`
      directory structure.
- [ ] Keep runtime/test evidence references stable and concrete.

**Evidence produced**

- Updated reviewer-facing links from Task 3 and the second-slice story to both
  `P31-S06` implementation plans.
- Updated second-slice status table and completion checklist.
- Optional tiny pointer clarification in first-slice or detailed-planning docs if
  truly needed.

## Acceptance Evidence

`P31-S06-E02` is ready to hand off when all of the following are true:

- `TASK_3_MINI_SPEC.md` links to both
  `ENGINEERING_TASK_P31_S06_E01_IMPLEMENTATION_PLAN.md` and
  `ENGINEERING_TASK_P31_S06_E02_IMPLEMENTATION_PLAN.md`,
- the second-slice story file links to both plans and includes per-task
  `Implementation plan:` lines,
- the second-slice **Slice completion checklist** records `P31-S06-E01` and
  `P31-S06-E02` as complete when true,
- the second-slice **Slice implementation status** table records the bounded
  second-slice completion honestly,
- the remaining deferred counted IDs and broader Task 3 / Task 4 work remain
  explicit,
- and slice implementation progress is still kept out of
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.

## Handoff To The Next Engineering Tasks

After `P31-S06-E02` lands:

- The second-slice story file becomes the stable reviewer-facing completion
  record for the bounded 2q/local-support increment.
- Broader Task 3 work can later decide when to migrate the active
  `correctness_evidence` builders onto the Phase 3.1 runtime path and package
  schema.
- Task 4 work can reuse the same deferred-family list and claim-boundary wording
  rather than inventing a parallel status surface.

## Risks / Rollback

- Risk: the second slice may be documented as if full Task 3 closure is complete.
  Rollback/mitigation: keep the deferred counted IDs, Aer work, and schema work
  explicitly listed in the status section.

- Risk: status may drift across multiple docs.
  Rollback/mitigation: keep the second-slice story file as the single status
  owner and use other docs only as pointers.

- Risk: plan links may be added inconsistently between the task mini-spec and the
  story file.
  Rollback/mitigation: mirror the same pair of relative links in both places and
  re-read them together after editing.
