# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor
Contract

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the frozen Phase 2 continuity anchor into explicit Task 2
descriptor behavior:

- the noisy XXZ `HEA` continuity workflow already accepted by Task 1 now emits
  a first positive partition-descriptor slice for Task 2,
- the continuity slice exposes stable partition ordering and stable references
  back to the canonical noisy planner surface,
- the continuity descriptor path is auditable enough that later stories can add
  shared methods-workload coverage, richer semantic guarantees, and broader
  audit packaging without replacing the anchor path,
- and the continuity descriptor route stays tied to the existing exact density
  workflow rather than to new Phase 4 workflow growth.

Out of scope for this story:

- shared descriptor coverage for the mandatory methods workloads owned by Story
  2,
- exact within-partition gate/noise-order closure owned by Story 3,
- reconstructible remapping and parameter-routing closure owned by Story 4,
- cross-workload provenance and audit-schema stability owned by Story 5,
- unsupported or lossy descriptor-boundary closure owned by Story 6,
- and runtime exactness, fused execution, or performance claims owned by later
  Phase 3 tasks.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_2_MINI_SPEC.md`,
  `TASK_2_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`,
  `P3-ADR-004`, `P3-ADR-008`, and `P3-ADR-009`.
- Task 1 already established the continuity workflow route into the canonical
  noisy planner surface, including the continuity workload IDs and supported
  entry-route labels.
- The current continuity-entry substrate already includes:
  - `build_phase3_continuity_planner_surface()` in
    `squander/partitioning/noisy_planner.py`,
  - the Phase 2 bridge inspection path exposed through
    `describe_density_bridge()`,
  - and the existing Task 1 continuity evidence and tests.
- The sequential `NoisyCircuit` executor remains the internal exact semantic
  oracle for later validation, but Story 1 does not yet attempt runtime
  agreement checks. Its purpose is to make the continuity descriptor contract
  reviewable before runtime work begins.
- Existing state-vector partitioning utilities may be reused as scaffolding for
  partition grouping or parameter bookkeeping, but the Story 1 contract is the
  descriptor surface itself rather than any legacy state-vector output format.
- Story 1 should establish one explicit positive continuity-descriptor slice
  without prematurely freezing all cross-workload schema choices that belong to
  later stories.

## Engineering Tasks

### Engineering Task 1: Freeze The Continuity Descriptor Slice And Review Boundary

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 names the exact continuity cases it owns.
- Story 1 defines successful descriptor emission as a reviewable descriptor
  contract, not as runtime execution or acceleration.
- The handoff from Story 1 to Stories 2 through 6 is explicit.

**Execution checklist**
- [ ] Freeze the required continuity descriptor case inventory around the noisy
      XXZ `HEA` workflow at the mandated 4, 6, 8, and 10 qubit sizes.
- [ ] Define what counts as successful Story 1 descriptor emission:
      stable partition ordering plus stable references back to the canonical
      noisy planner surface.
- [ ] Define the minimum continuity-descriptor metadata needed for Story 1
      review, such as workload ID, partition ordering, referenced canonical
      operation range or membership, and partition-span summary.
- [ ] Keep shared methods-workload coverage, richer semantic closure, and
      unsupported-boundary closure explicitly outside the Story 1 bar.

**Evidence produced**
- One stable Story 1 continuity-descriptor contract description.
- One reviewable continuity case inventory for the required anchor sizes.

**Risks / rollback**
- Risk: if Story 1 mixes descriptor closure with later runtime or performance
  goals, the first positive Task 2 slice will become hard to review.
- Rollback/mitigation: keep Story 1 focused on continuity-descriptor emission
  only.

### Engineering Task 2: Reuse The Existing Continuity Planner Surface As The Descriptor Input Contract

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- code | tests

**Definition of done**
- The continuity anchor reaches Task 2 descriptor generation from one explicit
  canonical planner-surface path.
- The implementation does not introduce a second competing interpretation of
  the same continuity workload.
- The descriptor adapter stays visibly rooted in the already accepted Task 1
  continuity path.

**Execution checklist**
- [ ] Reuse `build_phase3_continuity_planner_surface()` or the smallest shared
      successor as the input contract for Story 1 descriptor emission.
- [ ] Add only the minimal adapter layer needed to turn the continuity planner
      surface into descriptor records.
- [ ] Keep descriptor emission deterministic on the existing supported
      continuity fixtures.
- [ ] Avoid moving continuity descriptor logic into benchmark-only scripts when
      the planner module already exposes the supported continuity surface.

**Evidence produced**
- One reviewable continuity-to-descriptor adapter rooted in the canonical
  planner surface.
- Focused regression coverage proving the continuity anchor reaches that path.

**Risks / rollback**
- Risk: a second continuity-descriptor interpretation path can drift from the
  accepted Task 1 planner surface.
- Rollback/mitigation: make the Task 1 continuity planner surface the single
  source for Story 1 descriptor generation.

### Engineering Task 3: Define The Minimal Story 1 Descriptor Record Shape

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- docs | code | tests

**Definition of done**
- Story 1 defines one narrow positive descriptor record shape for the continuity
  anchor.
- The record shape is sufficient to review continuity partition ordering and
  canonical-operation references.
- The record shape remains small enough that later stories can extend it
  cleanly.

**Execution checklist**
- [ ] Freeze one descriptor record shape for Story 1 continuity cases.
- [ ] Include stable partition ordering and stable references to canonical
      operation indices or equivalent canonical-operation membership.
- [ ] Record continuity workload identity and the minimal partition-span summary
      needed for review.
- [ ] Keep richer cross-workload provenance closure and descriptor-wide audit
      vocabulary for Story 5 rather than overloading Story 1.

**Evidence produced**
- One stable Story 1 continuity descriptor record shape.
- One clear boundary between Story 1 review fields and later descriptor-audit
  fields.

**Risks / rollback**
- Risk: an oversized Story 1 record shape can blur the boundary between the
  first positive slice and later shared audit work.
- Rollback/mitigation: freeze only the minimum fields needed for continuity
  review.

### Engineering Task 4: Emit Continuity Descriptors For The Required Anchor Sizes

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported 4, 6, 8, and 10 qubit continuity cases emit descriptor sets through
  the shared Story 1 path.
- Descriptor emission stays deterministic across reruns.
- The emitted records remain auditable without consulting hidden runtime state.

**Execution checklist**
- [ ] Add descriptor emission for the required supported continuity sizes.
- [ ] Preserve stable partition ordering across reruns on deterministic
      continuity fixtures.
- [ ] Preserve canonical-operation references explicitly in the emitted records.
- [ ] Keep descriptor emission separate from later runtime execution or exactness
      checks.

**Evidence produced**
- Continuity descriptor outputs for the required anchor sizes.
- Regression coverage proving deterministic descriptor emission.

**Risks / rollback**
- Risk: continuity cases may appear supported while still relying on hidden
  planner or runtime state to interpret the emitted descriptors.
- Rollback/mitigation: record the positive continuity slice entirely in the
  emitted descriptor contract.

### Engineering Task 5: Add A Focused Story 1 Continuity-Descriptor Validation Gate

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 has a rerunnable validation layer dedicated to the continuity
  descriptor slice.
- The validation layer covers the required anchor sizes and emitted descriptor
  metadata.
- The validation gate remains narrower than later runtime semantic-preservation
  work.

**Execution checklist**
- [ ] Add a focused Task 2 regression slice in
      `tests/partitioning/test_planner_surface_descriptors.py` or a tightly related successor.
- [ ] Check descriptor emission for supported 4, 6, 8, and 10 qubit continuity
      cases.
- [ ] Assert stable partition ordering and stable canonical-operation references
      on the emitted records.
- [ ] Keep the Story 1 validation layer separate from shared methods-workload
      coverage and unsupported-descriptor tests.

**Evidence produced**
- One rerunnable Story 1 continuity-descriptor validation surface.
- Fast regression coverage for the first positive Task 2 slice.

**Risks / rollback**
- Risk: Story 1 may close with only informal inspection and no repeatable
  continuity-descriptor gate.
- Rollback/mitigation: require one dedicated validation layer before closure.

### Engineering Task 6: Emit A Stable Story 1 Continuity-Descriptor Artifact Bundle

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable artifact bundle or rerunnable
  checker for the continuity-descriptor slice.
- The output records enough metadata to prove continuity descriptor emission on
  supported cases.
- The artifact shape is stable enough for later Task 2 stories to extend.

**Execution checklist**
- [ ] Add a dedicated Story 1 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/continuity_descriptor/`).
- [ ] Record workload ID, partition ordering, canonical-operation references,
      and partition-span summaries for supported continuity cases.
- [ ] Record the rerun command and software metadata with the artifact.
- [ ] Keep the artifact narrow to descriptor emission rather than runtime or
      performance interpretation.

**Evidence produced**
- One stable Story 1 continuity-descriptor artifact bundle.
- One reusable output shape for later shared descriptor work.

**Risks / rollback**
- Risk: prose-only closure makes the first positive Task 2 slice hard to cite
  and easy to misstate later.
- Rollback/mitigation: emit one thin machine-reviewable bundle early.

### Engineering Task 7: Document The Story 1 Handoff To Later Descriptor Stories

**Implements story**
- `Story 1: The Frozen Phase 2 Continuity Workflow Emits One Auditable Descriptor Contract`

**Change type**
- docs

**Definition of done**
- Story 1 notes explain exactly what the continuity-descriptor slice closes.
- The continuity anchor is documented as the first positive descriptor path, not
  as the whole Task 2 contract.
- Developer-facing notes point to the Story 1 validation and artifact path.

**Execution checklist**
- [ ] Document the supported continuity-descriptor slice and its evidence
      surface.
- [ ] Explain that shared methods-workload coverage belongs to Story 2.
- [ ] Explain that exact within-partition ordering, reconstructibility,
      cross-workload audit stability, and unsupported-boundary closure belong to
      Stories 3 through 6.
- [ ] Record stable references to the Story 1 tests and artifact bundle.

**Evidence produced**
- Updated developer-facing notes for the Story 1 continuity-descriptor gate.
- One stable handoff reference for later Task 2 work.

**Risks / rollback**
- Risk: later Task 2 work may over-assume Story 1 closed the full descriptor
  contract.
- Rollback/mitigation: document the handoff boundaries explicitly.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- the frozen Phase 2 noisy XXZ `HEA` continuity workflow at 4, 6, 8, and 10
  qubits emits one auditable descriptor contract,
- the continuity-descriptor path is rooted in one explicit canonical
  planner-surface route rather than multiple competing interpretations,
- stable partition ordering and stable canonical-operation references are
  reviewable from the emitted descriptor records,
- one stable Story 1 validation command or artifact bundle proves descriptor
  emission for the continuity anchor,
- and shared methods-workload coverage, richer semantic closure, cross-workload
  audit stability, and unsupported-boundary closure remain clearly assigned to
  Stories 2 through 6.

## Implementation Notes

- Prefer reusing the existing Task 1 continuity planner surface over rebuilding
  continuity-descriptor emission from raw VQE internals.
- Keep Story 1 focused on the first positive descriptor slice, not on full
  semantic-preservation closure.
- Treat the continuity anchor as a required Task 2 workload, not as evidence
  that every supported workload already shares the final descriptor contract.
