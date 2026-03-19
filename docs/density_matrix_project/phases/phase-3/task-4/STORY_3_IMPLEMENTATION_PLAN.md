# Story 3 Implementation Plan

## Story Being Implemented

Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same
Fused-Capable Runtime Surface

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the first positive fused structured-runtime slice into one
shared fused-capable surface:

- any in-bounds continuity or micro-validation cases used for fused review reuse
  the same fused-capable runtime entry point established by Story 2,
- path labeling, output shape, and audit vocabulary stay aligned across
  structured, continuity, and microcase slices,
- no separate continuity-only or microcase-only fused harness is introduced,
- and Story 3 closes cross-surface reuse without requiring fused coverage on
  every continuity or microcase case.

Out of scope for this story:

- descriptor-local eligibility definition already owned by Story 1,
- the first positive structured fused-runtime slice already owned by Story 2,
- exact noisy-semantics preservation thresholds owned by Story 4,
- explicit fused versus unfused versus deferred classification closure owned by
  Story 5,
- stable fused-output packaging owned by Story 6,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 and 2 already define the eligibility rule and the first positive
  fused runtime path that Story 3 must reuse rather than replace.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, and `P3-ADR-009`.
- The existing Task 3 continuity and microcase runtime surfaces already provide
  the reference audit and output shapes Story 3 should stay aligned with:
  - `benchmarks/density_matrix/partitioned_runtime/continuity_runtime_validation.py`,
  - `benchmarks/density_matrix/partitioned_runtime/mandatory_workload_runtime_validation.py`,
  - `benchmarks/density_matrix/partitioned_runtime/runtime_output_validation.py`,
  - and their emitted bundles under `benchmarks/density_matrix/artifacts/phase3_task3/`.
- The continuity builder Story 3 should reuse already exists through
  `build_phase3_story1_continuity_vqe()` and
  `build_phase3_continuity_partition_descriptor_set()`.
- The microcase builders Story 3 should reuse already exist through
  `iter_story2_microcase_descriptor_sets()` in
  `benchmarks/density_matrix/planner_surface/workloads.py`.
- The shared runtime substrate remains the Task 3 runtime layer in
  `squander/partitioning/noisy_runtime.py`; Story 3 should not introduce a
  separate fused interface outside that runtime surface.
- Story 3 should prefer one reused fused-capable runtime path vocabulary and one
  reused result shape over multiple workload-local fused conventions.

## Engineering Tasks

### Engineering Task 1: Freeze The Shared Fused-Capable Reuse Rule For In-Bounds Cases

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines when continuity or micro-validation cases count as valid reuse
  of the shared fused-capable runtime surface.
- The rule is explicit enough to prevent continuity-only or microcase-only fused
  interfaces from appearing.
- Story 3 keeps mandatory structured fused execution separate from optional
  reuse slices.

**Execution checklist**
- [ ] Define which in-bounds continuity or microcase cases may be used for
      fused-capable reuse validation.
- [ ] Define what it means to reuse the same fused-capable runtime surface:
      same runtime entry point, the same runtime-path vocabulary, and the same
      output or audit shape, even when the actual runtime path remains the plain
      baseline because no fused island executed on a given in-bounds case.
- [ ] Define that fused coverage is not mandatory on every continuity or
      microcase case.
- [ ] Keep semantic-threshold and performance closure outside the Story 3 bar.

**Evidence produced**
- One stable Story 3 shared-surface reuse rule.
- One explicit boundary between reused fused-capable slices and the required
  structured positive path.

**Risks / rollback**
- Risk: Story 3 may blur into a demand that every continuity or microcase case
  must fuse, which is stronger than the phase contract requires.
- Rollback/mitigation: freeze the reuse rule around shared surface, not around
  universal fused coverage.

### Engineering Task 2: Reuse The Story 2 Fused Runtime Entry Point For Continuity And Microcase Slices

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- code | tests

**Definition of done**
- Continuity and microcase fused-capable runs use the same fused runtime entry
  point as the required structured workloads.
- Story 3 does not introduce a second fused execution API.
- Shared runtime-path labeling remains explicit and reviewable.

**Execution checklist**
- [ ] Reuse the Story 2 fused runtime entry point in
      `squander/partitioning/noisy_runtime.py` for any in-bounds continuity or
      microcase fused-capable runs.
- [ ] Preserve the same runtime-path vocabulary and provenance fields used by
      the structured fused path.
- [ ] Avoid continuity-only or microcase-only fused helpers that would become a
      second contract surface.
- [ ] Add focused checks proving the same runtime surface is reused.

**Evidence produced**
- One shared fused runtime entry path across structured and in-bounds reuse
  slices.
- Focused regression coverage for reused path labeling.

**Risks / rollback**
- Risk: secondary fused helpers can silently diverge from the structured fused
  runtime path while still appearing equivalent.
- Rollback/mitigation: route all in-bounds fused-capable reuse through the same
  runtime entry surface.

### Engineering Task 3: Select Representative Continuity And Microcase Reuse Fixtures

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 uses a small representative set of continuity and microcase fixtures
  for fused-capable reuse review.
- The fixture set is broad enough to expose reuse drift if it appears.
- The fixture set remains smaller than the full structured fused benchmark
  inventory.

**Execution checklist**
- [ ] Select at least one continuity-anchor descriptor set and at least one
      microcase descriptor set that can exercise or inspect the fused-capable
      surface.
- [ ] Prefer fixtures whose audit records will stress shared path labeling and
      shared output shape.
- [ ] Record which fixtures are used only for reuse validation versus those used
      for later semantic thresholds.
- [ ] Keep the reuse matrix small and stable.

**Evidence produced**
- One representative Story 3 continuity and microcase reuse matrix.
- One stable inventory of in-bounds reuse fixtures for later stories.

**Risks / rollback**
- Risk: Story 3 may close on hand-picked reuse cases that do not actually cover
  the continuity and microcase surfaces reviewers care about.
- Rollback/mitigation: freeze a small but representative reuse matrix.

### Engineering Task 4: Keep Runtime Path Labels, Output Shape, And Audit Vocabulary Aligned

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- code | tests | docs

**Definition of done**
- Shared fused-capable reuse cases expose the same runtime-path vocabulary as
  the required structured fused cases.
- Output and audit shapes remain aligned enough for later Story 6 packaging.
- Story 3 prevents workload-local relabeling of the same fused-capable path.

**Execution checklist**
- [ ] Reuse the same runtime-path label fields on structured, continuity, and
      microcase fused-capable outputs.
- [ ] Allow the actual runtime path to remain the plain baseline on reuse cases
      with no real fused coverage, while keeping the fused-capable entry surface
      and classification vocabulary shared.
- [ ] Reuse the same top-level provenance and summary field names where the
      fields overlap.
- [ ] Keep fused-capable reuse outputs compatible with the Task 3 result and
      audit record shape.
- [ ] Document any additive Story 3 metadata explicitly rather than renaming
      shared fields.

**Evidence produced**
- One aligned Story 3 path-label and output-shape rule.
- Focused checks proving shared field stability across reuse slices.

**Risks / rollback**
- Risk: continuity and microcase fused-capable cases may become structurally
  incomparable to the structured fused cases they are meant to validate.
- Rollback/mitigation: align path labels and overlapping fields explicitly.

### Engineering Task 5: Add A Focused Story 3 Shared-Surface Validation Gate

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 has a rerunnable validation layer dedicated to shared fused-capable
  surface reuse.
- The gate checks structured, continuity, and microcase reuse alignment.
- The validator remains narrower than later semantic-threshold and benchmark
  work.

**Execution checklist**
- [ ] Add focused Story 3 checks in `tests/partitioning/test_phase3_task4.py`.
- [ ] Add a Story 3 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_surface_reuse_validation.py` as the primary checker.
- [ ] Assert reuse of the same runtime entry point, runtime-path vocabulary, and
      output or audit shape across representative cases.
- [ ] Keep the validator focused on shared-surface reuse, not on benchmark
      thresholds.

**Evidence produced**
- One rerunnable Story 3 shared-surface validation surface.
- Fast regression coverage for fused-capable reuse stability.

**Risks / rollback**
- Risk: shared-surface drift may remain hidden until later bundle or benchmark
  work, when it becomes expensive to unwind.
- Rollback/mitigation: add a focused reuse gate early.

### Engineering Task 6: Emit A Stable Story 3 Surface-Reuse Bundle

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-reviewable bundle or rerunnable checker for
  shared fused-capable reuse.
- The bundle records structured, continuity, and microcase cases through one
  shared record shape.
- The bundle is reusable by later Story 6 output packaging.

**Execution checklist**
- [ ] Add a dedicated Story 3 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task4/story3_surface_reuse/`).
- [ ] Record representative structured, continuity, and microcase cases in one
      shared schema.
- [ ] Record runtime-path labels, provenance fields, and any fused-coverage
      summaries needed to prove shared-surface reuse.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 3 surface-reuse bundle or checker.
- One reusable audit surface for later fused packaging work.

**Risks / rollback**
- Risk: if Story 3 emits only prose claims, later reviewers will not be able to
  tell whether shared-surface reuse was actually enforced.
- Rollback/mitigation: emit one machine-reviewable reuse bundle.

### Engineering Task 7: Document The Story 3 Boundary Around Optional Reuse

**Implements story**
- `Story 3: In-Bounds Continuity And Micro-Validation Slices Reuse The Same Fused-Capable Runtime Surface`

**Change type**
- docs

**Definition of done**
- Story 3 notes explain what shared-surface reuse means and what it does not
  require.
- The continuity and microcase reuse bundle is documented as optional in-bounds
  reuse evidence rather than as the whole fused claim.
- Developer-facing notes point to the Story 3 tests and artifact path.

**Execution checklist**
- [ ] Document the supported shared-surface reuse rule and its evidence surface.
- [ ] Explain that required positive fused execution still centers on structured
      workloads from Story 2.
- [ ] Explain that semantic thresholds, explicit classification, stable output
      packaging, and threshold-or-diagnosis benchmarking belong to Stories 4
      through 7.
- [ ] Record stable references to the Story 3 tests and emitted bundle.

**Evidence produced**
- Updated developer-facing notes for Story 3 shared-surface reuse.
- One stable handoff reference for later Task 4 work.

**Risks / rollback**
- Risk: later work may overread Story 3 as universal fused coverage rather than
  as shared-surface reuse.
- Rollback/mitigation: document the reuse boundary explicitly.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- any continuity or micro-validation cases used for fused review reuse the same
  fused-capable runtime entry point as the structured fused path,
- shared runtime-path labels and overlapping output or audit fields stay aligned
  across structured, continuity, and microcase reuse slices,
- one stable Story 3 validator or artifact bundle proves that no second
  continuity-only or microcase-only fused surface was introduced,
- and semantic-threshold closure, explicit classification, stable fused-output
  packaging, and benchmark interpretation remain clearly assigned to later
  stories.

## Implementation Notes

- Prefer one shared fused-capable runtime surface over multiple convenient local
  helpers.
- Keep Story 3 focused on reuse and alignment, not on proving acceleration.
- Do not let optional continuity or microcase reuse expand into a requirement
  that every in-bounds case must fuse.
