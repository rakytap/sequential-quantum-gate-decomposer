# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor
Contract

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the Phase 3 methods-workload requirement into one explicit
descriptor contract:

- the required 2 to 4 qubit micro-validation cases emit partition descriptors
  through the same Task 2 contract as the continuity anchor,
- the mandatory structured noisy `U3` / `CNOT` families also emit descriptors
  through that same contract,
- deterministic workload identity, seed policy, and noise-placement vocabulary
  stay stable enough for reproducible descriptor evidence,
- and Story 2 closes shared positive descriptor coverage across mandatory
  workload classes without claiming runtime correctness thresholds, fused
  execution, or density-aware heuristic optimality.

Out of scope for this story:

- continuity-anchor descriptor closure already owned by Story 1,
- exact within-partition gate/noise-order closure owned by Story 3,
- reconstructible remapping and parameter-routing closure owned by Story 4,
- cross-workload provenance and audit-bundle stability owned by Story 5,
- unsupported or lossy descriptor-boundary closure owned by Story 6,
- and runtime execution, exactness, or performance claims owned by later Phase
  3 tasks.

## Dependencies And Assumptions

- Story 1 already established the continuity-anchor route into the Task 2
  descriptor surface and should remain the reference positive path for the
  continuity workload.
- The shared Task 2 descriptor substrate now lives in
  `squander/partitioning/noisy_planner.py` through
  `build_partition_descriptor_set()` and
  `build_phase3_continuity_partition_descriptor_set()`.
- Story 1 now also established the first rerunnable Task 2 validation surfaces
  in:
  - `tests/partitioning/test_planner_surface_descriptors.py`,
  - `benchmarks/density_matrix/planner_surface/continuity_descriptor_validation.py`,
  - and the emitted bundle under
    `benchmarks/density_matrix/artifacts/planner_surface/continuity_descriptor/`.
- The frozen workload requirements come from the Phase 3 support-matrix,
  workflow-anchor, and benchmark-minimum decisions in
  `DETAILED_PLANNING_PHASE_3.md`.
- Existing reusable workload material already exists in:
  - `benchmarks/density_matrix/planner_surface/`,
  - the Task 1 workload builders and helpers used for microcases and structured
    families,
  - and the shared canonical planner module
    `squander/partitioning/noisy_planner.py`.
- Story 2 should reuse those surfaces or add tightly scoped Task 2 successors,
  but it should not create multiple incompatible definitions of the mandatory
  microcases or structured families.
- The required gate family remains `U3` and `CNOT`; the required noise surface
  remains local single-qubit depolarizing, amplitude damping, and phase damping
  or dephasing.
- Story 2 closes shared descriptor coverage across mandatory workload classes.
  It does not need to decide final within-partition ordering rules, remapping
  semantics, or unsupported-boundary taxonomy.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Task 2 Workload Inventory And Descriptor Identity Scheme

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names the exact workload classes it owns.
- Every mandatory workload class has stable identifiers and deterministic
  construction rules.
- Story 2 distinguishes mandatory descriptor coverage from optional exploratory
  workload growth.

**Execution checklist**
- [ ] Freeze the required 2 to 4 qubit micro-validation inventory and its case
      identifiers for Task 2 descriptor work.
- [ ] Freeze the mandatory structured noisy family inventory for the methods
      descriptor slice.
- [ ] Freeze deterministic seed and construction rules for structured families.
- [ ] Freeze the required sparse, periodic, and dense local-noise placement
      patterns for the methods workload slice.

**Evidence produced**
- One stable Task 2 mandatory-workload inventory.
- One reviewable identifier and seed-policy scheme for reproducibility.

**Risks / rollback**
- Risk: if workload identity remains implicit, later descriptor evidence may mix
  required cases with exploratory additions.
- Rollback/mitigation: define stable workload IDs and deterministic construction
  rules before broadening descriptor coverage.

### Engineering Task 2: Build Or Reuse Descriptor Builders For The Required 2 To 4 Qubit Microcases

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- code | tests

**Definition of done**
- Required microcases can emit supported partition descriptors through the
  shared Task 2 contract.
- Microcase construction reuses existing workload-building surfaces where
  practical.
- Microcase descriptor emission stays inside the frozen support matrix.

**Execution checklist**
- [ ] Reuse or refactor the existing microcase builders under
      `benchmarks/density_matrix/planner_surface/` or a tightly related Task 2
      successor.
- [ ] Route required microcases through `build_partition_descriptor_set()`
      rather than benchmark-local output schemas.
- [ ] Keep the required microcase gate and noise vocabulary tightly bounded to
      the frozen support matrix.
- [ ] Add focused tests proving supported microcases emit descriptors through
      the shared Task 2 contract.

**Evidence produced**
- Reviewable descriptor-entry builders for mandatory microcases.
- Fast regression coverage for microcase descriptor representability.

**Risks / rollback**
- Risk: Story 2 may duplicate microcase logic across validation, benchmarks,
  and descriptor generation.
- Rollback/mitigation: centralize microcase construction behind one shared
  builder or adapter path.

### Engineering Task 3: Add Deterministic Structured Noisy `U3` / `CNOT` Family Descriptor Generators

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- code | benchmark harness

**Definition of done**
- Each required structured family can emit descriptors reproducibly.
- The generated families stay inside the frozen `U3` / `CNOT` plus local-noise
  support surface.
- Structured-family descriptor generation is separated cleanly from runtime
  benchmarking.

**Execution checklist**
- [ ] Extend the structured-family generation surface under
      `benchmarks/density_matrix/planner_surface/` or add a Task 2-specific
      successor for descriptor work.
- [ ] Encode deterministic seed rules and family identifiers directly in the
      generation surface.
- [ ] Encode the required sparse, periodic, and dense local-noise placement
      patterns as stable construction options.
- [ ] Keep larger performance-study variants outside the mandatory Story 2
      closure bar unless they reuse the same stable family IDs.

**Evidence produced**
- One stable structured-family descriptor generation surface.
- Deterministic family instances that can be rerun and audited.

**Risks / rollback**
- Risk: ad hoc structured-family generation will make later descriptor and
  benchmark interpretation unstable.
- Rollback/mitigation: freeze deterministic family construction and naming at
  Story 2 time.

### Engineering Task 4: Normalize All Mandatory Workload Classes Into One Shared Descriptor Schema

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- code | tests

**Definition of done**
- Continuity cases, microcases, and structured families all emit descriptors
  through one shared Task 2 contract.
- The descriptor shape does not depend on whether the case came from continuity
  lowering, direct microcase construction, or structured-family generation.
- Story 2 proves shared positive coverage without requiring identical upstream
  builders.

**Execution checklist**
- [ ] Reuse the shared descriptor-generation surface instead of introducing a
      workload-specific descriptor object model.
- [ ] Ensure the resulting descriptor records share the same top-level schema
      across workload classes.
- [ ] Add checks that mandatory workload cases do not require workload-specific
      descriptor semantics to be interpreted correctly.
- [ ] Keep richer cross-workload audit stability for Story 5 rather than
      overloading Story 2 with broader audit concerns.

**Evidence produced**
- One shared descriptor schema used across mandatory workload classes.
- Regression checks comparing schema-level coverage across workload classes.

**Risks / rollback**
- Risk: if each workload class emits a different descriptor contract, later
  semantic-preservation work will become hard to compare.
- Rollback/mitigation: enforce one shared positive descriptor surface even if
  builders differ upstream.

### Engineering Task 5: Add A Story 2 Descriptor-Coverage Matrix And Fast Regression Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 has a focused descriptor-coverage matrix covering required workload
  classes and required noise-placement patterns.
- Fast regression checks catch missing workload-family coverage or descriptor
  schema drift.
- The regression surface remains narrower than full runtime or numerical
  validation.

**Execution checklist**
- [ ] Extend `tests/partitioning/test_planner_surface_descriptors.py` with the Story 2
      mandatory-workload coverage matrix.
- [ ] Cover mandatory microcases and at least one deterministic instance of each
      required structured family.
- [ ] Cover the required noise-placement patterns where they matter to workload
      identity.
- [ ] Fail quickly when a mandatory workload class can no longer emit the shared
      descriptor contract.

**Evidence produced**
- Focused Story 2 regression coverage for shared descriptor coverage.
- Reviewable failure messages for missing mandatory workload support.

**Risks / rollback**
- Risk: without a focused gate, Story 2 regressions may only surface later in
  broad validation or benchmark runs.
- Rollback/mitigation: add a lightweight coverage matrix that runs early and
  deterministically.

### Engineering Task 6: Emit A Stable Mandatory-Workload Descriptor Bundle

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 emits one stable bundle summarizing descriptor emission across the
  mandatory workload matrix.
- The bundle records workload IDs, seed policy, noise-placement pattern, source
  type, and descriptor-summary metadata.
- The output is stable enough for later Task 2 stories to reuse.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/mandatory_workload/`).
- [ ] Emit machine-reviewable metadata for each required workload class and
      size.
- [ ] Record whether the case came from continuity lowering, direct microcase
      construction, or structured-family generation.
- [ ] Keep the bundle focused on shared descriptor coverage and identity rather
      than later exactness or performance interpretation.

**Evidence produced**
- One stable Story 2 mandatory-workload descriptor bundle.
- One reusable output shape for later benchmark and paper packaging.

**Risks / rollback**
- Risk: later bundles may mix required and optional workload cases if Story 2
  never emits its own clean inventory.
- Rollback/mitigation: emit one explicit mandatory-workload bundle before
  broader Task 2 validation begins.

### Engineering Task 7: Document And Run The Story 2 Shared-Contract Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Descriptor Contract`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Story 2 notes explain how mandatory workload classes reach the shared
  descriptor contract.
- Fast regression checks and the Story 2 bundle run successfully.
- Story 2 closes with rerunnable evidence rather than informal workload
  descriptions.

**Execution checklist**
- [ ] Document the mandatory-workload inventory, source labels, seed policy, and
      noise-placement vocabulary.
- [ ] Run focused Story 2 regression coverage for shared descriptor coverage.
- [ ] Run the Story 2 bundle-emission path and verify output.
- [ ] Record stable test and artifact references for Stories 3 through 6.

**Evidence produced**
- Passing Story 2 descriptor-coverage checks.
- One stable workload-bundle reference proving shared descriptor-contract
  coverage.

**Risks / rollback**
- Risk: Story 2 can appear finished while still leaving methods workloads
  under-specified and non-reproducible.
- Rollback/mitigation: require both passing checks and one stable emitted bundle
  before closure.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the required 2 to 4 qubit microcases and the mandatory structured noisy
  `U3` / `CNOT` families emit the same Task 2 descriptor contract as the
  continuity anchor,
- deterministic workload identifiers, seed rules, and noise-placement patterns
  are frozen for the mandatory methods workload slice,
- fast regression coverage detects missing mandatory workload-family coverage,
- one stable Story 2 bundle records descriptor emission across mandatory
  workload classes,
- and exact order/noise semantics, reconstructibility, shared audit stability,
  and unsupported-boundary closure remain clearly assigned to later stories.

## Implementation Notes

- Prefer extending the existing Task 1 workload builders over creating a second
  disconnected descriptor workload zoo.
- Keep Story 2 focused on workload identity and shared positive coverage, not on
  proving exactness or speedup.
- Treat seed policy and noise-placement vocabulary as part of the descriptor
  contract, not as incidental benchmark details.
