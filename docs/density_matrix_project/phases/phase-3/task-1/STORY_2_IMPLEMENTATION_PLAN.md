# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Mandatory Microcases And Structured Families Share One Canonical
Planner Contract

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns the Phase 3 methods-workload requirement into one explicit
planner-entry contract:

- the required 2 to 4 qubit micro-validation cases enter the same canonical
  planner surface as the continuity anchor,
- the mandatory structured noisy `U3` / `CNOT` families also enter that same
  planner surface,
- deterministic workload identity, seed policy, and noise-placement metadata are
  frozen tightly enough for reproducible planner-entry evidence,
- and Story 2 closes representability across mandatory workload classes without
  claiming runtime correctness thresholds, runtime acceleration, or heuristic
  optimality.

Out of scope for this story:

- continuity-anchor closure already owned by Story 1,
- richer audit and provenance schema closure owned by Story 3,
- optional exact lowering from legacy source surfaces owned by Story 4,
- unsupported planner-request closure owned by Story 5,
- and partition execution, fused runtime, or density-aware cost modeling owned
  by later Phase 3 tasks.

## Dependencies And Assumptions

- Story 1 already established the continuity-anchor route into the canonical
  planner surface and should remain the reference positive path for continuity.
- The shared Task 1 implementation substrate is now the Python-side canonical
  planner module `squander/partitioning/noisy_planner.py` rather than a new C++
  inspection surface.
- The frozen workload requirements come from the Phase 3 support matrix,
  workflow-anchor, and benchmark-minimum decisions in
  `DETAILED_PLANNING_PHASE_3.md`.
- Existing reusable workload material already exists in:
  - `benchmarks/density_matrix/circuits.py`,
  - `benchmarks/density_matrix/bridge_scope/bridge_validation.py`,
  - and the new lighter helper surface under
    `benchmarks/density_matrix/planner_surface/`.
- Story 2 may extend those surfaces or add a tightly scoped Phase 3-specific
  workload module, but it should not create multiple incompatible definitions of
  the mandatory microcases or structured families.
- The required gate family remains `U3` and `CNOT`; the required noise surface
  remains local single-qubit depolarizing, amplitude damping, and phase damping
  or dephasing.
- Story 2 closes shared planner-entry coverage across mandatory workloads. It
  does not need to decide how those workloads will later be partitioned or
  fused.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Phase 3 Workload Inventory And Identity Scheme

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names the exact workload classes it owns.
- Every mandatory workload class has stable identifiers and a deterministic
  construction policy.
- The story distinguishes mandatory workload coverage from optional exploratory
  workload growth.

**Execution checklist**
- [ ] Freeze the required 2 to 4 qubit micro-validation inventory and its case
      identifiers.
- [ ] Freeze the mandatory structured noisy family inventory for the methods
      workload slice.
- [ ] Freeze deterministic seed and construction rules for structured families.
- [ ] Freeze the required sparse, periodic, and dense local-noise placement
      patterns for the methods workload slice.

**Evidence produced**
- One stable mandatory-workload inventory for Story 2.
- One reviewable identifier and seed-policy scheme for reproducibility.

**Risks / rollback**
- Risk: if workload identity remains implicit, later evidence may mix supported
  mandatory cases with exploratory additions.
- Rollback/mitigation: define stable workload IDs and deterministic construction
  rules before implementation broadens.

### Engineering Task 2: Build Or Reuse Canonical Builders For The Required 2 To 4 Qubit Microcases

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- code | tests

**Definition of done**
- Required microcases can be materialized directly into the canonical planner
  surface.
- Microcase construction reuses existing density-matrix building surfaces where
  practical.
- Microcase canonicalization stays inside the frozen support matrix.

**Execution checklist**
- [ ] Reuse or refactor the existing microcase builders in
      `benchmarks/density_matrix/bridge_scope/bridge_validation.py` and
      `benchmarks/density_matrix/circuits.py`, with shared helper logic landing
      in `benchmarks/density_matrix/planner_surface/` when it belongs to Task 1
      broadly.
- [ ] Express microcases through
      `build_canonical_planner_surface_from_operation_specs()` or the smallest
      equivalent adapter in `squander/partitioning/noisy_planner.py` rather than
      through benchmark-local schemas.
- [ ] Keep the required microcase gate and noise vocabulary tightly bounded to
      the frozen support matrix.
- [ ] Add focused tests proving supported microcases enter the shared canonical
      planner contract.

**Evidence produced**
- Reviewable canonical-entry builders for mandatory microcases.
- Fast regression coverage for microcase representability.

**Risks / rollback**
- Risk: Story 2 may duplicate microcase logic across validation, benchmarks, and
  planner entry.
- Rollback/mitigation: centralize microcase construction behind one canonical
  builder or adapter path.

### Engineering Task 3: Add Deterministic Structured Noisy `U3` / `CNOT` Family Generators

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- code | benchmark harness

**Definition of done**
- Each required structured family can be generated reproducibly.
- The generated families stay inside the frozen `U3` / `CNOT` plus local-noise
  support surface.
- Structured-family generation is separated cleanly from runtime benchmarking.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/circuits.py` or add a Phase 3-specific
      successor under `benchmarks/density_matrix/planner_surface/` for
      structured noisy family generation.
- [ ] Encode deterministic seed rules and family identifiers directly in the
      generation surface.
- [ ] Encode the required sparse, periodic, and dense local-noise placement
      patterns as stable construction options.
- [ ] Keep larger performance-study variants outside the mandatory Story 2
      closure bar unless they reuse the same stable family IDs.

**Evidence produced**
- One stable structured-family generation surface for Story 2.
- Deterministic family instances that can be rerun and audited.

**Risks / rollback**
- Risk: ad hoc structured-family generation will make later benchmark and paper
  interpretation unstable.
- Rollback/mitigation: freeze deterministic family construction and naming at
  Story 2 time.

### Engineering Task 4: Normalize All Mandatory Workload Classes Into One Shared Planner-Entry Schema

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- code | tests

**Definition of done**
- Continuity cases, microcases, and structured families all enter one shared
  planner-entry contract.
- The planner-entry shape does not depend on whether the case came from VQE
  continuity lowering, direct microcase construction, or structured-family
  generation.
- Story 2 proves shared representability without requiring identical source
  builders.

**Execution checklist**
- [ ] Reuse the canonical planner constructors already established in
      `squander/partitioning/noisy_planner.py` instead of introducing a second
      planner-entry object model.
- [ ] Ensure the resulting planner-entry object or metadata surface shares the
      same top-level schema across workload classes.
- [ ] Add checks that mandatory workload cases do not require workload-specific
      planner semantics to be interpreted correctly.
- [ ] Keep richer audit fields or provenance expansion for Story 3 rather than
      overloading Story 2 with schema growth.

**Evidence produced**
- One shared planner-entry schema used across mandatory workload classes.
- Regression checks that compare schema-level representability across workload
  classes.

**Risks / rollback**
- Risk: if each workload class arrives through a different planner contract,
  later partitioning claims will become hard to compare or validate.
- Rollback/mitigation: enforce one shared planner-entry surface even if builders
  differ upstream.

### Engineering Task 5: Add A Story 2 Representability Matrix And Fast Regression Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 has a focused representability matrix that covers required workload
  classes and required noise-placement patterns.
- Fast regression checks catch missing workload-family coverage or schema drift.
- The regression surface remains narrower than full runtime-correctness
  validation.

**Execution checklist**
- [ ] Add a fast regression surface in a Phase 3-specific planner test file
      (for example `tests/partitioning/test_phase3_task1.py` or a tightly
      related successor).
- [ ] Cover mandatory microcases and at least one deterministic instance of each
      required structured family.
- [ ] Cover the required noise-placement patterns where they matter to workload
      identity.
- [ ] Fail quickly when a mandatory workload class can no longer reach the
      shared planner-entry contract.

**Evidence produced**
- Focused Story 2 regression coverage for workload representability.
- Reviewable failure messages for missing mandatory workload coverage.

**Risks / rollback**
- Risk: without a focused gate, Story 2 regressions may only surface later in
  broad benchmark runs.
- Rollback/mitigation: add a lightweight representability matrix that runs early
  and deterministically.

### Engineering Task 6: Emit A Stable Mandatory-Workload Planner Bundle

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 emits one stable bundle summarizing canonical planner entry across the
  mandatory workload matrix.
- The bundle records workload IDs, seed policy, noise-placement pattern, source
  type, and planner-entry summary metadata.
- The output is stable enough for later Task 1 stories and later Phase 3 tasks
  to reuse.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task1/story2_workloads/`).
- [ ] Emit machine-reviewable metadata for each required workload class and size.
- [ ] Record whether the case came from continuity lowering, direct microcase
      construction, or structured-family generation.
- [ ] Keep the bundle focused on representability and identity rather than later
      correctness or performance interpretation.

**Evidence produced**
- One stable Story 2 mandatory-workload planner bundle.
- One reusable output schema for later benchmark and paper packaging.

**Risks / rollback**
- Risk: later benchmark bundles may mix required and optional workload cases if
  Story 2 never emits its own clean inventory.
- Rollback/mitigation: emit one explicit mandatory-workload bundle before
  broader benchmarking begins.

### Engineering Task 7: Document And Run The Story 2 Shared-Contract Validation Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Story 2 notes explain how mandatory workload classes reach the shared planner
  contract.
- Fast regression checks and the Story 2 bundle run successfully.
- Story 2 closes with rerunnable evidence rather than with informal workload
  descriptions.

**Execution checklist**
- [ ] Document the mandatory-workload inventory, source labels, and seed policy.
- [ ] Run focused Story 2 regression coverage for workload representability.
- [ ] Run the Story 2 workload-bundle emission path and verify output.
- [ ] Record stable test and artifact references for Story 3 and later Phase 3
      work.

**Evidence produced**
- Passing Story 2 representability checks.
- One stable workload-bundle reference proving shared planner-contract coverage.

**Risks / rollback**
- Risk: Story 2 can appear finished while still leaving methods workloads
  under-specified and non-reproducible.
- Rollback/mitigation: require both passing checks and one stable emitted bundle
  before closure.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the required 2 to 4 qubit microcases and the mandatory structured noisy
  `U3` / `CNOT` families enter the same canonical planner-entry contract,
- deterministic workload identifiers, seed rules, and noise-placement patterns
  are frozen for the mandatory methods workload slice,
- fast regression coverage detects missing mandatory workload-family coverage,
- one stable Story 2 bundle records canonical planner entry across mandatory
  workload classes,
- and richer audit fields, optional legacy-source lowering, unsupported-case
  closure, and runtime semantics remain clearly assigned to later stories or
  later tasks.

## Implementation Notes

- Prefer extending existing density-matrix benchmark builders over creating a
  second disconnected workload zoo.
- Keep Story 2 focused on workload identity and shared representability, not on
  proving exactness or speedup.
- Treat seed policy and noise-placement vocabulary as part of the contract, not
  as incidental benchmark details.
