# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Mandatory Microcases And Structured Families Share The Same Executable
Runtime Surface

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the first positive continuity runtime slice into a shared
mandatory-workload runtime surface:

- the required 2 to 4 qubit micro-validation cases execute through the same
  partitioned runtime contract as the continuity anchor,
- the mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families execute
  through that same runtime surface,
- runtime behavior remains bounded to the frozen support matrix and workload
  inventory rather than widening into new benchmark families,
- and Story 2 closes shared positive workload coverage without claiming fused
  execution, final correctness thresholds, or performance conclusions.

Out of scope for this story:

- the positive continuity-runtime closure already owned by Story 1,
- direct descriptor-to-runtime audit stabilization owned by Story 3,
- partition-local semantic stress closure owned by Story 4,
- stable comparison-ready output packaging owned by Story 5,
- cross-workload runtime provenance stability owned by Story 6,
- runtime-stage unsupported-boundary closure owned by Story 7,
- and real fused execution, calibrated planning heuristics, or full performance
  packaging owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Story 1 already defines the first supported Task 3 runtime path on the
  continuity anchor, which Story 2 should widen rather than replace.
- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-007`, `P3-ADR-008`, and `P3-ADR-009`.
- Task 2 already established the mandatory workload descriptor substrate
  through:
  - `iter_story2_microcase_descriptor_sets()` and
    `iter_story2_structured_descriptor_sets()` in
    `benchmarks/density_matrix/planner_surface/workloads.py`,
  - the emitted bundle under
    `benchmarks/density_matrix/artifacts/planner_surface/mandatory_workload/`,
  - and the Task 2 shared schema checks in
    `tests/partitioning/test_planner_surface_descriptors.py`.
- The micro-validation inventory in `benchmarks/density_matrix/circuits.py`
  and the Aer comparison helper in
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already provide
  the right identity layer for the required small cases. Story 2 should reuse
  those identities where practical rather than inventing new microcase names.
- The existing density backend already exposes the exact sequential execution
  primitive for supported gate and noise families through `NoisyCircuit`,
  `DensityMatrix`, and `NoisyCircuit.apply_to()`.
- The likely shared implementation substrate therefore remains:
  - the Task 2 descriptor layer in `squander/partitioning/noisy_planner.py`,
  - the shared Task 3 runtime adapter in `squander/partitioning/noisy_runtime.py`,
  - the Story 1 continuity validator in
    `benchmarks/density_matrix/partitioned_runtime/continuity_runtime_validation.py`,
  - and benchmark or validation helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 2 should widen supported runtime coverage across mandatory workload
  classes, not create workload-local execution contracts or per-benchmark
  runtime semantics.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory Runtime Workload Inventory And Shared Closure Boundary

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one required runtime workload inventory across microcases and
  structured families.
- Story 2 defines shared runtime coverage as one common supported execution
  surface rather than separate benchmark-local adapters.
- The handoff from Story 2 to Stories 3 through 7 is explicit.

**Execution checklist**
- [ ] Freeze the required Task 3 runtime workload inventory around:
      required 2 to 4 qubit microcases, required 8 and 10 qubit structured
      families, and the frozen continuity anchor already owned by Story 1.
- [ ] Define what counts as successful Story 2 shared runtime coverage: one
      executable partitioned runtime contract across all mandatory workload
      classes.
- [ ] Keep broader workload growth, optional families, and full benchmark
      interpretation explicitly outside the Story 2 bar.
- [ ] Record the boundary between Story 2 shared workload coverage and later
      output, audit, and unsupported-boundary work.

**Evidence produced**
- One stable Story 2 workload inventory for the mandatory runtime surface.
- One reviewable boundary between positive workload coverage and later runtime
  concerns.

**Risks / rollback**
- Risk: if Story 2 leaves the mandatory runtime inventory loose, later benchmark
  and paper claims will be harder to audit consistently.
- Rollback/mitigation: freeze the mandatory runtime workload set before widening
  execution coverage.

### Engineering Task 2: Reuse The Task 2 Mandatory Descriptor Builders As The Shared Runtime Inputs

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- code | tests

**Definition of done**
- Required microcases and structured families reach Task 3 runtime execution
  from the existing Task 2 descriptor builders.
- The implementation does not introduce workload-specific runtime-only schemas.
- The supported runtime input surface remains visibly rooted in the accepted
  Task 2 descriptor contract.

**Execution checklist**
- [ ] Reuse `iter_story2_microcase_descriptor_sets()` and
      `iter_story2_structured_descriptor_sets()` or the smallest shared
      successors as the runtime inputs for Story 2.
- [ ] Route those descriptor sets through the same runtime entry point
      established by Story 1.
- [ ] Avoid benchmark-only adapters that bypass the shared descriptor contract
      for microcases or structured families.
- [ ] Keep execution deterministic on the fixed seed rules and workload IDs
      already frozen by Task 2.

**Evidence produced**
- One reviewable descriptor-to-runtime routing path for mandatory microcases.
- One reviewable descriptor-to-runtime routing path for mandatory structured
  families.

**Risks / rollback**
- Risk: workload-specific runtime wrappers can silently create multiple
  incompatible "supported" runtime contracts.
- Rollback/mitigation: reuse the existing Task 2 descriptor builders as the
  shared runtime inputs.

### Engineering Task 3: Extend The Story 1 Runtime Adapter To Cover Mandatory Methods Workloads

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- code | tests

**Definition of done**
- Story 1 continuity execution and Story 2 methods execution share one runtime
  entry surface.
- The runtime accepts supported microcase and structured-family descriptors
  without widening the frozen support matrix.
- The workload class does not change the meaning of supported partitioned
  execution.

**Execution checklist**
- [ ] Extend the Story 1 runtime adapter in
      `squander/partitioning/noisy_runtime.py` to accept required microcase
      descriptors.
- [ ] Extend that same adapter to accept required structured-family descriptors
      at 8 and 10 qubits.
- [ ] Keep support bounded to the frozen `U3` / `CNOT` plus required local-noise
      surface.
- [ ] Keep real fused execution, unsupported-boundary logic, and summary
      benchmark packaging out of the Story 2 implementation bar.

**Evidence produced**
- One shared runtime adapter covering continuity, microcase, and structured
  workloads.
- Focused regression coverage for the widened positive runtime surface.

**Risks / rollback**
- Risk: Story 2 may appear complete while continuity, microcase, and structured
  cases still rely on subtly different runtime semantics.
- Rollback/mitigation: widen the same runtime entry surface rather than adding
  workload-local paths.

### Engineering Task 4: Cross-Check Shared Runtime Semantics Across Workload Classes

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- tests

**Definition of done**
- Shared runtime semantics apply equally to continuity, microcase, and
  structured workloads.
- Runtime-path labels and supported execution status do not become
  workload-specific conventions.
- Schema drift in the positive runtime surface is caught early.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_partitioned_runtime.py` for shared
      runtime semantics across workload classes.
- [ ] Compare requested mode, runtime-path labels, partition-count summaries,
      and exact-output presence across continuity, microcase, and structured
      cases.
- [ ] Keep the checks narrow to shared runtime coverage rather than later
      correctness thresholds or unsupported-boundary logic.
- [ ] Fail quickly when supported runtime labeling differs across workload
      classes.

**Evidence produced**
- Fast regression coverage for shared runtime semantics.
- Reviewable cross-workload checks for the positive Task 3 runtime surface.

**Risks / rollback**
- Risk: runtime semantics may drift subtly across workload classes while still
  looking locally reasonable.
- Rollback/mitigation: cross-check the same positive rules on multiple required
  workload types.

### Engineering Task 5: Add A Focused Story 2 Mandatory-Workload Runtime Validation Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 has a rerunnable validation layer dedicated to mandatory shared
  runtime coverage.
- The validation layer covers required microcases and required structured
  families in addition to the Story 1 continuity path.
- The gate remains narrower than later output-shape, audit, and unsupported
  closure work.

**Execution checklist**
- [ ] Add a dedicated mandatory-workload runtime validation entry point under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `mandatory_workload_runtime_validation.py` as the primary Story 2
      validator.
- [ ] Check at least one required microcase at each mandated small-qubit scale.
- [ ] Check at least one required instance from each mandatory structured family
      at each required structured size.
- [ ] Keep the Story 2 gate focused on shared positive runtime coverage rather
      than on final benchmark reporting.

**Evidence produced**
- One rerunnable Story 2 mandatory-workload runtime validation surface.
- Fast regression coverage for shared positive runtime coverage.

**Risks / rollback**
- Risk: Story 2 may close with broad prose claims but no rerunnable proof that
  mandatory methods workloads actually share the runtime surface.
- Rollback/mitigation: require one dedicated shared-workload validation gate.

### Engineering Task 6: Emit A Stable Story 2 Mandatory-Workload Runtime Bundle

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable bundle or rerunnable checker for
  shared mandatory runtime coverage.
- The output records supported cases across the required microcase and
  structured-workload surface.
- The artifact shape is reusable by later Task 3 stories.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/mandatory_workload/`).
- [ ] Emit at least one supported microcase and one supported structured-family
      case through the shared runtime surface.
- [ ] Record workload identity, requested mode, runtime-path label, partition
      summary, and exact-output presence in the emitted bundle.
- [ ] Record rerun commands and software metadata with the bundle.

**Evidence produced**
- One stable Story 2 mandatory-workload runtime bundle or checker.
- One reusable positive runtime output shape for later Task 3 work.

**Risks / rollback**
- Risk: if Story 2 emits only ad hoc local artifacts, later audit and paper
  work will still lack one shared positive runtime surface to cite.
- Rollback/mitigation: emit one stable shared bundle early and extend it later.

### Engineering Task 7: Document And Run The Story 2 Shared-Runtime Gate

**Implements story**
- `Story 2: Mandatory Microcases And Structured Families Share The Same Executable Runtime Surface`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Story 2 workload surface.
- Fast regression checks and the Story 2 bundle run successfully.
- Story 2 closes with a stable review path for shared mandatory runtime
  coverage.

**Execution checklist**
- [ ] Document the required microcase and structured-family runtime surface.
- [ ] Explain how Story 2 extends Story 1 continuity runtime coverage without
      changing the runtime contract.
- [ ] Run focused shared-workload runtime checks and verify the emitted bundle.
- [ ] Record stable test and artifact references for Stories 3 through 7 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 2 shared-runtime regression checks.
- One stable Story 2 runtime-bundle or checker reference.

**Risks / rollback**
- Risk: Story 2 may appear complete while still leaving contributors unsure how
  the mandatory methods workloads are supposed to share the Task 3 runtime
  contract.
- Rollback/mitigation: document the shared runtime surface and require a
  rerunnable bundle.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- required microcases and mandatory structured noisy families execute through
  the same supported partitioned runtime surface as the continuity anchor,
- the positive runtime surface remains bounded to the frozen support matrix and
  workload inventory,
- shared runtime labels and result-presence expectations are stable across the
  mandatory workload classes,
- one stable Story 2 validation command or bundle proves shared positive
  runtime coverage,
- and direct handoff auditability, semantic stress, richer output packaging,
  audit stability, and unsupported-boundary closure remain clearly assigned to
  later stories.

## Implementation Notes

- Prefer widening the Story 1 runtime adapter over building workload-local Task
  3 execution paths.
- Reuse the exact workload IDs, seeds, and noise-pattern labels already frozen
  by Task 2 wherever practical.
- Treat Story 2 as the point where Task 3 becomes a shared runtime surface, not
  as the point where fused execution or final benchmark claims are closed.
