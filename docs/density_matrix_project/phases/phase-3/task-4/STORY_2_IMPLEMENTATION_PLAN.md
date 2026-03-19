# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Representative Required Structured Workloads Execute A Real Fused Path
End To End

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the Task 4 fused baseline into a real positive runtime result
on the mandatory Phase 3 methods surface:

- at least one required 8 or 10 qubit structured noisy `U3` / `CNOT` workload
  executes through a real fused path,
- the fused path is inside the same noisy partitioned runtime surface established
  by Task 3 rather than a disconnected benchmark-only executor,
- the emitted result is auditable as fused execution rather than as symbolic
  fusibility or a hidden fallback to the plain Task 3 baseline,
- and Story 2 closes the first positive fused runtime slice without yet claiming
  cross-surface reuse, semantic-threshold closure, or performance closure.

Out of scope for this story:

- the descriptor-rooted eligibility contract already owned by Story 1,
- continuity and micro-validation reuse owned by Story 3,
- exact noisy-semantics preservation around fused regions owned by Story 4,
- explicit fused versus unfused versus deferred classification closure owned by
  Story 5,
- stable fused-output packaging owned by Story 6,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- Story 1 already defines the minimum descriptor-rooted eligibility rule and the
  auditable span vocabulary Story 2 must execute rather than reinterpret.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-007`, and `P3-ADR-009`.
- Task 3 already established the executable partitioned density runtime substrate
  in `squander/partitioning/noisy_runtime.py`, including:
  - `validate_runtime_request()`,
  - `execute_partitioned_density()`,
  - `execute_sequential_density_reference()`,
  - and `build_runtime_audit_record()`.
- The current Task 3 mandatory-workload runtime evidence already provides the
  baseline structured-workload surface Story 2 must extend rather than bypass in
  `benchmarks/density_matrix/artifacts/phase3_task3/story2_workloads/`.
- The structured noisy workload builders Story 2 should reuse already exist in
  `benchmarks/density_matrix/planner_surface/workloads.py`, especially:
  - `STRUCTURED_FAMILY_NAMES`,
  - `STRUCTURED_QUBITS`,
  - `MANDATORY_NOISE_PATTERNS`,
  - `build_story2_structured_descriptor_set()`,
  - and `iter_story2_structured_descriptor_sets()`.
- The exact reference and metric helpers Story 2 should reuse already exist in
  `benchmarks/density_matrix/partitioned_runtime/common.py`, especially
  `execute_partitioned_with_reference()`,
  `PHASE3_RUNTIME_DENSITY_TOL`, and `PHASE3_RUNTIME_ENERGY_TOL`.
- The density backend does not currently expose a fused `NoisyCircuit` path, but
  it does expose `DensityMatrix.apply_local_unitary()` at the C++ level. Story
  2 should therefore prefer a real fused local-unitary path on descriptor-local
  1- and 2-qubit unitary islands over inventing a second benchmark-only fusion
  engine.
- Existing state-vector fusion-related helpers in
  `squander/partitioning/partition.py`, `squander/partitioning/ilp.py`, and
  `squander/partitioning/tools.py` may inspire candidate selection or ordering,
  but Story 2 must not let them replace the descriptor-rooted fused runtime
  contract.
- Story 2 should prefer one clearly labeled fused runtime-path extension over
  multiple ad hoc fused entry points.

## Engineering Tasks

### Engineering Task 1: Freeze The First Positive Structured Fused-Runtime Slice

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names the exact required workload slice it owns.
- Story 2 defines success as real fused execution on a representative required
  structured workload, not merely symbolic fusion analysis.
- The handoff from Story 2 to Stories 3 through 7 is explicit.

**Execution checklist**
- [ ] Freeze the minimum Story 2 positive bar around at least one required 8 or
      10 qubit structured noisy workload with a Story 1 eligible span.
- [ ] Define one explicit fused runtime-path label distinct from the Task 3
      `partitioned_density_descriptor_baseline` path.
- [ ] Define what counts as real fused execution for Story 2 review.
- [ ] Keep semantic-threshold closure, explicit classification closure, and
      performance closure outside the Story 2 bar.

**Evidence produced**
- One stable Story 2 structured fused-runtime contract description.
- One reviewable definition of the first positive fused path.

**Risks / rollback**
- Risk: Story 2 may close on symbolic fusibility or on a benchmark-only shortcut
  rather than on a real fused runtime result.
- Rollback/mitigation: freeze the positive fused-runtime bar before wiring the
  execution path.

### Engineering Task 2: Extend The Task 3 Runtime With One Real Fused Execution Branch

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- code | tests

**Definition of done**
- The Task 3 runtime layer gains one explicit fused execution branch rooted in
  the Story 1 eligibility surface.
- The fused branch executes inside the same supported noisy partitioned runtime
  surface rather than through a disconnected harness-only executor.
- The unfused baseline remains available and clearly distinguishable.

**Execution checklist**
- [ ] Extend `squander/partitioning/noisy_runtime.py` with one fused-capable
      execution entry point or one auditable fused execution mode inside the
      existing entry point.
- [ ] Reuse Story 1 eligible span summaries rather than recomputing fusibility
      privately inside benchmark code.
- [ ] Keep fused and unfused runtime paths explicitly distinguishable in emitted
      results.
- [ ] Preserve the Task 3 runtime as the shared implementation substrate instead
      of moving fused execution into scripts alone.

**Evidence produced**
- One reviewable fused runtime branch rooted in the Task 3 runtime layer.
- Focused regression coverage proving the fused branch is distinct from the
  plain baseline.

**Risks / rollback**
- Risk: a fused branch implemented only in a benchmark script will be hard to
  audit and easy to mislabel as product runtime behavior.
- Rollback/mitigation: make the fused path live in the shared runtime layer.

### Engineering Task 3: Lower Eligible Unitary-Island Spans Into Real Fused Runtime Work

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- code | tests

**Definition of done**
- Eligible descriptor-local unitary-island spans are lowered into real fused
  runtime work.
- Non-eligible regions remain on the supported unfused Task 3 path.
- The fused lowering stays auditable back to descriptor membership and
  partition-local metadata.

**Execution checklist**
- [ ] Add one lowering path from eligible descriptor spans to fused runtime work,
      such as one fused local unitary kernel applied through the shared density
      backend or the smallest auditable successor.
- [ ] Keep unfused descriptor members on the plain Task 3 execution path inside
      the same workload.
- [ ] Preserve span-level links back to canonical operation indices and
      partition-local qubit or parameter metadata.
- [ ] Add focused tests proving that actual fused execution occurs for at least
      one eligible structured span.

**Evidence produced**
- One reviewable eligible-span-to-fused-runtime lowering path.
- Focused tests tying real fused execution to descriptor-local spans.

**Risks / rollback**
- Risk: the runtime may report fused coverage while still executing the same
  work entirely through the unfused baseline.
- Rollback/mitigation: require one auditable lowering path from eligible spans
  to real fused runtime work.

### Engineering Task 4: Select Representative Required Structured Fixtures With Real Fused Coverage

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 uses representative required structured workloads that actually
  contain eligible fused spans.
- The fixture set is broad enough to support later performance and semantics
  work.
- The fixture set remains smaller than the full Story 7 benchmark package.

**Execution checklist**
- [ ] Reuse `iter_story2_structured_descriptor_sets()` as the primary source for
      required structured fixtures.
- [ ] Select at least one representative required 8 qubit case and one
      representative required 10 qubit case for the first fused positive path.
- [ ] Prefer fixtures whose sparse, periodic, or dense noise placement still
      leaves at least one clear eligible unitary island.
- [ ] Record which family, size, and noise pattern each Story 2 fused fixture
      represents.

**Evidence produced**
- One representative Story 2 structured fused-fixture set.
- One clear seed, family, and noise-pattern inventory for later reuse.

**Risks / rollback**
- Risk: Story 2 may close on a too-artificial fixture that does not represent
  the required structured workload surface.
- Rollback/mitigation: anchor the first fused path on actual required structured
  families and sizes.

### Engineering Task 5: Add A Focused Story 2 Structured Fused-Runtime Validation Gate

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 has a rerunnable validation layer dedicated to the first positive
  structured fused path.
- The validator checks real fused-path labeling and basic exactness against the
  sequential reference.
- The gate remains narrower than the later full benchmark and threshold package.

**Execution checklist**
- [ ] Add focused Task 4 regression coverage in
      `tests/partitioning/test_phase3_task4.py`.
- [ ] Add a Story 2 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `structured_fused_runtime_validation.py` as the primary checker.
- [ ] Assert actual fused-path labeling, no silent fallback to the plain Task 3
      baseline, and exact-output presence on supported Story 2 fixtures.
- [ ] Reuse the sequential reference path to check that the first positive fused
      slice remains exact enough for later Story 4 work.

**Evidence produced**
- One rerunnable Story 2 structured fused-runtime validation surface.
- Fast regression coverage for the first positive Task 4 runtime slice.

**Risks / rollback**
- Risk: Story 2 may close with only manual inspection and no repeatable positive
  fused-runtime gate.
- Rollback/mitigation: require one dedicated validator before closure.

### Engineering Task 6: Emit A Stable Story 2 Structured Fused-Runtime Artifact Bundle

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable bundle or rerunnable checker for
  the first structured fused-runtime slice.
- The artifact proves real fused-path execution on representative required
  structured workloads.
- The artifact shape is stable enough for later semantic and benchmark work to
  extend.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task4/story2_structured_fused_runtime/`).
- [ ] Record case provenance, fused runtime-path label, partition summaries,
      eligible-span references, fused-span summaries, and exactness metrics for
      Story 2 fixtures.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the bundle focused on the first positive structured fused path rather
      than on full performance interpretation.

**Evidence produced**
- One stable Story 2 structured fused-runtime bundle.
- One reusable positive fused-runtime surface for later Task 4 stories.

**Risks / rollback**
- Risk: a prose-only Story 2 closure would make the first positive fused result
  hard to cite and easy to overstate later.
- Rollback/mitigation: emit one thin machine-reviewable bundle early.

### Engineering Task 7: Document The Story 2 Handoff To Later Task 4 Work

**Implements story**
- `Story 2: Representative Required Structured Workloads Execute A Real Fused Path End To End`

**Change type**
- docs

**Definition of done**
- Story 2 notes explain exactly what the first positive fused-runtime slice
  closes.
- The structured fused-runtime bundle is documented as the first real fused
  path, not as the whole Task 4 contract.
- Developer-facing notes point to the Story 2 validation and artifact path.

**Execution checklist**
- [ ] Document the supported structured fused-runtime slice and its evidence
      surface.
- [ ] Explain that continuity and micro-validation reuse belong to Story 3.
- [ ] Explain that semantic thresholds, explicit classification, stable output
      packaging, and threshold-or-diagnosis benchmarking belong to Stories 4
      through 7.
- [ ] Record stable references to the Story 2 tests and emitted bundle.

**Evidence produced**
- Updated developer-facing notes for the Story 2 structured fused-runtime gate.
- One stable handoff reference for later Task 4 work.

**Risks / rollback**
- Risk: later Task 4 work may assume Story 2 already closed semantic thresholds,
  shared reuse, or performance claims.
- Rollback/mitigation: document the Story 2 handoff boundaries explicitly.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- at least one representative required structured workload executes through one
  real fused path inside the shared noisy partitioned runtime,
- the fused path is auditable back to Story 1 eligible descriptor spans,
- emitted results clearly distinguish the fused runtime path from the plain Task
  3 baseline path,
- one stable Story 2 structured fused-runtime validator or artifact bundle
  proves the first positive fused slice,
- and continuity reuse, semantic-threshold closure, explicit classification, and
  performance interpretation remain clearly assigned to later stories.

## Implementation Notes

- Prefer one clearly labeled fused runtime-path extension over multiple
  benchmark-local fused labels.
- Keep the first positive fused path small and auditable before trying to make it
  broad.
- Treat representative structured workloads as the primary Task 4 positive
  surface for Paper 2.
