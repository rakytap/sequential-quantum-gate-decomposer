# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And
Noise Semantics

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns Task 2 descriptor semantics into positive executable runtime
semantics:

- partition-local qubit remapping during runtime stays faithful to the Task 2
  descriptor fields,
- parameter routing during runtime stays faithful to the descriptor contract
  rather than hidden runtime conventions,
- exact gate and noise order remains explicit through execution on
  boundary-sensitive supported cases,
- and Story 4 closes positive partition-local semantic stress without claiming
  full threshold validation, final result-shape packaging, or unsupported
  runtime taxonomy closure.

Out of scope for this story:

- positive continuity and shared mandatory-workload coverage already owned by
  Stories 1 and 2,
- direct descriptor-to-runtime handoff auditability already owned by Story 3,
- stable comparison-ready output packaging owned by Story 5,
- cross-workload runtime provenance stability owned by Story 6,
- runtime-stage unsupported-boundary closure owned by Story 7,
- and real fused execution, full correctness-threshold packaging, or
  performance diagnosis owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 3 already define the supported positive runtime slices and
  direct handoff contract whose semantic meaning Story 4 must preserve.
- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-004`,
  `P3-ADR-005`, and `P3-ADR-008`.
- The shared Task 2 descriptor substrate already exposes the concrete fields
  Story 4 must honor in `squander/partitioning/noisy_planner.py`, including:
  - `local_to_global_qbits`,
  - `global_to_local_qbits`,
  - `requires_remap`,
  - `parameter_routing`,
  - `canonical_operation_indices`,
  - and per-member fields such as `local_target_qbit`,
    `local_control_qbit`, `local_qubit_support`, and `local_param_start`.
- Task 2 also already established positive semantic guards through
  `validate_partition_descriptor_set()` and
  `validate_partition_descriptor_set_against_surface()`. Story 4 should consume
  those guarantees rather than re-deriving descriptor validity informally at
  runtime.
- The existing density backend already exposes the execution primitive Story 4
  should reuse through `NoisyCircuit`, `DensityMatrix`, and
  `NoisyCircuit.apply_to()`.
- The likely shared implementation substrate therefore remains:
  - the Task 3 runtime layer in `squander/partitioning/noisy_runtime.py`,
  - the exact density backend under `squander/density_matrix/`,
  - and validation helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 4 should prefer auditable partition-local execution semantics and leave
  structured unsupported taxonomy to Story 7 and full threshold closure to Task
  6.

## Engineering Tasks

### Engineering Task 1: Freeze The Positive Runtime Semantic-Preservation Rule

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit positive runtime semantic-preservation rule for
  supported partition-local execution.
- The rule distinguishes faithful execution from hidden runtime-only semantic
  assumptions.
- The rule stays narrow enough that later stories can add output, audit, and
  unsupported-boundary details cleanly.

**Execution checklist**
- [ ] Freeze the supported meaning of partition-local qubit remapping,
      parameter routing, and exact gate/noise execution semantics for Task 3.
- [ ] Define what counts as faithful use of `local_to_global_qbits`,
      `global_to_local_qbits`, `requires_remap`, and `parameter_routing`.
- [ ] Define the minimum positive Story 4 audit fields needed to review runtime
      semantic preservation.
- [ ] Keep full threshold validation and unsupported-taxonomy closure outside
      the Story 4 bar.

**Evidence produced**
- One stable Story 4 runtime semantic-preservation contract.
- One clear boundary between positive semantic stress and later correctness or
  unsupported closure.

**Risks / rollback**
- Risk: if Story 4 defines runtime semantics too loosely, later exactness claims
  will rest on ambiguous runtime meaning.
- Rollback/mitigation: freeze the positive semantic-preservation rule before
  expanding later validation and benchmarking surfaces.

### Engineering Task 2: Build Auditable Partition-Local Execution Units From Descriptor Members

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- code | tests

**Definition of done**
- Supported runtime execution derives partition-local execution units directly
  from descriptor members.
- The runtime preserves exact within-partition gate and noise order explicitly.
- Partition-local execution does not rely on hidden runtime-only semantic state.

**Execution checklist**
- [ ] Reuse ordered descriptor `members` as the source for partition-local
      execution-unit construction.
- [ ] Preserve exact within-partition gate/noise order explicitly in those
      execution units.
- [ ] Represent noise operations inside the same ordered execution structure used
      for gate operations.
- [ ] Add focused tests proving ordered partition-local execution units survive
      runtime preparation.

**Evidence produced**
- Ordered partition-local execution units rooted in descriptor membership.
- Regression coverage for explicit ordered runtime preparation.

**Risks / rollback**
- Risk: runtime execution may appear descriptor-driven while still reconstructing
  order from hidden internal state.
- Rollback/mitigation: derive ordered partition-local execution units directly
  from the emitted descriptor members.

### Engineering Task 3: Apply Qubit Remapping And Parameter Routing From Descriptor Metadata Explicitly

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- code | tests

**Definition of done**
- Supported runtime execution applies qubit remapping from explicit descriptor
  metadata.
- Supported runtime execution applies parameter routing from explicit descriptor
  metadata.
- The implementation does not depend on implicit partition-local parameter order
  or implicit qubit-layout assumptions.

**Execution checklist**
- [ ] Reuse `local_to_global_qbits`, `global_to_local_qbits`, and
      `requires_remap` or the smallest auditable successors when partition-local
      indexing differs from the canonical surface.
- [ ] Reuse `parameter_routing`, `local_param_start`, and related descriptor
      fields to map the global parameter vector into partition-local execution.
- [ ] Add focused fixtures where partition-local indexing genuinely differs from
      the canonical planner surface.
- [ ] Add focused fixtures where multi-parameter `U3` routing stresses the
      runtime parameter map.

**Evidence produced**
- One reviewable runtime remapping layer rooted in descriptor metadata.
- One reviewable runtime parameter-routing layer rooted in descriptor metadata.

**Risks / rollback**
- Risk: runtime results may look plausible while still relying on implicit local
  indexing or parameter-order assumptions.
- Rollback/mitigation: make remapping and routing auditable from descriptor
  metadata directly.

### Engineering Task 4: Keep Noise Placement First-Class During Runtime Execution

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- code | tests

**Definition of done**
- Noise operations remain first-class runtime content inside supported
  partition-local execution.
- Runtime execution does not reduce noise placement to boundary markers or side
  metadata.
- Boundary-sensitive supported cases remain auditable at the execution layer.

**Execution checklist**
- [ ] Execute noise operations through the same ordered partition-local
      execution surface used for gate operations.
- [ ] Preserve the relative position of noise with respect to neighboring gates
      inside each partition.
- [ ] Keep execution honest on supported sparse, periodic, and dense local-noise
      placements.
- [ ] Add focused checks showing that runtime semantics do not silently degrade
      noise placement to boundary-only metadata.

**Evidence produced**
- Runtime execution traces or audit records with explicit gate and noise order.
- Focused tests proving noise placement remains first-class at execution time.

**Risks / rollback**
- Risk: a runtime may appear partitioned while silently degrading noise
  semantics to boundary-only conventions.
- Rollback/mitigation: keep noise explicit inside the ordered runtime execution
  model.

### Engineering Task 5: Add Boundary-Sensitive Story 4 Runtime Audit Cases

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 includes supported cases that stress partition-local remapping,
  multi-parameter routing, and noise-adjacent boundaries.
- The audit cases are strong enough to expose silent semantic drift during
  execution.
- The audit surface remains narrower than full threshold validation.

**Execution checklist**
- [ ] Select continuity and methods-oriented fixtures where partitions begin,
      end, or remap near noise-adjacent regions.
- [ ] Add focused runtime audit checks for remapping, parameter-routing, and
      explicit noise placement on those fixtures.
- [ ] Keep the checks at the execution-semantics level rather than at final
      threshold packaging.
- [ ] Record a small set of boundary-stressing fixtures that later Task 6 work
      can reuse.

**Evidence produced**
- Boundary-sensitive runtime audit cases.
- Reviewable evidence that supported runtime execution does not silently change
  descriptor semantics.

**Risks / rollback**
- Risk: semantic bugs near partition boundaries may remain hidden if Story 4
  tests only simple partitions.
- Rollback/mitigation: include explicit remapping and noise-boundary fixtures in
  the Story 4 gate.

### Engineering Task 6: Emit A Stable Story 4 Runtime-Semantics Bundle

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable audit bundle or rerunnable checker for runtime
  semantic preservation.
- The output records remapping, parameter-routing, and ordered gate/noise
  execution information in a machine-reviewable way.
- The bundle is reusable by later correctness-validation work.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task3/story4_semantics/`).
- [ ] Emit at least one continuity case and one methods-oriented case through
      the Story 4 runtime-semantics surface.
- [ ] Record partition remap summaries, parameter-routing summaries, and ordered
      gate/noise execution traces or equivalent audit fields in the emitted
      bundle.
- [ ] Keep the bundle focused on runtime semantics rather than final threshold
      verdicts.

**Evidence produced**
- One stable Story 4 runtime-semantics bundle.
- One reusable semantic-audit surface for later Task 6 and Task 7 work.

**Risks / rollback**
- Risk: if Story 4 emits only ephemeral debug output, later validation and paper
  preparation will not have a stable runtime-semantics artifact to cite.
- Rollback/mitigation: emit one machine-reviewable bundle and keep it narrow.

### Engineering Task 7: Document And Run The Story 4 Runtime-Semantics Gate

**Implements story**
- `Story 4: Runtime Execution Preserves Partition-Local Qubit, Parameter, And Noise Semantics`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Story 4 runtime semantic rules.
- Fast regression checks and the Story 4 bundle run successfully.
- Story 4 closes with a stable review path for positive runtime semantic
  preservation.

**Execution checklist**
- [ ] Document the explicit qubit-remapping, parameter-routing, and first-class
      noise-execution rules for supported Task 3 runtime behavior.
- [ ] Explain how Story 4 differs from Task 2 descriptor semantics and from
      later threshold-validation work.
- [ ] Run focused Story 4 regression coverage and verify
      `benchmarks/density_matrix/partitioned_runtime/runtime_semantics_validation.py`.
- [ ] Record stable test and artifact references for Stories 5 through 7 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 4 runtime-semantics regression checks.
- One stable Story 4 audit-bundle or checker reference.

**Risks / rollback**
- Risk: Story 4 may appear complete while still leaving implementers unsure how
  runtime semantic preservation is reviewed consistently.
- Rollback/mitigation: document the rules and require a rerunnable semantic
  bundle.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- supported runtime execution applies descriptor-defined remapping and parameter
  routing explicitly,
- exact gate/noise order and first-class noise placement remain visible through
  supported partition-local execution,
- boundary-sensitive supported cases expose no silent semantic drift at the
  execution layer,
- one stable Story 4 runtime-semantics bundle or checker exists for later reuse,
- and final threshold validation, richer output packaging, cross-workload audit
  stability, and unsupported-boundary closure remain clearly assigned to later
  stories or tasks.

## Implementation Notes

- Prefer explicit descriptor-derived partition-local execution over runtime-only
  conventions that have to be explained after the fact.
- Keep Story 4 focused on positive runtime semantics, not on the full negative
  taxonomy.
- Treat Task 2 descriptor fields as contract inputs that the runtime must honor,
  not as hints that can be reinterpreted privately.
