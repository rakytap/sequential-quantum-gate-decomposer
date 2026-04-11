# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And
Exposed Auditable At Runtime

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the minimum Phase 3 fused baseline into one explicit,
descriptor-rooted eligibility surface:

- eligible fused regions are defined against ordered
  `NoisyPartitionDescriptorMember` spans rather than against opaque planner-only
  internals,
- the minimum positive fused baseline is conservative unitary-island fusion
  inside noisy partitions,
- eligibility outcomes are auditable in runtime-facing records before broader
  fused execution or performance claims are made,
- and Story 1 closes the contract for "what counts as eligible" without yet
  claiming that every eligible span is executed through the real fused path.

Out of scope for this story:

- positive fused execution on representative structured workloads owned by Story
  2,
- continuity and micro-validation surface reuse owned by Story 3,
- exact noisy-semantics preservation owned by Story 4,
- fused versus unfused versus deferred classification closure owned by Story 5,
- stable fused output and provenance packaging owned by Story 6,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-004`,
  `P3-ADR-005`, and `P3-ADR-007`.
- Task 2 already established the descriptor substrate Story 1 must use in
  `squander/partitioning/noisy_planner.py`, including:
  - `NoisyPartitionDescriptorSet`,
  - `NoisyPartitionDescriptor`,
  - `NoisyPartitionDescriptorMember`,
  - ordered `members`,
  - `canonical_operation_indices`,
  - `local_to_global_qbits`,
  - `global_to_local_qbits`,
  - `requires_remap`,
  - and `parameter_routing`.
- Task 3 already established the executable runtime surface Story 1 must extend
  in `squander/partitioning/noisy_runtime.py`, including:
  - `validate_runtime_request()`,
  - `execute_partitioned_density()`,
  - `NoisyRuntimePartitionRecord`,
  - `NoisyRuntimeExecutionResult`,
  - and `build_runtime_audit_record()`.
- The existing Task 3 runtime artifacts already provide the shared positive audit
  surface Story 1 should align with:
  - `benchmarks/density_matrix/artifacts/partitioned_runtime/story3_handoff/`,
  - `benchmarks/density_matrix/artifacts/partitioned_runtime/story4_semantics/`,
  - and `benchmarks/density_matrix/artifacts/partitioned_runtime/story6_audit/`.
- The required workload builders Story 1 should use for representative eligibility
  inspection already exist in:
  - `build_phase3_continuity_partition_descriptor_set()` in
    `squander/partitioning/noisy_planner.py`,
  - `iter_story2_microcase_descriptor_sets()` in
    `benchmarks/density_matrix/planner_surface/workloads.py`,
  - and `iter_story2_structured_descriptor_sets()` in the same module.
- The existing state-vector partitioning and fusion stack under
  `squander/partitioning/partition.py`, `squander/partitioning/ilp.py`, and
  `squander/partitioning/tools.py` provides useful design prior art about
  `ilp-fusion`, `ilp-fusion-ca`, and `qiskit-fusion`, but Story 1 must not let
  those planner-oriented helpers become the contract-defining runtime truth.
- Story 1 should prefer one explicit descriptor-local eligibility rule and one
  auditable evidence vocabulary rather than multiple workload-local or strategy-
  local interpretations of fusibility.

## Engineering Tasks

### Engineering Task 1: Freeze The Descriptor-Local Fusion Eligibility Rule

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 defines one explicit descriptor-local eligibility rule for the minimum
  fused baseline.
- The rule is concrete enough to distinguish eligible spans from merely
  interesting spans.
- The rule stays narrow enough that later stories can add execution,
  classification, and benchmark behavior cleanly.

**Execution checklist**
- [ ] Freeze the minimum positive fused baseline as unitary-island fusion inside
      noisy partitions.
- [ ] Define what makes a span eligible: contiguous descriptor membership,
      supported `U3` / `CNOT` unitary-only content, one supported partition
      context, auditable remapping and parameter routing, and explicit noise or
      partition boundaries.
- [ ] Define what makes a span ineligible or deferred for the minimum Phase 3
      claim.
- [ ] Keep real fused execution, semantic thresholds, and benchmark closure
      outside the Story 1 bar.

**Evidence produced**
- One stable Story 1 fusion-eligibility rule.
- One clear boundary between eligibility definition and later fused execution
  closure.

**Risks / rollback**
- Risk: if eligibility remains informal, later fused-path claims will rest on
  ad hoc examples rather than on one contract-defining rule.
- Rollback/mitigation: freeze the minimum eligibility rule before building the
  positive fused path.

### Engineering Task 2: Build A Descriptor-Rooted Eligibility Classifier

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- code | tests

**Definition of done**
- Story 1 exposes one deterministic classifier for descriptor-local fused
  candidates.
- The classifier operates on Task 2 descriptor content rather than on hidden
  planner-only state.
- The classifier produces auditable span-level outputs that later stories can
  reuse directly.

**Execution checklist**
- [ ] Implement one eligibility-analysis surface in
      `squander/partitioning/noisy_runtime.py` or the smallest adjacent runtime
      helper.
- [ ] Derive fused-candidate spans from ordered descriptor `members`,
      `canonical_operation_indices`, and partition-local metadata.
- [ ] Keep the classifier deterministic for a fixed descriptor set.
- [ ] Avoid workload-specific allowlists or benchmark-only manual span tagging.

**Evidence produced**
- One reviewable descriptor-rooted fusion-eligibility classifier.
- Focused regression coverage for span-level classification determinism.

**Risks / rollback**
- Risk: a planner-oriented or workload-specific classifier can drift from the
  runtime contract and become hard to audit later.
- Rollback/mitigation: root the classifier in the emitted descriptor contract.

### Engineering Task 3: Expose Eligibility Outcomes Through The Shared Runtime Audit Surface

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- code | docs

**Definition of done**
- Story 1 eligibility outcomes fit alongside the existing Task 3 runtime audit
  surface where fields overlap.
- Eligibility evidence is machine-reviewable and comparable across workloads.
- Story 1 does not create a disconnected debug-only format for fusibility.

**Execution checklist**
- [ ] Define one eligibility summary shape that can sit beside the existing Task
      3 runtime provenance and partition summaries.
- [ ] Reuse overlapping provenance fields from `build_runtime_audit_record()`
      where they already match the Story 1 review surface.
- [ ] Add eligibility-specific fields such as candidate count, eligible span
      summaries, and ineligibility reasons without replacing the Task 3 audit
      vocabulary.
- [ ] Document how the Story 1 eligibility surface relates to later fused
      execution records.

**Evidence produced**
- One aligned Story 1 eligibility-audit record shape.
- One explicit mapping between Task 3 runtime audit fields and Story 1
  eligibility extensions.

**Risks / rollback**
- Risk: eligibility evidence may become structurally incomparable to the runtime
  evidence it is supposed to support.
- Rollback/mitigation: align overlapping fields with the Task 3 audit surface
  from the start.

### Engineering Task 4: Add A Representative Eligibility Matrix Across Mandatory Workload Classes

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 covers representative continuity, microcase, and structured workloads
  for eligibility inspection.
- The matrix is broad enough to show the contract is shared across workload
  classes.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Select at least one continuity descriptor set, one microcase descriptor
      set, and one structured-family descriptor set for eligibility inspection.
- [ ] Include at least one case with a clearly eligible unitary island.
- [ ] Include at least one case where explicit noise boundaries block or split a
      larger candidate.
- [ ] Keep the matrix representative and phase-contract-driven rather than
      enumerating every possible span shape.

**Evidence produced**
- One representative Story 1 eligibility matrix across mandatory workload
  classes.
- Reviewable fixtures for later fused execution and semantic validation work.

**Risks / rollback**
- Risk: eligibility rules may look coherent on one hand-picked case but fail to
  generalize across the actual Phase 3 workload surface.
- Rollback/mitigation: freeze a small but workload-spanning eligibility matrix.

### Engineering Task 5: Add Fast Story 1 Regression Coverage For Eligibility Stability

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- tests

**Definition of done**
- Story 1 has focused regression checks for eligibility stability.
- The checks prove eligibility outcomes are descriptor-rooted and repeatable.
- The regression slice remains narrower than later execution and threshold
  validation.

**Execution checklist**
- [ ] Add a dedicated Task 4 regression surface in
      `tests/partitioning/test_partitioned_runtime.py`.
- [ ] Assert that representative cases produce stable eligible-span summaries and
      stable ineligibility reasons.
- [ ] Assert that explicit noise boundaries split or block fused spans when the
      contract says they should.
- [ ] Keep the checks at the eligibility layer rather than prematurely requiring
      real fused execution.

**Evidence produced**
- Fast regression coverage for Story 1 eligibility stability.
- One repeatable test surface for later Task 4 work to extend.

**Risks / rollback**
- Risk: eligibility drift may remain hidden until benchmark or paper packaging.
- Rollback/mitigation: add a dedicated fast eligibility regression slice early.

### Engineering Task 6: Emit A Stable Story 1 Eligibility Bundle Or Rerunnable Checker

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable eligibility bundle or rerunnable
  checker.
- The bundle records representative eligible and ineligible spans with enough
  provenance to audit them later.
- The output shape is stable enough for Stories 2 through 7 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 1 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_eligibility_validation.py` as the primary checker.
- [ ] Add a dedicated Story 1 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/fused_eligibility/`).
- [ ] Emit case-level provenance, partition summaries, eligible-span summaries,
      and ineligibility reasons for representative workloads.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 1 eligibility bundle or checker.
- One reusable evidence surface for later fused execution work.

**Risks / rollback**
- Risk: prose-only eligibility closure will make later fused-path decisions hard
  to justify and easy to overstate.
- Rollback/mitigation: emit one thin machine-reviewable eligibility surface
  early.

### Engineering Task 7: Document The Story 1 Handoff To Later Task 4 Stories

**Implements story**
- `Story 1: Fusion Eligibility Is Defined Against The Descriptor Contract And Exposed Auditable At Runtime`

**Change type**
- docs

**Definition of done**
- Story 1 notes explain exactly what the eligibility surface closes.
- The eligibility bundle is documented as the contract for what may fuse, not
  as proof that a real fused path already exists.
- Developer-facing notes point to the Story 1 tests and artifact location.

**Execution checklist**
- [ ] Document the supported descriptor-local eligibility rule and its evidence
      surface.
- [ ] Explain that real fused execution belongs to Story 2.
- [ ] Explain that semantic thresholds, explicit classification, stable output
      packaging, and threshold-or-diagnosis benchmarking belong to Stories 4
      through 7.
- [ ] Record stable references to the Story 1 tests and emitted bundle.

**Evidence produced**
- Updated developer-facing notes for the Story 1 eligibility gate.
- One stable handoff reference for later Task 4 implementation work.

**Risks / rollback**
- Risk: later Task 4 work may over-assume Story 1 already proved real fused
  execution or benchmark value.
- Rollback/mitigation: document the handoff boundaries explicitly.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one explicit descriptor-rooted eligibility rule exists for the minimum Task 4
  fused baseline,
- representative continuity, microcase, and structured workloads can be
  inspected through that same eligibility surface,
- eligibility evidence is machine-reviewable and aligned with the Task 3 runtime
  audit vocabulary where fields overlap,
- one stable Story 1 eligibility bundle or rerunnable checker exists for later
  reuse,
- and real fused execution, semantic threshold closure, classification closure,
  and performance interpretation remain clearly assigned to later stories.

## Implementation Notes

- Prefer descriptor-rooted span analysis over planner-only fusion hints that
  would have to be justified after the fact.
- Treat explicit noise members as hard boundaries unless a separately validated
  exact rule is introduced later.
- Keep Story 1 focused on defining eligibility, not on proving acceleration.
