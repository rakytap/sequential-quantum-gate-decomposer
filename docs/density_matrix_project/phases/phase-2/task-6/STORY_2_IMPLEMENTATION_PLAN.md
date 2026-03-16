# Story 2 Implementation Plan

## Story Being Implemented

Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A
Reproducible Optimization Trace

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns the canonical Task 6 workflow contract into executable
end-to-end evidence at the two mandatory workflow-execution sizes:

- the canonical workflow is executed end to end at both 4 and 6 qubits through
  explicit `density_matrix` selection and the supported bridge plus noise path,
- one reproducible optimization trace at 4 or 6 qubits is frozen as mandatory
  training-loop evidence for the canonical workflow,
- the end-to-end and trace evidence use explicit status semantics and stable
  case identity rather than narrative-only reporting,
- and the resulting Story 2 gate stays narrow enough that Story 3 can add the
  full fixed-parameter 4 / 6 / 8 / 10 matrix and explicit 10-qubit anchor
  closure without redefining Story 2 trace semantics.

Out of scope for this story:

- fixed-parameter matrix closure across 4 / 6 / 8 / 10 qubits owned by Story 3,
- unsupported-case deterministic failure closure owned by Story 4,
- interpretation guardrails for optional, unsupported, or incomplete evidence
  owned by Story 5,
- top-level publication-ready provenance packaging owned by Story 6,
- widening the frozen support matrix, observable contract, or workflow family,
- and introducing a parallel optimizer or workflow execution harness that
  diverges from the canonical validation surfaces.

## Dependencies And Assumptions

- Story 1 now emits the canonical workflow-contract artifact through
  `benchmarks/density_matrix/workflow_evidence/workflow_contract_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_contract_bundle.json`.
  Story 2 must execute that same `workflow_id` and `contract_version`, and
  should now also reuse Story 1's emitted
  `thresholds.required_end_to_end_qubits` plus
  `input_contract.execution_modes.bounded_optimization_trace.canonical_trace_case_name`
  rather than duplicating those contract values in a second place.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already provides
  canonical execution surfaces for this story:
  - fixed-parameter workflow execution through `run_exact_regime_workflow_case()`,
  - bounded optimization traces through `run_optimization_trace()`,
  - and supported-path metadata via bridge fields.
- Task 5 already provides stable workflow and trace bundle semantics that Story 2
  should reuse where practical:
  - `task5_workflow_baseline_reference_validation.py`,
  - `trace_anchor_validation_validation.py`,
  - and artifact status language (`pass`, `fail`, `incomplete`, `completed`,
    `unsupported`).
- Existing Task 5 trace identity (`optimization_trace_4q`) provides a practical
  starting point for a canonical Story 2 trace case identity.
- Frozen phase decisions remain unchanged for this story:
  `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, `P2-ADR-013`,
  `P2-ADR-014`, and `P2-ADR-015`.
- Story 2 defines end-to-end execution readiness and reproducible trace
  availability for 4 and 6 qubits only; it does not close full exact-regime
  matrix coverage.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Story 2 End-To-End Inventory For 4q/6q Plus One Required Trace

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names one mandatory end-to-end inventory for canonical 4-qubit and
  6-qubit workflow execution.
- Story 2 freezes one mandatory reproducible optimization trace identity at 4 or
  6 qubits.
- Inventory IDs are stable enough for Story 3 to Story 6 handoff.

**Execution checklist**
- [ ] Freeze one canonical Story 2 4q end-to-end case identity and one canonical
      Story 2 6q end-to-end case identity.
- [ ] Freeze one canonical required trace case identity (4q or 6q) and mark it
      explicitly as mandatory Story 2 evidence.
- [ ] Record a stable mapping from Story 2 case IDs to the Story 1
      `phase2_xxz_hea_density_matrix_anchor_workflow` identity, `v1`
      contract version, required end-to-end qubit list, and canonical trace
      case name emitted by Story 1.
- [ ] Keep optional extra traces or exploratory end-to-end runs outside Story 2
      mandatory closure.

**Evidence produced**
- One named mandatory Story 2 end-to-end inventory for 4q and 6q.
- One named mandatory Story 2 optimization trace identity.

**Risks / rollback**
- Risk: if Story 2 case identity stays implicit, later stories cannot
  deterministically verify that the same workflow slice was executed.
- Rollback/mitigation: freeze stable case IDs and reuse them in artifacts,
  tests, and summaries.

### Engineering Task 2: Reuse Canonical Workflow And Trace Execution Surfaces Without Forking Them

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 execution is assembled from existing canonical workflow and trace
  surfaces instead of a parallel execution framework.
- Story 2 preserves existing bridge and status semantics where practical.
- The Story 2 layer remains orchestration and completeness logic, not a
  replacement for lower-level execution code.

**Execution checklist**
- [ ] Reuse `run_exact_regime_workflow_case()` for canonical fixed-parameter end-to-end
      case execution at 4 and 6 qubits.
- [ ] Reuse `run_optimization_trace()` plus `capture_case()` for canonical
      bounded training-loop trace execution.
- [ ] Reuse existing status and backend attribution fields instead of introducing
      a second status vocabulary.
- [ ] Keep any Story 2-specific wrapper minimal and focused on contract-level
      assembly.

**Evidence produced**
- One Story 2 assembly path rooted in canonical workflow and trace runners.
- Reviewable mapping from Story 2 outputs to existing execution functions.

**Risks / rollback**
- Risk: a parallel Story 2 runner can drift from canonical workflow behavior and
  produce incompatible outputs.
- Rollback/mitigation: treat existing runners as canonical and add only thin
  Task 6 orchestration around them.

### Engineering Task 3: Add Explicit Completeness And Status Rules For 4q/6q End-To-End Cases And Required Trace

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- code | tests | validation automation

**Definition of done**
- Story 2 can distinguish `complete pass`, `failed`, and `incomplete` states from
  machine-readable evidence.
- Missing mandatory 4q or 6q end-to-end cases, missing trace artifact, or
  missing status fields block Story 2 closure.
- Partial success cannot satisfy Story 2.

**Execution checklist**
- [ ] Add one completeness helper that checks mandatory Story 2 case IDs and
      required trace presence.
- [ ] Require explicit per-case status fields and one unambiguous aggregate
      Story 2 status.
- [ ] Treat missing mandatory case IDs, duplicate IDs, or missing status fields
      as incomplete evidence.
- [ ] Reject partial bundles where only one of the mandatory end-to-end sizes or
      only the trace is present.

**Evidence produced**
- Machine-readable completeness semantics for Story 2.
- Focused failure signals for missing end-to-end or trace evidence.

**Risks / rollback**
- Risk: Story 2 may appear complete from one favorable run without full required
  evidence.
- Rollback/mitigation: enforce explicit identity and status completeness checks.

### Engineering Task 4: Preserve Supported-Path Attribution And Training-Loop Metadata In Story 2 Output

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 output makes supported-path attribution explicit for both end-to-end
  cases and trace artifact.
- Trace metadata is sufficient to demonstrate training-loop relevance and
  reproducibility.
- Output preserves bridge-support and workflow-completion semantics.

**Execution checklist**
- [ ] Preserve backend identity and bridge-supported fields for mandatory Story 2
      cases.
- [ ] Preserve trace fields such as optimizer identity, parameter count,
      workflow completion, runtime, and bridge support.
- [ ] Keep required Story 2 evidence tied to the canonical workflow ID and case
      IDs.
- [ ] Ensure output distinguishes supported mandatory evidence from unsupported
      or excluded evidence.

**Evidence produced**
- Story 2 outputs with explicit supported-path attribution and trace metadata.
- Reviewable link between end-to-end cases, trace artifact, and canonical
  workflow identity.

**Risks / rollback**
- Risk: without explicit attribution, trace evidence may be misread as coming
  from a different or unsupported path.
- Rollback/mitigation: make supported-path and case identity fields mandatory in
  Story 2 outputs.

### Engineering Task 5: Add Focused Regression Tests For Story 2 End-To-End And Trace Gate Semantics

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- tests

**Definition of done**
- Fast tests cover positive Story 2 completeness and representative failure
  cases for missing mandatory evidence.
- Tests verify status semantics and case-identity invariants for Story 2.
- Regression coverage remains lightweight compared to full benchmark runs.

**Execution checklist**
- [ ] Add focused tests for mandatory case identity and status fields in
      `tests/density_matrix/test_density_matrix.py` or a tightly related
      successor.
- [ ] Add at least one negative test for missing 4q or 6q end-to-end evidence.
- [ ] Add at least one negative test for missing or malformed trace evidence.
- [ ] Keep full workflow execution in dedicated validation commands.

**Evidence produced**
- Focused regression coverage for Story 2 gate semantics.
- Reviewable failures that localize completeness and status regressions.

**Risks / rollback**
- Risk: schema drift in Story 2 may go unnoticed until later stories if no
  focused tests exist.
- Rollback/mitigation: enforce core Story 2 contract checks in fast tests.

### Engineering Task 6: Emit One Stable Story 2 End-To-End Plus Trace Artifact Bundle Or Rerunnable Command

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- benchmark harness | validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-readable bundle (or equivalent stable command)
  covering mandatory 4q/6q end-to-end evidence plus required trace evidence.
- Bundle includes stable case IDs, status semantics, and supported-path
  attribution fields.
- Artifact shape is stable enough for Story 3 to Story 6 references.

**Execution checklist**
- [ ] Add one Story 2 validation entry point (for example
      `benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py`).
- [ ] Emit one stable Story 2 artifact in a Task 6 artifact directory (for
      example `benchmarks/density_matrix/artifacts/workflow_evidence/`).
- [ ] Record generation command and provenance metadata in the artifact.
- [ ] Keep Story 2 artifact focused on end-to-end and trace evidence only.

**Evidence produced**
- One stable Story 2 end-to-end plus trace bundle or rerunnable command.
- One reusable schema for Story 3+ handoff.

**Risks / rollback**
- Risk: ad hoc Story 2 summaries can diverge and make later completeness checks
  fragile.
- Rollback/mitigation: freeze one machine-readable Story 2 bundle and command.

### Engineering Task 7: Document Story 2 Entry Points, Scope Boundaries, And Handoff To Story 3

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain Story 2 scope, rerun commands, and mandatory
  evidence IDs.
- Documentation states that Story 3 owns full 4 / 6 / 8 / 10 matrix closure and
  10-qubit anchor interpretation.
- Notes align with frozen Task 6 boundaries and avoid overclaiming completion.

**Execution checklist**
- [ ] Document Story 2 rerun command and emitted artifact location.
- [ ] Document mandatory Story 2 case IDs (4q, 6q, required trace).
- [ ] Document explicit handoff boundaries between Story 2 and Story 3.
- [ ] Keep unsupported-case and publication-bundle ownership aligned to Stories 4
      and 6.

**Evidence produced**
- Updated Story 2 developer-facing guidance and handoff notes.
- One stable documentation location for Story 2 execution scope.

**Risks / rollback**
- Risk: without clear handoff docs, Story 2 may be misread as closing full
  exact-regime matrix coverage.
- Rollback/mitigation: tie docs directly to artifact IDs and story ownership
  boundaries.

### Engineering Task 8: Run Story 2 Validation And Confirm End-To-End Plus Trace Readiness

**Implements story**
- `Story 2: The Canonical Workflow Executes End-To-End At 4 And 6 Qubits With A Reproducible Optimization Trace`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 2 tests pass.
- Story 2 artifact emission command succeeds and produces stable outputs.
- Story 2 completion is backed by rerunnable artifact evidence.

**Execution checklist**
- [ ] Run focused Story 2 regression tests.
- [ ] Run Story 2 artifact emission command and verify bundle output.
- [ ] Verify mandatory 4q/6q end-to-end and required trace entries are present
      with explicit statuses.
- [ ] Record test and artifact references for Story 3 handoff.

**Evidence produced**
- Passing Story 2 focused tests.
- Stable Story 2 artifact or command reference with mandatory entries present.

**Risks / rollback**
- Risk: Story 2 can appear complete from documentation updates alone while
  executable evidence remains missing or unstable.
- Rollback/mitigation: require successful reruns and machine-readable outputs as
  exit evidence.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- mandatory canonical end-to-end workflow cases at 4 and 6 qubits are frozen
  with stable IDs and explicit statuses,
- one mandatory reproducible optimization trace case is frozen with stable
  identity and explicit status,
- Story 2 completeness checks fail when any mandatory end-to-end or trace item
  is missing or malformed,
- Story 2 output preserves backend and supported-path attribution for mandatory
  cases,
- one stable Story 2 artifact or rerunnable command defines end-to-end plus
  trace evidence for this story,
- and full fixed-parameter matrix closure, unsupported-case closure,
  interpretation guardrails, and publication packaging remain clearly assigned to
  Stories 3 to 6.

## Implementation Notes

- Start from canonical execution surfaces in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`:
  `run_exact_regime_workflow_case()`, `capture_case()`, and `run_optimization_trace()`.
- Consume the emitted Story 1 contract artifact directly for canonical workflow
  identity, contract-version, required end-to-end qubit list, and canonical
  trace case name instead of restating those values in a second Task 6-only
  source.
- Reuse status and bundle semantics from Task 5 Story 2 and Story 3 outputs
  (`workflow_baseline_bundle.json`,
  `trace_anchor_bundle.json`, and `optimization_trace_4q.json`) where practical.
- Keep Story 2 as a contract-layer execution gate for 4q/6q plus required trace,
  not a replacement for Story 3 exact-regime matrix closure.
- Prefer one thin Story 2 bundle that references canonical lower-level fields
  rather than duplicating full raw payload structures.
