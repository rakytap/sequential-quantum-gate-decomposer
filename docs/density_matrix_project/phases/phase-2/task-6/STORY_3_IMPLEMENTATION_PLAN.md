# Story 3 Implementation Plan

## Story Being Implemented

Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10
Qubits With Exact-Regime Pass/Fail Interpretation

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns the canonical Task 6 workflow into a fixed-parameter
exact-regime matrix with explicit pass/fail semantics:

- the mandatory fixed-parameter matrix is frozen at 4, 6, 8, and 10 qubits,
- each mandatory size carries at least 10 fixed parameter vectors with stable
  parameter-set identities,
- matrix results are interpreted with explicit exactness and completion semantics
  rather than favorable-subset summaries,
- at least one documented 10-qubit anchor evaluation case is mandatory and
  explicitly linked to the canonical workflow identity,
- and Story 3 remains a matrix-closure layer so Story 4 to Story 6 can close
  unsupported boundaries, interpretation guardrails, and publication bundle
  integrity without redefining matrix semantics.

Out of scope for this story:

- the 4/6 end-to-end plus reproducible optimization-trace closure owned by
  Story 2,
- deterministic unsupported-case failure closure owned by Story 4,
- optional/unsupported/incomplete interpretation closure owned by Story 5,
- publication-ready top-level bundle assembly owned by Story 6,
- widening the frozen workflow family, support matrix, or observable contract,
- and runtime speed thresholds or acceleration claims as matrix completion gates.

## Dependencies And Assumptions

- Story 1 now emits the canonical workflow-contract artifact through
  `benchmarks/density_matrix/workflow_evidence/workflow_contract_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_contract_bundle.json`.
  Story 3 matrix evidence must reference that emitted `workflow_id` and
  `contract_version` directly, and should now also reuse Story 1's emitted
  `thresholds.required_workflow_qubits`,
  `thresholds.fixed_parameter_sets_per_size`, and
  `thresholds.documented_anchor_qubit` where practical.
- Story 2 now emits
  `benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/end_to_end_trace_bundle.json`.
  Story 3 should treat that bundle as the canonical 4q/6q end-to-end plus trace
  reference and should not re-own trace semantics.
- Canonical matrix execution surfaces already exist in
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py`:
  `build_exact_regime_parameter_sets()`, `run_exact_regime_workflow_matrix()`, and
  `build_exact_regime_workflow_bundle()`.
- Task 5 Story 2 already provides a mature matrix-closure pattern in
  `benchmarks/density_matrix/validation_evidence/workflow_baseline_validation.py`,
  including stable case IDs, stable parameter-set IDs, required case accounting,
  and mandatory pass-rate semantics.
- Frozen exact-regime and numeric decisions remain unchanged for this story:
  `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`.
- Story 3 should provide one matrix-level gate for Task 6; it should not reopen
  support-tier semantics or unsupported-case taxonomies handled elsewhere.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Story 3 Fixed-Parameter Matrix Inventory

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 names one canonical mandatory matrix inventory for 4/6/8/10 qubits.
- Parameter-set identity is frozen with at least 10 sets per required size.
- Mandatory matrix case IDs are stable and reusable by later stories.

**Execution checklist**
- [ ] Freeze one canonical mandatory qubit-size list: 4, 6, 8, 10.
- [ ] Freeze one canonical fixed-parameter-set inventory with at least 10 stable
      IDs per mandatory size.
- [ ] Freeze deterministic case naming that combines qubit size and
      parameter-set ID.
- [ ] Keep every Story 3 case explicitly tied to the emitted Story 1 workflow
      identity and threshold inventory rather than to implicit matrix-only
      naming.
- [ ] Keep optional exploratory matrix variants outside Story 3 mandatory
      closure.

**Evidence produced**
- One stable mandatory Story 3 matrix inventory with case IDs.
- One stable mandatory parameter-set inventory with IDs.

**Risks / rollback**
- Risk: implicit case inventories allow silent matrix shrinkage while still
  reporting apparent success.
- Rollback/mitigation: freeze explicit case and parameter-set IDs and validate
  against them.

### Engineering Task 2: Reuse Canonical Workflow-Matrix Harness Without Forking Execution Logic

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- code | validation automation

**Definition of done**
- Story 3 matrix execution reuses canonical matrix harness surfaces rather than a
  parallel workflow runner.
- Story 3 preserves existing matrix field semantics where practical.
- The Story 3 layer remains a contract-focused matrix gate.

**Execution checklist**
- [ ] Reuse `run_exact_regime_workflow_matrix()` for matrix case generation and
      execution.
- [ ] Reuse `build_exact_regime_parameter_sets()` as the default parameter-set source.
- [ ] Reuse `build_exact_regime_workflow_bundle()` fields where practical for
      aggregate matrix interpretation.
- [ ] Keep Story 3-specific logic focused on canonical workflow identity mapping
      and completeness checks.

**Evidence produced**
- One Story 3 assembly path rooted in canonical matrix execution functions.
- Reviewable traceability from Story 3 outputs to existing matrix code paths.

**Risks / rollback**
- Risk: parallel matrix harnesses can drift in thresholds or status semantics.
- Rollback/mitigation: preserve one canonical matrix execution substrate and
  layer Story 3 checks on top.

### Engineering Task 3: Add Explicit Completeness And Status Checks For Mandatory Matrix Coverage

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- code | tests | validation automation

**Definition of done**
- Story 3 can distinguish `complete pass`, `failed`, and `incomplete` matrix
  states.
- Missing mandatory case IDs, missing parameter-set IDs, duplicate IDs, or
  missing status fields block Story 3 closure.
- Partial matrix coverage cannot pass Story 3.

**Execution checklist**
- [ ] Add one completeness helper for required case IDs and per-size
      parameter-set coverage.
- [ ] Require explicit per-case status fields and one aggregate Story 3 status.
- [ ] Treat missing mandatory case identities or malformed status fields as
      incomplete evidence.
- [ ] Keep hand-selected favorable subsets from satisfying Story 3 closure.

**Evidence produced**
- Machine-readable matrix completeness semantics.
- Focused failure signals for missing or malformed matrix evidence.

**Risks / rollback**
- Risk: aggregate pass-rate summaries can hide missing matrix sections.
- Rollback/mitigation: validate explicit identity coverage before aggregate
  interpretation.

### Engineering Task 4: Preserve Exactness, Backend Attribution, And 10q Anchor Presence In Story 3 Summaries

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- code | validation automation

**Definition of done**
- Story 3 preserves explicit exactness interpretation (`<= 1e-8`) for mandatory
  matrix cases and frozen pass-rate semantics.
- Matrix output makes backend attribution explicit for required cases.
- 10-qubit anchor presence is explicit and machine-readable.

**Execution checklist**
- [ ] Preserve per-case and aggregate exactness fields used by canonical matrix
      outputs.
- [ ] Preserve explicit backend identity and supported-path fields for mandatory
      matrix cases.
- [ ] Add explicit `documented_10q_anchor_present` or equivalent field to Story 3
      summary if not already present.
- [ ] Ensure matrix summary clearly reports required-case pass-rate and
      unsupported-case counts.

**Evidence produced**
- Story 3 matrix summary with explicit exactness, attribution, and 10q-anchor
  presence fields.
- Reviewable proof that matrix closure remains inside the frozen exact regime.

**Risks / rollback**
- Risk: 10-qubit anchor can be implied but not verified if not represented
  explicitly in structured output.
- Rollback/mitigation: include explicit 10q anchor presence and case references
  in Story 3 summary.

### Engineering Task 5: Add Focused Regression Tests For Story 3 Matrix Gate Semantics

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- tests

**Definition of done**
- Fast tests validate Story 3 matrix identity and completeness rules.
- Tests include representative failures for missing case IDs, missing
  parameter-set coverage, or missing 10q anchor evidence.
- Regression layer remains lightweight versus full matrix runs.

**Execution checklist**
- [ ] Add focused Story 3 schema/completeness tests in
      `tests/density_matrix/test_density_matrix.py` or a related successor.
- [ ] Add negative tests for incomplete matrix coverage per required size.
- [ ] Add negative test for missing or hidden 10-qubit anchor reference.
- [ ] Keep full matrix execution in dedicated validation command paths.

**Evidence produced**
- Focused Story 3 regression coverage for matrix closure semantics.
- Reviewable failure outputs that localize matrix completeness regressions.

**Risks / rollback**
- Risk: matrix contract drift may remain undetected without targeted regression
  checks.
- Rollback/mitigation: enforce identity and 10q-anchor checks in fast tests.

### Engineering Task 6: Emit One Stable Story 3 Matrix Baseline Artifact Or Rerunnable Command

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- benchmark harness | validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-readable matrix-baseline artifact (or stable
  command that emits it).
- Artifact records mandatory case identities, parameter-set identities, exactness
  semantics, pass/fail summary, and 10q anchor presence.
- Artifact shape is stable enough for Story 5 and Story 6 consumption.

**Execution checklist**
- [ ] Add one Story 3 validation entry point (for example
      `benchmarks/density_matrix/workflow_evidence/matrix_baseline_validation.py`).
- [ ] Emit one stable Story 3 artifact under a Task 6 artifact directory.
- [ ] Record generation command, suite identity, thresholds, and provenance
      metadata in artifact output.
- [ ] Keep Story 3 artifact scope focused on fixed-parameter matrix closure.

**Evidence produced**
- One stable Story 3 matrix baseline artifact or rerunnable command.
- One reusable schema for downstream Task 6 interpretation and publication
  bundling.

**Risks / rollback**
- Risk: ad hoc matrix summaries can diverge and create inconsistent pass/fail
  interpretations across reruns.
- Rollback/mitigation: freeze one machine-readable Story 3 artifact format.

### Engineering Task 7: Document Story 3 Matrix Contract And Handoff Boundaries

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes state Story 3 matrix scope, required matrix inventory,
  and rerun entry points.
- Documentation explicitly separates Story 3 matrix closure from Story 2 trace
  closure and Story 4 unsupported-case closure.
- Notes remain aligned with frozen Task 6 contract language.

**Execution checklist**
- [ ] Document Story 3 rerun command and artifact path.
- [ ] Document required 4/6/8/10 matrix inventory and parameter-set identity
      assumptions.
- [ ] Document explicit handoff to Story 4 and Story 5 semantics.
- [ ] Keep wording aligned with `TASK_6_STORIES.md` and `TASK_6_MINI_SPEC.md`.

**Evidence produced**
- Updated Story 3 implementation-facing documentation.
- One stable documentation reference for Story 3 matrix gate semantics.

**Risks / rollback**
- Risk: unclear ownership can cause overlap or gaps between Story 3 matrix
  closure and neighboring stories.
- Rollback/mitigation: encode handoff boundaries directly in Story 3 guidance.

### Engineering Task 8: Run Story 3 Validation And Confirm Exact-Regime Matrix Readiness

**Implements story**
- `Story 3: The Mandatory Fixed-Parameter Workflow Matrix Covers 4, 6, 8, And 10 Qubits With Exact-Regime Pass/Fail Interpretation`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 tests pass.
- Story 3 matrix artifact command runs and produces stable output.
- Matrix closure evidence is rerunnable and machine-readable.

**Execution checklist**
- [ ] Run focused Story 3 regression tests.
- [ ] Run Story 3 matrix artifact generation command.
- [ ] Verify required 4/6/8/10 coverage, stable case IDs, stable parameter-set
      IDs, and explicit 10q anchor presence.
- [ ] Record test and artifact references for Story 4 handoff.

**Evidence produced**
- Passing Story 3 focused tests.
- Stable Story 3 matrix baseline artifact with required coverage present.

**Risks / rollback**
- Risk: Story 3 can look complete while matrix identity or anchor fields remain
  unstable.
- Rollback/mitigation: require successful rerun and explicit field validation as
  exit evidence.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- mandatory fixed-parameter matrix coverage exists for 4, 6, 8, and 10 qubits,
- each mandatory size includes at least 10 fixed parameter vectors with stable
  parameter-set IDs,
- matrix case IDs are stable and complete for the mandatory inventory,
- matrix output preserves explicit exactness and pass/fail semantics for required
  cases,
- at least one documented 10-qubit anchor evaluation case is explicit and linked
  to the canonical workflow identity,
- one stable Story 3 artifact or rerunnable command defines matrix closure for
  this story,
- and unsupported-case boundaries, interpretation guardrails, and publication
  bundle closure remain clearly assigned to Stories 4 to 6.

## Implementation Notes

- Reuse canonical matrix execution from
  `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` and matrix
  closure patterns from
  `benchmarks/density_matrix/validation_evidence/workflow_baseline_validation.py`.
- In the current Task 6 implementation flow, prefer the committed rich matrix
  surface in
  `benchmarks/density_matrix/artifacts/validation_evidence/workflow_baseline_bundle.json`
  when it preserves all required matrix metadata, instead of forcing a live
  rerun path to be the only source of Story 3 evidence.
- Consume the emitted Story 1 contract artifact directly for canonical workflow
  identity, contract-version, required workflow qubit list, required
  parameter-set count, and documented anchor qubit instead of duplicating those
  values in Story 3 code.
- Keep trace semantics in Story 2 and do not reintroduce trace dependency as a
  Story 3 closure gate.
- Preserve existing case/status vocabulary where possible (`required_cases`,
  `required_pass_rate`, `stable_case_ids_present`,
  `stable_parameter_set_ids_present`, `documented_10q_anchor_present`).
- Prefer one thin Story 3 bundle that references canonical lower-level outputs
  rather than duplicating full raw per-case payloads.
