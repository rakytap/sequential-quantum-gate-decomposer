# Story 8 Implementation Plan

## Story Being Implemented

Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case
Evidence

This is a Layer 4 engineering plan for implementing the eighth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one summary-consistency and claim-closure guardrail
surface:

- later benchmark and publication summaries are checked against the underlying
  Task 6 per-case records,
- only complete counted supported evidence can close downstream claims,
- excluded, unsupported, or deferred cases remain visible as claim-boundary
  evidence instead of disappearing from summaries,
- and Story 8 closes the contract for "how Task 6 evidence can be rolled up"
  without taking ownership of Task 7 performance thresholds or Task 8 paper
  prose.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- the bounded Qiskit Aer slice already owned by Story 3,
- output-integrity and continuity-energy emphasis already owned by Story 4,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- shared correctness-package assembly already owned by Story 7,
- and Task 7 performance-threshold satisfaction or Task 8 narrative framing.

## Dependencies And Assumptions

- Story 7 already defines the shared Task 6 correctness package Story 8 must
  interpret consistently rather than replace.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-008`,
  `P3-ADR-009`, `P3-ADR-010`, and the performance-claim-boundary item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Phase 2 already provides summary- and claim-closure precedent Story 8 should
  reuse where practical:
  - `benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py`,
  - `benchmarks/density_matrix/publication_claim_package/evidence_closure_validation.py`,
  - `benchmarks/density_matrix/publication_claim_package/claim_package_validation.py`,
  - and `benchmarks/density_matrix/workflow_evidence/workflow_publication_bundle.py`.
- Story 8 should not recompute per-case correctness from raw execution data. It
  should interpret and validate rollups against the Story 7 correctness package.
- The current implementation learning is that Story 8 should consume the Story 7
  package through one dedicated summary-consistency validator and keep the
  counted-status rule close to that package surface rather than scattering the
  same rollup logic across later consumers.
- Story 8 should treat excluded, unsupported, and deferred cases as required
  boundary evidence that must remain visible in later rollups.
- Story 8 should prefer one explicit counted-status and summary-consistency rule
  over one rule per downstream consumer.
- The natural implementation home for Task 6 summary-consistency validation is
  the new `benchmarks/density_matrix/correctness_evidence/` package, with
  `summary_consistency_validation.py` reading the Story 7 package directly.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Counted-Status And Summary-Closure Rule

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- docs | validation automation

**Definition of done**
- Story 8 defines one explicit rule for when a downstream summary may claim Task
  6 closure.
- The rule ties downstream rollups directly to counted supported per-case
  evidence in the Story 7 package.
- The rule is explicit enough that later Task 7 and Task 8 consumers can rely
  on it safely.

**Execution checklist**
- [ ] Freeze the rule that only complete counted supported Task 6 evidence may
      close a downstream benchmark or publication claim.
- [ ] Freeze the rule that excluded, unsupported, or deferred cases remain
      visible as claim-boundary evidence.
- [ ] Define one explicit summary-consistency rule for rollups derived from the
      Story 7 package.
- [ ] Keep Task 7 performance thresholds and Task 8 narrative framing outside
      the Story 8 bar.

**Evidence produced**
- One stable Task 6 counted-status and summary-closure rule.
- One explicit boundary between claim-closing evidence and visible boundary-only
  evidence.

**Risks / rollback**
- Risk: later benchmark or publication consumers may inflate claims by counting
  incomplete or excluded Task 6 evidence if Story 8 leaves closure semantics
  implicit.
- Rollback/mitigation: freeze one summary-closure rule before broad downstream
  packaging.

### Engineering Task 2: Reuse The Story 7 Correctness Package And Phase 2 Interpretation Precedent As The Base

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- docs | code

**Definition of done**
- Story 8 reuses the Story 7 correctness package as the direct interpretation
  substrate.
- Story 8 learns from the existing Phase 2 interpretation and claim-closure
  patterns where they strengthen auditability.
- Story 8 avoids creating a detached summary language that ignores per-case Task
  6 records.

**Execution checklist**
- [ ] Reuse the Story 7 package fields and bundle identity directly where they
      already match Story 8 needs.
- [ ] Review Phase 2 interpretation and claim-closure precedent and reuse only
      the parts that strengthen Task 6 summary auditability.
- [ ] Add only the Task 6-specific summary-consistency and counted-status fields
      needed for downstream rollups.
- [ ] Document where Story 8 intentionally extends Story 7 package semantics.

**Evidence produced**
- One reviewable mapping from the Story 7 package and Phase 2 interpretation
  precedent to the Task 6 summary guardrail surface.
- One explicit boundary between reused interpretation patterns and Story 8-
  specific fields.

**Risks / rollback**
- Risk: Story 8 may produce plausible rollups that still drift from per-case
  evidence because the summary logic is detached from the shared package.
- Rollback/mitigation: anchor Story 8 directly on the Story 7 correctness
  package.

### Engineering Task 3: Build The Task 6 Summary-Consistency Validation Harness

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- code | validation automation

**Definition of done**
- Story 8 has one reusable harness for validating summary-consistency and
  counted-status closure against the Story 7 package.
- The harness records rollup counts, closure flags, and carry-forward boundary
  evidence in a machine-reviewable way.
- The harness is reusable by later Task 7 benchmark and Task 8 publication
  consumers.

**Execution checklist**
- [ ] Add a dedicated Story 8 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `summary_consistency_validation.py` as the primary checker.
- [ ] Read the Story 7 correctness package directly rather than reconstructing
      per-case semantics from raw execution outputs.
- [ ] Record rolled-up counts, counted-status closure flags, and boundary-
      evidence carry-forward summaries.
- [ ] Keep the harness rooted in machine-reviewable package interpretation
      rather than prose-only summary checks.

**Evidence produced**
- One reusable Story 8 summary-consistency harness.
- One machine-reviewable rollup schema for downstream Task 6 consumers.

**Risks / rollback**
- Risk: summary-consistency logic may remain scattered across scripts and drift
  from the shared correctness package.
- Rollback/mitigation: centralize Story 8 in one stable interpretation entry
  point.

### Engineering Task 4: Define Explicit Claim-Closure And Boundary-Carry-Forward Semantics

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- code | tests

**Definition of done**
- Story 8 defines explicit machine-reviewable semantics for main closure flags,
  counted-status rollups, and boundary-evidence carry-forward.
- Excluded, unsupported, and deferred cases remain visible in downstream
  summaries rather than disappearing into aggregate totals.
- Story 8 avoids post hoc interpretation for basic claim-closure semantics.

**Execution checklist**
- [ ] Add explicit closure-flag and carry-forward fields to the Story 8 rollup
      surface or the smallest auditable successor.
- [ ] Define how excluded, unsupported, and deferred cases remain visible in
      benchmark and publication summaries.
- [ ] Ensure main closure flags cannot pass when required counted supported
      evidence is incomplete.
- [ ] Add focused regression checks for closure-field presence and semantic
      stability.

**Evidence produced**
- One explicit Story 8 claim-closure and boundary-carry-forward rule.
- Regression coverage for required closure-field stability.

**Risks / rollback**
- Risk: downstream summaries may look complete while hiding missing or excluded
  cases if Story 8 leaves closure semantics implicit.
- Rollback/mitigation: attach closure flags and carry-forward semantics directly
  to the rollup surface.

### Engineering Task 5: Add A Representative Rollup Matrix Across Benchmark And Publication Consumers

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- tests | validation automation

**Definition of done**
- Story 8 covers representative rollup consumers that depend on Task 6 summary
  semantics.
- The matrix is broad enough to show that one counted-status rule drives later
  benchmark and publication summaries consistently.
- The matrix remains representative and contract-driven rather than exhaustive
  over every downstream report shape.

**Execution checklist**
- [ ] Include at least one benchmark-facing rollup consumer pattern aligned with
      Task 7 needs.
- [ ] Include at least one publication-facing rollup consumer pattern aligned
      with Task 8 needs.
- [ ] Include at least one representative case where excluded or unsupported
      evidence remains visible in the rollup.
- [ ] Keep performance-threshold satisfaction and paper prose outside the Story
      8 matrix.

**Evidence produced**
- One representative Story 8 rollup-consumer matrix.
- One review surface for counted-status consistency across downstream consumers.

**Risks / rollback**
- Risk: Story 8 may appear correct for one consumer while drifting for another.
- Rollback/mitigation: freeze a small but cross-consumer rollup matrix early.

### Engineering Task 6: Emit A Stable Story 8 Summary-Consistency Bundle Or Rerunnable Checker

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- validation automation | docs

**Definition of done**
- Story 8 emits one stable machine-reviewable summary-consistency bundle or
  rerunnable checker.
- The bundle records rollup counts, closure flags, and carry-forward boundary
  semantics through one stable schema.
- The output is stable enough for direct citation in later benchmark and
  publication package review.

**Execution checklist**
- [ ] Add a dedicated Story 8 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story8_summary_consistency/`).
- [ ] Emit rollup counts, closure flags, and carry-forward semantics through one
      stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the relationship to the underlying Story 7 package explicit in the
      bundle summary.

**Evidence produced**
- One stable Story 8 summary-consistency bundle or checker.
- One direct citation surface for the Task 6 rollup guardrail.

**Risks / rollback**
- Risk: prose-only Story 8 closure will make later reviewers unable to tell
  whether rollups actually agree with the per-case evidence.
- Rollback/mitigation: emit one machine-reviewable summary-consistency bundle
  directly.

### Engineering Task 7: Document The Story 8 Decision Rule And Run The Summary Guardrail

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 6 counted-status and summary-
  consistency rule concretely.
- The Story 8 summary-consistency harness and bundle run successfully.
- Story 8 keeps Task 7 performance closure and Task 8 paper prose clearly out of
  the Task 6 interpretation guardrail itself.

**Execution checklist**
- [ ] Document the Story 8 counted-status closure rule and the carry-forward
      semantics for excluded, unsupported, and deferred evidence.
- [ ] Explain how later benchmark and publication rollups should consume the
      Story 7 package and Story 8 guardrail together.
- [ ] Run focused Story 8 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/summary_consistency_validation.py`.
- [ ] Record stable references to the Story 8 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 8 summary-consistency regression checks.
- One stable Story 8 summary-consistency bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 8 for performance closure or paper
  prose rather than the Task 6 interpretation guardrail it is meant to be.
- Rollback/mitigation: document the Story 8 boundary to later benchmark and
  publication work explicitly.

## Exit Criteria

Story 8 is complete only when all of the following are true:

- one explicit Task 6 counted-status and summary-closure rule exists,
- downstream rollups are validated directly against the shared Story 7
  correctness package,
- excluded, unsupported, and deferred cases remain visible as claim-boundary
  evidence in downstream summaries,
- one stable Story 8 summary-consistency bundle or rerunnable checker exists for
  direct citation,
- and Task 7 performance thresholds plus Task 8 narrative framing remain
  clearly outside the Story 8 closure bar.

## Implementation Notes

- Prefer one explicit counted-status rule over one interpretation convention per
  downstream consumer.
- Keep Story 8 focused on summary guardrails, not on proving benchmark
  performance or writing paper prose.
- Treat visible negative evidence as part of honest claim closure, not as
  something to filter out of downstream summaries.
