# Story 8 Implementation Plan

## Story Being Implemented

Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case
Evidence And Explicit Boundary Language

This is a Layer 4 engineering plan for implementing the eighth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one summary-consistency and bounded-claim guardrail
surface:

- later benchmark and publication summaries are checked against the underlying
  Task 7 per-case records,
- positive claims close only from counted representative evidence rather than
  from hand-filtered or diagnosis-only slices,
- diagnosis-only, excluded, unsupported, or deferred cases remain visible as
  limitation and claim-boundary evidence instead of disappearing from summaries,
- and Story 8 closes the contract for "how Task 7 evidence can be rolled up"
  without taking ownership of final paper prose itself.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- counted-supported benchmark gating already owned by Story 2,
- explicit positive-threshold review already owned by Story 3,
- the sensitivity surface already owned by Story 4,
- the comparable metric surface already owned by Story 5,
- the diagnosis-path bottleneck rule already owned by Story 6,
- and shared benchmark-package assembly already owned by Story 7.

## Dependencies And Assumptions

- Story 7 already defines the shared Task 7 benchmark package Story 8 must
  interpret consistently rather than replace.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-008`,
  `P3-ADR-009`, `P3-ADR-010`, and the performance-claim-boundary item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Task 6 already provides summary-consistency and counted-status precedent Story
  8 should reuse where practical through:
  - `benchmarks/density_matrix/correctness_evidence/summary_consistency_validation.py`,
  - `benchmarks/density_matrix/artifacts/phase3_task6/story8_summary_consistency/`,
  - and the shared Task 6 correctness package.
- Phase 2 already provides interpretation and claim-closure precedent Story 8
  should learn from where useful:
  - `benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py`,
  - `benchmarks/density_matrix/publication_claim_package/evidence_closure_validation.py`,
  - and `benchmarks/density_matrix/publication_claim_package/claim_package_validation.py`.
- Story 8 should not recompute per-case benchmark semantics from raw execution
  data. It should interpret and validate rollups against the Story 7 benchmark
  package.
- The current implementation learning is that Story 8 should keep counted-status
  rules, diagnosis-only labeling, and visible boundary evidence close to the
  shared package surface rather than scattering the same rollup logic across
  later consumers.
- Story 8 should prefer one explicit summary-consistency and bounded-claim rule
  over one rule per downstream consumer.
- The natural implementation home for Task 7 summary-consistency validation is
  the new `benchmarks/density_matrix/performance_evidence/` package, with
  `summary_consistency_validation.py` reading the Story 7 package directly.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Counted-Representative And Summary-Closure Rule

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- docs | validation automation

**Definition of done**
- Story 8 defines one explicit rule for when a downstream summary may claim Task
  7 closure.
- The rule ties positive claims directly to counted representative per-case
  evidence in the Story 7 package.
- The rule is explicit enough that later publication consumers can rely on it
  safely.

**Execution checklist**
- [ ] Freeze the rule that only counted representative Task 7 evidence may
      close a positive downstream benchmark or publication claim.
- [ ] Freeze the rule that diagnosis-only, excluded, unsupported, or deferred
      cases remain visible as limitation or boundary evidence.
- [ ] Define one explicit summary-consistency rule for rollups derived from the
      Story 7 package.
- [ ] Keep final paper prose and narrative framing outside the Story 8 bar.

**Evidence produced**
- One stable Task 7 counted-representative and summary-closure rule.
- One explicit boundary between claim-closing evidence and visible limitation or
  boundary-only evidence.

**Risks / rollback**
- Risk: later benchmark or publication consumers may inflate claims by counting
  diagnosis-only or excluded Task 7 evidence if Story 8 leaves closure
  semantics implicit.
- Rollback/mitigation: freeze one summary-closure rule before broad downstream
  packaging.

### Engineering Task 2: Reuse The Story 7 Benchmark Package And Task 6 Summary-Consistency Precedent As The Base

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- docs | code

**Definition of done**
- Story 8 reuses the Story 7 benchmark package as the direct interpretation
  substrate.
- Story 8 learns from the existing Task 6 summary-consistency and Phase 2
  claim-closure patterns where they strengthen auditability.
- Story 8 avoids creating a detached summary language that ignores per-case Task
  7 records.

**Execution checklist**
- [ ] Reuse the Story 7 package fields and bundle identity directly where they
      already match Story 8 needs.
- [ ] Review Task 6 summary-consistency and Phase 2 claim-closure precedent and
      reuse only the parts that strengthen Task 7 summary auditability.
- [ ] Add only the Task 7-specific summary-consistency and bounded-claim fields
      needed for downstream rollups.
- [ ] Document where Story 8 intentionally extends Story 7 package semantics.

**Evidence produced**
- One reviewable mapping from the Story 7 package and prior summary precedent to
  the Task 7 summary guardrail surface.
- One explicit boundary between reused interpretation patterns and Story 8-
  specific fields.

**Risks / rollback**
- Risk: Story 8 may produce plausible rollups that still drift from per-case
  evidence because the summary logic is detached from the shared package.
- Rollback/mitigation: anchor Story 8 directly on the Story 7 benchmark package.

### Engineering Task 3: Build The Task 7 Summary-Consistency Validation Harness

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- code | validation automation

**Definition of done**
- Story 8 has one reusable harness for validating summary-consistency and
  bounded-claim closure against the Story 7 package.
- The harness records rollup counts, closure flags, and limitation carry-forward
  evidence in a machine-reviewable way.
- The harness is reusable by later publication consumers.

**Execution checklist**
- [ ] Add a dedicated Story 8 validation driver under
      `benchmarks/density_matrix/performance_evidence/`, with
      `summary_consistency_validation.py` as the primary checker.
- [ ] Read the Story 7 benchmark package directly rather than reconstructing
      per-case semantics from raw execution outputs.
- [ ] Record rolled-up counts, closure flags, positive-versus-diagnosis
      semantics, and limitation carry-forward summaries.
- [ ] Keep the harness rooted in machine-reviewable package interpretation
      rather than prose-only summary checks.

**Evidence produced**
- One reusable Story 8 summary-consistency harness.
- One machine-reviewable rollup schema for downstream Task 7 consumers.

**Risks / rollback**
- Risk: summary-consistency logic may remain scattered across scripts and drift
  from the shared benchmark package.
- Rollback/mitigation: centralize Story 8 in one stable interpretation entry
  point.

### Engineering Task 4: Define Explicit Positive-Claim, Diagnosis-Only, And Boundary-Carry-Forward Semantics

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- code | tests

**Definition of done**
- Story 8 defines explicit machine-reviewable semantics for positive-claim
  closure flags, diagnosis-only rollups, and boundary-evidence carry-forward.
- Diagnosis-only, excluded, unsupported, and deferred cases remain visible in
  downstream summaries rather than disappearing into aggregate totals.
- Story 8 avoids post hoc interpretation for basic claim-closure semantics.

**Execution checklist**
- [ ] Add explicit closure-flag and carry-forward fields to the Story 8 rollup
      surface or the smallest auditable successor.
- [ ] Define how diagnosis-only, excluded, unsupported, and deferred cases
      remain visible in benchmark and publication summaries.
- [ ] Ensure positive-claim closure flags cannot pass when counted
      representative evidence is incomplete.
- [ ] Add focused regression checks for closure-field presence and semantic
      stability.

**Evidence produced**
- One explicit Story 8 positive-claim, diagnosis-only, and boundary-carry-
  forward rule.
- Regression coverage for required closure-field stability.

**Risks / rollback**
- Risk: downstream summaries may look complete while hiding diagnosis-only or
  excluded evidence if Story 8 leaves closure semantics implicit.
- Rollback/mitigation: attach closure flags and carry-forward semantics directly
  to the rollup surface.

### Engineering Task 5: Add A Representative Rollup Matrix Across Benchmark And Publication Consumers

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- tests | validation automation

**Definition of done**
- Story 8 covers representative rollup consumers that depend on Task 7 summary
  semantics.
- The matrix is broad enough to show that one counted-representative rule drives
  later benchmark and publication summaries consistently.
- The matrix remains representative and contract-driven rather than exhaustive
  over every downstream report shape.

**Execution checklist**
- [ ] Include at least one benchmark-facing rollup consumer pattern aligned with
      Task 7 internal summaries.
- [ ] Include at least one publication-facing rollup consumer pattern aligned
      with Task 8 needs.
- [ ] Include at least one representative case where diagnosis-only or excluded
      evidence remains visible in the rollup.
- [ ] Keep final paper prose outside the Story 8 matrix.

**Evidence produced**
- One representative Story 8 rollup-consumer matrix.
- One review surface for counted-representative consistency across downstream
  consumers.

**Risks / rollback**
- Risk: Story 8 may appear correct for one consumer while drifting for another.
- Rollback/mitigation: freeze a small but cross-consumer rollup matrix early.

### Engineering Task 6: Emit A Stable Story 8 Summary-Consistency Bundle Or Rerunnable Checker

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- validation automation | docs

**Definition of done**
- Story 8 emits one stable machine-reviewable summary-consistency bundle or
  rerunnable checker.
- The bundle records rollup counts, closure flags, and limitation carry-forward
  semantics through one stable schema.
- The output is stable enough for direct citation in later benchmark and
  publication package review.

**Execution checklist**
- [ ] Add a dedicated Story 8 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story8_summary_consistency/`).
- [ ] Emit rollup counts, closure flags, and carry-forward semantics through one
      stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the relationship to the underlying Story 7 benchmark package explicit
      in the bundle summary.

**Evidence produced**
- One stable Story 8 summary-consistency bundle or checker.
- One direct citation surface for the Task 7 rollup guardrail.

**Risks / rollback**
- Risk: prose-only Story 8 closure will make later reviewers unable to tell
  whether rollups actually agree with the per-case benchmark evidence.
- Rollback/mitigation: emit one machine-reviewable summary-consistency bundle
  directly.

### Engineering Task 7: Document The Task 7 Decision Rule And Run The Summary Guardrail

**Implements story**
- `Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 7 counted-representative closure rule
  and limitation carry-forward semantics concretely.
- The Story 8 summary-consistency harness and bundle run successfully.
- Story 8 keeps final paper prose clearly out of the interpretation guardrail
  itself.

**Execution checklist**
- [ ] Document the Story 8 counted-representative closure rule and the carry-
      forward semantics for diagnosis-only, excluded, unsupported, and deferred
      evidence.
- [ ] Explain how later benchmark and publication rollups should consume the
      Story 7 package and Story 8 guardrail together.
- [ ] Run focused Story 8 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/summary_consistency_validation.py`.
- [ ] Record stable references to the Story 8 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 8 summary-consistency regression checks.
- One stable Story 8 summary-consistency bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 8 for paper prose rather than for the
  Task 7 interpretation guardrail it is meant to provide.
- Rollback/mitigation: document the Story 8 boundary to later publication work
  explicitly.

## Exit Criteria

Story 8 is complete only when all of the following are true:

- one explicit Task 7 counted-representative and summary-closure rule exists,
- downstream rollups are validated directly against the shared Story 7 benchmark
  package,
- diagnosis-only, excluded, unsupported, and deferred cases remain visible as
  limitation or claim-boundary evidence in downstream summaries,
- one stable Story 8 summary-consistency bundle or rerunnable checker exists for
  direct citation,
- and final paper prose remains clearly outside the Story 8 closure bar.

## Implementation Notes

- Prefer one explicit counted-representative rule over one interpretation
  convention per downstream consumer.
- Keep Story 8 focused on summary guardrails, not on writing the paper prose.
- Treat visible limitation and boundary evidence as part of honest claim
  closure, not as something to filter out of downstream summaries.
