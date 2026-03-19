# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates
Stay Explicit With No Silent Fallback

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the Task 4 fused boundary into one explicit classification
surface:

- actually fused execution is distinguishable from supported-but-unfused
  execution and from deferred or unsupported fusion candidates,
- no case labeled as fused silently falls back to the plain Task 3 partitioned
  baseline,
- supported workloads that contain ineligible or deferred regions remain
  reviewable without overstating fused coverage,
- and Story 5 closes explicit fusion-classification behavior without yet claiming
  final output packaging or phase-level performance closure.

Out of scope for this story:

- descriptor-local eligibility definition already owned by Story 1,
- the first positive structured fused-runtime slice already owned by Story 2,
- shared fused-capable reuse already owned by Story 3,
- positive exact semantic-preservation closure already owned by Story 4,
- stable fused-output and provenance packaging owned by Story 6,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 through 4 already define the eligibility rule, positive fused path,
  shared fused-capable surface, and semantic-preservation rule that Story 5 must
  classify honestly.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`, and
  `P3-ADR-010`.
- Task 3 already provides the structured runtime negative surface Story 5 should
  align with through `NoisyRuntimeValidationError` and the emitted bundle under
  `benchmarks/density_matrix/artifacts/phase3_task3/story7_unsupported/`.
- Task 3 also already provides the positive runtime-audit surface Story 5 should
  align with through `build_runtime_audit_record()` and the emitted bundle under
  `benchmarks/density_matrix/artifacts/phase3_task3/story6_audit/`.
- Story 5 should prefer extending the shared Task 3 runtime result and audit
  vocabulary over inventing a disconnected fusion-only reporting format.
- The minimum Phase 3 contract explicitly requires three reviewable categories:
  - supported and actually exercised through the fused path,
  - supported but intentionally left unfused,
  - and deferred or unsupported as fusion candidates.
- Story 5 should treat explicit negative or deferred evidence as a required
  output, not as an optional appendix.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 4 Fusion-Classification Taxonomy

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one stable classification taxonomy for Task 4 fusion outcomes.
- The taxonomy distinguishes positive fused coverage from supported non-fusion
  and from deferred or unsupported candidates.
- The taxonomy stays explicit enough for validation, benchmarking, and paper
  packaging.

**Execution checklist**
- [ ] Freeze the minimum three-way classification vocabulary for Task 4:
      actually fused, supported-but-unfused, and deferred or unsupported fusion
      candidate.
- [ ] Define when a supported workload may remain on the plain Task 3 path while
      still staying inside the Task 4 contract.
- [ ] Define when a candidate is merely deferred versus when runtime failure or
      unsupported handling is required.
- [ ] Keep final output packaging and performance interpretation outside the
      Story 5 bar.

**Evidence produced**
- One stable Story 5 fusion-classification taxonomy.
- One explicit rule for supported non-fusion versus deferred or unsupported
  outcomes.

**Risks / rollback**
- Risk: if the classification taxonomy remains loose, later evidence may
  overstate fused coverage or hide deferred work.
- Rollback/mitigation: freeze one shared classification vocabulary before
  broadening artifact production.

### Engineering Task 2: Extend The Runtime Record With Explicit Fused-Coverage And Deferral Fields

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- code | tests

**Definition of done**
- Supported runtime results expose explicit fused-coverage and non-fusion
  summaries.
- Overlapping provenance and path-label fields remain aligned with the Task 3
  runtime record shape.
- Story 5 avoids creating a disconnected fusion-only record format.

**Execution checklist**
- [ ] Extend `NoisyRuntimeExecutionResult`, `build_runtime_audit_record()`, or
      the smallest auditable successors with fused-coverage and deferral fields.
- [ ] Record actually fused span counts and summaries separately from
      supported-but-unfused or deferred span summaries.
- [ ] Keep overlapping provenance and runtime-path fields aligned with Task 3.
- [ ] Add focused regression checks for record-shape stability.

**Evidence produced**
- One explicit Task 4 fused-coverage record shape.
- Regression coverage for shared record stability with additive fusion fields.

**Risks / rollback**
- Risk: fused classification data may drift into ad hoc metadata fields that are
  hard to compare and easy to misread.
- Rollback/mitigation: extend the shared runtime record shape explicitly.

### Engineering Task 3: Block Silent Fallback From Fused-Labeled Cases To The Plain Baseline

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- code | tests

**Definition of done**
- No case labeled as fused can silently resolve to the plain Task 3 baseline.
- Supported-but-unfused cases are clearly labeled as such.
- Out-of-contract fused attempts either remain explicit non-fusion outcomes or
  fail through a structured runtime boundary.

**Execution checklist**
- [ ] Add explicit checks that fused-labeled cases record actual fused-path
      execution.
- [ ] Add explicit checks that supported-but-unfused cases are not mislabeled as
      fused coverage.
- [ ] Reuse the Task 3 runtime negative surface where runtime failure is the
      honest outcome.
- [ ] Prevent silent baseline substitution when the requested review surface is
      fused execution.

**Evidence produced**
- One reviewable no-silent-fallback rule for fused-labeled cases.
- Focused regression coverage for fused-label honesty.

**Risks / rollback**
- Risk: later benchmark bundles may claim fused coverage while still reflecting
  only the unfused baseline path.
- Rollback/mitigation: make fused-label honesty enforceable in code and tests.

### Engineering Task 4: Align Task 4 Classification With The Shared Positive And Negative Runtime Surfaces

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- code | docs

**Definition of done**
- Story 5 classification records fit alongside the Task 3 positive audit and
  negative runtime surfaces where fields overlap.
- Deferred or unsupported fusion-candidate records stay machine-reviewable and
  comparable.
- Task 4 does not introduce a disconnected classification language.

**Execution checklist**
- [ ] Reuse the shared case-level provenance tuple on Story 5 classification
      records where it still applies.
- [ ] Keep overlapping fields aligned with the Task 3 runtime-audit and
      unsupported-runtime surfaces.
- [ ] Add the Task 4 classification fields without replacing the shared artifact
      vocabulary.
- [ ] Document how Story 5 records relate to supported Task 3 runtime audit
      records and unsupported runtime records.

**Evidence produced**
- One aligned Task 4 classification record shape.
- One explicit mapping between supported, unfused, deferred, and unsupported
  fusion evidence.

**Risks / rollback**
- Risk: Task 4 classification evidence may remain structurally incomparable to
  the Task 3 runtime evidence it depends on.
- Rollback/mitigation: align overlapping fields with the shared runtime
  surfaces.

### Engineering Task 5: Add A Representative Fusion-Classification Matrix

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative examples of all required classification
  categories.
- Negative or deferred cases are tied to explicit reasons rather than generic
  "not fused" labels.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Add at least one actually fused representative case.
- [ ] Add at least one supported-but-unfused representative case that still
      remains inside the supported workload matrix.
- [ ] Add at least one deferred or unsupported representative case tied to an
      explicit reason, such as an explicit noise boundary or an out-of-contract
      fused attempt.
- [ ] Keep the matrix small but contract-complete.

**Evidence produced**
- One representative Story 5 fusion-classification matrix.
- Reviewable examples for every required classification category.

**Risks / rollback**
- Risk: without a representative classification matrix, Story 5 may close with
  only positive examples and no proof that non-fused cases stay explicit.
- Rollback/mitigation: freeze a small but classification-complete matrix.

### Engineering Task 6: Emit A Stable Story 5 Fusion-Classification Bundle

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable classification bundle or rerunnable
  checker.
- The bundle records actually fused, supported-but-unfused, and deferred or
  unsupported outcomes through one shared schema.
- The output is reusable by later Story 6 packaging and Story 7 benchmarking.

**Execution checklist**
- [ ] Add a Story 5 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_classification_validation.py` as the primary checker.
- [ ] Add a dedicated Story 5 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task4/story5_classification/`).
- [ ] Emit representative cases through one stable schema with explicit
      classification and reason fields.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 5 fusion-classification bundle or checker.
- One reusable classification surface for later Task 4 work.

**Risks / rollback**
- Risk: prose-only Story 5 closure will make it easy to underreport deferred
  work or overreport fused coverage.
- Rollback/mitigation: emit one structured classification bundle and cite it
  directly.

### Engineering Task 7: Document The Story 5 No-Mislabeling Rule And Follow-On Boundary

**Implements story**
- `Story 5: Fused, Supported-But-Unfused, And Deferred Or Unsupported Candidates Stay Explicit With No Silent Fallback`

**Change type**
- docs

**Definition of done**
- Story 5 notes explain the no-mislabeling rule for fused coverage.
- Deferred channel-native noisy-block fusion remains documented as future work
  rather than hidden incompleteness.
- Developer-facing notes point to the Story 5 tests and bundle.

**Execution checklist**
- [ ] Document the Task 4 classification taxonomy and no-silent-fallback rule.
- [ ] Explain how supported-but-unfused and deferred or unsupported categories
      differ.
- [ ] Explain that channel-native fused noisy blocks remain outside the minimum
      Task 4 contract.
- [ ] Record stable references to the Story 5 tests and emitted bundle.

**Evidence produced**
- Updated developer-facing notes for Story 5 fusion classification.
- One stable handoff reference to Story 6 and Story 7.

**Risks / rollback**
- Risk: later readers may treat unfused or deferred evidence as an
  implementation accident rather than as an explicit contract outcome.
- Rollback/mitigation: document the classification rule and deferred boundary
  explicitly.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- actually fused, supported-but-unfused, and deferred or unsupported fusion
  candidates are explicitly distinguishable through one stable vocabulary,
- no case labeled as fused silently falls back to the plain Task 3 partitioned
  baseline,
- one stable Story 5 bundle or checker records representative cases across all
  required classification categories,
- overlapping fields remain aligned with the shared Task 3 positive and negative
  runtime surfaces,
- and final fused-output packaging and phase-level performance interpretation
  remain clearly assigned to later stories.

## Implementation Notes

- Treat explicit non-fusion evidence as part of the deliverable, not as a
  failure to be hidden.
- Keep the classification surface small and auditable rather than exhaustive and
  noisy.
- Do not let fused coverage, supported non-fusion, and deferred work collapse
  into one ambiguous label.
