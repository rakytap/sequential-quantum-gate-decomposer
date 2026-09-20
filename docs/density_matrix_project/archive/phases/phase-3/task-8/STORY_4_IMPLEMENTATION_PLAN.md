# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen
Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit evidence-closure and interpretation
guardrail for Paper 2:

- only mandatory, complete, supported evidence may close the main Paper 2 claim,
- the closure path must remain the frozen threshold-or-diagnosis rule rather
  than an informal speedup narrative,
- optional, exploratory, unsupported, or incomplete material stays visible as
  context or boundary evidence rather than being promoted into claim closure,
- and Story 4 closes the contract for "what evidence closes Paper 2" without
  taking ownership of supported-path wording, manifest packaging, or future-work
  framing.

Out of scope for this story:

- freezing the claim package itself, which is owned by Story 1,
- keeping publication surfaces aligned, which is owned by Story 2,
- claim-to-source traceability owned by Story 3,
- supported-path, no-fallback, bounded planner-claim, and benchmark-honesty
  wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- future-work positioning owned by Story 7,
- and package-level terminology, reviewer-entry, and summary-consistency
  guardrails owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 3 are already expected to freeze the claim package, align
  publication surfaces, and define the traceability map. Story 4 should
  interpret evidence closure against those stable surfaces rather than redefine
  them.
- The Phase 3 acceptance thresholds are already frozen in
  `DETAILED_PLANNING_PHASE_3.md` and must not be softened here.
- The emitted Task 6 bundle family already defines the correctness and
  unsupported-boundary substrate for Paper 2 closure, especially:
  - `correctness_package/correctness_package_bundle.json`,
  - `unsupported_boundary/unsupported_boundary_bundle.json`,
  - and `summary_consistency/summary_consistency_bundle.json`.
- The emitted Task 7 bundle family already defines the representative benchmark
  and diagnosis substrate for Paper 2 closure, especially:
  - `benchmark_package/benchmark_package_bundle.json`,
  - `diagnosis/diagnosis_bundle.json`,
  - `positive_threshold/positive_threshold_bundle.json`,
  - and `summary_consistency/summary_consistency_bundle.json`.
- Story 4 should treat the current implementation-backed state honestly: the
  representative benchmark package currently closes through the diagnosis branch
  rather than the positive-threshold branch.
- The natural implementation home for Task 8 evidence-closure validation is the
  same `benchmarks/density_matrix/publication_evidence/` package, with
  `evidence_closure_validation.py` as the Story 4 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/publication_evidence/`.
- Story 4 should validate evidence closure only. It should not itself build the
  top-level reviewer manifest or final package consistency surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Paper 2 Evidence-Closure Rule And Threshold-Or-Diagnosis Interpretation

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit Paper 2 evidence-closure rule.
- Story 4 defines one explicit threshold-or-diagnosis interpretation rule for
  the performance side of the claim.
- The rule is stable enough that later Task 8 stories can package it directly.

**Execution checklist**
- [ ] Freeze the rule that only mandatory, complete, supported evidence may
      close the main Paper 2 claim.
- [ ] Freeze the rule that the benchmark side closes through either the positive
      threshold path or the diagnosis-grounded path, but not through ad hoc
      favorable slices.
- [ ] Keep optional, exploratory, deferred, unsupported, or incomplete material
      explicitly outside claim-closing status.
- [ ] Treat omitted closure-rule statements as a Story 4 failure condition.

**Evidence produced**
- One stable Story 4 evidence-closure rule.
- One explicit boundary between claim-closing evidence and contextual or
  boundary-only evidence.

**Risks / rollback**
- Risk: without an explicit closure rule, Paper 2 can quietly promote favorable
  but incomplete evidence into the main claim.
- Rollback/mitigation: freeze the evidence-closure rule before broad publication
  packaging proceeds.

### Engineering Task 2: Reuse The Emitted Task 6 And Task 7 Bundle Surfaces As The Closure Substrate

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- docs | code

**Definition of done**
- Story 4 interprets Paper 2 closure directly from emitted Task 6 and Task 7
  bundles.
- Story 4 preserves the difference between correctness closure, unsupported-
  boundary visibility, positive-threshold evidence, and diagnosis-grounded
  limitation evidence.
- Story 4 avoids creating a detached publication-only evidence vocabulary.

**Execution checklist**
- [ ] Reuse the current Task 6 correctness-package and unsupported-boundary
      bundles directly where they already match Story 4 needs.
- [ ] Reuse the current Task 7 benchmark-package, positive-threshold, diagnosis,
      and summary-consistency bundles directly where they match Story 4 needs.
- [ ] Record which bundle surfaces contribute to positive closure, diagnosis
      closure, and explicit boundary-only carry-forward.
- [ ] Avoid replacing emitted bundle semantics with prose-only interpretations.

**Evidence produced**
- One reviewable mapping from emitted Task 6 and Task 7 bundle surfaces to Story
  4 closure semantics.
- One explicit boundary between reused emitted evidence and Story 4-specific
  closure fields.

**Risks / rollback**
- Risk: Story 4 may produce a plausible closure narrative that no longer matches
  the emitted evidence surfaces reviewers can inspect.
- Rollback/mitigation: anchor Story 4 directly on the emitted bundle surfaces.

### Engineering Task 3: Define The Story 4 Evidence-Closure Record Schema And Checker

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- code | validation automation

**Definition of done**
- Story 4 has one reusable evidence-closure checker.
- The checker records closure flags, supporting bundle references, and
  limitation-carry-forward status through one stable schema.
- The checker stays focused on claim closure rather than on supported-path
  wording or full manifest packaging.

**Execution checklist**
- [ ] Add a Story 4 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `evidence_closure_validation.py` as the primary validation surface.
- [ ] Define one stable Story 4 closure-record schema.
- [ ] Record whether closure proceeds through the positive-threshold path or the
      diagnosis path, together with explicit bundle references.
- [ ] Keep supported-path wording, future-work framing, and package-level
      consistency outside the Story 4 checker.

**Evidence produced**
- One reusable Story 4 evidence-closure checker.
- One stable Story 4 closure schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 4 can turn into vague editorial language if it lacks one concrete
  closure-validation surface.
- Rollback/mitigation: validate one machine-reviewable closure record directly.

### Engineering Task 4: Define Positive-Closure, Diagnosis-Closure, And Boundary-Only Carry-Forward Semantics

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- code | tests

**Definition of done**
- Story 4 defines explicit machine-reviewable semantics for positive closure,
  diagnosis-grounded closure, and non-closing boundary evidence.
- Diagnosis-grounded closure cannot be misread as a positive-threshold win.
- Unsupported, excluded, deferred, or incomplete evidence remains visible as
  boundary evidence rather than as silent omission.

**Execution checklist**
- [ ] Add explicit closure-status and carry-forward fields to the Story 4 record
      surface.
- [ ] Define how diagnosis-grounded closure is represented distinctly from
      positive-threshold closure.
- [ ] Define how unsupported, deferred, optional, or incomplete evidence remains
      visible without becoming claim-closing evidence.
- [ ] Add focused regression checks for closure-field presence and semantic
      stability.

**Evidence produced**
- One explicit Story 4 positive-closure, diagnosis-closure, and boundary-only
  rule.
- Regression coverage for required closure-field stability.

**Risks / rollback**
- Risk: downstream publication surfaces may blur positive and diagnosis closure
  if Story 4 leaves the semantics implicit.
- Rollback/mitigation: attach closure flags and boundary-carry-forward semantics
  directly to the Story 4 surface.

### Engineering Task 5: Add A Representative Evidence-Closure Matrix Across Positive, Diagnosis, And Boundary Cases

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 covers representative closure scenarios across the Paper 2 evidence
  package.
- The matrix is broad enough to show that one closure rule spans positive,
  diagnosis, and boundary-only cases consistently.
- The matrix remains representative and contract-driven rather than exhaustive
  over every record.

**Execution checklist**
- [ ] Include at least one representative positive-threshold scenario path, even
      if the current delivered package does not close through it.
- [ ] Include the current diagnosis-grounded closure path used by the delivered
      representative benchmark package.
- [ ] Include at least one explicit boundary-only case carried forward from the
      Task 6 unsupported-boundary surface.
- [ ] Keep final supported-path wording and reviewer-entry packaging outside the
      Story 4 matrix.

**Evidence produced**
- One representative Story 4 evidence-closure matrix.
- One review surface for cross-category closure semantics.

**Risks / rollback**
- Risk: Story 4 may appear correct for the current diagnosis path while remaining
  underspecified for other closure modes.
- Rollback/mitigation: freeze a small but cross-category closure matrix early.

### Engineering Task 6: Add Focused Regression Checks For Overclaiming And Improper Evidence Promotion

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- tests

**Definition of done**
- Fast checks catch improper promotion of optional, incomplete, unsupported, or
  diagnosis-only evidence into the main Paper 2 claim.
- Negative cases prove Story 4 fails when claim closure does not match the
  frozen rule.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_publication_evidence.py` or a
      tightly related successor for Story 4 evidence closure.
- [ ] Add negative checks for optional or incomplete evidence being treated as
      claim-closing.
- [ ] Add negative checks for diagnosis-only evidence being mislabeled as a
      positive-threshold win.
- [ ] Keep broader manuscript review and final prose polish outside the fast
      regression layer.

**Evidence produced**
- Focused regression coverage for Story 4 overclaiming failures.
- Reviewable failures for improper evidence promotion or misclassified closure.

**Risks / rollback**
- Risk: evidence-closure regressions are easy to miss because the paper may
  still read smoothly.
- Rollback/mitigation: add targeted checks for the highest-risk overclaiming
  modes.

### Engineering Task 7: Emit A Stable Story 4 Evidence-Closure Bundle

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable evidence-closure bundle or one
  stable rerunnable checker output.
- The output records closure flags, supporting bundle references, and boundary-
  carry-forward semantics through one stable schema.
- The output is stable enough for later manifest and package-consistency stories
  to consume directly.

**Execution checklist**
- [ ] Add one stable Story 4 output location under
      `benchmarks/density_matrix/artifacts/publication_evidence/evidence_closure/`.
- [ ] Emit one artifact such as `evidence_closure_bundle.json`.
- [ ] Record generation command, software metadata, and closure summary in the
      output.
- [ ] Keep the output focused on evidence closure rather than on reviewer-entry
      navigation or final package consistency.

**Evidence produced**
- One stable Story 4 evidence-closure bundle or rerunnable checker output.
- One reusable Story 4 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 4 closure will make later reviewers unable to tell
  whether Paper 2 really respects the frozen evidence bar.
- Rollback/mitigation: emit one machine-reviewable closure surface directly.

### Engineering Task 8: Document The Story 4 Rule And Run The Evidence-Closure Gate

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 4 validates, how to rerun it, and
  how it hands off to later Task 8 stories.
- The Story 4 checker and emitted artifact run successfully.
- Story 4 completion is backed by rerunnable evidence-closure validation rather
  than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 4 closure rule and evidence-carry-forward semantics.
- [ ] Make the Story 4 rule explicit:
      only mandatory, complete, supported evidence plus the frozen
      threshold-or-diagnosis interpretation may close the main Paper 2 claim.
- [ ] Explain how Story 4 hands off supported-path wording to Story 5 and
      manifest packaging to Story 6.
- [ ] Run focused Story 4 regression checks and verify the emitted Story 4
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 4 evidence closure.
- One stable Story 4 output proving honest Paper 2 evidence closure.

**Risks / rollback**
- Risk: Story 4 can look complete while still allowing favorable but improper
  evidence promotion in later publication edits.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 4.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one explicit Paper 2 evidence-closure rule defines which evidence may close
  the main claim,
- the closure path remains the frozen positive-threshold-or-diagnosis rule
  rather than an informal benchmark narrative,
- unsupported, deferred, optional, or incomplete evidence remains visible as
  context or boundary-only evidence instead of being promoted into claim closure,
- improper evidence promotion fails focused Story 4 checks,
- one stable Story 4 bundle or rerunnable checker captures the closure surface,
- and supported-path wording, manifest packaging, future-work framing, and
  package-level consistency remain clearly assigned to Stories 5 through 8.

## Implementation Notes

- Treat diagnosis-grounded closure as a first-class honest outcome, not as a
  weaker version of positive-threshold closure.
- Story 4 should be strict about evidence promotion. For Paper 2, the closure
  rule is part of the scientific claim, not editorial polish.
- Keep the Story 4 output thin and explicit. Later stories need one stable
  closure surface they can package, not another narrative review document.
- Honest boundary evidence matters here. Visible exclusions strengthen the paper
  when they preserve the claim boundary clearly.
