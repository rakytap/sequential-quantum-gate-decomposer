# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As
Mandatory Validation Success

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the frozen required/optional/unsupported/incomplete distinction
into an explicit phase-level interpretation layer for Task 5 results:

- the main Phase 2 validation claim remains tied to the mandatory Aer-centered
  package rather than to optional or favorable side evidence,
- unsupported and deferred requests remain visible as negative evidence without
  being counted as positive completion signals,
- missing mandatory artifacts, missing status checks, or malformed summaries are
  treated as incomplete evidence rather than partial success,
- and the resulting interpretation gate stays narrow enough that the final
  publication bundle can package the already-delivered semantics without
  redefining them.

Out of scope for this story:

- implementing unsupported backend, bridge, observable, or noise behavior
  itself,
- changing the frozen support-matrix, benchmark-minimum, or numeric-threshold
  decisions,
- redesigning the lower-level support-tier or unsupported-case taxonomy already
  fixed in earlier stories,
- the final Task 5 top-level provenance and publication bundle owned by Story 6,
- and adding new simulator baselines beyond the mandatory Aer-centered package.

## Dependencies And Assumptions

- Stories 1 to 4 are already in place: the local correctness gate, workflow
  matrix, trace-plus-anchor package, and metric-completeness gate already define
  the mandatory positive evidence that Story 5 must interpret conservatively.
- Task 5 Story 4 now emits a dedicated metric-completeness bundle for supported
  mandatory evidence. Story 5 should reuse that bundle as the canonical metric
  guardrail rather than infer metric completeness again from scattered lower-
  level fields.
- The current support-tier vocabulary already exists in
  `benchmarks/density_matrix/task4_support_tiers.py`, including
  `support_tier`, `case_purpose`, `counts_toward_mandatory_baseline`,
  `optional_reason`, and the boundary-class vocabulary for deferred and
  unsupported cases.
- Task 4 Story 3 already establishes explicit optional-case semantics, and
  Task 4 Story 4 already establishes explicit unsupported and deferred boundary
  semantics, including `unsupported_category`,
  `first_unsupported_condition`, `task4_boundary_class`, and `failure_stage`.
- Task 5 Stories 1 to 4 are expected to emit explicit completeness and status
  semantics; Story 5 interprets those semantics at the phase level rather than
  inferring closure from raw file presence alone.
- The frozen Task 5 interpretation contract is already implied by
  `P2-ADR-006`, `P2-ADR-014`, and `P2-ADR-015`:
  - Qiskit Aer remains the mandatory pass/fail baseline,
  - optional secondary baselines are supplemental only,
  - unsupported or deferred requests do not count as positive evidence,
  - and missing mandatory evidence blocks closure.
- Story 5 should define the phase-level interpretation and guardrail layer; it
  should not reopen the support split, unsupported error behavior, or lower-level
  completeness rules already frozen in earlier stories.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 5 Interpretation Vocabulary And Closure Rules

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- docs | validation automation

**Definition of done**
- Task 5 Story 5 names one canonical interpretation vocabulary for mandatory,
  optional, unsupported, deferred, and incomplete evidence.
- The closure rules explicitly define which evidence counts toward the main Phase
  2 validation claim and which evidence does not.
- The interpretation vocabulary stays aligned with the frozen Task 4 support-tier
  semantics rather than inventing a second taxonomy.

**Execution checklist**
- [ ] Freeze one Task 5 interpretation vocabulary rooted in the existing
      `support_tier` and `case_purpose` semantics.
- [ ] Define explicit closure rules for mandatory success, optional supplemental
      evidence, unsupported or deferred negative evidence, and incomplete
      evidence.
- [ ] Keep the vocabulary aligned with `required`, `optional`, `deferred`, and
      `unsupported` rather than ad hoc review labels.
- [ ] Record the vocabulary and closure rules in one stable implementation-facing
      location that later Task 5 stories can reuse.

**Evidence produced**
- A stable Task 5 interpretation vocabulary and closure-rule mapping.
- One reviewable mapping from frozen Task 5 contract decisions to result labels.

**Risks / rollback**
- Risk: drifting or inconsistent labels will let optional or incomplete evidence
  look more milestone-defining than the frozen contract allows.
- Rollback/mitigation: keep one explicit vocabulary tied directly to the phase
  ADRs and existing support-tier helpers and reuse it everywhere.

### Engineering Task 2: Reuse Existing Support-Tier And Unsupported-Case Schemas Without Renaming Them

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- code | validation automation

**Definition of done**
- Task 5 Story 5 interprets optional, deferred, and unsupported evidence through
  the existing support-tier and boundary schemas rather than via a new phase-
  specific label set.
- The phase-level interpretation layer preserves lower-level field names wherever
  practical so reviewers can trace a claim from summary to raw evidence without
  schema translation.
- Story 5 remains an interpretation layer, not a replacement for lower-level
  classification or unsupported-validation code.

**Execution checklist**
- [ ] Reuse `support_tier`, `case_purpose`,
      `counts_toward_mandatory_baseline`, and related support-tier fields from
      current bundles.
- [ ] Reuse existing negative-evidence fields such as
      `unsupported_category`, `first_unsupported_condition`,
      `task4_boundary_class`, and `failure_stage` where unsupported evidence is
      surfaced in Task 5 summaries.
- [ ] Preserve current required-case accounting fields such as
      `required_cases`, `required_passed_cases`, `required_pass_rate`, and
      `mandatory_baseline_completed`.
- [ ] Avoid introducing Task 5-only synonyms when an existing stable field name
      already carries the needed meaning.

**Evidence produced**
- One Task 5 Story 5 interpretation layer rooted in the canonical support-tier
  and unsupported-case schemas.
- Reviewable traceability from the phase-level interpretation gate to the
  existing case-level classification surfaces.

**Risks / rollback**
- Risk: a second phase-level label set would force reviewers to translate between
  bundles and make interpretation drift more likely.
- Rollback/mitigation: preserve lower-level classification field names wherever
  practical and add only the smallest derived Task 5 summary fields.

### Engineering Task 3: Compute The Main Phase 2 Validation Claim Only From Mandatory Complete Supported Evidence

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- code | tests | validation automation

**Definition of done**
- The Task 5 closure signal is computed only from mandatory, complete, supported
  evidence.
- Optional evidence, unsupported evidence, deferred-scope guards, and incomplete
  mandatory artifacts are excluded explicitly from the main pass/fail claim.
- Aggregate phase-level success cannot be satisfied by partial bundles or by
  favorable subsets.

**Execution checklist**
- [ ] Add or tighten one phase-level interpretation helper that computes closure
      only from mandatory, complete, supported evidence items.
- [ ] Exclude optional or unsupported cases explicitly rather than by omission or
      convention.
- [ ] Treat missing mandatory items, missing status fields, or malformed bundle
      semantics as incomplete evidence that blocks closure.
- [ ] Keep the aggregate phase-level pass/fail logic auditable and easy to
      reproduce from machine-readable inputs.

**Evidence produced**
- One explicit Task 5 closure rule for the main Phase 2 validation claim.
- Machine-readable proof that the main claim excludes optional, unsupported, and
  incomplete evidence.

**Risks / rollback**
- Risk: a single aggregate success flag can quietly mix mandatory success with
  optional or incomplete evidence.
- Rollback/mitigation: compute the phase-level closure signal only from explicit
  mandatory-baseline evidence and validate that rule directly.

### Engineering Task 4: Preserve Optional, Unsupported, And Incomplete Semantics In Phase-Level Summaries

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- code | validation automation

**Definition of done**
- Phase-level Task 5 summaries keep optional, unsupported, deferred, and
  incomplete evidence visible rather than flattening everything into one pass/fail
  line.
- Reviewers can tell from stored outputs why a result was counted toward closure,
  excluded as supplemental, or treated as incomplete.
- Optional, unsupported, and incomplete evidence remain separated in a
  machine-readable way suitable for later publication packaging.

**Execution checklist**
- [ ] Preserve explicit required-versus-optional accounting in phase-level
      summaries.
- [ ] Preserve unsupported and deferred evidence as negative evidence with stable
      boundary-class fields where applicable.
- [ ] Add the smallest explicit incomplete-evidence summary fields needed to make
      missing mandatory items or bad status semantics visible.
- [ ] Avoid flattening away why an artifact was excluded from the main claim.

**Evidence produced**
- Phase-level Task 5 summaries that distinguish mandatory, optional,
  unsupported, deferred, and incomplete evidence.
- Reviewable machine-readable interpretation fields for every excluded evidence
  class.

**Risks / rollback**
- Risk: if excluded evidence types are visible only in prose, reviewers may still
  misread the aggregate result.
- Rollback/mitigation: keep exclusion semantics explicit in the structured output,
  not only in documentation.

### Engineering Task 5: Add Focused Regression Tests For Task 5 Interpretation Rules

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- tests

**Definition of done**
- Fast automated tests prove that optional evidence stays supplemental,
  unsupported or deferred evidence stays negative, and incomplete mandatory
  evidence blocks Task 5 closure.
- Regression coverage is specific enough to catch interpretation drift without
  running the full publication bundle.
- The tests make clear that the main Phase 2 claim cannot be satisfied by
  optional or partial evidence alone.

**Execution checklist**
- [ ] Add focused tests for support-tier-aware closure semantics.
- [ ] Assert that optional evidence cannot satisfy the mandatory Task 5 closure
      rule.
- [ ] Add at least one representative negative test showing that missing or
      malformed mandatory evidence is treated as incomplete rather than as
      acceptable partial success.
- [ ] Keep full bundle generation in dedicated validation commands rather than the
      default fast test path.

**Evidence produced**
- Focused regression coverage for Task 5 interpretation semantics.
- Reviewable failures when optional, unsupported, or incomplete evidence is
  misclassified.

**Risks / rollback**
- Risk: without focused interpretation tests, optional or incomplete evidence can
  drift into the main success path silently.
- Rollback/mitigation: keep a small test surface that locks the interpretation
  contract down directly.

### Engineering Task 6: Emit One Stable Task 5 Story 5 Interpretation Summary Or Checker

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 can emit one stable machine-readable interpretation summary or one
  stable rerunnable checker that explains why Task 5 is closed, blocked, or
  incomplete.
- The output records the main closure signal, the excluded evidence classes, and
  the reasons those classes do not count toward the main claim.
- The output shape is stable enough that Story 6 can package it directly into the
  final Task 5 bundle.

**Execution checklist**
- [ ] Build the Task 5 Story 5 output as a thin checker or summary around the
      existing story outputs rather than a second copy of their raw data.
- [ ] Record the main closure signal plus explicit counts or references for
      optional, unsupported, deferred, and incomplete evidence.
- [ ] Keep the output narrow to interpretation semantics rather than mixing in
      publication-only provenance fields.
- [ ] Make the checker or summary stable enough for later Task 5 stories and
      paper-facing bundle assembly to reference directly.

**Evidence produced**
- One stable Task 5 Story 5 interpretation summary or rerunnable checker.
- A reusable output schema for later Task 5 publication-bundle assembly.

**Risks / rollback**
- Risk: ad hoc interpretation notes will drift and make later publication
  packaging harder to audit.
- Rollback/mitigation: define one thin structured interpretation output now and
  extend it incrementally.

### Engineering Task 7: Document The Task 5 Interpretation Rules And Their Hand-Offs

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Task 5 Story 5 validates, how to rerun it,
  and why it is the canonical interpretation guardrail for the phase.
- The notes make clear that Story 5 sits above the delivered evidence layers and
  below the final publication bundle.
- The documentation stays aligned with the frozen Phase 2 claim boundaries and
  does not overstate optional or partial evidence.

**Execution checklist**
- [ ] Document the Task 5 Story 5 checker or summary and its relationship to the
      lower-level story outputs.
- [ ] Make the main interpretation rule explicit:
      only mandatory, complete, supported evidence can close the main Task 5
      claim.
- [ ] Explain how Story 5 hands off final provenance packaging to Story 6.
- [ ] Keep optional secondary baselines and unsupported negative evidence visibly
      distinct in the notes.

**Evidence produced**
- Updated developer-facing guidance for the Task 5 Story 5 interpretation gate.
- One stable place where Story 5 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 5 is poorly documented, reviewers may still overread optional or
  partial evidence.
- Rollback/mitigation: tie the notes directly to the same checker or structured
  summary used for Story 5 validation.

### Engineering Task 8: Run Story 5 Validation And Confirm The Main Phase 2 Claim Cannot Be Inflated

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success`

**Change type**
- tests | validation automation

**Definition of done**
- The Task 5 Story 5 interpretation layer runs successfully end to end.
- Optional, unsupported, deferred, and incomplete evidence are all visible in the
  resulting outputs and excluded from the main closure signal as required.
- Story 5 completion is backed by stable outputs and rerunnable checks rather
  than by documentation alone.

**Execution checklist**
- [ ] Run the focused Story 5 regression tests for interpretation semantics.
- [ ] Run the dedicated Story 5 checker or summary-emission path.
- [ ] Verify that the main Task 5 closure signal is computed only from mandatory,
      complete, supported evidence.
- [ ] Record the stable test run and artifact references for Story 6 and later
      publication work.

**Evidence produced**
- Passing focused pytest coverage for Task 5 Story 5.
- A machine-readable interpretation summary or rerunnable checker proving the
  main Task 5 claim cannot be satisfied by optional, unsupported, or incomplete
  evidence.

**Risks / rollback**
- Risk: Story 5 can look complete while still allowing optional or incomplete
  evidence to influence the main success signal implicitly.
- Rollback/mitigation: treat the emitted interpretation output and closure-rule
  validation as part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one stable Task 5 interpretation vocabulary defines mandatory, optional,
  deferred, unsupported, and incomplete evidence,
- the main Phase 2 validation claim is computed only from mandatory, complete,
  supported evidence,
- optional evidence remains explicitly supplemental,
- unsupported and deferred requests remain explicitly negative evidence,
- incomplete mandatory evidence blocks closure explicitly,
- one stable validation command or checker plus one stable Task 5 Story 5 summary
  define the interpretation gate,
- and the final provenance and publication bundle remain clearly assigned to
  Story 6.

## Implementation Notes

- `benchmarks/density_matrix/task4_support_tiers.py` already provides the
  canonical support-tier vocabulary and mandatory-baseline accounting semantics.
  Story 5 should build on those helpers rather than invent a parallel taxonomy.
- Task 4 Story 3 and Story 4 already fix the optional and unsupported boundary
  language used across the current artifact bundles. Story 5 should preserve that
  language where practical.
- Story 1 to Story 4 outputs should already expose explicit completeness and
  status semantics. Story 5 should interpret those semantics at the phase level
  rather than infer closure from raw file presence or narrative description.
- In particular, Story 4 now packages metric completeness as its own phase-level
  artifact, so Story 5 can treat missing-metric failures as explicit incomplete
  evidence instead of reconstructing them from lower-level payloads.
- Story 5 is a phase-level interpretation and guardrail layer, not a new
  validation harness and not a second unsupported-case implementation surface.
- Story 6 should package the delivered semantics into the final Task 5
  publication bundle. Story 5 should focus on making the claim boundary
  impossible to overread before packaging begins.
