# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be
Counted As Task 6 Completion

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns the Task 6 required/optional/unsupported/incomplete distinction
into explicit machine-checkable interpretation guardrails:

- Task 6 completion is computed only from mandatory, complete, supported
  canonical-workflow evidence,
- optional evidence remains visible as supplemental and cannot satisfy closure
  by itself,
- unsupported and deferred evidence remains visible as negative boundary evidence
  and cannot be counted as success,
- missing mandatory artifacts or missing status semantics are treated as
  incomplete evidence that blocks closure,
- and Story 5 remains an interpretation layer so Story 6 can package the same
  semantics without redefining closure rules.

Out of scope for this story:

- implementing unsupported workflow behavior,
- changing frozen workflow inventory, support matrix, or threshold decisions,
- redefining lower-level unsupported diagnostics produced by Story 4,
- and top-level publication-bundle assembly and provenance closure owned by
  Story 6.

## Dependencies And Assumptions

- Story 1 now emits the canonical workflow-contract artifact through
  `benchmarks/density_matrix/workflow_evidence/workflow_contract_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_contract_bundle.json`.
  Story 5 interpretation should treat that emitted contract as the canonical
  identity anchor for all included and excluded evidence.
- Story 2 now emits
  `benchmarks/density_matrix/workflow_evidence/end_to_end_trace_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/end_to_end_trace_bundle.json`.
  Story 5 should interpret that bundle as the mandatory 4q/6q end-to-end plus
  required-trace evidence layer for Task 6 completion.
- Story 3 now emits
  `benchmarks/density_matrix/workflow_evidence/matrix_baseline_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/matrix_baseline_bundle.json`.
  Story 5 should interpret that bundle as the mandatory 4/6/8/10 fixed-
  parameter matrix evidence layer for Task 6 completion.
- Story 4 now emits
  `benchmarks/density_matrix/workflow_evidence/unsupported_workflow_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/workflow_evidence/unsupported_workflow_bundle.json`.
  Story 5 should interpret that bundle as the canonical unsupported/deferred
  negative-evidence layer for exclusion from the main Task 6 claim.
- The committed optional evidence surface in
  `benchmarks/density_matrix/artifacts/noise_support/optional_noise_classification_bundle.json`
  should be reused as the canonical supplemental optional-evidence layer.
- Task 5 Story 5 interpretation workflow in
  `benchmarks/density_matrix/validation_evidence/interpretation_validation.py` provides
  a proven pattern that Story 5 should reuse where practical.
- Existing support-tier and boundary taxonomy from
  `benchmarks/density_matrix/noise_support/support_tiers.py` should remain the canonical
  classification vocabulary.
- Story 4 outputs should already include explicit unsupported status and boundary
  diagnostics required for Story 5 interpretation, and should now also satisfy
  the Story 1-emitted `required_unsupported_case_fields` inventory so Story 5
  can rely on one canonical unsupported-case shape.
- Frozen interpretation expectations remain unchanged:
  `P2-ADR-007`, `P2-ADR-014`, and `P2-ADR-015`.
- Story 5 should compute and expose Task 6 interpretation guardrails, not
  replace lower-level validation logic.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 6 Interpretation Vocabulary And Closure Rules

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one stable interpretation vocabulary for mandatory, optional,
  unsupported, deferred, and incomplete evidence.
- Closure rules explicitly state which classes count toward Task 6 completion.
- Vocabulary stays aligned with existing support-tier semantics.

**Execution checklist**
- [ ] Freeze one Story 5 interpretation vocabulary rooted in existing
      support-tier fields.
- [ ] Freeze explicit closure rule:
      only mandatory, complete, supported evidence closes Task 6.
- [ ] Freeze explicit exclusion rules for optional, unsupported, deferred, and
      incomplete evidence classes.
- [ ] Record vocabulary and closure mapping in one stable implementation-facing
      location.

**Evidence produced**
- One stable Story 5 interpretation vocabulary.
- One explicit closure-rule map from evidence class to inclusion/exclusion.

**Risks / rollback**
- Risk: inconsistent labels across artifacts can allow optional or incomplete
  evidence to inflate closure claims.
- Rollback/mitigation: preserve one explicit vocabulary and reuse it across
  Story 5 and Story 6.

### Engineering Task 2: Reuse Existing Support-Tier And Unsupported-Case Schemas Without Renaming

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 interpretation layer reuses existing classification fields wherever
  practical.
- Story 5 avoids Task 6-only synonym fields unless strictly required.
- Traceability from Story 5 summaries to lower-level evidence remains direct.

**Execution checklist**
- [ ] Reuse existing support-tier fields such as `support_tier`,
      `case_purpose`, and `counts_toward_mandatory_baseline`.
- [ ] Reuse existing unsupported boundary fields where available:
      `unsupported_category`, `first_unsupported_condition`,
      `unsupported_reason`, and `failure_stage`.
- [ ] Preserve required-case accounting fields and status semantics from Story 2
      and Story 3 outputs.
- [ ] Add only minimal derived interpretation fields at Story 5 level.

**Evidence produced**
- Story 5 interpretation layer rooted in existing schemas.
- Reviewable field-level traceability from Story 5 summaries to raw evidence.

**Risks / rollback**
- Risk: introducing parallel classification terms can break interpretation
  consistency and auditability.
- Rollback/mitigation: preserve canonical field names and derive summaries
  minimally.

### Engineering Task 3: Compute The Main Task 6 Completion Signal Only From Mandatory Complete Supported Evidence

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- code | tests | validation automation

**Definition of done**
- Story 5 computes Task 6 completion from mandatory complete supported evidence
  only.
- Optional and unsupported evidence are explicitly excluded from completion
  computation.
- Missing mandatory evidence or malformed status semantics produce incomplete
  closure status.

**Execution checklist**
- [ ] Add one explicit Story 5 closure helper that computes main completion only
      from mandatory complete supported artifacts.
- [ ] Exclude optional and unsupported evidence through explicit logic, not by
      omission.
- [ ] Treat missing mandatory artifacts or missing required status fields as
      incomplete evidence that blocks closure.
- [ ] Keep aggregate closure computation auditable from machine-readable inputs.

**Evidence produced**
- One explicit Story 5 completion signal with documented computation rule.
- Machine-readable proof that excluded evidence classes cannot satisfy closure.

**Risks / rollback**
- Risk: aggregate pass flags can silently mix mandatory and supplemental
  evidence.
- Rollback/mitigation: enforce explicit inclusion and exclusion logic in Story 5
  computation.

### Engineering Task 4: Preserve Optional, Unsupported, Deferred, And Incomplete Evidence As Explicit Summary Classes

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 summary explicitly reports each excluded evidence class and why it was
  excluded.
- Story 5 output distinguishes positive closure evidence from supplemental and
  negative evidence classes.
- Incomplete-evidence reasons are visible and machine-readable.

**Execution checklist**
- [ ] Preserve explicit counters or lists for optional, unsupported, deferred,
      and incomplete evidence classes in Story 5 output.
- [ ] Preserve per-class status flags showing whether interpretation guardrails
      were respected.
- [ ] Add explicit incomplete-evidence diagnostics for missing mandatory cases or
      malformed statuses.
- [ ] Ensure summaries keep exclusion reasons auditable rather than prose-only.

**Evidence produced**
- Story 5 summaries with explicit per-class evidence accounting.
- Machine-readable exclusion diagnostics for incomplete and unsupported evidence.

**Risks / rollback**
- Risk: excluded evidence can be misread as accepted if summaries flatten all
  outcomes into one score.
- Rollback/mitigation: keep class-specific fields mandatory in Story 5 outputs.

### Engineering Task 5: Add Focused Regression Tests For Story 5 Interpretation Guardrails

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- tests

**Definition of done**
- Fast tests validate Story 5 closure rules and exclusion semantics.
- Tests include representative negative cases for optional-only evidence,
  unsupported-only evidence, and incomplete mandatory evidence.
- Regression coverage remains lightweight versus full bundle runs.

**Execution checklist**
- [ ] Add focused Story 5 interpretation tests in
      `tests/density_matrix/test_density_matrix.py` or a related successor.
- [ ] Add a test proving optional evidence cannot satisfy Task 6 completion.
- [ ] Add a test proving unsupported evidence remains negative-only.
- [ ] Add a test proving missing mandatory evidence yields incomplete status.

**Evidence produced**
- Focused Story 5 interpretation regression coverage.
- Reviewable failures for interpretation drift and misclassification.

**Risks / rollback**
- Risk: interpretation drift can occur silently when new artifact classes are
  added.
- Rollback/mitigation: enforce class-specific closure tests as part of fast CI.

### Engineering Task 6: Emit One Stable Story 5 Interpretation Bundle Or Rerunnable Command

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- benchmark harness | validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-readable interpretation bundle (or stable
  command) with explicit closure and exclusion semantics.
- Bundle includes closure signal, excluded evidence classes, and incomplete
  diagnostics.
- Artifact shape is stable enough for Story 6 publication packaging.

**Execution checklist**
- [ ] Add one Story 5 validation entry point (for example
      `benchmarks/density_matrix/workflow_evidence/workflow_interpretation_validation.py`).
- [ ] Emit one stable Story 5 interpretation artifact under Task 6 artifacts.
- [ ] Record generation command, suite identity, and provenance metadata.
- [ ] Keep Story 5 artifact focused on interpretation guardrails, not publication
      assembly.

**Evidence produced**
- One stable Story 5 interpretation bundle or rerunnable command.
- One reusable interpretation schema for Story 6 bundling.

**Risks / rollback**
- Risk: ad hoc interpretation summaries can diverge and become non-auditable.
- Rollback/mitigation: freeze one machine-readable Story 5 interpretation
  artifact format.

### Engineering Task 7: Document Story 5 Closure Rules And Handoff To Story 6

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain Story 5 closure rule, exclusion classes, and
  rerun entry points.
- Documentation states Story 6 owns final publication-facing package assembly.
- Notes remain aligned with Task 6 mini-spec and story contracts.

**Execution checklist**
- [ ] Document Story 5 rerun command and artifact output location.
- [ ] Document mandatory inclusion rule and exclusion class semantics.
- [ ] Document handoff to Story 6 publication bundle assembly.
- [ ] Keep wording aligned with `TASK_6_MINI_SPEC.md` and `TASK_6_STORIES.md`.

**Evidence produced**
- Updated Story 5 implementation-facing interpretation documentation.
- One stable documentation reference for Story 5 closure semantics.

**Risks / rollback**
- Risk: unclear handoff can cause Story 6 to reinterpret Story 5 semantics.
- Rollback/mitigation: encode Story 5 output contract and handoff expectations in
  one stable doc location.

### Engineering Task 8: Run Story 5 Validation And Confirm Interpretation Guardrail Readiness

**Implements story**
- `Story 5: Optional, Unsupported, And Incomplete Workflow Evidence Cannot Be Counted As Task 6 Completion`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 5 tests pass.
- Story 5 interpretation artifact command runs and emits stable output.
- Story 5 closure is backed by rerunnable machine-readable evidence.

**Execution checklist**
- [ ] Run focused Story 5 interpretation regression tests.
- [ ] Run Story 5 artifact emission command and verify output.
- [ ] Confirm closure signal and excluded evidence-class fields are present.
- [ ] Record test and artifact references for Story 6 handoff.

**Evidence produced**
- Passing Story 5 focused tests.
- Stable Story 5 interpretation bundle reference.

**Risks / rollback**
- Risk: Story 5 can appear complete without executable proof that guardrails are
  actually enforced.
- Rollback/mitigation: require tests plus emitted interpretation artifact as exit
  evidence.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one stable Story 5 interpretation vocabulary and closure rule is frozen,
- Task 6 completion signal is computed only from mandatory complete supported
  evidence,
- optional, unsupported, deferred, and incomplete evidence classes are explicit
  and excluded from completion by rule,
- missing mandatory evidence or malformed statuses produce incomplete closure
  state rather than partial success,
- one stable Story 5 artifact or rerunnable command emits interpretation
  semantics in machine-readable form,
- and publication-facing evidence packaging remains clearly assigned to Story 6.

## Implementation Notes

- Use Task 5 Story 5 implementation as the baseline pattern:
  `benchmarks/density_matrix/validation_evidence/interpretation_validation.py`.
- Preserve the emitted Story 1 workflow identity and contract version in Story 5
  summaries so closure is computed against one named canonical workflow rather
  than an inferred bundle family.
- Reuse support-tier and unsupported fields from existing helpers rather than
  introducing Task 6-only synonyms.
- Keep Story 5 as an interpretation layer over Stories 1 to 4 outputs; do not
  duplicate low-level validation logic in this story.
- Preserve stable class-specific summary fields so Story 6 can package them
  directly without reinterpretation.
