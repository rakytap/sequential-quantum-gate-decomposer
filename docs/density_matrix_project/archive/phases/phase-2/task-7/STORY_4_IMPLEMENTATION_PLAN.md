# Story 4 Implementation Plan

## Story Being Implemented

Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core
Phase 2 Claims

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns the frozen validation and workflow-evidence package into one
explicit documentation gate for what counts as a core Phase 2 claim:

- the mandatory evidence package is named explicitly rather than inferred from
  favorable examples,
- numeric thresholds, pass-rate expectations, runtime and peak-memory recording,
  and reproducibility expectations are described as required evidence rather than
  optional polish,
- optional, unsupported, and incomplete evidence are kept visible without being
  allowed to replace the main claim,
- and the resulting evidence-bar layer stays narrow enough that Story 5 can
  separate future work without re-explaining what the current evidence floor is.

Out of scope for this story:

- rerunning or redefining the validation package itself,
- broad support-surface classification owned by Story 3,
- future-work and non-goal separation owned by Story 5,
- and final terminology and cross-reference bundle closure owned by Story 6.

## Dependencies And Assumptions

- Story 3 already closes the support-surface classification that tells readers
  what kinds of cases count as required, optional, deferred, or unsupported.
- The mandatory validation and benchmark contract is already frozen in:
  - `TASK_5_MINI_SPEC.md`,
  - `TASK_5_STORIES.md`,
  - `P2-ADR-014`,
  - `P2-ADR-015`,
  - and the benchmark-minimum and numeric-threshold closures in
    `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- The canonical workflow evidence surface is already frozen in:
  - `TASK_6_MINI_SPEC.md`,
  - `TASK_6_STORIES.md`,
  - and the Task 6 publication bundle under
    `benchmarks/density_matrix/artifacts/workflow_evidence/`.
- Story 1 and later Task 7 stories should reuse one shared reader-facing entry
  surface in
  `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`.
  Story 4 should refine that same entry surface with evidence-bar guidance
  rather than create a disconnected evidence note.
- The Phase 2 paper-facing documents already mention the delivered evidence
  package. Story 4 should make that evidence floor explicit and auditable rather
  than inventing a separate paper-only interpretation.
- Story 4 is a documentation and interpretation layer only. It does not redefine
  thresholds, reclassify evidence, or add new mandatory workloads.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Story 4 Evidence Inventory And Claim Rules

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 names one canonical inventory of mandatory evidence required for core
  Phase 2 claims.
- The inventory covers the mandatory validation layers, workflow layers, anchor
  cases, thresholds, and reproducibility expectations.
- The inventory remains aligned with the frozen Task 5 and Task 6 contract.

**Execution checklist**
- [ ] Freeze one canonical Story 4 inventory covering:
      1 to 3 qubit micro-validation, 4 / 6 / 8 / 10 workflow matrix, at least
      one reproducible 4- or 6-qubit optimization trace, a documented 10-qubit
      anchor case, runtime and peak-memory recording, and the reproducibility
      bundle.
- [ ] Mark which evidence items are mandatory versus optional supplemental.
- [ ] Record the required threshold and pass-rate expectations attached to the
      mandatory evidence inventory.
- [ ] Keep the inventory rooted in the existing Task 5 and Task 6 contract.

**Evidence produced**
- One stable Story 4 evidence inventory and claim-rule mapping.
- One reviewable mapping from mandatory evidence items to frozen contract
  decisions.

**Risks / rollback**
- Risk: without an explicit evidence inventory, documentation can drift toward
  narrative-only claims supported by unclear subsets of evidence.
- Rollback/mitigation: freeze one explicit evidence inventory and use it
  everywhere Story 4 talks about closure.

### Engineering Task 2: Reuse Existing Task 5 And Task 6 Artifact Vocabulary Without Renaming It

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 reuses the current Task 5 and Task 6 evidence vocabulary wherever
  practical.
- Readers can trace evidence-bar wording directly to current bundles and task
  documents without schema translation.
- Story 4 remains a documentation gate over existing evidence, not a new bundle
  taxonomy.

**Execution checklist**
- [ ] Reuse existing labels for micro-validation matrix, workflow matrix,
      optimization trace, anchor case, reproducibility bundle, and pass/fail
      semantics where practical.
- [ ] Reuse current threshold language
      (`<= 1e-10`, `<= 1e-8`, `100%` pass rate, `rho.is_valid(tol=1e-10)`,
      `|Tr(rho) - 1| <= 1e-10`, `|Im Tr(H*rho)| <= 1e-10`) exactly.
- [ ] Reuse existing provenance and status language from the Task 5 and Task 6
      artifacts rather than paraphrasing it loosely.
- [ ] Avoid introducing Story 4-only evidence labels when a stable term already
      exists.

**Evidence produced**
- One Story 4 evidence-bar layer rooted in canonical Task 5 and Task 6
  vocabulary.
- Reviewable traceability from documentation claims to existing evidence fields.

**Risks / rollback**
- Risk: renamed evidence labels can make a stable validation package look like a
  different or broader claim.
- Rollback/mitigation: preserve existing evidence vocabulary unless a strictly
  smaller clarification layer is required.

### Engineering Task 3: Add Explicit Completeness And Status Rules For The Mandatory Evidence Package

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines which mandatory evidence items must be present for the core
  Phase 2 claim to remain valid.
- Missing mandatory items or missing status semantics fail Story 4 review.
- The completeness rules are stable enough for fast checks and later bundle
  packaging.

**Execution checklist**
- [ ] Mark the mandatory evidence layers and status expectations as explicit
      Story 4 completeness requirements.
- [ ] Require threshold and pass-rate wording to appear alongside the mandatory
      evidence package, not only in buried references.
- [ ] Require runtime, peak-memory, and reproducibility expectations to remain
      visible where Story 4 summarizes the evidence floor.
- [ ] Keep the completeness rules machine-checkable where practical.

**Evidence produced**
- One explicit Story 4 completeness rule set for mandatory evidence coverage.
- One reviewable list of required evidence-bar statements and statuses.

**Risks / rollback**
- Risk: documentation can mention the evidence package while still omitting one
  of the mandatory closure conditions.
- Rollback/mitigation: treat missing evidence-bar elements as Story 4 failure
  rather than as editorial cleanup.

### Engineering Task 4: Preserve Mandatory, Optional, Unsupported, And Incomplete Evidence Distinctions In Story 4 Outputs

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 outputs keep the main claim tied only to mandatory evidence.
- Optional, unsupported, and incomplete evidence remain visible but clearly
  excluded from the core claim.
- Reviewers can tell from Story 4 outputs why a given evidence class counts or
  does not count toward closure.

**Execution checklist**
- [ ] Preserve explicit distinction between mandatory evidence and optional
      supplemental evidence.
- [ ] Preserve explicit unsupported and incomplete evidence categories where they
      affect how claims should be read.
- [ ] Keep numeric thresholds, pass-rate rules, and provenance expectations tied
      to mandatory evidence rather than flattening them into generic
      "validation" language.
- [ ] Avoid wording that lets a favorable example or partial bundle stand in for
      the frozen evidence floor.

**Evidence produced**
- Story 4 outputs that distinguish mandatory, optional, unsupported, and
  incomplete evidence classes.
- Reviewable structured explanations of why the main claim rests only on the
  mandatory package.

**Risks / rollback**
- Risk: even accurate evidence references can be overread if exclusion semantics
  are not visible.
- Rollback/mitigation: keep the exclusion logic explicit in structured Story 4
  outputs, not only in prose footnotes.

### Engineering Task 5: Add Focused Regression Checks For Story 4 Evidence-Bar Semantics

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing mandatory evidence categories, missing threshold
  wording, or wording that allows optional or partial evidence to inflate the
  main claim.
- Negative cases show that Story 4 fails if the evidence floor becomes vague.
- Regression coverage remains documentation-focused and lightweight.

**Execution checklist**
- [ ] Add focused Story 4 checks in `tests/density_matrix/test_phase2_docs.py`
      or a tightly related successor.
- [ ] Add negative checks for missing micro-validation, missing workflow-matrix,
      missing optimization-trace or anchor wording, or missing reproducibility
      references.
- [ ] Add at least one check that fails when thresholds or `100%` pass-rate
      expectations are omitted from the Story 4 evidence-bar surface.
- [ ] Keep actual benchmark execution outside this fast documentation check
      layer.

**Evidence produced**
- Focused regression coverage for Story 4 evidence-bar semantics.
- Reviewable failure messages for missing or inflated evidence-bar statements.

**Risks / rollback**
- Risk: evidence-bar regressions can survive because they change how results are
  described, not how they are computed.
- Rollback/mitigation: lock the mandatory evidence-bar statements down with
  focused checks.

### Engineering Task 6: Emit One Stable Story 4 Evidence-Bar Summary Or Rerunnable Checker

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 can emit one stable machine-readable evidence-bar summary or one stable
  rerunnable checker.
- The output records mandatory evidence classes, thresholds, status expectations,
  and exclusion semantics for optional or incomplete evidence.
- The output is stable enough for Story 5 and Story 6 to consume directly.

**Execution checklist**
- [ ] Add one Story 4 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for evidence-bar summary
      emission.
- [ ] Emit one stable artifact in a Task 7 artifact directory
      (for example `benchmarks/density_matrix/artifacts/documentation_contract/`).
- [ ] Record source references, generation command, and evidence metadata with
      the emitted output.
- [ ] Keep the output narrow to evidence-bar semantics rather than broader
      roadmap or terminology work.

**Evidence produced**
- One stable Task 7 Story 4 evidence-bar summary or rerunnable checker.
- One reusable Story 4 output schema for later Task 7 handoffs.

**Risks / rollback**
- Risk: the evidence floor can remain spread across many docs and become hard to
  audit consistently.
- Rollback/mitigation: emit one thin structured Story 4 surface that makes the
  claim boundary explicit and reusable.

### Engineering Task 7: Document The Story 4 Evidence Gate And Handoff To Story 5

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 4 validates, how to rerun it, and
  why it is the canonical documentation gate for the Phase 2 evidence floor.
- The notes make clear that Story 4 closes the current evidence bar but not the
  roadmap and future-work separation owned by Story 5.
- The documentation stays aligned with the frozen Task 5 and Task 6 claim
  boundary.

**Execution checklist**
- [ ] Document the Story 4 summary or checker and its relationship to Task 5 and
      Task 6 evidence surfaces.
- [ ] Make the main Story 4 rule explicit:
      only the mandatory evidence package supports the core Phase 2 claim.
- [ ] Explain how Story 4 hands off current-versus-future boundary work to Story
      5.
- [ ] Keep optional, unsupported, and incomplete evidence visibly distinct in the
      notes.

**Evidence produced**
- Updated developer-facing guidance for the Task 7 Story 4 evidence gate.
- One stable location where Story 4 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 4 is poorly documented, future-work notes may be asked to
  compensate for unclear current evidence requirements.
- Rollback/mitigation: document Story 4 as the explicit evidence-floor gate
  before Story 5 begins.

### Engineering Task 8: Run Story 4 Validation And Confirm Mandatory Evidence-Bar Clarity

**Implements story**
- `Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 4 evidence-bar checks pass.
- The Story 4 summary or checker runs successfully and emits stable output.
- Story 4 closure is backed by rerunnable documentation evidence rather than by
  prose-only interpretation.

**Execution checklist**
- [ ] Run focused Story 4 regression checks for mandatory evidence-bar coverage.
- [ ] Run the Story 4 summary or checker command and verify emitted output.
- [ ] Confirm mandatory evidence classes, thresholds, pass-rate expectations,
      and exclusion semantics are present and complete.
- [ ] Record stable test and artifact references for Story 5 and later Task 7
      work.

**Evidence produced**
- Passing focused checks for Story 4 evidence-bar semantics.
- One stable Story 4 output proving the mandatory evidence floor is explicit and
  reviewable.

**Risks / rollback**
- Risk: Story 4 can look complete while still allowing favorable subsets or
  partial bundles to blur the main claim.
- Rollback/mitigation: require both passing checks and one stable emitted
  evidence-bar surface before closure.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one stable evidence inventory defines the mandatory package required for the
  core Phase 2 claim,
- thresholds, pass-rate expectations, runtime and peak-memory recording, and
  reproducibility expectations are documented as mandatory evidence attributes,
- optional, unsupported, and incomplete evidence remain explicitly visible but
  excluded from the main claim,
- missing mandatory evidence-bar statements fail Story 4 completeness checks,
- one stable Story 4 output or rerunnable checker captures the evidence-bar
  surface in structured form,
- and future-work separation and final terminology / cross-reference bundle
  closure remain clearly assigned to Stories 5 and 6.

## Implementation Notes

- Keep Story 4 tightly grounded in the Task 5 and Task 6 evidence bundle rather
  than in aspirational publication wording.
- Reuse `tests/density_matrix/test_phase2_docs.py` as the default fast
  documentation-regression surface unless Story 4 uncovers a strong reason to
  split bundle-specific tests away from it.
- Prefer explicit claim rules over broad phrases like "well validated" or
  "publication ready." Readers need to know exactly which evidence classes are
  mandatory.
- Treat thresholds and pass-rate expectations as first-class documentation
  requirements. If they disappear from the visible evidence-bar surface, the
  claim becomes softer than the frozen contract.
- Keep Story 4 separate from future-work discussion. First close the current
  evidence floor, then clarify what belongs to later phases.
