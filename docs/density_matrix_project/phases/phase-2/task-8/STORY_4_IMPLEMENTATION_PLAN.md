# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper
1 Claim

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 evidence-closure rule into one explicit publication
gate for Paper 1:

- the mandatory evidence floor is stated as the only basis for closing the main
  Paper 1 claim,
- optional, deferred, unsupported, or incomplete evidence remains visible but
  cannot be counted as milestone-closing proof,
- the Task 5 and Task 6 evidence package is reused as the authoritative evidence
  surface rather than replaced by persuasive prose,
- and the resulting closure layer stays narrow enough that Story 5 can focus on
  supported-path honesty and Story 7 can focus on top-level publication-package
  integrity.

Out of scope for this story:

- redefining the Paper 1 claim package already frozen by Story 1,
- cross-surface alignment already owned by Story 2,
- claim-to-source mapping already owned by Story 3,
- supported-path and exact-regime wording owned by Story 5,
- future-work and publication-ladder framing owned by Story 6,
- and final terminology plus reviewer-navigation bundle closure owned by Story
  7.

## Dependencies And Assumptions

- Stories 1 to 3 are already in place and provide the Paper 1 claim package,
  cross-surface alignment, and claim traceability surfaces.
- Story 3 is expected to emit
  `benchmarks/density_matrix/artifacts/phase2_task8/story3_claim_traceability_bundle.json`
  or an equivalent rerunnable checker. Story 4 should build on that traceability
  surface rather than recreate it.
- The canonical machine-readable evidence surface already exists at
  `benchmarks/density_matrix/artifacts/phase2_task6/task6_story6_publication_bundle.json`
  and preserves traceability to Task 5 validation layers.
- The mandatory evidence floor is already frozen by:
  - `TASK_5_MINI_SPEC.md`,
  - `TASK_6_MINI_SPEC.md`,
  - `DETAILED_PLANNING_PHASE_2.md`,
  - `P2-ADR-014`,
  - and `P2-ADR-015`.
- The mandatory closure rule is already frozen in the Phase 2 paper-facing docs:
  only mandatory, complete, supported evidence may close the main claim.
- Story 4 should define publication evidence semantics and closure rules only. It
  should not generate new benchmark evidence or alter the frozen thresholds.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Mandatory-Evidence And Claim-Closure Inventory

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 names one stable inventory of mandatory evidence items for Paper 1.
- Story 4 names one stable inventory of claim-closure rules for that evidence.
- The inventory stays aligned with the frozen Task 5 and Task 6 evidence
  contract.

**Execution checklist**
- [ ] Freeze one canonical list of mandatory evidence items required for the
      Paper 1 main claim.
- [ ] Freeze one explicit set of closure rules for mandatory, complete,
      supported evidence.
- [ ] Keep the inventory rooted in existing Phase 2 validation and workflow
      evidence rather than in ad hoc paper phrasing.
- [ ] Record the inventory in one stable Story 4 planning surface that later
      Task 8 stories can reference directly.

**Evidence produced**
- One stable Story 4 mandatory-evidence and claim-closure inventory.
- One reviewable mapping from evidence item to closure rule.

**Risks / rollback**
- Risk: if the evidence floor remains implicit, polished publication prose can
  overclose the main claim with too little evidence.
- Rollback/mitigation: freeze the mandatory evidence floor and closure rules
  explicitly before later manuscript editing proceeds.

### Engineering Task 2: Reuse Task 5 And Task 6 Evidence Semantics Without Rewriting Them

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 reuses the existing Task 5 and Task 6 evidence semantics wherever
  practical.
- Readers can trace closure logic directly to the frozen validation and workflow
  contract.
- Story 4 remains an evidence-closure layer, not a second evidence vocabulary.

**Execution checklist**
- [ ] Reuse the mandatory micro-validation, workflow-matrix, optimization-trace,
      acceptance-anchor, runtime, peak-memory, and reproducibility terminology
      already frozen by Task 5 and Task 6.
- [ ] Reuse the existing meaning of required, optional, deferred, unsupported,
      complete, and incomplete evidence.
- [ ] Reuse the Task 6 publication bundle as the top-level workflow-evidence
      surface and let it preserve linkage to Task 5 validation layers.
- [ ] Avoid introducing Story 4-only synonyms for evidence classes or closure
      semantics.

**Evidence produced**
- One Story 4 closure layer rooted in canonical Task 5 and Task 6 evidence
  language.
- Reviewable traceability from Paper 1 closure wording to the frozen evidence
  contract.

**Risks / rollback**
- Risk: renamed or paraphrased evidence classes can silently weaken the meaning
  of “mandatory” or “complete.”
- Rollback/mitigation: preserve the frozen evidence vocabulary unless a very
  small, explicitly mapped clarity improvement is required.

### Engineering Task 3: Encode Explicit Closure Rules For Mandatory, Optional, Deferred, Unsupported, And Incomplete Evidence

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 records one explicit closure rule for each evidence class relevant to
  Paper 1 interpretation.
- The rules make it impossible to confuse supplemental or boundary evidence with
  milestone-closing evidence.
- Missing or contradictory closure rules block Story 4 completion.

**Execution checklist**
- [ ] Define the mandatory closure rule for complete supported evidence.
- [ ] Define explicit non-closure rules for optional, deferred, unsupported, and
      incomplete evidence.
- [ ] Distinguish supplemental context from positive claim closure.
- [ ] Keep the rules machine-checkable where practical.

**Evidence produced**
- One explicit Story 4 closure-rule set for all relevant evidence classes.
- One reviewable list of allowed and forbidden closure semantics.

**Risks / rollback**
- Risk: evidence-class labels can remain present while still being interpreted
  inconsistently across paper surfaces.
- Rollback/mitigation: encode the closure semantics explicitly instead of
  assuming readers will infer them correctly.

### Engineering Task 4: Preserve The Full Mandatory Evidence Floor In Publication-Facing Summaries

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 preserves the entire mandatory evidence floor in publication-facing
  summaries.
- No mandatory evidence component silently disappears when the paper is
  compressed.
- Evidence-floor summaries remain stable enough for reviewer audit.

**Execution checklist**
- [ ] Keep 1 to 3 qubit micro-validation, 4 / 6 / 8 / 10 workflow matrix, at
      least one reproducible 4- or 6-qubit optimization trace, one documented
      10-qubit anchor case, runtime and peak-memory recording, and the
      reproducibility bundle visible in the Story 4 inventory.
- [ ] Preserve the `100%` pass interpretation for mandatory cases and the frozen
      numeric-threshold semantics in closure-oriented paper summaries.
- [ ] Distinguish mandatory evidence-floor statements from optional context or
      venue-shaping narrative.
- [ ] Treat omitted mandatory evidence-floor items as a Story 4 failure.

**Evidence produced**
- Story 4 outputs that preserve the complete mandatory evidence floor.
- Reviewable wording showing how publication summaries retain the full evidence
  bar.

**Risks / rollback**
- Risk: compressed paper surfaces can accidentally turn a full evidence package
  into a highlight reel.
- Rollback/mitigation: treat every mandatory evidence-floor element as a required
  publication-closure input.

### Engineering Task 5: Add Focused Regression Checks For Evidence Inflation And Incomplete Closure

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing mandatory evidence-floor statements, relaxed closure
  semantics, or optional evidence being counted as the main claim.
- Negative cases show that Story 4 fails if incomplete or unsupported evidence
  is described as sufficient.
- Regression coverage remains small and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 4 evidence semantics.
- [ ] Add negative checks for missing mandatory evidence-floor items or missing
      closure-rule statements.
- [ ] Add at least one check for optional or unsupported evidence being counted
      as positive main-claim closure.
- [ ] Keep benchmark execution and numeric recomputation outside this fast
      publication-check layer.

**Evidence produced**
- Focused regression coverage for Story 4 evidence-closure completeness.
- Reviewable failures when optional or incomplete evidence inflates the main
  Paper 1 claim.

**Risks / rollback**
- Risk: evidence inflation often appears as a wording issue until it damages
  scientific review.
- Rollback/mitigation: lock the closure semantics down with targeted checks.

### Engineering Task 6: Emit One Stable Story 4 Evidence-Closure Manifest Or Checker

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 can emit one stable machine-readable evidence-closure summary or one
  stable rerunnable checker.
- The output records the mandatory evidence floor, evidence classes, and closure
  status rules.
- The output is stable enough for Story 7 and reviewer-facing publication reuse.

**Execution checklist**
- [ ] Add one Story 4 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for evidence-closure
      summary emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/phase2_task8/story4_evidence_closure_bundle.json`.
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to evidence-floor and claim-closure semantics
      rather than full supported-path wording.

**Evidence produced**
- One stable Task 8 Story 4 evidence-closure manifest or rerunnable checker.
- One reusable Story 4 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: ad hoc evidence summaries are easy to misread and hard to validate
  consistently across evolving paper surfaces.
- Rollback/mitigation: define one thin structured Story 4 output that makes
  closure semantics explicit and rerunnable.

### Engineering Task 7: Document Story 4 Evidence Rules And Handoff To Story 5

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 4 validates, how to rerun it, and
  why it is the canonical evidence-closure gate for Task 8.
- The notes make clear that Story 4 closes evidence semantics but not
  supported-path wording or future-work framing.
- The documentation stays aligned with the frozen Phase 2 evidence contract.

**Execution checklist**
- [ ] Document the Story 4 manifest or checker and how it relates to Task 5 and
      Task 6 evidence surfaces.
- [ ] Make the main Story 4 rule explicit:
      only mandatory, complete, supported evidence closes the main Paper 1
      claim.
- [ ] Explain how Story 4 hands off supported-path wording to Story 5.
- [ ] Keep optional, deferred, unsupported, and incomplete evidence clearly
      outside positive closure.

**Evidence produced**
- Updated developer-facing guidance for the Task 8 Story 4 evidence gate.
- One stable location where Story 4 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 4 is poorly documented, later manuscript edits can soften the
  evidence bar without anyone noticing.
- Rollback/mitigation: document Story 4 as the explicit Paper 1 closure gate and
  keep its boundaries visible.

### Engineering Task 8: Run Story 4 Validation And Confirm The Main Paper Claim Closes Only On Mandatory Evidence

**Implements story**
- `Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 4 evidence-closure checks pass.
- The Story 4 manifest or checker runs successfully and emits stable output.
- Story 4 completion is backed by rerunnable evidence semantics rather than by
  persuasive prose alone.

**Execution checklist**
- [ ] Run focused Story 4 regression checks for evidence-floor and closure-rule
      coverage.
- [ ] Run the Story 4 manifest or checker command and verify emitted output.
- [ ] Confirm that mandatory evidence-floor items and non-closure rules for
      supplemental evidence are all present and aligned with the frozen Phase 2
      contract.
- [ ] Record stable test and artifact references for Story 5 and later Task 8
      work.

**Evidence produced**
- Passing focused checks for Story 4 evidence-closure completeness.
- One stable Story 4 output proving the main Paper 1 claim closes only on
  mandatory, complete, supported evidence.

**Risks / rollback**
- Risk: Story 4 can look complete while still allowing inflated closure language
  to survive in paper-facing prose.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 4.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- one stable mandatory-evidence inventory defines the evidence floor for the
  main Paper 1 claim,
- one stable closure-rule inventory defines how mandatory, optional, deferred,
  unsupported, and incomplete evidence must be interpreted,
- the full mandatory evidence floor remains visible in publication-facing
  summaries,
- relaxed closure semantics or missing evidence-floor items fail Story 4 checks,
- one stable Story 4 output or rerunnable checker captures the evidence-closure
  surface,
- and supported-path wording, future-work framing, and publication-package
  bundle closure remain clearly assigned to Stories 5 to 7.

## Implementation Notes

- Treat “mandatory, complete, supported” as an operational rule, not as a slogan.
  Story 4 should make that rule machine-checkable where practical.
- Reuse Task 5 and Task 6 evidence semantics directly. Story 4 should clarify
  closure logic, not create a second evidence taxonomy.
- Keep optional and unsupported evidence visible. The goal is honest exclusion,
  not erasure of boundary evidence.
- Do not let compressed paper surfaces drop the mandatory evidence floor. If a
  surface cannot carry all details, it still needs to preserve the closure rule.
- If Story 4 is strong, later Task 8 review can separate “is this enough
  evidence?” from “can we find the evidence at all?”
