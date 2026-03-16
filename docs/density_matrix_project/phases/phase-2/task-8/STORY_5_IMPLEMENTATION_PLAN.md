# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale
Honestly

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns the Task 8 scope-honesty requirement into one explicit
publication wording gate for the supported density-matrix path:

- the guaranteed VQE-facing density path is described clearly as the supported
  Paper 1 path rather than as a proxy for broader standalone capability,
- the exact-regime scale contract is documented honestly for 4 / 6 end-to-end
  execution and 8 / 10 benchmark-ready evaluation, with a documented 10-qubit
  acceptance anchor,
- broader circuit parity, broader workflow breadth, or broad scaling claims are
  kept outside the Paper 1 result,
- and the resulting supported-path layer stays narrow enough that Story 6 can
  focus on future-work framing while Story 7 can package the final terminology
  and reviewer-navigation bundle.

Out of scope for this story:

- redefining the Paper 1 claim package already frozen by Story 1,
- cross-surface alignment already owned by Story 2,
- claim-to-source traceability already owned by Story 3,
- evidence-floor and claim-closure interpretation already owned by Story 4,
- future-work and publication-ladder framing owned by Story 6,
- and final bundle-level terminology and navigation closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 to 4 are already in place and provide the Paper 1 claim package,
  publication-surface alignment, traceability, and evidence-closure rules.
- Story 4 is expected to emit
  `benchmarks/density_matrix/artifacts/publication_claim_package/evidence_closure_bundle.json`
  or an equivalent rerunnable checker. Story 5 should assume the evidence bar is
  already frozen and focus on supported-path wording.
- Supported-path semantics are already frozen by:
  - `TASK_1_MINI_SPEC.md`,
  - `TASK_2_MINI_SPEC.md`,
  - `TASK_3_MINI_SPEC.md`,
  - `TASK_4_MINI_SPEC.md`,
  - `TASK_6_MINI_SPEC.md`,
  - `PHASE_2_DOCUMENTATION_INDEX.md`,
  - and `P2-ADR-009` through `P2-ADR-013`.
- The canonical machine-readable workflow evidence surface remains
  `workflow_publication_bundle.json`, and the documentation entry surface
  remains `PHASE_2_DOCUMENTATION_INDEX.md`.
- Story 5 should reuse existing supported-path vocabulary rather than inventing
  a new description of the Phase 2 workflow.
- Story 5 defines publication wording for the supported path and exact regime.
  It should not widen support or introduce new benchmark claims.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Supported-Path And Exact-Regime Wording Inventory

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 names one stable inventory of required statements about the supported
  VQE-facing density path.
- The inventory covers backend selection, workflow identity, supported-path
  attribution, exact-regime scale, and acceptance-anchor wording.
- The inventory stays aligned with the frozen Phase 2 contract.

**Execution checklist**
- [ ] Freeze one canonical list of required Story 5 statements for supported-path
      and exact-regime honesty.
- [ ] Include explicit `density_matrix` selection, no silent fallback, canonical
      noisy XXZ plus `HEA` workflow identity, and exact `Re Tr(H*rho)` wording.
- [ ] Include the accepted scale wording for 4 / 6 full end-to-end execution,
      8 / 10 benchmark-ready evaluation, and the 10-qubit acceptance anchor.
- [ ] Record the inventory in one stable Story 5 planning surface that later
      Task 8 stories can reference directly.

**Evidence produced**
- One stable Story 5 supported-path and exact-regime wording inventory.
- One reviewable mapping from required statement to the frozen contract source.

**Risks / rollback**
- Risk: if required wording stays implicit, publication prose can sound
  technically strong while still overstating breadth or scale.
- Rollback/mitigation: freeze the required supported-path statements explicitly
  and validate them directly.

### Engineering Task 2: Reuse Existing Backend, Observable, Bridge, And Workflow Vocabulary Without Renaming It

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 reuses existing supported-path vocabulary wherever practical.
- Readers can trace wording directly to Task 1 through Task 6 documents and the
  Phase 2 documentation index without schema translation.
- Story 5 remains a scope-honesty layer, not a second contract language.

**Execution checklist**
- [ ] Reuse the existing backend labels `state_vector` and `density_matrix`
      exactly.
- [ ] Reuse the exact-observable wording rooted in `Re Tr(H*rho)` rather than
      adding broader observable phrases.
- [ ] Reuse the canonical workflow labels from Task 6 and the Phase 2
      documentation index where they already match the frozen contract.
- [ ] Avoid introducing Story 5-only synonyms for supported-path identity, exact
      regime, or acceptance-anchor status.

**Evidence produced**
- One Story 5 wording layer rooted in canonical Phase 2 supported-path
  vocabulary.
- Reviewable traceability from publication wording to existing contract terms.

**Risks / rollback**
- Risk: renamed terminology can silently blur the distinction between one
  supported workflow and broader noisy-backend capability.
- Rollback/mitigation: preserve the frozen supported-path vocabulary unless a
  stronger clarity reason requires a very small, explicitly mapped synonym set.

### Engineering Task 3: Encode Coverage Rules For Mandatory Supported-Path And Exact-Regime Statements

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 makes explicit which required supported-path and scale statements must
  appear in Paper 1.
- Missing mandatory statements block Story 5 closure.
- Coverage rules are stable enough for fast checks and later bundle review.

**Execution checklist**
- [ ] Mark explicit `density_matrix` selection, no silent fallback, canonical
      XXZ plus `HEA` workflow identity, supported VQE-facing path attribution,
      exact-regime size contract, and 10-qubit anchor status as mandatory
      covered statements.
- [ ] Distinguish mandatory supported-path statements from optional explanatory
      detail or comparative context.
- [ ] Add explicit completeness rules so missing supported-path or size-boundary
      statements fail Story 5 review.
- [ ] Keep coverage rules machine-checkable where practical.

**Evidence produced**
- One explicit completeness rule set for Story 5 required statements.
- One reviewable list of mandatory supported-path and exact-regime claims.

**Risks / rollback**
- Risk: publication prose can sound polished while omitting one critical scope
  boundary, such as no-fallback behavior or 10-qubit anchor meaning.
- Rollback/mitigation: treat missing required statements as a real Story 5
  failure instead of as a doc-nit.

### Engineering Task 4: Encode Forbidden Over-Broad Wording For Standalone Capability, Circuit Parity, And Scaling

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 names one explicit inventory of over-broad or misleading wording that
  must not appear as positive Paper 1 scope.
- The inventory covers broader standalone `NoisyCircuit` breadth, full
  `qgd_Circuit` parity, generic noisy-backend breadth, and broad scaling claims.
- High-risk overclaim patterns are treated as Story 5 failure conditions.

**Execution checklist**
- [ ] Define explicit forbidden wording classes for full `qgd_Circuit` parity,
      broad manual circuit reuse, generic noisy-backend generality, and scaling
      claims beyond the documented exact regime.
- [ ] Define forbidden wording that makes standalone `NoisyCircuit` capability
      sound equivalent to guaranteed VQE-facing support.
- [ ] Define forbidden wording that treats the 10-qubit anchor as a broad
      scalability result.
- [ ] Keep the overclaim inventory reviewable and machine-checkable where
      practical.

**Evidence produced**
- One Story 5 forbidden-wording inventory for supported-path overclaim risk.
- One reviewable list of high-risk scope-inflation patterns.

**Risks / rollback**
- Risk: a single over-broad sentence can undo careful evidence and claim-boundary
  work elsewhere in the paper.
- Rollback/mitigation: define the forbidden wording classes directly and check
  them as part of Story 5 completion.

### Engineering Task 5: Add Focused Regression Checks For Ambiguous Or Over-Broad Supported-Path Wording

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing required supported-path statements, missing
  exact-regime bounds, or over-broad capability wording.
- Negative cases show that Story 5 fails if broader capability is described as
  guaranteed Paper 1 support.
- Regression coverage remains small and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_publication_docs.py`
      or a tightly related successor for Story 5 wording coverage.
- [ ] Add negative checks for omitted explicit `density_matrix` selection, omitted
      no-fallback wording, or omitted canonical workflow identity.
- [ ] Add at least one check for missing 4 / 6 / 8 / 10 exact-regime wording or
      missing 10-qubit anchor status.
- [ ] Add at least one check for broader standalone capability or circuit parity
      being described as the guaranteed Paper 1 path.

**Evidence produced**
- Focused regression coverage for Story 5 supported-path and exact-regime
  wording.
- Reviewable failures when required scope boundaries are missing or inflated.

**Risks / rollback**
- Risk: wording regressions may not break any code and therefore can survive
  until paper review.
- Rollback/mitigation: lock the supported-path and exact-regime statements down
  with targeted checks.

### Engineering Task 6: Emit One Stable Story 5 Supported-Path Scope Summary Or Checker

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 can emit one stable machine-readable supported-path summary or one
  stable rerunnable checker.
- The output records the mandatory supported-path statements, size bounds, and
  overclaim exclusions.
- The output is stable enough for Story 7 and publication review to consume.

**Execution checklist**
- [ ] Add one Story 5 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for supported-path scope
      summary emission.
- [ ] Emit one stable artifact in a Task 8 artifact directory such as
      `benchmarks/density_matrix/artifacts/publication_claim_package/supported_path_scope_bundle.json`.
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to supported-path and exact-regime wording rather
      than future-work or terminology bundle closure.

**Evidence produced**
- One stable Task 8 Story 5 supported-path scope summary or rerunnable checker.
- One reusable Story 5 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: ad hoc scope wording is difficult to compare across evolving paper
  surfaces.
- Rollback/mitigation: define one thin structured Story 5 output that makes the
  supported path and size boundary explicit.

### Engineering Task 7: Document Story 5 Scope Rules And Handoff To Story 6

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 5 validates, how to rerun it, and
  why it is the canonical supported-path honesty gate for Task 8.
- The notes make clear that Story 5 closes supported-path wording but not
  future-work framing or final terminology packaging.
- The documentation stays aligned with the frozen Phase 2 scope boundary.

**Execution checklist**
- [ ] Document the Story 5 summary or checker and how it relates to Task 1
      through Task 6 plus the Phase 2 documentation index.
- [ ] Make the main Story 5 rule explicit:
      Paper 1 must describe the supported VQE-facing density path and exact
      regime honestly.
- [ ] Explain how Story 5 hands off future-work framing to Story 6.
- [ ] Keep unsupported or broader standalone capability clearly outside Story 5
      closure.

**Evidence produced**
- Updated developer-facing guidance for the Task 8 Story 5 scope-honesty gate.
- One stable location where Story 5 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 5 is poorly documented, later future-work prose can start doing
  the job of supported-path clarification badly.
- Rollback/mitigation: document Story 5 as the explicit supported-path gate and
  keep its handoff boundary visible.

### Engineering Task 8: Run Story 5 Validation And Confirm Honest Supported-Path Wording

**Implements story**
- `Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 5 wording checks pass.
- The Story 5 summary or checker runs successfully and emits stable output.
- Story 5 completion is backed by rerunnable scope evidence rather than by
  polished prose alone.

**Execution checklist**
- [ ] Run focused Story 5 regression checks for supported-path and exact-regime
      coverage.
- [ ] Run the Story 5 summary or checker command and verify emitted output.
- [ ] Confirm that explicit backend selection, no-fallback behavior, canonical
      workflow identity, exact-regime bounds, and 10-qubit anchor status are
      present.
- [ ] Record stable test and artifact references for Story 6 and later Task 8
      work.

**Evidence produced**
- Passing focused checks for Story 5 supported-path completeness.
- One stable Story 5 output proving the Paper 1 supported path and exact-regime
  scope are documented honestly.

**Risks / rollback**
- Risk: Story 5 can appear complete while still allowing broad capability or
  scaling implications to survive in publication prose.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 5.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one stable wording inventory defines the mandatory supported-path and
  exact-regime statements for Paper 1,
- the supported VQE-facing density path is described with explicit backend
  selection, no-fallback behavior, and canonical workflow identity,
- the 4 / 6 / 8 / 10 exact-regime contract and 10-qubit anchor status are
  documented honestly,
- broader standalone capability, broader circuit parity, or broad scaling claims
  fail Story 5 wording checks,
- one stable Story 5 output or rerunnable checker captures the supported-path
  publication surface,
- and future-work framing and final publication-package bundle closure remain
  clearly assigned to Stories 6 and 7.

## Implementation Notes

- Prefer to reuse Task 1 through Task 6 language and the Phase 2 documentation
  index rather than inventing a new publication scope vocabulary.
- Keep the exact-regime wording bounded and honest. Phase 2 is not the place to
  imply a broad scaling result from the 10-qubit anchor.
- Treat broader standalone `NoisyCircuit` capability as a high-risk overclaim
  zone. Story 5 should explicitly prevent that confusion.
- Missing no-fallback wording is a real contract failure. It weakens both
  technical review and scientific attribution.
- If Story 5 is strong, later future-work discussion can build on a stable
  supported-path boundary instead of quietly replacing it.
