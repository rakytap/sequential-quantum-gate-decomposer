# Story 2 Implementation Plan

## Story Being Implemented

Story 2: The Supported `density_matrix` Entry Surface And Canonical XXZ Workflow
Are Explained Unambiguously

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns the supported-entry and canonical-workflow wording into one
explicit documentation gate for Phase 2:

- the supported entry surface is described in a way that makes backend defaults,
  explicit `density_matrix` selection, and no-fallback behavior visible without
  code inspection,
- the canonical supported workflow is described consistently as noisy XXZ VQE
  with default `HEA`, explicit local noise insertion, and exact
  `Re Tr(H*rho)` evaluation,
- the accepted exact-regime size contract is documented honestly for 4, 6, 8,
  and 10 qubits with a 10-qubit anchor case,
- and the resulting wording layer stays narrow enough that Story 3 can focus on
  support-surface classification rather than re-explaining basic entry and
  workflow semantics.

Out of scope for this story:

- redefining backend semantics already frozen by Task 1 and the Phase 2 ADRs,
- broad support-surface classification across optional, deferred, and
  standalone-only capability owned by Story 3,
- validation-package interpretation owned by Story 4,
- future-work separation owned by Story 5,
- and final terminology and cross-reference bundle closure owned by Story 6.

## Dependencies And Assumptions

- Story 1 already provides the authoritative source-of-truth map for where the
  relevant contract statements live. Story 2 should reuse that map rather than
  rediscover sources from scratch.
- Story 1 now emits
  `benchmarks/density_matrix/task7_story1_contract_reference_validation.py`,
  writing
  `benchmarks/density_matrix/artifacts/phase2_task7/story1_contract_reference_map.json`,
  and establishes
  `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`
  as the stable entry point for the Phase 2 document bundle. Story 2 should
  refine that same entry surface rather than create a competing Phase 2 summary.
- Backend-selection semantics are already frozen in:
  - `TASK_1_MINI_SPEC.md`,
  - `P2-ADR-009`,
  - `DETAILED_PLANNING_PHASE_2.md`,
  - and the closed backend-selection item in
    `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Observable semantics are already frozen in:
  - `TASK_2_MINI_SPEC.md`,
  - `P2-ADR-010`,
  - and the observable-contract closure in the checklist.
- Canonical workflow and exact-regime expectations are already frozen in:
  - `TASK_6_MINI_SPEC.md`,
  - `TASK_6_STORIES.md`,
  - `P2-ADR-013`,
  - `P2-ADR-014`,
  - and `P2-ADR-015`.
- Phase 2 paper-facing documents already describe the canonical workflow. Story
  2 should align those descriptions with the authoritative task and ADR wording
  rather than create a new interpretation.
- Story 2 defines clarity around the supported entry and canonical workflow. It
  should not widen support or invent new workflow examples.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Supported-Entry And Canonical-Workflow Wording Inventory

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 names one canonical inventory of required statements about backend
  entry and canonical workflow behavior.
- The inventory covers backend default, explicit selection, no-fallback
  behavior, canonical workflow identity, and exact-regime scale.
- The inventory stays aligned with the frozen task and ADR contract.

**Execution checklist**
- [ ] Freeze one canonical list of required Story 2 statements for supported
      entry and workflow clarity.
- [ ] Include backend-default wording, explicit `density_matrix` selection, and
      hard-error no-fallback behavior.
- [ ] Include canonical workflow wording for XXZ plus `HEA` plus explicit local
      noise plus exact `Re Tr(H*rho)` evaluation.
- [ ] Include accepted scale wording for 4 / 6 end-to-end execution, 8 / 10
      evaluation-ready behavior, and the 10-qubit anchor case.

**Evidence produced**
- One stable Story 2 wording inventory for supported entry and workflow clarity.
- One reviewable mapping from required statement to the frozen contract source.

**Risks / rollback**
- Risk: if required wording stays implicit, different docs can sound plausible
  while describing different supported behavior.
- Rollback/mitigation: freeze one minimal inventory of mandatory statements and
  validate them directly.

### Engineering Task 2: Reuse Existing Backend, Observable, And Workflow Vocabulary Without Renaming It

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 reuses existing backend, observable, and workflow vocabulary wherever
  practical.
- Readers can trace wording directly to Task 1, Task 2, and Task 6 documents
  without schema translation.
- Story 2 remains a documentation-clarity layer, not a second contract language.

**Execution checklist**
- [ ] Reuse the existing backend labels `state_vector` and `density_matrix`
      exactly.
- [ ] Reuse exact-observable wording rooted in `Re Tr(H*rho)` rather than adding
      broader or alternate phrases.
- [ ] Reuse canonical workflow labels from current Task 6 and paper-facing
      surfaces where they already match the frozen contract.
- [ ] Avoid introducing Story 2-only synonyms for workflow identity, exact
      regime, or no-fallback semantics.

**Evidence produced**
- One Story 2 wording layer rooted in canonical Task 1, Task 2, and Task 6
  vocabulary.
- Reviewable traceability from user-facing guidance to existing contract terms.

**Risks / rollback**
- Risk: renamed or paraphrased terminology can silently weaken no-fallback or
  workflow-anchor clarity.
- Rollback/mitigation: preserve the frozen contract vocabulary unless a stronger
  clarity reason requires a very small, explicitly mapped synonym set.

### Engineering Task 3: Add Explicit Coverage Rules For Required Entry And Workflow Statements

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 makes explicit which required statements must appear in the supported
  entry and workflow guidance.
- Missing mandatory statements block Story 2 closure.
- The coverage rules are stable enough for fast checks and later bundle review.

**Execution checklist**
- [ ] Mark backend default, explicit density selection, no silent fallback,
      canonical XXZ plus `HEA` workflow identity, and exact-regime size contract
      as mandatory covered statements.
- [ ] Add explicit completeness rules so missing required statements fail Story 2
      review.
- [ ] Distinguish required statements from optional contextual explanations or
      examples.
- [ ] Keep coverage rules machine-checkable where practical.

**Evidence produced**
- One explicit completeness rule set for Story 2 required statements.
- One reviewable list of mandatory entry and workflow claims.

**Risks / rollback**
- Risk: documentation can sound polished while omitting one critical boundary,
  such as default backend or 10-qubit anchor status.
- Rollback/mitigation: treat missing required statements as a real Story 2
  failure instead of a doc-nit.

### Engineering Task 4: Preserve Supported-Path Attribution And Exact-Regime Scope In Story 2 Outputs

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 outputs preserve clear attribution to the supported VQE-facing density
  path.
- Exact-regime size claims remain explicit and bounded.
- Broader workflow or scaling claims are excluded from the supported-entry
  surface.

**Execution checklist**
- [ ] Keep explicit linkage between backend selection, VQE-facing execution, and
      canonical XXZ workflow wording.
- [ ] Preserve the distinction between full end-to-end 4 / 6 qubit behavior and
      benchmark-ready 8 / 10 qubit evaluation.
- [ ] Keep the 10-qubit anchor described as an acceptance anchor rather than as
      a broad scaling guarantee.
- [ ] Avoid wording that makes standalone density execution or broader VQA loops
      look like the guaranteed Phase 2 path.

**Evidence produced**
- Story 2 outputs with explicit supported-path attribution and exact-regime
  bounds.
- Reviewable wording showing how the canonical workflow is bounded.

**Risks / rollback**
- Risk: ambiguous scope language can turn one supported workflow into an implied
  family of workflows.
- Rollback/mitigation: keep supported-path attribution and size bounds explicit
  in every Story 2 output.

### Engineering Task 5: Add Focused Regression Checks For Ambiguous Or Over-Broad Entry Wording

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing supported-entry statements, fallback ambiguity, or
  over-broad workflow wording.
- Negative cases show that Story 2 fails if the canonical workflow or exact
  regime becomes unclear.
- Regression coverage remains small and documentation-focused.

**Execution checklist**
- [ ] Add focused checks in `tests/density_matrix/test_phase2_docs.py` or a
      tightly related successor for Story 2 wording coverage.
- [ ] Add negative checks for omitted backend default, omitted explicit
      `density_matrix` selection, or missing no-fallback wording.
- [ ] Add at least one check for missing canonical workflow identity or missing
      4 / 6 / 8 / 10 exact-regime wording.
- [ ] Keep benchmark execution and code-path validation outside this fast
      documentation check layer.

**Evidence produced**
- Focused regression coverage for Story 2 entry and workflow wording.
- Reviewable failures when required supported-path statements are missing.

**Risks / rollback**
- Risk: wording regressions may not break code, so they can survive unnoticed
  until external review.
- Rollback/mitigation: lock the critical statements down with targeted checks.

### Engineering Task 6: Emit One Stable Story 2 Supported-Entry And Workflow Reference Summary Or Checker

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 can emit one stable machine-readable summary or one stable rerunnable
  checker for supported-entry and canonical-workflow wording.
- The output records the mandatory Story 2 statements and their source links.
- The output is stable enough for Story 6 and paper-facing review to consume.

**Execution checklist**
- [ ] Add one Story 2 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for supported-entry
      summary emission.
- [ ] Emit one stable artifact in a Task 7 artifact directory
      (for example `benchmarks/density_matrix/artifacts/phase2_task7/`).
- [ ] Record source references, generation command, and scope notes in the
      output.
- [ ] Keep the output narrow to supported-entry and workflow clarity rather than
      broad support-matrix classification.

**Evidence produced**
- One stable Task 7 Story 2 reference summary or rerunnable checker.
- One reusable Story 2 output schema for later Task 7 handoffs.

**Risks / rollback**
- Risk: ad hoc prose updates are hard to compare and easy to overread.
- Rollback/mitigation: define one thin structured Story 2 output that makes the
  canonical supported entry and workflow surface explicit.

### Engineering Task 7: Document Story 2 Entry Points, Scope Boundaries, And Handoff To Story 3

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 2 validates, how to rerun it, and
  why it is the canonical entry-and-workflow clarity gate.
- The notes make clear that Story 2 closes supported-entry semantics but not the
  broader support-surface classification owned by Story 3.
- The documentation stays aligned with the frozen Phase 2 support boundary.

**Execution checklist**
- [ ] Document the Story 2 summary or checker and how it relates to Task 1,
      Task 2, and Task 6 surfaces.
- [ ] Make the main Story 2 rule explicit:
      required supported-entry and canonical-workflow statements must be present
      and unambiguous.
- [ ] Explain how Story 2 hands off broader support-surface classification to
      Story 3.
- [ ] Keep unsupported or deferred capability clearly outside Story 2 closure.

**Evidence produced**
- Updated developer-facing guidance for the Task 7 Story 2 clarity gate.
- One stable location where Story 2 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 2 is poorly documented, later support-surface notes may look
  like they are fixing basic entry semantics instead of building on them.
- Rollback/mitigation: document Story 2 as the explicit supported-entry gate and
  keep the handoff boundary visible.

### Engineering Task 8: Run Story 2 Validation And Confirm Unambiguous Supported-Path Wording

**Implements story**
- `Story 2: The Supported density_matrix Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 2 wording checks pass.
- The Story 2 summary or checker runs successfully and emits stable output.
- Story 2 completion is backed by rerunnable documentation evidence rather than
  by polished prose alone.

**Execution checklist**
- [ ] Run focused Story 2 regression checks for supported-entry and workflow
      coverage.
- [ ] Run the Story 2 summary or checker command and verify emitted output.
- [ ] Confirm that backend default, explicit selection, no-fallback wording,
      canonical workflow identity, and exact-regime size statements are present.
- [ ] Record stable test and artifact references for Story 3 and later Task 7
      work.

**Evidence produced**
- Passing focused checks for Story 2 wording completeness.
- One stable Story 2 output proving the supported entry surface and canonical
  workflow are documented unambiguously.

**Risks / rollback**
- Risk: Story 2 can appear complete while still allowing ambiguous or inflated
  workflow wording to survive.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 2.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one stable wording inventory defines the mandatory supported-entry and
  canonical-workflow statements,
- backend default, explicit `density_matrix` selection, and no-fallback behavior
  are documented unambiguously,
- the canonical XXZ plus `HEA` plus explicit-local-noise plus exact-observable
  workflow is documented unambiguously,
- the 4 / 6 / 8 / 10 exact-regime contract and 10-qubit anchor status are
  documented honestly,
- missing required statements fail Story 2 completeness checks,
- one stable Story 2 output or rerunnable checker captures the supported entry
  and workflow reference surface,
- and broader support-surface classification, evidence-bar interpretation,
  future-work separation, and terminology closure remain clearly assigned to
  Stories 3 to 6.

## Implementation Notes

- Prefer to reuse Task 1, Task 2, and Task 6 language rather than invent a new
  prose layer. Story 2 should reduce ambiguity, not add another vocabulary.
- Reuse `PHASE_2_DOCUMENTATION_INDEX.md` as the stable reader entry point and
  extend it with the supported-entry and canonical-workflow wording needed by
  Story 2 rather than creating a second competing index.
- Keep the exact-regime wording bounded and honest. Phase 2 is not the place to
  imply a broad scaling result from the 10-qubit anchor.
- Treat missing no-fallback wording as a real contract failure. Ambiguity here
  weakens both technical review and publication claims.
- Keep Story 2 focused on the supported path users are expected to follow.
  Optional, deferred, or broader standalone capability belongs to Story 3.
