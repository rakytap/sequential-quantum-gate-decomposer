# Story 7 Implementation Plan

## Story Being Implemented

Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay
Explicit

This is a Layer 4 engineering plan for implementing the seventh behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns Task 5 calibration into one explicit claim-boundary surface:

- approximation areas remain visible instead of being collapsed into one generic
  "density-aware" label,
- diagnosis-only or exploratory cases remain explicit non-claim evidence rather
  than silently enlarging the supported claim,
- deferred channel-native or broader Phase 4 branches remain documented as
  future work rather than hidden prerequisites,
- and Story 7 closes Task 5 boundary interpretation without reopening candidate
  selection, workload anchoring, or stable bundle packaging.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- workload-matrix anchoring already owned by Story 2,
- density-aware signal differentiation already owned by Story 3,
- correctness-gated evidence admissibility already owned by Story 4,
- supported-claim selection already owned by Story 5,
- and stable calibration-bundle packaging already owned by Story 6.

## Dependencies And Assumptions

- Stories 1 through 6 already define the candidate surface, workload matrix,
  density-aware signal surface, correctness gate, supported-claim selection, and
  stable bundle surface Story 7 must summarize honestly.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-010`, and the performance-claim-boundary item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- The deferred follow-on branches explicitly visible around Task 5 remain:
  - fully channel-native fused noisy blocks,
  - broader noisy VQE or VQA workflow growth,
  - calibration-aware and readout-oriented workflow features,
  - and broader approximate-scaling branches beyond the core Phase 3 result.
- Story 7 should treat visible approximation or non-claim cases as required
  scientific evidence, not as an optional appendix.
- Story 7 should prefer one explicit boundary vocabulary over repeated soft
  wording scattered across bundle summaries and paper notes.
- The existing future-work and claim-boundary validation areas under
  `benchmarks/density_matrix/documentation_contract/` and
  `benchmarks/density_matrix/publication_claim_package/` provide useful prior
  art for explicit boundary handling, but Story 7 must keep the Task 5 claim
  boundary tied to the specific calibration evidence surface established here.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 5 Approximation And Non-Claim Taxonomy

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines one stable taxonomy for approximation areas, deferred
  branches, and non-claim exploratory cases.
- The taxonomy is explicit enough that later reviewers can see exactly where the
  supported Task 5 claim stops.
- The taxonomy stays aligned with the frozen Phase 3 future-work boundary.

**Execution checklist**
- [ ] Freeze the minimum Story 7 taxonomy for approximation areas,
      diagnosis-only cases, exploratory non-claim cases, and deferred follow-on
      branches.
- [ ] Define which categories belong in the supported Task 5 claim boundary and
      which do not.
- [ ] Keep candidate selection, workload anchoring, and bundle packaging outside
      the Story 7 bar.
- [ ] Preserve alignment with the deferred channel-native and Phase 4 branches.

**Evidence produced**
- One stable Task 5 approximation and non-claim taxonomy.
- One explicit boundary between supported claims and visible future work.

**Risks / rollback**
- Risk: the supported Task 5 claim may be overstated if approximation and future
  work remain informal.
- Rollback/mitigation: freeze the boundary taxonomy before broadening summary
  prose.

### Engineering Task 2: Build An Explicit Boundary Record Surface

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- code | docs

**Definition of done**
- Story 7 exposes one boundary record shape for approximation and non-claim
  summaries.
- The surface links supported claims, approximation notes, and deferred branches
  to the same shared Task 5 provenance and claim-selection outputs.
- The record remains additive to the Story 6 bundle rather than a disconnected
  narrative-only appendix.

**Execution checklist**
- [ ] Define one boundary-summary record in
      `benchmarks/density_matrix/planner_calibration/` or the smallest adjacent
      helper.
- [ ] Record approximation categories, non-claim status, and deferred-branch
      references beside the selected supported claim.
- [ ] Keep overlapping fields aligned with the Story 6 bundle vocabulary.
- [ ] Document how the boundary record differs from the raw calibration record.

**Evidence produced**
- One reviewable Task 5 boundary record shape.
- One explicit mapping from supported claims to explicit non-claim categories.

**Risks / rollback**
- Risk: approximation and future-work language may remain detached from the
  underlying calibration records.
- Rollback/mitigation: attach boundary categories to one shared record surface.

### Engineering Task 3: Map Diagnosis-Only And Exploratory Cases Into Explicit Non-Claim Buckets

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- code | tests

**Definition of done**
- Story 7 keeps diagnosis-only and exploratory cases visible without letting
  them silently enlarge the supported Task 5 claim.
- Non-claim cases remain machine-reviewable and comparable.
- The mapping is explicit enough to prevent overclaiming in later summaries.

**Execution checklist**
- [ ] Add explicit boundary categories for diagnosis-only cases and exploratory
      non-claim cases.
- [ ] Prevent exploratory cases from being promoted into the supported claim
      without explicit selection through Story 5.
- [ ] Prevent diagnosis-only cases from being misread as positive supported
      evidence.
- [ ] Add focused regression checks for non-claim bucket honesty.

**Evidence produced**
- One explicit Task 5 non-claim classification rule.
- Regression coverage for diagnosis-only and exploratory-case labeling honesty.

**Risks / rollback**
- Risk: later summaries may use interesting exploratory cases to imply stronger
  supported claims than the evidence allows.
- Rollback/mitigation: make non-claim labeling explicit and test it directly.

### Engineering Task 4: Add A Representative Claim-Boundary Matrix

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- tests | validation automation

**Definition of done**
- Story 7 covers representative examples of supported claims, approximation
  areas, non-claim cases, and deferred branches.
- The matrix is broad enough to show that the boundary taxonomy works on real
  Task 5 outputs.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one supported claim case that remains fully inside the
      frozen Task 5 boundary.
- [ ] Include at least one approximation-area case that is still part of the
      supported Task 5 story but must be described narrowly.
- [ ] Include at least one diagnosis-only or exploratory non-claim case.
- [ ] Include at least one explicit deferred follow-on branch reference, such as
      channel-native fusion or broader Phase 4 workflow growth.

**Evidence produced**
- One representative Story 7 claim-boundary matrix.
- One review surface for supported claims versus visible non-claim categories.

**Risks / rollback**
- Risk: the boundary taxonomy may appear coherent in prose while remaining
  underspecified on representative case types.
- Rollback/mitigation: freeze a small but boundary-complete matrix.

### Engineering Task 5: Align Task 5 Boundary Language With Publication-Facing Claim Surfaces

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 aligns the Task 5 boundary summary with the broader Phase 3
  publication-claim surface.
- The Task 5 boundary language is explicit enough that later abstract, short
  paper, and full paper work can reuse it safely.
- Story 7 avoids reintroducing ambiguous optimality or generality language.

**Execution checklist**
- [ ] Document how the Task 5 supported claim boundary should be phrased for
      publication-facing work.
- [ ] Keep structural or state-vector baselines visible as reference rather than
      as hidden replacements.
- [ ] Keep deferred branches clearly named as future work rather than as
      unfinished prerequisites.
- [ ] Cross-check the boundary language against the broader Phase 3 claim
      boundary in planning and ADR documents.

**Evidence produced**
- One reviewable Task 5 publication-boundary phrasing surface.
- One explicit mapping from Task 5 boundary categories to later paper use.

**Risks / rollback**
- Risk: later paper work may accidentally overstate Task 5 through vague
  language even if the bundle data is honest.
- Rollback/mitigation: align the boundary wording explicitly before publication
  packaging.

### Engineering Task 6: Emit A Stable Story 7 Boundary Bundle Or Rerunnable Checker

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable boundary bundle or rerunnable
  checker.
- The bundle records supported claims, approximation areas, diagnosis-only or
  exploratory cases, and deferred branches through one stable schema.
- The output is stable enough for later review and publication work to cite
  directly.

**Execution checklist**
- [ ] Add a dedicated Story 7 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `calibration_boundary_validation.py` as the primary checker.
- [ ] Add a dedicated Story 7 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/planner_calibration/claim_boundary/`).
- [ ] Emit boundary-category summaries through one stable schema aligned with the
      Story 6 bundle.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 7 boundary bundle or checker.
- One direct citation surface for the explicit Task 5 claim boundary.

**Risks / rollback**
- Risk: if Task 5 boundary evidence remains only in prose, later reviewers may
  struggle to separate supported claims from future work.
- Rollback/mitigation: emit one stable machine-reviewable boundary surface.

### Engineering Task 7: Document The Future-Work Gate And Run The Story 7 Boundary Surface

**Implements story**
- `Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 5 supported-claim boundary and the
  visible future-work gate.
- The Story 7 boundary harness and bundle run successfully.
- Deferred channel-native and broader Phase 4 branches remain documented as
  explicit future work rather than hidden prerequisites.

**Execution checklist**
- [ ] Document the Task 5 approximation taxonomy and future-work gate.
- [ ] Explain how supported claims, non-claim cases, and deferred branches should
      be interpreted in Phase 3.
- [ ] Run focused Story 7 regression coverage and verify
      `benchmarks/density_matrix/planner_calibration/calibration_boundary_validation.py`.
- [ ] Record stable references to the Story 7 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 7 claim-boundary regression checks.
- One stable Story 7 boundary-bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake deferred follow-on branches for hidden
  implementation debt instead of explicit scientific scope boundaries.
- Rollback/mitigation: document the future-work gate and deferred boundary
  explicitly.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- one explicit taxonomy exists for supported Task 5 claims, approximation areas,
  diagnosis-only or exploratory non-claim cases, and deferred follow-on
  branches,
- representative Task 5 outputs can be categorized through one shared
  machine-reviewable boundary surface,
- publication-facing boundary language is aligned with the frozen Phase 3 claim
  boundary and does not overstate optimality or generality,
- one stable Story 7 boundary bundle or rerunnable checker exists for direct
  citation,
- and deferred channel-native fusion and broader Phase 4 workflow growth remain
  clearly documented as future work rather than hidden prerequisites.

## Implementation Notes

- Prefer explicit boundary categories over vague cautionary prose.
- Keep Story 7 focused on honest claim-boundary handling, not on reopening the
  supported-claim selection or stable bundle structure already closed by earlier
  stories.
- Treat visible future work as a strength of the contract, not as an admission
  that Task 5 is underspecified.
