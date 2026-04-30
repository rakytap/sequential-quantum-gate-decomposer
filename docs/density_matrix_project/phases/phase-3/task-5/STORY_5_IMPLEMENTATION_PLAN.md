# Story 5 Implementation Plan

## Story Being Implemented

Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner
Claim Surface

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the Task 5 calibration work into one supported-claim selection
surface:

- one documented benchmark-grounded rule selects the supported Task 5 planner
  claim surface from the candidate surface,
- supported calibrated claims remain distinct from comparison baselines and from
  diagnosis-only or exploratory cases,
- the selected claim remains tied to the frozen workload matrix and the
  correctness-gated calibration evidence,
- and Story 5 closes the contract for "which calibrated planner claim surface is
  actually supported" without yet claiming final bundle packaging or explicit
  future-work boundary summaries.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- workload-matrix anchoring already owned by Story 2,
- density-aware signal differentiation already owned by Story 3,
- correctness-gated evidence admissibility already owned by Story 4,
- stable calibration-bundle packaging already owned by Story 6,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- Stories 1 through 4 already define the candidate surface, workload matrix,
  density-aware signal surface, and correctness-gated positive evidence Story 5
  must select among honestly.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-009`, and `P3-ADR-010`.
- Story 5 should treat the outputs of the Story 4 correctness gate as the only
  admissible positive evidence for selecting the supported Task 5 claim.
- In the current Task 5 implementation, that supported claim is selected from
  the existing noisy planner's span-budget candidate settings rather than from a
  larger family of already-implemented noisy planner variants.
- Story 5 should preserve explicit visibility for structural or
  state-vector-oriented comparison baselines rather than hiding them after the
  supported claim is chosen.
- The supported claim must remain bounded to the frozen workload inventory and
  must not silently absorb exploratory or diagnosis-only cases.
- Story 5 should prefer one explicit claim-selection rule over narrative
  interpretation scattered across notebooks or paper prose.

## Engineering Tasks

### Engineering Task 1: Freeze The Supported-Claim Selection Rule

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one explicit benchmark-grounded rule for selecting the
  supported Task 5 planner claim.
- The rule is concrete enough that later reviewers can reproduce the supported
  claim from the recorded evidence.
- The rule remains bounded to the frozen workload matrix and correctness-gated
  evidence surface.

**Execution checklist**
- [ ] Freeze one explicit supported-claim selection rule for Task 5.
- [ ] Define which evidence surfaces the rule is allowed to consult.
- [ ] Define how comparison baselines remain visible without being selected as
      the supported claim silently.
- [ ] Keep final bundle packaging and explicit future-work boundary summaries
      outside the Story 5 bar.

**Evidence produced**
- One stable Task 5 supported-claim selection rule.
- One explicit boundary between selected supported claims and non-selected
  reference claims.

**Risks / rollback**
- Risk: planner selection may devolve into cherry-picking or narrative
  preference.
- Rollback/mitigation: freeze the claim-selection rule before publishing broad
  comparisons.

### Engineering Task 2: Build A Shared Claim-Selection Record Surface

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- code | docs

**Definition of done**
- Story 5 exposes one shared claim-selection record shape.
- The record links candidate identity, workload coverage, counted evidence, and
  selection verdicts in one auditable surface.
- The record remains additive to the shared Task 5 provenance vocabulary rather
  than replacing it.

**Execution checklist**
- [ ] Define one claim-selection record in
      `benchmarks/density_matrix/planner_calibration/` or the smallest adjacent
      helper.
- [ ] Record candidate identity, counted evidence summaries, and final selected
      or non-selected status explicitly.
- [ ] Keep overlapping provenance fields aligned with Stories 1 through 4.
- [ ] Document how selection records differ from raw calibration records.

**Evidence produced**
- One reviewable Task 5 claim-selection record shape.
- One explicit mapping from counted calibration evidence to supported-claim
  status.

**Risks / rollback**
- Risk: later claim summaries may become detached from the evidence that
  justifies them.
- Rollback/mitigation: define one shared claim-selection record before emitting
  broad summaries.

### Engineering Task 3: Keep Supported Claims, Comparison Baselines, And Exploratory Cases Explicit

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- code | tests

**Definition of done**
- Story 5 keeps selected supported claims distinct from comparison baselines and
  from exploratory or diagnosis-only cases.
- The record surface is explicit enough to prevent relabeling or accidental
  overclaiming.
- The distinction is enforced in machine-reviewable outputs rather than only in
  prose.

**Execution checklist**
- [ ] Add explicit claim-status categories for supported claim, comparison
      baseline, and exploratory or diagnosis-only case.
- [ ] Prevent comparison baselines from being relabeled as the supported claim
      silently.
- [ ] Prevent exploratory cases from expanding the supported claim boundary
      silently.
- [ ] Add focused regression checks for claim-status honesty.

**Evidence produced**
- One explicit Task 5 claim-status taxonomy.
- Regression coverage for supported-claim labeling honesty.

**Risks / rollback**
- Risk: later paper or benchmark work may overstate the Task 5 claim by
  collapsing supported, reference, and exploratory cases into one bucket.
- Rollback/mitigation: make claim-status labeling explicit and test it directly.

### Engineering Task 4: Add A Representative Claim-Selection Matrix

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative supported, comparison, and non-claim cases in
  one small review surface.
- The matrix is broad enough to show that the supported-claim rule operates
  across the actual Task 5 candidate and workload surfaces.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one selected supported claim case.
- [ ] Include at least one structural-only or state-vector-oriented comparison
      baseline case.
- [ ] Include at least one exploratory or diagnosis-only case that remains
      outside the supported claim.
- [ ] Keep the matrix small but claim-complete.

**Evidence produced**
- One representative Story 5 claim-selection matrix.
- One review surface for supported versus non-supported claim outcomes.

**Risks / rollback**
- Risk: the claim-selection rule may look coherent in prose but remain under-
  specified on actual case combinations.
- Rollback/mitigation: freeze a small but claim-complete matrix.

### Engineering Task 5: Add Deterministic Regression Coverage For Supported-Claim Selection

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- tests

**Definition of done**
- Story 5 has focused regression checks for deterministic supported-claim
  selection and labeling.
- The checks prove the same evidence surface yields the same selected claim for a
  fixed repository state and story configuration.
- The regression slice remains narrower than final bundle packaging and
  future-work boundary prose.

**Execution checklist**
- [ ] Add focused Story 5 regression coverage in
      `tests/partitioning/test_planner_calibration.py`.
- [ ] Assert stable claim-selection verdicts for representative supported and
      non-supported cases.
- [ ] Assert stable claim-status labeling for comparison baselines and
      exploratory cases.
- [ ] Keep the checks at the claim-selection layer rather than full publication
      packaging.

**Evidence produced**
- Fast regression coverage for Story 5 supported-claim selection stability.
- One repeatable test surface for later Task 5 work to extend.

**Risks / rollback**
- Risk: claim-selection drift may remain hidden until publication packaging or
  peer review.
- Rollback/mitigation: add a dedicated supported-claim regression slice early.

### Engineering Task 6: Emit A Stable Story 5 Claim-Selection Bundle Or Rerunnable Checker

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable claim-selection bundle or
  rerunnable checker.
- The bundle records selected supported claims, visible comparison baselines, and
  non-claim exploratory cases through one stable schema.
- The output shape is stable enough for Stories 6 and 7 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 5 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `calibrated_claim_selection_validation.py` as the primary checker.
- [ ] Add a dedicated Story 5 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/planner_calibration/claim_selection/`).
- [ ] Emit claim-selection verdicts, claim-status labels, and supporting evidence
      summaries through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 5 claim-selection bundle or checker.
- One reusable citation surface for the supported Task 5 planner claim.

**Risks / rollback**
- Risk: prose-only claim selection will make the supported Task 5 claim hard to
  audit and easy to reinterpret later.
- Rollback/mitigation: emit one thin machine-reviewable claim-selection surface
  early.

### Engineering Task 7: Document The Supported-Claim Handoff To Story 6 And Story 7

**Implements story**
- `Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface`

**Change type**
- docs

**Definition of done**
- Story 5 notes explain exactly which planner claim is selected and why.
- The selected supported claim is documented as ready for final packaging by
  Story 6, not yet as the final publication boundary summary.
- Developer-facing notes point to the Story 5 validator and artifact location.

**Execution checklist**
- [ ] Document the supported-claim selection rule and its admissible evidence
      inputs.
- [ ] Explain how comparison baselines remain visible after claim selection.
- [ ] Explain that final calibration-bundle packaging belongs to Story 6.
- [ ] Explain that explicit approximation and deferred-boundary summaries belong
      to Story 7.

**Evidence produced**
- Updated developer-facing notes for the Story 5 supported-claim gate.
- One stable handoff reference for later Task 5 implementation work.

**Risks / rollback**
- Risk: later Task 5 work may over-assume Story 5 already completed the final
  publication-facing boundary summary.
- Rollback/mitigation: document the handoff boundaries explicitly.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one explicit benchmark-grounded rule exists for selecting the supported Task 5
  planner claim,
- supported claims, comparison baselines, and exploratory or diagnosis-only
  cases remain explicitly distinguishable,
- representative supported and non-supported claim outcomes are recorded through
  one shared claim-selection surface,
- one stable Story 5 claim-selection bundle or rerunnable checker exists for
  later reuse,
- and stable final packaging plus explicit approximation and deferred-boundary
  handling remain clearly assigned to later stories.

## Implementation Notes

- Prefer one explicit claim-selection rule over a soft narrative consensus.
- Keep Story 5 focused on "which claim is supported," not yet on "how that claim
  is finally packaged for publication."
- Treat visible non-selected baselines as evidence of honesty, not as clutter.
