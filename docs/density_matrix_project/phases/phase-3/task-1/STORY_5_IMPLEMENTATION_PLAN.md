# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Unsupported Planner Requests Fail Before Execution With No Silent
Fallback

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns the Phase 3 no-fallback rule into an enforceable planner-entry
gate:

- requests outside the frozen canonical planner contract fail before execution,
- failure reporting records the first unsupported condition in a stable
  machine-reviewable way,
- benchmark and validation layers cannot silently substitute sequential density,
  state-vector, or hidden non-partitioned execution while still claiming Phase 3
  behavior,
- and the unsupported-boundary evidence is explicit enough to support honest
  Paper 2 scope language.

Out of scope for this story:

- positive continuity-entry closure owned by Story 1,
- positive mandatory-workload coverage owned by Story 2,
- richer audit schema owned by Story 3,
- optional positive exact legacy-source lowering owned by Story 4,
- and partition runtime correctness or performance diagnosis owned by later
  Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 4 already define the positive supported slices that Story 5
  protects.
- The shared Task 1 implementation substrate now includes
  `squander/partitioning/noisy_planner.py`, which already contains the first
  planner-entry validation hooks and should become the canonical home for
  Task 1 unsupported-boundary checks.
- Stories 1 and 2 have already emitted passing planner-surface bundles, so
  Story 5 should align unsupported-case records with that existing payload
  structure wherever practical.
- Story 3 confirmed that planner-side provenance and audit fields are already
  sufficient for Task 1 evidence, so Story 5 should attach unsupported metadata
  to that same planner-side schema rather than adding a separate error-only
  reporting layer.
- Story 4 now provides concrete negative legacy-source cases and planner-side
  exact-lowering failures, so Story 5 should extend that real unsupported
  vocabulary rather than re-derive source/gate/noise failure categories from
  scratch.
- The no-fallback rule is frozen at the phase level in `P3-ADR-003`,
  `P3-ADR-007`, and the Phase 3 checklist closure.
- Existing negative-evidence patterns already exist in:
  - `benchmarks/density_matrix/bridge_scope/unsupported_bridge_validation.py`,
  - `benchmarks/density_matrix/noise_support/unsupported_noise_validation.py`,
  - and the new planner-surface validation directory under
    `benchmarks/density_matrix/planner_surface/`.
- The current codebase does not yet expose a full Phase 3 `partitioned_density`
  runtime mode. Story 5 may therefore need a planner-entry preflight or harness
  gate before the runtime itself exists.
- Story 5 should introduce one explicit unsupported-boundary vocabulary for
  planner entry rather than allowing every script or wrapper to invent its own
  fallback language.

## Engineering Tasks

### Engineering Task 1: Freeze The Unsupported-Boundary Taxonomy For Planner Entry

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one stable unsupported-boundary taxonomy for Task 1.
- The taxonomy covers the minimum planner-entry failure categories needed by the
  frozen contract.
- Failure categories are separated cleanly from later runtime or numerical
  correctness failures.

**Execution checklist**
- [ ] Freeze the minimum unsupported categories for Task 1:
      source type, gate family, noise model, noise schedule metadata, malformed
      planner request, and disallowed mode claims.
- [ ] Freeze one first-unsupported-condition vocabulary for machine-reviewable
      artifacts.
- [ ] Distinguish planner-entry failure stages from later runtime or benchmark
      failures.
- [ ] Reuse the `NoisyPlannerValidationError` vocabulary from
      `squander/partitioning/noisy_planner.py` where it already aligns with the
      frozen Phase 3 contract.

**Evidence produced**
- One stable Story 5 unsupported-boundary taxonomy.
- One reviewable first-unsupported-condition vocabulary.

**Risks / rollback**
- Risk: if unsupported categories remain ad hoc, later validation bundles will
  not be comparable and silent fallback risks will be harder to detect.
- Rollback/mitigation: define one shared unsupported taxonomy before wiring the
  failure surfaces.

### Engineering Task 2: Add A Planner-Entry Preflight Validator For Phase 3 Requests

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- code | tests

**Definition of done**
- Phase 3 planner-entry requests pass through one explicit preflight validator.
- The validator can reject unsupported source, gate, noise, or mode conditions
  before execution begins.
- The validator does not rely on hidden fallback to make an unsupported request
  succeed.

**Execution checklist**
- [ ] Introduce one Phase 3 planner-entry validation surface in the planner or
      harness layer, with `squander/partitioning/noisy_planner.py` as the
      default implementation home.
- [ ] Route supported continuity, methods, and optional legacy-source requests
      through that validator before planner execution begins.
- [ ] Reject unsupported planner-entry conditions with stable category and
      first-condition metadata.
- [ ] Keep preflight validation separate from later partition-descriptor or
      runtime execution checks.

**Evidence produced**
- One explicit planner-entry preflight validator for Task 1.
- Focused regression coverage for preflight rejection behavior.

**Risks / rollback**
- Risk: unsupported cases may still reach downstream code paths where fallback
  or partial handling becomes hard to reason about.
- Rollback/mitigation: make planner-entry preflight mandatory before any Phase 3
  planner behavior is claimed.

### Engineering Task 3: Block Silent Fallback In Harnesses, Validation Surfaces, And Artifact Labels

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- code | validation automation | docs

**Definition of done**
- No benchmark or validation surface can claim Phase 3 planner behavior after a
  hidden fallback.
- Artifact labels make unsupported status visible rather than silently
  substituting another execution mode.
- The no-fallback rule is enforced even before the full partitioned runtime is
  delivered.

**Execution checklist**
- [ ] Review benchmark and validation entry points that may later claim Phase 3
      `partitioned_density` behavior.
- [ ] Add explicit status or mode labels that prevent unsupported cases from
      being recorded as supported partitioned cases.
- [ ] Ensure any temporary sequential-density oracle path is labeled as
      reference behavior, not as fallback success.
- [ ] Reuse the planner-surface bundle style established in Story 1 where
      practical instead of inventing benchmark-local fallback semantics.

**Evidence produced**
- Reviewable artifact-labeling rules for no-fallback behavior.
- Focused checks proving unsupported planner requests are not mislabeled as
  supported cases.

**Risks / rollback**
- Risk: even correct preflight logic can be undermined if harnesses or bundles
  relabel fallback behavior as supported planner execution.
- Rollback/mitigation: enforce no-fallback labeling rules wherever planner
  behavior is recorded.

### Engineering Task 4: Add A Representative Unsupported-Request Matrix For Task 1

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative unsupported requests across the Task 1 boundary.
- Negative cases produce stable category and first-condition metadata.
- Unsupported cases are tested at planner entry rather than only through later
  runtime failure.

**Execution checklist**
- [ ] Add representative negative cases for unsupported source types, unsupported
      gate families, unsupported noise models, invalid schedule metadata, and
      disallowed mode claims.
- [ ] Reuse or extend the patterns from unsupported bridge and unsupported noise
      validation bundles.
- [ ] Add a fast regression slice that asserts unsupported planner requests fail
      before execution.
- [ ] Keep the matrix representative and contract-driven rather than exhaustive
      over every impossible input combination.

**Evidence produced**
- Focused Story 5 regression coverage for representative unsupported requests.
- A representative unsupported-request matrix tied to the Task 1 contract.

**Risks / rollback**
- Risk: without representative negative coverage, unsupported behavior may be
  tested only opportunistically and fallback bugs can hide.
- Rollback/mitigation: freeze a small but contract-complete unsupported matrix
  and run it routinely.

### Engineering Task 5: Emit A Stable Story 5 Unsupported-Planner Bundle

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable unsupported-planner bundle.
- The bundle records unsupported category, first unsupported condition, failure
  stage, and no-fallback outcome for representative cases.
- The bundle is reusable for later benchmark interpretation and paper writing.

**Execution checklist**
- [ ] Add a dedicated Story 5 artifact location
      (for example `benchmarks/density_matrix/artifacts/planner_surface/`).
- [ ] Emit representative unsupported cases through one stable schema.
- [ ] Keep overlapping fields aligned with the existing Story 1 and Story 2
      planner-surface bundles so supported and unsupported cases remain
      comparable.
- [ ] Record rerun commands, software metadata, and no-fallback interpretation
      with the emitted bundle.
- [ ] Keep the bundle focused on planner-entry boundary behavior rather than on
      later runtime performance or exactness.

**Evidence produced**
- One stable Story 5 unsupported-planner bundle.
- One reusable schema for later publication and review packaging.

**Risks / rollback**
- Risk: unsupported behavior may remain documented only in prose, leaving later
  reviewers unable to tell how the no-fallback rule was actually enforced.
- Rollback/mitigation: emit one structured unsupported bundle and treat it as
  the canonical negative-evidence surface.

### Engineering Task 6: Document The No-Fallback Rule And Run The Story 5 Gate

**Implements story**
- `Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 1 unsupported boundary and the
  no-fallback rule.
- Fast regression checks and the Story 5 unsupported bundle run successfully.
- Story 5 closes with explicit negative evidence rather than only with positive
  supported-path claims.

**Execution checklist**
- [ ] Document the unsupported categories and first-condition vocabulary for Task
      1.
- [ ] Explain how the no-fallback rule applies before the full partition runtime
      exists.
- [ ] Run focused Story 5 regression coverage and verify the emitted unsupported
      bundle.
- [ ] Record stable test and artifact references for later Phase 3 tasks and
      paper packaging.

**Evidence produced**
- Passing Story 5 regression checks.
- One stable Story 5 unsupported-planner bundle or checker reference.

**Risks / rollback**
- Risk: if the no-fallback rule is not documented concretely, later benchmark
  or paper wording may drift back toward best-effort interpretation.
- Rollback/mitigation: document the rule and its emitted negative evidence
  together.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- unsupported Task 1 requests fail before execution with one stable unsupported
  category and first-unsupported-condition vocabulary,
- one explicit planner-entry preflight validator or equivalent gate enforces the
  unsupported boundary,
- benchmark and validation surfaces cannot silently relabel fallback behavior as
  supported Phase 3 planner behavior,
- one stable Story 5 unsupported bundle or checker exists for negative evidence,
- and later runtime, semantic-preservation, and performance work remain
  separated from planner-entry unsupported-boundary closure.

## Implementation Notes

- Prefer a single planner-entry unsupported vocabulary over multiple
  script-specific error taxonomies.
- Treat the sequential density path as an oracle and reference baseline, not as
  a silent rescue path for unsupported planner requests.
- Negative evidence is a required output of Task 1, not an optional appendix.
