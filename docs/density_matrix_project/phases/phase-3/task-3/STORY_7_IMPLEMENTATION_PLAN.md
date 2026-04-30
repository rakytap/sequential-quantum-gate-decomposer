# Story 7 Implementation Plan

## Story Being Implemented

Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent
Fallback

This is a Layer 4 engineering plan for implementing the seventh behavioral
slice from `TASK_3_STORIES.md`.

## Scope

This story turns the Task 3 negative runtime boundary into an enforceable
runtime-stage gate:

- runtime-stage unsupported or incomplete execution fails explicitly after
  successful descriptor generation rather than silently falling back,
- failure reporting records stable unsupported category, first unsupported
  condition, and failure-stage metadata in a machine-reviewable way,
- validation and benchmark surfaces cannot quietly relabel failed runtime cases
  as supported Task 3 behavior,
- and the unsupported-runtime evidence is explicit enough to support honest
  Paper 2 scope language.

Out of scope for this story:

- positive workload coverage already owned by Stories 1 and 2,
- direct descriptor-to-runtime handoff owned by Story 3,
- positive runtime semantic-preservation closure owned by Story 4,
- positive result-shape closure owned by Story 5,
- positive cross-workload audit stability owned by Story 6,
- and full correctness-threshold analysis, fused execution behavior, or
  performance diagnosis owned by later Phase 3 tasks.

## Dependencies And Assumptions

- Stories 1 through 6 already define the supported positive runtime slices that
  Story 7 must protect.
- The frozen source-of-truth contract is `TASK_3_MINI_SPEC.md`,
  `TASK_3_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`, and
  `P3-ADR-008`.
- Task 1 already established a structured unsupported vocabulary at
  planner-entry time through `NoisyPlannerValidationError`, including category,
  first unsupported condition, and failure-stage concepts that Story 7 should
  align with where practical.
- Task 2 already established a structured unsupported vocabulary at
  descriptor-generation time through `NoisyDescriptorValidationError`, which
  should remain distinguishable from the Task 3 runtime boundary.
- The current codebase does not yet expose a dedicated Task 3 runtime error
  class, so the natural implementation home for Story 7 includes either:
  - extending the Task 3 runtime layer in
    `squander/partitioning/noisy_runtime.py` with a structured runtime
    validation error type.
- Stories 5 and 6 now define the positive Task 3 result and audit surfaces;
  Story 7 should keep overlapping provenance and schema identity fields aligned
  with those supported outputs.
- The likely shared implementation substrate therefore remains:
  - the Task 3 runtime layer in `squander/partitioning/noisy_runtime.py`,
  - the existing descriptor and audit helpers under
    `squander/partitioning/noisy_planner.py`,
  - and validation or artifact-emission helpers under
    `benchmarks/density_matrix/partitioned_runtime/`.
- Story 7 should introduce one explicit unsupported-runtime vocabulary rather
  than allowing every test, script, or wrapper to invent its own failure
  taxonomy.
- Story 7 should treat negative evidence as a required output, not as an
  optional appendix to positive runtime cases.

## Engineering Tasks

### Engineering Task 1: Freeze The Unsupported-Runtime Taxonomy For Task 3

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- docs | validation automation

**Definition of done**
- Story 7 defines one stable unsupported-boundary taxonomy for Task 3 runtime
  execution.
- The taxonomy covers the minimum runtime-stage failure categories needed by the
  contract.
- Runtime-stage failures are separated clearly from earlier planner-entry and
  descriptor-generation failures.

**Execution checklist**
- [ ] Freeze the minimum unsupported categories for Task 3 runtime execution:
      descriptor-to-runtime mismatch, missing or unusable remap metadata at
      execution time, incomplete parameter-routing realization, unsupported
      runtime operation lowering, result-emission failure on an otherwise
      claimed supported case, malformed runtime request, and disallowed fallback
      substitution.
- [ ] Freeze one first-unsupported-condition vocabulary for
      machine-reviewable runtime artifacts.
- [ ] Distinguish runtime-stage failure stages from planner-entry and
      descriptor-generation stages explicitly.
- [ ] Align the Task 3 taxonomy with the Task 1 and Task 2 unsupported
      vocabularies wherever the meanings overlap.

**Evidence produced**
- One stable Story 7 unsupported-runtime taxonomy.
- One reviewable first-unsupported-condition vocabulary for Task 3 runtime
  failures.

**Risks / rollback**
- Risk: if unsupported runtime categories remain ad hoc, later validation and
  paper packaging will struggle to compare negative evidence consistently.
- Rollback/mitigation: define one shared runtime taxonomy before wiring the
  failure surfaces.

### Engineering Task 2: Add A Runtime Preflight Or Execution-Validation Gate

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- code | tests

**Definition of done**
- Supported Task 3 runtime requests pass through one explicit runtime
  preflight or execution-validation surface.
- The validator can reject runtime-stage unsupported or incomplete conditions
  before a case is mislabeled as supported success.
- The validator does not rely on hidden fallback or silent sequential
  substitution to make an unsupported request appear successful.

**Execution checklist**
- [ ] Introduce one Task 3 runtime validation surface in the runtime layer, with
      `squander/partitioning/noisy_runtime.py` as the default implementation
      home.
- [ ] Add a structured runtime validation error type aligned with the existing
      machine-reviewable error pattern used by `NoisyPlannerValidationError`
      and `NoisyDescriptorValidationError`, or reuse the smallest compatible
      successor.
- [ ] Route supported continuity and methods-oriented runtime requests through
      that gate before claiming Task 3 success.
- [ ] Keep runtime-stage validation separate from earlier descriptor-generation
      checks and from later threshold-verdict packaging.

**Evidence produced**
- One explicit Task 3 runtime validation gate.
- Focused regression coverage for runtime-stage rejection behavior.

**Risks / rollback**
- Risk: unsupported cases may still reach downstream bundle or benchmark code
  paths where fallback or partial success becomes hard to reason about.
- Rollback/mitigation: make runtime-stage validation mandatory before any Task 3
  success is recorded.

### Engineering Task 3: Detect Representative Runtime-Stage Unsupported Or Incomplete Conditions Explicitly

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- code | tests

**Definition of done**
- Runtime execution can detect the main unsupported or incomplete execution
  failures defined by Task 3.
- The checks are explicit enough to expose why a case failed rather than only
  that it failed.
- The supported positive Task 3 contract remains protected against silent
  erosion at runtime.

**Execution checklist**
- [ ] Add explicit checks for descriptor-to-runtime mismatches that make
      execution unauditable or incomplete.
- [ ] Add explicit checks for missing or unusable remap or routing data that
      only becomes visible at runtime preparation or execution time.
- [ ] Add explicit checks for runtime-stage inability to realize supported gate
      or noise operations from the claimed supported descriptor path.
- [ ] Add explicit checks for incomplete result emission on otherwise claimed
      supported runtime cases.
- [ ] Add explicit checks for hidden sequential or non-partitioned substitution
      when the request claims `partitioned_density` runtime behavior.

**Evidence produced**
- One reviewable set of runtime-stage unsupported checks for Task 3.
- Focused regression coverage tying concrete negative cases to concrete runtime
  failure categories.

**Risks / rollback**
- Risk: generic runtime failure handling may hide which contract guarantee was
  actually violated.
- Rollback/mitigation: detect and label each representative runtime-stage
  failure class explicitly.

### Engineering Task 4: Align Structured Runtime Diagnostics With The Shared Positive Audit Surface

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- code | docs

**Definition of done**
- Story 7 negative evidence fits alongside the shared positive runtime audit
  surface from Story 6 where fields overlap.
- Unsupported runtime records stay machine-reviewable and comparable.
- Task 3 does not introduce a disconnected error-only reporting format.

**Execution checklist**
- [ ] Reuse the shared case-level provenance tuple on unsupported runtime
      records where it still applies.
- [ ] Keep overlapping fields aligned with the Story 6 audit surface for easy
      comparison between supported and unsupported cases.
- [ ] Add the structured unsupported fields without replacing the shared
      artifact vocabulary.
- [ ] Document how unsupported runtime records relate to supported runtime-audit
      records.

**Evidence produced**
- One aligned Task 3 negative-evidence record shape.
- One explicit mapping between supported and unsupported runtime artifacts.

**Risks / rollback**
- Risk: unsupported evidence may remain structurally incomparable to supported
  evidence even if both look reasonable on their own.
- Rollback/mitigation: align overlapping fields with the shared positive audit
  surface.

### Engineering Task 5: Block Mislabeling In Validation, Harnesses, And Artifact Labels

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- code | validation automation | docs

**Definition of done**
- No validation or benchmark surface can claim supported Task 3 runtime
  behavior after a runtime-stage failure.
- Artifact labels make unsupported status visible rather than silently
  substituting a different execution path.
- The no-fallback rule is enforced on the delivered runtime surface itself.

**Execution checklist**
- [ ] Review validation and artifact-emission entry points that may later claim
      Task 3 runtime behavior.
- [ ] Add explicit status or support labels that prevent failed runtime cases
      from being recorded as supported.
- [ ] Ensure any sequential-density oracle path remains labeled as reference
      behavior rather than fallback success.
- [ ] Reuse the Task 3 audit-bundle style established in Story 6 where
      practical instead of inventing local failure semantics.

**Evidence produced**
- Reviewable artifact-labeling rules for Task 3 unsupported runtime cases.
- Focused checks proving failed runtime cases are not mislabeled.

**Risks / rollback**
- Risk: even correct runtime validation logic can be undermined if harnesses
  relabel failed cases as supported output.
- Rollback/mitigation: enforce status labeling wherever Task 3 behavior is
  recorded.

### Engineering Task 6: Add A Representative Unsupported-Runtime Matrix

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- tests | validation automation

**Definition of done**
- Story 7 covers representative unsupported cases across the Task 3 runtime
  boundary.
- Negative cases produce stable category, first-condition, and failure-stage
  metadata.
- Unsupported cases are tested at runtime rather than only through earlier
  planner or descriptor failure.

**Execution checklist**
- [ ] Add representative negative cases for descriptor-to-runtime mismatch,
      missing remap data at execution time, incomplete parameter-routing
      realization, runtime-stage operation-lowering failure, and hidden fallback
      attempts.
- [ ] Reuse or extend Task 1 and Task 2 unsupported-case patterns where they
      provide a good model for structured diagnostics.
- [ ] Add a fast regression slice in `tests/partitioning/test_partitioned_runtime.py`
      that asserts failure occurs at runtime and not earlier.
- [ ] Keep the matrix representative and contract-driven rather than exhaustive
      over every impossible runtime combination.

**Evidence produced**
- Focused Story 7 regression coverage for representative unsupported runtime
  requests.
- A representative unsupported-runtime matrix tied to the Task 3 contract.

**Risks / rollback**
- Risk: without representative negative coverage, unsupported runtime behavior
  may be tested only opportunistically and fallback bugs can hide.
- Rollback/mitigation: freeze a small but contract-complete runtime matrix and
  run it routinely.

### Engineering Task 7: Emit And Document A Stable Story 7 Unsupported-Runtime Bundle

**Implements story**
- `Story 7: Runtime-Stage Unsupported Or Incomplete Execution Fails With No Silent Fallback`

**Change type**
- validation automation | docs

**Definition of done**
- Story 7 emits one stable machine-reviewable unsupported-runtime bundle or
  rerunnable checker.
- The bundle records unsupported category, first unsupported condition, failure
  stage, and no-mislabeling outcome for representative runtime cases.
- Developer-facing notes explain the Task 3 runtime negative boundary
  concretely.

**Execution checklist**
- [ ] Add a dedicated Story 7 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/unsupported_runtime/`).
- [ ] Emit representative unsupported runtime cases through one stable schema.
- [ ] Record rerun commands, software metadata, and the no-mislabeling
      interpretation with the emitted bundle.
- [ ] Document how the Story 7 negative bundle complements the positive Story 6
      runtime-audit bundle.

**Evidence produced**
- One stable Story 7 unsupported-runtime bundle or checker.
- One stable developer-facing reference for the Task 3 runtime negative
  boundary.

**Risks / rollback**
- Risk: unsupported runtime behavior may remain documented only in prose,
  leaving later reviewers unable to tell how the Task 3 boundary was actually
  enforced.
- Rollback/mitigation: emit one structured negative bundle and document it
  together with the rule.

## Exit Criteria

Story 7 is complete only when all of the following are true:

- unsupported or incomplete Task 3 runtime cases fail explicitly with one stable
  unsupported taxonomy,
- runtime-stage failures expose stable category, first unsupported condition,
  and failure-stage metadata,
- validation and artifact surfaces cannot silently relabel failed runtime cases
  as supported Task 3 behavior,
- one stable Story 7 unsupported-runtime bundle or checker exists for negative
  evidence,
- and final correctness-threshold analysis and performance diagnosis remain
  separated from Task 3 unsupported-boundary closure.

## Implementation Notes

- Prefer one shared unsupported-runtime vocabulary over multiple
  script-specific error taxonomies.
- Treat negative evidence as a required output of Task 3, not as an optional
  appendix to supported cases.
- Keep the Task 3 unsupported boundary enforceable at runtime itself rather
  than through later summary scripts.
