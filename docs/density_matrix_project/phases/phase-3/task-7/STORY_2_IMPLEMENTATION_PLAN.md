# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving
Supported Cases

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one explicit counted-supported benchmark gate:

- positive benchmark evidence is gated on Task 6 counted-status closure rather
  than on raw timing or ad hoc favorable runs,
- counted benchmark cases remain joinable to supported `partitioned_density`
  execution, stable provenance, and planner-setting references,
- excluded benchmark cases stay explicit when correctness, provenance, or
  runtime-path requirements are missing,
- and Story 2 closes the contract for "which Task 7 cases may count as positive
  benchmark evidence" without yet claiming threshold satisfaction, sensitivity
  interpretation, diagnosis closure, or final package assembly.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- explicit positive-threshold interpretation already owned by Story 3,
- sensitivity analysis across benchmark knobs already owned by Story 4,
- the shared comparable metric surface already owned by Story 5,
- diagnosis-path bottleneck reporting already owned by Story 6,
- shared benchmark-package assembly already owned by Story 7,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- Story 1 already defines the benchmark matrix and representative review set
  Story 2 must gate consistently rather than redefine.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-004`,
  `P3-ADR-005`, and `P3-ADR-008`.
- Task 6 already emits the shared correctness package Story 2 should interpret
  directly through:
  - `benchmarks/density_matrix/correctness_evidence/correctness_bundle_validation.py`,
  - `benchmarks/density_matrix/correctness_evidence/bundle.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task6/`.
- Task 4 and Task 3 already provide stable runtime-path, fused-coverage, and
  execution provenance surfaces Story 2 should preserve rather than rename.
- Task 5 already emits the bounded supported planner-setting surface and claim-
  selection references Story 2 should carry forward into counted benchmark
  evidence.
- Story 2 should not recompute correctness from raw execution data. It should
  interpret counted versus excluded benchmark eligibility against the Task 6
  package and the shared runtime-path vocabulary.
- The natural implementation home for Task 7 counted-evidence gating is the new
  `benchmarks/density_matrix/performance_evidence/` package, with
  `counted_supported_validation.py` reading the Story 1 matrix and the Task 6
  correctness package directly.
- Story 2 should treat missing provenance, silent fallback, or unlabeled
  runtime-path identity as explicit exclusion reasons rather than as soft
  warnings.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Counted-Supported Benchmark Rule

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one explicit rule for when a Task 7 benchmark case may count
  as positive evidence.
- The rule ties counted benchmark evidence directly to Task 6 correctness
  closure, supported runtime-path identity, and stable provenance.
- The rule is explicit enough that later threshold, diagnosis, and summary
  stories can rely on it safely.

**Execution checklist**
- [ ] Freeze the rule that only Task 6 counted supported cases may count as
      positive Task 7 benchmark evidence.
- [ ] Freeze the rule that missing provenance, silent fallback, or unlabeled
      runtime-path identity force explicit exclusion.
- [ ] Define which Task 6 references and shared runtime fields every counted
      benchmark record must preserve.
- [ ] Keep threshold, sensitivity, diagnosis, and summary interpretation
      outside the Story 2 bar.

**Evidence produced**
- One stable Task 7 counted-supported benchmark rule.
- One explicit exclusion rule for missing correctness, provenance, or supported
  runtime identity.

**Risks / rollback**
- Risk: later benchmark consumers may inflate claims by counting cases whose
  correctness or provenance status is incomplete if Story 2 leaves the rule
  implicit.
- Rollback/mitigation: freeze one counted-supported rule before broad benchmark
  packaging.

### Engineering Task 2: Reuse The Shared Task 6 Correctness Package And Runtime-Path Surfaces As The Base

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- docs | code

**Definition of done**
- Story 2 reuses the Task 6 correctness package and shared runtime-path fields
  directly where they already match Task 7 needs.
- Story 2 keeps counted benchmark evidence aligned with supported Phase 3
  runtime and correctness vocabulary.
- Story 2 avoids creating a detached benchmark-counting language.

**Execution checklist**
- [ ] Reuse the Story 7 correctness-package fields from Task 6 directly where
      they already express counted versus excluded status.
- [ ] Reuse shared runtime-path and fusion-classification labels from Task 3 and
      Task 4 where they overlap with Task 7 benchmark records.
- [ ] Reuse Task 5 claim-selection references as the planner-setting join key
      for counted benchmark cases.
- [ ] Document any additive Task 7 gating fields explicitly rather than renaming
      shared provenance or correctness vocabulary.

**Evidence produced**
- One reviewable mapping from Task 6 correctness fields and shared runtime-path
  fields to Task 7 counted benchmark eligibility.
- One explicit boundary between reused Phase 3 fields and Story 2-specific
  gating fields.

**Risks / rollback**
- Risk: Task 7 may count a plausible-looking benchmark slice that later turns
  out to be inconsistent with Task 6 correctness rules if Story 2 drifts from
  the shared package.
- Rollback/mitigation: anchor Story 2 directly on the Task 6 package and shared
  runtime vocabulary.

### Engineering Task 3: Build The Task 7 Counted-Supported Benchmark Gate Harness

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 has one reusable harness for validating counted versus excluded
  benchmark eligibility.
- The harness records counted-status decisions and exclusion reasons in a
  machine-reviewable way.
- The harness is reusable by later threshold, sensitivity, and summary
  consumers.

**Execution checklist**
- [ ] Add a dedicated Story 2 validation driver under
      `benchmarks/density_matrix/performance_evidence/`, with
      `counted_supported_validation.py` as the primary checker.
- [ ] Read the Story 1 benchmark matrix and the Task 6 correctness package
      directly rather than reconstructing counted status from raw logs.
- [ ] Record counted versus excluded benchmark status together with the required
      Task 6 reference fields and runtime-path identity.
- [ ] Keep the harness rooted in machine-reviewable counted-status logic rather
      than prose-only interpretation.

**Evidence produced**
- One reusable Story 2 counted-supported benchmark harness.
- One machine-reviewable counted-status schema for later Task 7 consumers.

**Risks / rollback**
- Risk: counted-benchmark eligibility may remain scattered across notebooks or
  benchmark scripts and drift from the shared correctness package.
- Rollback/mitigation: centralize Story 2 in one stable validation entry point.

### Engineering Task 4: Tie Counted-Status And Correctness References Directly To Task 7 Benchmark Records

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- code | tests

**Definition of done**
- Story 2 defines explicit fields for counted-status closure and Task 6
  correctness references on the shared Task 7 benchmark records.
- Counted benchmark cases remain joinable to the exact Task 6 evidence that
  authorized them.
- Story 2 avoids post hoc interpretation for basic counted-versus-excluded
  semantics.

**Execution checklist**
- [ ] Add explicit counted-status and Task 6 reference fields to the Task 7
      record surface or the smallest auditable successor.
- [ ] Define how supported runtime identity, planner-setting references, and
      correctness references remain attached to counted benchmark cases.
- [ ] Ensure excluded cases retain explicit exclusion reasons rather than being
      dropped from the emitted surface.
- [ ] Add focused regression checks for required counted-status field presence
      and semantic stability.

**Evidence produced**
- One explicit Story 2 counted-status rule on shared Task 7 records.
- Regression coverage for required counted-status field stability.

**Risks / rollback**
- Risk: later Task 7 rollups may look correct while hiding which cases were
  actually authorized as counted evidence.
- Rollback/mitigation: attach counted-status and correctness references
  directly to the benchmark records.

### Engineering Task 5: Add A Representative Counted-And-Excluded Benchmark Matrix

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 covers representative counted and excluded benchmark cases that depend
  on the shared counted-supported rule.
- The matrix is broad enough to show that the rule works across the main Task 7
  benchmark slices.
- The matrix remains representative and contract-driven rather than exhaustive
  over every possible exclusion path.

**Execution checklist**
- [ ] Include at least one counted continuity-anchor benchmark case aligned with
      the Task 6 correctness package.
- [ ] Include at least one counted representative structured case aligned with
      the review set.
- [ ] Include at least one explicit excluded case driven by missing
      counted-supported eligibility.
- [ ] Keep threshold satisfaction, diagnosis closure, and summary rollups
      outside the Story 2 matrix.

**Evidence produced**
- One representative Story 2 counted-and-excluded benchmark matrix.
- One review surface for counted-status stability across benchmark slices.

**Risks / rollback**
- Risk: Story 2 may appear correct for one benchmark slice while drifting for
  another.
- Rollback/mitigation: freeze a small but cross-slice counted-status matrix
  early.

### Engineering Task 6: Emit A Stable Story 2 Counted-Supported Bundle Or Rerunnable Checker

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable counted-supported bundle or
  rerunnable checker.
- The bundle records counted versus excluded benchmark status and the Task 6
  correctness references through one stable schema.
- The output is stable enough for Story 3 and later Task 7 stories to build on
  directly.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story2_counted_supported/`).
- [ ] Emit counted and excluded benchmark records through one stable schema plus
      a stable bundle summary.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the relationship to the underlying Task 6 correctness package
      explicit in the bundle summary.

**Evidence produced**
- One stable Story 2 counted-supported bundle or checker.
- One direct citation surface for Task 7 counted benchmark eligibility.

**Risks / rollback**
- Risk: prose-only Story 2 closure will make later reviewers unable to tell
  which benchmark cases actually count toward Task 7 conclusions.
- Rollback/mitigation: emit one machine-reviewable counted-supported bundle
  directly.

### Engineering Task 7: Document The Task 7 Internal Benchmark-Counting Rule And Run The Gate

**Implements story**
- `Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 7 counted-supported rule concretely.
- The Story 2 counted-supported harness and bundle run successfully.
- Story 2 keeps threshold interpretation, diagnosis closure, and summary
  semantics clearly out of the gating rule itself.

**Execution checklist**
- [ ] Document the Story 2 counted-supported benchmark rule and the exclusion
      semantics for cases that fail the gate.
- [ ] Explain how later Task 7 stories should consume the Story 2 gate together
      with the Task 6 correctness package.
- [ ] Run focused Story 2 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/counted_supported_validation.py`.
- [ ] Record stable references to the Story 2 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 2 counted-supported regression checks.
- One stable Story 2 counted-supported bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 2 for threshold closure rather than
  for the counted-supported gate it is meant to provide.
- Rollback/mitigation: document the Story 2 handoff to later Task 7 stories
  explicitly.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one explicit Task 7 counted-supported benchmark rule exists,
- counted benchmark cases are validated directly against the shared Task 6
  correctness package and the supported runtime-path vocabulary,
- excluded benchmark cases retain visible exclusion reasons instead of silently
  disappearing from the Task 7 surface,
- one stable Story 2 counted-supported bundle or rerunnable checker exists for
  direct citation,
- and threshold satisfaction, sensitivity interpretation, diagnosis closure,
  package assembly, and summary guardrails remain clearly assigned to later
  stories.

## Implementation Notes

- Prefer one explicit counted-supported rule over one benchmark-counting
  convention per later consumer.
- Keep Story 2 focused on "which cases may count," not yet on "which counted
  cases win."
- Treat explicit exclusion as part of honest Task 7 evidence, not as noise to
  filter out.
