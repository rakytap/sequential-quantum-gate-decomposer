# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One
Comparable Measurement Surface

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one explicit comparable metric surface:

- runtime, peak memory, planning overhead, partition context, and fused-
  coverage behavior are recorded through one shared benchmark record shape,
- repeated-timing or median-timing interpretation remains auditable rather than
  hidden behind summary prose,
- counted and diagnosis-only cases use one comparable metric vocabulary where
  fields overlap,
- and Story 5 closes the contract for "how Task 7 measurements are recorded"
  without yet taking ownership of diagnosis interpretation, shared benchmark-
  package assembly, or summary-consistency guardrails.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- counted-supported benchmark eligibility already owned by Story 2,
- explicit positive-threshold interpretation already owned by Story 3,
- cross-knob sensitivity interpretation already owned by Story 4,
- diagnosis-path bottleneck reporting already owned by Story 6,
- shared benchmark-package assembly already owned by Story 7,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- Stories 1 through 4 already define the benchmark matrix, counted-supported
  gate, representative threshold set, and required sensitivity dimensions Story
  5 must record consistently rather than replace.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-006`, and `P3-ADR-009`.
- Task 3 already records the core runtime metrics Story 5 should reuse through
  `NoisyRuntimeExecutionResult`, including `runtime_ms`, `peak_rss_kb`,
  partition count, max partition span, and runtime-path identity.
- Task 4 already records fused-coverage and structured performance precedent in
  `benchmarks/density_matrix/partitioned_runtime/fused_performance_validation.py`;
  Story 5 should preserve direct compatibility with that surface where fields
  overlap.
- Task 5 already records planning time and planner-setting identity in
  `benchmarks/density_matrix/planner_calibration/`; Story 5 should carry that
  planning-overhead vocabulary into the Task 7 metric surface rather than
  inventing a second timing language.
- Story 5 should not decide by itself whether a case closes the positive
  threshold or the diagnosis path. It should provide the comparable measurement
  surface both paths rely on.
- The current implementation learning is that Task 7 needs repeated-run timing
  or an explicit median-computation procedure preserved in the artifacts rather
  than only final aggregate numbers.
- The natural implementation home for Task 7 metric recording is the new
  `benchmarks/density_matrix/performance_evidence/` package, with `records.py`,
  `common.py`, and `metric_surface_validation.py` as the shared Story 5
  surface.
- Story 5 should treat metric comparability across counted and diagnosis-only
  cases as part of the contract, not as a downstream formatting convenience.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Metric Inventory And Repeated-Timing Rule

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one explicit inventory of the metrics every Task 7 benchmark
  record must preserve where applicable.
- Story 5 defines one explicit repeated-timing or median-computation rule for
  performance interpretation.
- The metric inventory is explicit enough that later threshold, diagnosis, and
  summary work can rely on it safely.

**Execution checklist**
- [ ] Freeze the required Story 5 metric inventory around runtime, peak memory,
      planning time, partition count, max partition span, runtime-path
      identity, and fused-coverage summaries.
- [ ] Freeze the rule for how repeated timing or median timing is recorded and
      recomputed.
- [ ] Define which metric fields are required on counted cases and which may be
      optional or diagnosis-only depending on the slice.
- [ ] Keep diagnosis interpretation, package assembly, and summary semantics
      outside the Story 5 bar.

**Evidence produced**
- One stable Task 7 metric inventory.
- One explicit repeated-timing or median-computation rule for Task 7.

**Risks / rollback**
- Risk: later benchmark and paper work may compare incompatible metric surfaces
  if Story 5 leaves the inventory or timing rule implicit.
- Rollback/mitigation: freeze one metric inventory before broad package
  assembly.

### Engineering Task 2: Reuse The Shared Runtime, Planning, And Fused-Coverage Surfaces As The Base

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- docs | code

**Definition of done**
- Story 5 reuses the existing Task 3 runtime, Task 4 fused-coverage, and Task 5
  planning-overhead fields where they already fit the Task 7 contract.
- Story 5 keeps metric naming aligned with earlier Phase 3 benchmark surfaces.
- Story 5 avoids inventing a disconnected measurement vocabulary.

**Execution checklist**
- [ ] Reuse `NoisyRuntimeExecutionResult` field names directly where they match
      Story 5 needs.
- [ ] Reuse fused-coverage and representative structured benchmark fields from
      the Task 4 performance surface where they overlap.
- [ ] Reuse Task 5 planning-time and planner-setting references directly where
      they already express the required overhead surface.
- [ ] Document any additive Story 5 metric fields explicitly rather than
      renaming shared measurement vocabulary.

**Evidence produced**
- One reviewable mapping from the shared runtime, fused, and planning surfaces
  to the Task 7 metric surface.
- One explicit boundary between reused phase-wide metric names and Story 5-
  specific additions.

**Risks / rollback**
- Risk: Task 7 may produce plausible metrics that later consumers still need to
  translate manually because Story 5 drifted from the shared runtime surfaces.
- Rollback/mitigation: align Story 5 with existing Phase 3 metric vocabulary
  wherever practical.

### Engineering Task 3: Define A Shared Task 7 Metric Record And Comparable Measurement Surface

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- code | tests

**Definition of done**
- Story 5 defines one shared Task 7 metric record shape for counted and
  diagnosis-only cases where fields overlap.
- The record separates case identity, measured values, repeated-run timing
  details, and later interpretation fields cleanly.
- The shape remains stable across continuity and structured benchmark slices.

**Execution checklist**
- [ ] Define one shared Task 7 metric record in
      `benchmarks/density_matrix/performance_evidence/records.py` or the
      smallest auditable successor.
- [ ] Record runtime, peak memory, planning time, partition context,
      runtime-path identity, fused-coverage summaries, and repeated-timing
      details where applicable.
- [ ] Keep diagnosis reasons, summary rollups, and final claim-closure fields
      outside the Story 5 record bar.
- [ ] Add regression checks for top-level metric-schema stability.

**Evidence produced**
- One stable shared Task 7 metric record shape.
- Regression checks for metric-schema stability across counted and diagnosis-only
  cases.

**Risks / rollback**
- Risk: Story 5 outputs may remain individually plausible but structurally hard
  to compare across the Task 7 benchmark surface.
- Rollback/mitigation: freeze one shared metric record before broadening bundle
  emission.

### Engineering Task 4: Build The Comparable Metric-Emission Harness

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 has one reusable harness for emitting comparable metric records across
  the Task 7 benchmark surface.
- The harness preserves stable metric names and repeated-timing details.
- The harness is reusable by later threshold, diagnosis, package, and summary
  consumers.

**Execution checklist**
- [ ] Add a dedicated Story 5 validation driver under
      `benchmarks/density_matrix/performance_evidence/`, with
      `metric_surface_validation.py` as the primary checker.
- [ ] Emit counted and diagnosis-only benchmark records through one comparable
      metric surface.
- [ ] Record repeated-run measurements or the auditable median-computation
      procedure directly in the emitted output.
- [ ] Keep the harness rooted in comparable metric emission rather than in later
      threshold or diagnosis interpretation.

**Evidence produced**
- One reusable Story 5 comparable-metric harness.
- One machine-reviewable metric-emission schema for later Task 7 consumers.

**Risks / rollback**
- Risk: metric completeness may remain scattered across scripts and drift from
  the shared record shape.
- Rollback/mitigation: centralize Story 5 in one stable metric-emission entry
  point.

### Engineering Task 5: Add A Representative Metric-Completeness Matrix Across Counted And Diagnosis-Only Cases

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 proves that the required metric fields remain present across the main
  Task 7 benchmark slices.
- The matrix is broad enough to catch counted-versus-diagnosis metric drift
  early.
- The matrix remains representative and contract-driven rather than exhaustive
  over every benchmark record shape.

**Execution checklist**
- [ ] Include at least one counted benchmark case and one diagnosis-only case in
      the Story 5 metric-completeness matrix.
- [ ] Assert presence of runtime, peak memory, planning time, partition
      context, and runtime-path identity on the representative cases.
- [ ] Assert fused-coverage summaries and repeated-timing fields where the case
      class requires them.
- [ ] Keep diagnosis interpretation, package assembly, and summary semantics
      outside the Story 5 matrix.

**Evidence produced**
- One representative Story 5 metric-completeness matrix.
- One review surface for comparable metric stability across Task 7 case classes.

**Risks / rollback**
- Risk: Story 5 may look coherent on counted cases while diagnosis-only records
  drift silently.
- Rollback/mitigation: freeze a small but cross-class metric-completeness matrix
  early.

### Engineering Task 6: Emit A Stable Story 5 Metric-Surface Bundle Or Rerunnable Checker

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable metric-surface bundle or
  rerunnable checker.
- The bundle records comparable measurements and repeated-timing semantics
  through one stable schema.
- The output is stable enough for direct citation by later Task 7 stories.

**Execution checklist**
- [ ] Add a dedicated Story 5 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/performance_evidence/metric_surface/`).
- [ ] Emit comparable metric records through one stable schema plus a stable
      bundle summary.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep the relationship to the shared runtime, fused, and planning surfaces
      explicit in the bundle summary.

**Evidence produced**
- One stable Story 5 metric-surface bundle or checker.
- One direct citation surface for Task 7 comparable measurements.

**Risks / rollback**
- Risk: prose-only Story 5 closure will make later reviewers unable to tell
  whether compared benchmark records actually carried the same metric fields.
- Rollback/mitigation: emit one machine-reviewable metric-surface bundle
  directly.

### Engineering Task 7: Document The Comparable-Measurement Rule And Run The Story 5 Surface

**Implements story**
- `Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 5 metric inventory and repeated-
  timing rule concretely.
- The Story 5 metric harness and bundle run successfully.
- Story 5 keeps diagnosis interpretation and summary semantics clearly outside
  the measurement rule itself.

**Execution checklist**
- [ ] Document the Story 5 metric inventory and the repeated-timing or median-
      computation rule.
- [ ] Explain how later Task 7 stories should consume the Story 5 metric surface
      directly.
- [ ] Run focused Story 5 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/metric_surface_validation.py`.
- [ ] Record stable references to the Story 5 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 5 metric-surface regression checks.
- One stable Story 5 metric-surface bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 5 for performance interpretation
  rather than for the comparable measurement surface it is meant to provide.
- Rollback/mitigation: document the Story 5 handoff to threshold, diagnosis, and
  summary stories explicitly.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one explicit Task 7 metric inventory and repeated-timing rule exist,
- counted and diagnosis-only benchmark records emit comparable runtime, memory,
  planning, partition-context, and fused-coverage fields where they overlap,
- repeated-timing interpretation remains auditable rather than hidden behind
  aggregate prose,
- one stable Story 5 metric-surface bundle or rerunnable checker exists for
  direct citation,
- and diagnosis interpretation, package assembly, and summary guardrails remain
  clearly assigned to later stories.

## Implementation Notes

- Prefer additive metric-surface evolution over renaming shared Phase 3 runtime
  fields.
- In actual coding order, the shared metric-record primitives from Story 5 may
  land earlier than the Story 3, Story 4, or Story 6 verdict logic because
  those stories depend on one stable comparable measurement surface.
- Keep Story 5 focused on comparable measurement, not yet on interpreting why a
  case won or lost.
- Treat repeated-timing auditability as part of the measurement contract, not as
  a later documentation nicety.
