# Story 6 Implementation Plan

## Story Being Implemented

Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And
Profiler-Backed When Material

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns Task 7 into one explicit diagnosis-path surface:

- representative cases that do not meet the positive threshold are interpreted
  through benchmark-grounded bottleneck evidence rather than vague "still slow"
  language,
- profiler artifacts are attached when profiling materially affects the follow-
  on architecture conclusion,
- diagnosis-only cases remain explicit evidence rather than hidden failures or
  silent exclusions,
- and Story 6 closes the contract for "how Task 7 closes honestly when the
  measurable-benefit path does not win" without yet taking ownership of the
  shared benchmark-package assembly or summary-consistency guardrails.

Out of scope for this story:

- benchmark-matrix inventory already owned by Story 1,
- counted-supported benchmark eligibility already owned by Story 2,
- explicit positive-threshold interpretation already owned by Story 3,
- the required sensitivity matrix already owned by Story 4,
- the shared comparable metric surface already owned by Story 5,
- shared benchmark-package assembly already owned by Story 7,
- and summary-consistency plus bounded-claim guardrails already owned by Story
  8.

## Dependencies And Assumptions

- Stories 1 through 5 already define the benchmark matrix, counted-supported
  gate, threshold review set, sensitivity requirements, and metric surface
  Story 6 must interpret consistently rather than replace.
- The frozen source-of-truth contract is `TASK_7_MINI_SPEC.md`,
  `TASK_7_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-005`,
  `P3-ADR-008`, `P3-ADR-010`, and the performance-claim-boundary item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Task 4 already provides the direct Phase 3 diagnosis-path precedent through
  `benchmarks/density_matrix/partitioned_runtime/fused_performance_validation.py`,
  including benchmark-grounded diagnosis reasons such as:
  - limited fused coverage due to noise boundaries,
  - supported islands left unfused,
  - Python-level fused-kernel overhead or short unitary islands,
  - and no peak-memory reduction on representative cases.
- Story 6 should reuse that diagnosis vocabulary where practical rather than
  inventing a disconnected Task 7 bottleneck language.
- Task 6 already provides the correctness gate Story 6 should treat as a
  prerequisite. Diagnosis-only closure may explain limited benefit, but it
  should not weaken correctness requirements.
- Profiling artifacts are required only when they materially affect the
  architecture conclusion. Story 6 should not make profiling a blanket
  requirement for already clear benchmark outcomes.
- The natural implementation home for Task 7 diagnosis validation is the new
  `benchmarks/density_matrix/performance_evidence/` package, with
  `diagnosis_validation.py` as the primary checker and optional profile
  references carried through the Story 6 bundle.
- Story 6 should treat explicit follow-on-branch mapping as part of the
  diagnosis contract, not as an afterthought in paper prose.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 7 Diagnosis Rule And Profiler-Materiality Boundary

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one explicit rule for when a Task 7 case closes through the
  diagnosis path rather than the positive-threshold path.
- Story 6 defines one explicit boundary for when profiler artifacts are
  materially required.
- The rule is explicit enough that later package and summary work can rely on
  diagnosis-only outcomes safely.

**Execution checklist**
- [ ] Freeze the rule that diagnosis-only closure requires correctness-
      preserving representative cases plus benchmark-grounded bottleneck
      reasons.
- [ ] Freeze the rule that profiling artifacts are required only when they
      materially affect the follow-on architecture conclusion.
- [ ] Define how diagnosis-only cases remain visible rather than being treated
      as silent benchmark failures.
- [ ] Keep benchmark-package assembly and summary semantics outside the Story 6
      bar.

**Evidence produced**
- One stable Task 7 diagnosis rule.
- One explicit profiler-materiality boundary for Task 7.

**Risks / rollback**
- Risk: later benchmark and paper work may hand-wave limited performance instead
  of turning it into useful scientific evidence if Story 6 leaves the rule
  implicit.
- Rollback/mitigation: freeze one diagnosis rule before broad package
  interpretation.

### Engineering Task 2: Reuse The Existing Diagnosis Precedent And Shared Bottleneck Vocabulary As The Base

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- docs | code

**Definition of done**
- Story 6 reuses the Task 4 diagnosis precedent and shared bottleneck
  vocabulary where they already fit Task 7 needs.
- Story 6 keeps diagnosis language aligned with existing benchmark evidence.
- Story 6 avoids inventing a detached limitation vocabulary.

**Execution checklist**
- [ ] Reuse the Task 4 diagnosis reasons and benchmark-grounded bottleneck
      precedent where they already express Story 6 semantics.
- [ ] Reuse shared runtime, fused-coverage, and planner-setting context fields
      from earlier Task 7 stories where they help explain the bottleneck.
- [ ] Add only the Task 7-specific diagnosis fields needed for follow-on branch
      mapping and profiler references.
- [ ] Document where Story 6 intentionally extends earlier diagnosis
      vocabularies.

**Evidence produced**
- One reviewable mapping from the existing Phase 3 diagnosis precedent to the
  Task 7 diagnosis surface.
- One explicit boundary between reused diagnosis vocabulary and Story 6-
  specific additions.

**Risks / rollback**
- Risk: Story 6 may produce plausible limitation statements that later consumers
  cannot relate back to the current benchmark evidence.
- Rollback/mitigation: align Story 6 diagnosis with existing benchmark-grounded
  bottleneck vocabulary wherever practical.

### Engineering Task 3: Build The Task 7 Diagnosis Validation Harness

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 has one reusable harness for validating the diagnosis path on
  representative cases.
- The harness records diagnosis reasons, follow-on-branch mapping, and optional
  profiler references in a machine-reviewable way.
- The harness is reusable by later benchmark-package and summary consumers.

**Execution checklist**
- [ ] Add a dedicated Story 6 validation driver under
      `benchmarks/density_matrix/performance_evidence/`, with
      `diagnosis_validation.py` as the primary checker.
- [ ] Read representative cases, counted-supported status, and the shared metric
      surface directly rather than reconstructing diagnosis from prose.
- [ ] Record diagnosis reasons, follow-on-branch mapping, and profiler
      references where material.
- [ ] Keep the harness rooted in benchmark-grounded diagnosis rather than in
      paper-only explanation.

**Evidence produced**
- One reusable Story 6 diagnosis-validation harness.
- One machine-reviewable diagnosis schema for later Task 7 consumers.

**Risks / rollback**
- Risk: diagnosis logic may remain scattered across notes and console output and
  drift from the benchmark package.
- Rollback/mitigation: centralize Story 6 in one stable validation entry point.

### Engineering Task 4: Attach Diagnosis Reasons And Profile References Directly To Shared Task 7 Records

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- code | tests

**Definition of done**
- Story 6 defines explicit fields for diagnosis reasons, follow-on-branch
  mapping, and optional profiler references on the shared Task 7 records.
- Diagnosis-only cases remain structurally compatible with the shared benchmark
  surface.
- Story 6 avoids post hoc interpretation for basic diagnosis semantics.

**Execution checklist**
- [ ] Add explicit diagnosis-only, bottleneck-reason, follow-on-branch, and
      optional profiler-reference fields to the Task 7 record surface or the
      smallest auditable successor.
- [ ] Define how diagnosis reasons remain attached to stable workload identity,
      planner-setting references, and metric context.
- [ ] Ensure profiler-reference fields remain optional but validated when a case
      claims profiling materially affects the conclusion.
- [ ] Add focused regression checks for diagnosis-field presence and semantic
      stability.

**Evidence produced**
- One explicit Story 6 diagnosis rule on shared Task 7 records.
- Regression coverage for required diagnosis-field stability.

**Risks / rollback**
- Risk: later Task 7 rollups may label a case as diagnosis-only without
  preserving why it was diagnosed that way or what follow-on branch it points
  toward.
- Rollback/mitigation: attach diagnosis reasons and follow-on mapping directly
  to the benchmark records.

### Engineering Task 5: Add A Representative Diagnosis Matrix Across Main Bottleneck Classes

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- tests | validation automation

**Definition of done**
- Story 6 covers representative diagnosis-only cases across the main current
  bottleneck classes.
- The matrix is broad enough to show that diagnosis stays benchmark-grounded
  rather than collapsing to one generic "slow" label.
- The matrix remains representative and contract-driven rather than exhaustive
  over every possible performance limitation.

**Execution checklist**
- [ ] Include at least one diagnosis-only case driven by limited fused coverage
      or fragmentation across noise boundaries.
- [ ] Include at least one diagnosis-only case driven by persistent runtime
      overhead or lack of peak-memory reduction.
- [ ] Include at least one case where profiler references are absent because the
      benchmark diagnosis is already clear.
- [ ] Keep package assembly and summary semantics outside the Story 6 matrix.

**Evidence produced**
- One representative Story 6 diagnosis matrix across main bottleneck classes.
- One review surface for benchmark-grounded limitation reporting.

**Risks / rollback**
- Risk: Story 6 may appear correct for one limitation class while drifting for
  another.
- Rollback/mitigation: freeze a small but multi-reason diagnosis matrix early.

### Engineering Task 6: Emit A Stable Story 6 Diagnosis Bundle Or Rerunnable Checker

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable diagnosis bundle or rerunnable
  checker.
- The bundle records diagnosis-only cases, bottleneck reasons, follow-on-branch
  mapping, and optional profiler references through one stable schema.
- The output is stable enough for later Task 7 package and paper review to cite
  directly.

**Execution checklist**
- [ ] Add a dedicated Story 6 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task7/story6_diagnosis_path/`).
- [ ] Emit diagnosis-only benchmark cases through one stable schema plus a
      stable bundle summary.
- [ ] Record rerun commands, software metadata, and any referenced profiler
      artifact locations with the emitted bundle.
- [ ] Keep the relationship to the positive-threshold path explicit in the
      bundle summary.

**Evidence produced**
- One stable Story 6 diagnosis bundle or checker.
- One direct citation surface for Task 7 diagnosis-only benchmark evidence.

**Risks / rollback**
- Risk: prose-only Story 6 closure will make later reviewers unable to tell
  whether the diagnosis path was actually benchmark-grounded.
- Rollback/mitigation: emit one machine-reviewable diagnosis bundle directly.

### Engineering Task 7: Document The Diagnosis Path And Run The Story 6 Guardrail

**Implements story**
- `Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 6 diagnosis rule, profiler-
  materiality boundary, and follow-on-branch mapping concretely.
- The Story 6 diagnosis harness and bundle run successfully.
- Story 6 keeps benchmark-package assembly and summary semantics clearly outside
  the diagnosis guardrail itself.

**Execution checklist**
- [ ] Document the Story 6 diagnosis rule, the profiler-materiality boundary,
      and the follow-on-branch mapping.
- [ ] Explain how later Task 7 stories should consume the Story 6 diagnosis
      surface directly.
- [ ] Run focused Story 6 regression coverage and verify
      `benchmarks/density_matrix/performance_evidence/diagnosis_validation.py`.
- [ ] Record stable references to the Story 6 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 6 diagnosis regression checks.
- One stable Story 6 diagnosis bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 6 for broad architecture planning
  rather than for the bounded diagnosis path Task 7 is meant to provide.
- Rollback/mitigation: document the Story 6 boundary to package and summary work
  explicitly.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one explicit Task 7 diagnosis rule and profiler-materiality boundary exist,
- diagnosis-only cases remain benchmark-grounded and structurally compatible
  with the shared Task 7 benchmark surface,
- follow-on-branch mapping remains explicit rather than hidden in prose-only
  limitation wording,
- one stable Story 6 diagnosis bundle or rerunnable checker exists for direct
  citation,
- and shared benchmark-package assembly plus summary guardrails remain clearly
  assigned to later stories.

## Implementation Notes

- Prefer benchmark-grounded diagnosis language over soft "still slow" wording.
- In actual coding order, Story 6 should consume the shared metric-record and
  timing surface once Story 5 lands it rather than recreating measurement logic
  inside the diagnosis path.
- Keep Story 6 focused on honest limitation evidence, not on implementing the
  deferred follow-on branch itself.
- Treat profiler artifacts as supporting evidence when material, not as a
  blanket requirement for every diagnosis-only case.
