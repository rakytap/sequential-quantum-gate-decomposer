# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 into one exactness-gated positive-evidence surface:

- every counted supported `partitioned_density` case carries an explicit
  sequential `NoisyCircuit` comparison verdict,
- frozen internal exactness thresholds remain attached directly to the cases
  they govern,
- no-silent-fallback behavior remains part of the admissibility rule for Task 6
  positive evidence,
- and Story 2 closes the contract for "which supported cases pass the internal
  exactness gate" without yet claiming the external Aer slice, unsupported-
  boundary closure, or summary-consistency.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- the mandatory Qiskit Aer slice already owned by Story 3,
- density-validity and continuity-energy emphasis already owned by Story 4,
- runtime and fusion classification comparability already owned by Story 5,
- unsupported-boundary stage separation already owned by Story 6,
- full correctness-package assembly already owned by Story 7,
- and counted-status propagation into later summaries already owned by Story 8.

## Dependencies And Assumptions

- Story 1 already defines the mandatory Task 6 matrix and the stable case-
  identity surface Story 2 must gate honestly.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-004`,
  `P3-ADR-005`, and `P3-ADR-008`.
- Task 3 already provides the primary internal exact-reference substrate Story 2
  should reuse through:
  - `execute_partitioned_with_reference()` in
    `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - `PHASE3_RUNTIME_DENSITY_TOL` in the same module,
  - `runtime_semantics_validation.py`,
  - `mandatory_workload_runtime_validation.py`,
  - and `continuity_runtime_validation.py`.
- Task 4 already provides the fused-capable supported path Story 2 should reuse
  when a counted case exercises real fused execution through:
  - `execute_fused_with_reference()` in
    `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - `fused_semantics_validation.py`,
  - and `benchmarks/density_matrix/artifacts/phase3_task4/`.
- The current implementation learning is that Story 2 should run the shared Task
  6 positive matrix through the fused-capable runtime surface from the start and
  let the actual runtime path resolve to fused or baseline per case, rather than
  splitting baseline-only and fused-capable correctness into separate record
  families.
- The frozen internal thresholds remain:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - and `rho.is_valid(tol=1e-10)`.
- Story 2 should gate counted cases on the supported runtime surface itself,
  not on a disconnected correctness-only reconstruction path.
- Story 2 should treat silent fallback to sequential or non-partitioned
  execution as a disqualifying condition for positive Task 6 evidence.
- The natural implementation home for Task 6 internal-exactness validators is
  the new `benchmarks/density_matrix/correctness_evidence/` package, with
  `records.py` providing the shared positive record spine and
  `sequential_correctness_validation.py` exposing the Story 2 gate.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 6 Internal Positive-Counting Rule

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one explicit rule for when a supported case counts as
  internally exact.
- The rule ties counted status directly to the frozen sequential-baseline
  thresholds and the supported runtime surface.
- The rule is explicit enough that later Task 7 and Task 8 work can reuse it
  without reinterpretation.

**Execution checklist**
- [ ] Freeze the rule that only supported cases with recorded sequential
      comparison verdicts can count as positive Task 6 evidence.
- [ ] Define how Frobenius, trace-deviation, and density-validity checks gate
      counted status.
- [ ] Define how silent fallback and unlabeled runtime paths disqualify positive
      evidence.
- [ ] Keep the external Aer slice, unsupported-boundary closure, and summary
      interpretation outside the Story 2 bar.

**Evidence produced**
- One stable Task 6 internal positive-counting rule.
- One explicit boundary between counted supported cases and excluded cases.

**Risks / rollback**
- Risk: later summaries may count favorable runtime cases even when they do not
  preserve exact internal semantics.
- Rollback/mitigation: freeze the internal counting rule before broadening Story
  2 result surfaces.

### Engineering Task 2: Reuse The Shared Supported Runtime And Sequential Reference Surfaces

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- docs | code

**Definition of done**
- Story 2 reuses the existing supported runtime and sequential-reference
  surfaces wherever they already match the contract.
- Internal exactness checks remain auditable back to the supported Task 3 / Task
  4 outputs.
- Story 2 avoids inventing a second internal-reference language.

**Execution checklist**
- [ ] Reuse `execute_partitioned_with_reference()` for unfused supported cases.
- [ ] Reuse `execute_fused_with_reference()` for counted fused-path cases where
      the current Task 4 boundary supports them.
- [ ] Reuse the existing exact-output record shapes and metric helpers from Task
      3 and Task 4 where they overlap with Task 6 needs.
- [ ] Document which shared runtime and reference helpers Story 2 consumes
      directly.

**Evidence produced**
- One reviewable mapping from the supported runtime surfaces to the Task 6
  internal exactness gate.
- One explicit no-second-internal-reference-language rule for Story 2.

**Risks / rollback**
- Risk: Story 2 may drift into ad hoc comparison logic that later reviewers
  cannot align with the supported runtime surface.
- Rollback/mitigation: reuse the existing supported runtime and reference
  helpers directly.

### Engineering Task 3: Build The Task 6 Sequential Correctness Gate Harness

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 has one reusable harness for evaluating sequential-baseline exactness
  on counted supported cases.
- The harness records threshold metrics and counted-status fields beside the
  shared Story 1 case identity.
- The harness is reusable by later Story 4, Story 5, and Story 7 work.

**Execution checklist**
- [ ] Add a dedicated Story 2 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `sequential_correctness_validation.py` as the primary checker.
- [ ] Evaluate supported cases through the shared runtime and sequential
      reference surfaces.
- [ ] Record Frobenius, maximum-absolute-difference, trace-deviation, and
      density-validity metrics together with counted-status fields.
- [ ] Keep the harness rooted in supported runtime outputs rather than in
      notebook-only or summary-only reconstructions.

**Evidence produced**
- One reusable Story 2 sequential-correctness harness.
- One comparable verdict schema for counted and excluded supported cases.

**Risks / rollback**
- Risk: internal exactness logic may remain scattered across scripts and drift
  from the delivered runtime surface.
- Rollback/mitigation: centralize the Story 2 gate in one stable validation
  entry point.

### Engineering Task 4: Tie Internal Exactness Verdicts Directly To Task 6 Case Records

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- code | tests

**Definition of done**
- Story 2 case records expose internal exactness verdicts beside the workload,
  runtime-path, and planner-setting fields they govern.
- Positive counting cannot occur without the required internal metric fields
  being present.
- Story 2 avoids post hoc interpretation for basic internal pass or fail status.

**Execution checklist**
- [ ] Add internal-threshold fields to the shared Task 6 correctness record or
      the smallest auditable successor.
- [ ] Record Frobenius-norm difference, maximum absolute difference, trace
      deviation, density-validity status, and counted-status fields explicitly.
- [ ] Ensure counted supported records cannot omit those fields silently.
- [ ] Add focused regression checks for field presence and pass/fail stability.

**Evidence produced**
- One explicit internal-exactness record shape for Task 6.
- Regression coverage for required Story 2 field stability.

**Risks / rollback**
- Risk: positive evidence may appear stronger than it is if internal verdicts
  are not attached directly to the cases being counted.
- Rollback/mitigation: record exactness verdicts alongside every counted case.

### Engineering Task 5: Add A Representative Counted And Excluded Internal-Correctness Matrix

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 covers representative counted and excluded supported cases across the
  mandatory Task 6 workload classes and runtime paths.
- The matrix is broad enough to show the internal exactness gate is shared
  across unfused and fused-capable supported cases.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one required microcase, one continuity-anchor case, and
      one structured workload case in the internal-correctness matrix.
- [ ] Include at least one fused-capable supported case where the current Task 4
      runtime boundary makes that comparison real.
- [ ] Include at least one excluded case proving that missing internal verdicts
      or unsupported runtime-path labeling cannot count positively.
- [ ] Keep the matrix focused on internal exactness rather than later Aer or
      summary-consistency closure.

**Evidence produced**
- One representative Story 2 internal-correctness matrix.
- One review surface for counted versus excluded supported cases.

**Risks / rollback**
- Risk: Story 2 may appear coherent on one path while drifting silently on
  another supported runtime path.
- Rollback/mitigation: freeze a small but workload- and path-spanning internal
  matrix.

### Engineering Task 6: Emit A Stable Story 2 Sequential-Correctness Bundle Or Rerunnable Checker

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable sequential-correctness bundle or
  rerunnable checker.
- The bundle records counted and excluded supported cases with explicit internal
  metrics and pass/fail semantics.
- The output shape is stable enough for Stories 4 through 8 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 2 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story2_sequential_gate/`).
- [ ] Emit case identity, runtime-path identity, internal metrics, threshold
      verdicts, and counted-status fields through one stable schema.
- [ ] Record rerun commands and software metadata with the emitted bundle.
- [ ] Keep counted-status derivation explicit in the bundle summary.

**Evidence produced**
- One stable Story 2 sequential-correctness bundle or checker.
- One reusable positive-evidence gate for later Task 6 stories.

**Risks / rollback**
- Risk: prose-only internal exactness closure will make later benchmark and
  publication claims hard to defend.
- Rollback/mitigation: emit one thin machine-reviewable internal gate early.

### Engineering Task 7: Document The Story 2 Internal Counting Rule And Run The Gate

**Implements story**
- `Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Story 2 internal counted-versus-excluded
  rule.
- The Story 2 sequential-correctness harness and bundle run successfully.
- Story 2 makes clear that external Aer closure, unsupported-boundary closure,
  and summary-consistency remain assigned to later stories.

**Execution checklist**
- [ ] Document the Story 2 internal counted-status rule for supported Task 6
      cases.
- [ ] Explain how silent fallback and missing internal verdicts exclude a case
      from positive evidence.
- [ ] Run focused Story 2 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/sequential_correctness_validation.py`.
- [ ] Record stable references to the Story 2 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 2 sequential-correctness regression checks.
- One stable Story 2 sequential-correctness bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may mistake Story 2 counted-status closure for the full
  Task 6 correctness package.
- Rollback/mitigation: document the handoff from Story 2 to Stories 3 through 8
  explicitly.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one explicit Task 6 internal positive-counting rule exists for supported
  cases,
- counted supported cases are evaluated through the shared supported runtime and
  sequential reference surfaces,
- internal verdicts are attached directly to the case records they govern,
- one stable Story 2 sequential-correctness bundle or rerunnable checker exists
  for later reuse,
- and the external Aer slice, unsupported-boundary closure, full-package
  assembly, and summary-consistency guardrails remain clearly assigned to later
  stories.

## Implementation Notes

- Prefer explicit counted-status fields over prose such as "this case matches
  closely enough."
- Keep Story 2 focused on internal evidence admissibility, not yet on the full
  two-baseline package.
- Treat excluded supported cases as required scientific evidence, not as noise
  to hide.
