# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise
Placement

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the Task 5 calibration claim into one explicit density-aware
signal surface:

- planner ranking or selection changes are tied to mixed-state and noise-aware
  signals rather than to an unchanged state-vector FLOP proxy alone,
- the calibration record makes those signals reviewable through one shared
  machine-readable surface,
- the density-aware change remains comparable against structural-only or
  state-vector-oriented baselines,
- and Story 3 closes the contract for "what makes the planner density-aware"
  without yet claiming correctness-gated closure or final supported-claim
  selection.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- workload-matrix anchoring already owned by Story 2,
- correctness-gated positive evidence already owned by Story 4,
- final supported-claim selection already owned by Story 5,
- stable calibration-bundle packaging already owned by Story 6,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- Stories 1 and 2 already define the candidate surface and workload matrix Story
  3 must differentiate rather than reinterpret.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, and `P3-ADR-006`.
- The existing planner prior-art surface Story 3 should study for comparison
  language and baseline framing already exists in:
  - `squander/partitioning/tools.py`,
  - `squander/partitioning/kahn.py`,
  - `squander/partitioning/tdag.py`,
  - and `squander/partitioning/ilp.py`.
- The current Task 5 implementation surface is narrower than that prior-art
  list: the supported candidate surface is the existing noisy planner with
  auditable `max_partition_qubits` span-budget settings, while state-vector
  planners remain comparison references rather than direct Task 5 candidate
  implementations.
- Task 3 and Task 4 already expose the measured runtime signals Story 3 may use
  as calibration inputs through:
  - `NoisyRuntimeExecutionResult` in `squander/partitioning/noisy_runtime.py`,
  - `execute_partitioned_with_reference()` and `execute_fused_with_reference()`
    in `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - and the existing fused and unfused performance bundles under
    `benchmarks/density_matrix/artifacts/phase3_task3/` and
    `benchmarks/density_matrix/artifacts/phase3_task4/`.
- Story 3 should treat explicit noise placement, support size, partition count
  or span, planning time, runtime, peak memory, and where relevant fused-path
  coverage as legitimate calibration signals when they are recorded through one
  auditable surface.
- Story 3 should not claim global optimality or universal acceleration; it only
  needs to make the density-aware change explicit and reviewable on the current
  span-budget candidate surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Density-Aware Signal Vocabulary And Baseline Contrast Rule

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines one stable vocabulary for the signals that make the Task 5
  planner density-aware.
- The vocabulary is explicit enough that later bundles can show why planner
  ranking changed.
- The story also defines one explicit contrast rule against structural-only or
  state-vector-oriented baselines.

**Execution checklist**
- [ ] Freeze the minimum density-aware signal vocabulary around mixed-state
      support cost, explicit noise placement or density, partition count or
      span, planning time, runtime, peak memory, and where relevant
      fused-coverage opportunity.
- [ ] Define which signals are primary Task 5 evidence and which remain optional
      diagnostics.
- [ ] Freeze one explicit baseline-contrast rule against structural-only or
      state-vector-oriented scoring.
- [ ] Keep correctness gating and final claim selection outside the Story 3 bar.

**Evidence produced**
- One stable Task 5 density-aware signal vocabulary.
- One explicit baseline-contrast rule for later calibration review.

**Risks / rollback**
- Risk: the phrase "density-aware" may remain rhetorical if the signal
  vocabulary is not explicit.
- Rollback/mitigation: freeze the signal vocabulary before broadening metric
  collection.

### Engineering Task 2: Extend The Planning Surface With Reviewable Signal Fields

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- code | tests

**Definition of done**
- Story 3 exposes the density-aware signals through one reviewable planning or
  calibration surface.
- The signal fields are explicit enough to explain score, ranking, or selection
  changes on representative cases.
- The surface remains additive to the shared candidate and workload provenance
  surfaces.

**Execution checklist**
- [ ] Extend `squander/partitioning/noisy_planner.py` or the smallest adjacent
      Task 5 helper with explicit signal fields or summaries.
- [ ] Record candidate-level or case-level signal values without replacing the
      shared provenance tuple.
- [ ] Keep signal recording deterministic for a fixed workload and candidate
      configuration.
- [ ] Add focused regression checks for signal-field presence and shape
      stability.

**Evidence produced**
- One reviewable Task 5 signal-record surface.
- Regression coverage for signal-field stability.

**Risks / rollback**
- Risk: density-aware reasoning may remain buried in planner internals and hard
  to review later.
- Rollback/mitigation: emit the signal fields explicitly in one shared surface.

### Engineering Task 3: Reuse Runtime And Fused-Coverage Metrics As Calibration Inputs

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- code | docs

**Definition of done**
- Story 3 reuses the measured runtime surfaces already established by Tasks 3 and
  4 where those metrics materially support calibration.
- The story does not introduce a disconnected synthetic metric language when
  supported runtime evidence already exists.
- Fused-coverage metrics remain optional but explicit when they matter to
  candidate comparison.

**Execution checklist**
- [ ] Reuse runtime, memory, and reference-comparison helpers from
      `benchmarks/density_matrix/partitioned_runtime/common.py`.
- [ ] Reuse Task 4 fused-coverage summaries where fused-capable behavior matters
      to calibration.
- [ ] Keep runtime-derived signals auditable back to supported runtime outputs
      rather than to synthetic kernels alone.
- [ ] Document which runtime-derived signals Story 3 uses directly.

**Evidence produced**
- One explicit mapping from supported runtime metrics into Task 5 signal fields.
- One reviewable rule for when fused coverage matters to calibration.

**Risks / rollback**
- Risk: calibration may claim runtime-awareness while relying only on analytic or
  synthetic estimates that later work cannot compare directly.
- Rollback/mitigation: reuse the supported runtime metric surfaces wherever they
  are already available.

### Engineering Task 4: Add Contrast Cases That Stress Noise Placement And Mixed-State Support Cost

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- tests | validation automation

**Definition of done**
- Story 3 includes representative contrast cases that make density-aware signal
  differences visible.
- The contrast matrix is broad enough to show that planner behavior changes for
  noisy mixed-state reasons rather than only for generic partition-size reasons.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Add at least one contrast slice that varies explicit noise placement across
      sparse, periodic, and dense patterns.
- [ ] Add at least one contrast slice that changes mixed-state support pressure,
      such as partition span or workload structure, inside the frozen matrix.
- [ ] Add at least one case where fused-coverage opportunity is relevant to the
      recorded signal surface.
- [ ] Keep the contrast matrix representative and rerunnable.

**Evidence produced**
- One representative density-aware contrast matrix for Task 5.
- One reviewable set of cases that can show why planner ranking changes.

**Risks / rollback**
- Risk: density-aware claims may remain too abstract if no contrast cases make
  the signal differences visible.
- Rollback/mitigation: build a small but expressive contrast matrix early.

### Engineering Task 5: Preserve An Explicit Structural-Or-State-Vector Baseline Comparison

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- code | tests | docs

**Definition of done**
- Story 3 preserves at least one explicit structural-only or state-vector
  baseline comparison.
- The baseline comparison remains close enough to the supported Task 5 surface to
  show what changed scientifically.
- The comparison does not become a hidden replacement for the density-aware
  claim.

**Execution checklist**
- [ ] Keep at least one structural-only or unchanged state-vector-oriented
      baseline candidate in the review surface.
- [ ] Record side-by-side signal, score, ranking, or selected-configuration
      outputs for the supported density-aware candidate and the baseline.
- [ ] Document what is still shared versus what changed in the density-aware
      calibration surface.
- [ ] Prevent the baseline from being relabeled as the supported calibrated
      model silently.

**Evidence produced**
- One explicit Task 5 baseline-comparison surface.
- One reviewable explanation of the density-aware change relative to prior art.

**Risks / rollback**
- Risk: later readers may not be able to tell whether Task 5 delivered a real
  density-aware change or only a renamed baseline.
- Rollback/mitigation: preserve one explicit baseline comparison in the artifact
  surface.

### Engineering Task 6: Add Fast Story 3 Regression Coverage For Signal Sensitivity

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- tests

**Definition of done**
- Story 3 has focused regression checks for density-aware signal sensitivity.
- The checks prove that representative contrast cases actually change the
  recorded signal or selection surface.
- The regression slice remains narrower than full correctness or claim-selection
  closure.

**Execution checklist**
- [ ] Add focused Task 5 regression coverage in
      `tests/partitioning/test_phase3_task5.py`.
- [ ] Assert that representative contrast cases produce different signal or score
      outputs when explicit noise placement or support-cost pressure changes.
- [ ] Assert that the baseline-comparison fields remain present and reviewable.
- [ ] Keep the checks at the signal-surface layer rather than requiring final
      benchmark verdicts.

**Evidence produced**
- Fast regression coverage for Story 3 density-signal sensitivity.
- One repeatable test surface for later Task 5 work to extend.

**Risks / rollback**
- Risk: signal drift may remain hidden until broad benchmark or paper packaging.
- Rollback/mitigation: add a dedicated fast signal-sensitivity regression slice.

### Engineering Task 7: Emit A Stable Story 3 Density-Signal Bundle Or Rerunnable Checker

**Implements story**
- `Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 emits one stable machine-reviewable density-signal bundle or
  rerunnable checker.
- The bundle records signal values, baseline comparisons, and score or ranking
  outputs for representative contrast cases.
- The output shape is stable enough for Stories 4 through 7 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 3 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `density_signal_validation.py` as the primary checker.
- [ ] Add a dedicated Story 3 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task5/story3_density_signal/`).
- [ ] Emit representative signal summaries, baseline-comparison outputs, and
      score or ranking summaries for the supported contrast matrix.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 3 density-signal bundle or checker.
- One reusable evidence surface for correctness gating and later claim review.

**Risks / rollback**
- Risk: prose-only signal differentiation will make the density-aware claim hard
  to defend and easy to overstate.
- Rollback/mitigation: emit one thin machine-reviewable signal surface early.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one explicit density-aware signal vocabulary exists for the supported Task 5
  calibration surface,
- representative contrast cases show that planner behavior responds to
  mixed-state and noise-aware signals rather than to an unchanged state-vector
  proxy alone,
- at least one structural-only or state-vector-oriented baseline comparison is
  preserved explicitly,
- one stable Story 3 density-signal bundle or rerunnable checker exists for
  later reuse,
- and correctness gating, final supported-claim selection, stable final
  packaging, and explicit approximation-boundary handling remain clearly
  assigned to later stories.

## Implementation Notes

- Prefer measured and reviewable signal fields over verbal claims about planner
  sophistication.
- Keep Story 3 focused on demonstrating why the planner is density-aware, not
  yet on proving that every positive result counts toward the final claim.
- Treat baseline comparisons as a first-class scientific aid, not as an
  embarrassment to be hidden.
