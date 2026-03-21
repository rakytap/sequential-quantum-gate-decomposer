# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the Task 5 calibration claim into one explicit planner-candidate
surface:

- supported planner candidates and planner-setting variants are identified
  through one shared reviewable surface rather than through ad hoc tuning notes,
- the calibration contract records planner identity and settings before any
  benchmark-grounded claim is selected,
- the candidate surface stays aligned with the canonical noisy planner contract
  and the frozen workload matrix,
- and Story 1 closes the contract for "what exactly was calibrated" without yet
  claiming workload-matrix completeness, density-aware superiority, or final
  claim selection.

Out of scope for this story:

- calibration workload-matrix anchoring already owned by Story 2,
- density-aware signal differentiation already owned by Story 3,
- correctness-gated positive evidence already owned by Story 4,
- final supported-claim selection already owned by Story 5,
- stable calibration-bundle packaging already owned by Story 6,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, and
  `P3-ADR-006`.
- Task 1 already established the canonical noisy planner surface Story 1 must
  calibrate against in `squander/partitioning/noisy_planner.py`, especially:
  - `CanonicalNoisyPlannerSurface`,
  - `build_canonical_planner_surface_from_operation_specs()`,
  - and the stable provenance vocabulary for `requested_mode`, `source_type`,
    `entry_route`, `workload_family`, and `workload_id`.
- Task 2 already established the schema-versioned partition handoff surface Story
  1 should treat as the auditable planning result boundary through:
  - `NoisyPartitionDescriptorSet`,
  - `NoisyPartitionDescriptor`,
  - and `NoisyPartitionDescriptorMember`.
- Task 3 and Task 4 already established the runtime and fused-runtime metric
  surfaces Story 1 will eventually feed, especially through:
  - `NoisyRuntimeExecutionResult` in `squander/partitioning/noisy_runtime.py`,
  - `execute_partitioned_with_reference()` and `execute_fused_with_reference()`
    in `benchmarks/density_matrix/partitioned_runtime/common.py`,
  - and the existing fused and non-fused validation bundles under
    `benchmarks/density_matrix/artifacts/partitioned_runtime/` and
    `benchmarks/density_matrix/artifacts/partitioned_runtime/`.
- The existing state-vector planner stack under
  `squander/partitioning/kahn.py`, `squander/partitioning/tdag.py`,
  `squander/partitioning/ilp.py`, and `squander/partitioning/tools.py`
  provides useful prior art about planner families and tunable settings, but
  Story 1 must not let those legacy surfaces become the implicit calibration
  contract.
- The mandatory workload builders Story 1 should use when checking candidate
  coverage already exist through:
  - `build_phase3_continuity_partition_descriptor_set()` in
    `squander/partitioning/noisy_planner.py`,
  - `iter_story2_microcase_descriptor_sets()` in
    `benchmarks/density_matrix/planner_surface/workloads.py`,
  - and `iter_story2_structured_descriptor_sets()` in the same module.
- Story 1 should prefer one explicit candidate vocabulary spanning structural
  and optimization-guided baselines over one vocabulary per workload family.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 5 Planner-Candidate Taxonomy

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- docs | validation automation

**Definition of done**
- Story 1 defines one stable candidate taxonomy for the supported Task 5
  calibration surface.
- The taxonomy is explicit enough that later benchmark and paper work can say
  what was calibrated without guessing from code branches.
- The taxonomy remains narrow enough that later stories can add workload,
  correctness, and claim-selection behavior cleanly.

**Execution checklist**
- [ ] Freeze the minimum candidate vocabulary for Task 5, including structural
      baselines such as `kahn`, `tdag`, and `gtqcp`, plus any in-scope
      optimization-guided candidate or successor.
- [ ] Define how planner-setting variants are named and compared inside one
      candidate surface.
- [ ] Define what counts as one auditable candidate versus one exploratory or
      discarded tuning branch.
- [ ] Keep workload-matrix completeness, correctness gating, and final
      claim-selection closure outside the Story 1 bar.

**Evidence produced**
- One stable Task 5 planner-candidate taxonomy.
- One explicit boundary between candidate definition and later claim closure.

**Risks / rollback**
- Risk: if candidate identity remains informal, later calibration evidence will
  be hard to audit and easy to overstate.
- Rollback/mitigation: freeze the candidate taxonomy before broadening
  calibration runs.

### Engineering Task 2: Define A Shared Candidate Descriptor Surface

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- docs | code

**Definition of done**
- Story 1 defines one shared descriptor for Task 5 planner candidates.
- The descriptor records planner identity, planner settings, and claim-relevant
  metadata in one auditable shape.
- The surface is additive to the existing planner and runtime provenance
  vocabulary rather than a disconnected new language.

**Execution checklist**
- [ ] Define one candidate descriptor record in
      `squander/partitioning/noisy_planner.py` or the smallest adjacent helper.
- [ ] Record planner family, planner version or strategy ID, and explicit
      planner settings such as partition-size or heuristic parameters.
- [ ] Reuse the existing case-level provenance vocabulary where it already fits.
- [ ] Keep the candidate descriptor separate from later metric or verdict fields.

**Evidence produced**
- One reviewable Task 5 candidate descriptor shape.
- One explicit mapping between planner identity and planner settings.

**Risks / rollback**
- Risk: candidate identity may drift into free-form labels that later bundles
  cannot compare safely.
- Rollback/mitigation: define one shared candidate descriptor before building
  story-specific validators.

### Engineering Task 3: Build A Deterministic Candidate Enumeration Surface

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- code | tests

**Definition of done**
- Story 1 exposes one deterministic surface for enumerating supported planner
  candidates and setting variants.
- Enumeration is reproducible for a fixed repository state and fixed story
  configuration.
- The enumeration surface avoids hidden per-workload overrides or one-off
  benchmark-only candidate injection.

**Execution checklist**
- [ ] Implement one candidate-enumeration helper in
      `squander/partitioning/noisy_planner.py` or the smallest adjacent Task 5
      helper.
- [ ] Keep enumeration deterministic and reviewable for a fixed configuration.
- [ ] Avoid hidden workload-specific allowlists or manual candidate injection.
- [ ] Add focused tests proving candidate enumeration stability.

**Evidence produced**
- One deterministic Task 5 candidate-enumeration surface.
- Regression coverage for candidate-surface stability.

**Risks / rollback**
- Risk: candidate sets may change implicitly between benchmark runs and weaken
  calibration traceability.
- Rollback/mitigation: make enumeration deterministic and test it directly.

### Engineering Task 4: Align Candidate Records With Shared Workload And Runtime Provenance

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- code | docs

**Definition of done**
- Story 1 candidate records align with the shared Phase 3 provenance tuple where
  fields overlap.
- Candidate records can be joined cleanly with later workload, runtime, and
  fused-coverage evidence.
- Story 1 does not invent a planner-only provenance language that later stories
  must translate mentally.

**Execution checklist**
- [ ] Reuse overlapping provenance fields from the Task 1 and Task 2 audit
      vocabulary where they already match Story 1 needs.
- [ ] Document how candidate identity combines with workload identity and runtime
      settings in later Task 5 stories.
- [ ] Add only the Task 5-specific candidate fields needed for planner-family and
      setting review.
- [ ] Keep candidate descriptors structurally compatible with later artifact
      bundling.

**Evidence produced**
- One aligned Task 5 candidate-provenance rule.
- One explicit boundary between reused provenance fields and new candidate
  fields.

**Risks / rollback**
- Risk: later bundles may become hard to interpret if candidate identity and
  workload identity are recorded through unrelated vocabularies.
- Rollback/mitigation: align candidate records with the shared provenance tuple
  from the start.

### Engineering Task 5: Add A Representative Candidate Matrix Across Supported Planner Families

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- tests | validation automation

**Definition of done**
- Story 1 covers a representative candidate matrix across the supported planner
  families relevant to Task 5.
- The matrix is broad enough to show that Task 5 calibration is not about one
  hand-picked planner only.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one structural baseline candidate.
- [ ] Include at least one stronger structural or optimization-guided candidate
      when it is in scope for the frozen Phase 3 contract.
- [ ] Include at least one setting variant that exercises explicit candidate
      metadata such as partition-size or heuristic weights.
- [ ] Keep the matrix small enough to stay reviewable and reproducible.

**Evidence produced**
- One representative Task 5 candidate matrix.
- One review surface for cross-candidate comparison later in the task.

**Risks / rollback**
- Risk: the calibration narrative may collapse onto one planner family before the
  supported candidate surface is explicit.
- Rollback/mitigation: freeze a small but representative candidate matrix early.

### Engineering Task 6: Add Fast Story 1 Regression Coverage For Candidate-Surface Stability

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- tests

**Definition of done**
- Story 1 has focused regression checks for candidate-surface stability.
- The checks prove the candidate taxonomy and enumeration logic remain auditable
  and repeatable.
- The regression slice stays narrower than later calibration, correctness, and
  claim-selection work.

**Execution checklist**
- [ ] Add a dedicated Task 5 regression surface in
      `tests/partitioning/test_planner_calibration.py`.
- [ ] Assert stable candidate names, stable setting fields, and stable candidate
      ordering or canonicalization for representative cases.
- [ ] Assert that unsupported or exploratory candidate labels do not leak into
      the supported Story 1 surface silently.
- [ ] Keep the checks at the candidate layer rather than requiring full
      calibration runs.

**Evidence produced**
- Fast regression coverage for Story 1 candidate-surface stability.
- One repeatable test surface for later Task 5 work to extend.

**Risks / rollback**
- Risk: candidate drift may remain hidden until later benchmark packaging.
- Rollback/mitigation: add a dedicated fast candidate regression slice early.

### Engineering Task 7: Emit A Stable Story 1 Candidate-Audit Bundle Or Rerunnable Checker

**Implements story**
- `Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface`

**Change type**
- validation automation | docs

**Definition of done**
- Story 1 emits one stable machine-reviewable candidate-audit bundle or
  rerunnable checker.
- The bundle records candidate identity and settings with enough provenance to
  audit them later.
- The output shape is stable enough for Stories 2 through 7 to extend.

**Execution checklist**
- [ ] Add a dedicated Story 1 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `planner_candidate_audit_validation.py` as the primary checker.
- [ ] Add a dedicated Story 1 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/planner_calibration/planner_candidate_audit/`).
- [ ] Emit candidate descriptors, planner settings, and shared provenance fields
      for the representative Task 5 candidate surface.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 1 candidate-audit bundle or checker.
- One reusable evidence surface for later Task 5 stories.

**Risks / rollback**
- Risk: prose-only candidate closure will make later calibration claims hard to
  justify and easy to overstate.
- Rollback/mitigation: emit one thin machine-reviewable candidate surface early.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- one explicit Task 5 planner-candidate taxonomy exists for the supported
  calibration surface,
- supported planner candidates and setting variants can be enumerated through
  one deterministic shared surface,
- candidate records align with the shared Phase 3 provenance vocabulary where
  fields overlap,
- one stable Story 1 candidate-audit bundle or rerunnable checker exists for
  later reuse,
- and workload anchoring, density-aware differentiation, correctness gating,
  supported-claim selection, and approximation-boundary handling remain clearly
  assigned to later stories.

## Implementation Notes

- Prefer one shared candidate vocabulary over one vocabulary per benchmark
  harness.
- Keep Story 1 focused on "what is being calibrated," not yet on "which
  candidate wins."
- Treat exploratory candidate ideas as explicit non-claim branches unless they
  are promoted into the supported candidate surface.
