# Story 2 Implementation Plan

## Story Being Implemented

Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the Task 5 calibration claim into one frozen workload-matrix
surface:

- calibration is anchored to the mandatory continuity, micro-validation, and
  structured-workload inventory rather than to ad hoc exploratory examples,
- stable workload IDs, deterministic seed rules, and noise-pattern labels stay
  intact across the calibration package,
- the workload matrix remains aligned with the frozen Phase 3 support boundary
  and benchmark anchors,
- and Story 2 closes the contract for "which workloads define the calibration
  surface" without yet claiming density-aware differentiation, correctness
  gating, or final planner selection.

Out of scope for this story:

- planner-candidate identity already owned by Story 1,
- density-aware signal differentiation already owned by Story 3,
- correctness-gated positive evidence already owned by Story 4,
- final supported-claim selection already owned by Story 5,
- stable calibration-bundle packaging already owned by Story 6,
- and explicit approximation and deferred-boundary handling already owned by
  Story 7.

## Dependencies And Assumptions

- Story 1 already defines the supported planner-candidate surface Story 2 must
  anchor to workloads rather than reinterpret.
- The frozen source-of-truth contract is `TASK_5_MINI_SPEC.md`,
  `TASK_5_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-006`,
  `P3-ADR-007`, and `P3-ADR-009`.
- The required workload builders Story 2 should reuse already exist through:
  - `build_phase3_continuity_partition_descriptor_set()` in
    `squander/partitioning/noisy_planner.py`,
  - `iter_story2_microcase_descriptor_sets()` in
    `benchmarks/density_matrix/planner_surface/workloads.py`,
  - `iter_story2_structured_descriptor_sets()` in the same module,
  - `STRUCTURED_FAMILY_NAMES`,
  - `STRUCTURED_QUBITS`,
  - and `MANDATORY_NOISE_PATTERNS`.
- The existing Phase 3 workload selection helpers already provide useful prior
  art for Story 2 under `benchmarks/density_matrix/partitioned_runtime/`,
  especially `task4_case_selection.py`.
- Task 3 and Task 4 already established stable workload-facing validation and
  artifact patterns Story 2 should align with through the bundles under
  `benchmarks/density_matrix/artifacts/phase3_task3/` and
  `benchmarks/density_matrix/artifacts/phase3_task4/`.
- Story 2 should preserve the frozen Phase 3 workload identity layer rather than
  renaming workloads per planner candidate or calibration run.
- The mandatory calibration surface must remain bounded to:
  - 2 to 4 qubit microcases,
  - 4, 6, 8, and 10 qubit Phase 2 noisy XXZ `HEA` continuity cases,
  - and mandatory 8 and 10 qubit structured noisy `U3` / `CNOT` families with
    sparse, periodic, and dense local-noise patterns.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 5 Calibration Workload Inventory

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- docs | validation automation

**Definition of done**
- Story 2 defines one stable workload inventory for the supported Task 5
  calibration surface.
- The inventory is explicit enough that later stories can identify what counts
  as mandatory calibration evidence.
- The inventory stays aligned with the frozen Phase 3 support matrix and
  benchmark anchors.

**Execution checklist**
- [ ] Freeze the mandatory Task 5 workload inventory around continuity,
      microcase, and structured-workload slices already accepted at the phase
      level.
- [ ] Define which workload families, sizes, seeds, and noise patterns belong to
      the supported calibration surface.
- [ ] Define which cases are representative versus optional or exploratory.
- [ ] Keep broader workflow growth and broader support-surface expansion outside
      the Story 2 bar.

**Evidence produced**
- One stable Task 5 calibration workload inventory.
- One explicit boundary between mandatory calibration coverage and exploratory
  workload growth.

**Risks / rollback**
- Risk: calibration may drift into favorable ad hoc workloads and weaken the
  scientific claim boundary.
- Rollback/mitigation: freeze the mandatory inventory before broadening
  calibration runs.

### Engineering Task 2: Reuse The Shared Continuity, Microcase, And Structured Workload Builders

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- docs | code

**Definition of done**
- Story 2 reuses the existing shared Phase 3 workload builders instead of
  introducing a second workload language for calibration.
- The workload matrix remains auditable back to the already-frozen Phase 3
  builder surfaces.
- Story 2 avoids benchmark-only workload reconstruction logic.

**Execution checklist**
- [ ] Reuse `build_phase3_continuity_partition_descriptor_set()` for the
      continuity anchor.
- [ ] Reuse `iter_story2_microcase_descriptor_sets()` and
      `iter_story2_structured_descriptor_sets()` or the smallest auditable
      shared successors.
- [ ] Keep workload-family names, workload IDs, and noise-pattern labels aligned
      with the existing builder vocabulary.
- [ ] Document any Story 2-specific matrix metadata explicitly rather than
      renaming shared fields.

**Evidence produced**
- One reviewable mapping from shared Phase 3 workload builders to the Task 5
  calibration matrix.
- One explicit no-second-workload-language rule for Story 2.

**Risks / rollback**
- Risk: calibration comparisons may become hard to trust if Story 2 quietly uses
  different workload builders or naming than earlier tasks.
- Rollback/mitigation: reuse the shared workload builders directly.

### Engineering Task 3: Build The Task 5 Calibration Case-Selection Surface

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- code | validation automation

**Definition of done**
- Story 2 exposes one auditable case-selection surface for the Task 5 workload
  matrix.
- The surface records which cases are mandatory, representative, or exploratory.
- The case-selection logic is reusable by later Task 5 stories and validators.

**Execution checklist**
- [ ] Add a dedicated Task 5 case-selection helper under
      `benchmarks/density_matrix/planner_calibration/`, with
      `task5_case_selection.py` as the primary selector surface.
- [ ] Enumerate the mandatory continuity, microcase, and structured cases with
      stable metadata and support labels.
- [ ] Keep case selection deterministic for fixed seeds and fixed story
      configuration.
- [ ] Avoid planner-specific workload filtering hidden outside the case-selection
      surface.

**Evidence produced**
- One reusable Task 5 case-selection surface.
- One explicit record of which cases define the supported calibration matrix.

**Risks / rollback**
- Risk: later stories may each invent their own workload subsets and blur the
  supported calibration surface.
- Rollback/mitigation: centralize case selection in one shared helper.

### Engineering Task 4: Record Stable Workload Provenance And Calibration Metadata

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- code | docs

**Definition of done**
- Story 2 records the stable provenance fields needed to keep workload identity
  auditable across calibration runs.
- The recorded metadata is rich enough for later claim selection and bundle
  packaging.
- Story 2 keeps workload identity distinct from planner-candidate identity.

**Execution checklist**
- [ ] Record workload family, workload ID, source type, entry route, qubit
      count, seed or deterministic construction rule, and noise-pattern label.
- [ ] Record the planner settings that materially affect case identity, such as
      partition-size settings, separately from workload identity.
- [ ] Reuse the shared Phase 3 provenance tuple where fields already overlap.
- [ ] Keep workload identity stable across multiple candidate evaluations.

**Evidence produced**
- One stable Task 5 workload-provenance tuple.
- One explicit separation between workload identity and planner-setting identity.

**Risks / rollback**
- Risk: calibration summaries may become irreproducible if workload identity and
  planner settings are mixed together loosely.
- Rollback/mitigation: define a stable workload-provenance tuple early.

### Engineering Task 5: Add A Representative Calibration Matrix Across Mandatory Workload Classes

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- tests | validation automation

**Definition of done**
- Story 2 covers representative continuity, microcase, and structured cases for
  calibration review.
- The matrix is broad enough to show that the supported calibration surface
  spans the required workload classes.
- The matrix remains representative and contract-driven rather than exhaustive.

**Execution checklist**
- [ ] Include at least one continuity-anchor case from the frozen 4, 6, 8, and
      10 qubit surface.
- [ ] Include the required 2 to 4 qubit microcase slice.
- [ ] Include representative 8 and 10 qubit structured-family cases across the
      frozen mandatory noise-pattern vocabulary.
- [ ] Keep the representative matrix small enough to stay rerunnable and
      reviewable.

**Evidence produced**
- One representative Task 5 calibration matrix across mandatory workload
  classes.
- One small but contract-complete review surface for later stories.

**Risks / rollback**
- Risk: the workload matrix may look coherent on one family while drifting from
  the full supported Task 5 surface.
- Rollback/mitigation: freeze a small but workload-spanning representative
  matrix.

### Engineering Task 6: Emit A Stable Story 2 Workload-Matrix Bundle Or Rerunnable Checker

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- validation automation | docs

**Definition of done**
- Story 2 emits one stable machine-reviewable workload-matrix bundle or
  rerunnable checker.
- The bundle records representative mandatory cases with stable workload
  provenance and support labels.
- The output shape is stable enough for Stories 3 through 7 to consume directly.

**Execution checklist**
- [ ] Add a dedicated Story 2 validator under
      `benchmarks/density_matrix/planner_calibration/`, with
      `calibration_workload_matrix_validation.py` as the primary checker.
- [ ] Add a dedicated Story 2 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task5/story2_workload_matrix/`).
- [ ] Emit workload identity, workload-family metadata, seed rules,
      noise-pattern labels, and support labels for the representative matrix.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 2 workload-matrix bundle or checker.
- One reusable calibration-inventory surface for later Task 5 work.

**Risks / rollback**
- Risk: prose-only workload anchoring will make later calibration claims hard to
  audit and easy to reinterpret.
- Rollback/mitigation: emit one thin machine-reviewable workload-matrix surface
  early.

### Engineering Task 7: Document The Story 2 Handoff To Later Task 5 Stories

**Implements story**
- `Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory`

**Change type**
- docs

**Definition of done**
- Story 2 notes explain exactly which workload inventory it freezes.
- The workload-matrix bundle is documented as the supported calibration surface,
  not yet as proof of density-aware superiority or claim closure.
- Developer-facing notes point to the Story 2 validators and artifact location.

**Execution checklist**
- [ ] Document the mandatory continuity, microcase, and structured-workload
      inventory for Task 5.
- [ ] Explain that density-aware differentiation belongs to Story 3.
- [ ] Explain that correctness gating, claim selection, and boundary handling
      belong to Stories 4 through 7.
- [ ] Record stable references to the Story 2 validators, tests, and emitted
      bundle.

**Evidence produced**
- Updated developer-facing notes for the Story 2 workload-inventory gate.
- One stable handoff reference for later Task 5 implementation work.

**Risks / rollback**
- Risk: later Task 5 work may over-assume Story 2 already proved the planner
  claim rather than only freezing the workload matrix.
- Rollback/mitigation: document the handoff boundaries explicitly.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- one explicit mandatory workload inventory exists for the supported Task 5
  calibration surface,
- continuity, microcase, and structured workloads are selected through one
  shared auditable case-selection surface,
- stable workload IDs, seeds, and noise-pattern labels are recorded in one
  shared provenance shape,
- one stable Story 2 workload-matrix bundle or rerunnable checker exists for
  later reuse,
- and density-aware differentiation, correctness gating, supported-claim
  selection, stable final packaging, and explicit approximation-boundary
  handling remain clearly assigned to later stories.

## Implementation Notes

- Prefer one shared workload matrix over a growing collection of special-case
  benchmark subsets.
- Keep Story 2 focused on "which workloads define the claim surface," not yet on
  "what the calibrated planner concludes on them."
- Treat optional and exploratory workloads as explicit non-claim extensions
  unless they are promoted into the supported Story 2 inventory.
