# Story 2 Implementation Plan

## Story Being Implemented

Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime
Pass/Fail For The Anchor XXZ Workflow

This is a Layer 4 engineering plan for implementing the second behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the supported anchor workflow into the canonical phase-level
exact-regime pass/fail baseline:

- the mandatory workflow-scale validation baseline is defined by the full 4, 6,
  8, and 10 qubit matrix rather than by a hand-selected subset of favorable
  runs,
- each mandatory workflow size is evaluated on at least 10 fixed parameter
  vectors through the supported `density_matrix` path,
- exact noisy energy agreement, supported completion, and explicit pass/fail
  semantics remain visible at workflow scale,
- and the resulting workflow gate stays narrow enough that later stories can add
  the dedicated optimization-trace plus 10-qubit anchor package, the broader
  metric-completeness gate, interpretation rules for optional or incomplete
  evidence, and the final publication bundle without replacing the workflow
  matrix itself.

Out of scope for this story:

- the 1 to 3 qubit local correctness gate owned by Story 1,
- the dedicated reproducible 4- or 6-qubit optimization trace plus documented
  10-qubit anchor evidence package owned by Story 3,
- the broader internal-consistency and execution-stability closure owned by
  Story 4,
- optional, unsupported, and incomplete evidence interpretation closure owned by
  Story 5,
- the final Task 5 top-level provenance and publication bundle owned by Story 6,
- expanding the frozen observable, bridge, gate, or noise support surface,
- and introducing a new workflow validation harness that diverges from the
  existing canonical workflow bundle path.

## Dependencies And Assumptions

- Story 1 is already in place: the phase-level local correctness gate closes the
  mandatory 1 to 3 qubit micro-validation matrix and keeps the workflow claim
  grounded in trusted local evidence.
- Task 2 Story 4 already provides the canonical workflow-scale exactness harness
  in `benchmarks/density_matrix/story2_vqe_density_validation.py`, including
  `build_story4_parameter_sets()`, `run_story4_workflow_case()`, and
  `build_story4_workflow_bundle()`.
- Task 4 Story 5 already specializes that workflow harness for the required
  local-noise workflow baseline in
  `benchmarks/density_matrix/task4_story5_required_local_noise_workflow_validation.py`,
  including required-case accounting and trace-related summary fields.
- The current workflow bundle already exposes the most relevant phase-level
  pass/fail anchors:
  `required_cases`, `required_passed_cases`, `required_pass_rate`,
  `mandatory_baseline_completed`, `unsupported_status_cases`,
  `documented_10q_anchor_present`, `required_trace_present`,
  `required_trace_completed`, and `required_trace_bridge_supported`.
- The frozen Task 5 workflow gate is already defined by `P2-ADR-013`,
  `P2-ADR-014`, and `P2-ADR-015`:
  - mandatory workflow coverage at 4, 6, 8, and 10 qubits,
  - at least 10 fixed parameter vectors per mandatory workflow size,
  - maximum absolute energy error `<= 1e-8` on the mandatory workflow matrix,
  - `100%` pass rate on the mandatory workflow benchmark set,
  - and explicit `density_matrix` attribution with no unsupported-operation
    workarounds.
- The current Phase 2 planning language requires stable case identifiers and
  explicit status checks or equivalent manifest fields for mandatory workflow
  evidence; Story 2 must carry that requirement into the phase-level
  exact-regime gate.
- Story 2 should define the workflow-scale pass/fail baseline for Task 5; it
  should not reopen the workflow anchor, benchmark minimum, numeric thresholds,
  or support-matrix decisions already frozen at the phase level.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 5 Exact-Regime Workflow Inventory

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Task 5 Story 2 names one canonical mandatory workflow inventory for the 4, 6,
  8, and 10 qubit exact-regime gate.
- Every mandatory workflow case has a stable case identifier and parameter-set
  identifier.
- Optional exploratory workflows remain outside the mandatory Story 2 pass/fail
  set.

**Execution checklist**
- [ ] Review the current 4 / 6 / 8 / 10 qubit workflow inventory and freeze it as
      the canonical Task 5 workflow baseline.
- [ ] Confirm that each mandatory workflow size carries at least 10 fixed
      parameter vectors with stable parameter-set identity.
- [ ] Preserve stable case IDs and parameter-set IDs exactly enough that later
      Task 5 stories and publication artifacts can reference them directly.
- [ ] Keep optional exploratory workflow cases out of the mandatory Story 2
      closure rule.

**Evidence produced**
- One named mandatory Task 5 exact-regime workflow inventory.
- Stable workflow and parameter-set identifiers reusable across tests, manifests,
  and later Task 5 artifacts.

**Risks / rollback**
- Risk: if Task 5 Story 2 relies on an implicit workflow list, the phase can
  appear workflow-validated while omitting part of the required exact regime.
- Rollback/mitigation: freeze the mandatory workflow inventory in one canonical
  place and reuse the same identifiers everywhere.

### Engineering Task 2: Reuse The Canonical Workflow Harness And Required-Noise Workflow Bundle Without Forking Them

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- code | validation automation

**Definition of done**
- Task 5 Story 2 derives its workflow evidence from the canonical
  `story2_vqe_density_validation.py` workflow path and the Task 4 Story 5
  required-noise workflow wrapper rather than from a parallel benchmark
  framework.
- The Task 5 workflow baseline uses the same frozen metric names, workflow
  identities, and required-case semantics already accepted lower in the stack.
- Story 2 remains a phase-level interpretation layer, not a replacement for the
  underlying workflow harness.

**Execution checklist**
- [ ] Start from the canonical Story 4 workflow bundle and the Task 4 Story 5
      required local-noise workflow bundle instead of designing a new workflow
      harness.
- [ ] Keep the workflow contract explicitly tied to the supported XXZ plus `HEA`
      plus `density_matrix` plus required local-noise path.
- [ ] Reuse the existing threshold, case, and software metadata helpers where
      practical so Task 5 does not invent a second workflow tolerance scheme.
- [ ] Keep the Task 5 Story 2 layer thin enough that lower-level workflow cases
      still trace directly to the canonical validation code.

**Evidence produced**
- One Task 5 Story 2 assembly path rooted in the canonical workflow and
  required-noise workflow kernels.
- Reviewable traceability from the phase-level workflow gate to the existing
  case-level workflow validation surfaces.

**Risks / rollback**
- Risk: a parallel Task 5 workflow harness would create schema drift and
  conflicting interpretations of what counts as exact-regime workflow success.
- Rollback/mitigation: treat the existing workflow bundles as canonical and
  extend them only through thin Task 5-specific assembly logic.

### Engineering Task 3: Add Explicit Completeness And Status Checks For The Mandatory Workflow Matrix

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- code | tests | validation automation

**Definition of done**
- Task 5 Story 2 can distinguish `complete pass`, `failed`, and `incomplete`
  workflow-matrix states from machine-readable evidence.
- Missing mandatory case IDs, missing parameter-set coverage, duplicate case IDs,
  or missing status fields block Story 2 closure rather than disappearing inside
  an aggregate summary.
- The workflow gate uses the frozen `100%` pass-rate rule on the full mandatory
  matrix.

**Execution checklist**
- [ ] Add or tighten one completeness helper that checks the mandatory workflow
      inventory against the emitted workflow case set for Story 2.
- [ ] Require explicit per-case status fields and an unambiguous aggregate
      pass/fail state for the workflow baseline.
- [ ] Make missing mandatory cases, missing parameter-set coverage, or missing
      status checks fail the Task 5 Story 2 gate explicitly.
- [ ] Keep hand-selected favorable subsets from being summarized as workflow
      baseline success.

**Evidence produced**
- Machine-readable completeness semantics for the Task 5 workflow baseline.
- Focused failure signals for missing mandatory workflow cases or bad status
  fields.

**Risks / rollback**
- Risk: Story 2 can look green while silently omitting required workflow cases if
  completeness is inferred only from aggregate counts.
- Rollback/mitigation: validate exact mandatory case identity and status fields,
  not just summary totals.

### Engineering Task 4: Preserve Exactness, Backend Attribution, And Unsupported-Free Completion In The Phase-Level Workflow Baseline

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- code | validation automation

**Definition of done**
- The Task 5 Story 2 workflow view preserves exactness, backend attribution, and
  unsupported-free completion for every mandatory case.
- Reviewers can see from the phase-level workflow output that the matrix was
  satisfied through the supported `density_matrix` path rather than through
  optional or degraded behavior.
- The workflow summary keeps mandatory workflow counts and unsupported-status
  counts explicit.

**Execution checklist**
- [ ] Reuse the canonical workflow fields already present in the workflow bundle,
      especially `workflow_completed`, `bridge_supported_pass`,
      `required_cases`, `required_passed_cases`, `required_pass_rate`, and
      `unsupported_status_cases`.
- [ ] Ensure the phase-level Story 2 summary preserves maximum absolute energy
      error `<= 1e-8` as the mandatory workflow exactness rule.
- [ ] Keep explicit `density_matrix` attribution and unsupported-free completion
      visible in the Task 5 workflow baseline output.
- [ ] Avoid flattening away workflow-completion or backend-attribution semantics
      when assembling the phase-level view.

**Evidence produced**
- A Task 5 Story 2 workflow summary that preserves exactness, supported
  completion, and backend attribution.
- Reviewable proof that the workflow gate closes on the supported mandatory
  matrix rather than on degraded or unsupported outcomes.

**Risks / rollback**
- Risk: a thin phase-level summary can accidentally hide whether the matrix was
  actually supported, complete, and attributable to the correct backend.
- Rollback/mitigation: treat supported-completion and backend attribution as part
  of the workflow-baseline schema, not optional annotations.

### Engineering Task 5: Add Focused Regression Tests For Task 5 Story 2 Completeness Rules

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the positive Task 5 Story 2 workflow-baseline
  assembly path and representative failure conditions for incomplete workflow
  evidence.
- Regression coverage is specific enough to localize whether a Story 2 failure is
  caused by missing workflow cases, bad parameter-set accounting, or broken
  workflow summary logic.
- The fast test layer stays smaller than the full workflow validation command
  while still protecting the phase-level exact-regime gate.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` or a tightly related successor with Task 5
      Story 2 bundle or completeness tests.
- [ ] Add focused assertions for stable mandatory workflow case IDs, parameter-set
      IDs, and explicit status fields on the Task 5 workflow output.
- [ ] Add at least one representative negative test showing that a missing or
      malformed mandatory workflow case blocks Story 2 closure.
- [ ] Keep full matrix execution in the dedicated validation command rather than
      duplicating the entire workflow matrix inside pytest.

**Evidence produced**
- Focused pytest coverage for Task 5 Story 2 workflow-baseline completeness.
- Reviewable failures that localize phase-level workflow-baseline regressions.

**Risks / rollback**
- Risk: without focused completeness tests, Story 2 can regress at the manifest
  level while the lower-level workflow harness still looks healthy.
- Rollback/mitigation: add small schema- and completeness-level regression tests
  on top of the canonical workflow tests.

### Engineering Task 6: Emit One Stable Task 5 Story 2 Workflow-Baseline Artifact Or Rerunnable Command

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 2 can emit one stable machine-readable workflow-baseline artifact or one
  stable rerunnable command that defines the mandatory 4 / 6 / 8 / 10 qubit
  exact-regime matrix for Task 5.
- The output records stable workflow case identity, parameter-set coverage,
  thresholds, status fields, and aggregate pass/fail interpretation.
- The artifact shape is stable enough that later Task 5 stories can extend it
  without replacing the Story 2 evidence format.

**Execution checklist**
- [ ] Build the Task 5 Story 2 output as a thin wrapper, manifest, or summary
      around the canonical workflow bundles rather than a duplicate case dump
      with divergent semantics.
- [ ] Record suite identity, thresholds, software metadata, mandatory case IDs,
      parameter-set coverage, aggregate pass/fail status, and per-case status
      references or payloads.
- [ ] Keep the output narrow to the workflow baseline rather than mixing in the
      dedicated trace package, optional-evidence interpretation, or publication-
      only fields.
- [ ] Make the artifact or command stable enough for later Task 5 stories and
      paper-facing bundle assembly to reference directly.

**Evidence produced**
- One stable Task 5 Story 2 workflow-baseline artifact or rerunnable command.
- A reusable output schema for later Task 5 workflow-baseline assembly.

**Risks / rollback**
- Risk: ad hoc workflow-baseline summaries will drift and make later trace,
  interpretation, and publication layers harder to audit.
- Rollback/mitigation: define one thin structured output now and extend it
  incrementally.

### Engineering Task 7: Document The Phase-Level Exact-Regime Workflow Gate And Its Hand-Offs

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Task 5 Story 2 validates, how to rerun it,
  and why it is the canonical workflow-scale exact-regime gate for the phase.
- The notes make clear that Story 2 sits above the local correctness gate and
  below the dedicated trace-and-anchor, metric-completeness, interpretation, and
  publication stories.
- The documentation stays aligned with the frozen Phase 2 workflow thresholds and
  does not overclaim broader validation closure.

**Execution checklist**
- [ ] Document the Task 5 Story 2 validation entry point and its relationship to
      the canonical Story 4 and Task 4 Story 5 workflow bundles.
- [ ] Make the mandatory workflow gate explicit:
      stable workflow IDs, at least 10 parameter vectors per size,
      `<= 1e-8` energy error, supported completion, and `100%` pass rate.
- [ ] Explain how Story 2 hands off the dedicated trace-plus-anchor package,
      broader metric completeness, optional-evidence interpretation, and final
      provenance packaging to Stories 3 to 6.
- [ ] Keep optional secondary baselines and unsupported-case material clearly out
      of the Story 2 definition of done.

**Evidence produced**
- Updated developer-facing guidance for the Task 5 Story 2 exact-regime workflow
  gate.
- One stable place where Story 2 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 2 is poorly documented, later contributors may confuse it with
  the lower-level workflow harness or with the full Task 5 validation package.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  outputs used for Story 2 completion.

### Engineering Task 8: Run Story 2 Workflow Validation And Confirm The Mandatory Exact-Regime Matrix

**Implements story**
- `Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow`

**Change type**
- tests | validation automation

**Definition of done**
- The full mandatory Task 5 Story 2 workflow gate runs successfully end to end.
- Every mandatory 4 / 6 / 8 / 10 qubit workflow case satisfies the frozen
  workflow exactness threshold, supported-completion rules, and local-baseline
  completeness checks.
- Story 2 completion is backed by stable outputs and rerunnable commands rather
  than by code changes alone.

**Execution checklist**
- [ ] Run the focused Story 2 regression tests for the Task 5 workflow-baseline
      layer.
- [ ] Run the dedicated Story 2 validation command or artifact-emission path for
      the mandatory workflow baseline.
- [ ] Verify `100%` pass rate on the full mandatory workflow matrix and confirm
      no mandatory workflow case is missing from the emitted output.
- [ ] Record the stable test run and artifact references for later Task 5
      stories and publication work.

**Evidence produced**
- Passing focused pytest coverage for Task 5 Story 2.
- A machine-readable workflow-baseline artifact or rerunnable command reference
  with a `100%` pass rate on the mandatory workflow matrix.

**Risks / rollback**
- Risk: Story 2 can look complete while still lacking a reproducible proof that
  the workflow exact-regime gate is both full and auditable.
- Rollback/mitigation: treat the emitted workflow-baseline output and the
  complete pass rate as part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 2 is complete only when all of the following are true:

- the mandatory 4 / 6 / 8 / 10 qubit workflow inventory is frozen through one
  stable set of case identifiers and parameter-set identifiers,
- each mandatory workflow size has at least 10 fixed parameter vectors recorded,
- every mandatory workflow case satisfies maximum absolute energy error
  `<= 1e-8`,
- the workflow baseline enforces explicit completeness and status checks so
  missing or malformed mandatory cases cannot close Story 2,
- supported completion, backend attribution, and unsupported-free execution
  remain auditable in machine-readable output,
- one stable validation command and one stable Task 5 Story 2 artifact or
  manifest define the workflow baseline with a `100%` pass rate on the mandatory
  matrix,
- and the dedicated trace plus 10-qubit anchor package, broader
  metric-completeness closure, optional or incomplete evidence interpretation,
  and final publication packaging remain clearly assigned to later Task 5
  stories.

## Implementation Notes

- `benchmarks/density_matrix/story2_vqe_density_validation.py` already provides
  the canonical workflow harness, threshold metadata, and machine-readable
  workflow output shape. Story 2 should build on that surface rather than
  duplicate it.
- `benchmarks/density_matrix/task4_story5_required_local_noise_workflow_validation.py`
  already adds the required-case accounting and trace-related summary fields most
  relevant to Task 5 Story 2. Story 2 should reuse that wrapper or its bundle
  semantics directly where possible.
- The current workflow bundle fields
  `required_cases`, `required_passed_cases`, `required_pass_rate`,
  `mandatory_baseline_completed`, `unsupported_status_cases`, and
  `documented_10q_anchor_present` are the natural anchors for Task 5 Story 2
  exit logic and should not be renamed casually.
- `tests/VQE/test_VQE.py` already contains workflow-bundle assertions and is the
  most natural foothold for fast Story 2 regression coverage.
- Task 5 Story 2 is a phase-level workflow gate, not a second workflow harness.
  Its main job is to freeze completeness, interpretation, and auditability on
  top of the canonical lower-level workflow evidence.
