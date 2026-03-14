# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Supported Validation Cases Record Internal Consistency And Execution
Stability Alongside External Agreement

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the current exactness outputs into a phase-level metric
completeness gate so strong external agreement cannot hide invalid density
states, unstable execution, or missing required metadata:

- the phase-level validation baseline records internal consistency metrics for
  supported mandatory micro-validation and workflow cases,
- workflow-scale cases also record completion, runtime, peak memory, and a
  machine-readable notion of stable end-to-end execution,
- missing required metrics or missing status semantics are treated as incomplete
  evidence rather than silent success,
- and the resulting metrics gate stays narrow enough that later stories can add
  interpretation rules for optional or incomplete evidence and the final
  publication bundle without redefining the metric sources themselves.

Out of scope for this story:

- the mandatory 1 to 3 qubit local correctness inventory owned by Story 1,
- the full 4 / 6 / 8 / 10 workflow-matrix pass/fail baseline owned by Story 2,
- the dedicated trace-and-anchor package owned by Story 3,
- optional, unsupported, and incomplete evidence interpretation closure owned by
  Story 5,
- the final Task 5 provenance and publication bundle owned by Story 6,
- runtime speed or acceleration thresholds as completion criteria,
- and introducing a new metric harness that diverges from the canonical micro and
  workflow validation paths.

## Dependencies And Assumptions

- Stories 1 to 3 are already in place: the local correctness gate, workflow
  matrix, and trace-plus-anchor package already identify the supported mandatory
  evidence surfaces that must carry the required metrics.
- Task 5 Story 3 now provides a phase-level trace-and-anchor bundle while also
  preserving the canonical raw trace artifact; Story 4 should reuse that trace
  evidence surface when extending metric completeness to trace-relevant outputs
  rather than inventing another trace wrapper.
- The canonical micro-validation exactness harness already records the core
  internal-consistency metrics in
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py`, including
  `density_valid_pass`, `trace_pass`, `observable_pass`, and case-level status.
- The canonical workflow harness already records workflow-side execution and
  performance metrics in
  `benchmarks/density_matrix/story2_vqe_density_validation.py`, including
  `workflow_completed`, runtime fields, and process peak-memory capture such as
  `total_case_runtime_ms` and `process_peak_rss_kb`.
- Task 4 Story 2 and Story 5 already preserve required-noise and workflow-scale
  metric semantics in their specialized bundles and should remain compatible
  with Task 5 Story 4 rather than be translated into a new vocabulary.
- The frozen Task 5 metric contract is already defined by the planning and ADR
  set:
  - `rho.is_valid(tol=1e-10)`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `|Im Tr(H*rho)| <= 1e-10`,
  - workflow-completion status,
  - runtime,
  - peak memory,
  - and stability of end-to-end execution for supported mandatory workflow
    evidence.
- The current Phase 2 planning language requires explicit status checks or
  manifest-equivalent fields for mandatory metrics; Story 4 must carry that
  requirement into the metric-completeness gate.
- Story 4 should close the metric-completeness gate for supported mandatory
  evidence; it should not reopen the benchmark inventory, numeric thresholds, or
  support-surface decisions already frozen at the phase level.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 5 Metric Inventory And Applicability Matrix

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- docs | validation automation

**Definition of done**
- Task 5 Story 4 names one canonical required metric inventory for supported
  mandatory evidence.
- The inventory distinguishes which metrics apply to micro-validation cases and
  which additionally apply to workflow-scale and trace cases.
- Optional or exploratory metrics remain outside the mandatory Story 4 closure
  rule.

**Execution checklist**
- [ ] Freeze the required internal-consistency metrics for supported mandatory
      micro and workflow cases.
- [ ] Freeze the required workflow-side execution metrics:
      workflow-completion status, runtime, peak memory, and explicit
      end-to-end execution stability.
- [ ] Record which metrics are mandatory per artifact class so Story 4 does not
      rely on implicit applicability.
- [ ] Keep optional diagnostics outside the mandatory Story 4 definition of done.

**Evidence produced**
- One named mandatory Task 5 metric inventory and applicability map.
- Stable metric names reusable across summaries, tests, and later Task 5
  artifacts.

**Risks / rollback**
- Risk: if Task 5 Story 4 relies on an implicit metric list, contributors may
  omit required fields while still claiming metric-complete evidence.
- Rollback/mitigation: freeze the mandatory metric inventory in one canonical
  place and reuse the same names everywhere.

### Engineering Task 2: Reuse The Canonical Micro And Workflow Metric Sources Without Forking Them

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- code | validation automation

**Definition of done**
- Task 5 Story 4 derives its required metrics from the canonical micro-validation
  and workflow validation paths rather than from a separate metric harness.
- The phase-level metric view uses the same frozen metric names and pass/fail
  semantics already accepted lower in the stack wherever practical.
- Story 4 remains a phase-level completeness layer, not a replacement for the
  underlying measurement code.

**Execution checklist**
- [ ] Reuse the micro-validation metric fields already emitted by
      `validate_squander_vs_qiskit.py`.
- [ ] Reuse the workflow-completion and performance fields already emitted by
      `story2_vqe_density_validation.py` and related required-noise workflow
      wrappers.
- [ ] Add the smallest derived summary fields needed for Task 5 Story 4 only
      where current bundle semantics are insufficient.
- [ ] Keep the Task 5 Story 4 layer thin enough that lower-level metric behavior
      still traces directly to the canonical validation code.

**Evidence produced**
- One Task 5 Story 4 assembly path rooted in the canonical micro and workflow
  metric sources.
- Reviewable traceability from the phase-level metric gate to the existing
  case-level validation surfaces.

**Risks / rollback**
- Risk: a parallel Task 5 metric harness would create schema drift and make
  metric-completeness claims harder to audit.
- Rollback/mitigation: treat the existing micro and workflow metric sources as
  canonical and extend them only through thin Task 5-specific assembly logic.

### Engineering Task 3: Add Explicit Missing-Metric And Incomplete-Evidence Status Rules

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- code | tests | validation automation

**Definition of done**
- Task 5 Story 4 can distinguish `complete pass`, `failed`, and `incomplete`
  metric states from machine-readable evidence.
- Missing required metric fields, missing status fields, or malformed metric
  payloads block Story 4 closure rather than disappearing behind aggregate
  pass-rate summaries.
- The metric-completeness gate applies only to supported mandatory cases and
  treats missing required metrics as evidence failures.

**Execution checklist**
- [ ] Add or tighten one completeness helper that checks required metric presence
      for supported mandatory evidence items.
- [ ] Require explicit per-case or per-artifact status fields for metric-bearing
      outputs.
- [ ] Make missing required metrics fail the Task 5 Story 4 gate explicitly even
      when external exactness agreement is present.
- [ ] Keep unsupported, optional, or malformed metric-bearing outputs from being
      summarized as metric-complete success.

**Evidence produced**
- Machine-readable completeness semantics for the Task 5 metric-completeness
  gate.
- Focused failure signals for missing required metric fields or missing status
  semantics.

**Risks / rollback**
- Risk: Story 4 can look green while mandatory supported cases silently omit the
  required validity or execution metrics.
- Rollback/mitigation: validate exact required metric presence and status fields,
  not just aggregate case counts.

### Engineering Task 4: Preserve Internal-Consistency And Execution-Stability Metrics In Phase-Level Outputs

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- code | validation automation

**Definition of done**
- The Task 5 Story 4 output preserves internal-consistency metrics for supported
  mandatory cases and workflow-side execution-stability metrics where they
  apply.
- Reviewers can see from the phase-level output that mandatory evidence is
  physically valid, operationally stable, and not only externally accurate.
- The phase-level metric summary keeps workflow completion, runtime, peak memory,
  and end-to-end execution stability explicit.

**Execution checklist**
- [ ] Reuse the canonical micro metric fields
      `density_valid_pass`, `trace_pass`, and `observable_pass` or their bundle
      equivalents.
- [ ] Reuse the canonical workflow-side fields
      `workflow_completed`, `total_case_runtime_ms`, and
      `process_peak_rss_kb` or the nearest stable equivalents already present in
      the workflow bundle.
- [ ] Add the smallest explicit summary field needed to make `stability of
      end-to-end execution` reviewable at the Task 5 level if current workflow
      bundle semantics are too indirect.
- [ ] Avoid flattening away metric provenance when assembling the phase-level
      Story 4 view.

**Evidence produced**
- A Task 5 Story 4 metric summary that preserves internal-consistency and
  execution-stability evidence.
- Reviewable proof that supported mandatory evidence is metric-complete, not only
  externally accurate.

**Risks / rollback**
- Risk: a thin phase-level summary can accidentally preserve only external
  exactness and drop the physical-validity or execution-stability context.
- Rollback/mitigation: treat metric visibility as part of the Story 4 schema, not
  optional debugging detail.

### Engineering Task 5: Add Focused Regression Tests For Task 5 Story 4 Metric Completeness

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the positive Task 5 Story 4 assembly path and
  representative failure conditions for missing required metrics.
- Regression coverage is specific enough to localize whether a Story 4 failure is
  caused by missing micro metrics, missing workflow metrics, or broken phase-
  level summary logic.
- The fast test layer stays smaller than the full validation commands while still
  protecting the phase-level metric gate.

**Execution checklist**
- [ ] Extend `tests/density_matrix/test_density_matrix.py`,
      `tests/VQE/test_VQE.py`, or a tightly related successor with Task 5 Story 4
      bundle or completeness tests.
- [ ] Add focused assertions for required internal-consistency and workflow-side
      metric fields on the Task 5 Story 4 output.
- [ ] Add at least one representative negative test showing that missing required
      metrics block Story 4 closure.
- [ ] Keep full matrix and workflow execution in dedicated validation commands
      rather than duplicating them inside pytest.

**Evidence produced**
- Focused pytest coverage for Task 5 Story 4 metric completeness.
- Reviewable failures that localize phase-level metric regressions.

**Risks / rollback**
- Risk: without focused completeness tests, Story 4 can regress at the summary
  layer while lower-level exactness harnesses still look healthy.
- Rollback/mitigation: add small schema- and completeness-level regression tests
  on top of the canonical micro and workflow tests.

### Engineering Task 6: Emit One Stable Task 5 Story 4 Metric-Completeness Artifact Or Rerunnable Command

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 4 can emit one stable machine-readable metric-completeness artifact or
  one stable rerunnable command that defines the mandatory Task 5 metric gate.
- The output records required metric presence, status fields, aggregate
  completeness interpretation, and references back to the supporting canonical
  micro and workflow artifacts.
- The artifact shape is stable enough that later Task 5 stories can extend it
  without replacing the Story 4 evidence format.

**Execution checklist**
- [ ] Build the Task 5 Story 4 output as a thin wrapper, manifest, or summary
      around the canonical micro and workflow bundles rather than a duplicate
      case dump with divergent semantics.
- [ ] Record suite identity, required metric names, aggregate pass/fail status,
      and references to the metric-bearing supporting artifacts.
- [ ] Keep the output narrow to metric completeness rather than mixing in
      optional-evidence interpretation or publication-only fields.
- [ ] Make the artifact or command stable enough for later Task 5 stories and
      paper-facing bundle assembly to reference directly.

**Evidence produced**
- One stable Task 5 Story 4 metric-completeness artifact or rerunnable command.
- A reusable output schema for later Task 5 metric-completeness assembly.

**Risks / rollback**
- Risk: ad hoc metric summaries will drift and make later interpretation and
  publication layers harder to audit.
- Rollback/mitigation: define one thin structured output now and extend it
  incrementally.

### Engineering Task 7: Document The Task 5 Metric Gate And Its Hand-Offs

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Task 5 Story 4 validates, how to rerun it,
  and why it is the canonical metric-completeness gate for supported mandatory
  evidence.
- The notes make clear that Story 4 sits above the correctness, workflow, and
  trace gates and below the evidence-interpretation and publication stories.
- The documentation stays aligned with the frozen Phase 2 metric expectations and
  does not overclaim broader validation closure.

**Execution checklist**
- [ ] Document the Task 5 Story 4 validation entry point and its relationship to
      the canonical micro, workflow, and trace surfaces.
- [ ] Make the mandatory Story 4 metric gate explicit:
      density validity, trace preservation, Hermitian-observable consistency,
      workflow completion, runtime, peak memory, and end-to-end execution
      stability.
- [ ] Explain how Story 4 hands off optional-evidence interpretation and final
      provenance packaging to Stories 5 and 6.
- [ ] Keep runtime speed thresholds and broader performance claims clearly out of
      the Story 4 definition of done.

**Evidence produced**
- Updated developer-facing guidance for the Task 5 Story 4 metric gate.
- One stable place where Story 4 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 4 is poorly documented, later contributors may confuse required
  metric completeness with optional performance analysis.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  outputs used for Story 4 completion.

### Engineering Task 8: Run Story 4 Validation And Confirm Metric Completeness

**Implements story**
- `Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement`

**Change type**
- tests | validation automation

**Definition of done**
- The full mandatory Task 5 Story 4 metric gate runs successfully end to end.
- Supported mandatory evidence retains all required internal-consistency and
  execution-stability metrics, and the completeness checks pass.
- Story 4 completion is backed by stable outputs and rerunnable commands rather
  than by code changes alone.

**Execution checklist**
- [ ] Run the focused Story 4 regression tests for the Task 5 metric layer.
- [ ] Run the dedicated Story 4 validation command or artifact-emission path for
      the mandatory metric-completeness gate.
- [ ] Verify that no supported mandatory evidence item is missing a required
      internal-consistency or workflow-side metric.
- [ ] Record the stable test run and artifact references for later Task 5
      stories and publication work.

**Evidence produced**
- Passing focused pytest coverage for Task 5 Story 4.
- A machine-readable metric-completeness artifact or rerunnable command proving
  required metrics are present on supported mandatory evidence.

**Risks / rollback**
- Risk: Story 4 can look complete while still lacking a reproducible proof that
  required metrics were present and reviewable.
- Rollback/mitigation: treat the emitted metric-completeness output and the
  associated completeness checks as part of the exit criteria, not optional
  follow-up.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- the mandatory Task 5 metric inventory is frozen with explicit applicability
  across supported micro, workflow, and trace evidence where relevant,
- supported mandatory micro evidence preserves internal-consistency metrics,
- supported mandatory workflow evidence preserves workflow completion, runtime,
  peak memory, and explicit end-to-end execution stability semantics,
- the metric gate enforces explicit completeness and status checks so missing
  required metrics cannot close Story 4,
- one stable validation command and one stable Task 5 Story 4 artifact or
  manifest define the metric-completeness gate,
- and optional-evidence interpretation plus final publication packaging remain
  clearly assigned to later Task 5 stories.

## Implementation Notes

- `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already provides
  the canonical internal-consistency metrics for mandatory micro-validation
  cases. Story 4 should build on those fields rather than duplicate them.
- `benchmarks/density_matrix/story2_vqe_density_validation.py` already provides
  workflow-completion, runtime, and peak-memory capture for workflow-scale cases.
  Story 4 should reuse those fields and add only the smallest derived
  Task 5-level stability summary needed by the planning contract.
- Task 5 Story 3 now packages the supported canonical trace as a distinct
  phase-level evidence layer. Story 4 should preserve that separation and extend
  metric completeness onto the delivered trace surface rather than fold the raw
  trace back into Story 2 semantics.
- Task 4 Story 2 and Story 5 wrappers already preserve required-noise and
  workflow semantics in the specialized bundles. Story 4 should preserve
  compatibility with those outputs rather than rename their stable fields.
- `tests/density_matrix/test_density_matrix.py` and `tests/VQE/test_VQE.py`
  already provide footholds for metric-field assertions and are the most natural
  starting points for fast Story 4 regression coverage.
- Task 5 Story 4 is a phase-level metric-completeness gate, not a second metric
  harness. Its main job is to freeze completeness, visibility, and auditability
  on top of the canonical lower-level metric sources.
