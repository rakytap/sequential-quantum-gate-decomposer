# Story 3 Implementation Plan

## Story Being Implemented

Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And
A Documented 10-Qubit Anchor Case

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story ties the Task 5 validation baseline to one training-relevant loop and
one explicit exact-regime anchor so the phase is not judged only by fixed
parameter sweeps:

- at least one reproducible 4- or 6-qubit optimization trace demonstrates that
  the supported density-matrix path is usable inside a training-relevant loop,
- one documented 10-qubit anchor evaluation keeps the validation package tied to
  the accepted exact-regime boundary rather than only to smaller favorable
  examples,
- both evidence items are treated as mandatory named components of the Task 5
  validation baseline rather than optional enrichments,
- and the resulting trace-plus-anchor gate stays narrow enough that later
  stories can add broader metric-completeness rules, interpretation rules for
  optional or incomplete evidence, and the final publication bundle without
  redefining these two baseline artifacts.

Out of scope for this story:

- the 1 to 3 qubit local correctness gate owned by Story 1,
- the full 4 / 6 / 8 / 10 workflow-matrix pass/fail baseline owned by Story 2,
- the broader internal-consistency and execution-stability closure owned by
  Story 4,
- optional, unsupported, and incomplete evidence interpretation closure owned by
  Story 5,
- the final Task 5 top-level provenance and publication bundle owned by Story 6,
- broad optimizer-comparison science or multi-workflow studies,
- and any claim beyond the accepted exact regime frozen for Phase 2.

## Dependencies And Assumptions

- Stories 1 and 2 are already in place: the local correctness gate and the
  mandatory workflow-scale matrix already define the canonical correctness and
  exact-regime surfaces that the trace and 10-qubit anchor must belong to.
- Task 5 Story 2 now defines a trace-independent workflow baseline through the
  phase-level Task 5 workflow bundle; Story 3 should add the dedicated trace and
  anchor package on top of that workflow gate rather than fold the trace back
  into Story 2 closure semantics.
- The current optimization-trace and workflow bundle substrate already exists in
  `benchmarks/density_matrix/story2_vqe_density_validation.py`, especially
  `run_optimization_trace()` and the workflow-bundle summary fields
  `documented_10q_anchor_present`, `supported_trace_completed`, and
  `supported_trace_case_name`.
- Task 4 Story 5 already specializes those workflow surfaces for the required
  local-noise baseline in
  `benchmarks/density_matrix/task4_story5_required_local_noise_workflow_validation.py`,
  including `required_trace_case_name`, `required_trace_present`,
  `required_trace_completed`, and `required_trace_bridge_supported`. Story 3
  should reuse those trace semantics where helpful without inheriting Task 4
  Story 5's broader workflow-sufficiency closure as its own phase-level gate.
- The frozen Task 5 trace-and-anchor contract is already implied by
  `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`:
  - at least one reproducible 4- or 6-qubit optimization trace,
  - one documented 10-qubit anchor evaluation case,
  - explicit supported-path attribution for both artifacts,
  - and inclusion of both artifacts in the mandatory evidence package.
- The current Phase 2 planning language requires stable identifiers and explicit
  status checks for mandatory evidence items; Story 3 must carry that
  requirement into the trace and anchor package.
- Story 3 should close the mandatory trace-plus-anchor gate for Task 5; it
  should not reopen the workflow anchor, optimizer surface, or benchmark
  thresholds already frozen at the phase level.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 5 Trace-And-Anchor Evidence Inventory

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- validation automation | docs

**Definition of done**
- Task 5 Story 3 names one canonical mandatory optimization trace recipe and one
  canonical documented 10-qubit anchor evidence item.
- Both evidence items have stable identities suitable for manifests, tests, and
  later publication references.
- Optional extra traces or anchor variants remain outside the mandatory Story 3
  closure rule.

**Execution checklist**
- [ ] Freeze one supported 4- or 6-qubit optimization-trace recipe as the
      canonical Task 5 Story 3 trace.
- [ ] Freeze one documented 10-qubit anchor evidence item as the canonical Task 5
      Story 3 exact-regime anchor reference.
- [ ] Preserve stable case IDs and artifact identities for both evidence items.
- [ ] Keep optional extra traces or alternate anchor examples outside the
      mandatory Story 3 definition of done.

**Evidence produced**
- One named mandatory Task 5 optimization trace.
- One named mandatory Task 5 documented 10-qubit anchor evidence item.

**Risks / rollback**
- Risk: if Story 3 relies on an implicit choice of trace or anchor case, the
  phase can appear baseline-complete while reviewers cannot tell which evidence
  item closes the contract.
- Rollback/mitigation: freeze one canonical trace and one canonical anchor case
  and reuse those identities everywhere.

### Engineering Task 2: Reuse The Canonical Optimization-Trace And Workflow-Anchor Surfaces Without Forking Them

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- code | validation automation

**Definition of done**
- Task 5 Story 3 derives its trace and 10-qubit anchor evidence from the
  canonical `story2_vqe_density_validation.py` and Task 4 Story 5 workflow
  surfaces rather than from a second optimizer or anchor harness.
- The phase-level trace and anchor package uses the same supported-path
  vocabulary, workflow identities, and status semantics already accepted lower
  in the stack.
- Story 3 remains a phase-level interpretation layer, not a replacement for the
  underlying trace or workflow harness.

**Execution checklist**
- [ ] Start from `run_optimization_trace()` and the existing workflow-bundle
      summary logic instead of designing a new trace runner.
- [ ] Reuse the Task 4 Story 5 trace and anchor summary fields where practical:
      `required_trace_case_name`, `required_trace_present`,
      `required_trace_completed`, `required_trace_bridge_supported`, and
      `documented_10q_anchor_present`.
- [ ] Keep the dedicated trace and documented 10-qubit anchor package separate
      from the already-closed Task 5 Story 2 workflow-matrix pass/fail signal.
- [ ] Keep the trace-and-anchor contract explicitly tied to the supported XXZ
      plus `HEA` plus `density_matrix` plus required local-noise path.
- [ ] Keep the Task 5 Story 3 layer thin enough that lower-level trace and anchor
      behavior still traces directly to the canonical validation code.

**Evidence produced**
- One Task 5 Story 3 assembly path rooted in the canonical trace and anchor
  validation surfaces.
- Reviewable traceability from the phase-level trace-and-anchor gate to the
  existing workflow and trace code paths.

**Risks / rollback**
- Risk: a parallel Task 5 trace harness would create schema drift and make the
  training-loop usability claim harder to audit.
- Rollback/mitigation: treat the existing trace and anchor surfaces as canonical
  and extend them only through thin Task 5-specific assembly logic.

### Engineering Task 3: Add Explicit Completeness And Status Checks For The Trace-And-Anchor Pair

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- code | tests | validation automation

**Definition of done**
- Task 5 Story 3 can distinguish `complete pass`, `failed`, and `incomplete`
  trace-and-anchor states from machine-readable evidence.
- Missing trace artifacts, missing 10-qubit anchor presence, or missing status
  fields block Story 3 closure rather than disappearing inside a workflow
  summary.
- The mandatory Story 3 gate requires both artifacts, not only one of them.

**Execution checklist**
- [ ] Add or tighten one completeness helper that checks for both a valid
      supported trace and a documented 10-qubit anchor case.
- [ ] Require explicit status fields for the trace artifact and for the 10-qubit
      anchor-presence summary.
- [ ] Make missing mandatory trace or anchor evidence fail the Task 5 Story 3
      gate explicitly.
- [ ] Keep partial satisfaction of only one of the two evidence items from being
      summarized as Story 3 success.

**Evidence produced**
- Machine-readable completeness semantics for the Task 5 trace-and-anchor gate.
- Focused failure signals for missing trace or missing 10-qubit anchor evidence.

**Risks / rollback**
- Risk: Story 3 can look green while satisfying only the trace or only the
  10-qubit anchor requirement.
- Rollback/mitigation: validate the pair explicitly and treat both artifacts as
  co-required evidence items.

### Engineering Task 4: Preserve Training-Loop Metadata And Exact-Regime Anchor Attribution In Machine-Readable Output

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- code | validation automation

**Definition of done**
- The Task 5 Story 3 output preserves enough metadata to show why the trace is
  training-relevant and why the 10-qubit case qualifies as the accepted anchor.
- Reviewers can see from the phase-level output that both artifacts belong to
  the supported Task 5 path rather than to optional or standalone-only routes.
- The trace summary and anchor summary stay aligned with lower-level workflow
  status semantics.

**Execution checklist**
- [ ] Reuse the canonical trace fields already present in the trace artifact,
      including optimizer identity, parameter-count or parameter-state fields,
      energy history or bounded summary, and `workflow_completed`.
- [ ] Preserve supported-path attribution through fields such as
      `required_trace_bridge_supported` or the nearest equivalent supported-path
      status.
- [ ] Make the 10-qubit anchor presence explicit in the phase-level output
      without hiding it inside a large workflow bundle summary.
- [ ] Avoid flattening away why the trace and anchor belong to the supported
      mandatory Phase 2 path.

**Evidence produced**
- A Task 5 Story 3 output that preserves trace metadata and explicit 10-qubit
  anchor attribution.
- Reviewable proof that both evidence items belong to the supported mandatory
  baseline.

**Risks / rollback**
- Risk: a thin phase-level summary can keep only existence flags and lose the
  context needed to audit training-loop relevance or anchor attribution.
- Rollback/mitigation: preserve the small set of trace and anchor fields needed
  to explain why each artifact matters.

### Engineering Task 5: Add Focused Regression Tests For Task 5 Story 3 Completeness Rules

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the positive Task 5 Story 3 assembly path and
  representative failure conditions for incomplete trace-or-anchor evidence.
- Regression coverage is specific enough to localize whether a Story 3 failure is
  caused by a missing trace, missing 10-qubit anchor, or broken summary logic.
- The fast test layer stays smaller than the full trace-and-anchor validation
  command while still protecting the phase-level gate.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` or a tightly related successor with Task 5
      Story 3 bundle or completeness tests.
- [ ] Add focused assertions for stable trace identity, documented 10-qubit anchor
      presence, and explicit status fields on the Task 5 Story 3 output.
- [ ] Add at least one representative negative test showing that missing trace or
      missing anchor evidence blocks Story 3 closure.
- [ ] Keep full trace execution and full workflow-matrix execution in dedicated
      validation commands rather than duplicating them inside pytest.

**Evidence produced**
- Focused pytest coverage for Task 5 Story 3 trace-and-anchor completeness.
- Reviewable failures that localize phase-level trace-and-anchor regressions.

**Risks / rollback**
- Risk: without focused completeness tests, Story 3 can regress at the manifest
  level while lower-level workflow and trace runs still look healthy.
- Rollback/mitigation: add small schema- and completeness-level regression tests
  on top of the canonical trace and workflow tests.

### Engineering Task 6: Emit One Stable Task 5 Story 3 Trace-And-Anchor Artifact Or Rerunnable Command

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 3 can emit one stable machine-readable trace-and-anchor artifact or one
  stable rerunnable command that defines the mandatory Task 5 trace-plus-anchor
  package.
- The output records stable trace identity, 10-qubit anchor presence, status
  fields, supported-path attribution, and aggregate pass/fail interpretation.
- The artifact shape is stable enough that later Task 5 stories can extend it
  without replacing the Story 3 evidence format.

**Execution checklist**
- [ ] Build the Task 5 Story 3 output as a thin wrapper, manifest, or summary
      around the canonical trace and workflow-bundle outputs rather than a
      duplicate result dump with divergent semantics.
- [ ] Record suite identity, software metadata, trace identity, anchor identity
      or presence, aggregate pass/fail status, and per-artifact status
      references or payloads.
- [ ] Keep the output narrow to the dedicated trace-and-anchor package rather
      than mixing in the full workflow matrix, optional-evidence interpretation,
      or publication-only fields.
- [ ] Make the artifact or command stable enough for later Task 5 stories and
      paper-facing bundle assembly to reference directly.

**Evidence produced**
- One stable Task 5 Story 3 trace-and-anchor artifact or rerunnable command.
- A reusable output schema for later Task 5 trace-and-anchor assembly.

**Risks / rollback**
- Risk: ad hoc trace-and-anchor summaries will drift and make later metric,
  interpretation, and publication layers harder to audit.
- Rollback/mitigation: define one thin structured output now and extend it
  incrementally.

### Engineering Task 7: Document The Task 5 Trace-And-Anchor Gate And Its Hand-Offs

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Task 5 Story 3 validates, how to rerun it,
  and why it is the canonical trace-and-anchor gate for the phase.
- The notes make clear that Story 3 sits above the workflow matrix but below the
  broader metric-completeness, evidence-interpretation, and publication stories.
- The documentation stays aligned with the frozen Phase 2 trace and anchor
  expectations and does not overclaim broader validation closure.

**Execution checklist**
- [ ] Document the Task 5 Story 3 validation entry point and its relationship to
      the canonical workflow bundle and optimization-trace surfaces.
- [ ] Make the mandatory Story 3 gate explicit:
      one reproducible 4- or 6-qubit trace, one documented 10-qubit anchor
      case, explicit supported-path attribution, and stable status fields.
- [ ] Explain how Story 3 hands off broader metric completeness, evidence
      interpretation, and publication packaging to Stories 4 to 6.
- [ ] Keep optional extra traces or alternate anchor examples clearly out of the
      Story 3 definition of done.

**Evidence produced**
- Updated developer-facing guidance for the Task 5 Story 3 trace-and-anchor gate.
- One stable place where Story 3 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 3 is poorly documented, later contributors may confuse the
  mandatory trace-and-anchor package with optional workflow examples.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  outputs used for Story 3 completion.

### Engineering Task 8: Run Story 3 Validation And Confirm The Mandatory Trace-And-Anchor Pair

**Implements story**
- `Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case`

**Change type**
- tests | validation automation

**Definition of done**
- The full mandatory Task 5 Story 3 trace-and-anchor gate runs successfully end
  to end.
- The canonical trace completes through the supported path, and the documented
  10-qubit anchor evidence is present and explicit.
- Story 3 completion is backed by stable outputs and rerunnable commands rather
  than by code changes alone.

**Execution checklist**
- [ ] Run the focused Story 3 regression tests for the Task 5 trace-and-anchor
      layer.
- [ ] Run the dedicated Story 3 validation command or artifact-emission path for
      the mandatory trace-and-anchor package.
- [ ] Verify both the trace and the 10-qubit anchor are present, supported, and
      complete in the emitted output.
- [ ] Record the stable test run and artifact references for later Task 5
      stories and publication work.

**Evidence produced**
- Passing focused pytest coverage for Task 5 Story 3.
- A machine-readable trace-and-anchor artifact or rerunnable command reference
  proving both mandatory Story 3 evidence items are present and supported.

**Risks / rollback**
- Risk: Story 3 can look complete while still lacking a reproducible proof that
  the trace and anchor pair are both present and attributable to the supported
  path.
- Rollback/mitigation: treat the emitted trace-and-anchor output and paired
  completeness checks as part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one stable 4- or 6-qubit optimization-trace identity is frozen as the
  mandatory Task 5 trace,
- one documented 10-qubit anchor evidence item is frozen as the mandatory
  exact-regime anchor reference,
- explicit completeness and status checks prove that both mandatory evidence
  items are present,
- the emitted trace remains attributable to the supported density-matrix path,
- the emitted anchor evidence remains explicitly tied to the accepted 10-qubit
  exact-regime boundary,
- one stable validation command and one stable Task 5 Story 3 artifact or
  manifest define the paired trace-and-anchor gate,
- and broader metric-completeness closure, optional or incomplete evidence
  interpretation, and final publication packaging remain clearly assigned to
  later Task 5 stories.

## Implementation Notes

- `benchmarks/density_matrix/story2_vqe_density_validation.py` already provides
  the canonical optimization-trace and workflow-bundle surfaces. Story 3 should
  build on those surfaces rather than duplicate them.
- `benchmarks/density_matrix/task4_story5_required_local_noise_workflow_validation.py`
  already adds the trace- and workflow-summary fields most relevant to Story 3,
  especially `required_trace_case_name`, `required_trace_present`,
  `required_trace_completed`, `required_trace_bridge_supported`, and the 10-qubit
  anchor summary.
- Task 5 Story 2 now establishes the phase-level workflow-matrix closure without
  requiring the bounded trace. Story 3 should preserve that separation and add
  the trace-plus-anchor package as its own explicit evidence layer.
- The current workflow and trace artifacts already distinguish completion from
  unsupported or degraded behavior. Story 3 should preserve those semantics
  rather than flatten them into simple existence checks.
- `tests/VQE/test_VQE.py` already contains workflow-bundle and trace-bundle
  assertions and is the most natural foothold for fast Story 3 regression
  coverage.
- Task 5 Story 3 is a phase-level trace-and-anchor gate, not a new optimizer or
  workflow harness. Its main job is to freeze completeness, identity, and
  supported-path attribution for the two mandatory evidence items.
