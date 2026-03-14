# Story 5 Implementation Plan

## Story Being Implemented

Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow
Across The Mandatory Exact-Regime Sizes

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the already supported local-noise baseline and exactness gates
from Stories 1 to 4 into a workflow-scale acceptance package for the anchor XXZ
VQE path:

- the required local-noise baseline is exercised on the anchor workflow at 4, 6,
  8, and 10 qubits,
- each mandatory workflow size is evaluated on at least 10 fixed parameter
  vectors under the required local-noise schedule,
- exact noisy energy results are compared against Qiskit Aer at workflow scale
  with the frozen `<= 1e-8` threshold,
- workflow completion, density-validity, runtime, peak memory, and
  required-versus-optional case semantics are recorded,
- and at least one reproducible 4- or 6-qubit optimization trace plus one
  documented 10-qubit anchor evaluation demonstrate that the required local
  baseline is sufficient for the accepted exact regime.

Out of scope for this story:

- the 1 to 3 qubit exact micro-validation matrix already owned by Story 2,
- optional baseline classification already owned by Story 3 except where it must
  remain visible in workflow-scale summaries,
- unsupported and deferred hard-error behavior already owned by Story 4,
- the final publication-ready provenance bundle owned by Story 6,
- and any expansion beyond the frozen XXZ, `HEA`, `density_matrix`,
  `local_depolarizing` / `amplitude_damping` / `phase_damping` contract.

## Dependencies And Assumptions

- Story 1 is already in place: the required local-noise models execute on the
  supported VQE-facing density path and expose machine-readable bridge metadata
  through `task4_story1_required_local_noise_validation.py`.
- Story 2 is already in place: the mandatory 1 to 3 qubit exact micro-validation
  gate passes for the required local-noise surface through
  `validate_squander_vs_qiskit.py` and
  `task4_story2_required_local_noise_micro_validation.py`.
- Story 3 is already in place: optional whole-register depolarizing cases are
  classified explicitly and do not count toward mandatory completion through
  `task4_story3_optional_noise_classification_validation.py`.
- Story 4 is already in place: deferred and unsupported noise requests fail
  before execution and emit structured negative evidence rather than silently
  contaminating workflow-scale results. The current negative authority is
  `task4_story4_unsupported_noise_validation.py`, which emits
  `story4_unsupported_noise_bundle.json` and fixes the current negative
  vocabulary around `support_tier`, `unsupported_category`,
  `first_unsupported_condition`, `task4_boundary_class`, `failure_stage`, and
  `unsupported_status_cases`.
- The current workflow-level evidence surface already exists in
  `benchmarks/density_matrix/story2_vqe_density_validation.py`, including:
  `build_story4_parameter_sets()`, `run_story4_workflow_case()`,
  `build_story4_workflow_bundle()`, and `run_optimization_trace()`.
- The frozen workflow and threshold contract remains:
  - `P2-ADR-013`: 4 and 6 qubits must support full end-to-end workflow
    execution, including at least one reproducible optimization trace,
  - `P2-ADR-013`: 8 and 10 qubits must support benchmark-ready fixed-parameter
    evaluation,
  - `P2-ADR-014`: at least 10 fixed parameter vectors per mandatory workflow
    size plus a documented 10-qubit anchor case,
  - `P2-ADR-015`: `<= 1e-8` maximum absolute energy error and `100%` pass rate
    on the mandatory workflow benchmark set.
- Story 5 should close the workflow-scale sufficiency gate without reopening the
  support matrix, optional classification, or deferred-noise boundary.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 4, 6, 8, And 10 Qubit Required-Noise Workflow Matrix

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The mandatory workflow matrix explicitly names the required 4, 6, 8, and 10
  qubit anchor cases.
- Each required workflow size has at least 10 fixed parameter vectors and stable
  case identity.
- The matrix makes clear which cases are mandatory required-baseline cases and
  which optional workflow explorations remain outside Story 5 completion.

**Execution checklist**
- [ ] Freeze the mandatory workflow-size inventory for 4, 6, 8, and 10 qubits.
- [ ] Preserve at least 10 fixed parameter vectors per required workflow size in
      a reproducible way.
- [ ] Keep stable case IDs and parameter-set IDs for later artifact and doc
      references.
- [ ] Ensure the mandatory matrix is explicitly tied to the required local-noise
      schedule rather than to optional baselines.

**Evidence produced**
- A named mandatory workflow-scale matrix for the required local-noise exact
  regime.
- Stable workflow and parameter-set identifiers for later artifacts.

**Risks / rollback**
- Risk: an underspecified matrix can leave gaps in the exact-regime claim while
- still producing some favorable workflow results.
- Rollback/mitigation: make mandatory size, parameter-set, and required-noise
  coverage explicit before closing Story 5.

### Engineering Task 2: Keep The Workflow Harness Anchored To The Required Local-Noise Schedule

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | benchmark harness

**Definition of done**
- The workflow harness uses the required local-noise schedule consistently across
  mandatory workflow sizes.
- Optional whole-register depolarizing or later extension cases remain distinct
  from the mandatory workflow package.
- Workflow-scale results inherit the same required local-noise naming and bridge
  vocabulary as Stories 1 and 2.

**Execution checklist**
- [ ] Reuse or tighten the current required local-noise schedule in
      `build_story2_noise()` and the workflow harness that consumes it.
- [ ] Ensure `run_story4_workflow_case()` and related helpers remain tied to
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Keep optional whole-register depolarizing and any future justified
      extensions out of the mandatory workflow bundle.
- [ ] Preserve Task 4 support-tier semantics on any workflow-scale artifact
      fields added here.

**Evidence produced**
- One consistent required local-noise workflow recipe across 4, 6, 8, and 10
  qubits.
- Workflow artifacts aligned with the same required-noise vocabulary as earlier
  Task 4 stories.

**Risks / rollback**
- Risk: workflow helpers may silently drift toward optional or broader noise
  surfaces because they reuse general density-matrix utilities.
- Rollback/mitigation: keep the required workflow recipe explicit and reuse Task
  4 classification semantics wherever the workflow harness emits metadata.

### Engineering Task 3: Extend Or Tighten Workflow-Scale Exactness Checks At 8 And 10 Qubits

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | benchmark harness | validation automation

**Definition of done**
- Every mandatory workflow case compares exact noisy energy against Qiskit Aer.
- The workflow-scale exactness rule is enforced at `<= 1e-8` maximum absolute
  energy error.
- 8- and 10-qubit fixed-parameter cases are no longer only planning targets;
  they are emitted as reviewed required-baseline evidence.

**Execution checklist**
- [ ] Reuse or extend the current exact-energy comparison helpers for mandatory
      workflow cases at 4, 6, 8, and 10 qubits.
- [ ] Enforce the `<= 1e-8` workflow threshold and `100%` pass-rate rule on the
      mandatory workflow bundle.
- [ ] Ensure the documented 10-qubit anchor case is explicitly represented in the
      workflow bundle summary.
- [ ] Keep optional or unsupported outcomes excluded from the required workflow
      pass rate.

**Evidence produced**
- Workflow-scale exact observable comparisons against Qiskit Aer with explicit
  pass/fail status.
- A documented 10-qubit anchor case in the mandatory workflow bundle.

**Risks / rollback**
- Risk: workflow cases may appear successful based only on completion while
  violating the exactness threshold or lacking a documented 10-qubit anchor.
- Rollback/mitigation: make threshold and 10-qubit anchor presence part of the
  mandatory bundle validation logic.

### Engineering Task 4: Preserve Workflow Completion And Required-Baseline Purity In Bundle Semantics

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | tests | validation automation

**Definition of done**
- Workflow completion remains a distinct gate from numeric agreement.
- Required-baseline workflow cases are distinguishable from optional,
  unsupported, or degraded cases in machine-readable outputs.
- The workflow bundle cannot be satisfied by optional or unsupported cases.

**Execution checklist**
- [ ] Reuse `workflow_completed`, `bridge_supported_pass`, and related bundle
      fields in `build_story4_workflow_bundle()`.
- [ ] Add or tighten support-tier and counts-toward-mandatory-baseline semantics
      for workflow-scale Task 4 outputs where needed, using the Story 4 field
      names `support_tier`, `case_purpose`, and
      `counts_toward_mandatory_baseline`.
- [ ] Ensure unsupported or optional workflow results are excluded from the
      mandatory workflow completion signal.
- [ ] Keep the summary logic aligned with Story 3’s classification layer rather
      than inventing separate workflow-only semantics.
- [ ] Preserve Story 4 negative-schema compatibility for any workflow-level
      unsupported result by reusing `unsupported_category`,
      `first_unsupported_condition`, `task4_boundary_class`, and
      `failure_stage` rather than inventing workflow-only error labels.

**Evidence produced**
- Workflow-scale bundle semantics that keep required completion independent of
  optional or unsupported outcomes.
- Reviewable machine-readable completion fields for every mandatory case.

**Risks / rollback**
- Risk: optional or unsupported workflow results may accidentally inflate the
  exact-regime pass rate if bundle semantics drift.
- Rollback/mitigation: compute the mandatory workflow gate only from required,
  completed, supported cases.

### Engineering Task 5: Capture The Reproducible 4- Or 6-Qubit Optimization Trace As Part Of Story 5

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | validation automation

**Definition of done**
- At least one supported 4- or 6-qubit optimization trace is rerunnable and
  clearly tied to the required local-noise baseline.
- The trace records initial parameters, final parameters, optimizer choice,
  workflow completion, runtime, and bridge-supported status.
- The trace remains part of the mandatory workflow sufficiency evidence for Story
  5, not merely a separate publication-bundle concern.

**Execution checklist**
- [ ] Reuse or tighten `run_optimization_trace()` as the canonical bounded trace
      for Story 5.
- [ ] Ensure the trace is explicitly marked as required-baseline evidence rather
      than optional workflow enrichment.
- [ ] Record optimizer configuration, initial/final parameters, and completion
      metadata in machine-readable form.
- [ ] Keep broader optimizer-comparison science outside the Story 5 minimum.

**Evidence produced**
- One reproducible 4- or 6-qubit optimization-trace artifact tied to the
  required local-noise baseline.
- Stable rerun instructions for the chosen trace.

**Risks / rollback**
- Risk: an under-specified optimization trace weakens the “training-loop usable”
  claim even if fixed-parameter workflow cases pass.
- Rollback/mitigation: keep one bounded, reproducible trace as mandatory Story 5
  evidence rather than chasing optimizer breadth.

### Engineering Task 6: Record Runtime, Peak Memory, And Workflow Stability Alongside Exactness

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Mandatory workflow cases record runtime and peak-memory metrics in addition to
  exactness and completion.
- These metrics are reported consistently but are not treated as pass/fail
  thresholds for Story 5.
- Performance and stability metadata stays attached to the same required-baseline
  bundle rather than only to ad hoc logs.

**Execution checklist**
- [ ] Reuse the current runtime and peak-RSS capture already present in the
      workflow harness where possible.
- [ ] Ensure mandatory workflow cases serialize runtime, peak memory, and
      workflow stability fields in machine-readable outputs.
- [ ] Keep performance reporting structurally separate from the exactness
      pass/fail gate.
- [ ] Preserve bundle compatibility with the later publication bundle rather than
      inventing a separate performance-only file format.

**Evidence produced**
- Runtime and peak-memory metrics for the mandatory required-local-noise
  workflow cases.
- Machine-readable workflow stability metadata aligned with the benchmark
  minimum.

**Risks / rollback**
- Risk: performance metrics captured only in console logs are easy to lose and
  hard to reuse later.
- Rollback/mitigation: serialize them in the same stable workflow artifact
  surface as the exactness outputs.

### Engineering Task 7: Emit A Stable Task 4 Story 5 Workflow Bundle

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 5 produces a stable machine-readable workflow-scale artifact bundle for
  the required local-noise exact regime.
- The bundle includes backend, workflow identity, parameter-set identity,
  required/optional classification, exactness metrics, validity metrics,
  completion status, and performance metadata.
- The artifact shape is stable enough that Story 6 can assemble it directly into
  the final publication-facing provenance bundle.

**Execution checklist**
- [ ] Start from the current `build_story4_workflow_bundle()` output rather than
      inventing a parallel workflow bundle.
- [ ] Add Task 4 support-tier and mandatory-baseline semantics only where they
      materially improve workflow review.
- [ ] Distinguish completed, failed, optional, and unsupported outcomes clearly.
- [ ] Use stable file naming and case naming so later docs and paper artifacts
      can cite the workflow bundle directly.
- [ ] Keep the summary layer compatible with Story 4’s
      `unsupported_status_cases` / support-tier split so Story 6 can aggregate
      negative and workflow evidence without schema translation.

**Evidence produced**
- A stable Task 4 Story 5 workflow-scale artifact bundle.
- Reproducible commands that regenerate the mandatory workflow outputs.

**Risks / rollback**
- Risk: ad hoc bundle growth can create incompatible artifact formats between
  stories and weaken traceability.
- Rollback/mitigation: extend the existing workflow bundle carefully instead of
  forking it.

### Engineering Task 8: Run Story 5 Workflow Validation And Confirm The Required Baseline Closes The Exact Regime

**Implements story**
- `Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 5 regression checks pass.
- The mandatory 4/6/8/10 workflow bundle and required optimization trace are
  emitted successfully.
- Story 5 completion is backed by reviewable workflow-scale artifacts rather than
  by code changes alone.

**Execution checklist**
- [ ] Run focused Story 5 pytest or schema-level checks for workflow bundle
      semantics.
- [ ] Run the workflow-scale validation command that emits the mandatory bundle
      and trace.
- [ ] Verify `100%` pass rate on the mandatory workflow benchmark set and
      explicit documented presence of the 10-qubit anchor case.
- [ ] Record stable test and artifact references for later Task 4 docs and Story
      6 bundle assembly.

**Evidence produced**
- Passing focused Story 5 regression coverage.
- Stable workflow bundle and optimization-trace references proving the required
  baseline closes the exact regime.

**Risks / rollback**
- Risk: Story 5 can appear complete while still lacking auditable proof that the
  required local-noise workflow baseline closes the accepted exact regime.
- Rollback/mitigation: treat workflow bundle status, 10-qubit anchor presence,
  and required trace evidence as part of the exit gate, not optional follow-up.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- required local-noise workflow cases are recorded at 4, 6, 8, and 10 qubits,
- at least 10 fixed parameter vectors are present for each mandatory workflow
  size,
- every mandatory workflow case satisfies maximum absolute energy error
  `<= 1e-8`,
- the mandatory workflow bundle achieves a `100%` pass rate on required cases,
- one documented 10-qubit anchor evaluation case is present,
- one reproducible 4- or 6-qubit optimization trace is present and tied to the
  required local-noise baseline,
- and workflow-scale artifacts keep required cases distinct from optional,
  unsupported, and degraded outcomes.

## Implementation Notes

- `story2_vqe_density_validation.py` already contains the natural backbone for
  Story 5 through `build_story4_parameter_sets()`,
  `run_story4_workflow_case()`, `build_story4_workflow_bundle()`, and
  `run_optimization_trace()`. Story 5 should extend this surface rather than
  creating a separate workflow harness.
- Stories 1 to 4 already established the required-noise vocabulary, exactness
  gate, optional classification, and unsupported boundary. Story 5 should reuse
  those outputs rather than redefining them at workflow scale. In particular,
  Story 4 now fixes the negative vocabulary around `support_tier`,
  `unsupported_category`, `first_unsupported_condition`,
  `task4_boundary_class`, `failure_stage`, and `unsupported_status_cases`.
- The current workflow bundle already tracks fields such as
  `documented_10q_anchor_present`, `supported_trace_completed`, and
  `bridge_supported_pass`; these are the natural anchors for Story 5 exit logic.
- Story 6 should assemble the final publication-facing provenance surface. Story
  5 should focus on closing workflow-scale sufficiency and producing stable raw
  artifacts for Story 6 to consume.
