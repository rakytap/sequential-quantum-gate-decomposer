# Story 4 Implementation Plan

## Story Being Implemented

Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The
Mandatory Exact-Regime Sizes

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the supported bridge slice into a mandatory workflow-scale gate
across the documented exact-regime sizes:

- the anchor XXZ noisy VQE workflow crosses the documented bridge at 4, 6, 8,
  and 10 qubits,
- workflow-scale bridge output is recorded with explicit generated-`HEA` source
  and ordered local-noise provenance,
- mandatory workflow cases complete without unsupported-operation workarounds or
  manual circuit rewriting,
- and at least one reproducible 4- or 6-qubit optimization trace crosses the
  full backend-selection, bridge, noise, and observable path.

Out of scope for this story:

- the local 1 to 3 qubit bridge micro-validation matrix owned by Story 2,
- unsupported-case hard-error work owned by Story 3,
- final publication-ready provenance packaging owned by Story 5,
- numeric exactness thresholds and Aer agreement rules already owned by Task 2,
- and broad workload expansion beyond the frozen XXZ plus generated-`HEA`
  anchor workflow.

## Dependencies And Assumptions

- Stories 1 to 3 are already in place: the supported bridge path exists, the
  local bridge surface is validated, and unsupported cases are explicit.
- The existing workflow-level validation surface already exists in
  `benchmarks/density_matrix/story2_vqe_density_validation.py` and currently
  covers part of the eventual workflow package.
- Story 1 already added canonical supported-bridge fields to the fixed-parameter
  artifacts: `bridge_source_type`, `bridge_parameter_count`,
  `bridge_operation_count`, `bridge_gate_count`, `bridge_noise_count`, and
  `bridge_operations`.
- Story 2 now provides a dedicated local bridge-validation surface in
  `benchmarks/density_matrix/task3_story2_bridge_validation.py` with a stable
  `task3_story2_bridge_micro_validation_bundle.json` bundle and per-case bridge
  gate fields such as `source_pass`, `gate_pass`, `noise_pass`,
  `operation_match_pass`, and `execution_ready`.
- `validate_density_anchor_support()` already enforces several workflow-level
  bridge assumptions, including generated `HEA` source requirements and current
  optimizer boundaries for density traces.
- Current implementation-backed constraint from Story 1: any workflow helper that
  depends on `get_Qiskit_Circuit()` should first freeze a deterministic
  optimized parameter vector on the VQE instance.
- Current implementation-backed clarification from Story 2: generated `HEA`
  support now includes the 1-qubit U3-only bridge slice, but Story 4 remains
  anchored strictly on the 4/6/8/10 qubit workflow regime.
- The frozen workload-anchor decisions remain `P2-ADR-013`,
  `DETAILED_PLANNING_PHASE_2.md`, and `TASK_3_MINI_SPEC.md`.
- Story 4 should close workflow-scale bridge usability without broadening the
  supported source, gate, or noise surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 4, 6, 8, And 10 Qubit Bridge Workflow Matrix

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The mandatory workflow-scale bridge matrix explicitly names the required 4, 6,
  8, and 10 qubit anchor cases.
- Each workflow size has a stable case identity and bridge-focused evidence
  requirements.
- The matrix distinguishes mandatory cases from optional exploratory cases so
  Story 4 pass/fail status is unambiguous.

**Execution checklist**
- [ ] Define the required 4/6/8/10 workflow inventory for Task 3 bridge
      completion.
- [ ] Record stable case IDs and parameter-set IDs for later artifact and paper
      references.
- [ ] Make the generated-`HEA` source, XXZ Hamiltonian family, and required
      local-noise schedule explicit for each mandatory case.
- [ ] Keep optional exploratory workflow cases outside the mandatory Story 4
      gate.

**Evidence produced**
- A named mandatory workflow-scale bridge matrix for the documented exact regime.
- Stable case identifiers for workflow artifacts and later docs.

**Risks / rollback**
- Risk: an underspecified workflow matrix can leave gaps in the bridge-usability
  claim while still producing some favorable cases.
- Rollback/mitigation: make the mandatory size and case coverage explicit before
  implementation closes the story.

### Engineering Task 2: Extend The Workflow Harness To Cover The Full Mandatory Bridge Matrix

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | benchmark harness

**Definition of done**
- The existing workflow harness can execute the bridge-focused anchor path at 4,
  6, 8, and 10 qubits.
- The same supported source, backend, and local-noise contract is used across
  all mandatory workflow sizes.
- One documented 10-qubit bridge case exists as part of the mandatory exact
  operating regime.

**Execution checklist**
- [ ] Extend the current workflow harness to generate supported 8- and 10-qubit
      bridge cases in the same style as the existing 4- and 6-qubit cases.
- [ ] Keep the workflow anchored on the default generated `HEA` source, XXZ
      Hamiltonian family, and required local-noise schedule.
- [ ] Freeze deterministic parameter vectors before any Qiskit-export-based
      workflow comparison or bridge summary step.
- [ ] Reuse the same backend-explicit summary conventions already established by
      the earlier workflow validation surface.
- [ ] Ensure the 10-qubit bridge case is recorded explicitly rather than treated
      as an optional stretch run.

**Evidence produced**
- Workflow harness support for the mandatory 8- and 10-qubit bridge cases.
- One documented 10-qubit bridge evaluation case.

**Risks / rollback**
- Risk: ad hoc expansion to larger sizes can drift from the supported bridge
  contract or silently weaken reproducibility.
- Rollback/mitigation: keep 8/10-qubit expansion inside the same supported
  workflow recipe already used for 4/6-qubit anchor cases.

### Engineering Task 3: Record Workflow-Scale Bridge Summaries And Ordered Operation Metadata

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | validation automation

**Definition of done**
- Every mandatory workflow case records bridge-specific metadata, not only final
  scalar outputs.
- Workflow artifacts show enough ordered operation or summarized bridge metadata
  to prove the workflow crossed the documented bridge path.
- The recorded bridge metadata is stable enough for later provenance packaging.

**Execution checklist**
- [ ] Reuse the Story 1 inspection vocabulary for workflow-scale cases where
      practical.
- [ ] Treat the Story 1 fields `bridge_source_type`,
      `bridge_parameter_count`, `bridge_operation_count`, `bridge_gate_count`,
      `bridge_noise_count`, and `bridge_operations` as the canonical starting
      schema for supported workflow artifacts.
- [ ] Reuse Story 2's bridge-pass vocabulary (`source_pass`, `gate_pass`,
      `noise_pass`, `execution_ready`) where that improves consistency between
      the local bridge gate and the workflow-scale bridge gate.
- [ ] Record source type, ansatz, operation counts, and ordered bridge metadata
      or summaries for mandatory workflow cases in that same vocabulary.
- [ ] Keep the workflow bridge summary compact enough for large cases while still
      auditable.
- [ ] Avoid making Story 4 dependent on a full raw-operation dump for every
      large case if a stable summary is sufficient.

**Evidence produced**
- Workflow-scale bridge metadata for mandatory 4/6/8/10 cases.
- Stable bridge-summary output reusable by Story 5.

**Risks / rollback**
- Risk: workflow cases may appear bridge-complete based only on execution while
  lacking enough metadata to prove how the bridge actually behaved.
- Rollback/mitigation: require bridge-summary metadata as part of the mandatory
  workflow result schema.

### Engineering Task 4: Detect Workflow Completion And Ban Unsupported-Operation Workarounds

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | tests | validation automation

**Definition of done**
- Mandatory workflow cases are marked complete only when they run through the
  supported bridge without unsupported-operation workarounds.
- Validation output distinguishes supported-complete cases from unsupported,
  degraded, or manually rewritten cases.
- Workflow-scale results cannot silently mix supported and unsupported bridge
  behavior.

**Execution checklist**
- [ ] Record workflow completion status explicitly for each mandatory Story 4
      case.
- [ ] Ensure unsupported outcomes from Story 3 remain visibly unsupported and
      are excluded from supported workflow completion rates.
- [ ] Add focused checks that confirm the larger workflow cases do not rely on
      undocumented bridge exceptions.
- [ ] Keep no-manual-rewrite and no-hidden-reroute behavior explicit in
      workflow summaries and artifacts.

**Evidence produced**
- Workflow-completion status for each mandatory bridge case.
- Validation outputs that separate supported-complete cases from unsupported or
  degraded outcomes.

**Risks / rollback**
- Risk: unsupported-operation workarounds can contaminate workflow evidence while
  still producing plausible outputs.
- Rollback/mitigation: treat supported completion and unsupported-free execution
  as explicit gates in the result schema.

### Engineering Task 5: Capture One Reproducible Optimization Trace That Crosses The Bridge

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- code | validation automation

**Definition of done**
- At least one supported 4- or 6-qubit optimization trace can be rerun
  deterministically enough to prove the full workflow crosses the bridge.
- The trace is clearly tied to the bridged density path rather than to the
  legacy state-vector path.
- Story 4 keeps the optimization-trace requirement narrow and does not expand
  into broad optimizer-science claims.

**Execution checklist**
- [ ] Choose and freeze one supported 4- or 6-qubit optimization-trace recipe
      for Story 4.
- [ ] Record enough bridge metadata during the trace to prove the workflow used
      the documented bridge path.
- [ ] Reuse the same supported bridge field names from Story 1 so the trace
      artifact can join the later Story 5 bundle without a schema translation.
- [ ] Stay inside the currently supported density-backend optimizer boundary
      already enforced by the VQE validator.
- [ ] Keep broader optimizer comparisons and trainability claims out of the
      Story 4 minimum.

**Evidence produced**
- One reproducible optimization-trace artifact proving workflow-level bridge use.
- Stable rerun instructions for the chosen trace.

**Risks / rollback**
- Risk: an under-specified trace weakens the claim that the bridge supports a
  real end-to-end workflow.
- Rollback/mitigation: keep one bounded, reproducible trace as the mandatory
  Story 4 trace rather than chasing optimizer breadth.

### Engineering Task 6: Emit A Stable Workflow-Scale Bridge Artifact Bundle

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 4 produces a stable machine-readable artifact bundle for the mandatory
  workflow-scale bridge matrix.
- Each case includes source identity, backend, case identity, bridge metadata,
  completion status, and trace linkage where relevant.
- The artifact shape is stable enough that Story 5 can assemble it directly into
  the final bridge-provenance bundle.

**Execution checklist**
- [ ] Define the per-case and per-bundle fields required for mandatory Story 4
      bridge outputs.
- [ ] Keep bundle structure compatible with the current workflow-level artifact
      surface while extending it to the mandatory 4/6/8/10 bridge regime.
- [ ] Preserve compatibility with the Story 2 bridge bundle fields so the local
      bridge gate and workflow-scale bridge gate can be assembled together later
      without a bespoke schema translation layer.
- [ ] Distinguish completed, failed, degraded, and unsupported results clearly.
- [ ] Use stable file naming and case naming so later docs and papers can cite
      the workflow bundle directly.

**Evidence produced**
- A stable machine-readable Story 4 workflow-scale bridge artifact bundle.
- Reproducible commands that regenerate the mandatory workflow outputs.

**Risks / rollback**
- Risk: ad hoc output growth can create incompatible artifact formats across
  workflow sizes and make Story 5 bundle assembly brittle.
- Rollback/mitigation: define the workflow bundle schema first and make all
  mandatory cases conform to it.

### Engineering Task 7: Add Focused Regression Checks And Developer-Facing Workflow Notes

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- tests | docs | validation automation

**Definition of done**
- The workflow-scale bridge path has focused regression coverage where practical.
- Developer-facing notes explain what Story 4 validates, how to rerun it, and
  what counts as supported workflow completion.
- The documentation makes clear that Story 4 closes workflow-scale bridge
  usability, not the final publication-ready bundle.

**Execution checklist**
- [ ] Add focused regression checks for representative workflow-scale bridge
      completion behavior where they can run practically.
- [ ] Document the mandatory 4/6/8/10 bridge matrix, supported optimization
      trace, and completion rules.
- [ ] Explain how Story 4 depends on Stories 1 to 3 and hands off to Story 5.
- [ ] Keep broader workload expansion, numeric-threshold closure, and final
      publication packaging clearly out of Story 4 scope.

**Evidence produced**
- Focused regression coverage and developer-facing guidance for the workflow
  bridge gate.
- Stable rerun instructions for the Story 4 bridge artifact bundle.

**Risks / rollback**
- Risk: without clear notes, contributors may confuse Story 4 workflow bridge
  closure with full Task 3 provenance closure.
- Rollback/mitigation: document both the delivered gate and the remaining Story
  5 bundle work explicitly.

### Engineering Task 8: Run Story 4 Validation And Confirm The Workflow-Scale Bridge Gate

**Implements story**
- `Story 4: The Bridge Supports The Anchor XXZ Noisy VQE Workflow Across The Mandatory Exact-Regime Sizes`

**Change type**
- tests | validation automation

**Definition of done**
- The mandatory workflow-scale bridge matrix runs successfully end to end.
- Every required 4, 6, 8, and 10 qubit case completes through the supported
  bridge path.
- Story 4 completion is backed by reviewable workflow artifacts rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the dedicated Story 4 workflow bridge validation command or harness.
- [ ] Verify `100%` pass rate on the mandatory workflow bridge set.
- [ ] Confirm at least one documented 10-qubit bridge case and one reproducible
      4- or 6-qubit optimization trace exist in the generated evidence.
- [ ] Record the stable artifact and test references for later Task 3 docs and
      Story 5 bundle assembly.

**Evidence produced**
- A machine-readable Story 4 bridge artifact bundle with a `100%` pass rate on
  mandatory cases.
- Reviewable validation references for 4/6/8/10 workflow-scale bridge
  completion.

**Risks / rollback**
- Risk: Story 4 can appear complete while still lacking reproducible proof of
  workflow-scale bridge use across the mandatory regime.
- Rollback/mitigation: treat the workflow bundle, the full pass rate, the
  documented 10-qubit case, and the supported optimization trace as part of the
  exit gate.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- the mandatory 4, 6, 8, and 10 qubit anchor workflow bridge matrix is defined
  and executed,
- all mandatory workflow cases complete through the documented bridge without
  unsupported-operation workarounds or manual circuit rewriting,
- workflow artifacts record explicit generated-`HEA` source and bridge metadata
  for the mandatory cases,
- one reproducible 4- or 6-qubit optimization trace proves the full workflow
  crosses the bridge,
- and the results are available through one stable workflow-scale bridge bundle
  that Story 5 can assemble directly into the final provenance package.

## Implementation Notes

- `benchmarks/density_matrix/story2_vqe_density_validation.py` already provides
  the natural backbone for Story 4 and should be extended rather than replaced.
- The current `run_fixed_parameter_case()` / `build_story1_bridge_metadata()`
  surface already defines the canonical supported-bridge fields. Story 4 should
  extend that schema rather than create a second workflow-only bridge format.
- `benchmarks/density_matrix/task3_story2_bridge_validation.py` is now the
  authoritative local bridge gate. Story 4 should align its pass/fail vocabulary
  and artifact shape with that bundle where practical, while still keeping the
  workflow-scale outputs distinct.
- Story 4 should reuse the Story 1 inspection vocabulary and the Story 3
  unsupported status vocabulary where helpful, but keep the workflow bundle
  clearly distinct from the Story 2 micro-validation bundle.
- The current VQE-side validator already enforces several workflow-level
  constraints, including generated `HEA` source requirements and supported
  optimizer choices for density traces; Story 4 should build on that substrate
  rather than duplicate it.
- Story 2 also made the 1-qubit generated-`HEA` U3-only slice explicitly
  supported. That should remain a local validation fact only and should not
  widen the workflow anchor beyond the frozen 4/6/8/10 regime.
- Story 1 also surfaced a practical harness rule: freeze deterministic optimized
  parameters before any workflow helper depends on `get_Qiskit_Circuit()` for
  comparison or artifact generation.
- Story 5 remains responsible for turning the collected workflow and trace
  outputs into the final publication-ready bridge bundle.
