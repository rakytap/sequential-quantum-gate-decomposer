# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The
Mandatory Exact Regime

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_2_STORIES.md`.

## Scope

This story turns the supported exact noisy observable path into a mandatory
workflow-scale benchmark gate across the documented exact regime:

- the anchor `HEA` noisy VQE workflow is exercised at 4, 6, 8, and 10 qubits,
- each required workflow size is evaluated on at least 10 fixed parameter
  vectors,
- exact noisy energy results are compared against Qiskit Aer at workflow scale,
- workflow completion, density-validity, runtime, and peak-memory metrics are
  recorded,
- and all mandatory cases complete without unsupported-operation workarounds or
  implicit backend fallback.

Out of scope for this story:

- the 1 to 3 qubit micro-validation matrix owned by Story 2,
- unsupported-case hard-error work owned by Story 3,
- the final publication-ready reproducibility bundle owned by Story 5,
- broad simulator bake-offs beyond the required Aer baseline,
- and runtime or speedup pass/fail thresholds beyond recording required metrics.

## Dependencies And Assumptions

- Story 1 is already in place: the supported positive exact noisy VQE path
  exists for fixed-parameter anchor cases.
- Story 2 is already in place: the local 1 to 3 qubit correctness gate proves
  the required gate and local-noise surface.
- Story 3 is already in place: unsupported combinations fail explicitly rather
  than contaminating workflow-scale results.
- The current workflow-level evidence surface already exists in
  `benchmarks/density_matrix/story2_vqe_density_validation.py` and currently
  covers a subset of the final Story 4 package.
- The frozen benchmark and threshold decisions remain:
  `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`, and the benchmark-minimum /
  numeric-threshold sections of `DETAILED_PLANNING_PHASE_2.md`.
- Story 4 should close the 4 to 10 qubit workflow-scale exactness gate without
  broadening the supported observable, bridge, or noise surface.

## Engineering Tasks

### Engineering Task 1: Freeze The Mandatory 4, 6, 8, And 10 Qubit Workflow Matrix

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- The mandatory workflow-scale matrix explicitly names the required 4, 6, 8, and
  10 qubit anchor cases.
- Each required workflow size has at least 10 fixed parameter vectors and a
  stable case identity.
- The matrix makes clear which cases are mandatory, which metrics are required,
  and which 10-qubit case closes the exact-regime anchor.

**Execution checklist**
- [ ] Define the required workflow-size inventory for 4, 6, 8, and 10 qubits.
- [ ] Freeze at least 10 fixed parameter vectors per mandatory workflow size in a
      reproducible way.
- [ ] Record stable case IDs and parameter-set IDs for later artifact and paper
      references.
- [ ] Keep optional exploratory workflow cases outside the mandatory Story 4
      gate.

**Evidence produced**
- A named mandatory workflow-scale matrix for the exact 4 to 10 qubit regime.
- Stable case and parameter-set identifiers for benchmark artifacts.

**Risks / rollback**
- Risk: an underspecified workflow matrix can leave gaps in the exact-regime
  claim while still producing some favorable results.
- Rollback/mitigation: make the mandatory size and parameter-set coverage
  explicit before implementation closes the story.

### Engineering Task 2: Extend The Anchor Workflow Harness To Cover 8 And 10 Qubits

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- code | benchmark harness

**Definition of done**
- The existing anchor VQE validation harness can execute the exact noisy
  workflow-scale path at 4, 6, 8, and 10 qubits.
- The same supported backend, ansatz, Hamiltonian family, and local-noise
  contract are used across all mandatory workflow sizes.
- One documented 10-qubit anchor evaluation case exists as part of the exact
  operating-regime gate.

**Execution checklist**
- [ ] Extend the current anchor workflow harness to generate supported 8- and
      10-qubit cases in the same style as the existing 4- and 6-qubit cases.
- [ ] Keep the workflow anchored on the default `HEA` ansatz, XXZ Hamiltonian,
      and the required local-noise schedule.
- [ ] Reuse the same backend-explicit artifact and summary conventions already
      established by the earlier workflow-level validation surface.
- [ ] Ensure the 10-qubit anchor case is recorded explicitly rather than treated
      as an optional stretch run.

**Evidence produced**
- Workflow-level harness support for the mandatory 8- and 10-qubit anchor cases.
- One documented 10-qubit exact-regime anchor evaluation case.

**Risks / rollback**
- Risk: ad hoc expansion to larger sizes can drift from the supported workflow
  contract or silently weaken reproducibility.
- Rollback/mitigation: keep 8/10-qubit expansion inside the same supported
  workflow recipe already used for 4/6-qubit anchor cases.

### Engineering Task 3: Add Workflow-Scale Exact Observable Comparison And Threshold Checks

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- code | benchmark harness | validation automation

**Definition of done**
- Every mandatory workflow case compares exact noisy energy against Qiskit Aer.
- The workflow-scale exactness rule is enforced at `<= 1e-8` maximum absolute
  energy error.
- Density-validity and Hermitian-consistency checks remain recorded alongside
  energy agreement at workflow scale.

**Execution checklist**
- [ ] Reuse or extend the current exact-energy comparison helper path for the
      mandatory workflow cases.
- [ ] Enforce the `<= 1e-8` workflow-level absolute energy error threshold.
- [ ] Record `rho.is_valid(tol=1e-10)`, `|Tr(rho) - 1|`, and `|Im Tr(H*rho)|`
      on recorded workflow outputs where available.
- [ ] Make pass/fail interpretation explicit for each workflow case and for the
      mandatory matrix as a whole.

**Evidence produced**
- Workflow-scale exact observable comparisons against Aer with explicit pass/fail
  status.
- Recorded density-validity and Hermitian-consistency metrics for mandatory
  workflow cases.

**Risks / rollback**
- Risk: workflow-level cases may appear successful based only on completion while
  still violating the numeric exactness contract.
- Rollback/mitigation: make the `<= 1e-8` threshold and validity metrics part of
  the mandatory result schema, not optional annotations.

### Engineering Task 4: Detect Workflow Completion And Ban Unsupported-Operation Workarounds

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- code | tests | validation automation

**Definition of done**
- Mandatory workflow cases are marked complete only when they run through the
  supported density backend without unsupported-operation workarounds.
- Validation output distinguishes successful completion from unsupported,
  failed, or degraded cases.
- Workflow-scale results cannot silently mix supported and unsupported behavior.

**Execution checklist**
- [ ] Record workflow completion status explicitly for each mandatory case.
- [ ] Ensure unsupported-case results from Story 3 remain visibly unsupported and
      are excluded from the supported benchmark pass rate.
- [ ] Add focused checks that confirm the larger workflow cases do not rely on
      unsupported bridge, gate, or noise behavior.
- [ ] Keep no-fallback behavior explicit in workflow summaries and artifacts.

**Evidence produced**
- Workflow-completion status for each mandatory benchmark case.
- Validation outputs that separate supported-complete cases from unsupported or
  degraded outcomes.

**Risks / rollback**
- Risk: unsupported-operation workarounds can contaminate workflow-scale evidence
  while still producing numeric outputs.
- Rollback/mitigation: treat workflow completion and unsupported-free execution
  as explicit gates in the result schema.

### Engineering Task 5: Record Runtime And Peak-Memory Metrics Alongside Exactness

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Mandatory workflow cases record runtime and peak-memory metrics in addition to
  exactness and completion.
- These metrics are reported consistently but are not treated as pass/fail
  thresholds for Story 4.
- Performance metadata is aligned with the benchmark-minimum contract so later
  publication packaging can reuse it directly.

**Execution checklist**
- [ ] Extend the workflow-scale harness or companion performance tooling to
      record runtime for mandatory cases.
- [ ] Add peak-memory capture or the nearest practical workflow-level memory
      metric in the current environment.
- [ ] Keep runtime and memory reporting structurally separate from the exactness
      pass/fail gate.
- [ ] Ensure these metrics are serialized in machine-readable outputs rather than
      only printed in logs.

**Evidence produced**
- Runtime and peak-memory metrics for the mandatory workflow-scale cases.
- Benchmark outputs aligned with the frozen benchmark-minimum metric set.

**Risks / rollback**
- Risk: runtime-only logs without machine-readable capture are easy to lose and
  hard to reuse in later publication assembly.
- Rollback/mitigation: serialize performance metrics in the same stable artifact
  surface as the exactness outputs.

### Engineering Task 6: Emit A Stable Workflow-Scale Artifact Bundle

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 4 produces a stable machine-readable artifact bundle for the mandatory
  workflow-scale matrix.
- Each case includes backend, workflow identity, Hamiltonian metadata, parameter
  vector identity, exactness metrics, validity metrics, completion status, and
  performance metadata.
- The artifact shape is stable enough that Story 5 can assemble it directly into
  the publication-ready evidence bundle.

**Execution checklist**
- [ ] Define the per-case and per-bundle fields required for mandatory Story 4
      outputs.
- [ ] Keep bundle structure compatible with the current workflow-level artifact
      surface while extending it to the mandatory 4/6/8/10 regime.
- [ ] Distinguish completed, failed, and unsupported results clearly.
- [ ] Use stable file naming and case naming so later docs and papers can cite
      the workflow bundle directly.

**Evidence produced**
- A stable machine-readable Story 4 workflow-scale artifact bundle.
- Reproducible commands that regenerate the mandatory workflow outputs.

**Risks / rollback**
- Risk: ad hoc output growth can create incompatible artifact formats between the
  4/6 and 8/10 workflow cases.
- Rollback/mitigation: define the workflow bundle schema first and make all
  mandatory cases conform to it.

### Engineering Task 7: Add Focused Regression And Developer-Facing Workflow Notes

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- tests | docs | validation automation

**Definition of done**
- The workflow-scale path has focused regression coverage where it is practical.
- Developer-facing notes explain what Story 4 validates, how to rerun it, and
  which metrics are mandatory versus informative.
- The documentation makes clear that Story 4 closes the workflow-scale exactness
  gate but not yet the final publication-ready bundle.

**Execution checklist**
- [ ] Add focused regression checks for representative workflow-scale exactness
      and completion behavior where they can run practically.
- [ ] Document the mandatory 4/6/8/10 workflow matrix, the `<= 1e-8` threshold,
      and the supporting runtime/memory outputs.
- [ ] Explain how Story 4 depends on Stories 1 to 3 and hands off to Story 5.
- [ ] Keep optional secondary baseline comparisons and broader simulator bake-offs
      outside the mandatory Story 4 surface.

**Evidence produced**
- Focused regression coverage and developer-facing guidance for the workflow
  benchmark gate.
- Stable rerun instructions for the Story 4 artifact bundle.

**Risks / rollback**
- Risk: without clear notes, contributors may confuse Story 4 workflow-scale
  closure with full publication-bundle closure.
- Rollback/mitigation: document both the delivered gate and the remaining Story 5
  evidence packaging work explicitly.

### Engineering Task 8: Run Story 4 Validation And Confirm The Exact-Regime Gate

**Implements story**
- `Story 4: Workflow-Scale Anchor Sweeps Meet Exactness Thresholds Across The Mandatory Exact Regime`

**Change type**
- tests | validation automation

**Definition of done**
- The mandatory workflow-scale matrix runs successfully end-to-end.
- Every required 4, 6, 8, and 10 qubit case meets the workflow-level exactness
  threshold and completion requirements.
- Story 4 completion is backed by reviewable workflow artifacts rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the dedicated Story 4 workflow validation command or harness.
- [ ] Verify `100%` pass rate on the mandatory workflow benchmark set.
- [ ] Confirm at least one documented 10-qubit anchor evaluation case exists in
      the generated evidence.
- [ ] Record the stable artifact and test references for later Task 2 docs and
      Paper 1 evidence assembly.

**Evidence produced**
- A machine-readable Story 4 workflow artifact bundle with a `100%` pass rate on
  mandatory cases.
- Reviewable validation references for 4/6/8/10 workflow-scale exactness.

**Risks / rollback**
- Risk: Story 4 can appear complete while still lacking a reproducible proof of
  the full mandatory exact regime.
- Rollback/mitigation: treat the workflow bundle, the full pass rate, and the
  documented 10-qubit case as part of the exit gate, not optional follow-up.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- the mandatory 4, 6, 8, and 10 qubit anchor workflow matrix is defined and
  executed,
- each required workflow size has at least 10 fixed parameter vectors recorded,
- every mandatory workflow case compares `Re Tr(H*rho)` against Qiskit Aer with
  maximum absolute energy error `<= 1e-8`,
- all mandatory workflow cases complete without unsupported-operation
  workarounds and with explicit `density_matrix` backend provenance,
- runtime and peak-memory metrics are recorded for the mandatory workflow-scale
  cases,
- and one documented 10-qubit anchor evaluation case exists inside the stable
  Story 4 artifact bundle.

## Implementation Notes

- `benchmarks/density_matrix/story2_vqe_density_validation.py` already provides
  the natural backbone for Story 4 and should be extended rather than replaced.
- Story 4 should reuse the Story 2 and Story 3 artifact vocabulary where it is
  helpful, but keep the workflow-scale bundle clearly distinct from the 1 to 3
  qubit supported-only micro-validation bundle.
- `benchmarks/density_matrix/benchmark_perf.py` already captures runtime timing
  conventions and can inform the required performance-metric surface, but Story 4
  should keep exactness and performance structurally separate.
- Story 5 remains responsible for turning the collected workflow, trace, and
  reproducibility evidence into the final publication-ready bundle.
