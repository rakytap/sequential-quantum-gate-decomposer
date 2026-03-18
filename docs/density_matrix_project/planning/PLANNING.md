# High-Level Research And Implementation Plan

This document consolidates the validated codebase findings, the current
density-matrix roadmap, the PhD plan, and the design trade-offs discussed during
planning.

It is intentionally high-level. The goal is not to prescribe every engineering
step, but to define a research program that is:

- scientifically meaningful,
- technically feasible,
- publishable at multiple milestones,
- and tightly aligned with the main thesis theme:

> scalable methods for training quantum circuits under realistic noise models,
> combining advanced emulation frameworks with noise-aware optimization
> strategies.

## 1. Validated Starting Point

The current repository already provides two important foundations:

- a mature state-vector circuit partitioning and fusion path in
  `squander/partitioning` plus `qgd_Circuit` / `Gates_block`,
- and a real exact density-matrix backend based on `DensityMatrix`,
  `GateOperation`, `NoiseOperation`, and `NoisyCircuit`.

What is already true:

- density matrices, basic noisy channels, tests, and benchmark scripts exist,
- the density backend is usable for standalone exact noisy simulation,
- Phase 2 Tasks 1 to 6 now provide an integrated VQE-facing density path with
  explicit backend selection, exact `Re Tr(H*rho)` Hermitian-energy
  evaluation, required local-noise support, one canonical noisy XXZ VQE
  workflow contract, and a layered machine-checkable publication bundle
  covering the workflow contract, end-to-end 4/6 qubit evidence, 4/6/8/10
  exact-regime matrix evidence, deterministic unsupported-boundary evidence,
  and interpretation guardrails, with top-level semantic-closure checks that
  validate contract completeness, workflow-gate completeness, unsupported-case
  boundary integrity, and claim-closure rules rather than only artifact
  presence,
- the partitioning stack is mature for state-vector simulation and wide-circuit
  unitary optimization.

What remains intentionally open for later phases:

- the integrated density path is currently anchored to generated-`HEA` VQE
  workflows rather than broad decomposition and custom-circuit parity,
- partitioning and gate fusion are not yet available in the density-matrix path,
- the current partitioning cost model is still state-vector-oriented,
- and density-backend gradient routing plus broader circuit-source support remain
  future work.

## 2. Guiding Principles

The following principles should govern the project:

- Keep exact dense density matrices as the scientific anchor during the early
  phases.
- Prioritize realistic local noise models over toy global noise when making
  scientific claims.
- Judge performance work by its value for noisy training workflows, not only by
  raw simulator throughput.
- Reuse the existing state-vector partitioning infrastructure where possible,
  but do not pretend its current objective is already density-aware.
- Delay high-risk architectural generalization until benchmark evidence justifies
  it.
- Maintain reproducibility as a first-class output: code, configs, benchmarks,
  and validation data.

## 2.1 Success Conditions For A Meaningful PhD Project

The project should be considered scientifically successful only if all of the
following become true by the end of the main research path:

- the work answers training-related scientific questions, not only simulator
  implementation questions,
- the exact density-matrix backend remains strong enough to serve as a trusted
  reference for at least the early and middle phases,
- realistic local noise models are used in the core studies,
- the benchmark suite includes training-relevant circuits and not only synthetic
  simulation kernels,
- the performance work is tied to concrete experimental reach, such as more
  optimizer runs, larger sweeps, or broader trainability studies,
- the optimizer studies are rigorous enough to produce conclusions beyond a
  single task or ansatz,
- the trainability phase produces a reusable dataset and publication-grade
  evidence,
- and the code, experiment configuration, and validation methodology are all
  reproducible enough to survive external review.

If these conditions are not met, the project risks becoming an engineering effort
without a strong thesis-level scientific contribution.

## 3. Recommended Research Path

The recommended path is:

1. finish exact noisy training integration,
2. add density-aware acceleration through adapter-based unitary-island
   partitioning and fusion,
3. use the resulting backend for noisy optimizer and trainability studies,
4. only then branch into channel-native fusion or approximate scaling methods if
   the exact backend has reached its practical limit.

This path maximizes the chance of a coherent PhD narrative and multiple strong
papers.

## 4. Phases

### Phase 1: Exact Mixed-State Foundation

### Status

`Complete`

### Purpose

Establish an exact mixed-state and noisy-circuit baseline inside SQUANDER without
disrupting the existing state-vector path.

### Main Outputs

- `DensityMatrix` core,
- `NoisyCircuit` execution path,
- initial noise channels,
- Python bindings,
- tests and Qiskit comparison scripts.

### Scientific Value

This phase creates the exact reference engine needed for later optimization and
trainability work.

### Publication Potential

Useful for a workshop or software / validation paper, but not necessarily the
strongest standalone full paper.

### Exit Criteria

- exact mixed-state functionality validated,
- noisy circuits reproducible against a trusted reference backend,
- Phase 1 accepted as aligned with the PhD plan.

### Phase 2: Exact Noisy Training Backend Integration

### Status

`In Progress (Tasks 1-6 now deliver the canonical workflow contract and publication-facing evidence bundle; paper drafting and follow-on phase handoff remain active)`

### Main Goal

Turn the density-matrix module from a standalone simulator into a usable backend
for noisy variational workflows.

### Technical Focus

- backend selection for state-vector versus density-matrix workflows,
- exact Hermitian-energy path `Re Tr(H*rho)` for the supported XXZ workflow,
- generated-`HEA` bridge from `qgd_Variational_Quantum_Eigensolver_Base` into
  density-matrix execution,
- additional realistic local noise channels as required by the first training
  studies,
- stronger validation and benchmark coverage around small and medium noisy
  circuits.

### Recommended Scope Boundary

Do not attempt full density-aware partitioning or channel-native fusion as part
of the core Phase 2 deliverable. Phase 2 should make exact noisy training
possible and reproducible first.

### Primary Scientific Question

Can SQUANDER support exact noisy variational circuit evaluation in a way that is
scientifically reliable and practically usable?

### Main Deliverables

- a selectable density-matrix backend in the main workflow,
- exact noisy energy / observable evaluation,
- benchmark suite for noisy training-relevant circuits,
- one canonical noisy XXZ VQE workflow contract with explicit supported,
  optional, deferred, and unsupported boundaries,
- and a machine-checkable validation and publication-evidence package with stable
  workflow and case IDs, explicit status checks, metric completeness, and
  interpretation guardrails.

### Publication Target

First major paper: exact noisy backend integration.

### Exit Criteria

- `Tr(H*rho)` validated,
- exact noisy training loop runs end-to-end,
- noisy emulation stable around the exact target regime,
- machine-checkable benchmark and validation bundles exist for the delivered
  Phase 2 support surface,
- only mandatory, complete, supported evidence closes the main Phase 2 claim,
- and paper-quality benchmark, validation, and evidence-packaging outputs exist.

### Phase 3: Density-Aware Partitioning, Fusion, And Acceleration

### Main Goal

Bring the scientific value of the existing partitioning/fusion subsystem into the
density-matrix path.

### Technical Focus

- adapter-based unitary-island partitioning with noise barriers,
- density-unitary fused block execution,
- density-aware cost-model design and calibration,
- AVX/GPU-oriented acceleration of key density operations,
- benchmark comparison between sequential density execution and partitioned /
  fused density execution.

### Architectural Decision

Use the barrier-based unitary-island path first:

- partition contiguous unitary segments,
- preserve noise order exactly,
- use the current partitioning stack as a planner,
- validate every fused result against the unfused density baseline.

### Primary Scientific Question

Can partitioning and gate-fusion ideas, which are already useful for state-vector
 simulation, produce meaningful gains for exact density-matrix workloads relevant
 to noisy training?

### Main Deliverables

- density-aware partitioning prototype,
- benchmark-calibrated cost model,
- performance study over circuit families, partition sizes, and noise densities,
- documented limitations of barrier-based fusion.

### Decision Gate

At the end of Phase 3, decide whether:

- the barrier-based approach is sufficient for the project's scientific goals,
- or whether an IR-first / channel-native fusion branch should be prototyped.

### Publication Target

Major methods / systems paper.

### Exit Criteria

- exact agreement with the unfused density path,
- measurable speedups on representative noisy workloads,
- clear understanding of where partitioning helps and where it does not.

### Phase 4: Noisy VQA Workflows And Optimizer Studies

### Main Goal

Use the integrated and accelerated exact noisy backend to study training
behavior, not just simulation throughput.

### Technical Focus

- end-to-end noisy VQA workflows,
- optimizer comparison under exact local noise,
- BLS and baseline methods under matched conditions,
- entropy, purity, and gradient instrumentation,
- reproducible experiment management and configuration logging.

### Primary Scientific Question

How do optimizer choices and workflow design interact with realistic local noise
when the backend is exact rather than approximate or shot-driven?

### Main Deliverables

- optimizer comparison datasets,
- one or more realistic application cases,
- reproducible noisy training workflows suitable for publication and later thesis
  experiments.

### Publication Target

Applications / optimization paper.

### Exit Criteria

- at least one full noisy training workflow is robust and reproducible,
- optimizer comparisons are strong enough for publication,
- experiments can be scaled into the Phase 5 trainability campaign.

### Phase 5: Trainability Analysis Under Realistic Noise

### Main Goal

Deliver the central scientific results of the PhD.

### Technical Focus

- gradient variance studies,
- entropy and expressivity metrics,
- barren-plateau diagnostics,
- unital versus non-unital noise comparisons,
- depth/noise/locality sweeps across representative ansatze and tasks.

### Primary Scientific Question

How do realistic noise models change trainability, entropy growth, expressivity,
and barren-plateau behavior in variational quantum circuits?

### Main Deliverables

- publication-grade dataset,
- phase diagrams and statistical summaries,
- robust conclusions about noise-aware training design,
- and thesis-level synthesis of the full project.

### Publication Target

Main thesis science paper and strongest-impact result.

### Exit Criteria

- complete analysis dataset exists,
- conclusions are stable across repeated experiments,
- figures and tables are ready for publication and thesis inclusion.

## 5. Parallel Tracks That Should Not Derail The Main Path

The following tracks are valuable, but should remain subordinate to the main
five-phase plan unless benchmark evidence strongly justifies accelerating them.

### 5.1 Channel-Native / Superoperator Fusion

Why it matters:

- it may outperform barrier-based fusion for sufficiently noise-dense circuits,
- it is architecturally elegant,
- and it could become a strong follow-on methods paper.

Why it is not first:

- it is much more invasive,
- it is harder to validate,
- and it is not needed to reach the first major scientific results.

### 5.2 Stochastic Trajectories

Why it matters:

- it can reuse the stronger state-vector path,
- and it may extend the accessible qubit range later.

Why it is not first:

- the exact density backend should remain the reference engine first,
- and the PhD gains more from exactness early than from larger but approximate
  scale.

### 5.3 MPDO And Other Tensor-Based Mixed-State Methods

Why it matters:

- it is the natural next step once exact dense density matrices become limiting,
- and it opens a separate scaling paper.

Why it is not first:

- it introduces approximation immediately,
- and the project still needs the exact backend as a scientific anchor.

### 5.4 Hardware-Specific Scaling Tracks

Examples:

- GPU specialization,
- distributed memory,
- Groq / data-flow acceleration,
- and storage-assisted scaling.

These are valuable and align with the broader research plan, but they should be
fed by the exact-backend results rather than replacing them.

## 6. Decision Gates

To keep the project coherent, use explicit decision gates.

### DG-1: After Phase 2

Question:

- Is the exact noisy backend integrated enough to support reproducible training
  experiments?

If no:

- do not move major effort into density-aware partitioning yet.

### DG-2: During Phase 3

Question:

- Does barrier-based unitary-island fusion give enough benefit on target noisy
  workloads?

If yes:

- keep the barrier-based architecture as the default.

If no:

- prototype channel-native fusion as a separate research branch.

### DG-3: Before Large-Scale Phase 5 Experiments

Question:

- Is exact dense density-matrix simulation still sufficient for the most
  important trainability experiments?

If no:

- introduce trajectories or MPDO-style methods, but benchmark them against the
  exact backend first.

## 7. Benchmark And Validation Matrix

The following evaluation matrix should be built up progressively across phases.

### Baseline Frameworks

- Qiskit Aer density-matrix backend,
- one additional simulator or framework when feasible, such as QuEST, Qulacs,
  or QuTiP depending on the experiment type,
- and the existing unfused sequential density path as the internal exact
  baseline.

### Circuit Families

- hardware-efficient ansatze,
- chemistry-inspired ansatze,
- QAOA-like circuits,
- random or structured partitioning benchmarks,
- and at least one training-relevant application family.

### Noise Families

- local depolarizing,
- dephasing / phase damping,
- amplitude damping,
- whole-register depolarizing only as an optional regression or stress-test
  baseline,
- generalized amplitude damping or coherent over-rotation only when a later
  justified benchmark extension requires them,
- and readout / shot-noise plus calibration-aware variants only in later phases.

### Metrics

- fidelity / trace distance versus trusted references,
- energy / observable error,
- purity and entropy,
- gradient norms and variance,
- wall-clock runtime,
- memory footprint,
- speedup from partitioning / fusion,
- optimizer convergence quality.

## 8. Recommended Scope Cuts

To protect the main scientific path, the project should avoid the following
premature expansions:

- full noisy circuit re-synthesis as an early main result,
- full gate parity before training-relevant coverage exists,
- channel-native fused noisy partitions before benchmark evidence demands them,
- approximate scaling methods before the exact backend is fully integrated,
- publication claims based mainly on global depolarizing toy workloads.

## 9. Suggested Calendar Alignment

The following mapping keeps the new planning consistent with the existing PhD
roadmap while sharpening the technical priorities:

| Calendar window | Recommended emphasis |
|---|---|
| Fall 2025 | Phase 1 foundation work |
| Spring 2026 | Phase 2 backend integration, canonical workflow contract, and publication bundle delivered (Tasks 1-6) |
| Fall 2026 | Phase 2 paper packaging and Phase 3 acceleration begins |
| Spring 2027 | Phase 3 matures into benchmark-driven methods work |
| 2027 to 2028 | Phase 4 noisy optimization workflows |
| 2028 onward | Phase 5 trainability and final thesis science |

This mapping is approximate. The more important point is dependency order:

- exact backend first,
- density-aware acceleration second,
- optimizer science third,
- trainability science fourth,
- optional large-scale approximation branches only when justified.

## 10. Bottom Line

The most successful and meaningful version of this PhD project is not "build the
most general noisy simulator first." It is:

- build a trusted exact noisy backend,
- integrate it into training workflows,
- accelerate it with density-aware partitioning and fusion,
- and use it to answer real questions about training under realistic noise.

That path gives the project:

- the cleanest scientific claims,
- the strongest publication sequence,
- and the best alignment with the stated PhD research goal.
