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
- Phase 2 is complete and now provides an integrated VQE-facing density path
with explicit backend selection, exact `Re Tr(H*rho)` Hermitian-energy
evaluation, required local-noise support, one frozen canonical noisy XXZ VQE
workflow contract, and a layered machine-checkable publication bundle
covering the workflow contract, end-to-end 4/6 qubit evidence, 4/6/8/10
exact-regime matrix evidence, deterministic unsupported-boundary evidence,
and interpretation guardrails, with top-level semantic-closure checks that
validate contract completeness, workflow-gate completeness, unsupported-case
boundary integrity, and claim-closure rules rather than only artifact
presence,
- Phase 3 is complete and now provides a canonical noisy mixed-state planner
surface, schema-backed descriptor sets, an executable partitioned density
runtime, limited real fused execution on eligible substructures, and layered
correctness, performance, and publication-evidence packages,
- the partitioning stack is mature for state-vector simulation and wide-circuit
unitary optimization.

What remains intentionally open for later phases:

- the completed Phase 2 density path is intentionally anchored to one generated-
`HEA` canonical workflow rather than broad circuit-source parity,
- the delivered Phase 3 planner-calibration result is intentionally bounded to
the audited support surface rather than full density-aware parity across every
planner variant and circuit source,
- and broader VQE/VQA feature growth, including density-backend gradient
routing, richer workflow support, and optimizer studies, remains Phase 4+
work,
- fully channel-native fused noisy blocks remain a benchmark-driven follow-on
branch rather than part of the delivered baseline,
- and approximate scaling paths such as stochastic trajectories or MPDO-style
methods remain outside the current exact-backend critical path.

## 2. Guiding Principles

The following principles should govern the project:

- Keep exact dense density matrices as the scientific anchor during the early
phases.
- Prioritize realistic local noise models over toy global noise when making
scientific claims.
- Judge performance work by its value for noisy training workflows, not only by
raw simulator throughput.
- Reuse the existing state-vector partitioning infrastructure where possible,
but do not pretend its current objective or data model is already
density-aware and noise-aware.
- Treat noisy mixed-state circuits as first-class inputs to partitioning/fusion
work instead of treating noise only as partition-boundary metadata.
- Keep broader VQE/VQA feature expansion out of the bounded Phase 3 baseline;
Phase 4 is the earliest home for new workflow surface beyond the frozen
Phase 2 contract.
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

The recommended dependency order is:

1. preserve and package the exact noisy training integration achieved in
  Phase 2,
2. build later work on top of the delivered Phase 3 noise-aware
  partitioning/fusion baseline rather than reopening its core scope by default,
3. broaden VQE/VQA workflows, gradients, and optimizer studies only after that
  Phase 3 backend exists,
4. use the resulting exact backend for full trainability studies,
5. only then branch into more invasive fusion models or approximate scaling
  methods if the exact backend has reached its practical limit.

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

`Complete (core technical scope delivered; paper drafting and packaging continue as a parallel publication track)`

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

### Phase 3: Noise-Aware Partitioning And Fusion For Mixed-State Circuits

### Status

`Complete (minimum backend contract delivered; planner calibration is bounded and the current performance rule closes through diagnosis-grounded benchmark evidence while paper packaging continues in parallel)`

### Main Goal

Extend SQUANDER's partitioning and gate-fusion subsystem so noisy
density-matrix circuits are first-class inputs rather than post hoc unitary
islands separated by opaque noise boundaries.

### Technical Focus

- represent noise channels and density-matrix execution semantics inside
partition planning and fusion contracts,
- make partitioning decisions aware of noise placement, channel density, and
mixed-state execution cost,
- adapt fusion/runtime interfaces so partitions derived from noisy circuits
preserve exact gate/noise order,
- benchmark correctness and performance against the unfused sequential density
baseline on representative noisy circuit families,
- and only add optional density-kernel acceleration, including AVX-focused work,
when profiling and benchmark evidence shows that it materially supports the
mixed-state partition/fusion runtime.

### Architectural Decision

Start from the existing partitioning/fusion assets, but raise the Phase 3
contract:

- noisy circuits must be valid planner inputs,
- partition descriptors must retain exact noise placement and mixed-state
semantics,
- unitary-only islands may remain an internal optimization tactic, but not the
definition of the Phase 3 problem,
- every partitioned/fused result must be validated against the unfused density
baseline.

### Minimum Acceptable Implementation

Phase 3 is **not** complete if it only delivers a noise-aware planner/runtime
representation on paper.

The minimum acceptable implementation must include:

- an executable partitioned path for noisy mixed-state circuits end to end,
- at least one real fused execution mode for eligible substructures inside that
noisy-circuit runtime,
- and correctness plus benchmark evidence on representative noisy workloads.

The minimum acceptable implementation does **not** require:

- channel-native fused noisy blocks,
- monolithic CPTP or superoperator fusion of arbitrary gate+noise regions,
- or a proof that every useful noisy partition must already execute as one
fused noisy block.

In other words, Phase 3 must deliver more than planner-only representation, but
it does not require the most invasive fully fused noisy-block architecture as
its minimum closure condition.

### Primary Scientific Question

Can SQUANDER's partitioning/fusion subsystem be extended so exact noisy
mixed-state circuits remain semantically faithful while still yielding
meaningful acceleration on target workloads?

### Main Deliverables

- a partitioning representation that accepts noisy mixed-state circuits as
first-class inputs,
- a noise-aware partitioning objective / heuristic with benchmark calibration,
- an executable mixed-state partition/fusion runtime that preserves exact noise
ordering and includes at least one real fused execution path for eligible
substructures,
- a correctness and performance study over circuit families, noise placements,
and noise densities,
- documented limitations and a clear handoff to any more invasive follow-on
fusion work.

### Decision Gate Resolution

The delivered Phase 3 evidence resolves this gate in favor of keeping the
native noise-aware planner/runtime plus limited real fused execution as the
default main path.

Channel-native / IR-first fusion remains a benchmark-driven follow-on branch,
not a prerequisite for baseline Phase 3 closure.

### Publication Target

Major methods / systems paper.

### Exit Criteria

- partitioning and fusion accept noisy mixed-state circuits without reducing
noise to partition-boundary-only metadata,
- Phase 3 delivers more than planner-only representation by providing an
executable partitioned path plus at least one real fused execution mode on
representative noisy circuits,
- exact agreement with the unfused density path holds on representative noisy
workloads,
- channel-native fused noisy blocks remain optional follow-on work rather than a
minimum Phase 3 closure requirement,
- measurable speedups or, at minimum, benchmark-grounded evidence clearly
explains where noise-aware partitioning helps and where additional
architecture is required.

### Current Phase 3 Evidence Findings

The current fused-runtime implementation has now realized the minimum fused
backend in
one conservative concrete form:

- descriptor-local unitary-island fusion on 1- and 2-qubit spans using the
density backend's local-unitary primitive,
- an additive extension of the shared partitioned runtime and audit surface
rather than a second private runtime schema,
- and real fused coverage plus exact agreement with the sequential density
baseline on representative 8- and 10-qubit structured workloads.

The current benchmark outcome closes the Phase 3 performance rule through the
diagnosis branch rather than the positive-threshold branch. On representative
layered nearest-neighbor sparse workloads, the current fused baseline remains
slower at 8 and 10 qubits and shows no peak-memory reduction, primarily because
supported islands still remain partially unfused and the present Python-level
fused-kernel path adds overhead. This now argues for benchmark-driven follow-on
optimization or a more invasive channel-native branch, not for stronger
acceleration claims in the baseline Phase 3 result.

Current performance-evidence findings now make that diagnosis package concrete
at the phase
level. The emitted `performance_evidence` bundles record:

- `34` counted supported benchmark cases total,
- `4` continuity-anchor cases,
- `30` structured benchmark cases across the required families, sizes,
noise-placement patterns, and rerun-sensitive seed slices,
- `6` representative review cases, one primary-seed sparse structured case for
each required family and size,
- `0` positive-threshold pass cases,
- `6` diagnosis-only representative cases,
- and carry-forward of the `17` explicit correctness-evidence boundary cases
into the summary
layer.

The current benchmark closure is therefore not only a fused-runtime finding
anymore. It is a fully emitted performance-and-sensitivity package that closes
the Phase 3 rule through diagnosis-grounded benchmark evidence rather than
through a speedup claim.

### Phase 4: Broader Noisy VQE/VQA Workflows And Optimizer Studies

### Main Goal

Build on the completed Phase 2 workflow and the Phase 3 backend to broaden the
VQE/VQA surface, rather than adding those features during the partitioning
phase.

### Technical Focus

- broader circuit-source support beyond the generated-`HEA` Phase 2 contract,
- density-backend gradient and optimizer routing,
- end-to-end noisy VQE/VQA workflows,
- optimizer comparison under exact local noise,
- BLS and baseline methods under matched conditions,
- entropy, purity, and gradient instrumentation,
- reproducible experiment management and configuration logging.

### Primary Scientific Question

How do workflow design and optimizer choices interact with realistic local noise
once the exact backend and noise-aware partitioning path are both in place?

### Main Deliverables

- broader noisy VQE/VQA support beyond the frozen Phase 2 workflow,
- density-backend gradient/optimizer infrastructure for the supported Phase 4
surface,
- optimizer comparison datasets,
- one or more realistic application cases,
- reproducible noisy training workflows suitable for publication and later thesis
experiments.

### Publication Target

Applications / optimization paper.

### Exit Criteria

- at least one noisy VQE/VQA workflow beyond the Phase 2 baseline is robust and
reproducible,
- supported density-backend gradient and optimizer flows are stable enough for
optimizer studies,
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

- it may outperform the first native noise-aware partition/runtime for
sufficiently noise-dense circuits,
- it is architecturally elegant,
- and it could become a strong follow-on methods paper.

Why it is not first:

- it is much more invasive,
- it is harder to validate,
- and it should follow the benchmark evidence produced by the Phase 3
noise-aware baseline rather than replace it.

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

**Scope note:** The labels `DG-1` … here are **program-level** gates for this
document only. Phase detailed plans (`DETAILED_PLANNING_PHASE_2.md`,
`DETAILED_PLANNING_PHASE_3.md`, …) define their own numbered gates in context;
when cross-referencing, use the phase document’s wording as the checklist, and
treat the questions below as the rollup across phases.

### DG-1: After Phase 2

Status:

- Satisfied in the delivered roadmap: Phase 2 proved the exact noisy backend is
integrated enough to support reproducible experiments on the frozen canonical
workflow, so Phase 3 proceeded as the main technical track.

### DG-2: At Phase 3 Completion

Status:

- Answered `yes` for baseline closure: the native noise-aware partitioned
runtime plus limited real fused execution satisfies the Phase 3 exit criteria
on the delivered bounded support surface.
- The performance rule closes through diagnosis-grounded benchmark evidence,
not through positive-threshold speedup cases.
- Channel-native / IR-first fusion remains deferred follow-on work unless a
dedicated branch is opened with stronger benchmark justification.

### DG-3: Phase 3 To Phase 4 Handoff

Question:

- Are remaining limitations primarily broader workflow surface, gradients, and
optimizer-facing features, rather than unresolved Phase 3 backend semantics or
performance architecture?

If yes:

- proceed to Phase 4 broader noisy VQE/VQA work.

If no:

- close Phase 3 backend debt before expanding workflow scope.

### DG-4: Before Large-Scale Phase 5 Experiments

Question:

- Is exact dense density-matrix simulation still sufficient for the most
important trainability experiments?

If yes:

- run Phase 5 at the planned scale on the exact backend and evidence bundle.

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

The original calendar mapping is now partly historical because Phases 1-3 have
been delivered in code and documentation. The more important point remains the
dependency order:

- exact backend first,
- noise-aware partitioning/fusion second,
- broader VQE/VQA and optimizer science third,
- trainability science fourth,
- optional large-scale approximation branches only when justified.

Current roadmap emphasis:


| Period                        | Recommended emphasis                                                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Completed milestone window    | Phase 1 exact mixed-state foundation, Phase 2 exact noisy workflow integration, and Phase 3 bounded partitioning/fusion backend all delivered |
| Current publication window    | Phase 2 and Phase 3 manuscript packaging, figure polishing, and venue shaping                                                                 |
| Next implementation milestone | Phase 4 broader noisy VQE/VQA workflows, gradients, and optimizer studies                                                                     |
| Later thesis milestones       | Phase 5 trainability studies plus any benchmark-justified scaling branches                                                                    |


## 10. Bottom Line

The most successful and meaningful version of this PhD project is not "build the
most general noisy simulator first." It is to preserve the dependency order
that has now been partially realized:

- build and keep a trusted exact noisy backend,
- establish a bounded noise-aware partitioning/fusion baseline for mixed-state
circuits,
- broaden VQE/VQA workflows and optimizer studies only after that backend
exists,
- and use it to answer real questions about training under realistic noise.

That path gives the project:

- the cleanest scientific claims,
- the strongest publication sequence,
- and the best alignment with the stated PhD research goal.

