# Detailed Planning for Phase 3

This document is the Phase 3 source of truth for scope, task goals,
acceptance criteria, validation expectations, and Paper 2 evidence.

Primary Phase 3 theme:

> turn the exact noisy density backend from a sequential reference path into a
> native noise-aware partitioning and limited-fusion backend that remains
> semantically exact on representative noisy mixed-state workloads.

This is a specification document, not an implementation log.

## 0.1 Starting Baseline After Phase 2

Phase 3 begins from a stronger baseline than earlier planning drafts assumed.

What is already closed before Phase 3 starts:

- Phase 2 delivered a frozen exact noisy workflow contract with explicit
  backend selection, exact Hermitian-energy evaluation through
  `Re Tr(H*rho)`, a canonical noisy XXZ `HEA` workflow, and a
  machine-checkable publication bundle,
- the exact mixed-state reference path now exists in two useful forms:
  - the sequential `NoisyCircuit` execution contract in the density module,
  - and the workflow-facing Phase 2 anchor path that reaches the density
    backend without broadening the VQE/VQA surface,
- the state-vector partitioning stack in `squander/partitioning` remains mature
  for `qgd_Circuit` / `Gates_block` workloads,
- and the architectural extension points for Phase 3 are already named in
  `ARCHITECTURE.md`.

What remains intentionally open at the start of Phase 3:

- noisy mixed-state circuits are not yet native planner inputs,
- partition descriptors do not yet carry explicit noise placement and mixed-state
  semantics as first-class data,
- the exact density backend does not yet have an executable partitioned runtime
  plus real fused execution for noisy workloads,
- and the current partitioning cost model is still state-vector-oriented and
  noise-blind.

## 0.2 Phase 2 Learnings That Carry Into Phase 3

Phase 2 produced several contract lessons that should be treated as fixed Phase
3 planning assumptions:

- explicit support-tier vocabulary (`required`, `optional`, `deferred`,
  `unsupported`) is necessary to keep the Paper 2 claim auditable,
- silent fallback behavior is scientifically unacceptable; unsupported
  partitioned-density requests must fail explicitly,
- the sequential density path is now the exact internal ground truth that Phase
  3 must preserve,
- machine-checkable manifests and stable case identifiers are required for
  publication-facing reproducibility,
- and Phase 3 should optimize the frozen Phase 2 workflow baseline before
  expanding Phase 4 workflow surface.

## 0.3 Algorithm Landscape Overview

This section summarizes the current algorithm-selection judgment for Phase 3.
The detailed literature-grounded assessment is maintained in the companion note
[PHASE_3_ALGORITHM_LANDSCAPE.md](PHASE_3_ALGORITHM_LANDSCAPE.md).

Current public research signal:

- the literature is strong on state-vector partitioning and gate fusion, with
  direct relevance from `TDAG`, `GTQCP`, `QGo`, `QMin`, and related work,
- the literature is also growing on exact noisy density-matrix runtime
  acceleration, computational reuse, and hardware-aware dense mixed-state
  execution,
- but there is still comparatively little public work on native exact noisy
  mixed-state partition planners in which noise operations are first-class
  planner inputs and the runtime preserves explicit gate/noise ordering,
- which means that adapting SQUANDER's existing state-vector partitioning stack
  to the canonical noisy mixed-state planner surface remains a plausible Phase 3
  methods contribution rather than only an engineering port.

Phase 3 baseline recommendation:

- use a canonical noisy mixed-state operation DAG as the planner input,
- apply noise-aware dependency-respecting partitioning rather than barrier-only
  unitary partitioning,
- use conservative real fused execution through unitary-island fusion inside
  noisy partitions as the minimum fused baseline,
- and calibrate partition selection with a density-aware benchmark-informed
  objective rather than reusing the current state-vector FLOP model unchanged.

Applicability of the current SQUANDER partitioning stack:

- `kahn`, `tdag`, and `gtqcp` are appropriate Phase 3 heuristic baselines once
  noise operations become first-class nodes and the selection logic is measured
  on noisy density workloads,
- `ilp`, `ilp-fusion`, and especially `ilp-fusion-ca` are strong Phase 3
  candidates once their cost model and admissibility logic are redefined for
  exact noisy mixed-state execution,
- the existing `qiskit`, `qiskit-fusion`, and `bqskit-*` integrations remain
  useful comparison baselines, but they should not define the core Paper 2
  contribution,
- and the current state-vector cost model remains implementation scaffolding or
  comparison context only, not the final Phase 3 scientific claim.

Boundary to later work:

- multilevel or hypergraph partitioning and computational-reuse methods are
  strong optional Phase 3 extensions when the native baseline plateaus,
- local channel or Kraus fusion on very small support may be justified inside
  the Phase 3 runtime if it preserves the exact contract cleanly,
- and fully channel-native, Liouville-space, or superoperator-based noisy-block
  fusion remains a benchmark-driven follow-on branch beyond the baseline Phase 3
  closure condition.

## 0.4 Task 1 And Task 2 Learnings That Now Constrain Remaining Work

Task 1 and Task 2 implementation converted several Phase 3 planning assumptions
into real contract surfaces that later tasks should now treat as fixed unless a
cross-task phase decision is explicitly reopened.

The main implementation learnings are:

- the planner-side canonical noisy mixed-state surface and the Task 2 partition
  descriptor contract now both exist as schema-versioned, machine-reviewable
  handoff surfaces rather than only as planning abstractions,
- the mandatory workload matrix is no longer hypothetical:
  - the Phase 2 continuity anchor is implemented as required 4, 6, 8, and 10
    qubit cases,
  - the external micro-validation slice is implemented as a deterministic 2 to
    4 qubit microcase inventory,
  - and the structured methods slice is implemented with stable family IDs,
    seed rules, and sparse/periodic/dense local-noise placement vocabulary,
- unsupported-boundary evidence is already layered:
  - planner-entry unsupported behavior is now a machine-reviewable evidence
    surface from Task 1,
  - descriptor-generation unsupported and lossy behavior is now a separate
    machine-reviewable evidence surface from Task 2,
  - so later runtime correctness work should extend those layers rather than
    collapse them into one generic failure bucket,
- and machine-reviewable artifact bundles are now part of the real Phase 3
  contract:
  - later tasks should emit reusable bundles or rerunnable checkers,
  - and they should validate summary correctness and provenance completeness,
    not only bundle presence or schema shape.

Practical implication for Tasks 3 through 8:

- the remaining work should consume the frozen canonical-surface and
  descriptor-set contracts directly,
- the benchmark and correctness packages should build on the already-frozen
  workload IDs, seed rules, and negative-evidence taxonomies,
- and later planning language should stop referring to planner outputs and
  benchmark evidence only in generic terms when the actual review surfaces are
  now known.

## 1. Purpose

Phase 3 exists to bridge the current gap between:

- the exact noisy mixed-state backend already delivered and integrated for one
  canonical workflow in Phase 2, and
- the scalable exact backend needed to make later noisy VQE/VQA and
  trainability studies practical on richer benchmark matrices.

Phase 3 is the first phase where the density-matrix track becomes a performance
and methods result rather than only an exactness and integration result.

## 2. Source-of-Truth Hierarchy

If multiple documents overlap, interpret them in the following order of
authority for Phase 3.

### Tier 1: Strategic Planning Constraints

These documents define the authoritative Phase 3 intent and trade-off
boundaries:

- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`
- `docs/density_matrix_project/planning/REFERENCES.md`

### Tier 2: Accepted Roadmap And Milestone Wording

These documents define accepted milestone wording and must remain consistent
with the Phase 3 plan:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`

### Tier 3: Current Architecture And Frozen Phase 2 Baseline

These documents define the delivered baseline that Phase 3 extends:

- `docs/density_matrix_project/ARCHITECTURE.md`
- `docs/density_matrix_project/README.md`
- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`

### Tier 4: Legacy Or Supportive Context

These sources are informative, but do not override the above:

- existing benchmark notes and exploratory scripts,
- `docs/density_matrix_project/phases/phase-3/PHASE_3_ALGORITHM_LANDSCAPE.md`
  as the phase-local interpretive note for algorithm selection, literature
  mapping, and baseline-versus-follow-on classification,
- older flat API references not updated for the phase structure,
- and exploratory ideas about channel-native fusion, trajectories, or MPDOs
  that are not part of the accepted critical path.

## 3. Traceability Matrix

| Requirement area | Upstream source | Phase 3 interpretation |
|---|---|---|
| Freeze Phase 2 and optimize next | `PLANNING.md`, `ADRs.md`, `RESEARCH_ALIGNMENT.md` | Phase 3 must optimize the frozen exact noisy baseline before any Phase 4 workflow growth |
| Noisy mixed-state circuits as first-class planner inputs | `PLANNING.md`, `ADRs.md`, `ARCHITECTURE.md` | Partition planning must see noise placement and mixed-state execution semantics directly |
| Exact gate/noise-order preservation | `PLANNING.md`, `ARCHITECTURE.md` | Partition descriptors and runtime must preserve the sequential `NoisyCircuit` contract |
| Executable partitioned runtime plus real fused execution | `PLANNING.md`, `PUBLICATIONS.md`, `CHANGELOG.md` | Planner-only representation is insufficient for Phase 3 closure |
| Density-aware cost model | `PLANNING.md`, `ADRs.md` | Phase 3 must move beyond the state-vector FLOP model, but only after correctness-first baseline work |
| Realistic local noise as the scientific target | `ADRs.md`, `PLANNING.md`, `PUBLICATIONS.md` | Mandatory benchmarks focus on local depolarizing, amplitude damping, and phase damping |
| Internal and external validation baselines | `PLANNING.md`, `PUBLICATIONS.md`, `REFERENCES.md` | Sequential `NoisyCircuit` is the required internal exact baseline and Qiskit Aer remains the primary external reference |
| Paper 2 readiness | `PUBLICATIONS.md`, `REFERENCES.md` | Phase 3 must produce a methods-grade evidence package with correctness, performance, sensitivity, and limitation reporting |

## 4. Spec-Driven Development Principles for Phase 3

Phase 3 should follow spec-driven development in the following sense:

1. Freeze cross-task scope and semantic decisions before implementation details
   drift into architecture.
2. Define required observable behavior of the partitioned runtime before choosing
   specific planner or kernel tactics.
3. Treat unsupported and deferred mixed-state partitioning cases as explicit
   outcomes, not accidental gaps.
4. Require traceability from the Phase 3 methods claim to benchmark and
   reproducibility evidence.
5. Keep Phase 3 focused on noise-aware partitioning/fusion, not broader Phase 4
   workflow growth.
6. Use Paper 2 evidence needs to define what "done" means for this phase.

In practice, this means the Phase 3 contract must answer:

- what noisy mixed-state partitioning behavior is required,
- what counts as semantic preservation,
- what the minimum executable partitioned/fused result is,
- what benchmark evidence closes the acceleration claim,
- and what remains deliberately out of scope for later phases.

## 5. Mission Statement

Phase 3 delivers a native noise-aware partitioning and limited-fusion backend
for exact mixed-state circuits, including:

- first-class noisy circuit planner inputs,
- partition descriptors that preserve exact gate/noise semantics,
- an executable partitioned density-matrix runtime,
- at least one real fused execution path for eligible substructures,
- a benchmark-calibrated density-aware planning objective,
- and validation strong enough to support the main Phase 3 methods paper.

## 6. In Scope

The following are in scope for Phase 3.

### 6.1 First-Class Noisy Mixed-State Planner Inputs

- A canonical Phase 3 internal circuit representation equivalent to an ordered
  `NoisyCircuit` operation stream built from `GateOperation` and
  `NoiseOperation`.
- A documented contract for how partition planning sees:
  - gate order,
  - noise placement,
  - qubit support,
  - and parameter ordering.
- Compatibility with the frozen Phase 2 canonical workflow without expanding the
  user-facing VQE/VQA surface.

### 6.2 Exact Semantic Preservation Across Partitioning

- Partition descriptors that retain exact gate/noise order and any remapping
  metadata needed for faithful execution.
- Explicit unsupported-case behavior for circuits or operations that cannot be
  represented safely in the Phase 3 contract.
- Validation rules that compare the partitioned runtime against the sequential
  density baseline rather than against an implementation-specific shortcut.

### 6.3 Executable Partitioned Runtime

- An end-to-end partitioned density execution path for the mandatory benchmark
  matrix.
- Explicit selection of the partitioned path in tests or benchmark harnesses.
- No silent internal fallback from partitioned execution to the sequential
  baseline inside a benchmark that claims Phase 3 behavior.

### 6.4 Real Fused Execution For Eligible Substructures

- At least one real fused execution mode for eligible substructures inside the
  noisy partitioned runtime.
- The fused mode may still be unitary-island-based internally if exact noisy
  semantics are preserved at the runtime boundary.
- Fully channel-native fused noisy blocks are not required for the minimum Phase
  3 result.

### 6.5 Noise-Aware Planning Objective

- A structural noise-aware partitioning heuristic baseline,
- followed by a benchmark-calibrated density-matrix objective or heuristic,
- with explicit documentation of where the current state-vector model is reused
  only as implementation scaffolding and where Phase 3 claims become
  density-specific.

### 6.6 Validation, Benchmarking, And Publication Evidence

- Correctness validation against the sequential `NoisyCircuit` baseline on all
  mandatory cases,
- external exactness comparison against Qiskit Aer on the required microcases,
- reproducible performance characterization over representative noisy circuit
  families,
- and a Paper 2 evidence package that records both strengths and limitations of
  the Phase 3 baseline.

## 7. Out of Scope

The following are explicitly outside the Phase 3 core deliverable.

- broader noisy VQE/VQA workflow growth beyond the frozen Phase 2 canonical
  workflow,
- density-backend gradient routing and optimizer studies,
- full `qgd_Circuit` gate parity as a completion requirement,
- fully channel-native or superoperator-native fusion of arbitrary gate+noise
  regions,
- stochastic trajectories, MPDO-style approximate scaling, or other approximate
  replacements for the exact reference backend,
- readout-noise, shot-noise, and calibration-aware workflow features as the main
  Phase 3 path,
- and AVX-level kernel tuning as a standalone success claim independent of an
  executable noise-aware partitioned runtime.

These topics may be mentioned as future work, but they must not be presented as
Phase 3 commitments.

## 8. Why Phase 3 Must Precede Phase 4

Phase 3 and Phase 4 are related, but they are not interchangeable roadmap
milestones.

Phase 3 optimizes the exact noisy backend through noise-aware partitioning and
limited real fusion. Phase 4 broadens noisy VQE/VQA surface, gradients, and
optimizer-facing workflows on top of that backend.

### 8.1 Convincing Technical Argument

From a technical perspective, Phase 4 needs a stable accelerated backend to
stand on.

After Phase 2, the project already has:

- a frozen exact noisy workflow contract,
- a sequential exact `NoisyCircuit` execution baseline,
- and a mature state-vector partitioning/fusion subsystem that can be reused
  structurally.

However, before Phase 3 completes, the project still lacks:

- a native planner input for noisy mixed-state circuits,
- partition descriptors that preserve exact noise placement,
- an executable partitioned density runtime,
- and benchmark-calibrated evidence about where partitioning/fusion helps on the
  intended workloads.

If Phase 4 were moved ahead of Phase 3, the project would broaden workflow
surface on top of a backend whose main remaining limitation is still execution
cost. That creates several technical risks:

- broader workflows could be added on top of a performance bottleneck that is
  not yet understood,
- optimizer and gradient work would target an unstable backend cost profile,
- benchmark narratives could drift toward workflow breadth rather than backend
  scalability,
- and later performance conclusions would be weakened because the unoptimized
  backend, rather than the intended Phase 3 methods result, would shape the
  experimental design.

In short, Phase 2 created the exact workflow baseline. Phase 3 makes that
baseline performance-meaningful. Phase 4 should only broaden workflow surface
after the Phase 3 backend exists.

### 8.2 Convincing Scientific Argument

From a scientific perspective, the order is equally important.

The PhD theme is not only "support more noisy workflows." It is scalable methods
for training under realistic noise. The next question after Phase 2 is therefore:

- can exact noisy mixed-state execution be accelerated in a way that preserves
  semantics and remains scientifically trustworthy?

Only after that question is answered does the next one become meaningful:

- how do richer noisy workflows, gradients, and optimizers behave on top of that
  accelerated exact backend?

This order strengthens the publication ladder:

1. Phase 2 yields the exact noisy integration paper.
2. Phase 3 yields the noise-aware partitioning/fusion methods paper.
3. Phase 4 yields the broader optimizer and workflow paper.

If the order were reversed, the project would risk a weaker scientific story:

- broader workflows on what exact performance baseline?
- optimizer conclusions under what backend cost model?
- and workflow growth instead of a clear methods contribution at the point where
  the methods question is most natural?

### 8.3 Practical Interpretation

The practical interpretation is:

- Phase 3 benchmarks should remain training-relevant, but they should not depend
  on new Phase 4 features,
- exploratory channel-native fusion work may run in parallel as a research
  branch, but it should not replace the minimum Phase 3 closure condition,
- and Phase 3 completion remains the correct handoff point for broader Phase 4
  noisy VQE/VQA expansion.

## 9. Assumptions

Phase 3 planning assumes:

- the Phase 2 canonical noisy XXZ `HEA` workflow remains frozen and valid as the
  continuity anchor,
- the current sequential `NoisyCircuit` executor remains the internal exact
  reference path for Phase 3 semantic validation,
- the current partitioning stack in `squander/partitioning` plus the fusion
  boundary around `Gates_block::apply_to()` can be reused structurally, even
  though their current semantics are state-vector-oriented,
- the main required Phase 3 gate surface can remain workload-driven and centered
  on `U3` and `CNOT` plus local noise, with broader gates only added when a
  mandatory benchmark or validation microcase requires them,
- local depolarizing, amplitude damping, and phase damping remain the mandatory
  scientific noise families,
- Qiskit Aer remains the primary external exact reference simulator,
- and optional density-kernel tuning should be driven by profiler evidence
  rather than treated as a prerequisite architecture commitment.

## 10. Success Conditions

Phase 3 is only scientifically successful if all of the following become true:

- noisy mixed-state circuits are accepted as first-class planner inputs rather
  than reduced to barrier-only metadata,
- partition descriptors preserve exact gate/noise order and parameter routing
  strongly enough to reproduce the sequential density baseline within the frozen
  thresholds,
- the partitioned density path executes end to end on the mandatory benchmark
  matrix,
- at least one real fused execution mode exists for eligible substructures
  inside that partitioned runtime,
- the benchmark package shows either measurable benefit on representative noisy
  workloads or a benchmark-grounded diagnosis of the dominant remaining
  bottleneck,
- external exactness and internal semantic-preservation evidence are both
  publication-ready,
- and broader Phase 4 workflow growth remains outside the Phase 3 success claim.

### 10.1 Frozen Implementation Contract Decisions

The following decisions are part of the Phase 3 contract.

#### Planner Input Decision

Phase 3 uses a canonical noisy mixed-state planner surface.

Contract:

- The canonical internal Phase 3 input is an ordered noisy operation sequence
  equivalent to `NoisyCircuit` operations built from `GateOperation` and
  `NoiseOperation`.
- Every mandatory Phase 3 benchmark case must be representable in that canonical
  surface before partition planning begins.
- The frozen Phase 2 workflow remains a required source workload, but it is
  judged through the canonical noisy mixed-state representation rather than
  through new VQE-facing feature growth.
- Lowering from `qgd_Circuit` or `Gates_block` into the canonical Phase 3
  surface is allowed and desirable when exact, but it is not the sole defining
  contract for phase completion.
- Unsupported circuit sources or unsupported operations in `partitioned_density`
  mode must hard-error before execution. No silent fallback to the sequential
  path is allowed inside a benchmark that claims Phase 3 behavior.

Trade-offs:

- Choosing a canonical noisy operation surface makes Phase 3 semantics explicit
  and keeps the phase aligned with the actual density backend.
- The cost is that full direct `qgd_Circuit` parity remains deferred instead of
  being implied by the planner contract.

#### Semantic Preservation Decision

Phase 3 makes semantic preservation a non-negotiable contract item.

Contract:

- Partition descriptors must retain:
  - operation order,
  - qubit support,
  - parameter slices or equivalent parameter-routing metadata,
  - and any remapping metadata needed to execute the partitioned circuit exactly.
- Noise placement is part of the semantic model and cannot be reduced to
  out-of-band partition-boundary markers.
- Reordering across noise boundaries is not part of the required Phase 3
  contract unless an exact equivalence rule is separately documented and
  validated.
- Every required partitioned execution result must be comparable to the
  sequential density baseline using the numeric thresholds below.

Trade-offs:

- This strict contract limits aggressive optimization, but it keeps correctness
  and publication claims defensible.
- It also preserves a clean handoff point if later work explores more invasive
  channel-native fusion.

#### Runtime And Fused Execution Decision

Phase 3 must deliver more than planner-only representation.

Contract:

- The minimum required runtime result is an executable partitioned density path
  for the mandatory workload matrix.
- At least one real fused execution mode must run eligible substructures inside
  that partitioned runtime.
- The fused path may remain unitary-island-based internally if:
  - the surrounding noisy semantics are preserved exactly,
  - and the fused path is benchmarked on real Phase 3 workloads rather than on
    only synthetic kernels.
- Fully channel-native fused noisy blocks are explicitly deferred beyond the
  Phase 3 minimum.
- The sequential density path remains the benchmark reference and correctness
  oracle, not an implicit implementation fallback.

Trade-offs:

- This baseline is strong enough to support Paper 2 without forcing the most
  invasive architecture from the beginning.
- The cost is that some later performance gains may remain unavailable until a
  follow-on channel-native branch is justified.

#### Cost Model Decision

Phase 3 adopts a correctness-first path to density-aware planning.

Contract:

- The Phase 3 implementation sequence is:
  1. establish native noisy-circuit correctness and executable partitioned
     runtime,
  2. implement structural noise-aware planning behavior on the supported noisy
     planner surface,
  3. calibrate a density-specific benchmark-informed objective, heuristic, or
     bounded candidate-setting policy,
  4. optionally tune kernels if profiling shows that they materially affect the
     benchmark outcome.
- The existing state-vector FLOP model in `squander/partitioning/tools.py` may
  be reused as scaffolding or a comparison point, but it does not define the
  Phase 3 claim.
- In the current delivered Task 5 result, the supported calibration surface is
  the existing noisy planner with auditable `max_partition_qubits` span-budget
  settings; broader adapted `kahn` / `tdag` / `gtqcp` / `ilp` /
  `ilp-fusion-ca` families remain design-space or comparison references until
  they are separately implemented on the noisy planner path.
- Phase 3 should avoid "optimal partitioning" language until the density-aware
  heuristic or objective has been benchmark-calibrated on the required workload
  matrix.
- If the benchmark-grounded rule yields close or rerun-sensitive winners inside
  the bounded candidate family, the supported claim should be the auditable
  selection rule plus explicit claim boundary, not one permanently frozen
  winner identity.
- Planner-time overhead must be reported as part of the benchmark package.

Trade-offs:

- Correctness-first sequencing reduces scientific risk and avoids premature
  optimality claims.
- The downside is that the strongest cost-model claim may arrive later in the
  implementation cycle than the first executable partitioned runtime.

#### Support Matrix Decision

Phase 3 freezes the following support surface.

| Area | Required in Phase 3 | Optional in Phase 3 | Deferred beyond core Phase 3 |
|---|---|---|---|
| Canonical input surface | ordered noisy mixed-state operation stream equivalent to `NoisyCircuit` plus Phase 2 continuity lowering into that surface | additional exact lowering paths from existing `qgd_Circuit` / `Gates_block` workflows | full direct parity for every circuit source |
| Gate families | `U3`, `CNOT` for the mandatory benchmark matrix | additional gates already exposed by `NoisyCircuit` when a mandatory microcase or secondary benchmark genuinely needs them | full `qgd_Circuit` parity, multi-controlled gates, arbitrary `Composite` or `Adaptive` structures as a minimum requirement |
| Noise models | local single-qubit depolarizing, local amplitude damping, local phase damping / dephasing | whole-register depolarizing as a baseline; generalized amplitude damping or coherent unitary error only when a justified extension needs them | correlated multi-qubit noise, calibration-aware noise, readout-noise or shot-noise workflow features, non-Markovian noise |
| Unsupported behavior | explicit hard error for unsupported `partitioned_density` requests | documented opt-in extensions | silent fallback, silent omission of noise operations, or hidden sequential-mode substitution |

Trade-offs:

- Keeping the mandatory surface centered on the Phase 2 gate/noise anchor keeps
  Phase 3 bounded around the methods question rather than gate-surface growth.
- The cost is that broader gate families remain future extensions instead of
  being claimed as part of the minimum Paper 2 contribution.

#### Workflow And Benchmark Anchor Decision

Phase 3 uses both continuity and methods-oriented benchmark anchors.

Contract:

- The mandatory continuity anchor is the frozen Phase 2 noisy XXZ `HEA` workflow
  executed through the exact density path without adding new Phase 4 workflow
  features.
- The mandatory methods-oriented circuit families are:
  - layered nearest-neighbor `U3` / `CNOT` noisy circuits,
  - seed-fixed random layered `U3` / `CNOT` noisy circuits,
  - and one structured partitioning stress family expressed in the same required
    gate surface.
- The mandatory noise-placement patterns are:
  - sparse local-noise insertion after selected entangling regions,
  - periodic local-noise insertion at layer boundaries,
  - and dense local-noise insertion after every logical layer.
- The mandatory correctness scale coverage is:
  - external micro-validation at 2 to 4 qubits,
  - internal partitioned-versus-sequential correctness at 4, 6, 8, and 10
    qubits,
  - and mandatory performance recording at 8 and 10 qubits for the structured
    families.
- Larger exact-regime cases may be added when they materially strengthen Paper
  2, but they are not required for the minimum Phase 3 closure claim.

Trade-offs:

- The continuity anchor preserves direct relevance to the Phase 2 workflow and
  later noisy training studies.
- The structured methods families create a stronger partitioning/fusion paper
  than the continuity workflow alone.
- The cost is that Phase 3 intentionally does not claim broad algorithm-family
  coverage.

#### Benchmark Minimum Decision

The Phase 3 minimum benchmark package is frozen as follows.

Mandatory:

- internal exact baseline:
  - sequential `NoisyCircuit` execution for every required Phase 3 case,
- external exact baseline:
  - Qiskit Aer density-matrix simulation for all 2 to 4 qubit microcases and a
    representative small continuity subset,
- circuit classes:
  - 2 to 4 qubit micro-validation circuits that stress partition boundaries,
    each required noise model, and representative noise placements,
  - the frozen Phase 2 noisy XXZ `HEA` continuity cases at 4, 6, 8, and 10
    qubits,
  - structured noisy `U3` / `CNOT` partitioning families at 8 and 10 qubits,
  - at least 3 seed-fixed circuit instances per mandatory structured family and
    size,
- metrics:
  - density agreement versus sequential baseline,
  - exactness versus Qiskit Aer on the external microcases,
  - energy agreement on the continuity anchor,
  - runtime,
  - peak memory,
  - planning time,
  - partition count and qubit span,
  - and fused-path coverage for eligible substructures,
- reproducibility artifacts:
  - circuit-family identifiers,
  - seeds or deterministic construction rules,
  - noise-placement pattern and parameters,
  - partition-size and planner settings,
  - software version or commit,
  - raw benchmark outputs,
  - and profiler artifacts for hotspot cases when profiling is used to justify
    kernel work or follow-on architecture decisions.

Optional:

- one additional simulator baseline such as QuEST or Qulacs when it materially
  strengthens Paper 2,
- larger exact-regime cases beyond the minimum 10-qubit required matrix,
- and exploratory channel-native comparison branches that are clearly separated
  from the minimum claim.

Trade-offs:

- This benchmark floor is strong enough to support a methods paper without
  turning Phase 3 into a broad multi-framework bake-off.
- The downside is that the phase remains deliberately bounded around a few
  representative workloads rather than every possible noisy circuit family.

#### Numeric Acceptance Threshold Decision

Phase 3 go or no-go thresholds are frozen as:

- external micro-validation exactness:
  - maximum Frobenius-norm density difference `<= 1e-10` against Qiskit Aer on
    the 2 to 4 qubit required microcases,
- internal semantic-preservation exactness:
  - maximum Frobenius-norm density difference `<= 1e-10` between partitioned and
    sequential execution on all mandatory correctness cases,
  - `|Tr(rho) - 1| <= 1e-10` and `rho.is_valid(tol=1e-10)` on recorded required
    outputs,
- continuity-anchor observable agreement:
  - maximum absolute energy error `<= 1e-8` on the required 4, 6, 8, and 10
    qubit Phase 2 continuity cases,
- benchmark completeness:
  - `100%` pass rate on the mandatory external microcases,
  - `100%` pass rate on the mandatory internal correctness matrix,
  - and at least one benchmarked real fused execution mode on eligible
    substructures,
- performance-evidence threshold:
  - either at least one required 8- or 10-qubit structured case shows median
    wall-clock speedup `>= 1.2x` or peak-memory reduction `>= 15%` versus the
    sequential baseline without correctness loss,
  - or the required benchmark package plus profiling evidence explicitly shows
    why the native Phase 3 baseline does not yet accelerate that case and
    justifies the follow-on architecture decision gate,
- Paper 2 readiness threshold:
  - correctness thresholds above pass,
  - reproducibility artifacts are complete,
  - the structured sensitivity matrix is recorded,
  - and limitations are documented honestly where speedups do not appear.

Trade-offs:

- These thresholds are strict enough to make semantic-preservation claims
  defensible.
- They avoid making Phase 3 success depend on an unrealistic global speedup
  claim across all noisy workloads.

## 11. Phase 3 Task Breakdown

Each task below is a goal, not an implementation recipe.

### Task 1: Canonical Noisy Planner Surface

#### Goal

Define the Phase 3 canonical noisy mixed-state input surface and its supported
entry paths.

#### Why It Exists

Without a canonical planner surface, Phase 3 risks becoming an adapter patchwork
instead of a clear methods result.

#### Success Looks Like

- the planner input contract is explicit,
- the frozen Phase 2 continuity workflow reaches that contract,
- and unsupported input cases are defined.

#### Evidence Required

- a documented canonical representation,
- explicit entry and unsupported-case semantics,
- and at least one required workload family routed through that contract.

### Task 2: Partition Descriptor And Semantic-Preservation Contract

#### Goal

Define the minimum partition descriptor metadata needed to preserve exact noisy
mixed-state semantics.

#### Why It Exists

Noise-aware partitioning is not scientifically credible if gate/noise order or
parameter routing become ambiguous.

#### Success Looks Like

- partition descriptors retain exact order and mapping information,
- noise placement is first-class,
- and semantic preservation is measurable.

#### Evidence Required

- descriptor specification,
- correctness criteria tied to the sequential baseline,
- and negative-case handling for unsupported transformations.

### Task 3: Executable Partitioned Density Runtime

#### Goal

Deliver an end-to-end partitioned density execution path that consumes the
frozen Task 2 partition-descriptor contract on the mandatory workload matrix.

#### Why It Exists

Planner-only representation is not enough to support the Phase 3 methods claim.

#### Success Looks Like

- the partitioned path executes required cases end to end from the validated
  Task 2 descriptor set rather than from hidden planner state or
  workload-specific ad hoc adapters,
- explicit partitioned mode selection exists in the harness,
- and no silent sequential fallback remains.

#### Evidence Required

- required benchmark runs using the partitioned path on the frozen continuity,
  microcase, and structured-workload matrix,
- explicit descriptor-to-runtime handoff evidence showing that the runtime
  consumes the supported Task 2 contract rather than a second private runtime
  schema,
- explicit baseline comparison to the sequential runtime,
- and documented unsupported-case behavior that remains distinguishable from the
  already-frozen planner-entry and descriptor-generation negative-evidence
  layers.

### Task 4: Real Fused Execution On Eligible Substructures

#### Goal

Provide at least one real fused execution mode on explicitly defined eligible
descriptor-level substructures inside the noisy partitioned runtime.

#### Why It Exists

Paper 2 should claim more than partition descriptors and scheduling; it needs a
real runtime result.

#### Success Looks Like

- an eligible substructure defined against the supported Task 2 descriptor
  contract actually executes through a fused path,
- that fused path is benchmarked on representative noisy circuits,
- surrounding noise semantics remain exact,
- and unsupported or deferred fusion candidates remain explicit instead of
  silently entering a fused path.

#### Evidence Required

- benchmarked fused-path cases,
- semantic-preservation validation,
- structured evidence for what classes of fusion are:
  - supported and actually exercised,
  - left intentionally unfused but still supported,
  - or explicitly deferred beyond the minimum Phase 3 claim.

#### Current Implementation Findings

- The delivered minimum fused path is descriptor-local unitary-island fusion on
  1- and 2-qubit spans using the density backend local-unitary primitive rather
  than a separate private fusion-only runtime.
- The shared Task 3 runtime contract remained additive: the runtime schema
  stayed stable while fused-region classifications and an explicit fused runtime
  path were added on top of the existing audit surface.
- Representative 8- and 10-qubit layered nearest-neighbor sparse structured
  workloads now exercise a real fused path and preserve exact noisy semantics
  against the sequential density baseline within the frozen thresholds.
- The current Task 4 benchmark result closes the performance-evidence rule via
  the diagnosis branch rather than the positive-threshold branch. On those
  representative structured cases, the current fused baseline remains slower and
  does not reduce peak memory, mainly because supported islands remain partially
  unfused and the present Python-level fused-kernel path adds overhead.

### Task 5: Noise-Aware Planning Heuristic And Calibration

#### Goal

Define and calibrate a density-aware planning heuristic, objective, or bounded
candidate-setting policy for the required workload matrix and the now-frozen
canonical-surface plus descriptor-level contract.

#### Why It Exists

Reusing a state-vector cost model would weaken the scientific Phase 3 claim.

#### Success Looks Like

- the planner responds to mixed-state cost, explicit noise placement, and the
  frozen workload knobs that the implemented benchmark inventory already makes
  reproducible,
- calibration is benchmark-informed rather than purely analytic,
- and the supported claim boundary is explicit.

#### Evidence Required

- heuristic, objective, or bounded candidate-setting definition,
- calibration benchmark data on the frozen continuity, microcase, and
  structured-workload inventory with stable workload IDs, seed rules, and
  noise-pattern labels,
- a machine-reviewable calibration bundle or equivalent rerunnable checker that
  records planner settings, workload identity, and calibration outputs,
- and documentation of where the model remains approximate.

#### Current Implementation Findings

- The delivered minimum Task 5 result is a benchmark-calibrated selection rule
  over auditable `max_partition_qubits` span-budget candidates on the existing
  noisy planner surface rather than a broader family of separately implemented
  noisy `kahn` / `tdag` / `gtqcp` / `ilp` variants.
- The current calibration package is machine-reviewable and anchored to the
  frozen continuity, microcase, and structured-workload inventory with stable
  workload IDs, seed rules, and noise-pattern labels.
- The external Qiskit Aer slice remains valid on the required microcases and the
  representative small continuity slice, but the benchmark-grounded selected
  winner inside the bounded candidate family has proved rerun-sensitive rather
  than permanently frozen.
- The current supported Task 5 claim should therefore be phrased around the
  auditable selection rule, bounded candidate family, and explicit comparison
  baselines rather than around one invariant winner identity.

### Task 6: Correctness Validation And Unsupported-Boundary Evidence

#### Goal

Define the minimum correctness package required before any acceleration claim is
treated as valid, while keeping planner-entry, descriptor-generation, and
runtime-only unsupported boundaries explicitly separated.

#### Why It Exists

Phase 3 remains exact-first. Performance without trustworthy semantics would
weaken both the paper and the broader PhD path.

#### Success Looks Like

- external and internal correctness checks are explicit,
- planner-entry unsupported cases, descriptor-generation unsupported cases, and
  runtime-only unsupported or deferred cases remain distinguishable in the
  evidence package,
- required pass thresholds are frozen,
- and unsupported or deferred cases are documented.

#### Evidence Required

- Qiskit Aer micro-validation,
- sequential-versus-partitioned density comparisons,
- structured unsupported-boundary artifacts covering:
  - planner-entry unsupported cases,
  - descriptor-generation unsupported or lossy cases,
  - and runtime-only unsupported or deferred behavior where later tasks expose
    it.

### Task 7: Performance And Sensitivity Benchmark Package

#### Goal

Produce the representative performance matrix needed to support the Phase 3
methods paper, with stable per-case provenance and validated rolled-up
benchmark summaries.

#### Why It Exists

Phase 3 should explain where noise-aware partitioning helps, not only that a
partitioned runtime exists.

#### Success Looks Like

- runtime and memory behavior are characterized on representative workloads,
- sensitivity to partition size, noise placement, and the frozen workload
  identity knobs already implemented for the methods matrix is recorded,
- every benchmark record remains auditable through stable per-case provenance,
- and the remaining bottlenecks are explicit.

#### Evidence Required

- mandatory runtime and memory matrix with per-case provenance, support labels,
  and raw case records,
- planning-overhead and fused-coverage metrics,
- summary-consistency checks that validate rolled-up benchmark counts and
  classifications against the underlying case records,
- and profiler artifacts when they materially affect conclusions.

### Task 8: Paper 2 Evidence And Documentation Bundle

#### Goal

Define the exact Paper 2 evidence package and supporting documentation for Phase
3 as a manifest over the emitted positive and negative evidence bundles.

#### Why It Exists

Publication readiness should shape the methods phase from the start rather than
being retrofitted after implementation.

#### Success Looks Like

- Paper 2 has a clear claim boundary,
- the benchmark package maps directly to that claim,
- supported and unsupported evidence both appear in the package where they
  matter to the claim boundary,
- summary and provenance checks make the rolled-up package auditable,
- and deferred branches are visible instead of hidden.

#### Evidence Required

- Phase 3 narrative claim set,
- reproducibility manifest expectations tied to the emitted bundle or checker
  surfaces rather than to prose-only descriptions,
- explicit inclusion rules for both positive supported-path evidence and
  negative unsupported-boundary evidence,
- summary-consistency and provenance-completeness checks for the publication
  bundle,
- and traceability from Phase 3 tasks to publication sections.

## 12. Full-Phase Acceptance Criteria

Phase 3 is complete only if all of the following are true:

- the Phase 3 planner accepts noisy mixed-state circuits without reducing noise
  to partition-boundary-only metadata,
- partition descriptors preserve exact gate/noise order and parameter routing
  strongly enough to satisfy the thresholds in Section 10.1,
- the partitioned density runtime executes the mandatory benchmark matrix end to
  end,
- at least one real fused execution mode is used and benchmarked on eligible
  substructures,
- exact agreement with the sequential density baseline holds on the required
  correctness matrix and Qiskit Aer agreement holds on the required microcases,
- the benchmark package records runtime, memory, planning overhead, and fusion
  coverage on the representative noisy workloads,
- and the Paper 2 evidence bundle documents both the achieved benefit and the
  remaining limitations honestly.

These criteria are intentionally aligned with:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`

## 13. Validation And Benchmark Matrix

### 13.1 Primary Internal Baseline

- Sequential `NoisyCircuit` execution is the required internal exact reference
  for every mandatory Phase 3 case.

### 13.2 Primary External Baseline

- Qiskit Aer density-matrix simulation is the required external reference on the
  mandatory 2 to 4 qubit microcases and representative small continuity cases.

### 13.3 Optional Secondary Baselines

- QuEST or Qulacs may be added when they materially strengthen the Paper 2
  narrative, but they are secondary to the sequential density baseline and
  Qiskit Aer.

### 13.4 Workload Classes

- 2 to 4 qubit micro-validation circuits stressing:
  - partition-boundary behavior,
  - required local noise models,
  - and representative noise placements,
- Phase 2 continuity cases:
  - noisy XXZ `HEA` workflow instances at 4, 6, 8, and 10 qubits,
- structured Phase 3 methods families:
  - layered nearest-neighbor `U3` / `CNOT` circuits at 8 and 10 qubits,
  - seed-fixed random layered `U3` / `CNOT` circuits at 8 and 10 qubits,
  - and one structured partitioning stress family at 8 and 10 qubits.

### 13.5 Noise Classes And Placements

- local single-qubit depolarizing,
- local amplitude damping,
- local phase damping / dephasing,
- sparse local-noise placement,
- periodic layer-boundary local-noise placement,
- dense layer-wise local-noise placement,
- whole-register depolarizing only as an optional baseline,
- and generalized amplitude damping or coherent unitary error only as justified
  extensions beyond the minimum claim.

### 13.6 Metrics

- Frobenius-norm density difference,
- energy error on continuity-anchor cases,
- trace preservation and density validity,
- runtime,
- peak memory footprint,
- planner runtime,
- partition count and qubit span,
- fused-path coverage for eligible substructures,
- and reproducibility completeness of the benchmark artifact bundle.

## 14. Risks

### Risk 1: Planner-Only Closure

The implementation could produce a strong representation layer without a real
runtime result.

Mitigation:

- keep executable partitioned runtime plus real fused execution as an explicit
  completion requirement.

### Risk 2: Semantic Drift Across Noise Boundaries

Partitioning or fusion could appear fast while silently changing exact noisy
semantics.

Mitigation:

- preserve exact order as a contract item,
- and validate every mandatory case against the sequential density baseline.

### Risk 3: Weak Performance Narrative

The native noise-aware baseline could fail to show clear benefit on the chosen
workloads.

Mitigation:

- require structured sensitivity benchmarks,
- profiler-backed bottleneck reporting,
- and explicit architecture-decision framing when speedups are limited.

### Risk 4: Scope Drift Into Phase 4

Phase 3 could absorb broader VQE/VQA, gradient, or optimizer work.

Mitigation:

- keep the frozen Phase 2 workflow as the continuity anchor,
- and reject new Phase 4 surface growth as a Phase 3 blocker.

### Risk 5: Premature Optimality Claims

The project could overstate a density-aware heuristic before it is calibrated.

Mitigation:

- separate structural heuristic closure from benchmark-calibrated claims,
- and document the model boundary explicitly.

### Risk 6: Architecture Overreach

Channel-native fused noisy blocks could be treated as the minimum bar too early.

Mitigation:

- keep that branch deferred behind a benchmark-driven decision gate,
- and make the native noisy partitioned runtime the minimum Phase 3 baseline.

## 15. Decision Gates

### DG-1: Phase 3 Completion Gate

Question:

- does the native Phase 3 backend accept noisy mixed-state planner inputs,
  execute the mandatory partitioned runtime, preserve exact semantics, and
  produce a benchmark package strong enough to support Paper 2?

If no:

- Phase 3 is not complete,
- even if partial planning or representation work exists.

### DG-2: Native Baseline Versus Channel-Native Follow-On

Question:

- do the mandatory benchmarks show that the native noise-aware partitioned
  runtime plus limited fused execution is sufficient for the main Phase 3
  scientific claim?

If yes:

- keep the native Phase 3 architecture as the main path,
- and treat channel-native fusion as later work.

If no:

- use the required benchmark and profiling package to justify a dedicated
  follow-on branch for more invasive channel-native fusion.

### DG-3: Handoff To Phase 4

Question:

- are the remaining limitations primarily about broader workflow surface,
  gradients, and optimizer-facing features rather than unresolved Phase 3
  backend semantics or performance architecture?

If yes:

- proceed to Phase 4 broader noisy VQE/VQA work.

If no:

- the project still has unresolved Phase 3 backend debt.

## 16. Non-Goals

To avoid later ambiguity, the following are explicit non-goals of this phase:

- proving fully general noisy-block fusion,
- delivering broad new VQE/VQA APIs,
- solving density-backend gradients end to end,
- maximizing qubit count beyond the honest exact-regime paper narrative,
- and replacing the exact backend with approximate scaling methods.

## 17. Expected Outcome

At the end of Phase 3, the project should have:

- a native noise-aware partitioning contract for exact mixed-state circuits,
- an executable partitioned density runtime,
- at least one benchmarked real fused execution mode,
- a validated benchmark package showing where the approach helps and where it
  still falls short,
- a reproducible Paper 2 evidence bundle,
- and a clean handoff into Phase 4, where broader noisy workflows and optimizer
  science can build on the stabilized backend.

That is the minimum outcome required for Phase 3 to count as a meaningful
methods step toward the broader PhD objective.
