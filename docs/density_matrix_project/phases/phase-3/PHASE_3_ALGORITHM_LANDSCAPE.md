# Phase 3 Algorithm Landscape

This note elaborates the compact algorithm-selection overview in
`DETAILED_PLANNING_PHASE_3.md`.

Purpose:

- summarize which partitioning and fusion approaches currently look strongest
  for exact noisy mixed-state simulation,
- separate literature-backed options from Phase 3 novelty opportunities,
- assess how the existing SQUANDER state-vector partitioning stack transfers to
  noisy circuits,
- and classify what belongs in baseline Phase 3, optional Phase 3 extension
  work, and later follow-on branches.

Interpretation rule:

- `DETAILED_PLANNING_PHASE_3.md` remains the phase contract and source of truth,
- this note is an interpretive planning aid for algorithm choice and research
  positioning,
- and `REFERENCES.md` remains the planning source of truth for curated formal
  citations.

## 1. Current Public Research Signal

### 1.1 Areas With Strong Existing Literature

The public literature is already strong in the following areas:

- state-vector partitioning and planner design, especially through `TDAG`,
  `GTQCP`, and `QGo`,
- gate-fusion methods for state-vector simulation, especially the control-aware
  perspective in `QMin` and the runtime-oriented fusion viewpoint in Nguyen et
  al.,
- exact density-matrix and HPC runtime design, especially the mixed-state
  systems context represented by Li et al., Doi and Horii, `QuEST`, `Qulacs`,
  and related references,
- newer multilevel and hypergraph-based partitioning work such as `A Multilevel
  Framework for Partitioning Quantum Circuits`,
- and noise-aware or fidelity-aware partitioning in distributed or NISQ
  settings, including `Fidelipart`.

### 1.2 Areas With Growing But Indirect Literature

There is also meaningful but less directly aligned literature on:

- noisy simulation acceleration via computational reuse, especially `TQSim`,
- hardware-aware exact noisy density-matrix runtime optimization, especially
  `TANQ-Sim`,
- and Liouville-space, superoperator, and channel-oriented representations that
  are relevant to aggressive noisy fusion, even when they are not yet presented
  as a direct Phase 3-style partitioning architecture.

### 1.3 Areas That Still Look Comparatively Sparse

The public literature still looks comparatively sparse on one specific point
that matters most for Phase 3:

- native exact noisy mixed-state partition planners in which noise operations
  are first-class planner inputs,
- partition descriptors preserve explicit gate/noise order and parameter
  structure,
- and the resulting runtime executes partitioned noisy mixed-state circuits end
  to end with real fused execution rather than only planner representation.

This gap matters because it means Phase 3 can still produce a methods
contribution even if it reuses ideas from the state-vector partitioning
literature. The novelty comes from the noisy mixed-state contract and runtime,
not only from inventing a planner family from scratch.

## 2. Main Algorithm Families Relevant To Phase 3

### 2.1 Noise-Aware DAG Partitioning

Core idea:

- represent the circuit as a dependency DAG,
- promote both gate and noise operations to explicit nodes,
- and choose partitions that respect execution order and a bounded qubit support
  while preserving exact noisy semantics.

Why it looks strong:

- it is the closest conceptual extension of the current SQUANDER planning stack,
- it matches the Phase 3 contract that noise operations become first-class
  planner inputs,
- and it provides the safest correctness-first baseline for exact noisy mixed-
  state simulation.

Best fit for Phase 3:

- very strong,
- especially as the baseline planner family for the first executable noisy
  partitioned runtime.

### 2.2 ILP-Based Partition Selection

Core idea:

- enumerate or characterize candidate parts,
- then solve partition selection with an objective that balances support size,
  dependency order, and expected execution cost.

Why it looks strong:

- it gives the cleanest route to a paper-quality optimization story for small
  and medium exact workloads,
- it supports explicit ablations between heuristic and optimization-guided
  partition selection,
- and it is particularly attractive once the cost model is recalibrated for
  density-matrix execution rather than state-vector FLOPs.

Main caveat:

- an ILP that reuses the current state-vector cost model unchanged is not yet a
  density-aware Phase 3 result,
- so the planner structure transfers well, but the objective does not.

Best fit for Phase 3:

- very strong,
- especially as the strongest optimization-focused baseline once the noisy DAG
  semantics and runtime are already working.

### 2.3 Multilevel And Hypergraph Partitioning

Core idea:

- coarsen the circuit graph or hypergraph,
- optimize partitions at multiple levels,
- then refine.

Why it looks promising:

- recent work suggests better scalability and stronger search quality than
  single-level heuristics on larger circuits,
- noise-aware weighting can be introduced at the hyperedge level,
- and this family is a natural next heuristic when local greedy or direct DAG
  methods start to plateau.

Main caveat:

- most current work in this family is framed around distributed execution or
  hardware fidelity rather than exact dense mixed-state simulation,
- so it is relevant inspiration, but not a drop-in exact noisy mixed-state
  algorithm.

Best fit for Phase 3:

- strong optional extension,
- especially if the simpler DAG and ILP baselines prove too myopic on the
  structured benchmark families.

### 2.4 Computational Reuse And Dynamic Subcircuit Reuse

Core idea:

- partition a noisy circuit into reusable subcircuits,
- cache or reuse repeated intermediate results across many noisy runs or many
  repeated evaluations.

Why it looks promising:

- VQE-like workloads naturally repeat the same circuit family many times with
  different parameters,
- `TQSim` shows that dynamic subcircuit reuse can provide real benefit in noisy
  simulation settings,
- and this line of work aligns well with the repeated-evaluation character of
  the Phase 2 continuity workflow.

Main caveat:

- this family is not a direct replacement for native noisy partition planning,
- because its main gain often comes from workload repetition and reuse patterns
  rather than from the partition descriptor and fused-runtime contract itself.

Best fit for Phase 3:

- useful optional extension,
- but not the defining baseline for the main Paper 2 claim.

### 2.5 Fusion Families

#### Unitary-Island Fusion

Core idea:

- fuse consecutive unitary regions between noise operations,
- preserve noise boundaries exactly,
- and execute those fused unitary regions inside the noisy partitioned runtime.

Why it looks strong:

- it is the safest exact baseline,
- it satisfies the Phase 3 requirement for at least one real fused execution
  mode,
- and it fits naturally with the existing SQUANDER fusion lineage.

Best fit for Phase 3:

- baseline required fused path.

#### Control-Aware Unitary Fusion

Core idea:

- favor fused blocks whose control and target structure gives a better effective
  execution cost.

Why it looks strong:

- `QMin` and the existing `ilp-fusion-ca` logic point to real value in
  control-aware grouping,
- and this may remain beneficial in density-matrix execution when the support
  stays small and the runtime uses a local-kernel model.

Main caveat:

- the cost model must be recalibrated for mixed-state execution,
- because density-matrix cost growth is harsher than state-vector cost growth.

Best fit for Phase 3:

- very strong,
- and one of the best novelty anchors when adapted from the current SQUANDER
  stack.

#### Local Channel Or Kraus Fusion

Core idea:

- combine very small-support gate-plus-noise subsequences into one local noisy
  update,
- typically by composing the local channel exactly.

Why it looks promising:

- it could reduce repeated memory traffic in noise-dense local regions,
- and it stays closer to the exact noisy semantics than large-scale
  superoperator blocks.

Main caveat:

- support size grows quickly,
- and the implementation gets harder as soon as the fused region is no longer
  very local.

Best fit for Phase 3:

- optional extension inside the baseline runtime,
- as long as the support remains very small and the semantic contract stays
  simple.

#### Superoperator Or Liouville-Space Fusion

Core idea:

- represent fused gate-plus-noise regions directly as channels or
  superoperators,
- then apply those fused noisy blocks to vectorized or otherwise transformed
  mixed-state objects.

Why it looks powerful:

- it is the clearest route to aggressive noisy-block fusion when noise density is
  high,
- and it may ultimately outperform conservative unitary-island fusion on some
  workloads.

Main caveat:

- it is architecturally invasive,
- it pushes support growth and representation size quickly,
- and it goes beyond the minimum Phase 3 closure condition already frozen in the
  planning set.

Best fit for Phase 3:

- benchmark-driven follow-on branch,
- not the baseline required architecture.

## 3. Applicability Of The Existing SQUANDER State-Vector Stack

The most important planning conclusion is:

- the current SQUANDER partitioning algorithms are structurally reusable for
  Phase 3,
- but they are not Phase 3-ready unchanged because their node semantics and
  cost model remain state-vector-oriented.

### 3.1 What Transfers Well

The following elements transfer well:

- dependency-graph construction and topological reasoning,
- bounded-support partition selection,
- candidate-group enumeration,
- heuristic-versus-ILP comparison structure,
- and the notion that fusion should be evaluated as a runtime concern rather
  than only as a symbolic grouping concern.

### 3.2 What Does Not Transfer Unchanged

The following elements do not transfer unchanged:

- the current state-vector FLOP model,
- any logic that assumes noise sits outside the planner as a boundary-only
  condition,
- any interpretation that support size in mixed-state execution behaves like the
  state-vector case,
- and any benchmark claim that does not validate directly against the sequential
  exact density baseline.

## 4. Assessment Of Each Current SQUANDER Strategy

| Strategy | Transferability to noisy circuits | Main change needed | Phase 3 status |
|---|---|---|---|
| `kahn` | high | promote noise operations to first-class nodes and add density-aware evaluation of partitions | design-space heuristic baseline |
| `tdag` | high | reinterpret dependency groups on a noisy operation DAG and benchmark against noisy workloads | design-space heuristic baseline |
| `gtqcp` | high | same as `tdag`, with workload calibration rather than state-vector-only intuition | design-space heuristic baseline |
| `ilp` | high | replace the state-vector objective with a density-aware benchmark-informed objective | design-space optimization baseline |
| `ilp-fusion` | medium to high | redefine fusion admissibility and cost for exact noisy mixed-state execution | design-space strong candidate |
| `ilp-fusion-ca` | high | keep the control-aware idea, but recalibrate it for density-matrix runtime cost | design-space especially strong candidate |
| `qiskit` | medium | use only as an external comparison prepartition strategy | comparison baseline |
| `qiskit-fusion` | medium | use only as a comparison baseline, not as the main claim surface | comparison baseline |
| `bqskit-Quick` / `Scan` / `Greedy` / `Cluster` | medium | use as external heuristic comparisons rather than native noisy mixed-state planners | comparison baseline |

This table is the broader Phase 3 planner design space, not a statement that
every listed strategy is already delivered in the current Task 5 baseline. The
implemented Task 5 calibration surface is narrower: it currently calibrates a
bounded family of auditable `max_partition_qubits` span-budget settings on the
existing noisy planner surface, while the adapted `kahn` / `tdag` / `gtqcp` /
`ilp` / `ilp-fusion` / `ilp-fusion-ca` families remain valid design-space and
comparison references until they are separately implemented for the noisy
planner path.

## 5. Recommended Phase 3 Baseline Stack

The most coherent Phase 3 baseline stack is:

### 5.1 Planner Input

- canonical noisy mixed-state operation DAG aligned with `NoisyCircuit`,
- exact lowering from the frozen Phase 2 continuity workflow into that surface.

### 5.2 Planner Ladder

The recommended implementation and evaluation ladder is:

1. `kahn` as the simplest correctness-first heuristic baseline,
2. `tdag` and `gtqcp` as stronger structural heuristics,
3. `ilp` and adapted `ilp-fusion` / `ilp-fusion-ca` as the optimization-guided
   baseline,
4. multilevel or hypergraph extensions only if the simpler baselines plateau.

Current Task 5 implementation finding:

- the delivered benchmark-facing calibration surface is narrower than the full
  longer-horizon ladder above,
- the current supported Task 5 result calibrates auditable span-budget
  candidates on the existing noisy planner surface,
- and the broader adapted `kahn` / `tdag` / `gtqcp` / `ilp` / `ilp-fusion` /
  `ilp-fusion-ca` family should currently be read as Phase 3 design-space or
  comparison context rather than as already-delivered noisy planner variants.

### 5.3 Required Fused Runtime Baseline

- unitary-island fusion inside noisy partitions is the most defensible minimum
  fused path,
- with control-aware fusion logic as the strongest near-term enhancement,
- and local channel fusion considered only where support remains very small and
  exact semantics are easy to validate.

Current Task 4 implementation findings support this choice. The first delivered
real fused path is exactly this conservative descriptor-local unitary-island
baseline, realized through the density backend's local-unitary primitive rather
than through a broader channel-native architecture. The resulting benchmark
signal currently favors the diagnosis branch of the Phase 3 performance rule:
representative structured workloads preserve exact semantics with real fused
coverage, but the current baseline remains slower and does not yet reduce peak
memory, which strengthens the case for benchmark-driven follow-on optimization
instead of premature architecture escalation.

### 5.4 Validation Backbone

- sequential `NoisyCircuit` remains the internal exact oracle,
- Qiskit Aer remains the external exact reference on the required microcases and
  the current 4-qubit continuity slice,
- and every planner or fusion claim is judged against the frozen noisy semantics
  before performance claims are interpreted.

Current Task 6 implementation findings now make this backbone concrete. The
shared `benchmarks/density_matrix/correctness_evidence/` pipeline emits one
machine-reviewable correctness package over:

- `25` counted supported cases,
- a bounded external slice of `4` Qiskit Aer cases,
- explicit runtime-path and fused-region classifications on the positive surface,
- and `17` explicit unsupported-boundary cases across planner-entry,
  descriptor-generation, and runtime-stage layers.

This means the validation backbone is no longer only conceptual. It is now an
implemented artifact bundle family under
`benchmarks/density_matrix/artifacts/phase3_task6/` plus a rerunnable
`task6_validation_pipeline.py` entry point.

Current Task 7 implementation findings now extend that backbone into the
benchmark layer. The shared
`benchmarks/density_matrix/performance_evidence/` pipeline emits one
machine-reviewable benchmark package over:

- `34` counted supported benchmark cases,
- `4` continuity-anchor cases,
- `30` structured benchmark cases across the required families, sizes,
  noise-placement patterns, and rerun-sensitive seed slices,
- `6` representative review cases, one primary-seed sparse structured case for
  each required family and size,
- `0` positive-threshold pass cases,
- `6` diagnosis-only representative cases,
- and carry-forward of the `17` explicit Task 6 boundary cases into the
  benchmark-summary layer.

This means the benchmark package is no longer only a planning concept either. It
is now an implemented artifact bundle family under
`benchmarks/density_matrix/artifacts/phase3_task7/` plus a rerunnable
`task7_validation_pipeline.py` entry point. The delivered baseline therefore
supports diagnosis-grounded benchmark closure directly, not only the earlier
Task 4 fused-runtime observation.

## 6. Phase Boundary Classification

### 6.1 Baseline Phase 3 Content

The following should be treated as clear baseline Phase 3 content:

- canonical noisy mixed-state planner inputs,
- noise-aware DAG partitioning,
- benchmark-calibrated planning on the existing noisy planner surface through a
  bounded auditable candidate family,
- with adapted `kahn`, `tdag`, `gtqcp`, `ilp`, `ilp-fusion`, and
  `ilp-fusion-ca` retained as design-space or comparison references until they
  are separately implemented on the noisy planner path,
- unitary-island fusion as the minimum real fused execution path,
- density-aware benchmark-calibrated planning,
- and comparison against the sequential density baseline and Qiskit Aer.

### 6.2 Strong Optional Phase 3 Extensions

The following are strong optional Phase 3 extensions when benchmark evidence
justifies them:

- multilevel or hypergraph partitioning,
- computational reuse for repeated noisy evaluations,
- local channel or Kraus fusion on very small support,
- and profiler-driven kernel tuning that materially affects the benchmark
  outcome.

### 6.3 Beyond Baseline Phase 3

The following should remain outside the baseline Phase 3 closure claim:

- fully channel-native fused noisy blocks,
- Liouville-space or superoperator fusion as the primary architecture,
- approximate scaling methods such as trajectories or MPDOs,
- and broader noisy VQE/VQA feature growth or density-backend gradients.

## 7. Main Planning Conclusion

The main planning conclusion is:

- yes, adapted versions of the current SQUANDER state-vector partitioning
  algorithms are part of Phase 3,
- and that adaptation is likely one of the most novel aspects of the phase,
- because the novelty is not just "use TDAG again," but "make noisy mixed-state
  circuits native planner objects with exact semantic-preservation and real
  fused runtime execution."

Current implementation finding:

- the delivered Task 5 surface has not yet closed that entire broader algorithm
  family as separate noisy planner implementations,
- instead it currently calibrates a bounded span-budget candidate family on the
  existing noisy planner surface and keeps the broader ladder as explicit
  design-space context,
- and the supported Task 5 claim should therefore be phrased against the
  auditable selection rule and bounded candidate family rather than against one
  permanently frozen winning planner identity.

Current Task 7 benchmark finding:

- the emitted benchmark package is now aligned with that bounded Task 5 claim
  surface rather than with the longer-horizon planner ladder,
- the current diagnosis closure therefore speaks about the delivered
  `span_budget_q2` baseline plus its measured sensitivity surface,
- and the broader adapted `kahn` / `tdag` / `gtqcp` / `ilp` / `ilp-fusion` /
  `ilp-fusion-ca` ladder should still be read as design-space or follow-on
  context rather than as already-benchmarked noisy-planner implementations.

The best baseline therefore is not to invent an entirely unrelated planner
family first. It is to:

- start from the existing SQUANDER partitioning structure,
- promote noise operations to first-class planner inputs,
- redefine the cost model around exact noisy mixed-state execution,
- deliver conservative real fusion through unitary islands,
- and reserve fully channel-native fusion for the benchmark-driven follow-on
  branch if the baseline still leaves the dominant bottleneck unresolved.
