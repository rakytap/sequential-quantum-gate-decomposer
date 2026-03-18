# Density Matrix Planning ADRs

This document records the planning-stage architecture decisions for extending
SQUANDER's density-matrix work into a publishable PhD research program.

Scope:

- exact noisy simulation,
- density-matrix integration with training workflows,
- partitioning and gate fusion for density matrices,
- scientific trade-offs needed to maximize feasible, defensible research output.

Status legend:

- `Accepted`: recommended default path for the PhD critical path.
- `Deferred`: important alternative, but not on the shortest path to meaningful
  scientific results.
- `Rejected for now`: useful idea in principle, but not justified at the current
  project stage.

## Decision Summary

| ADR | Title | Status |
|---|---|---|
| ADR-001 | Prioritize exact noisy emulation and training integration over noisy circuit re-synthesis | Accepted |
| ADR-002 | Use adapter-based unitary-island partitioning with noise barriers as the first density-matrix fusion path | Accepted |
| ADR-003 | Introduce a density-matrix-aware partitioning cost model only after the correctness baseline is in place | Accepted |
| ADR-004 | Prioritize realistic local noise models over global whole-register noise as the main scientific target | Accepted |
| ADR-005 | Keep exact dense density matrices as the reference backend through the first three research phases | Accepted |
| ADR-006 | Expand gate coverage in a workload-driven order rather than chasing full `qgd_Circuit` parity immediately | Accepted |
| ADR-007 | Evaluate IR-first channel-native fusion only after a benchmark-driven decision gate | Deferred |
| ADR-008 | Treat stochastic trajectories and MPDO-style methods as later scaling branches, not the initial architecture | Deferred |

## ADR-001: Prioritize Exact Noisy Emulation And Training Integration Over Noisy Circuit Re-Synthesis

### Status

`Accepted`

### Context

The current codebase has two strong but distinct assets:

- a mature partitioning and fusion path for `qgd_Circuit` / `Gates_block`,
- a new exact density-matrix backend built around `DensityMatrix` and
  `NoisyCircuit`.

The density-matrix backend is already scientifically useful for exact noisy
simulation, validation against Qiskit Aer, and small-to-medium-scale mixed-state
experiments. By contrast, noisy circuit re-synthesis or noisy wide-circuit
optimization would require combining:

- exact noisy simulation,
- noisy cost functions,
- decomposition / compilation quality,
- and optimizer behavior

into one early research claim.

That is architecturally attractive, but scientifically risky.

### Decision

The primary research target is:

- exact noisy emulation,
- density-matrix integration into VQA / training workflows,
- and scalable methods that improve noisy training experiments.

Noisy circuit re-synthesis and noise-aware wide-circuit compilation are deferred
until the exact noisy training backend is mature.

### Rationale

- Exact noisy simulation is easier to validate than noisy compilation.
- The PhD theme is scalable training under realistic noise, not primarily noisy
  circuit compilation.
- The strongest early papers come from clean exactness and reproducible
  comparisons, not from combining too many moving parts.
- This path aligns with the existing `ARCHITECTURE.md`, `CHANGELOG.md`, and
  `RESEARCH_ALIGNMENT.md` trajectory, where backend integration and `Tr(H*rho)`
  are the next explicit milestones.

### Consequences

- Early density-matrix work should be evaluated on simulation and training tasks,
  not on noisy circuit-size reduction claims.
- `Wide_Circuit_Optimization` remains a useful conceptual reference, but not the
  first integration target for density matrices.
- Performance work is judged by how much it improves exact noisy training
  workflows, not only by raw simulator throughput on synthetic circuits.

### Alternatives Considered

- **Immediate noisy circuit re-synthesis**: deferred because it creates a much
  broader and harder-to-defend scientific claim too early.

## ADR-002: Use Adapter-Based Unitary-Island Partitioning With Noise Barriers As The First Density-Matrix Fusion Path

### Status

`Accepted`

### Context

The current partitioning subsystem is built around `qgd_Circuit` and
state-vector fusion:

- the planner lives in `squander/partitioning`,
- the runtime fusion happens in `Gates_block::apply_to()`,
- the current cost model is state-vector-oriented,
- and `NoisyCircuit` currently executes sequentially.

At the same time, the density-matrix backend already has the key local primitive
needed for fused unitary execution:

- `DensityMatrix::apply_local_unitary(...)`.

Noise channels are not unitary and therefore do not naturally fit the existing
fusion path.

### Decision

The first density-matrix partitioning/fusion implementation should:

- keep `NoisyCircuit` as the density-matrix public API,
- treat noise operations as explicit fusion barriers,
- partition only contiguous unitary islands,
- adapt those islands into the existing circuit/gate partitioning machinery,
- and execute each fused unitary island through a density-matrix local-unitary
  block operation.

### Rationale

- This reuses the strongest current assets with the least architectural risk.
- It preserves exact semantics because noise order is unchanged.
- It gives a clean validation path against the current sequential
  `NoisyCircuit` executor and Qiskit Aer.
- It aligns with the most scientifically feasible early claim:
  "partitioning and fusion accelerate exact density-matrix simulation without
  changing results."

### Consequences

- Noise-dense circuits may fragment into many small unitary islands, limiting
  speedup.
- The first performance gains will be strongest for circuits where local noise is
  realistic but not inserted after every elementary gate.
- The architecture remains compatible with later channel-native fusion, but does
  not require it up front.

### Alternatives Considered

- **Channel-native fused CPTP blocks from the start**: deferred to ADR-007.
- **No partitioning until full backend integration**: rejected because it delays
  a strong Phase 3 methods paper without scientific benefit.

## ADR-003: Introduce A Density-Matrix-Aware Partitioning Cost Model Only After The Correctness Baseline Is In Place

### Status

`Accepted`

### Context

The current partitioning cost model in `squander/partitioning/tools.py` is
state-vector-specific. It scales with `2^n`, assumes one-sided gate application,
and is currently used by `ilp-fusion` and `ilp-fusion-ca` to optimize
state-vector execution.

Directly reusing that objective for density matrices would be scientifically and
architecturally misleading because density-matrix evolution has different:

- state size,
- arithmetic intensity,
- cache behavior,
- and left/right application costs.

### Decision

The project should not claim density-aware optimal partitioning until there is a
separate, benchmark-calibrated density-matrix cost model.

Implementation sequence:

1. establish the density-matrix correctness baseline,
2. implement unitary-island fusion with simple structural heuristics,
3. then introduce a density-matrix-aware cost model and retune
   `ilp-fusion` / `ilp-fusion-ca` for density execution.

### Rationale

- Correctness-first is the safest scientific strategy.
- Early benchmark calibration is more trustworthy than premature analytic claims.
- The density-matrix crossover points for useful block sizes will almost
  certainly differ from the state-vector ones.

### Consequences

- Phase 2 claims should center on integration and exactness, not "optimal"
  density partitioning.
- Phase 3 becomes the natural home for a density-aware methods paper.
- Existing state-vector partitioning can still be reused structurally before the
  density-specific objective is finalized.

### Alternatives Considered

- **Immediate analytic port of the state-vector FLOP model**: rejected for now
  because it would likely overstate accuracy and understate validation needs.

## ADR-004: Prioritize Realistic Local Noise Models Over Global Whole-Register Noise As The Main Scientific Target

### Status

`Accepted`

### Context

The current density-matrix module already supports:

- depolarizing noise,
- amplitude damping,
- phase damping.

However, whole-register depolarizing noise is scientifically weak as the main
workload model for a PhD focused on realistic noisy training. It also interacts
poorly with partitioning because a global channel naturally defeats local block
structure.

### Decision

The primary scientific noise targets should be local, device-motivated models:

- single- and few-qubit depolarizing noise,
- amplitude damping / generalized amplitude damping,
- phase damping / dephasing,
- coherent over-rotation or coherent unitary error,
- shot noise and readout noise,
- and later calibration-aware noise models.

Whole-register depolarizing noise remains useful as:

- a baseline,
- a sanity-check model,
- and a worst-case stress test,

but not as the main research workload.

### Rationale

- Realistic local noise makes the research more publishable and more relevant to
  noisy VQA training.
- Local channels preserve meaningful block structure and therefore make
  density-matrix partitioning/fusion scientifically worth studying.
- Unital versus non-unital local noise is directly relevant to the trainability
  questions in the PhD plan.

### Consequences

- Noise-model expansion should be guided by training experiments, not only by
  simulator completeness.
- Benchmarks and papers should avoid over-relying on whole-register depolarizing
  examples unless they are clearly labeled as baselines.

### Alternatives Considered

- **Keep global depolarizing as the dominant model**: rejected because it weakens
  both scientific relevance and partitioning usefulness.

## ADR-005: Keep Exact Dense Density Matrices As The Reference Backend Through The First Three Research Phases

### Status

`Accepted`

### Context

Exact dense density matrices scale poorly, but they provide:

- exact mixed-state semantics,
- clean validation against reference simulators,
- and a trustworthy scientific anchor for later approximate methods.

For a PhD centered on realistic noisy training, exactness is especially valuable
because it disentangles noise effects from approximation error.

### Decision

Exact dense density-matrix simulation is the reference backend through:

- Phase 1: exact mixed-state foundation,
- Phase 2: noisy training integration,
- Phase 3: density-aware partitioning/fusion and acceleration.

Approximate scaling methods are allowed later, but only after they can be
benchmarked against this exact reference on overlapping problem sizes.

### Rationale

- Early papers are stronger when they can claim exactness.
- Exact results provide a trustworthy baseline for optimizer and trainability
  studies.
- This keeps later scaling branches scientifically anchored instead of turning
  the project into an approximation study too early.

### Consequences

- The early qubit range is constrained, but scientifically defensible.
- Training studies in the exact regime should be preferred to weaker,
  larger-scale approximate claims until the reference backend is mature.
- Later approximate methods must be benchmarked against exact density-matrix
  results where possible.

### Alternatives Considered

- **Adopt approximate methods before exact integration is complete**: rejected
  because it weakens early scientific claims and complicates validation.

## ADR-006: Expand Gate Coverage In A Workload-Driven Order Rather Than Chasing Full `qgd_Circuit` Parity Immediately

### Status

`Accepted`

### Context

`NoisyCircuit` currently exposes a useful but narrower gate set than
`qgd_Circuit`. Full parity would be convenient, but it is not required to reach
publishable Phase 2 and Phase 3 results if the supported set is aligned with the
target VQA workloads.

### Decision

Gate coverage should be expanded in the following order:

1. retain and thoroughly validate the existing single-qubit and controlled-gate
   subset,
2. prioritize gates that are common in noisy VQA workloads and partitioning
   experiments,
3. add less common gates only when required by benchmark suites or applications.

Priority expansion candidates:

- `RZZ` and closely related entangling gates for chemistry / hardware-efficient
  ansatze,
- additional controlled rotations or two-qubit parametrized gates if benchmark
  circuits require them,
- broader gate parity only after the training pipeline is stable.

### Rationale

- Publishable work needs representative scope, not immediate total scope.
- Workload-driven expansion is faster and easier to validate.
- It keeps the critical path focused on noisy training and density-aware
  acceleration.

### Consequences

- Some gate families will remain unsupported in early density partitioning work.
- Benchmark suites must be chosen to match the supported gate set until coverage
  expands.

### Alternatives Considered

- **Immediate full gate parity with `qgd_Circuit`**: rejected for now because it
  delays more important scientific milestones.

## ADR-007: Evaluate IR-First Channel-Native Fusion Only After A Benchmark-Driven Decision Gate

### Status

`Deferred`

### Context

A more ambitious long-term design would introduce a shared IR with:

- explicit qubit support,
- unitary and channel traits,
- and fused channel blocks represented as Kraus bundles, PTMs, or Liouville-space
  superoperators.

This is a powerful research direction, but it is significantly more invasive
than adapter-based unitary-island fusion.

### Decision

Channel-native fusion is deferred until after the project has:

- a working exact noisy training backend,
- unitary-island density fusion,
- and benchmark evidence that noise barriers are the main remaining bottleneck.

Decision gate:

- if Phase 3 benchmarks show that barrier costs dominate realistic target
  workloads, prototype channel-native fusion as a dedicated research branch;
- otherwise, keep it as a future extension rather than the main architecture.

### Rationale

- This protects the PhD critical path.
- It avoids spending a large amount of engineering effort before there is strong
  benchmark evidence that the extra complexity is needed.
- It keeps the project aligned with publishable results at each phase.

### Consequences

- Early papers will not claim fully fused noisy partitions.
- The codebase remains open to later IR-first work without depending on it now.

### Alternatives Considered

- **Make channel-native fusion the main architecture immediately**: rejected for
  now due to implementation risk and unclear early scientific payoff.

## ADR-008: Treat Stochastic Trajectories And MPDO-Style Methods As Later Scaling Branches, Not The Initial Architecture

### Status

`Deferred`

### Context

If the project eventually needs to push beyond the practical range of exact dense
density matrices, two major approximate directions become attractive:

- stochastic trajectory / Monte Carlo wavefunction methods,
- tensor-based mixed-state methods such as MPDOs.

Both are scientifically relevant, but they are not the best starting point for
the current project because the exact backend is still becoming integrated into
training workflows.

### Decision

Trajectory and MPDO-style methods are deferred until:

- the exact density-matrix backend is fully benchmarked and integrated,
- and the project has identified the exact workloads that remain out of reach.

They should be introduced as later scaling branches and benchmarked against the
exact backend on overlapping problem sizes.

### Rationale

- Exact density matrices provide the best validation anchor.
- Later approximate branches can be scientifically stronger if they are justified
  by clear scaling limits in the exact regime.
- This preserves a coherent thesis narrative: exact noisy training first, then
  principled scaling.

### Consequences

- The initial architecture stays simpler.
- Large-scale approximate methods remain available as future publications or
  extensions rather than immediate core dependencies.

### Alternatives Considered

- **Lead with trajectory or MPDO methods**: rejected for now because it weakens
  the early exactness-centered scientific story.
