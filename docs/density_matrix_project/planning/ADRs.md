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
| ADR-001 | Freeze Phase 2 and prioritize Phase 3 partitioning work over broader VQE/VQA growth | Accepted |
| ADR-002 | Make noise channels and density semantics first-class in partitioning and fusion | Accepted |
| ADR-003 | Introduce a noise-aware density-matrix partitioning cost model only after native noisy-circuit correctness is in place | Accepted |
| ADR-004 | Prioritize realistic local noise models over global whole-register noise as the main scientific target | Accepted |
| ADR-005 | Keep exact dense density matrices as the reference backend through the first four research phases | Accepted |
| ADR-006 | Expand gate coverage in a workload-driven order rather than chasing full `qgd_Circuit` parity immediately | Accepted |
| ADR-007 | Evaluate IR-first channel-native fusion only after a benchmark-driven decision gate | Deferred |
| ADR-008 | Treat stochastic trajectories and MPDO-style methods as later scaling branches, not the initial architecture | Deferred |

## ADR-001: Freeze Phase 2 And Prioritize Phase 3 Partitioning Work Over Broader VQE/VQA Growth

### Status

`Accepted`

### Context

The current codebase has two strong but distinct assets:

- a mature partitioning and fusion path for `qgd_Circuit` / `Gates_block`,
- a new exact density-matrix backend built around `DensityMatrix` and
  `NoisyCircuit`.

Phase 2 now already delivers:

- backend selection for the canonical exact noisy workflow,
- exact `Re Tr(H*rho)` observable evaluation,
- one frozen supported noisy XXZ workflow contract,
- and publication-facing validation bundles for that scope.

The next tempting step would be to keep expanding the VQE/VQA surface
immediately with broader workflow support, gradients, and richer training-loop
features. At the same time, the project still needs a clearly defined Phase 3
methods milestone inside the partitioning/fusion subsystem. Noisy circuit
re-synthesis and noisy wide-circuit compilation remain even broader claims.

### Decision

The primary sequencing decision is:

- treat the Phase 2 canonical workflow as accomplished and frozen,
- make Phase 3 about noise-aware partitioning/fusion rather than new VQE/VQA
  surface growth,
- defer broader VQE/VQA features such as density-backend gradient routing and
  richer workflow support to Phase 4 or later,
- and keep noisy circuit re-synthesis / noise-aware wide-circuit compilation
  outside the near-term critical path.

### Rationale

- Freezing Phase 2 keeps the first major claim stable and publishable.
- It gives Phase 3 a clear methods target instead of mixing partitioning work
  with new workflow-surface promises.
- The PhD theme is still scalable training under realistic noise, but the next
  architectural bottleneck is inside partitioning/fusion rather than the
  already-accomplished minimal workflow integration.
- Noisy circuit compilation remains scientifically interesting, but it still
  creates a broader and harder-to-defend claim than the revised Phase 3 scope.

### Consequences

- Phase 3 benchmarks may still use training-relevant circuits, but they should
  not depend on new VQE/VQA features beyond the Phase 2 contract.
- Gradient routing, broader circuit-source support, and richer optimizer-facing
  workflows become explicit Phase 4 items.
- `Wide_Circuit_Optimization` remains a useful conceptual reference, but not the
  first integration target for density matrices.
- Performance work is judged by how much it improves exact noisy training
  workflows, not only by raw simulator throughput on synthetic circuits.

### Alternatives Considered

- **Continue expanding VQE/VQA features immediately after Phase 2**: rejected
  for now because it weakens the clarity of the Phase 3 partitioning/fusion
  milestone.
- **Immediate noisy circuit re-synthesis**: deferred because it creates a much
  broader and harder-to-defend scientific claim too early.

## ADR-002: Make Noise Channels And Density Semantics First-Class In Partitioning And Fusion

### Status

`Accepted`

### Context

The current partitioning subsystem is built around `qgd_Circuit` and
state-vector fusion:

- the planner lives in `squander/partitioning`,
- the runtime fusion happens in `Gates_block::apply_to()`,
- the current cost model is state-vector-oriented,
- and `NoisyCircuit` currently executes sequentially.

The simplest port would treat noise operations as opaque barriers, partition
only contiguous unitary islands, and keep the planner effectively unitary-first.
That reuses existing infrastructure, but it is too narrow for the revised
Phase 3 goal where noise should be a natural part of any circuit at any level
and the partitioning logic itself must be noise-aware.

### Decision

The Phase 3 partitioning/fusion contract should:

- keep `NoisyCircuit`-style noisy mixed-state semantics visible to the planner,
- treat noisy mixed-state circuits as valid partitioning inputs,
- retain exact noise placement and execution ordering inside partition
  descriptors and runtime interfaces,
- make partitioning decisions noise-aware rather than state-vector-aware plus
  external barriers,
- require an executable partitioned runtime with at least one real fused
  execution mode on eligible substructures,
- and allow unitary-island execution as an internal optimization tactic without
  making it the definition of the Phase 3 problem.

### Rationale

- It matches the actual target workload, where noise can appear throughout a
  circuit rather than only between pre-existing unitary islands.
- It avoids architecturally misrepresenting noise as out-of-band metadata.
- It still gives a clean validation path against the current sequential
  `NoisyCircuit` executor and Qiskit Aer.
- It aligns the Phase 3 claim with the project requirement that partitioning
  itself be noise-aware.

### Consequences

- Phase 3 becomes a broader planner/runtime change than a barrier-only adapter
  layer.
- Early implementations may still exploit unitary substructure, but the planner
  must understand noise explicitly.
- Planner-only representation is not enough to close the phase; there must be an
  executable noisy-circuit path with real fused execution on at least some
  representative cases.
- Fully channel-native fused noisy blocks remain optional follow-on work rather
  than the minimum Phase 3 threshold.
- The architecture remains compatible with later channel-native fusion, but does
  not require the most invasive form up front.

### Alternatives Considered

- **Boundary-only unitary-island partitioning with noise barriers**: rejected as
  the phase-defining contract because it treats noise as an external exception
  instead of part of the circuit model.
- **Channel-native fused CPTP blocks from the start**: deferred to ADR-007.
- **No partitioning until full backend integration**: rejected because it delays
  a strong Phase 3 methods paper without scientific benefit.

## ADR-003: Introduce A Noise-Aware Density-Matrix Partitioning Cost Model Only After Native Noisy-Circuit Correctness Is In Place

### Status

`Accepted`

### Context

The current partitioning cost model in `squander/partitioning/tools.py` is
state-vector-specific. It scales with `2^n`, assumes one-sided gate
application, is blind to noise placement, and is currently used by
`ilp-fusion` and `ilp-fusion-ca` to optimize state-vector execution.

Directly reusing that objective for density matrices would be scientifically and
architecturally misleading because noisy density-matrix evolution has different:

- state size,
- arithmetic intensity,
- cache behavior,
- left/right application costs,
- and structural effects from noise placement and channel density.

### Decision

The project should not claim density-aware or noise-aware optimal partitioning
until there is a separate, benchmark-calibrated density-matrix cost model.

Implementation sequence:

1. establish the native noisy-circuit correctness baseline,
2. implement noise-aware partitioning with simple structural heuristics,
3. then introduce a benchmark-calibrated density-matrix cost model and retune
   `ilp-fusion` / `ilp-fusion-ca` (or a successor planner) for density
   execution.

### Rationale

- Correctness-first is the safest scientific strategy.
- Early benchmark calibration is more trustworthy than premature analytic
  claims.
- The useful crossover points will differ from the state-vector ones because
  the planner must react to both mixed-state cost and noise structure.

### Consequences

- Phase 2 claims should center on integration and exactness, not "optimal"
  density partitioning.
- Phase 3 claims should center on native noisy-circuit partitioning plus
  benchmark-calibrated heuristics, not premature optimality language.
- AVX-level kernel tuning is optional support work in Phase 3, not a
  phase-defining success criterion by itself; only pursue it if profiling and
  benchmark evidence identify a material hotspot in the native noisy-circuit
  runtime.
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

## ADR-005: Keep Exact Dense Density Matrices As The Reference Backend Through The First Four Research Phases

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
- Phase 3: noise-aware partitioning/fusion plus optional benchmark-driven
  acceleration support,
- Phase 4: broader noisy VQE/VQA workflows and optimizer studies.

Approximate scaling methods are allowed later, but only after they can be
benchmarked against this exact reference on overlapping problem sizes.

### Rationale

- Early papers are stronger when they can claim exactness.
- Exact results provide a trustworthy baseline for broader VQE/VQA, optimizer,
  and trainability studies.
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
publishable Phase 2 and Phase 3 results if the supported set is aligned with
target noisy mixed-state benchmarks and later VQE/VQA workloads.

### Decision

Gate coverage should be expanded in the following order:

1. retain and thoroughly validate the existing single-qubit and controlled-gate
   subset,
2. prioritize gates that are common in Phase 3 partitioning benchmarks and later
   Phase 4 VQE/VQA workloads,
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
- It keeps the critical path focused on noise-aware partitioning first and noisy
  training expansion second.

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
than the planned Phase 3 noise-aware baseline.

### Decision

Channel-native fusion is deferred until after the project has:

- a working exact noisy training backend,
- a native noise-aware density partitioning baseline,
- and benchmark evidence that more native channel handling is the main remaining
  bottleneck.

Decision gate:

- if Phase 3 benchmarks show that the native noise-aware baseline is still
  bottlenecked by the lack of channel-native fusion, prototype channel-native
  fusion as a dedicated research branch;
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
