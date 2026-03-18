# Phase 2 ADRs

This document records the detailed architecture and scope decisions that apply
specifically to Phase 2.

It refines, but does not override, the upstream planning decisions in:

- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`

If this document appears to conflict with the upstream planning set, the conflict
must be resolved in favor of the upstream planning set unless the Phase 2
document explicitly states that it is tightening the scope for this phase only.

## Status Legend

- `Accepted`: part of the Phase 2 contract.
- `Deferred`: deliberately postponed beyond the core Phase 2 deliverable.
- `Rejected for Phase 2`: explicitly excluded from this phase.

## Phase 2 Decision Summary

| ADR | Title | Status |
|---|---|---|
| P2-ADR-001 | Use spec-driven Phase 2 documentation as the implementation contract | Accepted |
| P2-ADR-002 | Focus Phase 2 on exact noisy backend integration, not density-aware acceleration | Accepted |
| P2-ADR-003 | Keep exact dense density matrices as the Phase 2 scientific reference backend | Accepted |
| P2-ADR-004 | Prioritize realistic local noise support over global toy-noise emphasis | Accepted |
| P2-ADR-005 | Use workload-driven support and explicitly defer full gate parity | Accepted |
| P2-ADR-006 | Require publication-grade validation and benchmark evidence from the start | Accepted |
| P2-ADR-007 | Define success around one or more exact noisy workflows in the documented exact regime | Accepted |
| P2-ADR-008 | Defer partitioning, fusion, gradients, and approximate scaling to later phases | Deferred |
| P2-ADR-009 | Make backend selection explicit and fail unsupported combinations early | Accepted |
| P2-ADR-010 | Limit Phase 2 observables to exact Hermitian energy evaluation in the VQE path | Accepted |
| P2-ADR-011 | Use a partial HEA-first bridge into `NoisyCircuit` | Accepted |
| P2-ADR-012 | Freeze a workload-driven Phase 2 support matrix | Accepted |
| P2-ADR-013 | Anchor Phase 2 on noisy XXZ VQE in the exact regime | Accepted |
| P2-ADR-014 | Require a minimum Aer-centered benchmark package | Accepted |
| P2-ADR-015 | Use numeric exactness and workflow-readiness thresholds | Accepted |

## P2-ADR-001: Use Spec-Driven Phase 2 Documentation As The Implementation Contract

### Status

`Accepted`

### Context

Phase 2 sits at a transition point:

- Phase 1 delivered a working standalone density-matrix module,
- but the research program now needs an integrated backend suitable for noisy
  training workflows and Paper 1.

Without a clear contract, implementation can drift toward:

- undocumented assumptions,
- premature optimization work,
- or publication claims that exceed delivered scope.

### Decision

Phase 2 work must be guided by explicit documentation-first contracts covering:

- scope,
- non-goals,
- task goals,
- acceptance criteria,
- validation evidence,
- and publication claims.

The Phase 2 document set in `docs/density_matrix_project/phases/phase-2/` is the
working contract for the phase.

### Rationale

- Phase 2 is the first phase where the density backend must serve research
  workflows rather than only exist as infrastructure.
- A spec-first approach is necessary to keep implementation aligned with the
  broader PhD goals and Paper 1.
- The project already has enough complexity that undocumented intent would create
  rework later.

### Consequences

- Every major Phase 2 deliverable must map back to a documented acceptance
  criterion.
- Unsupported and deferred items are not accidental omissions; they are explicit
  decisions.
- Publication drafts can be prepared in parallel with implementation because the
  claims are specified ahead of time.

### Rejected Alternatives

- Rely on informal planning notes or implementation order alone.
- Treat the paper drafting as a downstream activity after implementation.

### Upstream Alignment

Refines:

- `ADR-001`
- `PLANNING.md` source-of-truth and milestone logic

## P2-ADR-002: Focus Phase 2 On Exact Noisy Backend Integration, Not Density-Aware Acceleration

### Status

`Accepted`

### Context

The upstream planning already separates:

- Phase 2 backend integration,
- and Phase 3 density-aware partitioning, fusion, and acceleration.

The current codebase still lacks:

- backend selection in the main workflow,
- `Tr(H*rho)` expectation values,
- and a documented bridge from current circuit representations into the density
  path.

These are more important to the research program than early acceleration work.

### Decision

The core Phase 2 result is exact noisy backend integration for training
workflows.

Phase 2 will not present density-aware partitioning, barrier-based fusion, or
other acceleration work as a core deliverable.

### Rationale

- Integration is the shortest path to scientific usefulness.
- Paper 1 is defined in the publication strategy as an integration paper, not an
  acceleration paper.
- Premature acceleration work would dilute the phase and complicate validation.

### Consequences

- Any Phase 2 performance discussion should remain secondary and bounded to
  baseline runtime characterization or workflow feasibility.
- The phase is complete when exact noisy workflows are usable and validated, not
  when they are maximally optimized.

### Rejected Alternatives

- Merge Phase 2 and Phase 3 into a single backend-plus-acceleration effort.
- Introduce density-aware partitioning before the backend is integrated into the
  target workflow.

### Upstream Alignment

Refines:

- `ADR-001`
- `ADR-002`
- `ADR-003`

## P2-ADR-003: Keep Exact Dense Density Matrices As The Phase 2 Scientific Reference Backend

### Status

`Accepted`

### Context

The scientific value of Phase 2 depends on exact noisy evaluation. Approximate
methods would increase scale, but they would also introduce approximation error
before the exact training path is fully established.

### Decision

Phase 2 treats exact dense density matrices as the reference backend for:

- observable evaluation,
- validation,
- benchmark comparison,
- and first noisy training workflows.

Approximate methods are not part of the Phase 2 core.

### Rationale

- Exactness makes Phase 2 claims much easier to defend.
- Paper 1 benefits more from trusted correctness than from pushing to larger but
  approximate sizes.
- The exact backend is already present and validated at the standalone level.

### Consequences

- The main exact operating regime should be documented honestly rather than
  overstated.
- Phase 2 benchmark and workflow choices must stay inside the exact regime.
- Later approximate scaling branches must be benchmarked against this reference.

### Rejected Alternatives

- Use trajectories, MPDOs, or other approximate paths as a Phase 2 substitute.
- Frame Phase 2 around scale expansion instead of exact noisy integration.

### Upstream Alignment

Refines:

- `ADR-005`

## P2-ADR-004: Prioritize Realistic Local Noise Support Over Global Toy-Noise Emphasis

### Status

`Accepted`

### Context

The upstream planning and ADRs already favor realistic local noise over
whole-register toy channels. Phase 2 needs this decision to be made concrete,
because the first noisy workflows and Paper 1 should not be built around weak
noise models that do not serve the PhD theme well.

### Decision

Phase 2 prioritizes local, training-relevant noise support:

- local depolarizing noise,
- local phase damping / dephasing,
- local amplitude damping,
- generalized amplitude damping if required by the first target workflow,
- and coherent unitary or over-rotation error if required by the first target
  workflow.

Whole-register depolarizing remains allowed as:

- a baseline,
- a regression check,
- or a stress-test model,

but not as the main scientific workload.

### Rationale

- The PhD goal is realistic noisy training, not toy-noise benchmarking.
- Local noise aligns better with the later optimizer and trainability studies.
- Local noise also creates a more meaningful path toward Phase 3 and beyond.

### Consequences

- Every requested Phase 2 noise model should be justified by a target workflow
  or publication need.
- Noise support should be documented as required, optional, or deferred.

### Rejected Alternatives

- Make whole-register depolarizing the default scientific model for Phase 2.
- Expand noise scope without a workflow-driven justification.

### Upstream Alignment

Refines:

- `ADR-004`

## P2-ADR-005: Use Workload-Driven Support And Explicitly Defer Full Gate Parity

### Status

`Accepted`

### Context

The density-matrix backend does not yet match the full gate surface of
`qgd_Circuit`. Full parity is valuable long term, but Phase 2 needs only enough
support to enable representative noisy workflows and strong validation.

### Decision

Phase 2 support should be defined by:

- the target noisy workflow,
- the benchmark suite required for validation,
- and the publication claim set for Paper 1.

Unsupported or deferred gates are acceptable if they are explicitly documented.

### Rationale

- Workload-driven support is scientifically sufficient and operationally faster.
- Full gate parity is a poor use of Phase 2 effort if it delays backend
  integration or validation.
- Clear unsupported-case behavior is better than implicit partial support.

### Consequences

- Phase 2 must include a support matrix, not just a list of current APIs.
- Workflow choice and benchmark choice must be compatible with the documented
  support surface.

### Rejected Alternatives

- Require full `qgd_Circuit` gate parity before declaring Phase 2 successful.
- Hide support limitations behind vague documentation.

### Upstream Alignment

Refines:

- `ADR-006`

## P2-ADR-006: Require Publication-Grade Validation And Benchmark Evidence From The Start

### Status

`Accepted`

### Context

Paper 1 is expected to be the first major standalone publication. If validation
and benchmark standards are added only after implementation, the resulting paper
is likely to be weaker, narrower, and harder to defend.

### Decision

Phase 2 validation and benchmark requirements are part of the core phase
contract, not optional polish.

At minimum, Phase 2 must define:

- the primary external reference backend,
- the target workload classes,
- the target noise classes,
- the exactness metrics,
- and the evidence needed to support Paper 1.

### Rationale

- The Phase 2 scientific value comes from exact noisy integration plus strong
  evidence, not from integration alone.
- This keeps implementation aligned with publication outcomes and PhD milestones.

### Consequences

- Qiskit Aer remains the required primary reference for validation.
- The benchmark suite must include training-relevant circuits, not only
  simulation kernels.
- Runtime and memory characterization remain important even though Phase 2 is not
  the acceleration phase.

### Rejected Alternatives

- Leave benchmark design until after implementation.
- Accept internal correctness evidence without a strong external comparison.

### Upstream Alignment

Refines:

- `PUBLICATIONS.md`, especially Phase 2 Paper 1 requirements
- `ADR-001`
- `ADR-004`

## P2-ADR-007: Define Success Around One Or More Exact Noisy Workflows In The Documented Exact Regime

### Status

`Accepted`

### Context

It is possible to complete technical backend work without proving that it
supports an actual noisy training workflow. That would make Phase 2 a weak
scientific milestone.

The current roadmap documents stability around roughly 10 qubits as a realistic
acceptance point for this phase.

### Decision

Phase 2 success is defined around at least one representative exact noisy
workflow that:

- uses the density backend,
- evaluates noisy observables correctly,
- runs end-to-end in the documented exact regime,
- and is reproducible enough to support Paper 1.

The exact regime should be described honestly and consistently, with about
10-qubit stability treated as the current acceptance anchor rather than a hard
theoretical limit.

### Rationale

- This prevents the phase from becoming a backend integration exercise detached
  from its research purpose.
- It directly supports the handoff to Phase 4 optimizer studies and Phase 5
  trainability experiments.

### Consequences

- Workflow selection becomes part of the Phase 2 contract.
- Benchmark and support choices must be aligned with that workflow.

### Rejected Alternatives

- Define completion purely in terms of backend hooks and API surface.
- Avoid specifying a realistic exact operating regime.

### Upstream Alignment

Refines:

- `CHANGELOG.md`
- `RESEARCH_ALIGNMENT.md`
- `PUBLICATIONS.md`

## P2-ADR-008: Defer Partitioning, Fusion, Gradients, And Approximate Scaling To Later Phases

### Status

`Deferred`

### Context

Several technically attractive branches sit immediately adjacent to Phase 2:

- density-aware partitioning,
- unitary-island fusion,
- channel-native fusion,
- gradient routing for density-matrix optimization,
- stochastic trajectories,
- MPDO-style methods,
- and hardware-specific scaling paths.

All of them are relevant to the full PhD, but not all of them belong in this
phase.

### Decision

The following are formally deferred beyond the core Phase 2 deliverable:

- density-aware partitioning and gate fusion,
- channel-native or superoperator fusion,
- full density-matrix gradient path,
- trajectory-based approximate scaling,
- MPDO-style approximate scaling,
- and major acceleration claims beyond baseline workflow viability.

### Rationale

- Deferral protects the clarity of Paper 1.
- These branches depend on a usable exact backend and therefore belong naturally
  to later phases.
- Forcing them into Phase 2 would increase risk and reduce focus.

### Consequences

- Phase 2 documents must call these items out explicitly as future work.
- Phase 3 becomes the first legitimate phase for claiming density-aware
  acceleration.
- Approximate methods should not be used to rescue a weak exact integration
  result.

### Rejected Alternatives

- Treat any of these deferred items as implicit stretch goals inside Phase 2.
- Use them to compensate for missing core Phase 2 integration work.

### Upstream Alignment

Refines:

- `ADR-002`
- `ADR-003`
- `ADR-007`
- `ADR-008`

## P2-ADR-009: Make Backend Selection Explicit And Fail Unsupported Combinations Early

### Status

`Accepted`

### Context

Phase 2 needs backend selection to be concrete enough for implementation and
benchmark interpretation.

Without a frozen contract, several ambiguities remain:

- whether density execution is opt-in or automatic,
- whether existing state-vector workflows change behavior by default,
- and whether unsupported density workflows silently fall back to the
  state-vector path.

Those ambiguities are especially risky for Paper 1, where a benchmark claim
must be tied to a clearly selected backend.

### Decision

Backend selection is an explicit workflow-level decision on
`qgd_Variational_Quantum_Eigensolver_Base` or an equivalent VQE-facing
configuration entry point.

Phase 2 allows exactly two backend modes:

- `state_vector`,
- `density_matrix`.

Additional contract details:

- `state_vector` remains the default when no backend is specified,
- `density_matrix` must be selected explicitly for exact noisy mixed-state
  workflow claims,
- no implicit `auto` mode is part of the Phase 2 contract,
- and unsupported density combinations fail with a hard pre-execution error
  rather than silently falling back.

### Rationale

- Keeping `state_vector` as the default preserves backward compatibility and
  limits migration risk.
- Requiring explicit `density_matrix` selection sacrifices some convenience, but
  it avoids scientifically ambiguous fallback behavior.
- Early hard errors are preferable to silently benchmarking the wrong backend.

### Consequences

- Phase 2 implementation must expose a clearly documented backend switch.
- Benchmarks and validation logs can state unambiguously which backend was used.
- Unsupported mixed-state requests become explicit scope-boundary outcomes
  rather than silent surprises.

### Rejected Alternatives

- Introduce an implicit `auto` mode that guesses when to switch backends.
- Make `density_matrix` the default before the support surface is stable.
- Silently fall back to `state_vector` when density execution is unsupported.

### Upstream Alignment

Refines:

- `P2-ADR-002`
- `P2-ADR-007`
- `RESEARCH_ALIGNMENT.md` Phase 2 acceptance wording

## P2-ADR-010: Limit Phase 2 Observables To Exact Hermitian Energy Evaluation In The VQE Path

### Status

`Accepted`

### Context

Phase 2 must freeze what `Tr(H*rho)` actually means at the workflow level.

The current VQE path already accepts a Hermitian sparse Hamiltonian, but the
project could still drift toward:

- an underspecified generic observable API,
- support claims that exceed the first benchmark family,
- or a mixed exact-plus-shot-measurement story that is too broad for this phase.

### Decision

Phase 2 observable support is limited to exact real-valued Hermitian energy
evaluation through:

- `E(theta) = Re Tr(H*rho(theta))`,
- using the existing Hermitian sparse-Hamiltonian input path of
  `qgd_Variational_Quantum_Eigensolver_Base`,
- with the canonical benchmark family anchored on XXZ Hamiltonians composed of
  `XX`, `YY`, `ZZ`, and local `Z` terms.

The following remain outside the core Phase 2 contract:

- arbitrary non-Hermitian observables,
- general POVMs,
- batched multi-observable APIs,
- and shot-noise or readout estimation as the main acceptance path.

### Rationale

- This matches the current VQE interface and keeps the integration target clear.
- Exact Hermitian energy evaluation is the minimum scientifically meaningful
  observable contract for noisy VQE in Phase 2.
- Deferring broader observable APIs keeps the phase focused and the validation
  story clean.

### Consequences

- Paper 1 claims center on exact noisy energy evaluation, not generic
  measurement infrastructure.
- Validation can use a precise reference comparison against Qiskit Aer.
- Later phases remain free to add richer measurement interfaces without
  retroactively changing the Phase 2 success definition.

### Rejected Alternatives

- Define a broad generic observable framework as part of the Phase 2 minimum.
- Restrict Phase 2 to diagonal or `ZZ`-only Hamiltonians.
- Blend exact observable validation with shot-noise estimation as one
  inseparable contract.

### Upstream Alignment

Refines:

- `P2-ADR-003`
- `P2-ADR-006`
- `PUBLICATIONS.md` Paper 1 observable requirement

## P2-ADR-011: Use A Partial HEA-First Bridge Into `NoisyCircuit`

### Status

`Accepted`

### Context

The density-matrix backend is structurally separate from the main VQE circuit
path. Phase 2 therefore needs an explicit bridge decision.

The main trade-off is:

- broad circuit compatibility versus
- a smaller bridge that is guaranteed for the first real workflow.

### Decision

The Phase 2 bridge is partial and HEA-first.

Mandatory bridge contract:

- source representation: the circuit generated by
  `qgd_Variational_Quantum_Eigensolver_Base.Generate_Circuit()` for the default
  `HEA` ansatz,
- target representation: `NoisyCircuit` plus `GateOperation` and ordered
  `NoiseOperation` insertion,
- unsupported operations: hard pre-execution error naming the first unsupported
  operation.

Optional Phase 2 extension:

- user-supplied `qgd_Circuit` or `Gates_block` inputs are allowed only when all
  operations lower cleanly to the documented required gate families.

Full `qgd_Circuit` parity is not part of the Phase 2 contract.

### Rationale

- The HEA-generated VQE path is the most defensible implementation anchor for
  Paper 1.
- A partial bridge keeps the phase bounded around a real workflow instead of an
  abstract compatibility target.
- Hard-error behavior is safer than silently dropping or rewriting unsupported
  operations.

### Consequences

- The bridge can be judged against a concrete workflow instead of an open-ended
  gate-compatibility list.
- Broader manual circuit reuse becomes an optional extension, not a blocking
  requirement.
- Documentation must clearly separate guaranteed bridge behavior from
  non-guaranteed future parity.

### Rejected Alternatives

- Require full `qgd_Circuit` parity before Phase 2 can start implementation.
- Permit silent partial lowering of unsupported circuits.
- Leave the bridge implicit and let implementation define support ad hoc.

### Upstream Alignment

Refines:

- `P2-ADR-005`
- `P2-ADR-008`
- `ARCHITECTURE.md` Phase 2 integration targets

## P2-ADR-012: Freeze A Workload-Driven Phase 2 Support Matrix

### Status

`Accepted`

### Context

The earlier Phase 2 ADRs already favored workload-driven support and realistic
local noise. What remained open was the exact support surface.

That support surface must balance:

- scientific relevance,
- implementation feasibility,
- and protection against scope inflation.

### Decision

Phase 2 freezes the following support matrix.

Required:

- gate families: `U3`, `CNOT`,
- noise models: local single-qubit depolarizing, local amplitude damping, local
  phase damping or dephasing.

Optional:

- additional gates already exposed by `NoisyCircuit` when a test or
  comparison microbenchmark genuinely needs them,
- whole-register depolarizing as a regression or stress-test baseline,
- generalized amplitude damping or coherent over-rotation only if a justified
  benchmark extension requires them.

Deferred:

- full `qgd_Circuit` gate parity,
- correlated multi-qubit noise,
- readout noise as a density-backend feature,
- calibration-aware noise,
- and non-Markovian noise.

### Rationale

- `U3` plus `CNOT` is enough to support the default HEA workflow cleanly.
- Mandatory local noise keeps Phase 2 aligned with the realistic-noise research
  theme.
- Making whole-register depolarizing optional preserves its value as a baseline
  while preventing it from becoming the main scientific target.

### Consequences

- Phase 2 implementation must likely extend Phase 1 by adding local
  single-qubit depolarizing support.
- Benchmarks and workflow claims must stay inside the frozen support surface.
- Any expansion beyond this matrix becomes an explicit scope decision rather
  than accidental feature creep.

### Rejected Alternatives

- Freeze no support matrix and let implementation discover the boundary later.
- Require immediate full gate parity.
- Treat whole-register depolarizing as sufficient for the main Paper 1 noise
  narrative.

### Upstream Alignment

Refines:

- `P2-ADR-004`
- `P2-ADR-005`

## P2-ADR-013: Anchor Phase 2 On Noisy XXZ VQE In The Exact Regime

### Status

`Accepted`

### Context

Phase 2 already needed at least one representative noisy workflow, but the
workflow anchor was still too generic.

Without a concrete anchor, the phase could still be evaluated against:

- a workflow that is too synthetic,
- a workflow that depends on unsupported features,
- or several half-supported workflows instead of one publishable one.

### Decision

The Phase 2 anchor workflow is noisy VQE ground-state estimation for a 1D XXZ
spin chain with local `Z` field using:

- `qgd_Variational_Quantum_Eigensolver_Base`,
- the default `HEA` ansatz,
- explicit local noise insertion,
- and exact energy evaluation through `Re Tr(H*rho)`.

Scale contract:

- 4 and 6 qubits must support full end-to-end workflow execution, including at
  least one reproducible optimization trace,
- 8 and 10 qubits must support benchmark-ready fixed-parameter evaluation,
- and 10 qubits is the acceptance anchor for the documented exact regime.

### Rationale

- This workflow is directly aligned with the current VQE interface.
- XXZ Hamiltonians are rich enough to stress the observable path while still
  fitting the frozen Phase 2 Hamiltonian family.
- Requiring full 10-qubit optimization would expand cost and risk without
  materially improving the Phase 2 scientific claim.

### Consequences

- Phase 2 success is judged against one concrete noisy VQE workflow, not a vague
  class of possible future workflows.
- Broader algorithm families remain possible later, but they are not needed to
  declare Phase 2 complete.
- The exact-regime claim is anchored honestly rather than being overstated.

### Rejected Alternatives

- Leave the workflow anchor generic.
- Split attention across several equal-priority workflow anchors.
- Require the full optimizer study that properly belongs to Phase 4.

### Upstream Alignment

Refines:

- `P2-ADR-007`
- `PUBLICATIONS.md` Paper 1 workflow requirement

## P2-ADR-014: Require A Minimum Aer-Centered Benchmark Package

### Status

`Accepted`

### Context

The benchmark requirement for Phase 2 was already accepted in principle, but the
minimum package was still open.

That left uncertainty around:

- which benchmark classes are mandatory,
- which baselines are required,
- and what reproducibility material must exist before Paper 1 claims are safe.

### Decision

Phase 2 requires the following minimum benchmark package.

Mandatory:

- Qiskit Aer density-matrix simulation as the primary external baseline,
- 1 to 3 qubit micro-validation circuits covering each required gate and noise
  contract,
- anchor XXZ noisy VQE circuits at 4, 6, 8, and 10 qubits,
- at least 10 fixed parameter vectors per mandatory workflow size,
- at least one reproducible 4- or 6-qubit optimization trace,
- metrics for absolute energy error, density validity, runtime, peak memory, and
  workflow completion,
- and a reproducibility bundle containing the Hamiltonian, ansatz, noise
  schedule, seeds, versions or commit, and raw results.

Optional:

- one additional simulator baseline if it materially strengthens the paper.

### Rationale

- Qiskit Aer is already the strongest required external reference.
- The micro-validation layer catches local correctness regressions that could be
  hidden by workflow-only benchmarks.
- The workflow layer proves that the backend is useful in the intended research
  setting.

### Consequences

- Paper 1 evidence is defined before implementation rather than assembled after
  the fact.
- Phase 2 benchmark work remains bounded and reproducible.
- Broader simulator bake-offs are clearly optional, not silently expected.

### Rejected Alternatives

- Leave benchmark design to post-implementation paper drafting.
- Require a large multi-framework comparison before Phase 2 can proceed.
- Use only one or two favorable workflow cases without a micro-validation layer.

### Upstream Alignment

Refines:

- `P2-ADR-006`
- `PUBLICATIONS.md` Paper 1 evidence package

## P2-ADR-015: Use Numeric Exactness And Workflow-Readiness Thresholds

### Status

`Accepted`

### Context

Before this ADR, the Phase 2 acceptance language was qualitatively strong but
not numerically closed.

That was enough for planning, but not enough for:

- a strict implementation gate,
- a reproducible benchmark pass or fail rule,
- or a publication-ready claim about exactness.

### Decision

Phase 2 acceptance uses the following numeric thresholds:

- mandatory acceptance coverage at 4, 6, 8, and 10 qubits,
- maximum absolute energy error `<= 1e-10` on the 1 to 3 qubit microcases,
- maximum absolute energy error `<= 1e-8` on the mandatory 4, 6, 8, and 10
  qubit workflow parameter sweeps,
- `rho.is_valid(tol=1e-10)` pass condition and `|Tr(rho) - 1| <= 1e-10` on
  recorded validation outputs,
- `|Im Tr(H*rho)| <= 1e-10` for the exact observable path,
- `100%` pass rate on the mandatory micro-validation matrix,
- `100%` pass rate on the mandatory workflow benchmark set,
- and at least one documented 10-qubit anchor evaluation case plus the full
  reproducibility bundle.

### Rationale

- Exactness claims should be strict enough to defend scientifically.
- Different thresholds for microcases and workflow cases reflect the difference
  between tiny correctness kernels and larger dense mixed-state benchmarks.
- No runtime speed threshold is added, because Phase 2 is not the acceleration
  phase.

### Consequences

- Phase 2 completion becomes a measurable gate rather than a qualitative
  judgment call.
- Benchmark and paper drafting can refer to the same pass or fail rules.
- Performance remains characterized, but not treated as the main completion
  criterion.

### Rejected Alternatives

- Keep Phase 2 acceptance purely qualitative.
- Add acceleration or runtime pass thresholds to Phase 2.
- Accept partial benchmark passes as sufficient for the main Phase 2 claim.

### Upstream Alignment

Refines:

- `P2-ADR-006`
- `P2-ADR-007`

## Phase 2 Decision Gate Summary

At the end of Phase 2, the following questions must all have a positive answer:

1. Can the density-matrix backend be selected explicitly and used in the anchor
   noisy VQE workflow without fallback?
2. Does the `Re Tr(H*rho)` observable path meet the numeric exactness thresholds
   against Qiskit Aer?
3. Does the frozen gate and noise support matrix cover the anchor workflow
   across the documented 4 to 10 qubit exact regime?
4. Is the benchmark and reproducibility package strong enough to draft Paper 1
   honestly and confidently?

If the answer to any of these questions is no, then Phase 2 is incomplete even
if some underlying implementation work exists.
