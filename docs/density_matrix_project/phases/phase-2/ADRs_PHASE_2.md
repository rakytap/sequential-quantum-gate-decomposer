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

## Phase 2 Decision Gate Summary

At the end of Phase 2, the following questions must all have a positive answer:

1. Can the density-matrix backend be selected and used in the intended noisy
   workflow?
2. Is the `Tr(H*rho)` observable path validated strongly enough for publication?
3. Does the supported gate and noise scope cover at least one representative
   exact noisy workflow?
4. Is the evidence package strong enough to draft Paper 1 honestly and
   confidently?

If the answer to any of these questions is no, then Phase 2 is incomplete even
if some underlying implementation work exists.
