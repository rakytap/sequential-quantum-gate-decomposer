# Research Alignment to PhD Plan

This document maps the density matrix project to the PhD plan
"Scalable Training of Noisy Quantum Circuits" and clarifies what is already
aligned versus what remains for later milestones.

Primary audience: PhD committee, supervisor, and research reviewers.

## Recommended Reading Order

This document is self-contained. For implementation details referenced below,
see the companion engineering docs:
- `CHANGELOG.md` (delivered and planned scope),
- `ARCHITECTURE.md` (extension points),
- `API_REFERENCE.md` (current API),
- `SETUP.md` (build and verification).

## Scope Used for Alignment

- Scope is limited to density-matrix-specific work in SQUANDER.
- GPU kernel development is treated as a parallel track, not a gating item for
density-matrix phase progression.
- Stochastic trajectory methods are deferred to later stages.
- Broader VQE/VQA feature growth beyond the accomplished Phase 2 canonical
  workflow is deferred to Phase 4 and later.

## Phase 1 Validation Against PhD Fall 2025 Milestone

PhD milestone text (Phase 1 / Fall 2025): introduce density matrices, implement a few
channels, and start integration into baseline workflows.

Alignment assessment:

- Introduced density matrices: complete (`DensityMatrix` and mixed-state ops).
- Implemented initial channels: complete (depolarizing, amplitude damping, phase
damping).
- Baseline integration start: partial by design (module integrated in build and
package; deep VQA integration deferred to later phases).

Verdict: **Phase 1 is aligned** with the expected Phase 1 research scope.

Review takeaway:

- Foundation goals are complete.
- Phase 2 exact noisy backend integration is complete for one canonical
  workflow.
- Phase 3 now targets noise-aware partitioning/fusion rather than additional
  VQE/VQA surface growth.
- Research-facing noisy VQE/VQA studies broaden in Phase 4 after the
  partitioning/fusion milestone.

## PhD Milestone Mapping

| Phase | PhD Milestone |
|---|---|
| Phase 2 | Exact noisy backend integration for one canonical workflow (Spring 2026) |
| Phase 3 | Noise-aware partitioning and fusion for mixed-state circuits (Fall 2026) |
| Phase 4 | Broader noisy VQE/VQA integration, gradients, and optimizer comparisons (Spring 2027) |
| Phase 5 | Trainability analysis under noise (Fall 2027) |

Roadmap shorthand used in the documentation contract:

- Phase 3 | Full noise module and validation
- Phase 4 | Noisy VQA integration and optimizer comparisons


## Objective-by-Objective Mapping

### 1) Noise modeling and simulation

- Phase 1:
  - baseline unital and non-unital channels implemented.
- Phase 2:
  - exact noisy backend integrated for one canonical workflow,
  - exact observable path and validation bundle established.
- Phase 3:
  - extend partitioning/fusion so noise channels and density-matrix semantics are
    first-class in the circuit model,
  - add calibration and fidelity validation workflows required by partitioning
    benchmarks.

### 2) Trainability analysis under noise

- Phase 5 primary target:
  - gradient variance studies,
  - entropy and expressivity metrics,
  - barren plateau diagnostics by depth/noise/locality.

### 3) Barren plateau mitigation in noisy VQA

- Phase 4:
  - broaden the noisy backend into richer VQE/VQA training loops,
  - add density-backend gradient and optimizer support for the supported
    Phase 4 surface,
  - benchmark BLS and baseline optimizers under noise.
- Phase 5:
  - convert results into robust training guidelines.

### 4) Software dissemination

- Already open-source in branch scope.
- Documentation restructured to separate:
  - setup/use,
  - API,
  - architecture,
  - research alignment and milestone traceability.

## Acceptance Criteria by Phase

- Phase 2:
  - VQE backend switch can execute density-matrix path,
  - `Tr(H*rho)` expectation value path validated,
  - canonical noisy workflow contract and validation bundle complete.
- Phase 3:
  - partitioning/gate fusion accepts noisy mixed-state circuits without reducing
    noise to partition-boundary-only metadata,
  - Phase 3 includes an executable partitioned path with at least one real
    fused execution mode rather than only a planner/runtime representation,
  - partitioning decisions are noise-aware for density workloads,
  - partitioned/fused execution matches the sequential density baseline,
  - channel-native fused noisy blocks are not required for minimum Phase 3
    closure.
- Phase 4:
  - end-to-end noisy VQE/VQA training loop beyond the Phase 2 baseline is
    functional,
  - density-backend gradient and optimizer routing support the supported Phase 4
    surface,
  - reproducible optimizer comparison experiments.
- Phase 5:
  - complete trainability analysis dataset,
  - publication-grade figures/tables and documented conclusions.

## Relationship to Existing SQUANDER Work

This project extends SQUANDER's existing strengths in:

- high-performance classical emulation,
- large-circuit decomposition,
- optimizer implementations for variational workflows.

The density-matrix track adds the missing noisy mixed-state layer required for
realistic trainability research and noise-aware optimization, and Phase 3 now
extends that mixed-state semantics into SQUANDER's partitioning/fusion stack
before Phase 4 broadens the VQE/VQA surface again.

## References Used in This Alignment

- SQUANDER publications listed in the repository root `README.md`:
  1. Rakyta, Zimboras (Quantum, 2022)
  2. Rakyta, Zimboras (arXiv:2203.04426)
  3. Rakyta et al. (JCP, 2024)
  4. Nadori et al. (Quantum, 2025)
- PhD plan milestone text supplied with this project update.

