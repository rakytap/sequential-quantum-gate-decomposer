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
- Integration depth is intentionally staged into phases 2-4.
- Research-facing noisy VQA studies begin after integration milestones.

## PhD Milestone Mapping

| Phase | PhD Milestone |
|---|---|
| Phase 2 | Noise models and preliminary noisy emulation (Spring 2026) |
| Phase 3 | Full noise module and validation (Fall 2026) |
| Phase 4 | Noisy VQA integration and optimizer comparisons (Spring 2027) |
| Phase 5 | Trainability analysis under noise (Fall 2027) |


## Objective-by-Objective Mapping

### 1) Noise modeling and simulation

- Phase 1:
  - baseline unital and non-unital channels implemented.
- Phases 2-3:
  - expand channel set based on experiment demand,
  - add calibration and fidelity validation workflows.

### 2) Trainability analysis under noise

- Phase 5 primary target:
  - gradient variance studies,
  - entropy and expressivity metrics,
  - barren plateau diagnostics by depth/noise/locality.

### 3) Barren plateau mitigation in noisy VQA

- Phase 4:
  - integrate noisy backend into VQA training loop,
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
  - noisy emulation stable around 10 qubits.
- Phase 3:
  - gradient support for density backend,
  - expanded noise channels validated against reference simulator.
- Phase 4:
  - end-to-end noisy VQA training loop functional,
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
realistic trainability research and noise-aware optimization.

## References Used in This Alignment

- SQUANDER publications listed in the repository root `README.md`:
  1. Rakyta, Zimboras (Quantum, 2022)
  2. Rakyta, Zimboras (arXiv:2203.04426)
  3. Rakyta et al. (JCP, 2024)
  4. Nadori et al. (Quantum, 2025)
- PhD plan milestone text supplied with this project update.

