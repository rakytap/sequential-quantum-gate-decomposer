# Phase 2 Documentation Index

This document is the stable entry point for the Phase 2 documentation contract.
It is intended to help developers, reviewers, and paper readers find the
authoritative Phase 2 contract without reverse-engineering code or inferring
scope from scattered notes.

This index summarizes and links the Phase 2 contract. It does not override the
underlying planning, ADR, mini-spec, checklist, validation, or paper-facing
sources.

## Source-of-Truth Hierarchy

| Document class | Primary files | Role |
|---|---|---|
| Phase 2 documentation entry point | `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md` | Stable starting point for navigation, topic lookup, and citation of the Phase 2 document bundle |
| Phase contract | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | Defines the full Phase 2 scope, frozen implementation contract, acceptance criteria, risks, and expected outcome |
| Phase decisions | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md` | Records the accepted Phase 2 decisions, rationale, consequences, and rejected alternatives |
| Readiness and closure | `docs/density_matrix_project/phases/phase-2/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | Records how the open pre-implementation contract items were closed and what implementation-ready means |
| Task contracts | `docs/density_matrix_project/phases/phase-2/task-1/TASK_1_MINI_SPEC.md` through `docs/density_matrix_project/phases/phase-2/task-8/TASK_8_MINI_SPEC.md` | Define required behavior, unsupported behavior, acceptance evidence, affected interfaces, and publication relevance per Phase 2 task |
| Workflow evidence surface | `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json` | Canonical machine-readable evidence package for the delivered Phase 2 workflow-backed claim |
| Paper-facing summaries | `docs/density_matrix_project/phases/phase-2/ABSTRACT_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_NARRATIVE.md`, `docs/density_matrix_project/phases/phase-2/PAPER_PHASE_2.md` | Publication-facing framing of the delivered Phase 2 claim and evidence |
| Roadmap context | `docs/density_matrix_project/planning/PLANNING.md`, `docs/density_matrix_project/planning/PUBLICATIONS.md`, `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`, `docs/density_matrix_project/CHANGELOG.md` | Broader roadmap, publication ladder, milestone wording, and future-phase context |

## Topic Map

| Topic | Primary source | Supporting sources | Layer |
|---|---|---|---|
| Backend selection | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, `docs/density_matrix_project/phases/phase-2/task-1/TASK_1_MINI_SPEC.md` | phase contract + task contract |
| Observable scope | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-2/TASK_2_MINI_SPEC.md` | phase contract + task contract |
| Bridge scope | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-3/TASK_3_MINI_SPEC.md` | phase contract + task contract |
| Support matrix | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-4/TASK_4_MINI_SPEC.md` | phase contract + task contract |
| Workflow anchor | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-6/TASK_6_MINI_SPEC.md` | phase contract + task contract |
| Benchmark minimum | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-5/TASK_5_MINI_SPEC.md`, `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json` | phase contract + evidence surface |
| Numeric thresholds | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/task-5/TASK_5_MINI_SPEC.md`, `docs/density_matrix_project/phases/phase-2/task-6/TASK_6_MINI_SPEC.md` | phase contract + task contract |
| Non-goals | `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md` | `docs/density_matrix_project/planning/PLANNING.md`, `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`, `docs/density_matrix_project/CHANGELOG.md` | phase contract + roadmap context |
| Publication evidence surface | `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json` | `docs/density_matrix_project/planning/PUBLICATIONS.md`, `docs/density_matrix_project/phases/phase-2/ABSTRACT_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_NARRATIVE.md`, `docs/density_matrix_project/phases/phase-2/PAPER_PHASE_2.md` | evidence surface + publication-facing |
| Paper 1 claim package | `docs/density_matrix_project/phases/phase-2/task-8/TASK_8_MINI_SPEC.md` | `docs/density_matrix_project/planning/PUBLICATIONS.md`, `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`, `docs/density_matrix_project/phases/phase-2/ABSTRACT_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_PHASE_2.md`, `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_NARRATIVE.md`, `docs/density_matrix_project/phases/phase-2/PAPER_PHASE_2.md`, `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json` | task contract + publication-facing + evidence surface |

## Stable Evidence Entry Point

The authoritative machine-readable evidence surface for the delivered Phase 2
workflow-backed claim is:

- `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`

That manifest links the canonical workflow contract, end-to-end trace bundle,
workflow-matrix bundle, unsupported-boundary bundle, and interpretation bundle,
while preserving traceability to the underlying validation-evidence layers.

## Supported Entry And Canonical Workflow

The supported Phase 2 entry surface and canonical workflow are intentionally
narrow and should be read as contract statements, not as broad future-facing
aspirations.

- `state_vector` remains the default backend when no explicit backend is
  selected.
- `density_matrix` must be selected explicitly for exact noisy mixed-state
  claims.
- No implicit `auto` mode or silent fallback is part of the Phase 2 contract.
- The canonical supported Phase 2 workflow is noisy VQE ground-state estimation
  for a 1D XXZ spin chain with local `Z` field using the default `HEA` ansatz,
  explicit local noise insertion, and exact `Re Tr(H*rho)` evaluation.
- Full end-to-end workflow execution is required at 4 and 6 qubits.
- Benchmark-ready fixed-parameter evaluation is required at 8 and 10 qubits.
- The documented 10-qubit case is the acceptance anchor for the current exact
  regime.

Primary sources for these statements:

- `docs/density_matrix_project/phases/phase-2/task-1/TASK_1_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/task-6/TASK_6_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`

## Guaranteed VQE-Facing Support Surface

The guaranteed Phase 2 VQE-facing density path is generated `HEA` only.
Broader standalone `NoisyCircuit` capability does not automatically imply
guaranteed VQE-facing support.

| Support class | Phase 2 status | Contract statement |
|---|---|---|
| Circuit source | Required | Generated `HEA` lowered from the VQE-facing path is the guaranteed source surface |
| Gate families | Required | Required gate families: `U3`, `CNOT` |
| Local noise models | Required | Required local noise models: local depolarizing, local phase damping or dephasing, and local amplitude damping |
| Whole-register depolarizing | Optional | Whole-register depolarizing remains optional and does not count as required local-noise support |
| Additional extensions | Optional | Generalized amplitude damping and coherent over-rotation remain optional benchmark extensions rather than delivered required support |
| Broader noise or circuit capability | Deferred | Correlated multi-qubit noise, readout noise, calibration-aware models, non-Markovian noise, and broader manual circuit parity remain deferred |
| Unsupported promotion | Unsupported | Full `qgd_Circuit` parity is not part of the Phase 2 guarantee, and unsupported circuit or noise requests must not be promoted into guaranteed workflow support |

Boundary examples:

- Full `qgd_Circuit` parity is not part of the Phase 2 guarantee.
- Broader standalone `NoisyCircuit` capability does not automatically imply
  guaranteed VQE-facing support.
- Whole-register depolarizing remains optional rather than required.

Primary sources for these support-surface statements:

- `docs/density_matrix_project/phases/phase-2/task-3/TASK_3_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/task-4/TASK_4_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`

## Mandatory Evidence Bar

The core Phase 2 claim is supported only by the mandatory evidence package
defined below. Favorable examples, optional baselines, unsupported boundary
cases, or incomplete bundles must not replace that package.

Mandatory evidence package:

- mandatory 1 to 3 qubit micro-validation matrix,
- mandatory 4 / 6 / 8 / 10 qubit fixed-parameter workflow matrix with 10
  parameter vectors per required size,
- at least one reproducible 4- or 6-qubit optimization trace,
- one documented 10-qubit anchor case,
- runtime and peak-memory recording for mandatory workflow evidence,
- and the backend-explicit reproducibility bundle rooted in
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`.

Mandatory thresholds and closure rules:

- mandatory micro-validation accuracy: `<= 1e-10`,
- mandatory workflow-matrix accuracy: `<= 1e-8`,
- `100%` pass rate on the mandatory evidence package,
- only mandatory, complete, supported evidence closes the core Phase 2 claim,
- optional whole-register depolarizing remains supplemental,
- unsupported, deferred, or incomplete evidence remains excluded from the core
  claim.

Primary sources for the evidence-bar statements:

- `docs/density_matrix_project/phases/phase-2/task-5/TASK_5_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/task-6/TASK_6_MINI_SPEC.md`
- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
- `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`

## Future Work And Non-Goals

Phase 2 remains the exact noisy backend integration milestone. The items below
may be discussed as future work or roadmap context, but they are not current
Phase 2 commitments.

| Topic | Later phase or status | Phase 2 boundary statement |
|---|---|---|
| Density-aware partitioning, fusion, and acceleration | Future work for Phase 3 | Density-aware partitioning, fusion, and acceleration are future work for Phase 3, not current Phase 2 commitments |
| Gradient support and approximate scaling | Future work beyond core Phase 2 | Gradient support and approximate scaling methods remain future work beyond the core Phase 2 milestone |
| Broader noisy-VQA integration and optimizer comparisons | Future work for Phase 4 | Broader noisy-VQA integration and optimizer comparisons are future work for Phase 4, not current Phase 2 commitments |
| Trainability analysis | Future work for Phase 5 | Trainability analysis is future work for Phase 5, not current Phase 2 commitments |
| Current milestone identity | Phase 2 status | Phase 2 remains the exact noisy backend integration milestone rather than an umbrella for later acceleration or trainability work |

Non-goal reminder:

- These topics are not current Phase 2 commitments.

Primary sources for the future-work boundary:

- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`
- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`

## Terminology And Consistency Notes

Use the following terms consistently across Phase 2 docs:

- `state_vector`: the default backend when no explicit density path is selected.
- `density_matrix`: the explicitly selected mixed-state backend for the frozen
  Phase 2 workflow-backed claim.
- canonical workflow: noisy XXZ VQE with default `HEA`, explicit local noise,
  and exact `Re Tr(H*rho)` evaluation.
- exact regime: full end-to-end workflow execution at 4 and 6 qubits, with
  benchmark-ready fixed-parameter evaluation at 8 and 10 qubits.
- acceptance anchor: the documented 10-qubit case used as the current exact-
  regime anchor.
- required / optional / deferred / unsupported: the explicit support-tier
  vocabulary used for support-surface, evidence, and claim-closure semantics.
- reproducibility bundle: the backend-explicit evidence package rooted in
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`.
- future work and non-goal: labels that keep later-phase work visible without
  turning it into a current Phase 2 commitment.

## Reading Order

When a reviewer needs the Phase 2 contract in the fastest correct order, use:

1. `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`
2. `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
3. `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`
4. `docs/density_matrix_project/phases/phase-2/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
5. the relevant task mini-spec, usually `task-1` through `task-8`
6. `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`
7. the paper-facing document that matches the review level:
   `ABSTRACT_PHASE_2.md`, `SHORT_PAPER_PHASE_2.md`,
   `SHORT_PAPER_NARRATIVE.md`, or `PAPER_PHASE_2.md`
