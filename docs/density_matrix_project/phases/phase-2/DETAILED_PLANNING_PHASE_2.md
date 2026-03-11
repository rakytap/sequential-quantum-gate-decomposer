# Detailed Planning for Phase 2

This document is the Phase 2 source of truth for scope, task goals,
acceptance criteria, validation expectations, and research-facing deliverables.

Primary Phase 2 theme:

> turn the density-matrix module from a standalone exact noisy simulator into a
> usable, validated backend for noisy variational workflows.

This is a specification document, not an implementation log.

## 1. Purpose

Phase 2 exists to bridge the current gap between:

- the standalone exact mixed-state backend already delivered in Phase 1, and
- the noisy training workflows required by the broader PhD research program on
  scalable training under realistic noise models.

Phase 2 is the first phase where the density-matrix work becomes a complete
scientific instrument rather than only an enabling module.

## 2. Source-of-Truth Hierarchy

If multiple existing documents overlap, interpret them in the following order of
authority for Phase 2.

### Tier 1: Strategic Planning Constraints

These documents define the authoritative Phase 2 intent and trade-off
boundaries:

- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`
- `docs/density_matrix_project/planning/REFERENCES.md`

### Tier 2: Existing Project Roadmap and Milestone Definitions

These documents define accepted milestone wording and must remain consistent
with the Phase 2 plan:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`

### Tier 3: Architecture and Existing Module Baseline

These documents describe the current delivered system and named integration
targets:

- `docs/density_matrix_project/ARCHITECTURE.md`
- `docs/density_matrix_project/README.md`
- `docs/density_matrix_project/SETUP.md`
- `docs/density_matrix_project/phases/phase-1/API_REFERENCE_PHASE_1.md`

### Tier 4: Legacy or Supportive Context

These sources are informative, but do not override the above:

- older top-level references to flat API documentation,
- existing branch-specific comments in examples or benchmark notes,
- exploratory planning notes that are not reflected in the Phase 2 planning set.

## 3. Traceability Matrix

| Requirement area | Upstream source | Phase 2 interpretation |
|---|---|---|
| Backend selection | `CHANGELOG.md`, `RESEARCH_ALIGNMENT.md`, `PLANNING.md` | Phase 2 must define and validate a selectable density-matrix execution path |
| `Tr(H*rho)` expectation value | `ARCHITECTURE.md`, `CHANGELOG.md`, `RESEARCH_ALIGNMENT.md`, `PUBLICATIONS.md` | Phase 2 must deliver a validated observable-evaluation path suitable for noisy training workflows |
| Bridge from existing circuit/gate representations | `ARCHITECTURE.md`, `CHANGELOG.md`, `PLANNING.md` | Phase 2 must specify how current circuit workflows reach the density backend without requiring Phase 3 partitioning/fusion |
| Realistic local noise | `ADRs.md`, `PLANNING.md`, `PUBLICATIONS.md` | Phase 2 prioritizes local noise models required for first noisy training studies |
| Exact dense density matrices as reference | `ADRs.md`, `PLANNING.md` | Phase 2 remains exact-first and does not substitute approximate methods for the core deliverable |
| No density-aware partitioning/fusion yet | `PLANNING.md`, `ADRs.md` | Phase 2 explicitly excludes Phase 3 acceleration work from the core commitment |
| Publication readiness | `PUBLICATIONS.md`, `PLANNING.md` | Phase 2 must be documented and validated in a way that directly supports Paper 1 |

## 4. Spec-Driven Development Principles for Phase 2

Phase 2 should follow spec-driven development in the following sense:

1. Define contracts, scope boundaries, and success criteria before any
   implementation decisions are treated as fixed.
2. Separate required behavior from possible implementation choices.
3. Maintain explicit traceability from milestone goals to validation evidence.
4. Treat unsupported cases and deferred work as first-class documented outcomes.
5. Use publication evidence requirements to guide what counts as done.
6. Keep task descriptions goal-oriented rather than implementation-prescriptive.

In practice, this means Phase 2 documentation should answer:

- what must be true when Phase 2 ends,
- what is explicitly not part of Phase 2,
- how correctness and scientific usefulness will be demonstrated,
- and what evidence is required before Phase 2 can be declared complete.

## 5. Mission Statement

Phase 2 delivers an exact noisy backend integration layer for SQUANDER's
variational workflows, including:

- backend selection,
- exact noisy expectation-value evaluation,
- circuit-to-density-backend bridging,
- realistic local noise support needed for initial studies,
- and validation strong enough to support the first major paper.

## 6. In Scope

The following are in scope for Phase 2.

### 6.1 Backend Integration

- A documented density-matrix backend path that can be selected within the
  relevant training and decomposition workflows.
- A clear user-facing and developer-facing contract for how backend choice is
  expressed and interpreted.

### 6.2 Exact Noisy Observable Evaluation

- A documented and validated expectation-value path for `Tr(H*rho)`.
- Support for observable evaluation sufficient for at least one noisy
  variational workflow.

### 6.3 Circuit and Representation Bridging

- A documented bridge from existing gate/circuit representations to the
  density-matrix execution path.
- Defined behavior for supported, unsupported, and fallback cases.

### 6.4 Workload-Driven Noise Expansion

- Noise-model support required for the first noisy training studies.
- Priority for realistic local noise over global toy noise.

### 6.5 Validation and Publication Evidence

- External reference comparison,
- reproducible benchmark definitions,
- and a publication-ready evidence package for Paper 1.

## 7. Out of Scope

The following are explicitly outside the Phase 2 core deliverable.

- density-aware partitioning and gate fusion,
- channel-native fusion or superoperator fusion,
- gradient-path completion for density-matrix optimization,
- AVX-focused density-kernel acceleration as a main milestone,
- stochastic trajectories or MPDO-based approximation paths,
- full `qgd_Circuit` gate parity,
- and noisy circuit re-synthesis as a primary result.

These topics may be mentioned as future work, but must not be presented as Phase
2 commitments.

## 8. Assumptions

Phase 2 planning assumes:

- the current exact density-matrix backend remains the scientific reference path,
- the main exact target regime is around the current documented Phase 2
  acceptance range of about 10 qubits,
- the existing state-vector backend remains intact and continues to serve as the
  default or comparison path where appropriate,
- at least one representative noisy VQA workflow can be specified and validated
  inside the exact regime,
- and Qiskit Aer remains the primary external reference simulator.

## 9. Success Conditions

Phase 2 is only scientifically successful if all of the following become true:

- the density-matrix backend can be selected in the relevant workflow,
- exact noisy expectation values required by training are available and
  validated,
- at least one noisy training workflow is executable end-to-end in the exact
  regime,
- realistic local noise models are used in the core demonstrations,
- unsupported or deferred cases are explicitly documented,
- benchmark and validation results are strong enough to support Paper 1,
- and the resulting system is useful for the later optimizer and trainability
  studies rather than being only a software bridge.

## 10. Phase 2 Task Breakdown

Each task below is a goal, not an implementation recipe.

### Task 1: Backend Selection Contract

#### Goal

Define and validate how the density-matrix backend is selected and how that
selection interacts with the current state-vector-first workflows.

#### Why It Exists

Without a stable backend-selection contract, the density backend remains a
standalone tool instead of a usable research backend.

#### Success Looks Like

- backend choice is a documented first-class concept,
- expected user-facing behavior is defined,
- and interaction with existing workflows is unambiguous.

#### Evidence Required

- documented backend-selection semantics,
- defined supported entry points,
- and validation that the density path can actually be invoked in the intended
  workflow class.

### Task 2: Exact Noisy Expectation-Value Path

#### Goal

Define and validate a `Tr(H*rho)` observable-evaluation path sufficient for
noisy variational training.

#### Why It Exists

Observable evaluation is the minimum scientific requirement for the density
backend to participate in VQA or similar training loops.

#### Success Looks Like

- the expectation-value contract is documented,
- observable evaluation is numerically trustworthy,
- and at least one training-relevant use case depends on it successfully.

#### Evidence Required

- agreement against a trusted reference backend,
- documented observable-scope assumptions,
- and benchmark or test evidence showing stable exact noisy evaluation.

### Task 3: Circuit-to-Density Backend Bridge

#### Goal

Define how existing circuit and gate representations are translated or routed
into the density-matrix execution path.

#### Why It Exists

The current density backend is structurally separate from the main circuit path.
This task turns that separation into a documented, usable bridge rather than an
implicit gap.

#### Success Looks Like

- supported circuit-entry modes are documented,
- unsupported cases are defined,
- and the bridge is sufficient for the target noisy workflow.

#### Evidence Required

- explicit support matrix,
- defined behavior for unsupported gate/noise combinations,
- and at least one end-to-end workflow that depends on the bridge.

### Task 4: Phase 2 Noise Support Baseline

#### Goal

Define the realistic local noise models required for the first noisy training
studies and document what is mandatory, optional, and deferred.

#### Why It Exists

Phase 2 should not drift into either unrealistically weak toy-noise studies or
an unbounded effort to implement every possible noise model.

#### Success Looks Like

- the Phase 2 noise scope is finite,
- it is aligned with realistic local noisy training use cases,
- and it is sufficient for the first training demonstrations.

#### Evidence Required

- a noise support matrix,
- rationale for why each required model is included,
- and benchmark or validation plans tied to those models.

### Task 5: Validation Baseline

#### Goal

Define the minimum exactness and correctness evidence required before Phase 2 can
be considered scientifically usable.

#### Why It Exists

Phase 2 is exact-first. Its scientific value depends on trusted correctness.

#### Success Looks Like

- numerical correctness criteria are explicit,
- reference comparisons are specified,
- and acceptance thresholds are defined.

#### Evidence Required

- exactness comparison plan against Qiskit Aer,
- internal consistency checks,
- and a documented pass/fail interpretation for Phase 2 claims.

### Task 6: Noisy Workflow Demonstration Goal

#### Goal

Specify at least one end-to-end noisy workflow that Phase 2 must support.

#### Why It Exists

Phase 2 should be judged by usefulness to noisy training research, not only by
backend availability.

#### Success Looks Like

- one representative workflow is chosen,
- the workflow is executable inside the exact regime,
- and its artifacts are reproducible enough for publication.

#### Evidence Required

- workflow definition,
- expected input/output contract,
- and demonstration-ready validation or benchmark design.

### Task 7: Documentation and User-Facing Clarity

#### Goal

Document the Phase 2 support surface so that the backend can be used, reviewed,
and discussed without reverse-engineering implementation details.

#### Why It Exists

Paper 1 and future implementation both depend on clear contracts and explicit
scope boundaries.

#### Success Looks Like

- clear developer-facing and research-facing documentation exists,
- Phase 2 non-goals are visible,
- and future work is separated from current commitments.

#### Evidence Required

- coherent Phase 2 document bundle,
- terminology consistency,
- and explicit alignment with the planning and roadmap docs.

### Task 8: Paper 1 Evidence Package

#### Goal

Define the exact evidence required for the first major publication tied to Phase
2.

#### Why It Exists

Publication readiness should shape the phase, not be retrofitted after the fact.

#### Success Looks Like

- Paper 1 has a clear contribution boundary,
- required evidence is defined before implementation,
- and benchmark/validation demands are aligned with the publication strategy.

#### Evidence Required

- abstract-level claim set,
- short-paper structure,
- full-paper structure,
- and traceability from Phase 2 deliverables to publication sections.

## 11. Full-Phase Acceptance Criteria

Phase 2 is complete only if all of the following are true:

- the VQE or equivalent workflow can execute the density-matrix backend path,
- `Tr(H*rho)` is validated for the intended workload class,
- exact noisy emulation is stable in the documented exact regime,
- the bridge from current circuit/gate representations is documented and usable,
- realistic local noise support is sufficient for the first noisy training
  studies,
- at least one end-to-end noisy workflow is specified and supported,
- and the publication evidence package for Paper 1 is complete enough to support
  abstract, short-paper, and full-paper drafting.

These criteria are intentionally aligned with:

- `docs/density_matrix_project/CHANGELOG.md`
- `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`

## 12. Validation and Benchmark Matrix

### 12.1 Primary External Baseline

- Qiskit Aer density-matrix simulation is the required primary reference.

### 12.2 Optional Secondary Baselines

- One additional simulator or framework may be used when it materially improves
  the publication evidence, but this is secondary to Qiskit Aer.

### 12.3 Workload Classes

- small-to-medium exact noisy circuits,
- at least one training-relevant variational workflow,
- representative ansatz classes appropriate to the supported gate set,
- and benchmark cases that stress the observable-evaluation path.

### 12.4 Noise Classes

- local depolarizing,
- dephasing / phase damping,
- amplitude damping,
- generalized amplitude damping if required by the target workload,
- coherent unitary or over-rotation error if required by the target workload,
- optional readout or shot-noise discussion where relevant to the workflow.

### 12.5 Metrics

- numerical agreement with trusted references,
- observable error,
- runtime,
- memory footprint,
- stability of end-to-end execution,
- and reproducibility of the workflow and benchmark setup.

## 13. Risks

### Risk 1: Phase 2 Scope Drift

If Phase 2 starts to absorb density-aware partitioning, fusion, or gradient-path
work, it will lose focus and weaken Paper 1.

Mitigation:

- enforce out-of-scope boundaries,
- and treat acceleration as Phase 3 unless strictly required for Phase 2
  usability.

### Risk 2: Gate-Coverage Mismatch

The target workflow may depend on unsupported gate families.

Mitigation:

- use workload-driven gate support decisions,
- choose representative workflows compatible with the documented support matrix,
- and explicitly document unsupported cases.

### Risk 3: Noise-Scope Inflation

Trying to support too many realistic noise models too early can delay the first
scientifically useful backend integration.

Mitigation:

- require every Phase 2 noise model to justify itself via a target workflow or
  publication need.

### Risk 4: Weak Validation Package

If the reference comparisons are too narrow, Paper 1 will read as a software
note rather than a strong methods paper.

Mitigation:

- define publication evidence requirements before implementation,
- and keep Qiskit Aer validation central.

### Risk 5: Scientific Contribution Dilution

Phase 2 could become a pure integration effort without a strong research story.

Mitigation:

- keep the document bundle centered on exact noisy training usability and
  publication-grade evidence,
- not only on technical integration.

## 14. Decision Gates

### DG-1: Phase 2 Completion Gate

Question:

- does the integrated backend support a reproducible exact noisy workflow that
  is strong enough to underpin Paper 1?

If no:

- Phase 2 is not complete,
- even if partial backend integration exists.

### DG-2: Handoff to Phase 3

Question:

- are the remaining limitations primarily about performance and execution cost,
  rather than missing workflow integration?

If yes:

- proceed to density-aware partitioning, fusion, and acceleration.

If no:

- the project still has unresolved Phase 2 integration debt.

## 15. Non-Goals

To avoid later ambiguity, the following are explicit non-goals of this phase:

- proving density-aware partitioning benefit,
- delivering fused density-matrix execution,
- solving density-matrix gradients end to end,
- maximizing qubit count beyond the exact reference regime,
- and producing large-scale approximation results.

## 16. Expected Outcome

At the end of Phase 2, the project should have:

- an exact noisy backend that is integrated into the relevant workflow,
- a validated expectation-value path,
- a realistic local noise baseline for early studies,
- a reproducible evidence package for Paper 1,
- and a clean handoff into Phase 3, where performance acceleration becomes the
  main question.

That is the minimum outcome required for Phase 2 to count as a meaningful step
toward the broader PhD objective.
