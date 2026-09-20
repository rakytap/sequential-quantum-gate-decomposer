# Task 4 Stories

This document decomposes Phase 2 Task 4 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_4_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, `P2-ADR-004`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`. They describe behavioral
slices, not implementation chores.

Story ordering is intentional:

1. make the required realistic local-noise baseline visible on the supported
   VQE path,
2. validate each mandatory local model and mixed schedules on small exact cases,
3. keep optional extensions clearly separate from the mandatory baseline,
4. lock unsupported and deferred noise requests behind deterministic
   pre-execution failures,
5. prove the required baseline is sufficient on the anchor XXZ workflow sizes,
6. preserve noise provenance in reproducible artifacts for publication claims.

## Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations

**User/Research value**
- This story makes the Phase 2 realistic local-noise baseline observable on the
  actual VQE-facing density path rather than only in planning text or
  standalone backend APIs.

**Given / When / Then**
- Given explicit `density_matrix` backend selection, the default generated
  `HEA` circuit from `qgd_Variational_Quantum_Eigensolver_Base`, and a
  supported XXZ anchor workflow request.
- When the workflow applies one of the required local noise models on the
  documented density path.
- Then the request executes with explicit ordered local noise operations for
  that model on the supported VQE-facing path rather than through a toy
  whole-register substitute or a standalone-only route.

**Scope**
- In: the positive path for `local single-qubit depolarizing`,
  `local amplitude damping`, and `local phase damping / dephasing` on the
  supported VQE workflow.
- Out: micro-validation exactness thresholds, optional noise extensions,
  unsupported-case coverage, and workflow-scale reproducibility packaging.

**Acceptance signals**
- Positive supported-path cases exist for each required local noise model on the
  generated `HEA` density workflow.
- Stored evidence makes the noise model identity, explicit insertion order, and
  target placement auditable for each required model.
- Supported requests do not rely on whole-register depolarizing to stand in for
  the required local-noise baseline.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 4 goal and evidence
  requirements; `TASK_4_MINI_SPEC.md` required behavior.
- ADR decision(s): `P2-ADR-004`, `P2-ADR-012`, `P2-ADR-013`.

## Story 2: Required Local Noise Models And Mixed Schedules Validate Cleanly On 1 To 3 Qubit Exact Microcases

**User/Research value**
- This story gives early scientific confidence that the mandatory Phase 2
  local-noise baseline is numerically trustworthy before broader workflow
  claims depend on it.

**Given / When / Then**
- Given 1 to 3 qubit exact noisy validation cases that exercise each required
  local noise model individually and at least one mixed schedule composed only
  from required local models.
- When those cases are executed on the supported density-matrix path and
  compared against Qiskit Aer.
- Then the mandatory local-noise baseline satisfies the frozen micro-validation
  exactness and density-validity thresholds without exceptions or degraded
  behavior.

**Scope**
- In: 1 to 3 qubit micro-validation coverage for each required local model,
  mixed required-model schedules, and the frozen exactness thresholds.
- Out: 4 to 10 qubit workflow-scale completion, optional extension models, and
  broader optimizer behavior.

**Acceptance signals**
- Required microcases cover `local single-qubit depolarizing`,
  `local amplitude damping`, and `local phase damping / dephasing`
  individually, plus at least one mixed required-noise schedule.
- All required microcases meet maximum absolute energy error `<= 1e-10`,
  `rho.is_valid(tol=1e-10)`, `|Tr(rho) - 1| <= 1e-10`, and
  `|Im Tr(H*rho)| <= 1e-10`.
- The mandatory micro-validation matrix achieves a `100%` pass rate.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 13 workload
  classes and noise classes; `TASK_4_MINI_SPEC.md` acceptance evidence.
- ADR decision(s): `P2-ADR-012`, `P2-ADR-014`, `P2-ADR-015`.

## Story 3: Optional Noise Extensions Remain Clearly Optional And Do Not Substitute For The Mandatory Baseline

**User/Research value**
- This story protects the scientific meaning of Phase 2 by ensuring optional
  baselines or justified extensions do not blur what the milestone actually
  requires.

**Given / When / Then**
- Given Phase 2 validation, benchmark, or regression material that may include
  whole-register depolarizing or a justified extension such as generalized
  amplitude damping or coherent over-rotation.
- When a reviewer inspects the documented support surface and evidence bundle.
- Then it is clear which noise models are mandatory, which are optional, and
  Phase 2 completion does not depend on optional evidence alone.

**Scope**
- In: required versus optional classification, optional extension labeling, and
  the rule that whole-register depolarizing is not the main scientific
  baseline.
- Out: promoting optional models into required scope, deferred-noise handling,
  and unsupported-case error semantics.

**Acceptance signals**
- The support matrix and Task 4 evidence explicitly label required, optional,
  and deferred noise classes.
- Any benchmark or regression artifact using whole-register depolarizing or a
  justified extension is marked as optional rather than milestone-defining.
- A reviewer can determine from the stored evidence that the required local
  baseline is complete independently of optional cases.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` support-matrix decision
  and Task 4 evidence requirements; `TASK_4_MINI_SPEC.md` required behavior.
- ADR decision(s): `P2-ADR-004`, `P2-ADR-012`.

## Story 4: Deferred Or Unsupported Noise Requests Fail Before Execution With The First Unsupported Noise Condition

**User/Research value**
- This story keeps the Task 4 boundary scientifically honest by turning
  out-of-scope noise requests into explicit, reviewable failures rather than
  ambiguous behavior.

**Given / When / Then**
- Given a density-path request that includes a deferred or unsupported noise
  class, such as correlated multi-qubit noise, readout noise,
  calibration-aware noise, non-Markovian noise, or an unsupported schedule
  element.
- When the user requests execution through the documented `density_matrix`
  workflow.
- Then the request fails deterministically before execution, names the first
  unsupported noise condition, and does not silently substitute, drop, or
  reroute the noise request.

**Scope**
- In: pre-execution hard-error behavior for unsupported or deferred noise
  classes and unsupported schedule elements owned by Task 4.
- Out: general backend-fallback semantics already frozen by Task 1 and future
  scope expansions that require a new phase-level decision.

**Acceptance signals**
- Negative cases cover at least one deferred or unsupported request in each
  major Task 4 boundary class: unsupported model family and unsupported schedule
  element.
- Unsupported requests do not silently fall back to another model, to a
  noiseless path, or to `state_vector`.
- Error output or stored diagnostics identify the first unsupported noise
  condition for each negative case.

**Traceability**
- Phase requirement(s): `TASK_4_MINI_SPEC.md` unsupported behavior;
  `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase acceptance criteria.
- ADR decision(s): `P2-ADR-012`.

## Story 5: The Required Local-Noise Baseline Supports The Anchor XXZ Workflow Across The Mandatory Exact-Regime Sizes

**User/Research value**
- This story proves the Task 4 baseline is not only locally correct but
  sufficient for the actual Phase 2 noisy-training workflow that matters for
  paper claims.

**Given / When / Then**
- Given supported XXZ noisy VQE anchor cases at 4, 6, 8, and 10 qubits using
  the generated `HEA` circuit path and noise schedules composed from required
  local models.
- When those cases are executed through the density-matrix workflow.
- Then the mandatory workflow package completes within the frozen exact-regime
  acceptance surface, including one reproducible 4- or 6-qubit optimization
  trace and one documented 10-qubit anchor evaluation.

**Scope**
- In: workflow-scale completion for the required 4, 6, 8, and 10 qubit fixed
  parameter cases, one reproducible 4- or 6-qubit optimization trace, and one
  documented 10-qubit anchor case using the required local-noise baseline.
- Out: optional noise sweeps, runtime speed thresholds, and later multi-workflow
  studies.

**Acceptance signals**
- Mandatory 4, 6, 8, and 10 qubit anchor cases execute with required local-noise
  schedules on the supported density path.
- Mandatory workflow cases meet maximum absolute energy error `<= 1e-8` and a
  `100%` pass rate on the mandatory benchmark set.
- At least one reproducible 4- or 6-qubit optimization trace and one documented
  10-qubit anchor evaluation rely on the required local-noise baseline rather
  than optional models.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase
  acceptance criteria and Section 13 workload classes; `TASK_4_MINI_SPEC.md`
  acceptance evidence.
- ADR decision(s): `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`.

## Story 6: Noise Artifacts Preserve Model Identity, Placement, And Classification For Reproducible Publication Evidence

**User/Research value**
- This story makes Task 4 evidence reviewable and publishable by preserving how
  each supported or unsupported noise case should be interpreted.

**Given / When / Then**
- Given supported Task 4 validation cases, workflow evidence, and negative
  unsupported-case artifacts.
- When reproducibility and benchmark material is recorded for the Phase 2 noise
  baseline.
- Then a reviewer can determine the noise model names, insertion order, target
  locations, parameters or probabilities, required versus optional
  classification, backend, and unsupported-case diagnostics for each stored
  case.

**Scope**
- In: Task 4-specific provenance, classification metadata, artifact completeness,
  and publication-ready evidence packaging for the noise baseline.
- Out: full paper drafting, broad simulator comparisons, and artifact standards
  unrelated to the Task 4 noise contract.

**Acceptance signals**
- Stored Task 4 evidence records noise model identity, insertion order, target
  location, and relevant parameters for supported cases.
- Evidence distinguishes required local-noise cases from optional extension or
  regression cases.
- Negative evidence preserves unsupported-case diagnostics that identify the
  first failing noise condition.

**Traceability**
- Phase requirement(s): `TASK_4_MINI_SPEC.md` acceptance evidence and
  publication relevance; `DETAILED_PLANNING_PHASE_2.md` Task 4 evidence
  requirements.
- ADR decision(s): `P2-ADR-004`, `P2-ADR-012`, `P2-ADR-014`.
