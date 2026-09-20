# Task 5 Stories

This document decomposes Phase 2 Task 5 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_5_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, `P2-ADR-006`, `P2-ADR-010`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`. They describe behavioral
slices, not implementation chores.

Story ordering is intentional:

1. lock the local correctness kernel before workflow claims are trusted,
2. define the mandatory workflow-scale exact-regime pass/fail baseline,
3. tie validation to one training-relevant trace and the accepted 10-qubit
   anchor,
4. require internal consistency and execution-stability metrics in addition to
   external agreement,
5. keep optional, unsupported, and incomplete evidence from inflating the main
   Phase 2 claim,
6. preserve stable case identity and provenance so Task 5 evidence is directly
   reviewable and publishable.

## Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline

**User/Research value**
- This story catches small exactness, ordering, and physical-validity
  regressions before the project relies on larger workflow-scale evidence.

**Given / When / Then**
- Given supported 1 to 3 qubit validation cases that cover each required gate
  family and each required local noise model, plus at least one mixed schedule
  composed only from required local-noise models.
- When those cases are executed on the supported density-matrix path and
  compared against Qiskit Aer.
- Then the local correctness baseline is explicit, stable, and complete enough
  that later workflow claims do not depend on hand-picked favorable examples.

**Scope**
- In: the mandatory micro-validation matrix, stable case identifiers, required
  gate-and-noise coverage, exactness checks, density-validity checks, and the
  `100%` micro-matrix pass rule.
- Out: 4 to 10 qubit workflow sweeps, optimization traces, optional secondary
  baselines, and publication bundle assembly.

**Acceptance signals**
- The mandatory micro-validation matrix covers each required gate family and
  each required local noise model individually, plus at least one mixed
  required-noise schedule.
- All mandatory microcases satisfy maximum absolute energy error `<= 1e-10`,
  `rho.is_valid(tol=1e-10)`, `|Tr(rho) - 1| <= 1e-10`, and
  `|Im Tr(H*rho)| <= 1e-10`.
- The mandatory micro-validation matrix has stable case identifiers and a
  `100%` pass rate.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 5 evidence
  requirements; Section 13 workload classes and metrics;
  `TASK_5_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-010`, `P2-ADR-012`, `P2-ADR-014`, `P2-ADR-015`.

## Story 2: The Mandatory 4 To 10 Qubit Workflow Matrix Defines Exact-Regime Pass/Fail For The Anchor XXZ Workflow

**User/Research value**
- This story proves the Phase 2 validation baseline is meaningful at workflow
  scale rather than only on tiny exactness kernels.

**Given / When / Then**
- Given supported XXZ noisy VQE anchor cases at 4, 6, 8, and 10 qubits using
  the default `HEA` ansatz and at least 10 fixed parameter vectors per
  mandatory workflow size.
- When those cases are executed through the supported density-matrix workflow
  and compared against Qiskit Aer.
- Then the exact-regime pass/fail baseline is defined by the full mandatory
  matrix rather than by a hand-selected subset of favorable cases.

**Scope**
- In: the mandatory 4 / 6 / 8 / 10 qubit fixed-parameter workflow matrix, the
  `<= 1e-8` exactness threshold, workflow completion, and the `100%`
  workflow-matrix pass rule.
- Out: optimization-trace details, optional simulator bake-offs, runtime speed
  thresholds, and broader workflow families.

**Acceptance signals**
- At least 10 fixed parameter vectors are recorded for each mandatory workflow
  size.
- Every mandatory workflow case is attributable to the supported
  `density_matrix` path and satisfies maximum absolute energy error `<= 1e-8`
  versus Qiskit Aer.
- The mandatory workflow matrix completes with a `100%` pass rate and no
  unsupported-operation workarounds.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase
  acceptance criteria; Section 13 workload classes and metrics;
  `TASK_5_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`.

## Story 3: The Validation Baseline Includes A Reproducible Optimization Trace And A Documented 10-Qubit Anchor Case

**User/Research value**
- This story keeps Task 5 tied to the real noisy-training workflow and the
  accepted exact-regime anchor instead of reducing validation to fixed sweeps
  alone.

**Given / When / Then**
- Given the supported Phase 2 density-matrix workflow and the mandatory
  validation package.
- When a reviewer inspects the training-relevant and exact-regime anchor
  evidence.
- Then the package includes at least one reproducible 4- or 6-qubit
  optimization trace and one documented 10-qubit anchor evaluation that show
  the validated path is both workflow-relevant and acceptance-ready.

**Scope**
- In: one reproducible 4- or 6-qubit optimization trace, one documented
  10-qubit anchor evaluation, and their inclusion in the mandatory evidence
  package.
- Out: a broad optimizer study, multi-workflow comparisons, runtime pass
  thresholds, and claims beyond the accepted exact regime.

**Acceptance signals**
- One reproducible 4- or 6-qubit optimization trace exercises the full
  backend-selection, bridge, noise, and observable path end to end.
- One documented 10-qubit anchor evaluation is present in the mandatory Task 5
  evidence bundle.
- Reviewers can see from stored artifacts that both cases belong to the
  supported Phase 2 path rather than to an optional or standalone-only route.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Section 12 full-phase
  acceptance criteria; Section 13 workload classes; `TASK_5_MINI_SPEC.md`
  required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-013`, `P2-ADR-014`, `P2-ADR-015`.

## Story 4: Supported Validation Cases Record Internal Consistency And Execution Stability Alongside External Agreement

**User/Research value**
- This story prevents strong external agreement from masking invalid density
  states, unstable execution, or incomplete evidence recording.

**Given / When / Then**
- Given supported mandatory validation cases across the micro-validation and
  workflow-scale matrices.
- When results are recorded for the Task 5 evidence package.
- Then each supported case carries the internal consistency and execution
  metrics needed to judge whether the exactness claim is physically valid,
  operationally stable, and reproducible.

**Scope**
- In: density-validity checks, trace-preservation, Hermitian-observable
  consistency, workflow-completion status, runtime, peak memory, and stability
  of end-to-end execution.
- Out: runtime speed thresholds, acceleration claims, and optional secondary
  baseline comparisons beyond the required recorded metrics.

**Acceptance signals**
- Mandatory recorded outputs preserve `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`, and `|Im Tr(H*rho)| <= 1e-10` where those checks
  apply.
- Mandatory workflow artifacts record workflow-completion status, runtime, peak
  memory, and stability of end-to-end execution.
- A mandatory case with missing required consistency or execution metrics is
  treated as incomplete Task 5 evidence rather than as a silent pass.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 5 evidence
  requirements; Section 13 metrics; `TASK_5_MINI_SPEC.md` required behavior and
  acceptance evidence.
- ADR decision(s): `P2-ADR-010`, `P2-ADR-014`, `P2-ADR-015`.

## Story 5: Optional, Unsupported, And Incomplete Evidence Cannot Masquerade As Mandatory Validation Success

**User/Research value**
- This story keeps the scientific meaning of the Phase 2 milestone honest by
  separating the mandatory claim from helpful but non-defining evidence.

**Given / When / Then**
- Given optional secondary baseline results, unsupported requests, negative
  boundary cases, or a validation bundle with missing mandatory items.
- When Task 5 results are summarized for developers, reviewers, or paper
  drafting.
- Then the main Phase 2 validation claim remains tied to the mandatory
  Aer-centered package, and optional, unsupported, or incomplete evidence is
  clearly marked so it cannot be mistaken for milestone-closing success.

**Scope**
- In: mandatory versus optional baseline interpretation, unsupported-case
  exclusion from positive evidence, and incompleteness rules for missing
  mandatory cases or failed status checks.
- Out: implementing unsupported backend, bridge, observable, or noise behavior
  itself, and any new scope decision that would change the frozen Phase 2
  contract.

**Acceptance signals**
- Optional secondary baselines are labeled supplemental and do not replace the
  required Qiskit Aer pass/fail baseline.
- Unsupported or deferred requests appear only as negative evidence and are not
  counted as positive Task 5 completion signals.
- Missing mandatory cases, missing status checks, or hand-selected favorable
  subsets block Task 5 closure rather than being treated as acceptable partial
  success.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 5 evidence
  requirements; Section 12 full-phase acceptance criteria;
  `TASK_5_MINI_SPEC.md` unsupported behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-006`, `P2-ADR-014`, `P2-ADR-015`.

## Story 6: Validation Artifacts Preserve Stable Case Identity, Thresholds, And Pass/Fail Provenance For Publication Evidence

**User/Research value**
- This story makes Task 5 evidence reviewable, reproducible, and citable in the
  Phase 2 publication package instead of leaving correctness claims implicit.

**Given / When / Then**
- Given supported mandatory validation cases and any stored negative evidence
  used to explain boundaries.
- When the Task 5 reproducibility bundle and manifests are assembled.
- Then a reviewer can audit stable case identifiers, thresholds, backend,
  Hamiltonian, ansatz, circuit-source or bridge metadata, noise schedule,
  parameters or seeds, versions or commit, raw results, and case-level pass or
  fail interpretation directly from stored artifacts.

**Scope**
- In: stable case identity, manifest status fields, threshold recording,
  provenance metadata, raw-result preservation, and publication-facing artifact
  completeness for Task 5.
- Out: paper writing itself, broad simulator comparison studies, and artifact
  standards unrelated to the Task 5 validation baseline.

**Acceptance signals**
- Stored manifests or equivalent bundle summaries preserve stable case
  identifiers and explicit status fields for mandatory validation cases.
- The Task 5 reproducibility bundle records thresholds, backend identity,
  Hamiltonian, ansatz, circuit-source or bridge route, noise schedule,
  parameters or seeds, versions or commit, and raw results.
- A reviewer can determine from stored artifacts why each case passed, failed,
  or was excluded from the mandatory Phase 2 claim.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 5 evidence
  requirements; Section 13 metrics and reproducibility expectations;
  `TASK_5_MINI_SPEC.md` acceptance evidence and publication relevance.
- ADR decision(s): `P2-ADR-006`, `P2-ADR-014`, `P2-ADR-015`.
