# Task 5: Validation Baseline

This mini-spec turns Phase 2 Task 5 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-006`, `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-012`, `P2-ADR-013`,
`P2-ADR-014`, and `P2-ADR-015`, plus the closed backend-selection,
workflow-anchor, benchmark-minimum, and numeric-threshold items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen baseline
choice, observable scope, support-matrix scope, workflow anchor, or numeric
acceptance decisions.

## Given / When / Then
- Given explicit `density_matrix` backend selection, the frozen Phase 2 XXZ
  noisy VQE anchor, and validation workloads inside the documented gate, noise,
  bridge, and observable contract.
- When the mandatory validation package compares supported SQUANDER outputs
  against the required external reference and evaluates internal consistency,
  workflow completion, stability of end-to-end execution, and reproducibility
  completeness.
- Then Phase 2 claims are backed by a documented pass/fail baseline that
  distinguishes supported success, unsupported boundary behavior, and
  incomplete evidence without relying on ad hoc favorable examples.

## Assumptions and dependencies
- Task 1 provides explicit backend selection and hard-error no-fallback
  behavior for `density_matrix`.
- Task 2 provides the exact noisy observable contract `Re Tr(H*rho)` and the
  phase-frozen exactness metrics used by the validation baseline.
- Task 3 provides the supported `HEA`-first bridge and auditable lowering
  metadata needed to show which execution path was validated.
- Task 4 provides the required / optional / deferred noise classification and
  the mandatory local-noise surface that the validation package must exercise.
- Qiskit Aer density-matrix simulation is the required primary external
  baseline; any secondary simulator comparison is optional only when it
  materially strengthens publication evidence.
- The benchmark minimum remains the closed Phase 2 package: 1 to 3 qubit
  micro-validation circuits, 4 / 6 / 8 / 10 qubit workflow cases, at least 10
  fixed parameter vectors per mandatory workflow size, at least one
  reproducible 4- or 6-qubit optimization trace, and a full reproducibility
  bundle.
- Runtime and peak memory are required recorded metrics for the validation
  package, but they are not Phase 2 pass thresholds.
- This task defines the minimum correctness and completeness evidence for Phase
  2 claims; it does not define new noise models, new observables, broader
  workflow families, or acceleration targets.

## Required behavior
- Task 5 freezes the minimum validation baseline required before Phase 2 can be
  presented as scientifically usable or publication-ready.
- The mandatory validation baseline is layered rather than example-driven: it
  must include both exact micro-validation and workflow-scale evidence.
- Qiskit Aer density-matrix simulation is the required primary external
  reference for all mandatory exactness claims in the Phase 2 package.
- The mandatory micro-validation layer covers 1 to 3 qubit circuits that
  exercise each required gate and each required local noise model individually,
  plus at least one mixed schedule composed only from required local-noise
  models.
- The mandatory workflow layer covers the anchor XXZ noisy VQE path with the
  default `HEA` ansatz at 4, 6, 8, and 10 qubits, with at least 10 fixed
  parameter vectors per mandatory workflow size.
- At least one reproducible 4- or 6-qubit optimization trace must demonstrate
  the supported density-matrix path inside a training-relevant loop rather than
  only on fixed parameter sweeps.
- Internal consistency checks are mandatory on recorded supported outputs even
  when external agreement is good: density validity, trace preservation, and
  Hermitian-observable consistency must be evaluated and recorded.
- The frozen numeric thresholds are part of the Task 5 contract:
  `<= 1e-10` maximum absolute energy error on the mandatory 1 to 3 qubit
  micro-validation matrix,
  `<= 1e-8` maximum absolute energy error on the mandatory 4 / 6 / 8 / 10
  qubit workflow matrix,
  `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`,
  and `|Im Tr(H*rho)| <= 1e-10` on recorded validation outputs.
- The validation baseline requires a `100%` pass rate on the mandatory
  micro-validation matrix and a `100%` pass rate on the mandatory workflow
  benchmark set; partial passes do not close Task 5.
- A documented 10-qubit anchor evaluation case is required as part of the
  accepted exact-regime evidence package.
- The validation package must record workflow completion, runtime, and peak
  memory alongside stability of end-to-end execution and the correctness
  metrics so the exact-regime evidence is auditable and reproducible.
- Reproducibility artifacts for mandatory evidence must record the Hamiltonian
  family, ansatz, backend, circuit source or bridge metadata, noise schedule,
  parameter vectors or seeds, versions or commit, raw results, thresholds, and
  pass/fail summaries.
- Mandatory validation artifacts must use stable case identifiers and explicit
  status checks or equivalent manifest fields so reruns, completeness checks,
  and publication references remain auditable.
- Supported, optional, deferred, and unsupported cases must remain explicitly
  distinguished in validation artifacts and summaries so publication claims do
  not overstate the guaranteed Phase 2 surface.
- Optional secondary baselines may supplement the package, but they must not
  replace the mandatory Aer-centered pass/fail interpretation.
- Existing favorable examples or partial traces may seed the validation bundle,
  but they do not satisfy Task 5 unless the full mandatory matrix and the
  documented pass/fail interpretation are present.

## Unsupported behavior
- Claiming Task 5 is complete from only one or two favorable workflow examples,
  from only 4- and 6-qubit cases, or from standalone density-backend tests that
  do not cover the frozen benchmark minimum.
- Treating workflow-scale agreement without the 1 to 3 qubit micro-validation
  layer as sufficient correctness evidence.
- Treating micro-validation agreement without the 4 / 6 / 8 / 10 workflow layer
  as sufficient evidence that the supported workflow is scientifically usable.
- Replacing Qiskit Aer with another simulator as the only external reference for
  the mandatory Phase 2 validation package.
- Accepting partial benchmark passes, skipped mandatory cases, hand-selected
  favorable subsets, or undocumented reruns as sufficient for the main Phase 2
  claim.
- Counting unsupported, silently rerouted, or silently degraded requests as
  successful mandatory validation cases.
- Using runtime speed as a mandatory Phase 2 acceptance threshold, or using the
  absence of a speed threshold as a reason to omit runtime and peak-memory
  recording entirely.
- Claiming validation of broader observable families, broader workflow families,
  or larger exact-regime scales than the documented 4 / 6 / 8 / 10 qubit
  anchor package.
- Treating a reproducibility bundle with missing backend identity, Hamiltonian
  metadata, noise schedule, threshold values, or raw results as publication-
  ready evidence.

## Acceptance evidence
- A documented validation matrix identifies the mandatory micro-validation and
  workflow cases, gives them stable case identifiers, and records their purpose,
  scope, and pass/fail status.
- Qiskit Aer comparisons cover the 1 to 3 qubit mandatory micro-validation
  matrix and satisfy maximum absolute energy error `<= 1e-10` with `100%`
  pass rate.
- Those mandatory micro-validation artifacts record
  `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`,
  and `|Im Tr(H*rho)| <= 1e-10` on supported recorded outputs.
- Qiskit Aer comparisons cover the mandatory 4 / 6 / 8 / 10 qubit anchor XXZ
  workflow matrix with at least 10 fixed parameter vectors per required size and
  satisfy maximum absolute energy error `<= 1e-8` with `100%` pass rate.
- At least one reproducible 4- or 6-qubit optimization trace demonstrates the
  supported density-matrix workflow end to end.
- At least one documented 10-qubit anchor evaluation case is present in the
  mandatory evidence bundle.
- Recorded workflow artifacts include workflow-completion status, runtime, and
  peak-memory metrics, plus stability of end-to-end execution, alongside
  correctness and validity metrics.
- Unsupported-case artifacts or negative tests show that out-of-scope requests
  fail explicitly and are not counted as positive Task 5 evidence.
- The reproducibility bundle records the Hamiltonian, ansatz, backend,
  circuit-source or bridge route, noise schedule, parameter vectors or seeds,
  versions or commit, thresholds, raw results, and case-level pass/fail
  interpretation.
- Stored manifests or equivalent bundle summaries preserve stable case
  identifiers and explicit status fields for mandatory validation cases.
- Traceability target: satisfy the Phase 2 Task 5 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criteria requiring
  validated exactness, documented workflow stability in the exact regime, and a
  publication-ready evidence package.
- Traceability target: satisfy the Aer-centered benchmark-minimum and
  numeric-threshold decisions frozen in `P2-ADR-014` and `P2-ADR-015`.

## Affected interfaces
- Validation and benchmark harnesses that compare SQUANDER outputs against
  Qiskit Aer and compute the required internal consistency metrics.
- Case-definition surfaces for the mandatory micro-validation matrix and the
  mandatory workflow-scale matrix.
- Result schemas, manifests, and completeness checkers that determine whether
  the full mandatory evidence package is present.
- Reproducibility bundle assembly and publication-facing artifact packaging for
  the Phase 2 abstract, short paper, and full paper.
- User-facing or developer-facing reporting surfaces that summarize pass/fail
  status, thresholds, backend identity, and unsupported-case boundaries.
- Change classification: additive for validation automation and artifact
  packaging, but stricter for incomplete or ambiguous evidence because missing
  mandatory items block the scientific-usability claim.

## Publication relevance
- Supports Paper 1's central claim that the Phase 2 density-matrix backend is
  validated against a trusted external reference with explicit, reproducible
  pass/fail rules.
- Provides the correctness baseline needed to write the Phase 2 abstract, short
  paper, and full paper without overstating evidence strength.
- Keeps publication claims scientifically defensible by requiring both local
  correctness evidence and workflow-scale exact-regime evidence.
- Supplies the auditable validation package that later phases can extend when
  they compare acceleration, broader workflows, or additional references against
  the frozen Phase 2 baseline.
