# Task 4: Phase 2 Noise Support Baseline

This mini-spec turns Phase 2 Task 4 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-004`, `P2-ADR-012`, `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`,
plus the closed support-matrix, workflow-anchor, benchmark-minimum, and
numeric-threshold items in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It
does not reopen the scientific choice of realistic local noise, the required /
optional / deferred support split, or the Phase 2 benchmark thresholds.

## Given / When / Then
- Given explicit `density_matrix` backend selection, the anchor XXZ noisy VQE
  workflow, and an ordered noise schedule composed only from the frozen Phase 2
  support surface.
- When the workflow lowers the supported `HEA` circuit into the density-matrix
  path and applies the requested local noise insertions for micro-validation,
  fixed-parameter benchmarks, or the anchor optimization trace.
- Then the mandatory local noise models execute as explicit, auditable local
  operations, optional models remain clearly non-mandatory, and unsupported or
  deferred noise requests fail before execution with deterministic diagnostics.

## Assumptions and dependencies
- Task 1 provides explicit backend selection and hard-error no-fallback
  behavior for `density_matrix`.
- Task 3 provides the supported `HEA`-first bridge into `NoisyCircuit` with
  ordered `NoiseOperation` insertion.
- Task 2 defines the exact noisy energy path and the Aer-centered exactness
  thresholds; Task 4 defines which noise classes must reach that path, not the
  observable mathematics itself.
- The guaranteed mandatory gate surface remains `U3` and `CNOT`; optional extra
  gates may appear only when a documented test or comparison case needs them and
  do not widen the required Task 4 noise claim.
- The acceptance package remains the Phase 2 benchmark minimum: 1 to 3 qubit
  micro-validation circuits, anchor XXZ workflow cases at 4, 6, 8, and 10
  qubits, at least 10 fixed parameter vectors per mandatory workflow size, at
  least one reproducible 4- or 6-qubit optimization trace, and the full
  reproducibility bundle.
- This task defines the scientific and workflow-facing noise baseline for Phase
  2; it does not define readout modeling, hardware-calibration ingestion,
  gradient behavior, acceleration, or later trainability study design.

## Required behavior
- Task 4 freezes a finite, workflow-driven Phase 2 noise contract rather than
  an open-ended list of possible future models.
- The mandatory Phase 2 local-noise baseline is:
  `local single-qubit depolarizing`, `local amplitude damping`, and
  `local phase damping / dephasing`.
- These three required local models must be usable on the documented
  density-matrix VQE path and on the mandatory 1 to 3 qubit micro-validation
  circuits.
- Phase 2 completion cannot be claimed using only whole-register depolarizing
  noise; whole-register depolarizing is optional only as a regression or
  stress-test baseline.
- The guaranteed positive path is the anchor XXZ noisy VQE workflow using
  `qgd_Variational_Quantum_Eigensolver_Base`, the default `HEA` ansatz, and
  explicit ordered local noise insertion.
- Noise insertion for supported cases must remain explicit, ordered, and
  auditable rather than implicit, heuristic, or reconstructed after the fact.
- The supported contract is local and per-operation or per-location in the
  sense needed by the bridge and reproducibility bundle; recorded evidence must
  make model identity, placement, and parameters reviewable.
- Mixed schedules composed from the required local noise models may be used when
  validation or workflow cases need them, but they must stay inside the frozen
  support surface and preserve documented insertion order.
- Optional Phase 2 noise items are limited to:
  whole-register depolarizing as a regression or stress-test baseline, and
  generalized amplitude damping or coherent over-rotation only when a justified
  benchmark extension requires them.
- Any optional extension must remain clearly marked optional in docs,
  benchmarks, and publication claims unless a new phase-level decision promotes
  it into the required baseline.
- Deferred noise classes remain out of the mandatory Phase 2 contract:
  correlated multi-qubit noise, readout noise as a density-backend feature,
  calibration-aware models, and non-Markovian noise.
- If a requested noise model or noise schedule falls outside the frozen support
  matrix, the request must fail before execution on the documented VQE-facing
  density path.
- The required failure path must not silently substitute another noise model,
  silently drop noise, silently route to another backend, or silently collapse a
  local-noise request into a whole-register baseline.
- The required local-noise baseline must be sufficient for the Phase 2 anchor
  XXZ workflow at the exact-regime sizes frozen for 4, 6, 8, and 10 qubit
  acceptance coverage.
- Support classification as required, optional, or deferred must be consistent
  across implementation, tests, benchmark artifacts, and the Phase 2 paper
  bundle.

## Unsupported behavior
- Treating whole-register depolarizing as the main scientific Phase 2 noise
  result or as sufficient evidence that the realistic local-noise baseline is
  complete.
- Claiming generalized amplitude damping, coherent over-rotation, or additional
  `NoisyCircuit`-only models as mandatory without a new phase-level scope
  decision.
- Treating correlated multi-qubit noise, readout noise, calibration-aware
  models, or non-Markovian noise as required Phase 2 support.
- Silent fallback from an unsupported local-noise request to another noise
  model, to a noiseless path, or to `state_vector`.
- Hidden or heuristic noise placement that cannot be reconstructed from the
  recorded evidence.
- Using standalone `NoisyCircuit` breadth to overstate the guaranteed VQE-facing
  Phase 2 support surface.
- Claiming Task 4 covers hardware-calibrated device modeling, noise learning, or
  the broader optimization studies planned for later phases.
- Returning ambiguous unsupported-case failures that do not identify the first
  unsupported noise condition or schedule element.

## Acceptance evidence
- A documented support matrix identifies the required, optional, and deferred
  Phase 2 noise classes and matches `DETAILED_PLANNING_PHASE_2.md`,
  `P2-ADR-012`, and the closed support-matrix item in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Qiskit Aer micro-validation comparisons cover 1 to 3 qubit circuits that
  exercise each required local noise model individually and at least one mixed
  required-noise schedule on the mandatory gate surface.
- Those micro-validation cases satisfy the frozen numeric thresholds:
  maximum absolute energy error `<= 1e-10`,
  `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`,
  and `|Im Tr(H*rho)| <= 1e-10` on recorded validation outputs.
- Anchor XXZ workflow comparisons against Qiskit Aer cover the mandatory 4, 6,
  8, and 10 qubit sizes with at least 10 fixed parameter vectors per mandatory
  workflow size, and the workload package collectively exercises the required
  local-noise baseline.
- The mandatory workflow benchmark cases satisfy the frozen workflow thresholds:
  maximum absolute energy error `<= 1e-8` and `100%` pass rate on the mandatory
  benchmark set.
- At least one reproducible 4- or 6-qubit optimization trace exercises a
  supported required local-noise schedule end-to-end inside the anchor VQE
  workflow.
- At least one documented 10-qubit anchor evaluation case demonstrates that the
  required local-noise baseline reaches the exact-regime acceptance anchor.
- Negative tests or validation cases show that deferred or unsupported noise
  requests fail before execution and do not silently degrade into optional or
  alternate behavior.
- The reproducibility bundle records the Hamiltonian family, ansatz, backend,
  noise model names, insertion order, target locations, parameters or
  probabilities, seeds, versions or commit, raw results, and unsupported-case
  diagnostics where applicable.
- Traceability target: satisfy the Phase 2 Task 4 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criterion requiring a
  bounded realistic local-noise baseline rather than a toy global-noise-only
  study.
- Traceability target: satisfy the support-matrix, workflow-anchor,
  benchmark-minimum, and numeric-threshold decisions frozen in `P2-ADR-012`,
  `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`.

## Affected interfaces
- The VQE-facing noise-configuration or noise-insertion surface used with
  `qgd_Variational_Quantum_Eigensolver_Base`.
- The bridge boundary that lowers the generated `HEA` circuit into
  `NoisyCircuit` and preserves ordered local noise insertion.
- `NoisyCircuit` and `NoiseOperation` construction or adapter surfaces used to
  represent the required local-noise models on the supported density path.
- Pre-execution validation and error reporting for unsupported noise models,
  unsupported schedules, or out-of-scope optional and deferred requests.
- Benchmark, validation, and reproducibility tooling that must record the noise
  schedule with enough detail for auditability and paper-ready evidence.
- User-facing and paper-facing support-matrix documentation that distinguishes
  required support from optional extensions and deferred scope.
- Change classification: additive for the supported realistic local-noise
  baseline, but stricter for ambiguous or unsupported noise requests, which
  become explicit hard failures rather than undocumented behavior.

## Publication relevance
- Supports Paper 1's central claim that Phase 2 studies use realistic local
  noise models rather than relying mainly on whole-register toy noise.
- Provides the contract behind the specific baseline noise claims already
  reflected in the Phase 2 abstract, short paper, and full paper:
  local depolarizing, local phase damping or dephasing, and local amplitude
  damping.
- Keeps publication claims scientifically defensible by separating required
  baseline models from optional benchmark extensions and explicitly deferred
  noise families.
- Supplies auditable evidence for later comparisons between different local
  noise classes without overstating the guaranteed Phase 2 support surface.
