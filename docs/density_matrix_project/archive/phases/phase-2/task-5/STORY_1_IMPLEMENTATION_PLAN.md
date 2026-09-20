# Story 1 Implementation Plan

## Story Being Implemented

Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local
Correctness Baseline

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_5_STORIES.md`.

## Scope

This story turns the frozen 1 to 3 qubit micro-validation contract into the
canonical local-correctness gate for the whole Phase 2 validation baseline:

- the phase-level local correctness baseline is defined by the mandatory 1 to 3
  qubit micro-validation matrix rather than by ad hoc tiny examples,
- the baseline reuses the existing exactness kernel and required-noise coverage
  surfaces instead of introducing a second overlapping micro-validation
  framework,
- stable case identifiers, explicit status checks, and complete pass/fail
  semantics make it impossible to close the local baseline with skipped or
  hand-selected subsets,
- and the resulting local gate is narrow enough that later stories can add the
  4 / 6 / 8 / 10 workflow-scale matrix, the optimization trace plus 10-qubit
  anchor evidence, broader execution-stability metrics, and the final
  publication bundle without replacing the local correctness layer.

Out of scope for this story:

- the mandatory 4 / 6 / 8 / 10 workflow-scale exact-regime matrix owned by
  Story 2,
- the reproducible 4- or 6-qubit optimization trace and documented 10-qubit
  anchor case owned by Story 3,
- workflow-completion, runtime, peak-memory, and end-to-end execution-stability
  closure for the full Task 5 package owned by Story 4,
- optional-baseline, unsupported-case, and incomplete-evidence interpretation
  closure owned by Story 5,
- the final Task 5 provenance and publication bundle owned by Story 6,
- expanding the frozen Phase 2 gate, noise, observable, or support-matrix
  contract,
- and introducing a new low-level exactness harness that diverges from the
  existing Story 2 / Task 4 micro-validation path.

## Dependencies And Assumptions

- Task 2 Story 2 already provides the canonical exactness kernel in
  `benchmarks/density_matrix/validate_squander_vs_qiskit.py`, including
  `Re Tr(H*rho)` comparison against Qiskit Aer, density-validity checks,
  trace-preservation checks, Hermitian-observable consistency checks, and the
  canonical `micro_validation_bundle.json` output shape.
- Task 4 Story 2 already specializes that canonical harness for the required
  local-noise baseline in
  `benchmarks/density_matrix/noise_support/required_local_noise_micro_validation.py`,
  including required-noise coverage, mixed-sequence-order auditability,
  support-tier semantics, and mandatory-baseline classification.
- `MANDATORY_MICROCASES` in
  `benchmarks/density_matrix/circuits.py` already provides the natural stable
  inventory for the mandatory 1 to 3 qubit cases.
- `tests/density_matrix/test_density_matrix.py` already provides useful footholds
  for Story 1 through representative Story 2 microcase tests and the Task 4
  Story 2 bundle-schema test.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already consumes
  the canonical micro-validation bundle inside the broader workflow package, so
  Task 5 Story 1 should preserve schema compatibility where possible instead of
  forking into a disconnected artifact vocabulary.
- The frozen Task 5 numeric gate is already defined by `P2-ADR-014` and
  `P2-ADR-015`:
  - maximum absolute energy error `<= 1e-10` on mandatory 1 to 3 qubit
    microcases,
  - `rho.is_valid(tol=1e-10)`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `|Im Tr(H*rho)| <= 1e-10`,
  - and `100%` pass rate on the mandatory micro-validation matrix.
- The current Phase 2 planning language now requires stable case identifiers and
  explicit status checks or equivalent manifest fields for mandatory validation
  evidence; Story 1 must carry that requirement into the local correctness gate.
- Story 1 should define the phase-level local correctness baseline; it should
  not reopen the support split, the workflow anchor, the benchmark minimum, or
  the numeric thresholds already frozen at the phase level.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Task 5 Local-Correctness Micro-Validation Inventory

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Task 5 Story 1 names one canonical mandatory 1 to 3 qubit micro-validation
  inventory for the phase-level local correctness gate.
- Every mandatory case has a stable identifier, declared purpose, and clear
  mapping to the required gate and local-noise coverage it is meant to prove.
- Optional or exploratory microcases remain outside the mandatory Task 5 Story 1
  pass/fail set.

**Execution checklist**
- [ ] Review `MANDATORY_MICROCASES` as the canonical case inventory for
      the Task 5 local-correctness gate.
- [ ] Confirm the inventory covers the frozen required gate family and each
      required local-noise model individually, plus the mandatory mixed
      required-noise sequence.
- [ ] Preserve stable case IDs and purpose strings exactly enough that later
      Task 5 stories and publication artifacts can reference them directly.
- [ ] Keep optional or exploratory microcases outside the mandatory Story 1
      closure rule.

**Evidence produced**
- One named mandatory Task 5 local-correctness micro-validation inventory.
- Stable required case identifiers reusable across tests, manifests, and later
  Task 5 artifacts.

**Risks / rollback**
- Risk: if Task 5 Story 1 relies on an implicit or drifting case list, the phase
  can appear locally validated while skipping part of the required contract.
- Rollback/mitigation: freeze the mandatory inventory in one canonical place and
  reuse the same case IDs everywhere.

### Engineering Task 2: Reuse The Canonical Exactness And Required-Noise Validation Kernels Without Forking Them

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- code | validation automation

**Definition of done**
- Task 5 Story 1 derives its local correctness evidence from the canonical
  `validate_squander_vs_qiskit.py` exactness path and the Task 4 Story 2
  required-noise wrapper rather than from a third overlapping micro-validation
  framework.
- The Task 5 local baseline uses the same frozen metric names, threshold values,
  and core case semantics already accepted lower in the stack.
- Story 1 remains a phase-level interpretation layer, not a replacement for the
  underlying exactness kernel.

**Execution checklist**
- [ ] Start from the canonical Story 2 exactness bundle and the Task 4 Story 2
      required local-noise bundle instead of designing a new low-level harness.
- [ ] Keep the comparison contract explicitly tied to `Re Tr(H*rho)` against
      Qiskit Aer density-matrix simulation.
- [ ] Reuse the existing threshold metadata and software metadata helpers where
      possible so Task 5 does not invent a second tolerance scheme.
- [ ] Keep the Task 5 Story 1 layer thin enough that lower-level case behavior
      still traces directly to the canonical validation code.

**Evidence produced**
- One Task 5 Story 1 assembly path rooted in the canonical exactness and
  required-noise kernels.
- Reviewable traceability from the phase-level local gate to the existing
  case-level validation surfaces.

**Risks / rollback**
- Risk: a parallel Task 5 micro-validation harness would create schema drift and
  conflicting interpretations of what counts as local correctness.
- Rollback/mitigation: treat the existing Story 2 and Task 4 Story 2 outputs as
  canonical and extend them only through thin Task 5-specific assembly logic.

### Engineering Task 3: Add Explicit Completeness And Status Checks For The Local Correctness Gate

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- code | tests | validation automation

**Definition of done**
- Task 5 Story 1 can distinguish `complete pass`, `failed`, and `incomplete`
  local-baseline states from machine-readable evidence.
- Missing mandatory case IDs, duplicate case IDs, missing status fields, or
  skipped required cases block Story 1 closure rather than disappearing inside a
  partial summary.
- The local correctness gate uses the frozen `100%` pass-rate rule on the full
  mandatory matrix.

**Execution checklist**
- [ ] Add or tighten one completeness helper that checks the mandatory inventory
      against the emitted case set for Story 1.
- [ ] Require explicit per-case status fields and an unambiguous aggregate
      pass/fail state for the local baseline.
- [ ] Make missing mandatory cases, missing status checks, or mismatched case IDs
      fail the Task 5 Story 1 gate explicitly.
- [ ] Keep partial subsets or hand-selected favorable cases from being summarized
      as local-baseline success.

**Evidence produced**
- Machine-readable completeness semantics for the Task 5 local correctness gate.
- Focused failure signals for missing mandatory cases or missing status fields.

**Risks / rollback**
- Risk: Story 1 can look green while silently omitting required cases if
  completeness is inferred only from aggregate counts.
- Rollback/mitigation: validate exact mandatory case identity and status fields,
  not just summary totals.

### Engineering Task 4: Preserve Auditable Required Coverage And Mixed-Sequence Order In The Phase-Level Local Baseline

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- code | validation automation

**Definition of done**
- The Task 5 Story 1 local-baseline view preserves auditable required coverage
  for gate families, required local-noise models, and the mixed required-noise
  sequence.
- Reviewers can see from the phase-level local-baseline output that the gate is
  not only numerically correct but also coverage-complete for the frozen local
  contract.
- Mixed-sequence order remains visible enough that Story 1 does not collapse the
  local baseline into scalar pass/fail alone.

**Execution checklist**
- [ ] Reuse the canonical per-case coverage signals from the existing Story 2 /
      Task 4 Story 2 result schema:
      `required_gate_coverage_pass`,
      `required_noise_model_coverage_pass`,
      `noise_sequence_match_pass`,
      `mixed_sequence_order_pass`, and `operation_audit_pass`.
- [ ] Ensure the phase-level Story 1 summary makes covered required models and
      mixed-sequence status visible without forcing readers to reverse-engineer
      lower-level code.
- [ ] Keep the local-baseline vocabulary aligned with the frozen required names:
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Avoid flattening away the ordered mixed-sequence semantics when assembling
      the Task 5 local-baseline view.

**Evidence produced**
- A Task 5 Story 1 local-baseline summary that preserves required coverage and
  mixed-sequence auditability.
- Reviewable proof that the local baseline covers the frozen required slice, not
  only aggregate exactness counts.

**Risks / rollback**
- Risk: a thin phase-level summary can accidentally hide which required models or
  ordered mixed sequence were actually validated.
- Rollback/mitigation: treat coverage and mixed-sequence visibility as part of
  the local-baseline schema, not optional debug output.

### Engineering Task 5: Add Focused Regression Tests For Task 5 Story 1 Completeness Rules

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the positive Task 5 Story 1 baseline assembly path
  and representative failure conditions for incomplete local evidence.
- Regression coverage is specific enough to localize whether a Story 1 failure is
  caused by a missing mandatory case, bad status semantics, or broken baseline
  summary logic.
- The fast test layer stays smaller than the full validation command while still
  protecting the phase-level local gate.

**Execution checklist**
- [ ] Extend `tests/density_matrix/test_density_matrix.py` or a tightly related
      successor with Task 5 Story 1 bundle or completeness tests.
- [ ] Add focused assertions for stable mandatory case IDs and explicit status
      fields on the Task 5 local-baseline output.
- [ ] Add at least one representative negative test showing that a missing or
      malformed mandatory case blocks Story 1 closure.
- [ ] Keep full matrix execution in the dedicated validation command rather than
      duplicating the whole baseline inside pytest.

**Evidence produced**
- Focused pytest coverage for Task 5 Story 1 local-baseline completeness.
- Reviewable failures that localize phase-level local correctness regressions.

**Risks / rollback**
- Risk: without focused completeness tests, Story 1 can regress at the manifest
  level while the lower-level exactness harness still looks healthy.
- Rollback/mitigation: add small schema- and completeness-level regression tests
  on top of the canonical case-level tests.

### Engineering Task 6: Emit One Stable Task 5 Story 1 Local-Correctness Artifact Or Rerunnable Command

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 1 can emit one stable machine-readable local-correctness artifact or one
  stable rerunnable command that defines the mandatory 1 to 3 qubit baseline for
  Task 5.
- The output records stable case identity, thresholds, status fields, required
  coverage, and aggregate pass/fail interpretation for the local gate.
- The artifact shape is stable enough that later Task 5 stories can extend it
  without replacing the Story 1 evidence format.

**Execution checklist**
- [ ] Build the Task 5 Story 1 output as a thin wrapper, manifest, or summary
      around the canonical micro-validation outputs rather than a duplicate case
      dump with divergent semantics.
- [ ] Record suite identity, thresholds, software metadata, mandatory case IDs,
      aggregate pass/fail status, and per-case status references or payloads.
- [ ] Keep the output narrow to the local correctness baseline rather than mixing
      in workflow-scale, optimization-trace, or publication-only fields.
- [ ] Make the artifact or command stable enough for later Task 5 stories and
      paper-facing bundle assembly to reference directly.

**Evidence produced**
- One stable Task 5 Story 1 local-correctness artifact or rerunnable command.
- A reusable output schema for later Task 5 baseline assembly.

**Risks / rollback**
- Risk: ad hoc local-baseline summaries will drift and make later Task 5
  workflow-scale and publication layers harder to audit.
- Rollback/mitigation: define one thin structured output now and extend it
  incrementally.

### Engineering Task 7: Document The Phase-Level Local Correctness Gate And Its Hand-Offs

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Task 5 Story 1 validates, how to rerun it,
  and why it is the canonical local correctness gate for the phase.
- The notes make clear that Story 1 sits above Task 2 Story 2 and Task 4 Story 2
  and below the workflow-scale, training-trace, and publication-oriented Task 5
  stories.
- The documentation stays aligned with the frozen Phase 2 thresholds and does
  not overclaim broader validation closure.

**Execution checklist**
- [ ] Document the Task 5 Story 1 validation entry point and its relationship to
      the canonical Story 2 and Task 4 Story 2 bundles.
- [ ] Make the mandatory local gate explicit:
      stable case IDs, `<= 1e-10` energy error, density-validity pass,
      trace-preservation, `|Im Tr(H*rho)| <= 1e-10`, and `100%` pass rate.
- [ ] Explain how Story 1 hands off workflow-scale pass/fail, training traces,
      broader stability metrics, optional evidence interpretation, and final
      provenance packaging to Stories 2 to 6.
- [ ] Keep optional secondary baselines and unsupported-case material clearly out
      of the Story 1 definition of done.

**Evidence produced**
- Updated developer-facing guidance for the Task 5 Story 1 local correctness
  gate.
- One stable place where Story 1 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 1 is poorly documented, later contributors may confuse it with
  the lower-level exactness harness or with the full Phase 2 validation package.
- Rollback/mitigation: tie the notes directly to the same command and artifact
  outputs used for Story 1 completion.

### Engineering Task 8: Run Story 1 Local-Baseline Validation And Confirm The Mandatory Micro-Validation Gate

**Implements story**
- `Story 1: The Mandatory 1 To 3 Qubit Micro-Validation Matrix Defines The Local Correctness Baseline`

**Change type**
- tests | validation automation

**Definition of done**
- The full mandatory Task 5 Story 1 local correctness gate runs successfully end
  to end.
- Every mandatory 1 to 3 qubit microcase satisfies the frozen exactness and
  density-validity thresholds, and the local-baseline completeness checks pass.
- Story 1 completion is backed by stable outputs and rerunnable commands rather
  than by code changes alone.

**Execution checklist**
- [ ] Run the focused Story 1 regression tests for the Task 5 local-baseline
      layer.
- [ ] Run the dedicated Story 1 validation command or artifact-emission path for
      the mandatory local correctness baseline.
- [ ] Verify `100%` pass rate on the full mandatory 1 to 3 qubit matrix and
      confirm no mandatory case is missing from the emitted output.
- [ ] Record the stable test run and artifact references for later Task 5
      stories and publication work.

**Evidence produced**
- Passing focused pytest coverage for Task 5 Story 1.
- A machine-readable local-correctness artifact or rerunnable command reference
  with a `100%` pass rate on the mandatory matrix.

**Risks / rollback**
- Risk: Story 1 can look complete while still lacking a reproducible proof that
  the local correctness gate is both full and auditable.
- Rollback/mitigation: treat the emitted local-baseline output and the complete
  pass rate as part of the exit criteria, not optional follow-up.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- the mandatory 1 to 3 qubit micro-validation inventory is frozen through one
  stable set of case identifiers,
- every mandatory microcase satisfies maximum absolute energy error
  `<= 1e-10`,
- every mandatory microcase satisfies `rho.is_valid(tol=1e-10)`,
  `|Tr(rho) - 1| <= 1e-10`, and `|Im Tr(H*rho)| <= 1e-10`,
- the local correctness gate enforces explicit completeness and status checks so
  missing or malformed mandatory cases cannot close Story 1,
- required gate and noise coverage, plus mixed-sequence order, remain auditable
  in machine-readable output,
- one stable validation command and one stable Task 5 Story 1 artifact or
  manifest define the local correctness baseline with a `100%` pass rate on the
  mandatory matrix,
- and workflow-scale pass/fail, optimization-trace and 10-qubit anchor
  evidence, broader execution-stability closure, optional/incomplete evidence
  interpretation, and final publication packaging remain clearly assigned to
  later Task 5 stories.

## Implementation Notes

- `benchmarks/density_matrix/validate_squander_vs_qiskit.py` already provides
  the canonical exactness kernel, threshold metadata, software metadata, and
  machine-readable micro-validation output shape. Story 1 should build on that
  kernel rather than duplicate it.
- `benchmarks/density_matrix/noise_support/required_local_noise_micro_validation.py`
  already adds the required-noise coverage, mixed-sequence-order, support-tier,
  and mandatory-baseline semantics most relevant to Task 5 Story 1. Story 1
  should reuse that wrapper or its bundle semantics directly where possible.
- `MANDATORY_MICROCASES` in
  `benchmarks/density_matrix/circuits.py` already carries the natural mandatory
  case inventory. Story 1 should preserve those case IDs instead of inventing a
  second Task 5-only naming scheme.
- `tests/density_matrix/test_density_matrix.py` already contains representative
  Story 2 microcase coverage and a Task 4 Story 2 bundle-schema test. Those are
  the most natural footholds for fast Story 1 regression coverage.
- `benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py` already consumes
  the canonical micro-validation bundle inside the larger workflow package.
  Story 1 should preserve compatibility where practical so later Task 5 stories
  can reuse the same artifact IDs and status semantics directly.
- Task 5 Story 1 is a phase-level local correctness gate, not a second low-level
  micro-validation framework. Its main job is to freeze completeness,
  interpretation, and auditability on top of the canonical lower-level evidence.
