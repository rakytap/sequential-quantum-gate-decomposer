# Engineering tasks — M-F1a slice 1: q4 baseline partitioned tracer
> **Status:** planning review closed code-ready by Architect on 2026-10-05 under
> ADR-F1A-008 · **Slice:** M-F1a slice 1 ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Scope:** red-first goals for `TASK_1_MINI_SPEC.md` ·
> **Traces:** REQ-001, REQ-002, REQ-004, REQ-005, REQ-006, REQ-007 ·
> QA-001, QA-008, QA-009 ·
> **Stop rule:** route/oracle disagreement is reported through planning handback; never
> change the oracle, predicate, workload, parameters, planner setting, or denominator ·
> **Authorization:** Research Manager G-01 and ADR-F1A-009 authorize Step 4b for the q4
> baseline cell only; local C1 awaits Reviewer clearance · **No push/PR**

## ET-1 — Prove the q4 baseline output under the exact QA-001 predicate

**Implements delivery story**
- DS-1 — A researcher receives the exact q4 baseline verdict.

**Change type**
- tests | evidence code

**Definition of done**
- One reusable predicate, shared by the fast acceptance test and tracer bundle, reports the
  four accepted QA-001 quantities, finiteness, individual pass flags, `qa001_pass`, and the
  first failing measure.
- Finiteness is evaluated before the eigensolver.
- `lambda_min(rho)` uses the existing `DensityMatrix.eigenvalues()` result on `rho` as
  produced; no symmetrization, Hermiticity gate, `rho_is_valid`, or QA-010 enters the result.
- The real q4 continuity workload uses shipped default partition width 2, realizes the
  baseline path with more than one partition and no actually-fused region, and passes the
  predicate against `execute_sequential_density_reference`.
- A genuine disagreement fails with route, anchor, workload, and first failing measure and
  causes planning handback rather than contract changes.

**Tests to write first in the existing baseline runtime test surface**
- `test_mf1a_q4_baseline_matches_sequential_under_exact_qa001`
- `test_mf1a_qa001_rejects_nonfinite_entry_before_eigensolver`
- `test_mf1a_qa001_rejects_frobenius_or_max_abs_excess`
- `test_mf1a_qa001_rejects_trace_excess`
- `test_mf1a_qa001_rejects_lambda_below_floor`
- `test_mf1a_qa001_rejects_eigensolver_failure`
- `test_mf1a_qa001_rejects_nonfinite_eigenvalue_or_reported_residual`
- `test_mf1a_qa001_accepts_values_exactly_on_each_inclusive_threshold`
- `test_mf1a_qa001_reports_first_failure_in_frozen_evaluation_order`
- `test_mf1a_qa001_uses_existing_eigenvalues_contract_on_asymmetric_input`

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the failing positive and negative tests to `tests/partitioning/test_partitioned_runtime.py`
- [ ] Run the focused fast command and confirm failures encode the missing predicate/record
- [ ] Add the smallest evidence-layer behavior that satisfies the frozen predicate
- [ ] Refactor without changing fields, ordering, or thresholds; rerun the focused tests
- [ ] Stop and create planning handback if the real route disagrees with the oracle

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/partitioning/test_partitioned_runtime.py -m "density_matrix and not slow" -k "mf1a_q4_baseline or mf1a_qa001" -v`
- QA-001 fitness evidence for REQ-002 and REQ-006.

**Risks / rollback**
- Risk: an older validity helper silently adds Hermiticity or a different eigenvalue floor.
- Mitigation: negative tests and a single predicate source forbid older helper semantics.
- Rollback: remove only the additive predicate integration and focused tests.

## ET-2 — Emit a truthful provisional tracer record with complete provenance

**Implements delivery stories**
- DS-2 — A reviewer sees a truthful tracer boundary.
- DS-3 — A reproducer regenerates the tracer from the named command.

**Change type**
- tests | evidence code

**Definition of done**
- The slice-scoped manifest contains exactly the reviewed q4 baseline cell and rejects any
  extra or missing executed cell.
- The record contains every route-realization, QA-001, schema, planner-setting, parameter,
  provenance, and claim-boundary field required by the mini-spec.
- `milestone_counted` and `completeness_claim` remain false.
- Clean tracked/untracked state is captured before artifact writes. Dirty or incomplete
  provenance makes the tracer explicitly non-counted and fails the tracer bundle/command.
- Schema ids, canonical builders, field types, SHA-256 identity policy, and output path match
  the persisted-contract table in the mini-spec.
- Aer and energy-continuity status cannot change the tracer predicate, status, or count
  status.

**Tests to write first in the existing correctness-evidence validator surface**
- `test_mf1a_q4_baseline_manifest_accepts_only_the_reviewed_cell`
- `test_mf1a_q4_baseline_record_has_complete_provenance_and_claim_boundary`
- `test_mf1a_q4_baseline_dirty_pre_run_is_non_counted`
- `test_mf1a_q4_baseline_aer_and_energy_context_are_non_counted`
- `test_mf1a_q4_baseline_record_rejects_fused_realization`

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add failing contract tests to `tests/partitioning/evidence/test_correctness_evidence.py`
- [ ] Confirm red for missing manifest, provenance, and boundary fields
- [ ] Add the minimum sibling record and validation behavior
- [ ] Refactor without importing older counted predicates or performance evidence
- [ ] Rerun focused fast and validator commands

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_q4_baseline -v`
- Manifest/provenance evidence for REQ-001, REQ-004, REQ-006, and QA-008.

**Risks / rollback**
- Risk: a passing tracer is mistaken for a complete milestone.
- Mitigation: mandatory false completeness/count flags and explicit claim-boundary tests.
- Rollback: remove the additive sibling schema/record and its tests; older records remain.

## ET-3 — Regenerate the sibling tracer bundle through the named command

**Implements delivery story**
- DS-3 — A reproducer regenerates the tracer from the named command.

**Change type**
- tests | evidence code | tooling

**Definition of done**
- The existing correctness pipeline registers one separately versioned q4 baseline tracer
  bundle at the exact path and schema versions frozen in the mini-spec.
- Provenance capture occurs before any artifact write.
- The bundle status depends on q4 route realization, exact-set validation, QA-001, provenance,
  and regeneration comparison only; Aer and energy context are non-counted.
- Repeated execution at the same revision and manifest version reproduces categorical fields
  exactly; each current QA-001 value independently passes, then absolute current-versus-prior
  drift is at most `1e-10` for Frobenius/max-abs/trace and `1e-12` for `lambda_min`, using
  the exact JSON keys frozen in the mini-spec.
- Missing fields, categorical drift, excessive residual drift, or QA-001 disagreement makes
  the tracer bundle fail and identifies the first failure.
- Older package schemas, records, artifacts, and statuses remain separately identified and
  unchanged.
- The process-exit aggregate adds
  `correctness_evidence_mf1a_q4_baseline_bundle_v1`, requires it to be present and passing,
  and retains every other currently registered suite except exactly
  `correctness_evidence_external_correctness` and the whole
  `correctness_evidence_output_integrity` suite.
- The command exits 0 only when the sibling and every other included suite pass. It exits 1
  when the sibling is missing or fails provenance, QA-001, or regeneration, or when another
  included suite fails.
- The two excluded suites keep their own recorded statuses. Aer or energy failure cannot
  change sibling status or M-F1a acceptance exit. No second top-level command is added.

**Tests to write first in the existing correctness-evidence validator surface**
- `test_mf1a_q4_baseline_bundle_schema_and_summary`
- `test_mf1a_q4_baseline_pipeline_registration_and_prewrite_provenance`
- `test_mf1a_q4_baseline_regeneration_accepts_frozen_residuals`
- `test_mf1a_q4_baseline_regeneration_rejects_categorical_or_residual_drift`
- `test_mf1a_q4_baseline_failure_reports_route_anchor_workload_and_measure`
- `test_mf1a_q4_baseline_exit_aggregate_requires_present_passing_sibling`
- `test_mf1a_q4_baseline_exit_aggregate_excludes_exactly_external_and_output_integrity`
- `test_mf1a_q4_baseline_exit_aggregate_keeps_every_other_registered_suite`
- `test_mf1a_q4_baseline_exit_aggregate_preserves_excluded_statuses`
- `test_mf1a_q4_baseline_exit_aggregate_fails_for_sibling_or_included_suite`

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add failing bundle and integration tests first
- [ ] Confirm red for absent sibling registration and regeneration fields
- [ ] Add the minimum pipeline integration without changing historical package semantics
- [ ] Run focused validators, then the named pipeline from a clean pre-run state
- [ ] Rerun to exercise regeneration comparison
- [ ] Refactor without changing the command, artifact root, or claim boundary

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_q4_baseline -v`
- `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
- Regeneration evidence for REQ-004, REQ-006, REQ-007, and QA-008.

**Risks / rollback**
- Risk: process-exit aggregation accidentally drops another registered suite or rewrites an
  excluded suite's own status.
- Mitigation: exact-set aggregate tests plus separately versioned sibling status.
- Rollback: remove only the tracer registration, sibling bundle behavior, tests, and
  generated tracer artifact.

## ET-4 — Prove non-interference and historical immutability

**Implements delivery story**
- DS-4 — A maintainer sees upstream and historical behavior untouched.

**Change type**
- tests | verification

**Definition of done**
- Focused boundary tests prove the tracer does not import or reuse older validity/counting
  predicates, qiskit/Aer gates, cost-selection artifacts, or performance evidence.
- The local upstream-suite preflight passes with state-vector default and density opt-in
  checks unchanged.
- Static tracked/untracked review shows no change to the archive, performance-evidence tree,
  transitive Phase-3.1 workload inventory, or counted-matrix test.
- No CI configuration changes, pull request, performance/timing command, or current-state
  doc edits occur in the slice.
- The actual Linux CI closure gate remains `build-and-test-linux` through the existing
  `workflow_dispatch` trigger only; it is not invoked or claimed by Step 4a.

**Tests to write first**
- `test_mf1a_q4_baseline_evidence_respects_import_and_claim_boundaries` in
  `tests/partitioning/evidence/test_correctness_evidence.py`

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the failing boundary test
- [ ] Make it pass by keeping the tracer inside the sibling evidence boundary
- [ ] Run the local upstream preflight
- [ ] Run both static historical-review commands and record empty outputs
- [ ] Do not dispatch CI, alter triggers, or create a pull request in this pass

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_q4_baseline -v`
- `conda run -n qgd --no-capture-output pytest tests/ -x -v --tb=line --ignore=tests/decomposition/test_wide_circuit_optimization.py`
- The static review commands in `TASK_1_MINI_SPEC.md` §5.

**Risks / rollback**
- Risk: broad evidence imports execute out-of-scope Aer, energy, or timing behavior.
- Mitigation: boundary test and sibling-only imports.
- Rollback: remove the boundary test with the sibling tracer; protected paths stay unchanged.

## Trace rollup

| Task | Stories | Requirements | QA fitness | Command / lane |
|------|---------|--------------|------------|----------------|
| ET-1 | DS-1 | REQ-002, REQ-006 | QA-001 | `pytest tests/partitioning/test_partitioned_runtime.py -k "mf1a_q4_baseline or mf1a_qa001"` · fast pytest |
| ET-2 | DS-2, DS-3 | REQ-001, REQ-004, REQ-006 | QA-008 | `pytest tests/partitioning/evidence/test_correctness_evidence.py -k mf1a_q4_baseline` · evidence validators |
| ET-3 | DS-3 | REQ-004, REQ-006, REQ-007 | QA-008 | `python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` · correctness pipeline |
| ET-4 | DS-4 | REQ-005, REQ-007 | QA-008, QA-009 | `pytest tests/ -x` plus mini-spec §5 static review · local preflight / repository review |

REQ-003 and REQ-008 are explicitly deferred by `DELIVERY_STORIES.md`; no task in this
slice implements them.

## Closed code-ready planning verdict

**Planning review closed as code-ready by Architect on 2026-10-05 under ADR-F1A-008, for the
q4 baseline cell only (`phase2_xxz_hea_q4_continuity` on
`partitioned_density_descriptor_baseline`). Strict `SLICE_MISSING_CLOSEOUT` is recorded as
expected until Step 4b. No waiver, no placeholder.** Every task has an owning story,
objective done criteria, named red-first tests in existing test surfaces, evidence lanes,
risks, rollback, and the closed G-07 exit aggregate. This verdict records code-ready planning
only; G-01 governs the separate q4-only implementation authorization. No `CLOSEOUT.md`
exists before delivery.
