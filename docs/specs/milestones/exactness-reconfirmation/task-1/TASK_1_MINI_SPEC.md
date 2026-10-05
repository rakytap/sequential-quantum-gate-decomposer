# Task / Work Package 1: q4 baseline partitioned exactness tracer
> **Status:** planning review closed code-ready by Architect on 2026-10-05 under
> ADR-F1A-008 · **Slice:** M-F1a slice 1 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base revision:** `1cb3d20c` ·
> **Scope:** `phase2_xxz_hea_q4_continuity` through
> `partitioned_density_descriptor_baseline` against the sequential oracle ·
> **Traces:** REQ-001, REQ-002, REQ-004, REQ-005, REQ-006, REQ-007 ·
> QA-001, QA-008, QA-009 · ADR-F1A-001, ADR-F1A-002, ADR-F1A-004…006 ·
> **Authorization:** Research Manager G-01 and ADR-F1A-009 authorize Step 4b for the q4
> baseline cell only; C1 is `a50ae79f`; C2 is `a2928bf1` · **No push/PR**

## 1. Purpose and slice boundary

This slice tracer proves the thinnest end-to-end M-F1a path: one reviewed q4 baseline
partitioned cell, one sequential-oracle comparison, the accepted QA-001 predicate, one
revision-pinned sibling evidence record, the named regeneration command, and the
non-interference and historical boundaries.

It does not prove M-F1a completeness. It creates no case, task, or claim for fused, strict,
or hybrid execution, and none for the 6-, 8-, or 10-qubit anchors. REQ-003 is deferred
because baseline execution is not evaluation mode. REQ-008 remains a milestone-close
obligation because current-state docs describe shipped truth.

## 2. Frozen tracer cell

| Field | Slice contract |
|-------|----------------|
| Route | `partitioned_density_descriptor_baseline` |
| Existing entry | `execute_partitioned_density` with fusion disabled |
| Anchor | 4 qubits |
| Workload | `phase2_xxz_hea_q4_continuity` |
| Workload source | existing `build_phase2_continuity_vqe(4)` continuity fixture |
| Descriptor source | existing `build_phase3_continuity_partition_descriptor_set` |
| Planner setting | shipped default `max_partition_qubits = 2`, recorded explicitly |
| Parameters | existing deterministic `build_initial_parameters` vector, recorded in full |
| Seed policy | deterministic workload; no random seed |
| Oracle | `execute_sequential_density_reference` on the same descriptor and parameters |
| Required realization | requested and realized paths are baseline; partition count is greater than one; no region is classified `actually_fused`; exact output is present |

Architect review of this mini-spec is the independent review of this one tracer cell under
ADR-F1A-001. The cell is provisional slice evidence, not the frozen milestone denominator.
Its record must state `milestone_counted = false` and `completeness_claim = false`; later
milestone counting must re-execute it under the fully reviewed manifest.

## 3. Required behavior

### 3.1 Exact QA-001 predicate

For the candidate `rho` produced by the baseline route and the sequential-oracle output from
identical ordered operations and parameters, the slice records and requires:

- `||delta rho||_F <= 1e-10`;
- `max_ij |delta rho_ij| <= 1e-10`;
- `|Tr(rho) - 1| <= 1e-10`;
- `lambda_min(rho) >= -1e-12`; and
- every candidate entry and every reported residual is finite.

All clauses must pass. `lambda_min(rho)` is the minimum value returned by the existing
`DensityMatrix.eigenvalues()` contract: LAPACK `zheev`, upper triangle of `rho` as stored.
Finiteness is checked before calling the eigensolver; an eigensolver failure or non-finite
eigenvalue fails the record. No symmetrization, Hermiticity gate, or QA-010 measure is added.

Each quantity and pass flag is emitted. Failure identifies the first failing measure,
route, anchor, and workload. The slice then stops and reports the disagreement; it does not
change the oracle, predicate, workload, parameters, planner setting, or case boundary.

### 3.2 Slice record and claim boundary

The sibling M-F1a record must include:

- schema and manifest versions;
- route, anchor, workload, planner setting, deterministic parameter vector, and seed policy;
- requested and realized paths, partition count, exact-output flag, and fused-region summary;
- all QA-001 values and pass flags, the first failure or null, and `qa001_pass`;
- `milestone_counted = false`, `completeness_claim = false`, and a claim-boundary statement
  that this is only the q4 baseline tracer;
- no claim that M-F1a or protocol (i)–(v) plus Aer is complete.

Aer and energy-continuity data are absent from the tracer's counted predicate. If inherited
context is present elsewhere in the command's older outputs, it remains explicitly
non-counted and cannot change this record or bundle status. The tracer does not open M4.

The persisted contract is frozen as follows:

| Item | Slice value |
|------|-------------|
| Owning module | `benchmarks/density_matrix/correctness_evidence/mf1a_q4_baseline_validation.py` |
| Builder convention | existing `build_cases` / `build_artifact_bundle` registry contract |
| Manifest schema | `correctness_evidence_mf1a_q4_baseline_manifest_v1` |
| Record schema | `correctness_evidence_mf1a_q4_baseline_case_v1` |
| Bundle schema | `correctness_evidence_mf1a_q4_baseline_bundle_v1` |
| Output | `benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json` |
| Canonical workload import | `benchmarks.density_matrix.planner_surface.common.build_phase2_continuity_vqe` |
| Canonical parameter import | `benchmarks.density_matrix.partitioned_runtime.common.build_initial_parameters` |
| Allowed dependencies | standard library, numpy, existing planner/runtime/density bindings, and correctness-evidence I/O; no qiskit, performance evidence, cost selection, or older counted predicate |
| Content identity | SHA-256 over bytes for the extension and any consumed input artifact |

Required JSON fields have stable types: schema ids, route, workload, claim boundary, count
status, and first failure are strings or null; anchor/count fields are integers; parameters
and dirty paths are arrays; provenance and predicate values are objects; pass/count flags are
booleans. Required fields may be extended only by a new schema version.

### 3.3 Provenance and regeneration

Before artifact writes, the tracer captures the full clean implementation revision (distinct
from the planning-base revision), tracked/untracked clean-state result, exact command, `qgd`
interpreter/environment identity, dependency versions,
extension identity, the frozen cell, tolerances, full parameter vector, and identities of
any consumed input artifacts. It emits `clean_start` and `provenance_pass`. A dirty or
incomplete pre-run makes the tracer bundle and command fail as well as remaining non-counted.

The existing named command remains:
`conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`.
It emits the separately versioned M-F1a tracer bundle under the existing correctness
artifact root without changing older schemas or artifacts. A repeated run at the same
manifest version reproduces categorical fields exactly and keeps each residual within its
own frozen comparator, or the tracer bundle fails. Specifically, each current value must
first pass QA-001 and then satisfy
`abs(current - prior) <= 1e-10` for `frobenius_norm_diff`, `max_abs_diff`, and
`trace_abs_deviation`, and `abs(current - prior) <= 1e-12` for `lambda_min`.
The compared keys are
`cases[0].qa001.{frobenius_norm_diff,max_abs_diff,trace_abs_deviation,lambda_min}`.
Input and extension identities are sorted arrays of objects with exactly `path` and
`sha256` keys; both keys compare exactly.

The process-exit aggregate for the existing command adds the sibling
`correctness_evidence_mf1a_q4_baseline_bundle_v1` result and includes every other currently
registered suite except exactly:

- `correctness_evidence_external_correctness`; and
- the whole `correctness_evidence_output_integrity` suite, whose status is the conjunction
  of output integrity and continuity energy.

The command exits 0 only when the sibling bundle is present with `status = pass` and every
other included registered suite passes. It exits 1 when the sibling bundle is missing or
fails, including provenance, QA-001, or regeneration mismatch, or when another included suite
fails. The two excluded suites retain their own recorded statuses unchanged. Aer or energy
failure cannot change the sibling status or the M-F1a acceptance exit. This contract changes
only the existing command's aggregate set; it adds no second top-level command.

### 3.4 Non-interference and historical boundaries

The slice changes no public runtime API, channel, planner objective, CI trigger, or
state-vector default. Local upstream-suite execution is preflight. The actual
`build-and-test-linux` result counts for closure only when invoked through the existing
`workflow_dispatch` trigger. A pull request is not a closure path and is not authorized.

The frozen archive, performance-evidence tree, transitive Phase-3.1 workload inventory, and
counted-matrix test remain unchanged against `1cb3d20c`. The frozen 26-case matrix is not
executed, relabelled, recounted, or adopted by this slice. Kraus bundles remain primary;
Choi and Liouville remain witnesses.

## 4. Unsupported behavior

- Any non-baseline route or any anchor other than q4 in the tracer bundle.
- Any workload other than `phase2_xxz_hea_q4_continuity`.
- A fused realization presented as baseline success.
- Any Hermiticity, symmetrized-eigenvalue, QA-010, Aer, or energy gate in QA-001.
- New channels or changes to delivered local depolarizing, amplitude-damping, or
  phase-damping semantics.
- Timing, cost-selection, optimizer, AVX, GPU, VQA, gradient, or at-least-1.2x work.
- Changes to the frozen 26-case assets, runtime public APIs, CI triggers, roadmap, product
  statement, initial requirements, or current-state docs.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / CI gate | Expected result | Owner artifact |
|----------|---------------|-------------------|-----------------|----------------|
| REQ-002, REQ-006, QA-001 | fast fitness and q4 acceptance | `conda run -n qgd --no-capture-output pytest tests/partitioning/test_partitioned_runtime.py -m "density_matrix and not slow" -k "mf1a_q4_baseline or mf1a_qa001" -v` | the real q4 baseline output passes every exact predicate clause; inclusive boundaries, non-finite values, eigensolver errors, and deterministic first-failure ordering are pinned | DS-1; ET-1 |
| REQ-001, REQ-004, QA-008 | record/manifest contract | `conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_q4_baseline -v` | exactly one provisional tracer cell; route realization, provenance, schema, and claim boundary validate | DS-2; ET-2 |
| REQ-004, REQ-006, QA-008 | pipeline acceptance | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | aggregate includes the present/passing sibling and every registered suite except exactly external correctness and the whole output-integrity suite; exits 1 for missing/failing sibling or any other included failure; excluded statuses remain recorded and cannot affect M-F1a acceptance | DS-3; ET-3 |
| REQ-007 | static historical review | `git diff --exit-code 1cb3d20c -- docs/density_matrix_project/archive/phases/phase-3-1 benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/planner_surface/workloads.py tests/partitioning/evidence/test_phase31_counted_matrix_validation.py` plus the same paths under `git status --porcelain --untracked-files=all --` | both outputs empty; no historical timing builder runs | DS-4; ET-4 |
| REQ-005, QA-009 | local preflight and closure CI | `conda run -n qgd --no-capture-output pytest tests/ -x -v --tb=line --ignore=tests/decomposition/test_wide_circuit_optimization.py`; closure gate `.github/workflows/ci.yml` job `build-and-test-linux` through `workflow_dispatch` only | local preflight passes; at closure the authorized manual CI run passes; defaults remain unchanged | DS-4; ET-4 |
| REQ-001, REQ-002, REQ-004…007 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` | zero errors; slice remains explicitly incomplete until a future closeout exists | this mini-spec |

## 6. Affected interfaces

- Existing runtime interfaces used unchanged:
  `execute_partitioned_density`, `execute_sequential_density_reference`,
  `DensityMatrix.eigenvalues()`.
- Additive evidence surface: one separately versioned M-F1a q4-baseline tracer record and
  bundle, registered under the existing correctness pipeline and artifact root.
- Test surfaces extended: existing baseline runtime test module and correctness-evidence
  validator module.
- Breaking changes: none.

## 7. Release and rollback

The slice is additive. Rollback removes only the M-F1a tracer registration, its sibling
schema/record/bundle, its tests, and generated tracer artifact. Runtime behavior, public
interfaces, older evidence, historical assets, CI configuration, and current-state docs
remain untouched.

## 8. Closed code-ready planning verdict

**Planning review closed as code-ready by Architect on 2026-10-05 under ADR-F1A-008, for the
q4 baseline cell only (`phase2_xxz_hea_q4_continuity` on
`partitioned_density_descriptor_baseline`).** Step 5 `CLOSEOUT.md` is shipped for that cell.
C2 is `a2928bf1`. No waiver or
placeholder was used. The G-07 process-exit aggregate is unchanged.
