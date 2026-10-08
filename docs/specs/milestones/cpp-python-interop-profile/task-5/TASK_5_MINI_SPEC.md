# Task 5: attribution routes at widths 6 and 8
> **Status:** closed by `task-5/CLOSEOUT.md` · **Verdict:** C1 `3a8a0d74`; C2-w6 `78ba7108`; C2-w8 `3feffb07`; (g) PASS both widths · **Slice:** M-F5a task-5 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 · CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-006, ADR-F5A-010 ·
> **Scope:** three timed routes at widths 6 and 8, plus a required R-strict refusal row at each width. No overhead ratio. No reduction ·
> **Gate:** C1 `3a8a0d74`, C2-w6 `78ba7108`, C2-w8 `3feffb07`. (g) PASS at both widths. `task-5/CLOSEOUT.md` records the slice ·
> **Parent:** task-4 C1 `6be6282f`, C1-ET4 `431a5808`, C2 `5f63a9d6`, w4 routes bundle `6584be2b…` · handback `98eec857…` unchanged ·
> **RM:** `2026-10-08-mf5a-task4-w4-routes-interpret.md` (`788f688d…`). Attribution evidence at width 4 only. No claim generalises until widths 6 and 8 are interpreted

## 1. Why this slice

Width-4 attribution is counted. Q1b asks for the same three timed routes at widths 6 and 8, each with an R-strict refusal row, without waiting on a live R-strict path. One slice covers both widths because they share one flag, one schema, and one validator. REQ-004 stays open. `milestone_counted` stays false. CAP-004 stays hold-the-line. This file does not open M-F1b. Landing widths 6 and 8 completes the 4/6/8 three-route set that ADR-F5A-010 names before any option-C pack (Research Manager, then PhD Manager, then Zoltán). This slice proposes none.

## 2. Rows, divisors, and bundles

Each width writes its own bundle. The names are `interop_profile_bundle_routes_w6.json` and `interop_profile_bundle_routes_w8.json`, under `benchmarks/density_matrix/artifacts/interop_profile/`. Suite ids are `interop_attribution_routes_task5_w6_v1` and `interop_attribution_routes_task5_w8_v1`.

The descriptor is the same builder as task-4: unpack `vqe, _hamiltonian = build_task_evaluator(width)` and pass `vqe` to `build_phase3_continuity_partition_descriptor_set` with `max_partition_qubits` 2. A read-only probe at `5f63a9d6` matched the task-2 and task-3 pins:

| Width | Parameters | Operations | Gates | Noise | Workload label | Partitions | Divisor |
|-------|------------|------------|-------|-------|----------------|------------|---------|
| 6 | 30 | 18 | 15 | 3 | `phase2_xxz_hea_q6_continuity` | 7 | 73728 |
| 8 | 42 | 24 | 21 | 3 | `phase2_xxz_hea_q8_continuity` | 9 | 1572864 |

The divisors are the counted E-VQE bundles, not a new measurement. `interop_profile_bundle_w6.json` (`5257bad2…`) has `operation_count` 18 and throughput divisor 73728 (`18 × 4^6`). `interop_profile_bundle_w8.json` (`1712dce9…`) has `operation_count` 24 and throughput divisor 1572864 (`24 × 4^8`). The route divisor is that product when each timed route executes every operation. Width 4 stays 3072 (12 operations). Width 6 is 73728. Width 8 is 1572864. The executed-operation check is: the partition members total the bridge `operation_count`, and every partition is executed. Primitive-call counts are not operation counts. If that check fails, the run stops and hands back. The parameter vector is the E-VQE vector for that width (`build_initial_parameters` on `build_task_evaluator(width)`), the same workload ADR-F5A-006 already froze. It is not a second vector.

The workload labels are the builder's format strings. They are not a second circuit.

Timed rows are R-base, R-fused, and R-hybrid. Throughput numerator is the apply component. The one-sided bound, mean plus `1.644854 * s / sqrt(1000)` with `ddof=1`, applies to orchestration and to apply only. S-g is not retuned and is not applied. No row carries an overhead ratio, a lower twin, or a QA-007 ratio.

R-hybrid's label comes from the executed `partition_runtime_class` list. The ordered pins, which a counted list must match or the run hands back, are: width 6 `phase31_channel_native`, `phase3_unitary_island_fused`, `phase31_channel_native`, then four `phase3_unitary_island_fused`; width 8 the same prefix, then six `phase3_unitary_island_fused`. Width 4 keeps its existing five-class tuple and is not copied onto these widths. The lane and the validator take workload label, bridge counts, divisor, pinned command, and this tuple from the suite id or `qbit_num`. Width 4's constants stay unchanged.

## 3. Counted protocol

Per timed route, per width: discard 50 warm-up calls, then record 1000 calls. `clean_start` true, or write nothing. Affinity is CPU 0. The four thread values are `"1"`. `unset PYTHONPATH` is a required launch step on its own line. It is not inside the pinned `provenance.command`, matching the width-4 lane. `provenance.command` is that pinned string, not the process argv. The lane also pins itself to the lowest allowed CPU, so `affinity_cpu` 0 is the recorded field. The Tester log is the launch evidence.

Width 6. The pinned `provenance.command` is the `taskset` line only. `cd` and `unset PYTHONPATH` sit in the launch block and are not part of that string:

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
unset PYTHONPATH
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 6 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w6.json
```

Width 8, the same way. The pinned `provenance.command` is again the `taskset` line only:

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
unset PYTHONPATH
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 8 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w8.json
```

The Tester runs the widths in this order, and never concurrently. `clean_start` detection is not narrowed. First, width 6 from an empty porcelain at the implementation commit. That bundle's `implementation_revision` is that commit. Second, C2-w6 commits only `interop_profile_bundle_routes_w6.json`. Third, width 8 from an empty porcelain at C2-w6. That bundle's `implementation_revision` is C2-w6. Fourth, C2-w8 commits only `interop_profile_bundle_routes_w8.json`. `task-5/CLOSEOUT.md` is a later docs commit, not part of either C2. Width 8 is a background job and is not killed on a short timeout. This draft runs neither command. A rename of `R_STRICT_RAISE_CODE_W4` is optional and stays inside the allowlist; the raise code itself is width-independent.

## 4. Tester time budget

Allow 1 minute at width 6. Width 8 is a background job; allow 10 minutes. These sentences are planning allowances for the Tester. They are not route-cost results. R-strict refuses, with no timings.

Width-6 and width-8 route rows will be interpreted by the Research Manager. No claim generalises from width 4 until then. QA-008's fitness function for this slice is `qa008_route_categorical_exact(committed, regenerated)` in `attribution_route_validation.py`. It returns when 100 % of the categorical leaves of the two bundles are equal and raises otherwise. Categorical means every leaf except the timing leaves (each sample's `orchestration_ns` and `apply_component_ns`; `mean_ns` and `upper_bound_95_ns` under `orchestration` and `apply_component`; `mean_ns_per_op` and `upper_bound_95_ns_per_op` under `throughput`) and the run-identity leaves (`provenance.implementation_revision`, each `provenance.extension_identities[*].sha256`, and `provenance.density_matrix_cpp_sha256`). A key present in only one bundle fails. A clean-clone regeneration rebuilds the extensions, so the run-identity leaves may differ, as they did in the task-4 (g); the (g) record reports them, and they are not categorical. Throughput has no numeric margin.

## 5. Unsupported

- An overhead ratio on any route row. A lower twin. A QA-007 label on a route row.
- Closing REQ-004. `milestone_counted=true`. A live R-strict path. A timed R-strict row.
- The binding or dispatch reduction. A C++ edit. Kernel, fusion, AVX, or GPU work. An edit under `squander/**`.
- R-oracle. A second workload. Editing 17/9/0 or the 26-case matrix. A VQA campaign. M-F1b, in any form.
- Overwriting `interop_profile_bundle.json`, `interop_profile_bundle_w6.json`, `interop_profile_bundle_w8.json`, or `interop_profile_bundle_routes_w4.json`.
- "M-F5a complete", "attribution routes profiled", "reduction shipped", "four-route shipped", "REQ-004 met", speedup, or at-least-1.2×.
- A numeric regeneration margin on route throughput. Throughput is not a (g) gate.
- Demo GO. A push or a pull request. N8. An edit of ADR-F1A-006.

## 6. Evidence matrix

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-001, REQ-004 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | each of widths 6 and 8 passes as three timed rows plus the R-strict refusal row. A number on R-strict, a missing R-strict row, an overhead field, or the phrases "four-route shipped", "REQ-004 met", and "speedup" fails. The committed w4 routes bundle still passes | DS-1 |
| REQ-004, REQ-006 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q` | `--attribution-routes` accepts widths 4, 6, and 8 and refuses every other width and the three E-VQE names. Width 6 writes only `interop_profile_bundle_routes_w6.json`. Width 8 writes only `interop_profile_bundle_routes_w8.json`. Each routes name is bound to its width: at width 6 or 8, `interop_profile_bundle_routes_w4.json` (the committed `6584be2b…` file) or the other width's routes name fails before a route runs, and at width 4, `interop_profile_bundle_routes_w6.json` or `interop_profile_bundle_routes_w8.json` fails before a route runs. Width 4's default and pinned output stay `interop_profile_bundle_routes_w4.json`, so the task-4 reproduce command still runs. Replacing `test_attribution_routes_refuses_non_width_four` is an authorized contract change, not a weakened test. Without the flag, widths 4, 6, and 8 stay the E-VQE path and a `*_routes_*` name is refused. Under the flag, "writes only" binds inside the artifacts directory, compared by resolved path; outside it every name refusal above still holds and any other path is accepted (`test_attribution_output_width_bound_or_outside_artifacts`). A refused width raises `attribution routes allow widths 4, 6, and 8 only` | DS-2 |
| REQ-004 | doc review | this mini-spec §2 and §3 | divisors 73728 and 1572864, the two bundle names, and the two provenance commands | DS-1 |
| REQ-005, REQ-008 | repo review | `git diff --exit-code 5f63a9d60eba7ab59176addfa0ef071e4865c74a -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json` | empty | DS-2 |
| REQ-005, REQ-006, REQ-007, QA-009 | repo review | `git diff --exit-code 5f63a9d60eba7ab59176addfa0ef071e4865c74a -- squander benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py tests/VQE/test_VQE.py docs/density_matrix_project/archive docs/specs/milestones/cpp-python-interop-profile/INITIAL_REQUIREMENTS.md` | empty | DS-2 |
| REQ-005 | repo review | `git diff --name-only 064a6f4f62ff0af96febf7a5df3e2915b18db881 -- . ':(exclude)docs/specs'` | a subset of `benchmarks/density_matrix/interop_profile/attribution_route_lane.py`, `benchmarks/density_matrix/interop_profile/attribution_route_validation.py`, `benchmarks/density_matrix/interop_profile/validation_pipeline.py`, `tests/VQE/test_vqe_interop_harness.py`, and `tests/VQE/test_vqe_interop_bundle_validation.py`; from C2-w6 on, also `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w6.json`; from C2-w8 on, also `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w8.json`. Any other path fails. `docs/specs` is excluded because the code-ready, C0, and closeout commits change it lawfully | DS-2 |
| QA-008 | benchmark evidence pipeline, (g) per width | the §3 command for width 6 from a clean checkout of C2-w6, and for width 8 from a clean checkout of C2-w8, each with its `--output` under `/tmp` (`benchmarks/density_matrix/interop_profile/validation_pipeline.py`); then `qa008_route_categorical_exact(committed, regenerated)` | returns for both widths (§4); the regenerated files are not committed | DS-1 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | after `task-5/CLOSEOUT.md`, normal and `--strict` both exit 0 with no findings. No waiver. No placeholder closeout | DS-3 |

## 7. Verdict

**Closed by `task-5/CLOSEOUT.md`.** C1 is `3a8a0d74`. C2-w6 is `78ba7108`. C2-w8 is `3feffb07`. (g) is PASS at both widths. Lint: normal and `--strict` both exit 0 with no findings. REQ-004 stays open. The QA-007 10 % bar is unchanged. ADR-F5A-010's decision is unchanged. The Research Manager has not vetoed N-y; it stays FYI. Interpretation of the width-6 and width-8 rows belongs to the Research Manager.
