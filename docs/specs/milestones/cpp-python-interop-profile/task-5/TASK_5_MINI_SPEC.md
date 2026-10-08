# Task 5: attribution routes at widths 6 and 8
> **Status:** not-ready · **Slice:** M-F5a task-5 · **Verdict:** Step 4a docs gate, not Step 4b ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 · CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-006, ADR-F5A-010 ·
> **Scope:** three timed routes at widths 6 and 8, plus a required R-strict refusal row at each width. No overhead ratio. No reduction ·
> **Gate:** SDD stage `step-4a`. No stamp. No counted w6 or w8 run in this draft ·
> **Parent:** task-4 C1 `431a5808`, C2 `5f63a9d6`, w4 routes bundle `6584be2b…` · handback `98eec857…` unchanged

## 1. Why this slice

Width-4 attribution is counted. Q1b asks for the same three timed routes at widths 6 and 8, each with an R-strict refusal row, without waiting on a live R-strict path. One slice covers both widths because they share one flag, one schema, and one validator. REQ-004 stays open. `milestone_counted` stays false. CAP-004 stays hold-the-line. This file does not open M-F1b.

## 2. Rows, divisors, and bundles

Each width writes its own bundle. The names are `interop_profile_bundle_routes_w6.json` and `interop_profile_bundle_routes_w8.json`, under `benchmarks/density_matrix/artifacts/interop_profile/`. Suite ids are `interop_attribution_routes_task5_w6_v1` and `interop_attribution_routes_task5_w8_v1`.

The descriptor is the same builder as task-4: unpack `vqe, _hamiltonian = build_task_evaluator(width)` and pass `vqe` to `build_phase3_continuity_partition_descriptor_set` with `max_partition_qubits` 2. A read-only probe at `5f63a9d6` matched the task-2 and task-3 pins:

| Width | Parameters | Operations | Gates | Noise | Workload label | Partitions | Divisor |
|-------|------------|------------|-------|-------|----------------|------------|---------|
| 6 | 30 | 18 | 15 | 3 | `phase2_xxz_hea_q6_continuity` | 7 | 73728 |
| 8 | 42 | 24 | 21 | 3 | `phase2_xxz_hea_q8_continuity` | 9 | 1572864 |

The divisors are the counted E-VQE bundles, not a new measurement. `interop_profile_bundle_w6.json` (`5257bad2…`) has `operation_count` 18 and throughput divisor 73728 (`18 × 4^6`). `interop_profile_bundle_w8.json` (`1712dce9…`) has `operation_count` 24 and throughput divisor 1572864 (`24 × 4^8`). The route divisor is that product when each timed route executes every operation, as width 4 did (12 operations, divisor 3072). Primitive-call counts are not operation counts. If a counted route executes a different operation count, the run stops and hands back.

The workload labels are the builder's format strings. They are not a second circuit.

Timed rows are R-base, R-fused, and R-hybrid. Throughput numerator is the apply component. The one-sided bound, mean plus `1.644854 * s / sqrt(1000)` with `ddof=1`, applies to orchestration and to apply only. S-g is not retuned and is not applied. No row carries an overhead ratio, a lower twin, or a QA-007 ratio.

R-hybrid's label comes from the executed `partition_runtime_class` list. The probe saw, at width 6, two `phase31_channel_native` and five `phase3_unitary_island_fused`, and at width 8, two `phase31_channel_native` and seven `phase3_unitary_island_fused`. A counted list that disagrees is a handback. The width-4 five-class tuple is not reused.

## 3. Counted protocol

Per timed route, per width: discard 50 warm-up calls, then record 1000 calls. `clean_start` true, or write nothing. Affinity is CPU 0. The four thread values are `"1"`. `unset PYTHONPATH` is a required launch step on its own line. It is not inside the pinned `provenance.command`, matching the width-4 lane. `provenance.command` is that pinned string, not the process argv. The lane also pins itself to the lowest allowed CPU, so `affinity_cpu` 0 is the recorded field. The Tester log is the launch evidence.

Width 6, after `cd` to the repo and `unset PYTHONPATH`:

```bash
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 6 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w6.json
```

Width 8, the same way:

```bash
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 8 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w8.json
```

The Tester runs width 8 as a background job and does not kill it on a short timeout. This draft runs neither command.

## 4. Tester time budget

Allow 1 minute at width 6. Width 8 is a background job; allow 10 minutes. These sentences are planning allowances for the Tester. They are not route-cost results. R-strict refuses, with no timings.

Width-6 and width-8 route rows will be interpreted by the Research Manager. No claim generalises from width 4 until then.

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
| REQ-004, REQ-006 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q` | `--attribution-routes` accepts widths 6 and 8 and refuses every other width and the three E-VQE names. Without the flag, widths 4, 6, and 8 stay the E-VQE path and a `*_routes_*` name is refused | DS-2 |
| REQ-004 | doc review | this mini-spec §2 and §3 | divisors 73728 and 1572864, the two bundle names, and the two provenance commands | DS-1 |
| REQ-005, REQ-008 | repo review | `git diff --exit-code 5f63a9d60eba7ab59176addfa0ef071e4865c74a -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | task-5 stage `step-4a` keeps its absent closeout a warning in both modes. No waiver. No placeholder closeout | DS-3 |

## 7. Verdict

**Not-ready.** SDD stage stays `step-4a`. Step 4b is not authorized. Task-5 cannot be stamped code-ready while `task-4/CLOSEOUT.md` is absent: task-4's strict `SLICE_MISSING_CLOSEOUT` error fails the SDD code-ready gate, which allows only task-5's own step-4a absent-closeout warning. The code-ready pass reads that closeout. No counted width-6 or width-8 command runs from this text. REQ-004 stays open. The QA-007 10 % bar is unchanged. ADR-F5A-010's decision is unchanged; the Q1b paragraph in the amendments file is a wording correction of the raise code.
