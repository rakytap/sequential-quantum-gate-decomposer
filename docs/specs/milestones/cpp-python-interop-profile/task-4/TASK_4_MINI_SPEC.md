# Task 4: four attribution routes, no overhead ratio
> **Status:** C2 `5f63a9d6` · width-4 routes counted · **Slice:** M-F5a task-4 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 · CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-005, ADR-F5A-006, ADR-F5A-010 ·
> **Scope:** R-base, R-fused, R-hybrid timed at width 4. R-strict is a required refusal row. No \(O\). No reduction ·
> **Gate:** SDD stage `step-4b-authorized`. C1 `431a5808`. C2 `5f63a9d6`. Width-4 rows do not close REQ-004 ·
> **RM:** Option A 2026-10-07 (`b0edc658…`). Q1a 2026-10-07 (`5e3c8222…`). ADR-F5A-010 is in the w4 lane ·
> **Tip:** `5f63a9d60eba7ab59176addfa0ef071e4865c74a` · bundle `6584be2b…` · handback `98eec857…` unchanged · three counted E-VQE bundles stay ·
> **Pair, inventory, no-O rule, kernel/fusion/AVX boundary:** unchanged

## 1. Why this slice is the thinnest next row

E-VQE at 4, 6, and 8 is counted, and the Research Manager has interpreted that set.
The remaining counted inventory in ADR-F5A-001 is the four attribution routes, as one
set. They share one no-\(O\) rule, one descriptor handback, and the width-4 anchor
already pinned in task-1. One draft covers all four. A route-by-route split would
repeat that handback. Widths 6 and 8 for these routes are task-5. G-08, G-09, and the milestone
close stay later.

The authorized planning sentence is: "E-VQE equal-work interop overhead measured at
4/6/8 qubits under S-g Measure; one-sided 95 % UB on \(O\) is below 5 % at every width
(A4 false → CAP-004 hold-the-line); QA-007 10 % bar frozen and met on those cells."
It is not a route-row label, not a bundle phrase, and not "M-F5a complete".

| Candidate | Why it waits |
|-----------|----------------|
| Route rows at 6 and 8 | REQ-004 still needs those widths before milestone close. This slice does not close REQ-004 |
| R-oracle | E1 default-exclude. Task-1 already labels C++ `apply_to` from the E-VQE pair. This slice adds no diagnosis row |
| Binding or dispatch reduction | A4 has fired. CAP-004 is hold-the-line. The reduction is not made |
| G-08, G-09, Demo, full-milestone review | Close-time work. Demo stays No GO. No Opus review in this draft |
| Estimator change | S-g stays the E-VQE rule. This slice does not retune it |

## 2. What a route row is

Width-4 rows do not close REQ-004. Widths 6 and 8 remain for a later slice.
Each row calls one existing public entry on the descriptor built below:

| Id | Entry | Apply label |
|----|-------|-------------|
| R-base | `execute_partitioned_density` | C++ `NoisyCircuit.apply_to` |
| R-fused | `execute_partitioned_density_fused` | the executed apply |
| R-strict | `execute_partitioned_density_channel_native` | numpy Kraus |
| R-hybrid | `execute_partitioned_density_channel_native_hybrid` | the executed class |

The descriptor is the existing
`build_phase3_continuity_partition_descriptor_set` in
`squander/partitioning/noisy_descriptor.py`, which calls
`build_phase3_continuity_planner_surface` in
`squander/partitioning/noisy_planner_surface_builders.py`. Unpack
`vqe, _hamiltonian = build_task_evaluator(4)` from
`benchmarks/density_matrix/interop_profile/interop_lane.py` and pass `vqe`, not
the tuple. The surface is `vqe.describe_density_bridge()` (`source_type`
`generated_hea`), not a hand-built operation list. A structural probe at `6654ede4`
accepted it: parameters 18, operations 12, gates 9, noise 3, gate sequence U3, U3,
CNOT, U3, U3, CNOT, U3, U3, CNOT, and noise `local_depolarizing`,
`amplitude_damping`, `phase_damping`. `max_partition_qubits` stays the default 2.
The builder's default label `phase2_xxz_hea_q4_continuity` is not a second circuit.
If that call raises, or the counts disagree with the bridge, the slice hands back
and does not invent specs.

Throughput is nanoseconds per density-matrix entry. The numerator is the route
apply component, as task-1 §7. The divisor is `operation_count * 4^4` =
\(12\times 256 = 3072\). Orchestration time, \(T_\mathrm{public}\), and
\(T_\mathrm{lower}\) are not the numerator. The one-sided 95 % bound, mean plus
`1.644854 * s / sqrt(n)` with `ddof=1`, applies to orchestration time and to the
apply component only. S-g Measure stays the E-VQE \(O\) estimator and is not
retuned and not applied here. The row publishes no \(O\), no \(T_\mathrm{lower}\)
twin, and no QA-007 ratio. `milestone_counted` stays false. Warm-up, affinity, and
the single-thread launch are the N-34 rule. The width-4 counted run is C2 `5f63a9d6`.
This file does not re-run it.

## 3. Unsupported

- \(O\) on any attribution route. A lower-boundary twin. A QA-007 label on a route row.
- Closing REQ-004 from width-4 rows alone. Widths 6 and 8 remain.
- The binding or dispatch reduction. Kernel, fusion, AVX, or GPU edits. A C++ edit.
- R-oracle, unless a later diagnosis row carries the E1 sentence. This slice does not.
- Estimator change. Dropping samples. A new CPU mask. VQA. The 26-case matrix.
- Overwriting `interop_profile_bundle.json`, `interop_profile_bundle_w6.json`, or `interop_profile_bundle_w8.json`.
- "M-F5a complete", "attribution routes profiled", "reduction shipped", speedup, at-least-1.2×, or GHA.
- Demo GO. A push or a pull request. N8. An edit of ADR-F1A-006.
- `milestone_counted=true`. A live R-strict path. Closing REQ-004. "Four-route shipped". Width-6 or width-8 route campaigns inside this slice.

## 3a. Refusal row and the counted width-4 command

ADR-F5A-010 is in the width-4 lane. C1 is `431a5808`. C2 is `5f63a9d6`. The counted bundle is `interop_profile_bundle_routes_w4.json` (`6584be2b…`). The R-strict row is required, `status` is `handback_refused`, and it cites `channel_native_noise_presence` plus STEP_4A_HANDBACK `98eec857`. Any timing, ns/op, upper bound, or \(O\) on that row fails. Omitting the row fails. Widths 6 and 8 are `task-5/`, each with a refusal row. The live raise code at those widths is `channel_native_noise_presence`, the same code as width 4. `pure_unitary_partition` is the hybrid classifier route reason. Do not wait on a live R-strict path. Throughput is not a (g) gate. This pack sets no numeric regeneration margin on route throughput.

The counted protocol is not the C1 builder. That builder uses 3 samples, no warm-up, and no provenance. `--attribution-routes` counted mode discards 50 warm-up calls per timed route and then records 1000 calls per timed route. ADR-F5A-004 requires at least 1000 counted calls after discarded warm-up. The bundle pins `provenance.command` equal to the command below, warm-up 50, affinity, the four thread values `"1"`, the arithmetic-mean estimator and the one-sided bound on orchestration and apply, the implementation revision, host, CPU, compiler and flags, dependencies, workload, and claim boundary. `clean_start` must be true or nothing is written. The divisor is 3072. Claim boundary records the wrap set: R-hybrid does not wrap `NoisyCircuit.apply_to` (N-1). Both `perf_counter_ns` reads sit inside the apply-timer `with`, after the patches are installed and before they are restored, so the patch cost is not in orchestration (N-2).

The width-4 Developer step landed in `431a5808`. It touched `attribution_route_lane.py`, `attribution_route_validation.py`, `validation_pipeline.py` (additive `--attribution-routes` only; without the flag, `--width` 4, 6, and 8 and the three E-VQE validators stay as at `95204ac6…`, and a `*_routes_*` output name is refused), and `tests/VQE/test_vqe_interop_harness.py` plus `tests/VQE/test_vqe_interop_bundle_validation.py`. Negatives: a number on the R-strict row fails; a missing R-strict row fails; \(O\) on any row fails. N-19: the validator rejects the phrases "four-route shipped", "REQ-004 met", and "speedup". The width-6 and width-8 lift, and the validator depth checks, are task-5. Do not edit `squander/**`, the three counted E-VQE bundles, or the width-4 routes bundle.

The counted width-4 command, already run for C2, is:

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 4 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json
```

`--width` without `--attribution-routes` stays the E-VQE path. The flag refuses the three counted filenames.

## 4. Evidence matrix

The width-4 counted bundle is C2. This matrix does not re-time it.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-001, REQ-004 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | four ids pass only as three timed rows plus the R-strict refusal row, with no \(O\). A number on R-strict, a missing R-strict row, or \(O\) on any row fails. The phrases "four-route shipped", "REQ-004 met", and "speedup" fail (N-19). The three counted bundles are not the fixtures | DS-1 |
| REQ-004, REQ-006 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q` | `--attribution-routes` refuses the three counted names before any route runs. Without the flag, `--width` 4, 6, and 8 are unchanged, and a `*_routes_*` name is refused | DS-2 |
| REQ-004, REQ-008 | doc review | this mini-spec §2 | the four entries are the ADR-F5A-001 names. Unpack `vqe` from `build_task_evaluator(4)` before `build_phase3_continuity_partition_descriptor_set`. Width-4 rows do not close REQ-004 | DS-1 |
| REQ-005 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` | empty. No reduction diff | DS-2 |
| REQ-006 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | no `task-4/CLOSEOUT.md` and no `task-5/CLOSEOUT.md`. Normal mode exits 0. `--strict` exits 1 on task-4 `SLICE_MISSING_CLOSEOUT` because that stage is `step-4b-authorized`. Task-5 is `step-4a`, so its absent closeout stays a warning in both modes. No waiver. No placeholder | DS-3 |

## 5. Verdict

**Width-4 C2 landed. Not a milestone close.** SDD stage stays `step-4b-authorized`. C1 is `431a5808`. C2 is `5f63a9d6`. The routes bundle is `6584be2b…`. C2 binder `/tmp/rev-mf5a-t4-c2-w4/REVIEW.md` (`7b48bbc6…`). Tester evidence `/tmp/mf5a-t4-counted-w4/`. Q2 and B1–B3 landed in `6be6282f`; C1 adds the ADR-F5A-010 refusal row and counted mode. Q1 is Option A. Width-4 rows do not close REQ-004. REQ-004 stays open for R-strict and for widths 6 and 8. Task-5 is not-ready. Section 4 keeps the refusal negatives and the N-19 phrase list. N-5 and N-8 through N-11 stay test or process carries from the N-7 re-fix. Checklist §13 records that C2 was bundle-only.

| Finding | Disposition |
|---------|-------------|
| A route row might be given an \(O\) | §3. No lower twin |
| The anchor might have no descriptor | §2. Existing builder; bridge counts 18 / 12 / 9 / 3. A raise is a handback |
| A4 might be read as permission to reduce | RM ALIGN. CAP-004 is hold-the-line |
| The authorized sentence might be copied into a bundle | §1. Planning text only |
| R-oracle might be added to fill the apply label | E1. Task-1 already labels `apply_to` |
| `INITIAL_REQUIREMENTS.md` still says `[confirm]` | The RM upload freezes the bar for the E-VQE cells. The file sentence waits for milestone close |
