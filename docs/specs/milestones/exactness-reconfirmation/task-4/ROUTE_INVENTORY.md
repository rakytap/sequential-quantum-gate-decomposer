# Advertised-route inventory — M-F1a Slice C.0 (G-03)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect · not counted evidence ·
> **Slice:** M-F1a C.0 · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Inventory revision:** `0e8299e9f48361ece2cc1665d81d4b2a35b4f156` ·
> **Extension sha256 (on-disk, not rebuilt):**
> `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` ·
> **Date:** 2026-10-05 · **Reviewer:** Squander Architect C.0 re-close (`/tmp/c0-step4a/C0_STEP4A_RECLOSE.md`) ·
> **Traces:** REQ-001 · ADR-F1A-001, ADR-F1A-003 · **No push/PR**

This artifact is not counted evidence. The shipped q4 cell is baseline route verified.
Claim for Slice B: q4 baseline regenerated at clean C2. Unknown cells stay
`TBD inventory`. No workload file was edited.

## Shared fields

| Field | Value for every row |
|-------|---------------------|
| `advertised` | `yes` |
| `advertisement_source` | `docs/specs/ROADMAP.md` delivered M3 (partitioned execution with unitary-island fusion) and M3A (strict/hybrid channel-native); `docs/specs/ARCHITECTURE_OVERVIEW.md` §2 partitioned-density runtime and §3 noisy-planning language. Exports alone do not define advertisement (ADR-F1A-001). |
| `entry_location` | Definitions in `squander/partitioning/noisy_runtime_core.py`; re-exported from `squander/partitioning/noisy_runtime.py` lines 24–27 and 54–57 |
| `parameter_and_seed_policy` | `build_initial_parameters` is `np.linspace(0.05, 0.05 * n, n)`. Counted cells use `benchmarks/density_matrix/partitioned_runtime/common.py:29` (the copy the q4 cell uses); `tests/partitioning/fixtures/runtime.py:22` is the mirror. Structured seeds use `DEFAULT_STRUCTURED_SEED` `20260318` (`benchmarks/density_matrix/planner_surface/workloads.py:21`) |
| `historical_protection` | `ADR-F1A-005 protected` where the builder is `benchmarks/density_matrix/planner_surface/workloads.py` (strict q4, fused q8/q10, and any hybrid row that uses a structured family): call only, never edit or re-label; covered by the REQ-007 static diff. `none` for continuity rows (`planner_surface/common.py` is not on the ADR-F1A-005 list) |
| `max_partition_qubits` (pinned, O-11) | `2` for every cell of all four routes (Architect, C.0 Step 4a review, 2026-10-05). Equals `DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS` (`squander/partitioning/noisy_types.py:7`) and the shipped q4 cell (`mf1a_q4_baseline_validation.py:47`). The historical Phase-3 correctness bundles used `span_budget_q4` (4); they are context only and not comparable cell for cell. Each C-slice manifest records `planner_setting.max_partition_qubits = 2`. A change returns to Research Manager. |

Public entries (all present; no `entry_missing` row):

| `public_entry` | Line |
|----------------|------|
| `execute_partitioned_density` | `noisy_runtime_core.py:816` |
| `execute_partitioned_density_fused` | `noisy_runtime_core.py:968` |
| `execute_partitioned_density_channel_native` | `noisy_runtime_core.py:980` |
| `execute_partitioned_density_channel_native_hybrid` | `noisy_runtime_core.py:992` |

Oracle for every later counted cell: `execute_sequential_density_reference` (`noisy_runtime_core.py:1011`). QA-001 and G-07 stay the q4 contracts. This inventory changes neither.

The `tests/partitioning/fixtures/` copies used by the cited tests are byte-identical mirrors (Architect check).

## Sixteen rows

`shown` only when an existing test calls the route's public entry on the named workload at this anchor at budget 2 and asserts the route witness. Committed artifacts, tests of other entries, and other budgets are `existing_evidence`.

| route_id | q | candidate_workload_id | workload_source | realization_rule | witness fields | realized | evidence | existing_evidence | flag | proposed_slice |
|----------|---|----------------------|-----------------|------------------|----------------|----------|----------|-------------------|------|----------------|
| `partitioned_density_descriptor_baseline` | 4 | `phase2_xxz_hea_q4_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(4)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | requested and realized baseline | `runtime_path` baseline, `actual_fused_execution` false | `shown` | shipped q4 cell; baseline route verified | M-F1a q4 bundle | `none` | shipped; not in C.4 |
| `partitioned_density_descriptor_baseline` | 6 | `phase2_xxz_hea_q6_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(6)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | same | `runtime_path` baseline | `shown` | `test_phase3_partitioned_runtime_continuity_runtime_executes_supported_anchor` parametrizes `[4, 6]` | Phase-3 continuity test; not an M-F1a QA-001 record | `none` | C.4 |
| `partitioned_density_descriptor_baseline` | 8 | `phase2_xxz_hea_q8_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(8)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | same | `runtime_path` baseline | `not_shown` | builder exists; the continuity runtime test does not include 8 | `benchmarks/density_matrix/artifacts/partitioned_runtime/continuity_runtime/continuity_runtime_bundle.json` (Phase-3, `fb865857`; baseline realized, 9 partitions, budget 2) | `none` | C.4 |
| `partitioned_density_descriptor_baseline` | 10 | `phase2_xxz_hea_q10_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(10)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | same | `runtime_path` baseline | `not_shown` | builder exists; no baseline-path test at 10 | same `continuity_runtime_bundle.json` (Phase-3, `fb865857`; baseline realized, 11 partitions, budget 2) | `none` | C.4 |
| `partitioned_density_descriptor_fused_unitary_islands` | 4 | `phase2_xxz_hea_q4_continuity` | same continuity builders | ≥1 genuinely fused region | `actual_fused_execution`, `fused_region_count` | `not_shown` | eligibility surface records `eligible_unitary_region_count > 0` on the first continuity case while `runtime_path` stays baseline (`fused_eligibility_validation.py`) | Phase-3 `sequential_correctness` and `runtime_classification` bundles show the fused request `actually_fused` on continuity q4–q10 at budget 4 (`span_budget_q4`). At budget 2, `benchmarks/density_matrix/artifacts/partitioned_runtime/fused_eligibility/eligibility_bundle.json` shows 4 fusable regions on q4 continuity. Hybrid S08 labels 3 partitions `phase3_unitary_island_fused` via `_execute_partition_with_optional_fusion(allow_fusion=True)`. q4 M-F1a cell is baseline, `fused_region_count` 0 | `none` | C.1 |
| `partitioned_density_descriptor_fused_unitary_islands` | 6 | `phase2_xxz_hea_q6_continuity` | same continuity builders | same | same | `not_shown` | continuity iterator includes 6; no test asserts `fused_region_count > 0` at 6 on the fused entry at budget 2 | same budget-4 Phase-3 bundles. Hybrid S10 labels 5 partitions `phase3_unitary_island_fused` via the same fusion call | `none` | C.1 |
| `partitioned_density_descriptor_fused_unitary_islands` | 8 | `layered_nearest_neighbor_q8_sparse_seed20260318` | `benchmarks/density_matrix/planner_surface/workloads.py::build_structured_descriptor_set("layered_nearest_neighbor", qbit_num=8, noise_pattern="sparse", seed=20260318)` at budget 2 | same | same | `shown` | `tests/partitioning/evidence/test_partitioned_runtime_fusion_matrices.py::test_partitioned_runtime_structured_fused_runtime_executes_representative_cases` (outside default collection; needs `-o addopts=""`). Its first-passing selection is the named id per `artifacts/partitioned_runtime/structured_fused_runtime/structured_fused_runtime_bundle.json` (`fused_region_count` 12). q8 is also covered by `tests/partitioning/test_partitioned_runtime_fusion.py:28-38` in the default lane. C.1 freezes the id in its manifest and never re-selects by outcome | The same q8 workload realized baseline at budget 4 (Phase-3 `sequential_correctness` bundle) | `none` | C.1 |
| `partitioned_density_descriptor_fused_unitary_islands` | 10 | `layered_nearest_neighbor_q10_sparse_seed20260318` | `benchmarks/density_matrix/planner_surface/workloads.py::build_structured_descriptor_set("layered_nearest_neighbor", qbit_num=10, noise_pattern="sparse", seed=20260318)` at budget 2 | same | same | `shown` | same test; first-passing selection is the named id (`fused_region_count` 20). C.1 freezes the id and never re-selects by outcome | recorded at budget 2 in that bundle; budget-4 Phase-3 bundles are a different pin | `none` | C.1 |
| `phase31_channel_native` | 4 | `phase31_local_support_q4_spectator_embedding_smoke` | `benchmarks/density_matrix/planner_surface/workloads.py::build_phase31_microcase_descriptor_set` | strict request executes an eligible motif | motif `candidate_kind`, `fused_region_count` | `shown` | `test_phase31_channel_native_public_4q_smoke_matches_sequential` (`fused_region_count == 2`, targets `(0, 1)` and `(2, 3)`) | Phase-3.1 smoke; not M-F1a counted | `none` | C.3 |
| `phase31_channel_native` | 6 | `TBD inventory` | no existing builder output at q6 is strict-eligible at budget 2: continuity and all five structured families (any noise pattern or seed; `build_structured_descriptor_set` accepts `qbit_num=6`) have pure-unitary partitions, so strict preflight raises `channel_native_noise_presence` (Architect C.0 review probe) | same | same | `not_shown` | no strict-entry test at 6 | none on a historical builder; RM-1(a) adds an M-F1a-only family at C.3 | `no_eligible_workload` | C.3 |
| `phase31_channel_native` | 8 | `TBD inventory` | no existing builder output at q8 is strict-eligible at budget 2; `phase31_pair_repeat_q8_dense_seed20260318` has 23 of 34 partitions pure-unitary | same | same | `not_shown` | Architect C.0 review classification probe at `0e8299e9` | none on a historical builder; RM-1(a) adds an M-F1a-only family at C.3 | `no_eligible_workload` | C.3 |
| `phase31_channel_native` | 10 | `TBD inventory` | no existing builder output at q10 is strict-eligible at budget 2; `phase31_pair_repeat_q10_dense_seed20260318` has 38 of 56 partitions pure-unitary | same | same | `not_shown` | Architect C.0 review classification probe at `0e8299e9` | none on a historical builder; RM-1(a) adds an M-F1a-only family at C.3 | `no_eligible_workload` | C.3 |
| `phase31_channel_native_hybrid` | 4 | `phase2_xxz_hea_q4_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(4)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | hybrid path, full per-partition labels, ≥1 channel-native partition | `partition_runtime_class`, `partition_route_reason` | `shown` | `test_phase31_s08_e01_counted_hybrid_continuity_q4_matches_sequential_oracle`: 5 partitions, 2 `phase31_channel_native`, 3 `phase3_unitary_island_fused` | Phase-3.1 continuity; not M-F1a counted | `none` | C.2 |
| `phase31_channel_native_hybrid` | 6 | `phase2_xxz_hea_q6_continuity` | `benchmarks/density_matrix/planner_surface/common.py::build_phase2_continuity_vqe(6)` plus `squander/partitioning/noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | same | same | `shown` | `test_phase31_s10_e01_…q6…`: 7 partitions, 2 channel-native, 5 fused | same | `none` | C.2 |
| `phase31_channel_native_hybrid` | 8 | `phase31_pair_repeat` q8 dense seed `20260318` | `benchmarks/density_matrix/planner_surface/workloads.py::build_phase31_structured_descriptor_set` | same | same | `not_shown` | `test_phase31_hybrid_structured_pair_repeat_q8_dense_smoke` checks hybrid `runtime_path` and a Frobenius bound; it does not assert a channel-native partition | non-counted smoke | `none` | C.2 |
| `phase31_channel_native_hybrid` | 10 | `phase31_pair_repeat` q10 | `benchmarks/density_matrix/planner_surface/workloads.py::build_phase31_structured_descriptor_set` | same | same | `not_shown` | builder exists; no hybrid test at 10 | none | `none` | C.2 |

## Counts

| Bucket | Count |
|--------|------:|
| Rows | 16 |
| `advertised` yes | 16 |
| `entry_missing` | 0 |
| `not_advertised` | 0 |
| `realized` shown | 7 (baseline 4 and 6; fused 8 and 10; strict 4; hybrid 4 and 6) |
| `realized` not_shown | 9 |
| `realized` contradicted | 0 |
| `no_eligible_workload` | 3 (strict q6, q8, q10; historical builders only) |
| `other` | 0 |

## Flags and RM-1(a)

- **Strict q6, q8, q10 `no_eligible_workload`.** At the pinned budget 2 no existing builder yields a strict-eligible workload at q6, q8, or q10 (Architect C.0 review, classification-only probe at `0e8299e9`); only q4 has one. ADR-F1A-001 keeps all three cells. This inventory drops none and changes no counted set, oracle, tolerance, or G-07.
- **RM-1(a), decided.** Those three cells are realizable via a new M-F1a-only strict-eligible family designed in C.3. The family is a width-n generalization of the q4 spectator-embedding motif so every budget-2 partition carries a noisy U3/CNOT motif and clears `channel_native_noise_presence`. Ops and rates: U3 and CNOT; local depolarizing / amplitude damping / phase damping at 0.10 / 0.05 / 0.07. Comparison is the sequential oracle, not a self-match. Residuals about 1e-15 to 1e-13 are expected; a value near 1e-10 is a finding. Concrete ids, builders, seeds, and `planner_setting.max_partition_qubits = 2` freeze in the C.3 manifest before any counted run. The new module is not `workloads.py` and not a Phase-3 or Phase-3.1 builder. RM-1(a) rejects option (b) as the Architect review stated it (record the three cells as not realizable). Narrowing strict to microcase widths is a separate non-option under ADR-F1A-001. C.3 Step 4a includes that design and cannot close until the design is reviewed and frozen. C.1 and C.2 do not wait on it.
- **No oracle, QA-001, counted-set, or G-07 change** is proposed.
- **O-11.** Pinned at 2 for all four routes (Architect). A change returns to Research Manager.

## Proposed C-slice order

C.2 (hybrid) and C.3 (strict) stay separate (Architect, C.0 review). Do not merge any slices. Fused and channel-native groups stay ahead of baseline. C.1 and C.2 proceed through their own Step 4a, code-ready, and ADR-F1A-009 closes without waiting on the new workload.

| Slice | Route | Anchors | Close |
|-------|-------|---------|-------|
| C.1 | fused unitary islands | 4, 6, 8, 10 | ADR-F1A-009 (a)–(g); does not wait on C.3 |
| C.2 | hybrid channel-native | 4, 6, 8, 10 | ADR-F1A-009 (a)–(g); does not wait on C.3 |
| C.3 | strict channel-native | 4, 6, 8, 10 | ADR-F1A-009 (a)–(g) after the RM-1(a) family is reviewed and frozen in that slice's Step 4a |
| C.4 | baseline | 6, 8, 10 (q4 shipped) | ADR-F1A-009 (a)–(g) |

C.0 itself does not run `validation_pipeline.py`. Its artifact is this file. C.0 is committed alone by ET-C0-4 before any C.1 write; it is not folded into C.1's C1. That commit is not a counted (c)/(g) run.

## Test-runtime notes for later C-slices

q10 wall time is `TBD — measure` before any q10 run (plan §7). A 10-qubit density matrix is 2^20 complex entries. Run q10 in a new tmux session; do not attach to another agent's session. Selectors stay targeted (`-k` / `-m "not slow"`). `tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2` stays deselected. Do not fix it. Do not claim a full state-vector lane is green.
