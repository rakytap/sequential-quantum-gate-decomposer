# Counted manifest — M-F1a task-9

> **Status:** Frozen; Research Manager freeze record in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` §10 · **Slice:** M-F1a task-9 ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Planning-base HEAD:** `a81be56baba7e97156532524978cc878d347fa87` ·
> **Inventory:** `task-4/ROUTE_INVENTORY.md` ·
> **Traces:** REQ-001, REQ-004 · ADR-F1A-001, ADR-F1A-004 ·
> **No push/PR**

This is the 16-cell manifest the counted bundle must match. It is not counted evidence. No cell is dropped, merged, or swapped. Rows 2–16 use the C-slice default workload. Row 1 is the task-1 pin, not a C-slice default. q8 and q10 use different families because those are the defaults C.1 and C.2 froze, not because this slice chose them.

Research Manager accepted `claim_boundary` and `completeness_claim` below on 2026-10-06. The freeze record is in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` §10, not in this file.

## 1. Shared fields

| Field | Value |
|-------|--------|
| Anchors | 4, 6, 8, 10 only |
| `max_partition_qubits` | 2 (O-11) |
| Parameters | `build_initial_parameters(n)` at `benchmarks/density_matrix/partitioned_runtime/common.py:29`: `linspace(0.05, 0.05*n, n)` |
| `det` | `seed_policy` `deterministic_workload_no_random_seed` |
| `sfs` | `seed_policy` `structured_family_seed_20260318` |
| Order | rows 1–16 below. Exact-set checks this order |

`n` is `descriptor_set.parameter_count`. The counted exact-set check covers route, anchor, workload, `max_partition_qubits`, `seed_policy`, and `n`. A miss fails at `manifest_exact_set`.

## 2. Sixteen cells

| # | Route | q | Workload id | n | Seed | C.0 | Pinned by |
|---|-------|---|-------------|---|------|-----|-----------|
| 1 | `partitioned_density_descriptor_baseline` | 4 | `phase2_xxz_hea_q4_continuity` | 18 | det | `:44` | task-1 |
| 2 | `partitioned_density_descriptor_baseline` | 6 | `phase2_xxz_hea_q6_continuity` | 30 | det | `:45` | C.4 task-8 |
| 3 | `partitioned_density_descriptor_baseline` | 8 | `phase2_xxz_hea_q8_continuity` | 42 | det | `:46` | C.4 task-8 |
| 4 | `partitioned_density_descriptor_baseline` | 10 | `phase2_xxz_hea_q10_continuity` | 54 | det | `:47` | C.4 task-8 |
| 5 | `partitioned_density_descriptor_fused_unitary_islands` | 4 | `phase2_xxz_hea_q4_continuity` | 18 | det | `:48` | C.1 task-5 |
| 6 | `partitioned_density_descriptor_fused_unitary_islands` | 6 | `phase2_xxz_hea_q6_continuity` | 30 | det | `:49` | C.1 task-5 |
| 7 | `partitioned_density_descriptor_fused_unitary_islands` | 8 | `layered_nearest_neighbor_q8_sparse_seed20260318` | 72 | sfs | `:50` | C.1 task-5 |
| 8 | `partitioned_density_descriptor_fused_unitary_islands` | 10 | `layered_nearest_neighbor_q10_sparse_seed20260318` | 120 | sfs | `:51` | C.1 task-5 |
| 9 | `phase31_channel_native` | 4 | `phase31_local_support_q4_spectator_embedding_smoke` | 18 | det | `:52` | C.3 task-7 |
| 10 | `phase31_channel_native` | 6 | `mf1a_strict_spectator_embed_q6` | 27 | det | `:53` | C.3 task-7 (C.0 was `TBD inventory`; RM-1(a)) |
| 11 | `phase31_channel_native` | 8 | `mf1a_strict_spectator_embed_q8` | 36 | det | `:54` | C.3 task-7 (C.0 was `TBD inventory`; RM-1(a)) |
| 12 | `phase31_channel_native` | 10 | `mf1a_strict_spectator_embed_q10` | 45 | det | `:55` | C.3 task-7 (C.0 was `TBD inventory`; RM-1(a)) |
| 13 | `phase31_channel_native_hybrid` | 4 | `phase2_xxz_hea_q4_continuity` | 18 | det | `:56` | C.2 task-6 |
| 14 | `phase31_channel_native_hybrid` | 6 | `phase2_xxz_hea_q6_continuity` | 30 | det | `:57` | C.2 task-6 |
| 15 | `phase31_channel_native_hybrid` | 8 | `phase31_pair_repeat_q8_dense_seed20260318` | 138 | sfs | `:58` | C.2 task-6 (C.0 `:58` named the family, `dense`, and seed `20260318`, not the id) |
| 16 | `phase31_channel_native_hybrid` | 10 | `phase31_pair_repeat_q10_dense_seed20260318` | 228 | sfs | `:59` | C.2 task-6 (C.0 had no noise pattern or seed) |

The fused route id is `partitioned_density_descriptor_fused_unitary_islands`.

## 3. Builders (call only)

The counted module does not call these. It calls the five existing `build_cases(provenance=...)` functions. The builders are recorded so the freeze names the workload source. `workloads.py` is not edited.

| Rows | Builder |
|------|---------|
| 1 | `mf1a_q4_baseline_validation.build_cases`. `planner_surface/common.py::build_phase2_continuity_vqe(4)` plus `noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set`, `max_partition_qubits=2`. Entry `execute_partitioned_density` (`noisy_runtime_core.py:816`), `allow_fusion=False` |
| 2–4 | `mf1a_baseline_validation.build_cases`. Same continuity builders and the same baseline entry |
| 5–6 | same continuity builders. Entry `execute_partitioned_density_fused` (`noisy_runtime_core.py:968`) |
| 7–8 | `workloads.py::build_structured_descriptor_set("layered_nearest_neighbor", qbit_num=q, noise_pattern="sparse", seed=20260318, max_partition_qubits=2)`. ADR-F1A-005, call only. Same fused entry |
| 9 | `workloads.py::build_phase31_microcase_descriptor_set(id, max_partition_qubits=2)`. ADR-F1A-005, call only |
| 10–12 | `mf1a_strict_validation.py::build_mf1a_strict_spectator_descriptor_set(q)`. Not `workloads.py` |
| 13–14 | same continuity builders as rows 1–4. Entry `execute_partitioned_density_channel_native_hybrid` (`noisy_runtime_core.py:992`) |
| 15–16 | `workloads.py::build_phase31_structured_descriptor_set("phase31_pair_repeat", qbit_num=q, noise_pattern="dense", seed=20260318, max_partition_qubits=2)`. ADR-F1A-005, call only. Same hybrid entry |

Rows 9–12 use `execute_partitioned_density_channel_native` (`noisy_runtime_core.py:980`), `allow_fusion=False`.

## 4. `claim_boundary`

**Accepted by Research Manager, 2026-10-06.**

```text
Counted M-F1a denominator evidence for the ADR-F1A-001 route-by-anchor set: partitioned_density_descriptor_baseline, partitioned_density_descriptor_fused_unitary_islands, phase31_channel_native, and phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with max_partition_qubits 2, each against execute_sequential_density_reference under QA-001. No complete M-F1a, state-vector, external-protocol, Aer, energy, or frozen-matrix claim.
```

Per cell, the closeout says "`<route>` route verified at q`<n>`" with the route id from column 2. It does not say "baseline route verified" for a fused, strict, or hybrid cell. The milestone-level wording waits for the milestone closeout and the Research Manager report.

## 5. `completeness_claim`

**Accepted by Research Manager, 2026-10-06.** Value: `false`.

Reason: The counted bundle is the route-by-anchor denominator only. The Linux CI job (G-04, G6), the current-state docs (G-05, G7), the milestone closeout, the full milestone review, and the Research Manager report are still open. `true` would move the claim, which returns to Research Manager.

## 6. Freeze record

The freeze record is not in this file.

The freeze record is in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` §10 and names this file's sha256. Any edit after that record voids it.
