# Canonical noisy planner surface — concrete workload instances

Companion to [`CANONICAL_NOISY_PLANNER_SURFACE_SOURCES.md`](./CANONICAL_NOISY_PLANNER_SURFACE_SOURCES.md). This file lists **specific `workload_id` strings** (or patterns) wired in tests and benchmarks, with **`source_type`** and **where that pairing is defined or fixed**.

Unless noted, `source_type` is either passed explicitly to a builder or inherited from bridge metadata (`generated_hea` for Phase 2 continuity VQE bridges).

**Where the code lives:** Surface builders and continuity helpers are implemented in `squander/partitioning/noisy_planner_surface_builders.py` and re-exported from `squander.partitioning.noisy_planner`. Descriptor sets that wrap continuity surfaces are built in `squander/partitioning/noisy_descriptor.py` (`build_phase3_continuity_partition_descriptor_set`, etc.), also re-exported from `noisy_planner`.

---

## Table 1 — Continuity / bridge (`source_type`: `generated_hea`)

| workload_id | Where defined / fixed | Notes |
|-------------|----------------------|--------|
| `phase2_xxz_hea_q{N}_continuity` for `N ∈ {4, 6, 8, 10}` | Default when `workload_id` omitted: `squander/partitioning/noisy_planner_surface_builders.py` — `build_phase3_continuity_planner_surface` (format `phase2_xxz_hea_q{}_continuity` from `bridge_metadata["qbit_num"]`; import via `squander.partitioning.noisy_planner`) | Used across parametrized tests (`tests/partitioning/test_planner_surface_entry.py`, `test_planner_surface_descriptors.py`), fusion fixtures (`tests/partitioning/fixtures/fusion_cases.py`), benchmark continuity suites (`benchmarks/density_matrix/planner_surface/continuity_surface_validation.py`, `continuity_descriptor_validation.py`, `descriptor_ordering_validation.py`, `planner_audit_validation.py`), partitioned-runtime fusion selection (`benchmarks/density_matrix/partitioned_runtime/fusion_case_selection.py`), correctness evidence (`benchmarks/density_matrix/correctness_evidence/case_selection.py` explicit `workload_id=`), performance evidence (`benchmarks/density_matrix/performance_evidence/case_selection.py`), planner calibration (`benchmarks/density_matrix/planner_calibration/case_selection.py`). |
| `phase2_xxz_hea_q4_continuity` (explicit) | Literals in `tests/partitioning/test_planner_surface_entry.py` (`preflight_planner_request`), `tests/partitioning/test_planner_surface_descriptors.py` (`preflight_descriptor_request`), `benchmarks/density_matrix/planner_surface/unsupported_descriptor_validation.py` (`preflight_descriptor_request` negative case), `benchmarks/density_matrix/planner_surface/descriptor_ordering_validation.py` (metadata) | Same `source_type` as default continuity; ID pinned for preflight / descriptor-boundary cases. |

---

## Table 2 — Microcases (`source_type`: `microcase_builder`)

| workload_id | Where defined | Primary consumers |
|-------------|---------------|-------------------|
| `microcase_2q_entangler_local_depolarizing` | `case_name` in `mandatory_microcase_definitions()` — `benchmarks/density_matrix/planner_surface/workloads.py` and mirror `tests/partitioning/fixtures/workloads.py` | `build_microcase_surface` / `build_microcase_descriptor_set`; planner-surface benchmarks; correctness / calibration iterators; unsupported planner validation (base specs). |
| `microcase_3q_mixed_local_noise_sequence` | Same tuple in `mandatory_microcase_definitions()` | Same as above. |
| `microcase_4q_partition_boundary_triplet` | Same tuple in `mandatory_microcase_definitions()` | Same; also `descriptor_ordering_validation.py`, `unsupported_descriptor_validation.py` (tampered descriptors), `planner_calibration/case_selection.py` density-signal filter `DENSITY_SIGNAL_MICROCASE_IDS`. |

---

## Table 2B — Phase 3.1 counted microcases (`source_type`: `microcase_builder`)

These IDs are now present in the executable workload catalogs as **Phase 3.1
helpers**, but they are **not** yet wired into the default Phase 3 evidence
pipelines. They mirror the frozen Phase 3.1 counted microcase surface.

| workload_id | Where defined | Primary consumers |
|-------------|---------------|-------------------|
| `phase31_microcase_1q_u3_local_noise_chain` | `phase31_microcase_definitions()` in `benchmarks/density_matrix/planner_surface/workloads.py` and mirror `tests/partitioning/fixtures/workloads.py` | `build_microcase_surface`, `build_phase31_microcase_descriptor_set`, Phase 3.1 planning helpers in correctness evidence. |
| `phase31_microcase_2q_cnot_local_noise_pair` | Same | Same. |
| `phase31_microcase_2q_multi_noise_entangler_chain` | Same | Same. |
| `phase31_microcase_2q_dense_same_support_motif` | Same | Same. |

---

## Table 3 — Structured families (`source_type`: `structured_family_builder`)

**ID pattern** (canonical in code): `{family}_q{qbit_num}_{noise_pattern}_seed{seed}` with:

- `family ∈ {layered_nearest_neighbor, seeded_random_layered, partition_stress_ladder}` (`STRUCTURED_FAMILY_NAMES` in `benchmarks/density_matrix/planner_surface/workloads.py` and mirror `tests/partitioning/fixtures/workloads.py`)
- `qbit_num ∈ {8, 10}` (`STRUCTURED_QUBITS`)
- `noise_pattern ∈ {sparse, periodic, dense}` (`MANDATORY_NOISE_PATTERNS`)
- `seed` — see below

| Seed(s) | Where the workload_id is formed | Notes |
|---------|--------------------------------|--------|
| `20260318` | `DEFAULT_STRUCTURED_SEED` in `workloads.py` / fixtures; `build_structured_surface` / `iter_structured_surfaces` | Full Cartesian product: **3 × 2 × 3 = 18** IDs (e.g. `layered_nearest_neighbor_q8_sparse_seed20260318`). Used by default in correctness evidence, planner calibration, mandatory workload surface/descriptor benchmarks, descriptor ordering (first `iter_structured_descriptor_sets()` item), etc. |
| `20260318` only (hard-coded suffix) | `benchmarks/density_matrix/partitioned_runtime/mandatory_workload_runtime_validation.py` — builds `"{family}_q{qbit_num}_{noise_pattern}_seed20260318"` with `STRUCTURED_VALIDATION_NOISE_PATTERN` | Subset of the default seed grid (noise pattern fixed by that module’s constant). |
| `20260318`, `20260319`, `20260320` | `benchmarks/density_matrix/performance_evidence/case_selection.py` — `_structured_seed_noise_pairs()` uses `PERFORMANCE_EVIDENCE_PRIMARY_STRUCTURED_SEED` (= `DEFAULT_STRUCTURED_SEED`) and `PERFORMANCE_EVIDENCE_ADDITIONAL_STRUCTURED_SEEDS`; inventory builder enumerates the same | Extra structured cases for performance evidence (still `structured_family_builder`). |

---

## Table 3B — Phase 3.1 counted structured families (`source_type`: `structured_family_builder`)

These IDs are now present in the executable workload catalogs as **Phase 3.1
helpers**, but they are **not** yet part of the default Phase 3 evidence
package builders.

**ID pattern**: `{family}_q{qbit_num}_{noise_pattern}_seed{seed}` with:

- primary families `phase31_pair_repeat`, `phase31_alternating_ladder`,
- control family `layered_nearest_neighbor`,
- `qbit_num ∈ {8, 10}`,
- primary noise patterns `{periodic, dense}`,
- control noise pattern `{sparse}`,
- primary seeds `{20260318, 20260319, 20260320}`,
- control seed `20260318`.

| Family / seed policy | Where defined | Notes |
|----------------------|---------------|-------|
| `phase31_pair_repeat`, `phase31_alternating_ladder` with primary seeds/patterns | `iter_phase31_structured_surfaces()` / `iter_phase31_structured_descriptor_sets()` in `benchmarks/density_matrix/planner_surface/workloads.py` and mirror fixtures | Mirrors the frozen counted positive-method slice in Phase 3.1 planning. |
| `layered_nearest_neighbor` sparse control (`seed=20260318`) | Same helpers | Control family retained for “Phase 3 sufficient?” comparisons on the same evaluation grid. |

Phase 3.1 planning helpers also exist in:

- `benchmarks/density_matrix/correctness_evidence/case_selection.py`
- `benchmarks/density_matrix/performance_evidence/case_selection.py`

These helpers intentionally do **not** alter the default Phase 3 case caches
until Phase 3.1 implementation work wires them in.

---

## Table 4 — Legacy QGD lowering (`source_type`: `legacy_qgd_circuit_exact`)

| workload_id | Where defined | Purpose |
|-------------|---------------|---------|
| `legacy_manual_u3_cnot` | `tests/partitioning/test_planner_surface_entry.py`; `benchmarks/density_matrix/planner_surface/legacy_exact_lowering_validation.py` | Happy-path legacy lowering audit. |
| `legacy_manual_with_noise` | Same files | Legacy circuit + `density_noise` schedule. |
| `legacy_manual_with_h` | Same files | Expected failure (unsupported `H` gate). |
| `legacy_manual_bad_noise_index` | Same files | Expected failure (invalid `after_gate_index`). |
| `legacy_descriptor_audit` | `benchmarks/density_matrix/planner_surface/descriptor_audit_validation.py`; `tests/partitioning/test_planner_surface_descriptors.py` | Descriptor audit record for legacy slice. |
| `legacy_descriptor_shape` | `tests/partitioning/test_planner_surface_descriptors.py` (`test_descriptor_audit_records_share_shape_across_supported_workloads`) | Second legacy ID for schema-shape comparison only. |

---

## Table 5 — Test-only / non-enum `source_type`

| workload_id | source_type | Where defined | Notes |
|-------------|-------------|---------------|-------|
| `story3_singleton_unitary_segments` | `test` (not in `SUPPORTED_PLANNER_SOURCE_TYPES`) | `tests/partitioning/test_partitioned_runtime.py` — `build_canonical_planner_surface_from_operation_specs` | Bypasses preflight enum; used only to exercise fusion path metadata. |

---

## Table 6 — Bridge tampering and preflight negatives (`workload_id` in failing requests)

These IDs appear in **calls that are expected to fail** before a valid surface exists, or in **tampered bridge** builds that still use bridge `source_type` (`generated_hea`).

| workload_id | Declared / used in | Outcome |
|-------------|-------------------|---------|
| `broken_bridge_case` | `tests/partitioning/test_planner_surface_entry.py` — `build_canonical_planner_surface_from_bridge_metadata` with corrupted `bridge["operations"]` | `NoisyPlannerValidationError` (noise insertion); bridge metadata still implies `generated_hea`. |
| `unsupported_source_case` | `tests/partitioning/test_planner_surface_entry.py` — `preflight_planner_request(source_type="binary_import", ...)` | Rejected at preflight (`source_type` unsupported). |
| `missing_payload_case` | `test_planner_surface_entry.py` — preflight with no payload | `malformed_request`. |
| `unsupported_noise_case` | `test_planner_surface_entry.py` — preflight with `readout_noise` in specs | `noise_type` (with `microcase_builder`). |
| `wrong_mode_case` | `test_planner_surface_entry.py` — preflight `requested_mode="state_vector"` | `mode` (with `microcase_builder`). |
| `unsupported_source_type` | `benchmarks/density_matrix/planner_surface/unsupported_planner_validation.py` | Bundle negative case. |
| `missing_source_payload` | Same | Bundle negative case. |
| `unsupported_mode_claim` | Same (`generated_hea` + bridge + wrong mode) | Bundle negative case. |
| `unsupported_noise_model` | Same (`microcase_builder` + extra noise op) | Bundle negative case. |
| `legacy_gate_family_h` | Same (`legacy_qgd_circuit_exact` + `Circuit` with `H`) | Bundle negative case. |
| `invalid_noise_insertion_index` | Same (`microcase_builder` + bad `source_gate_index`) | Bundle negative case. |

---

## JSON artifacts

Committed bundles under `benchmarks/density_matrix/artifacts/**` echo the same `workload_id` / `source_type` pairs after validation scripts run. They are **outputs**, not definitions; the Python sources in the tables above are authoritative for intent.
