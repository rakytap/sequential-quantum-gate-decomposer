# `CanonicalNoisyPlannerSurface` — provenance

Primary implementation: `squander/partitioning/noisy_planner.py`.

---

## Stored provenance

`CanonicalNoisyPlannerSurface` carries `**source_type**` and `**workload_id**` as the workload identity for audits. The `**provenance**` property and nested copy inside `**to_dict()**` include only:


| Field         | Role                                                                               |
| ------------- | ---------------------------------------------------------------------------------- |
| `source_type` | Supported Phase 3 origin label (`SUPPORTED_PLANNER_SOURCE_TYPES`).                 |
| `workload_id` | Instance id (continuity default, microcase name, structured id, or caller string). |


Related top-level fields (not inside `provenance`): `schema_version`, `requested_mode`, qubit/parameter counts, operations.

`NoisyPartitionDescriptorSet` and runtime handoff types mirror the same `**provenance**` shape (`source_type` + `workload_id` only).

---

## Grouping and logging

Separate “entry route” or “workload family” strings are **not** part of the Phase 3 planner or descriptor contract: provenance is `source_type` plus `workload_id` only. Benchmarks and tools that need a coarse grouping or display label should use `**source_type**` (and optionally `**workload_id**` for finer grain). For example, planner calibration metadata may set `workload_family` to the same value as `source_type` when a single extra grouping field is convenient for downstream records.

---

## Builders and preflight

- `**build_canonical_planner_surface_from_bridge_metadata**`: `bridge_metadata`, `workload_id`, optional `source_type` override (else `bridge_metadata["source_type"]`).
- `**build_canonical_planner_surface_from_operation_specs**`: `qbit_num`, `source_type`, `workload_id`, `operation_specs`.
- `**preflight_planner_request**`: `source_type`, `workload_id`, and exactly one of `bridge_metadata`, `operation_specs`, or `legacy_circuit`; operation-spec branch requires `qbit_num`.
- `**build_phase3_continuity_planner_surface**`: bridge from VQE + default or explicit `workload_id`.

Descriptor preflight (`**preflight_descriptor_request**`) follows the same surface preflight rules, then `**build_partition_descriptor_set**`.

`**validate_partition_descriptor_set_against_surface**` checks `requested_mode`, `source_type`, `workload_id`, `qbit_num`, and `parameter_count` match between surface and descriptor set (plus structural operation equality).

---

## Where to look in the repo


| Area                              | Location                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Surface / descriptor / validation | `squander/partitioning/noisy_planner.py`                                                                                 |
| Runtime result provenance         | `squander/partitioning/noisy_runtime.py` (`NoisyRuntimeExecutionResult`)                                                 |
| Tests                             | `tests/partitioning/test_planner_surface_entry.py`, `test_planner_surface_descriptors.py`, `test_partitioned_runtime.py` |
| Benchmark surfaces / bundles      | `benchmarks/density_matrix/planner_surface/`                                                                             |
| Phase 3 API overview              | `docs/density_matrix_project/phases/phase-3/API_REFERENCE_PHASE_3.md`                                                    |


