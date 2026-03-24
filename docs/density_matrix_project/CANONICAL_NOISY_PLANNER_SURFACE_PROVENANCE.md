# CanonicalNoisyPlannerSurface — provenance and implied route / family

Primary implementation: `squander/partitioning/noisy_planner.py`.

---

## Stored provenance

`CanonicalNoisyPlannerSurface` carries **`source_type`** and **`workload_id`** as the workload identity for audits. The **`provenance`** property and nested copy inside **`to_dict()`** include only:

| Field | Role |
| --- | --- |
| `source_type` | Supported Phase 3 origin label (`SUPPORTED_PLANNER_SOURCE_TYPES`). |
| `workload_id` | Instance id (continuity default, microcase name, structured id, or caller string). |

Related top-level fields (not inside `provenance`): `schema_version`, `requested_mode`, qubit/parameter counts, operations.

`NoisyPartitionDescriptorSet` and runtime handoff types mirror the same **`provenance`** shape (`source_type` + `workload_id` only).

---

## Implied entry route and workload family

**`entry_route` and `workload_family` are not stored** on surfaces or descriptors. For each supported `source_type`, the canonical labels are defined by the Phase 3 constants and exposed as helpers:

- **`phase3_entry_route_for_source_type(source_type)`**
- **`phase3_workload_family_for_source_type(source_type)`**

These map:

| `source_type` | Entry route constant | Workload family constant |
| --- | --- | --- |
| `generated_hea` | `PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY` | `PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY` |
| `microcase_builder` | `PHASE3_ENTRY_ROUTE_MICROCASE` | `PHASE3_WORKLOAD_FAMILY_MICROCASE` |
| `structured_family_builder` | `PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY` | `PHASE3_WORKLOAD_FAMILY_STRUCTURED` |
| `legacy_qgd_circuit_exact` | `PHASE3_ENTRY_ROUTE_LEGACY_EXACT` | `PHASE3_WORKLOAD_FAMILY_LEGACY` |

Benchmarks that still need a human-readable “family” string for grouping (for example planner calibration) derive **`workload_family`** from `phase3_workload_family_for_source_type` when building metadata, not from the descriptor object.

---

## Builders and preflight

- **`build_canonical_planner_surface_from_bridge_metadata`**: `bridge_metadata`, `workload_id`, optional `source_type` override (else `bridge_metadata["source_type"]`).
- **`build_canonical_planner_surface_from_operation_specs`**: `qbit_num`, `source_type`, `workload_id`, `operation_specs`.
- **`preflight_planner_request`**: `source_type`, `workload_id`, and exactly one of `bridge_metadata`, `operation_specs`, or `legacy_circuit`; operation-spec branch requires `qbit_num`.
- **`build_phase3_continuity_planner_surface`**: bridge from VQE + default or explicit `workload_id`.

Descriptor preflight (**`preflight_descriptor_request`**) follows the same surface preflight rules, then **`build_partition_descriptor_set`**.

**`validate_partition_descriptor_set_against_surface`** checks `requested_mode`, `source_type`, `workload_id`, `qbit_num`, and `parameter_count` match between surface and descriptor set (plus structural operation equality).

---

## Where to look in the repo

| Area | Location |
| --- | --- |
| Surface / descriptor / validation | `squander/partitioning/noisy_planner.py` |
| Runtime result provenance | `squander/partitioning/noisy_runtime.py` (`NoisyRuntimeExecutionResult`) |
| Tests | `tests/partitioning/test_planner_surface_entry.py`, `test_planner_surface_descriptors.py`, `test_partitioned_runtime.py` |
| Benchmark surfaces / bundles | `benchmarks/density_matrix/planner_surface/` |
| Phase 3 API overview | `docs/density_matrix_project/phases/phase-3/API_REFERENCE_PHASE_3.md` |
