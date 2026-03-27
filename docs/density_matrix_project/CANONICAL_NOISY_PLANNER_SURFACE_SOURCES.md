# Canonical noisy planner surface — sources and usage

This document inventories **where** a `CanonicalNoisyPlannerSurface` is produced, which **`source_type`** values the Phase 3 contract recognizes, and **where** those surfaces are consumed. Primary implementation: `squander/partitioning/noisy_planner.py` (`SUPPORTED_PLANNER_SOURCE_TYPES`, builder functions, descriptor pipeline).

**Note:** `preflight_planner_request()` is the only public entry that enforces `source_type ∈ SUPPORTED_PLANNER_SOURCE_TYPES`. Direct calls to `build_canonical_planner_surface_from_operation_specs()` (and similar) do not re-check the enum unless they go through preflight.

---

## Table 1 — Supported `source_type` values (workload provenance)

| source_type | Meaning (contract role) | Typical creation path | Created in repository (representative) | Used elsewhere (downstream) |
|-------------|-------------------------|----------------------|----------------------------------------|-----------------------------|
| `generated_hea` | Phase 2 noisy HEA / density-bridge anchor workloads | `describe_density_bridge()` → `build_canonical_planner_surface_from_bridge_metadata` or `build_phase3_continuity_planner_surface` | `noisy_planner.build_phase3_continuity_planner_surface`; `tests/partitioning/test_planner_surface_entry.py`; `tests/partitioning/test_planner_surface_descriptors.py`; `benchmarks/density_matrix/planner_surface/continuity_surface_validation.py`; `benchmarks/density_matrix/planner_surface/planner_audit_validation.py`; `benchmarks/density_matrix/planner_surface/descriptor_ordering_validation.py`; `preflight_planner_request(..., bridge_metadata=...)` in `tests/partitioning/test_planner_surface_entry.py` and `benchmarks/density_matrix/planner_surface/unsupported_planner_validation.py` | Descriptor set via `build_phase3_continuity_partition_descriptor_set` (internally builds surface then `build_partition_descriptor_set`); audit via `build_planner_audit_record` / `build_bridge_overlap_report`; benchmark bundles that record `provenance.source_type` |
| `microcase_builder` | Small fixed operation-spec workloads for API and schema tests | `build_canonical_planner_surface_from_operation_specs` with explicit `source_type` | `benchmarks/density_matrix/planner_surface/workloads.py`; `tests/partitioning/fixtures/workloads.py`; `preflight_planner_request(..., operation_specs=...)` in `unsupported_planner_validation.py` | `build_partition_descriptor_set` from workloads helpers; `tests/partitioning/test_planner_surface_descriptors.py`; mandatory workload descriptor/surface benchmarks |
| `structured_family_builder` | Parameterized family workloads built from specs | Same as `microcase_builder` | `benchmarks/density_matrix/planner_surface/workloads.py`; `tests/partitioning/fixtures/workloads.py`; `preflight_planner_request` in `unsupported_planner_validation.py` | Same as `microcase_builder` |
| `legacy_qgd_circuit_exact` | Lowering from `qgd_Circuit` / `Gates_block`-like objects (`get_Gates`, `get_Qbit_Num`) plus optional `density_noise` | `build_canonical_planner_surface_from_legacy_circuit` → `build_canonical_planner_surface_from_operation_specs`; `build_canonical_planner_surface_from_qgd_circuit` fixes this `source_type` | `tests/partitioning/test_planner_surface_entry.py`; `tests/partitioning/test_planner_surface_descriptors.py`; `benchmarks/density_matrix/planner_surface/legacy_exact_lowering_validation.py`; `benchmarks/density_matrix/planner_surface/descriptor_audit_validation.py`; `preflight_planner_request(..., legacy_circuit=...)` in `unsupported_planner_validation.py` | `build_partition_descriptor_set` in descriptor audit; planner audit records; continuity vs legacy cross-checks in tests |

---

## Table 2 — Builder APIs that instantiate `CanonicalNoisyPlannerSurface`

| API | Role | Where defined | Direct callers in repository |
|-----|------|---------------|------------------------------|
| `CanonicalNoisyPlannerSurface(...)` | Frozen dataclass constructor | `squander/partitioning/noisy_planner.py` | Only inside `build_canonical_planner_surface_from_bridge_metadata` and `build_canonical_planner_surface_from_operation_specs` |
| `build_canonical_planner_surface_from_bridge_metadata` | Map bridge dict (`operations`, `qbit_num`, `parameter_count`, `source_type`) to surface | `noisy_planner.py` | `build_phase3_continuity_planner_surface`; `preflight_planner_request`; `tests/partitioning/test_planner_surface_entry.py` |
| `build_canonical_planner_surface_from_operation_specs` | Map iterable of operation spec dicts to surface | `noisy_planner.py` | `build_canonical_planner_surface_from_legacy_circuit`; `preflight_planner_request`; `benchmarks/density_matrix/planner_surface/workloads.py`; `tests/partitioning/fixtures/workloads.py`; `tests/partitioning/test_partitioned_runtime.py` |
| `build_canonical_planner_surface_from_legacy_circuit` | Lower legacy circuit + optional noise specs to operation_specs path | `noisy_planner.py` | `build_canonical_planner_surface_from_qgd_circuit`; `preflight_planner_request` |
| `build_canonical_planner_surface_from_qgd_circuit` | Convenience wrapper with `source_type="legacy_qgd_circuit_exact"` | `noisy_planner.py` | `tests/partitioning/test_planner_surface_entry.py`; `tests/partitioning/test_planner_surface_descriptors.py`; `benchmarks/density_matrix/planner_surface/legacy_exact_lowering_validation.py`; `benchmarks/density_matrix/planner_surface/descriptor_audit_validation.py` |
| `preflight_planner_request` | Single entry: validates `source_type`, requires exactly one of `bridge_metadata` / `operation_specs` / `legacy_circuit` | `noisy_planner.py` | `preflight_descriptor_request` (internal); `tests/partitioning/test_planner_surface_entry.py`; `benchmarks/density_matrix/planner_surface/unsupported_planner_validation.py` |
| `build_phase3_continuity_planner_surface` | `vqe.describe_density_bridge()` then bridge builder | `noisy_planner.py` | `build_phase3_continuity_partition_descriptor_set`; tests and benchmarks listed for `generated_hea` continuity |

---

## Table 3 — Downstream use of a `CanonicalNoisyPlannerSurface` (same module and dependents)

| Consumer | Purpose | Location |
|----------|---------|----------|
| `_validate_surface` | Contiguity, gate/noise support, parameter metadata | `noisy_planner.py` (all builders) |
| `build_planner_audit_record` | Serialized audit payload (`to_dict` + summary) | `noisy_planner.py`; `tests/partitioning/test_planner_surface_entry.py`; `benchmarks/.../planner_audit_validation.py`; `benchmarks/.../legacy_exact_lowering_validation.py` |
| `build_bridge_overlap_report` | Compare surface to bridge metadata fields | `noisy_planner.py`; `tests/partitioning/test_planner_surface_entry.py`; `benchmarks/.../planner_audit_validation.py` |
| `build_partition_descriptor_set` | Greedy partition + `NoisyPartitionDescriptorSet` + `validate_partition_descriptor_set_against_surface` | `noisy_planner.py`; `tests/partitioning/test_planner_surface_descriptors.py`; `tests/partitioning/test_partitioned_runtime.py`; `tests/partitioning/fixtures/workloads.py`; `benchmarks/density_matrix/planner_surface/workloads.py`; `benchmarks/.../descriptor_audit_validation.py` |
| `_validate_descriptor_request` | `max_partition_qubits` vs `max_qubit_span` | `noisy_planner.py` (via `build_partition_descriptor_set`) |
| `validate_partition_descriptor_set_against_surface` | Provenance and per-operation alignment | `noisy_planner.py`; `tests/partitioning/test_planner_surface_descriptors.py`; `benchmarks/.../unsupported_descriptor_validation.py` |

Surfaces produced inside `build_phase3_continuity_partition_descriptor_set` are not returned to callers; only the resulting `NoisyPartitionDescriptorSet` is. That path still **creates** a surface internally.

---

## Related upstream: `describe_density_bridge`

VQE method `describe_density_bridge()` (`squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py`) feeds `build_phase3_continuity_planner_surface` and overlap/bridge tests. It is also invoked directly in tests and benchmarks (e.g. `tests/VQE/test_VQE.py`, `benchmarks/density_matrix/bridge_scope/`, workflow and noise-support bundles) for bridge validation, not always to build a full planner surface.

---

## Artifact lineage

Benchmark JSON under `benchmarks/density_matrix/artifacts/planner_surface/` stores emitted audit/descriptor bundles; they **record** `source_type` / provenance from prior validation runs rather than constructing surfaces themselves.
