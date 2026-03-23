# CanonicalNoisyPlannerSurface — provenance attributes and combinations

This note inventories how **provenance** is represented on `CanonicalNoisyPlannerSurface`, which string vocabularies the codebase defines, which **(source_type, entry_route, workload_family)** triples appear in supported call paths and tests, and where those definitions live.

Primary implementation: `squander/partitioning/noisy_planner.py`.

---

## Provenance attributes on the surface

`CanonicalNoisyPlannerSurface` stores provenance as four string fields on the dataclass. The **`provenance`** property returns exactly these four keys (see `CanonicalNoisyPlannerSurface.provenance` and `to_dict()` in `noisy_planner.py`).

| Field | Role |
| --- | --- |
| `source_type` | Labels the **origin** of the workload (e.g. bridge from a VQE builder, synthetic microcase, structured family generator, or legacy circuit lowering). Must be one of `SUPPORTED_PLANNER_SOURCE_TYPES` when using `preflight_planner_request`. |
| `entry_route` | Identifies **how** the workload entered the Phase 3 planner API (which lowering or generation path). |
| `workload_family` | Coarse **family** grouping for audits, benchmarks, and continuity checks (often aligned with `entry_route` in this repo). |
| `workload_id` | Stable **instance** identifier within a family (case name, structured workload id, continuity default id, or caller-supplied string). |

Related fields on the same surface that are **not** inside the `provenance` dict but appear alongside it in `to_dict()`:

| Field | Role |
| --- | --- |
| `schema_version` | Canonical planner schema id (`CANONICAL_PLANNER_SCHEMA_VERSION`). |
| `requested_mode` | Execution mode; Phase 3 validation currently requires `partitioned_density` (`PARTITIONED_DENSITY_MODE`). |

Operation-level records use `source_gate_index` on `CanonicalNoisyPlannerOperation` to tie noise rows to a preceding gate index; that is **not** part of the top-level `provenance` dict.

---

## Frozen string vocabularies (constants)

Defined at the top of `squander/partitioning/noisy_planner.py`:

**Entry routes**

| Constant | String value |
| --- | --- |
| `PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY` | `phase2_continuity_lowering` |
| `PHASE3_ENTRY_ROUTE_MICROCASE` | `phase3_microcase_generation` |
| `PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY` | `phase3_structured_family_generation` |
| `PHASE3_ENTRY_ROUTE_LEGACY_EXACT` | `phase3_legacy_exact_lowering` |

**Workload families**

| Constant | String value |
| --- | --- |
| `PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY` | `phase2_continuity_workflow` |
| `PHASE3_WORKLOAD_FAMILY_MICROCASE` | `phase3_micro_validation` |
| `PHASE3_WORKLOAD_FAMILY_STRUCTURED` | `phase3_structured_family` |
| `PHASE3_WORKLOAD_FAMILY_LEGACY` | `phase3_legacy_exact_lowering` |

**Supported `source_type` values** (`SUPPORTED_PLANNER_SOURCE_TYPES`)

| Value | Typical use in this repo |
| --- | --- |
| `generated_hea` | Density bridge metadata from Phase 2–style HEA / VQE workloads (`describe_density_bridge()`). |
| `microcase_builder` | Small mandatory operation-spec workloads in benchmarks and tests. |
| `structured_family_builder` | Larger structured synthetic workloads (families × qubit counts × noise patterns). |
| `legacy_qgd_circuit_exact` | Legacy `qgd_Circuit` (or compatible) lowering via `get_Gates()` / `get_Qbit_Num()`. |

---

## Provenance combinations (triple + workload_id pattern)

The APIs **`build_canonical_planner_surface_from_bridge_metadata`**, **`build_canonical_planner_surface_from_operation_specs`**, and **`preflight_planner_request`** take `source_type`, `entry_route`, and `workload_family` as **caller-provided strings** (except legacy lowering, which fixes route and family internally). So many triples are *syntactically* possible if they pass validation. Below are the **combinations that the codebase explicitly wires or asserts** as the intended Phase 3 shapes.

### 1. Phase 2 continuity (bridge metadata)

| Attribute | Value |
| --- | --- |
| `source_type` | `generated_hea` (from `bridge_metadata["source_type"]`; continuity VQE matches this) |
| `entry_route` | `phase2_continuity_lowering` (`PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY`) |
| `workload_family` | `phase2_continuity_workflow` (`PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY`) |
| `workload_id` | Default `phase2_xxz_hea_q{qbit_num}_continuity` when not overridden |

**Description:** Builds the surface from `vqe.describe_density_bridge()` and pins route/family to the Phase 2 continuity path.

**Where:** `build_phase3_continuity_planner_surface()` in `squander/partitioning/noisy_planner.py`; tests and benchmarks under `tests/partitioning/test_planner_surface_entry.py`, `tests/partitioning/test_planner_surface_descriptors.py`, `tests/partitioning/test_partitioned_runtime.py`, and `benchmarks/density_matrix/planner_surface/continuity_surface_validation.py`.

### 2. Microcase operation specs

| Attribute | Value |
| --- | --- |
| `source_type` | `microcase_builder` |
| `entry_route` | `phase3_microcase_generation` (`PHASE3_ENTRY_ROUTE_MICROCASE`) |
| `workload_family` | `phase3_micro_validation` (`PHASE3_WORKLOAD_FAMILY_MICROCASE`) |
| `workload_id` | Per-case name (e.g. `microcase_2q_entangler_local_depolarizing`) |

**Description:** Mandatory small circuits built from explicit operation spec lists for validation and descriptor audits.

**Where:** `benchmarks/density_matrix/planner_surface/workloads.py` (`build_microcase_surface`, `iter_microcase_surfaces`); asserted in `tests/partitioning/test_planner_surface_entry.py` and related partitioning tests.

### 3. Structured family operation specs

| Attribute | Value |
| --- | --- |
| `source_type` | `structured_family_builder` |
| `entry_route` | `phase3_structured_family_generation` (`PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY`) |
| `workload_family` | `phase3_structured_family` (`PHASE3_WORKLOAD_FAMILY_STRUCTURED`) |
| `workload_id` | Generated per family / qubit count / noise pattern (see `workloads.py`) |

**Description:** Scaled synthetic workloads for stress and audit bundles.

**Where:** `benchmarks/density_matrix/planner_surface/workloads.py` (`build_structured_surface` path); asserted in `tests/partitioning/test_planner_surface_entry.py` and related tests.

### 4. Legacy QGD circuit lowering

| Attribute | Value |
| --- | --- |
| `source_type` | `legacy_qgd_circuit_exact` (default; overridable in `build_canonical_planner_surface_from_legacy_circuit`, fixed in `build_canonical_planner_surface_from_qgd_circuit`) |
| `entry_route` | `phase3_legacy_exact_lowering` (`PHASE3_ENTRY_ROUTE_LEGACY_EXACT`) — **always** set inside `build_canonical_planner_surface_from_legacy_circuit` |
| `workload_family` | `phase3_legacy_exact_lowering` (`PHASE3_WORKLOAD_FAMILY_LEGACY`) — **always** set there |
| `workload_id` | Caller-supplied |

**Description:** Lowers a legacy circuit object into canonical operations, optionally inserting noise from `density_noise` specs.

**Where:** `build_canonical_planner_surface_from_legacy_circuit` and `build_canonical_planner_surface_from_qgd_circuit` in `squander/partitioning/noisy_planner.py`; tests in `tests/partitioning/test_planner_surface_entry.py`, `tests/partitioning/test_planner_surface_descriptors.py`, and benchmarks under `benchmarks/density_matrix/planner_surface/legacy_exact_lowering_validation.py`.

---

## Other combinations (API vs. in-repo usage)

- **`preflight_planner_request`** dispatches on payload shape (`bridge_metadata`, `operation_specs`, or `legacy_circuit` exclusively). For **bridge** and **operation_specs** branches it **requires** `entry_route` and `workload_family` and passes them through with the caller’s `source_type`, as long as `source_type ∈ SUPPORTED_PLANNER_SOURCE_TYPES`. The codebase does not enumerate every valid triple; descriptor validation later checks that descriptor provenance matches the surface (`provenance_mismatch` path in `noisy_planner.py`).
- **`build_canonical_planner_surface_from_bridge_metadata`** can resolve `source_type` from `bridge_metadata["source_type"]` when the optional `source_type` argument is omitted; **route and family remain caller-supplied**. The continuity helper is the main in-repo example pairing `generated_hea` with the phase2 continuity route/family.
- **Tests** that call `preflight_planner_request` with `microcase_builder` reuse the same microcase **entry_route** / **workload_family** as the benchmark workloads when asserting validation behavior (`tests/partitioning/test_planner_surface_entry.py`).

---

## Where provenance is serialized or checked

| Location | Purpose |
| --- | --- |
| `CanonicalNoisyPlannerSurface.provenance` / `to_dict()` | Audit-friendly nested `provenance` object in the full surface dict. |
| `build_planner_audit_record()` | Copies `provenance` into planner audit records. |
| Descriptor builders and validators in `noisy_planner.py` | Compare `source_type`, `entry_route`, and `workload_family` between surface and descriptor payloads (e.g. provenance mismatch errors). |
| `tests/partitioning/test_planner_surface_*.py`, `tests/partitioning/test_partitioned_runtime.py` | Assert expected provenance for continuity, microcase, structured, and legacy paths. |
| `benchmarks/density_matrix/planner_surface/*.py` | Construct surfaces and bundles with the combinations above. |

---

## Summary table (primary triples)

| `source_type` | `entry_route` | `workload_family` | Primary builder / entry |
| --- | --- | --- | --- |
| `generated_hea` | `phase2_continuity_lowering` | `phase2_continuity_workflow` | `build_phase3_continuity_planner_surface`; bridge + explicit metadata in tests |
| `microcase_builder` | `phase3_microcase_generation` | `phase3_micro_validation` | `build_canonical_planner_surface_from_operation_specs` in `workloads.py` |
| `structured_family_builder` | `phase3_structured_family_generation` | `phase3_structured_family` | `build_canonical_planner_surface_from_operation_specs` in `workloads.py` |
| `legacy_qgd_circuit_exact` | `phase3_legacy_exact_lowering` | `phase3_legacy_exact_lowering` | `build_canonical_planner_surface_from_legacy_circuit` / `from_qgd_circuit` |

`workload_id` is always caller- or helper-defined and is not limited to a fixed vocabulary in code.
