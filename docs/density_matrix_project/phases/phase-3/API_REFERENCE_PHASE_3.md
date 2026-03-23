# Density Matrix API Reference and Usage Guide (Through Phase 3)

This document is the consolidated API and usage guide for the density-matrix track
after **Phase 3** completion. It extends
[API_REFERENCE_PHASE_1.md](../phase-1/API_REFERENCE_PHASE_1.md), which remains
the detailed source of truth for core `squander.density_matrix` types
(`DensityMatrix`, `NoisyCircuit`, legacy noise channels, `OperationInfo`).

**Primary audience:** developers and researchers wiring exact mixed-state
simulation, noisy variational workflows, and partitioned density execution.

---

## Why Phases 2 and 3 matter (scientific role and delivered results)

- **Phase 1** delivered an exact dense mixed-state simulator: `DensityMatrix`,
  `NoisyCircuit`, and standalone noise channels. That is the computational
  reference for everything that follows.

- **Phase 2** turned that module into a **usable scientific instrument** for
  noisy variational physics: selectable execution backend on
  `qgd_Variational_Quantum_Eigensolver_Base`, ordered **fixed** local noise
  insertions on the density path, and exact Hermitian energy evaluation
  consistent with **Re Tr(H ρ)** on the supported anchor workflow. The
  workflow is validated against Qiskit Aer and packaged in machine-checkable
  evidence under `benchmarks/density_matrix/` (see
  `artifacts/workflow_evidence/` and publication bundles).

- **Phase 3** adds a **noise-aware partitioning and runtime layer** that treats
  gates and noise as first-class ordered operations: a schema-versioned **canonical
  planner surface**, **partition descriptors** (qubit remapping, parameter routing,
  semantic preservation), and an **executable partitioned runtime** with a real
  **unitary-island fusion** mode where admissible. Correctness is closed against
  sequential `NoisyCircuit` execution; performance claims are intentionally
  diagnosis-grounded (exactness first, speedup not guaranteed on the audited
  surface).

Together, these phases support reproducible noisy VQE-style studies with a
clear exact baseline, explicit unsupported boundaries, and a path toward
structured larger-scale workloads without silently changing noise semantics.

---

## Imports (typical combined usage)

```python
# Core mixed-state simulation (Phase 1 + bindings extensions)
from squander.density_matrix import (
    DensityMatrix,
    NoisyCircuit,
    NoiseChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)

# Variational workflow entry (Phase 2)
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base,
)

# Partitioned exact noisy execution (Phase 3)
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    build_phase3_continuity_planner_surface,
    build_phase3_continuity_partition_descriptor_set,
    build_partition_descriptor_set,
    preflight_planner_request,
    preflight_descriptor_request,
    NoisyPlannerValidationError,
    NoisyDescriptorValidationError,
)
from squander.partitioning.noisy_runtime import (
    execute_partitioned_density,
    execute_partitioned_density_fused,
    execute_sequential_density_reference,
    build_runtime_audit_record,
    NoisyRuntimeValidationError,
)
```

---

## Phase 1 recap (see Phase 1 doc for full detail)

The following remain unchanged in spirit from Phase 1; see
[API_REFERENCE_PHASE_1.md](../phase-1/API_REFERENCE_PHASE_1.md).

| Area | Summary |
|------|---------|
| `DensityMatrix` | Constructors, `trace`, `purity`, `entropy`, `apply_unitary`, `partial_trace`, etc. |
| `NoisyCircuit` | Gate builders, `add_depolarizing` / `add_amplitude_damping` / `add_phase_damping` (parametric or fixed), `apply_to`, `get_operation_info` |
| `OperationInfo` | `name`, `is_unitary`, `param_count`, `param_start` |
| Legacy channels | `DepolarizingChannel`, `AmplitudeDampingChannel`, `PhaseDampingChannel` |

**Parameter ordering:** parameters are consumed in the order parametric operations
were appended to the circuit.

---

## Phase 2 extensions

### `DensityMatrix`: local multi-qubit unitary (fusion and advanced use)

- `apply_local_unitary(u_kernel: np.ndarray[complex], target_qbits: list[int]) -> None`
  - Applies a **square** unitary of size `2^k × 2^k` on the listed qubits using
    the optimized local kernel (`k = len(target_qbits)`).
  - Used internally by the Phase 3 fused partitioned runtime for eligible
    unitary islands; advanced users may call it directly when fusing unitaries
    by hand.

### `NoisyCircuit`: local single-qubit depolarizing (workflow-aligned noise)

These align the circuit API with the **local** depolarizing channel name used in
the Phase 2/3 bridge and partitioned runtime (`local_depolarizing` in planner
metadata).

- `add_local_depolarizing(target)` — one parameter (parametric).
- `add_local_depolarizing(target, error_rate)` — fixed rate (zero parameters).

Whole-register depolarizing remains `add_depolarizing(qbit_num, ...)` as in Phase 1.

### `qgd_Variational_Quantum_Eigensolver_Base` (Phase 2 integration)

Constructor signature (relevant parts):

```python
qgd_Variational_Quantum_Eigensolver_Base(
    Hamiltonian,
    qbit_num,
    config=None,
    accelerator_num=0,
    *,
    backend=None,           # "state_vector" (default) or "density_matrix"
    density_noise=None,     # ordered fixed local noise; density backend only
)
```

- **`backend`**
  - `"state_vector"`: legacy behavior.
  - `"density_matrix"`: exact mixed-state evaluation path for the **supported**
    anchor workflow (generated **HEA**, U3/CNOT path as validated). Unsupported
    combinations raise **hard errors** from C++ validation (no silent fallback).

- **`density_noise`**
  - List of dicts, each with:
    - `channel`: canonical `local_depolarizing`, `amplitude_damping`, `phase_damping`
      (aliases: `depolarizing` → local depolarizing, `dephasing` → phase damping).
    - `target`: int qubit index.
    - `after_gate_index`: int, index in the **gate** sequence after which the
      channel is inserted.
    - `value`: float strength (`error_rate`, `gamma`, or dephasing `lambda`;
      for phase damping you may use `lambda_param` instead of `value`).
  - Normalized and stored on the Python object as `vqe.density_noise`.

- **`set_Density_Matrix_Noise(density_noise)`**  
  Replaces the ordered noise list (same schema as constructor).

- **`describe_density_bridge() -> dict`**
  - Reviewable metadata: `source_type`, qubit and parameter counts, ordered
    `operations` (gates and noise with `operation_class`, `param_start`, etc.).
  - This dict is the handoff format for **`build_canonical_planner_surface_from_bridge_metadata`** in Phase 3.

- **Energy evaluation**
  - `Optimization_Problem(parameters)` uses the density backend path when
    `backend="density_matrix"`, evaluating the real energy consistent with the
    sparse Hamiltonian and **Re Tr(H ρ)** on the supported surface.
  - **Gradient-based optimization is not supported** on the density backend;
    use optimizers that do not require `Optimization_Problem_Grad`, or use
    `state_vector` for gradient studies.

### Example: density backend + local noise + bridge metadata

```python
import numpy as np
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base,
)

# Hamiltonian: use your sparse Hamiltonian object as elsewhere in SQUANDER
vqe = qgd_Variational_Quantum_Eigensolver_Base(
    Hamiltonian,
    qbit_num,
    config={},
    backend="density_matrix",
    density_noise=[
        {
            "channel": "amplitude_damping",
            "target": 0,
            "after_gate_index": 0,
            "value": 0.01,
        },
    ],
)
vqe.set_Ansatz("HEA")
vqe.Generate_Circuit(layers=1, inner_blocks=1)

theta = np.zeros(vqe.get_Parameter_Num(), dtype=np.float64)
energy = float(vqe.Optimization_Problem(theta))

bridge = vqe.describe_density_bridge()
assert bridge["source_type"] == "generated_hea"
assert bridge["noise_count"] >= 1
```

---

## Phase 3: canonical planner surface (`squander.partitioning.noisy_planner`)

Phase 3 introduces a **contract-first** pipeline:

1. Build a **`CanonicalNoisyPlannerSurface`** (ordered gate + noise operations,
   schema `phase3_canonical_noisy_planner_v1`).
2. Build a **`NoisyPartitionDescriptorSet`** (schema
   `phase3_noisy_partition_descriptor_v1`) with partitions respecting a
   `max_partition_qubits` span budget and preserving operation order and noise
   placement.
3. Execute via **`noisy_runtime`** (below).

### Supported Phase 3 gate and noise names (strict preflight)

- Gates: `U3`, `CNOT` (normalized aliases: `u`, `u3`, `cx`, `cnot`).
- Noise: `local_depolarizing`, `amplitude_damping`, `phase_damping` (aliases
  include `dephasing` → phase damping).

### Mode

- Only **`PARTITIONED_DENSITY_MODE`** (`"partitioned_density"`) is accepted for
  Phase 3 preflight and runtime.

### Key constants (selection)

| Name | Role |
|------|------|
| `CANONICAL_PLANNER_SCHEMA_VERSION` | Planner schema tag |
| `PHASE3_DESCRIPTOR_SCHEMA_VERSION` | Descriptor schema tag |
| `DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS` | Default span budget (2) |
| `PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY` | Continuity workload route id |
| `PHASE3_ENTRY_ROUTE_MICROCASE` / `STRUCTURED_FAMILY` / `LEGACY_EXACT` | Other routes |

### Exceptions

- **`NoisyPlannerValidationError`**: planner entry / surface validation;
  structured fields: `category`, `first_unsupported_condition`, `failure_stage`,
  `source_type`, `requested_mode`, `reason`; **`to_dict()`** for logging.
- **`NoisyDescriptorValidationError`**: descriptor generation / validation;
  includes `workload_id`.

### Data classes (high level)

- **`CanonicalNoisyPlannerOperation`**: one ordered operation (gate or noise)
  with indices, qubit fields, parameter metadata, `fixed_value` for fixed noise.
- **`CanonicalNoisyPlannerSurface`**: full surface + **`to_dict()`** for audits.
- **`NoisyPartitionDescriptorMember` / `NoisyPartitionDescriptor` /
  `NoisyPartitionDescriptorSet`**: partition-local view with
  `local_to_global_qbits`, **`parameter_routing`** `(global_start, local_start, count)`,
  and per-member local qubit indices.

### Main functions

| Function | Purpose |
|----------|---------|
| `preflight_planner_request(...)` | Validate inputs; return `CanonicalNoisyPlannerSurface` (exactly one of `bridge_metadata`, `operation_specs`, `legacy_circuit`) |
| `build_canonical_planner_surface_from_bridge_metadata(...)` | From `vqe.describe_density_bridge()`-shaped dict |
| `build_phase3_continuity_planner_surface(vqe, ...)` | Convenience for Phase 2 continuity VQE |
| `build_canonical_planner_surface_from_qgd_circuit(circuit, workload_id=..., density_noise=...)` | Legacy `qgd_Circuit` lowering + optional noise specs |
| `build_partition_descriptor_set(surface, max_partition_qubits=...)` | Descriptor set from surface |
| `build_phase3_continuity_partition_descriptor_set(vqe, ...)` | VQE + partition in one step |
| `preflight_descriptor_request(...)` | Planner preflight + descriptor build |
| `build_planner_audit_record(surface, metadata=...)` | JSON-serializable audit |
| `build_bridge_overlap_report(surface, bridge_metadata)` | Compare surface vs bridge |
| `build_descriptor_audit_record(descriptor_set, metadata=...)` | Descriptor audit |

### Example: continuity surface and descriptors from a configured VQE

```python
from benchmarks.density_matrix.planner_surface.common import build_phase2_continuity_vqe
from squander.partitioning.noisy_planner import (
    build_phase3_continuity_partition_descriptor_set,
)

vqe, hamiltonian, _ = build_phase2_continuity_vqe(4)
descriptor_set = build_phase3_continuity_partition_descriptor_set(
    vqe,
    max_partition_qubits=2,
)
# descriptor_set.partitions: each partition carries ordered members + routing
```

### Example: manual operation specs (micro-benchmark style)

```python
from squander.partitioning.noisy_planner import (
    build_canonical_planner_surface_from_operation_specs,
    build_partition_descriptor_set,
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
)

surface = build_canonical_planner_surface_from_operation_specs(
    qbit_num=2,
    source_type="microcase_builder",
    entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
    workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
    workload_id="my_2q_case",
    operation_specs=[
        {"kind": "gate", "name": "U3", "target_qbit": 0, "param_count": 3},
        {
            "kind": "noise",
            "name": "phase_damping",
            "target_qbit": 0,
            "source_gate_index": 0,
            "fixed_value": 0.02,
            "param_count": 0,
        },
    ],
)
descriptor_set = build_partition_descriptor_set(surface, max_partition_qubits=2)
```

---

## Phase 3: partitioned runtime (`squander.partitioning.noisy_runtime`)

### Execution entry points

- **`execute_partitioned_density(descriptor_set, parameters, *, runtime_path=..., allow_fusion=False)`**
  - Runs partition-by-partition, building per-partition `NoisyCircuit` views with
    **local parameter routing**, starting from `|0…0⟩⟨0…0|`.
  - Default path: baseline (no fusion acceleration in the sense of fused
    kernels).
  - With **`allow_fusion=True`** (and appropriate `runtime_path`), eligible
    **unitary islands** (consecutive unitary ops on ≤2 qubits) may be merged and
    applied via **`DensityMatrix.apply_local_unitary`**.

- **`execute_partitioned_density_fused(descriptor_set, parameters)`**  
  Shorthand for fused unitary-island execution.

- **`execute_sequential_density_reference(descriptor_set, parameters) -> DensityMatrix`**
  - Flattens descriptor members in order into one global `NoisyCircuit` and
    applies with the **global** parameter indexing. Use this as the semantic
    reference when comparing against partitioned results.

### Result type: `NoisyRuntimeExecutionResult`

Notable fields and properties:

- **`density_matrix`**: final `DensityMatrix` (**clone** of internal state).
- **`runtime_path`**, **`runtime_ms`**, **`peak_rss_kb`**: path label and coarse
  measurement (useful for evidence scripts).
- **`partitions`**: tuple of **`NoisyRuntimePartitionRecord`** (per-partition
  circuit shape, routing, operation names).
- **`fused_regions`**: tuple of **`NoisyRuntimeFusedRegionRecord`** (fusion
  classification: fused vs supported-but-unfused vs deferred).
- **Semantic helpers:** `rho_is_valid`, `purity`, `trace_deviation`,
  `actual_fused_execution`, `fused_region_count`, etc.
- **`to_dict(include_density_matrix=False)`**, **`build_exact_output_record(...)`**,
  **`build_runtime_audit_record(result, metadata=...)`** for manifests and papers.

### Exception

- **`NoisyRuntimeValidationError`**: preflight or execution failures with
  structured metadata (`to_dict()`).

### Example: partitioned vs sequential agreement

```python
import numpy as np
from squander.partitioning.noisy_runtime import (
    execute_partitioned_density,
    execute_sequential_density_reference,
)

parameters = np.zeros(descriptor_set.parameter_count, dtype=np.float64)
# ... fill parameters ...

result = execute_partitioned_density(descriptor_set, parameters)
rho_part = result.density_matrix_numpy()
rho_seq = execute_sequential_density_reference(descriptor_set, parameters).to_numpy()

assert np.allclose(rho_part, rho_seq, atol=1e-9, rtol=0.0)
assert result.rho_is_valid
```

### Example: fused path (when eligible)

```python
from squander.partitioning.noisy_runtime import execute_partitioned_density_fused

fused_result = execute_partitioned_density_fused(descriptor_set, parameters)
# fused_result.actual_fused_execution may be True when islands were fused
```

---

## Observable energy from a stored density matrix

For a Hermitian Hamiltonian `H` (dense NumPy or product with `rho`):

```python
import numpy as np

def hermitian_energy_real(H, rho_np: np.ndarray) -> float:
    return float(np.real(np.trace(H @ rho_np)))
```

Match this against `vqe.Optimization_Problem(parameters)` on the density backend
for consistency checks on the supported workflow.

---

## Evidence, benchmarks, and reproducibility

The following locations support claims and regression checking:

- `benchmarks/density_matrix/workflow_evidence/` — Phase 2 workflow validation,
  Qiskit comparison, publication bundle inputs.
- `benchmarks/density_matrix/partitioned_runtime/` — Phase 3 runtime handoff,
  correctness vs sequential reference, unsupported-case inventory.
- `benchmarks/density_matrix/planner_surface/` — Phase 3 planner/descriptor
  workloads (continuity, microcases, structured families).
- `benchmarks/density_matrix/publication_evidence/` — claim packages and manifest
  checks.

Pytest coverage includes `tests/partitioning/test_partitioned_runtime.py`,
`test_partitioned_runtime_fusion.py`, and planner surface tests.

---

## Common pitfalls (Phases 1–3)

- **VQE density backend** is not a general arbitrary-circuit API: it requires
  the validated **generated HEA** path; custom structures or unsupported gates
  fail explicitly.
- **`density_noise` is only meaningful with `backend="density_matrix"`**; the
  Python layer normalizes channels and the C++ layer enforces consistency.
- **Do not mix up** `add_depolarizing(n_qubits, …)` (multi-qubit channel) with
  planner/runtime naming **`local_depolarizing`** (single-qubit local channel).
- **Phase 3 strict support**: only **U3**, **CNOT**, and the three local noise
  names on the audited surface; anything else should raise structured validation
  errors rather than silently changing behavior.
- **Parameter vectors** must match `descriptor_set.parameter_count` (runtime)
  or `circuit.parameter_num` (`NoisyCircuit`).
- **Partitioned execution** preserves global semantics only when descriptors are
  built from validated surfaces; use **`execute_sequential_density_reference`**
  when asserting bitwise agreement with a flattened circuit.

---

## Document hierarchy

| Phase | Primary API doc |
|-------|-----------------|
| 1 | [API_REFERENCE_PHASE_1.md](../phase-1/API_REFERENCE_PHASE_1.md) |
| 2–3 | This document (Phase 1 + integration + partitioning) |

For milestone scope and acceptance language, see `CHANGELOG.md` and
`phases/phase-2/DETAILED_PLANNING_PHASE_2.md`,
`phases/phase-3/DETAILED_PLANNING_PHASE_3.md`.
