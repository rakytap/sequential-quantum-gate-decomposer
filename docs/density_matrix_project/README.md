# Density Matrix Project for SQUANDER

This directory documents the density-matrix track of SQUANDER.

The objective is to move SQUANDER from ideal pure-state simulation to
noise-aware mixed-state simulation, then make noise a first-class concern in
the partitioning and gate-fusion stack, and only after that broaden noisy
VQE/VQA workflow capabilities.

## Why This Project Exists

SQUANDER's existing simulation flow is state-vector based. That is efficient for
ideal circuits but cannot represent mixed states produced by noise channels.

Density matrices are required for:

- realistic noisy emulation (depolarizing, T1, T2, and future channels),
- trainability studies under noise (gradient collapse, barren plateaus),
- reproducible noisy VQA experiments.

## Current Status

Phase 1 is complete and established the foundation for the density-matrix project:

- `DensityMatrix` C++ core with quantum properties and partial trace,
- `NoisyCircuit` unified gate + noise execution path,
- three implemented noise channels (depolarizing, amplitude damping, phase damping),
- Python bindings in `squander.density_matrix`,
- dedicated tests and Qiskit comparison benchmarks.

Phase 2 is complete and established the current exact noisy-workflow baseline:

- backend selection between state-vector and density-matrix execution,
- exact Hermitian-energy evaluation via `Re Tr(H*rho)`,
- one canonical noisy XXZ workflow contract with explicit supported and
  deferred boundaries,
- and machine-checkable validation/publication bundles for that frozen support
  surface.

Phase 3 is complete and established the noise-aware partitioning/fusion foundation:

- canonical noisy mixed-state planner and descriptor surfaces that treat gates
  and noise as first-class planner inputs,
- an executable partitioned density runtime with limited real fused execution on
  eligible substructures,
- machine-checkable correctness, performance, and publication-evidence bundles
  grounded in sequential `NoisyCircuit` and Qiskit Aer baselines,
- and a bounded planner-calibration result with diagnosis-grounded performance
  closure, while channel-native fusion and broader workflow growth remain
  deferred.

Phase 3.1 is complete on its frozen v1 slice as a **bounded decision-study**
follow-on for channel-native / superoperator fusion. The phase-local record in
[`archive/phases/phase-3-1/`](archive/phases/phase-3-1/) includes the
implemented strict/hybrid runtime surfaces, the full counted 26-row
whole-workload matrix, a machine-readable decision artifact, a recorded
pre-publication review state of `decision-study-ready`, and finalized
phase-local paper surfaces in decision-study mode.

## Delivered Phases

| Phase | Goal                                                            | Status   |
| ----- | --------------------------------------------------------------- | -------- |
| 1     | Foundation: density matrices + initial noise channels           | Complete |
| 2     | Exact noisy backend integration for one canonical workflow       | Complete |
| 3     | Noise-aware partitioning and gate fusion for mixed-state circuits | Complete |
| 3.1   | Channel-native / superoperator fusion follow-on (optional)      | Complete (bounded decision study) |

These phases were planned and delivered under the earlier phase-based
convention; their contracts, evidence reviews, and paper surfaces are archived
read-only under [`archive/`](archive/README.md).

## What Comes Next

Further work is planned with the spec-driven development stack described in
[`docs/sdd-skills-guide.md`](../sdd-skills-guide.md), under the spec root
[`docs/specs/`](../specs/README.md): a product statement (`CAP-*`, `QA-*`), an
outcome roadmap (`M#`) that records Phases 1–3.1 as delivered, then one
milestone at a time — requirements (`REQ-*`), Layer 1 contract, and vertical
slices with closeouts. The current-state references
[`ARCHITECTURE_OVERVIEW.md`](../specs/ARCHITECTURE_OVERVIEW.md) and
[`TECH_STACK.md`](../specs/TECH_STACK.md) are already in place.

Notes:

- Scope here is density-matrix-specific work.
- GPU kernel development is tracked separately and can be integrated per
  milestone as available.

## Documentation Map

- [`CHANGELOG.md`](CHANGELOG.md): delivered phase outputs and status policy.
- [`../specs/ARCHITECTURE_OVERVIEW.md`](../specs/ARCHITECTURE_OVERVIEW.md):
  current-state architecture (containers, contexts, flows, risks, ADR links).
- [`../specs/TECH_STACK.md`](../specs/TECH_STACK.md): build, environments, test
  and evidence lanes.
- [`SETUP.md`](SETUP.md): environment setup, build, verification, troubleshooting.
- [`RESEARCH_ALIGNMENT.md`](RESEARCH_ALIGNMENT.md): mapping of delivered phases
  to the PhD plan.
- [`archive/`](archive/README.md): delivered Phase 1–3.1 contracts, the
  program-level plan and ADRs, API references at closure.

## Minimal Hello World

```python
from squander.density_matrix import DensityMatrix, NoisyCircuit
import numpy as np

rho = DensityMatrix(qbit_num=2)
circuit = NoisyCircuit(2)
circuit.add_H(0)
circuit.add_CNOT(1, 0)
circuit.apply_to(np.array([]), rho)
```

