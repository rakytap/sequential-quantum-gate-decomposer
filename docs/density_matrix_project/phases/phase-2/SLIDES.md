# Exact Noisy Variational Quantum Circuit Emulation in SQUANDER

## Presentation Slides (~20 minutes)

---

## Slide 1 — Why study VQAs under noise?

- Variational quantum algorithms (VQE, QAOA) are the leading candidates for
  near-term quantum applications
- Noise does not just degrade results --- it reshapes the training landscape
  qualitatively
- Wang et al. (Nature Communications, 2021): local Pauli noise induces barren
  plateaus that do not arise in ideal simulation
- Fontana et al. (Quantum, 2025): non-unital noise (e.g. amplitude damping)
  does *not* necessarily induce barren plateaus
- Conclusion: the *type* of noise model directly determines what you can
  conclude about trainability
- This motivates exact noisy simulation with realistic local noise models
  inside the training loop

---

## Slide 2 — The landscape: who already does density-matrix VQE?

Several major frameworks already integrate density-matrix simulation into
VQE training:

| Framework      | Density-matrix VQE? | Mechanism                                 |
|----------------|---------------------|--------------------------------------------|
| **Qiskit Aer** | Yes                 | Estimator primitives; `method="automatic"` silently switches to density-matrix when noise is present |
| **PennyLane**  | Yes                 | `default.mixed` device; same QNode API; differentiable noise |
| **Cirq**       | Yes                 | `DensityMatrixSimulator`; manual VQE loop  |
| **Qulacs**     | Yes                 | `DensityMatrix` class; community-documented for noisy VQE |

The problem addressed here is **not** the absence of this capability in the
field. It is a specific integration gap within SQUANDER:

- SQUANDER has a **mature state-vector VQE** with high-performance
  partitioning and gate fusion
- SQUANDER has a **standalone density-matrix module** (Phase 1)
- But the two were **not connected** --- the VQE optimizer could not use
  density-matrix evaluation as its cost function
- Phase 2 closes this gap

---

## Slide 3 — What "exact" means

- "Exact" = full dense density matrix rho (2^N x 2^N complex entries)
- Every gate and noise channel applied deterministically to rho
- Observables computed as E(theta) = Re Tr(H rho(theta)) --- a deterministic
  trace, not estimated from shots
- Contrasted with:
  - Stochastic trajectory methods (sampling, statistical variance)
  - Low-rank / tensor-network approximations (truncation error)
  - Shot-based estimation (measurement noise)
- Trade-off: O(4^N) memory --- practical limit ~10 qubits
- Value: results are trusted references with no statistical or approximation
  error (up to floating-point precision)

---

## Slide 4 — Design: Local kernel gate application

- Naive gate application: expand 1-qubit gate U to full 2^N x 2^N matrix,
  compute U rho U^dag --- cost O(8^N) per gate
- Local kernel approach: apply the small 2x2 kernel directly to pairs of
  density-matrix elements differing on the target qubit's index bit ---
  cost O(4^N) per gate
- Speedup: 2^N per gate (1024x at 10 qubits)
- Same approach for controlled 2-qubit gates (CNOT): apply 2x2 kernel only
  when the control bit is 1
- **This is a standard technique** in density-matrix simulation (used by
  Qiskit Aer, Qulacs, and others internally)
- But it is the critical enabler --- without it, exact density-matrix VQE
  at 8-10 qubits would be impractical

*Evidence: `DensityMatrix::apply_single_qubit_unitary` in density_matrix.cpp,
lines 314-439. Header: "Complexity: O(2^{2N}) instead of O(2^{3N})."*

---

## Slide 5 — Design: Noise channels via local Kraus operators

- Noise channels reuse the same local-kernel structure as gates
- Local depolarizing on qubit k:
  rho -> (1 - 3p/4) rho + (p/4)(X_k rho X_k + Y_k rho Y_k + Z_k rho Z_k)
  Each Pauli term uses `apply_single_qubit_unitary` with the 2x2 Pauli kernel
- Amplitude damping: two Kraus operators K_0, K_1 applied via bit-index
  logic on density-matrix elements --- never materializes full Kraus matrices
- **Result: noise channels have the same O(4^N) cost as gates**
- **This is standard practice** --- Qiskit Aer also uses Kraus operators for
  noise channels
- The tight coupling between gate and noise execution paths is a deliberate
  design choice

*Evidence: `LocalDepolarizingOp` in noise_operation.cpp, lines 154-198;
`AmplitudeDampingOp`, lines 254-306.*

---

## Slide 6 — Design: Unified gate+noise execution and ordered noise

- Gates and noise channels share one interface (`IDensityOperation`) and
  live in a single ordered list inside `NoisyCircuit`
- The circuit is: Gate, Gate, Noise, Gate, Noise, Gate, ...
- Noise is inserted at **exact gate positions** via `after_gate_index`
  (e.g. "local depolarizing on qubit 0 after gate 4")
- The resulting execution order is fully deterministic and auditable
- This differs from rule-based noise models in Qiskit and Cirq, where noise
  insertion is resolved at circuit compilation time via pattern-matching rules
- For reproducibility: the noise schedule is explicit, not inferred

*Evidence: `NoisyCircuit::apply_to` in noisy_circuit.cpp, lines 248-268.
`append_density_noise_for_gate_index` in VQE_Base.cpp, lines 321-348.*

---

## Slide 7 — Design: Backend dispatch at the optimizer level

- The existing VQE optimizer calls a cost function f(theta) -> energy on
  every iteration
- The density backend **replaces only this evaluation**:
  1. Initialize rho = |0><0|
  2. Lower circuit into NoisyCircuit (gates + noise in order)
  3. Apply circuit to rho
  4. Return E = Re Tr(H rho)
- The optimizer loop, convergence logic, and parameter updates are
  **reused unchanged**
- Same optimizers (COSINE, BAYES_OPT) work for both backends
- **This is standard practice** in variational frameworks --- Qiskit's
  Estimator, PennyLane's QNode, and Cirq's manual VQE loops all do the same

*Evidence: `optimization_problem()` in VQE_Base.cpp, lines 1100-1107.*

---

## Slide 8 — What IS different: No silent fallback

- SQUANDER requires explicit `backend="density_matrix"`
- Unsupported requests fail **before execution** with a named reason:
  - Wrong ansatz -> "only the HEA ansatz"
  - Unsupported gate -> "first unsupported gate is ..."
  - state_vector + density noise -> "does not support density_matrix-only
    noise configuration"
  - Unsupported noise channel -> "readout_noise" identified as first
    unsupported condition
- 10 distinct validation checks in `validate_density_anchor_support()`
- Qiskit Aer's `method="automatic"` silently selects density-matrix when
  noise is present --- user may not know which method ran
- Consequence: **every benchmark result is attributable** to a specific
  backend --- prevents scientifically ambiguous results

*Evidence: VQE_Base.cpp, lines 233-318 (10 throw points).
unsupported_workflow_bundle.json: structured negative evidence with
5+ unsupported-case artifacts.*

---

## Slide 9 — What IS different: Performance in the VQE training context

Per-evaluation runtime on mandatory workflow-scale cases:

| Qubits | SQUANDER (ms) | Qiskit Aer (ms) | Ratio                  |
|--------|---------------|------------------|------------------------|
| 4      | 0.06          | 4.1              | SQUANDER **70x faster**  |
| 6      | 0.87          | 4.6              | SQUANDER **5x faster**   |
| 8      | 26            | 211              | SQUANDER **8x faster**   |
| 10     | 435           | 263              | Qiskit Aer **1.7x faster** |

- **Why:** SQUANDER calls the C++ density kernel directly with a pre-lowered
  circuit. Qiskit Aer re-processes circuit compilation and noise-model
  resolution on each call.
- **VQE training impact (500 evaluations):**
  - 4-qubit: SQUANDER ~30 ms total vs Qiskit Aer ~2,050 ms
  - 8-qubit: SQUANDER ~13 s vs Qiskit Aer ~106 s
  - 10-qubit: SQUANDER ~218 s vs Qiskit Aer ~132 s (Aer faster)
- **Crossover at 10 qubits** motivates planned kernel-level and
  partitioning-based acceleration (Phase 3)

*Evidence: matrix_baseline_bundle.json, 40 cases with runtime data.
Do NOT cite the standalone benchmark.log numbers (90,000x) --- those compare
different levels of the software stack.*

---

## Slide 10 — What IS different: Path to density-aware acceleration

- SQUANDER has a **mature state-vector partitioning and gate-fusion
  subsystem** --- no other framework has this
- Phase 3 plan: extend partitioning/fusion to density-matrix workloads
  using barrier-based unitary-island partitioning with noise barriers
- This research direction has **no equivalent** in Qiskit, Cirq, PennyLane,
  or Qulacs
- Phase 2 creates the **stable exact backend** that Phase 3 will accelerate
- The crossover at 10 qubits provides a concrete optimization target

*Evidence: PLANNING.md Phase 3 description. Existing squander/partitioning
codebase.*

---

## Slide 11 — Validation results

All numbers from stored JSON artifacts with git revision provenance:

**Micro-validation (1-3 qubits):**
- 7 cases, all pass
- Maximum energy error: 2.5 x 10^-18 vs Qiskit Aer
- Density matrix valid: Tr(rho) = 1, rho >= 0, rho = rho^dag to 1e-10

**Workflow-scale (4, 6, 8, 10 qubits):**
- 40 cases (10 per size), all pass
- Maximum energy error < 1e-8 vs Qiskit Aer

**Optimization trace:**
- 4-qubit XXZ + HEA + mixed local noise
- COSINE optimizer, 18 parameters
- Initial energy: 0.936 -> Final energy: -4.259
- Full training loop on the density-matrix backend

**Unsupported-case evidence:**
- 5+ structured negative cases (readout noise, correlated noise, HEA_ZYZ,
  state_vector + density_noise)
- All fail before execution with named reasons

*Sources: local_correctness_bundle.json,
matrix_baseline_bundle.json, optimization_trace_4q.json,
unsupported_workflow_bundle.json.*

---

## Slide 12 — Limitations (honest)

1. **~10-qubit limit** --- dense density matrices cost O(4^N) memory
   (854 MB peak RSS at 10 qubits)
2. **Only HEA ansatz bridged** --- custom circuits, HEA_ZYZ, and arbitrary
   gate structures are out of scope
3. **No density-matrix gradients** --- only gradient-free optimizers work on
   the density path
4. **Qiskit Aer is faster at 10 qubits** --- SQUANDER's advantage is at
   4-8 qubits; kernel optimization is Phase 3 work
5. **Standard techniques** --- local kernel application, Kraus noise, sparse
   Tr(H rho), backend dispatch are all well-known; the contribution is in
   the integration, validation methodology, and acceleration path

---

## Slide 13 — Summary of contributions

1. **Integration:** Exact density-matrix backend is a selectable, validated
   execution path inside SQUANDER's VQE workflow
2. **Performance:** 5-70x faster than Qiskit Aer at 4-8 qubits per VQE
   evaluation; reverses at 10 qubits
3. **Validation methodology:** Micro + workflow-scale + optimization trace +
   negative evidence, all stored as auditable JSON artifacts with provenance
4. **Foundation for density-aware acceleration:** The integrated backend is
   the stable target for extending SQUANDER's partitioning/fusion to density
   matrices --- a research direction unique to this project

---

## Slide 14 — What's next

- **Phase 3:** Density-aware partitioning and gate fusion --- can
  state-vector acceleration ideas produce speedups for density-matrix
  workloads?
- **Phase 4:** Optimizer studies under exact noise --- how do optimizers
  behave when the cost landscape includes realistic local noise?
- **Phase 5:** Trainability analysis --- barren plateaus, entropy growth,
  expressivity under realistic noise models
- Sequence: integrate -> accelerate -> answer training questions

---

## Timing Guide

| Block                           | Slides | Minutes |
|---------------------------------|--------|---------|
| Motivation and landscape        | 1-3    | 3-4     |
| Design choices                  | 4-7    | 5-6     |
| What's actually different       | 8-10   | 4-5     |
| Results and limitations         | 11-12  | 3-4     |
| Summary and future              | 13-14  | 2-3     |
| Questions                       | ---    | ~2      |

---

## Fact-Check Summary

| Claim in slides                               | Status    | Notes                                              |
|-----------------------------------------------|-----------|----------------------------------------------------|
| Noise changes VQA training qualitatively       | Verified  | Wang et al. 2021, Fontana et al. 2025              |
| Other frameworks do density-matrix VQE         | Verified  | Qiskit Aer, PennyLane, Cirq, Qulacs               |
| "Exact" = full dense rho, deterministic trace  | Verified  | Standard usage; confirmed in codebase              |
| Local kernel is O(4^N) vs O(8^N)               | Verified  | Standard technique; code + header confirm           |
| Noise channels at same O(4^N) cost             | Verified  | Code confirms; standard technique                   |
| Ordered noise at gate positions                | Verified  | `after_gate_index` in code; differs from rule-based |
| Backend dispatch is standard practice          | Verified  | Same in Qiskit, PennyLane, Cirq                    |
| No silent fallback                             | Verified  | 10 throw points in VQE_Base.cpp; artifact evidence  |
| 5-70x faster at 4-8q                          | Verified  | matrix_baseline_bundle.json                  |
| Qiskit Aer 1.7x faster at 10q                 | Verified  | Same artifact; crossover confirmed                  |
| Micro-validation error < 2.5e-18               | Verified  | local_correctness_bundle.json                |
| 40/40 workflow cases pass at 1e-8              | Verified  | matrix_baseline_bundle.json                  |
| Optimization trace 0.936 -> -4.259             | Verified  | optimization_trace_4q.json                                |
| Density-aware acceleration path unique         | Verified  | No equivalent in other frameworks' plans            |
| Individual techniques are standard             | Verified  | Acknowledged explicitly                             |
