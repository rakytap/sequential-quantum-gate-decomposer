# Density Matrix Integration for SQUANDER

**Phase 1 Complete ✅** - Full density matrix support with noise modeling

---

## Quick Start

### What is This Project?

**Goal:** Add density matrix support to SQUANDER to enable realistic quantum noise simulation.

**Current State:** SQUANDER simulates ideal quantum circuits using state vectors (pure states only).

**Phase 1 Delivered:** SQUANDER now supports density matrices for mixed-state simulation with noise! ✅

**Why:** Realistic simulations of NISQ (Noisy Intermediate-Scale Quantum) devices require modeling noise, which state vectors cannot represent.

**Quick Start:** See [SETUP.md](SETUP.md) to get started with Phase 1.

---

### Three Implementation Approaches

| Approach | Complexity | Max Qubits | Best For |
|----------|------------|------------|----------|
| **Tier 1: Minimal** | Low | 10 | Prototyping, quick wins |
| **Tier 2: Optimized** | Medium | 12 | Production use |
| **Tier 3: Stochastic** | High | 20+ | Research, large systems |

**Note:** Implement sequentially (1 → 2 → 3).

---

### Key Benefits

**Scientific:**
- ✅ Simulate realistic quantum noise (decoherence, gate errors)
- ✅ Model open quantum systems (system-environment coupling)
- ✅ Validate quantum error correction codes
- ✅ Study mixed state entanglement

**Software:**
- ✅ Modernized Python bindings (88% less code)
- ✅ Better memory safety (automatic reference counting)
- ✅ Improved maintainability (easier to add new features)
- ✅ Competitive with Qiskit/Cirq

---

## Modernization Opportunity: Python Bindings

### Current Situation
```cpp
// Manual C API - 200 lines per gate type
static PyObject* add_CNOT(...) {
    if (!PyArg_ParseTupleAndKeywords(...)) return NULL;
    // ... 50 lines of boilerplate ...
}
```

### With pybind11
```cpp
// Automatic binding - 2 lines per gate type
.def("add_CNOT", &Circuit::add_CNOT, 
     py::arg("target"), py::arg("control"))
```

**Impact:** 
- 88% less code (6,659 → 770 lines)
- Better memory safety
- Faster development of new features
- Improved documentation

**Trade-off:** +30% compile time, +10-20% binary size

**Note:** Phase 1 uses pybind11. nanobind is a modern alternative, but requires more mature tooling support.

---

## 🎓 Key Concepts 

### State Vectors vs. Density Matrices

**State Vector (Pure State):**
```python
# |ψ⟩ = α|0⟩ + β|1⟩
psi = np.array([alpha, beta])  # Size: 2^n
```
- Represents **one** quantum state
- Requires |α|² + |β|² = 1
- Cannot represent noise or mixed states

**Density Matrix (Mixed State):**
```python
# ρ = p₁|ψ₁⟩⟨ψ₁| + p₂|ψ₂⟩⟨ψ₂| + ...
rho = np.array([[ρ₀₀, ρ₀₁],
                [ρ₁₀, ρ₁₁]])  # Size: 2^n × 2^n
```
- Represents **statistical ensemble** of quantum states
- Can represent noise, decoherence, measurement effects
- Pure state is special case: ρ = |ψ⟩⟨ψ|

### Why Density Matrices?

**Example: Depolarizing Noise**

State vector (before):
```python
|ψ⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
```

After 1% depolarizing noise:
```python
# Cannot represent with state vector!
# Need density matrix:
ρ = 0.99 |+⟩⟨+| + 0.01 I/2
  = [[0.5,    0.495],     # Not a pure state!
     [0.495, 0.5   ]]
```


---

## Roadmap

### Phase 1: Foundation ✅ COMPLETE
**Goal:** Basic density matrix support (Tier 1)

**Status:** Complete and tested ✅

**Delivered:**
- ✅ Core C++ implementation (DensityMatrix, NoisyCircuit, 3 noise channels)
- ✅ Python interface via pybind11
- ✅ Testing & validation (22 Python + 8 C++ tests)
- ✅ Integration with existing SQUANDER (zero code changes)
- ✅ Modern CMake build system
- ✅ Comprehensive documentation

**See:** [phase1-isolated/README.md](phase1-isolated/README.md) for usage examples

---

### Phase 2: 


---

### Phase 3: 


---

### Phase 4: 


---
### Phase 5: 


---

### Performance Trade-off

| Metric | State Vector | Density Matrix (Exact) | Density Matrix (Stochastic) |
|--------|--------------|------------------------|------------------------------|
| **Memory** | O(2^n) | O(4^n) | O(2^n) |
| **Time per gate** | O(2^n) | O(4^n) | O(N × 2^n) |
| **Max qubits (8GB RAM)** | ~27 | ~13 | ~27 |
| **Accuracy** | Exact | Exact | Statistical (~1/√N) |

---

## External Resources

**Modern CMake:**
- [Effective CMake (2018+)](https://www.youtube.com/watch?v=bsXLMQ6WgIk)
- [Modern CMake Practices](https://cliutils.gitlab.io/modern-cmake/)

**pybind11:**
- [Official Tutorial](https://pybind11.readthedocs.io/en/stable/basics.html)
- [NumPy Integration](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)


**C++/Python Binding:**
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [nanobind Documentation](https://nanobind.readthedocs.io/)


**Performance Optimization:**
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [TBB Documentation](https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2021-12/overview.html)

**Similar Projects:**
- [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) (IBM)
- [Qulacs](https://github.com/qulacs/qulacs) (Keio University)
- [QuEST](https://github.com/QuEST-Kit/QuEST) (Oxford)

---

## 📚 Documentation Guide

For detailed documentation, see:

1. **[README.md](README.md)** - Documentation index and navigation guide
2. **[SETUP.md](SETUP.md)** - Build and installation instructions
3. **[phase1-isolated/README.md](phase1-isolated/README.md)** - Phase 1 usage examples
4. **[phase1-isolated/PHASE1_DESIGN.md](phase1-isolated/PHASE1_DESIGN.md)** - Design rationale
5. **[phase1-isolated/PHASE1_IMPLEMENTATION.md](phase1-isolated/PHASE1_IMPLEMENTATION.md)** - API reference

---

## 📄 License & Attribution

**SQUANDER:** Apache-2.0 License  
**Documentation:** Apache-2.0 License  
**Authors:** SQUANDER Contributors (2020-2025)

**Acknowledgments:**
- Original SQUANDER architecture: Peter Rakyta, Zoltán Zimborás
- AVX kernels inspired by: Qulacs (MIT License, properly attributed)

---

*Last Updated: November 1, 2025*  

