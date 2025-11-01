# Density Matrix Project Documentation

**Phase 1 Release**  

---

## 📚 Documentation Guide

This directory contains comprehensive documentation for adding density matrix support to SQUANDER. Choose your reading path based on your needs:

### ⚡ Quick Reference (2 minutes)

**Experienced users wanting a cheat sheet:**
- See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API reference card with common patterns

### 🚀 Quick Start (5 minutes)

**New users wanting to try Phase 1:**
1. Start with [SETUP.md](SETUP.md) - Build and test the implementation
2. Try examples in [phase1-isolated/README.md](phase1-isolated/README.md)
3. Explore [PHASE1_IMPLEMENTATION.md](phase1-isolated/PHASE1_IMPLEMENTATION.md) for API details

### 📖 Complete Reading Path

**For reviewers and contributors:**

| Document | Purpose | Audience |
|----------|---------|----------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | API cheat sheet, common patterns | Experienced users |
| [DENSITY_MATRIX_PROJECT_README.md](DENSITY_MATRIX_PROJECT_README.md) | Project overview, motivation, 3-phase roadmap | Everyone |
| [SETUP.md](SETUP.md) | Environment setup, build steps, verification | Users, developers |
| [phase1-isolated/README.md](phase1-isolated/README.md) | Phase 1 quick start with examples | Users |
| [phase1-isolated/PHASE1_DESIGN.md](phase1-isolated/PHASE1_DESIGN.md) | Design decisions and rationale | Reviewers, architects |
| [phase1-isolated/PHASE1_IMPLEMENTATION.md](phase1-isolated/PHASE1_IMPLEMENTATION.md) | Implementation details, API reference | Developers |

---

## 🎯 What's in Phase 1?

Phase 1 delivers **foundation-level density matrix support** with:

✅ **Core C++ Implementation**
- Full density matrix class with quantum properties (trace, purity, entropy)
- Integration with existing SQUANDER gates via wrapper pattern
- Circuit builder for density matrix evolution
- Three noise channel implementations (depolarizing, amplitude damping, phase damping)

✅ **Python Interface**
- Clean subpackage: `from squander.density_matrix import DensityMatrix`
- Direct C++ bindings via pybind11
- Seamless NumPy integration

✅ **Testing & Validation**
- 22 Python unit tests (all passing)
- 8 C++ unit tests
- Validated code examples
- Integration tests with existing SQUANDER

✅ **Modern Build System**
- Modern CMake with INTERFACE libraries
- Zero modifications to existing SQUANDER code
- Always enabled (no build flags needed)

---

## 📋 Document Contents

### DENSITY_MATRIX_PROJECT_README.md
- **What:** Overall project vision and roadmap
- **Contains:**
  - Why density matrices matter for quantum simulation
  - State vectors vs. density matrices (with examples)
  - 3-phase implementation plan
  - Performance trade-offs
  - External resources

### SETUP.md
- **What:** How to build and verify Phase 1
- **Contains:**
  - Prerequisites and dependencies
  - Step-by-step build instructions
  - Quick verification tests
  - Troubleshooting guide
  - Platform-specific notes

### phase1-isolated/README.md
- **What:** Phase 1 user guide with working examples
- **Contains:**
  - Quick start examples
  - Basic usage patterns
  - Noise simulation examples
  - Integration with existing SQUANDER
  - Modern CMake benefits

### phase1-isolated/PHASE1_DESIGN.md
- **What:** Phase 1 architectural decisions and rationale
- **Contains:**
  - Design principles (non-invasive, modern CMake)
  - Directory structure
  - CMake modernization approach
  - Future enhancement considerations

### phase1-isolated/PHASE1_IMPLEMENTATION.md
- **What:** What was actually implemented in Phase 1
- **Contains:**
  - Complete feature list
  - API reference for all classes
  - Python usage patterns

---

## 🔗 External Resources

**Related Technologies:**
- pybind11: https://pybind11.readthedocs.io/
- Modern CMake: https://cliutils.gitlab.io/modern-cmake/
- NumPy C API: https://numpy.org/doc/stable/reference/c-api/

**Similar Projects (for reference):**
- Qiskit Aer: https://github.com/Qiskit/qiskit-aer
- Qulacs: https://github.com/qulacs/qulacs
- QuEST: https://github.com/QuEST-Kit/QuEST

---

## 📄 License

This documentation and implementation follow the same Apache-2.0 license as the main SQUANDER project.

*Last Updated: November 1, 2025*  
