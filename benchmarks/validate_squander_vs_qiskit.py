"""
Validation: SQUANDER vs Qiskit Noisy Circuit Simulation

Validates SQUANDER's density matrix simulation against Qiskit using
MANUAL NOISE INSERTION for identical circuit structures.

Run with: python benchmarks/validate_squander_vs_qiskit.py
"""

import numpy as np
from circuits import CIRCUITS_BY_QUBITS
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import state_fidelity


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute trace distance: D(ρ,σ) = (1/2)||ρ - σ||_1"""
    diff = rho1 - rho2
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.maximum(eigenvalues, 0)))


def validate(name: str, squander_rho: np.ndarray, qiskit_rho, verbose=True):
    """Compare two density matrices and print results."""
    # Convert to numpy arrays if DensityMatrix objects
    squander_arr = np.asarray(squander_rho)
    qiskit_arr = np.asarray(qiskit_rho)

    fidelity = state_fidelity(
        QiskitDensityMatrix(squander_arr), QiskitDensityMatrix(qiskit_arr)
    )
    max_diff = np.max(np.abs(squander_arr - qiskit_arr))

    sq_purity = np.real(np.trace(squander_arr @ squander_arr))

    status = "✅" if fidelity > 0.99999 else ("⚠️" if fidelity > 0.999 else "❌")

    if verbose:
        print(f"  {name:<45} F={fidelity:.10f} {status}  purity={sq_purity:.4f}")

    return fidelity, max_diff, sq_purity


def run_validation():
    """Run validation on all circuits."""
    print("=" * 70)
    print("  SQUANDER vs Qiskit: Comprehensive Validation")
    print("=" * 70)

    results = []

    for n_qubits in sorted(CIRCUITS_BY_QUBITS.keys()):
        circuits = CIRCUITS_BY_QUBITS[n_qubits]

        # Determine section title
        if n_qubits <= 2:
            # Check if mixed or gates only
            mixed = [c for c in circuits if "mixed" in c[0]]
            gates = [c for c in circuits if "gates" in c[0]]

            if mixed:
                print(f"\n--- {n_qubits}-QUBIT CIRCUITS (MIXED) ---", flush=True)
                for name, builder_fn in mixed:
                    sq, qk = builder_fn().run()
                    results.append((name, *validate(name, sq, qk)))

            if gates:
                print(f"\n--- {n_qubits}-QUBIT CIRCUITS (GATES ONLY) ---", flush=True)
                for name, builder_fn in gates:
                    sq, qk = builder_fn().run()
                    results.append((name, *validate(name, sq, qk)))
        else:
            print(f"\n--- {n_qubits}-QUBIT CIRCUITS (MIXED) ---", flush=True)
            for name, builder_fn in circuits:
                sq, qk = builder_fn().run()
                results.append((name, *validate(name, sq, qk)))

    return results


def print_summary(results):
    """Print validation summary."""
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, fid, max_diff, purity in results:
        status = "✅ PASS" if fid > 0.99999 else "❌ FAIL"
        if fid <= 0.99999:
            all_passed = False
        print(f"  {name:<20} Fidelity={fid:.10f} MaxDiff={max_diff:.2e} {status}")

    print("\n" + "-" * 70)
    total = len(results)
    passed = sum(1 for _, fid, _, _ in results if fid > 0.99999)
    print(f"  Total: {passed}/{total} tests passed")

    if all_passed:
        print("\n  🎉 ALL TESTS PASSED - SQUANDER validated against Qiskit!")
    else:
        print("\n  ⚠️  Some tests failed - investigation needed")

    print("=" * 70)


def main():
    results = run_validation()
    print_summary(results)


if __name__ == "__main__":
    main()
