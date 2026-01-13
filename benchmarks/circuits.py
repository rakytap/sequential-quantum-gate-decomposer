"""
Shared Test Circuits: SQUANDER vs Qiskit

Contains DualBuilder class and circuit factory functions for use in both
validation and benchmarking scripts.

Test circuits:
- 1 qubit: 5 mixed ops, 5 gates (no noise)
- 2 qubits: 5/7/10 mixed ops, 5/7/10 gates (no noise)
- 3 qubits: 5/10/15/20/25 mixed ops
- 4 qubits: 15/20/25 mixed ops
- 5 qubits: 30 mixed ops (commented out - slow with Qiskit)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

from squander.density_matrix import DensityMatrix, NoisyCircuit


class DualBuilder:
    """Build identical circuits in both SQUANDER and Qiskit frameworks."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.sq = NoisyCircuit(n_qubits)
        self.qk = QuantumCircuit(n_qubits)
        self.ops = []

    # === Single-qubit gates ===
    def H(self, t):
        self.sq.add_H(t)
        self.qk.h(t)
        self.ops.append(f"H({t})")

    def X(self, t):
        self.sq.add_X(t)
        self.qk.x(t)
        self.ops.append(f"X({t})")

    def Y(self, t):
        self.sq.add_Y(t)
        self.qk.y(t)
        self.ops.append(f"Y({t})")

    def Z(self, t):
        self.sq.add_Z(t)
        self.qk.z(t)
        self.ops.append(f"Z({t})")

    def S(self, t):
        self.sq.add_S(t)
        self.qk.s(t)
        self.ops.append(f"S({t})")

    def T(self, t):
        self.sq.add_T(t)
        self.qk.t(t)
        self.ops.append(f"T({t})")

    def Sdg(self, t):
        self.sq.add_Sdg(t)
        self.qk.sdg(t)
        self.ops.append(f"Sdg({t})")

    def Tdg(self, t):
        self.sq.add_Tdg(t)
        self.qk.tdg(t)
        self.ops.append(f"Tdg({t})")

    def SX(self, t):
        self.sq.add_SX(t)
        self.qk.sx(t)
        self.ops.append(f"SX({t})")

    # === Two-qubit gates ===
    def CNOT(self, tgt, ctrl):
        self.sq.add_CNOT(tgt, ctrl)
        self.qk.cx(ctrl, tgt)
        self.ops.append(f"CNOT({ctrl}->{tgt})")

    def CZ(self, tgt, ctrl):
        self.sq.add_CZ(tgt, ctrl)
        self.qk.cz(ctrl, tgt)
        self.ops.append(f"CZ({ctrl},{tgt})")

    # === Noise channels ===
    def depolarizing(self, p):
        self.sq.add_depolarizing(self.n, error_rate=p)
        self.qk.append(depolarizing_error(p, self.n), list(range(self.n)))
        self.ops.append(f"Depol({p})")

    def amp_damp(self, t, gamma):
        self.sq.add_amplitude_damping(t, gamma=gamma)
        self.qk.append(amplitude_damping_error(gamma), [t])
        self.ops.append(f"AD({t},{gamma})")

    def phase_damp(self, t, lam):
        self.sq.add_phase_damping(t, lambda_param=lam)
        self.qk.append(phase_damping_error(lam), [t])
        self.ops.append(f"PD({t},{lam})")

    # === Execution methods ===
    def run(self):
        """Execute both and return density matrices as numpy arrays."""
        sq_rho = self.run_squander()
        qk_rho = self.run_qiskit()

        # Alternative way to run Qiskit only:
        # from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
        # qk_rho = QiskitDensityMatrix.from_label(
        #     '0' * self.n).evolve(self.qk).data
        return sq_rho.to_numpy(), qk_rho

    def run_squander(self):
        """Execute SQUANDER only."""
        sq_rho = DensityMatrix(self.n)
        self.sq.apply_to(np.array([]), sq_rho)
        return sq_rho

    def run_qiskit(self):
        """Execute Qiskit only using optimal Aer density matrix simulation.

        Uses AerSimulator with method='density_matrix' which:
        - Directly simulates the density matrix evolution
        - Natively handles QuantumError noise channels (depolarizing, amplitude/phase damping)
        - Is more efficient than qiskit.quantum_info.DensityMatrix.evolve() for noisy circuits
        """
        # Create circuit copy with density matrix save instruction
        qc = self.qk.copy()
        qc.save_density_matrix()

        # Use AerSimulator with density_matrix method - optimal for noisy circuits
        # This backend directly evolves the density matrix through all operations
        # including noise channels, without needing to sample or use Kraus operators
        simulator = AerSimulator(method="density_matrix")

        # Run with shots=1 since density matrix simulation is deterministic
        result = simulator.run(qc, shots=1).result()

        return result.data()["density_matrix"]


# ===================================================================
# 1-QUBIT CIRCUITS
# ===================================================================


def build_1q_5ops_mixed():
    """1 qubit, 5 mixed operations."""
    b = DualBuilder(1)
    b.H(0)
    b.depolarizing(0.05)
    b.X(0)
    b.amp_damp(0, 0.03)
    b.S(0)
    return b


def build_1q_5ops_gates():
    """1 qubit, 5 gate operations (no noise)."""
    b = DualBuilder(1)
    b.H(0)
    b.X(0)
    b.S(0)
    b.T(0)
    b.H(0)
    return b


# ===================================================================
# 2-QUBIT CIRCUITS - MIXED
# ===================================================================


def build_2q_5ops_mixed():
    """2 qubits, 5 mixed operations."""
    b = DualBuilder(2)
    b.H(0)
    b.CNOT(1, 0)
    b.depolarizing(0.04)
    b.S(1)
    b.amp_damp(0, 0.02)
    return b


def build_2q_7ops_mixed():
    """2 qubits, 7 mixed operations."""
    b = DualBuilder(2)
    b.H(0)
    b.H(1)
    b.CNOT(1, 0)
    b.depolarizing(0.05)
    b.X(0)
    b.phase_damp(1, 0.03)
    b.CZ(0, 1)
    return b


def build_2q_10ops_mixed():
    """2 qubits, 10 mixed operations."""
    b = DualBuilder(2)
    b.H(0)
    b.H(1)
    b.CNOT(1, 0)
    b.depolarizing(0.03)
    b.S(0)
    b.T(1)
    b.amp_damp(0, 0.02)
    b.CNOT(0, 1)
    b.phase_damp(1, 0.025)
    b.H(0)
    return b


# ===================================================================
# 2-QUBIT CIRCUITS - GATES ONLY (NO NOISE)
# ===================================================================


def build_2q_5ops_gates():
    """2 qubits, 5 gate operations (no noise)."""
    b = DualBuilder(2)
    b.H(0)
    b.CNOT(1, 0)
    b.S(1)
    b.T(0)
    b.H(1)
    return b


def build_2q_7ops_gates():
    """2 qubits, 7 gate operations (no noise)."""
    b = DualBuilder(2)
    b.H(0)
    b.H(1)
    b.CNOT(1, 0)
    b.S(0)
    b.T(1)
    b.CNOT(0, 1)
    b.H(0)
    return b


def build_2q_10ops_gates():
    """2 qubits, 10 gate operations (no noise)."""
    b = DualBuilder(2)
    b.H(0)
    b.H(1)
    b.CNOT(1, 0)
    b.S(0)
    b.T(1)
    b.X(0)
    b.Z(1)
    b.CNOT(0, 1)
    b.Sdg(0)
    b.H(1)
    return b


# ===================================================================
# 3-QUBIT CIRCUITS - MIXED
# ===================================================================


def build_3q_5ops_mixed():
    """3 qubits, 5 mixed operations."""
    b = DualBuilder(3)
    b.H(0)
    b.CNOT(1, 0)
    b.depolarizing(0.04)
    b.S(2)
    b.amp_damp(0, 0.02)
    return b


def build_3q_10ops_mixed():
    """3 qubits, 10 mixed operations."""
    b = DualBuilder(3)
    b.H(0)
    b.H(1)
    b.CNOT(1, 0)
    b.depolarizing(0.05)
    b.S(2)
    b.CNOT(2, 1)
    b.amp_damp(0, 0.03)
    b.T(1)
    b.phase_damp(2, 0.02)
    b.H(2)
    return b


def build_3q_15ops_mixed():
    """3 qubits, 15 mixed operations."""
    b = DualBuilder(3)
    b.H(0)
    b.H(1)
    b.H(2)
    b.CNOT(1, 0)
    b.depolarizing(0.04)
    b.CNOT(2, 1)
    b.amp_damp(0, 0.025)
    b.S(0)
    b.T(1)
    b.phase_damp(2, 0.02)
    b.CNOT(0, 2)
    b.depolarizing(0.03)
    b.X(1)
    b.amp_damp(2, 0.02)
    b.H(0)
    return b


def build_3q_20ops_mixed():
    """3 qubits, 20 mixed operations."""
    b = DualBuilder(3)
    b.H(0)
    b.H(1)
    b.H(2)
    b.CNOT(1, 0)
    b.CNOT(2, 1)
    b.depolarizing(0.04)
    b.S(0)
    b.T(1)
    b.Sdg(2)
    b.amp_damp(0, 0.02)
    b.CNOT(0, 2)
    b.phase_damp(1, 0.025)
    b.X(0)
    b.Y(1)
    b.Z(2)
    b.depolarizing(0.03)
    b.CNOT(2, 0)
    b.amp_damp(1, 0.015)
    b.H(2)
    b.phase_damp(0, 0.02)
    return b


def build_3q_25ops_mixed():
    """3 qubits, 25 mixed operations."""
    b = DualBuilder(3)
    b.H(0)
    b.H(1)
    b.H(2)
    b.CNOT(1, 0)
    b.CNOT(2, 1)
    b.depolarizing(0.035)
    b.S(0)
    b.T(1)
    b.SX(2)
    b.amp_damp(0, 0.02)
    b.CNOT(0, 2)
    b.phase_damp(1, 0.02)
    b.X(0)
    b.Y(1)
    b.Z(2)
    b.depolarizing(0.025)
    b.CNOT(2, 0)
    b.amp_damp(1, 0.015)
    b.Tdg(0)
    b.Sdg(1)
    b.phase_damp(2, 0.018)
    b.CNOT(1, 2)
    b.depolarizing(0.02)
    b.H(0)
    b.H(1)
    return b


# ===================================================================
# 4-QUBIT CIRCUITS - MIXED
# ===================================================================


def build_4q_15ops_mixed():
    """4 qubits, 15 mixed operations."""
    b = DualBuilder(4)
    b.H(0)
    b.H(1)
    b.H(2)
    b.H(3)
    b.CNOT(1, 0)
    b.CNOT(3, 2)
    b.depolarizing(0.03)
    b.S(0)
    b.T(1)
    b.amp_damp(2, 0.02)
    b.CNOT(2, 1)
    b.phase_damp(3, 0.025)
    b.X(0)
    b.depolarizing(0.02)
    b.H(3)
    return b


def build_4q_20ops_mixed():
    """4 qubits, 20 mixed operations."""
    b = DualBuilder(4)
    b.H(0)
    b.H(1)
    b.H(2)
    b.H(3)
    b.CNOT(1, 0)
    b.CNOT(3, 2)
    b.depolarizing(0.03)
    b.S(0)
    b.T(1)
    b.Sdg(2)
    b.amp_damp(3, 0.02)
    b.CNOT(2, 1)
    b.CNOT(0, 3)
    b.phase_damp(1, 0.02)
    b.X(0)
    b.Y(2)
    b.depolarizing(0.025)
    b.amp_damp(0, 0.015)
    b.H(3)
    b.phase_damp(2, 0.018)
    return b


def build_4q_25ops_mixed():
    """4 qubits, 25 mixed operations."""
    b = DualBuilder(4)
    b.H(0)
    b.H(1)
    b.H(2)
    b.H(3)
    b.CNOT(1, 0)
    b.CNOT(3, 2)
    b.depolarizing(0.025)
    b.S(0)
    b.T(1)
    b.SX(2)
    b.Tdg(3)
    b.amp_damp(0, 0.02)
    b.CNOT(2, 1)
    b.CNOT(0, 3)
    b.phase_damp(1, 0.018)
    b.X(0)
    b.Y(2)
    b.Z(3)
    b.depolarizing(0.02)
    b.CNOT(3, 0)
    b.amp_damp(2, 0.015)
    b.phase_damp(3, 0.015)
    b.Sdg(1)
    b.H(0)
    b.H(2)
    return b


# ===================================================================
# 5-QUBIT CIRCUITS - MIXED
# ===================================================================


def build_5q_30ops_mixed():
    """5 qubits, 30 mixed operations."""
    b = DualBuilder(5)
    # Layer 1: Initial superposition (5 ops)
    b.H(0)
    b.H(1)
    b.H(2)
    b.H(3)
    b.H(4)
    # Layer 2: Entanglement + noise (5 ops)
    b.CNOT(1, 0)
    b.CNOT(2, 1)
    b.CNOT(3, 2)
    b.CNOT(4, 3)
    b.depolarizing(0.02)
    # Layer 3: Single-qubit + noise (5 ops)
    b.S(0)
    b.T(1)
    b.amp_damp(2, 0.025)
    b.X(3)
    b.Z(4)
    # Layer 4: Cross-entanglement + noise (5 ops)
    b.CNOT(0, 2)
    b.CNOT(1, 3)
    b.phase_damp(4, 0.02)
    b.CNOT(2, 4)
    b.depolarizing(0.015)
    # Layer 5: More gates + noise (5 ops)
    b.H(0)
    b.H(2)
    b.amp_damp(1, 0.018)
    b.Sdg(3)
    b.Tdg(4)
    # Layer 6: Final layer + noise (5 ops)
    b.CNOT(3, 1)
    b.CNOT(4, 0)
    b.phase_damp(2, 0.015)
    b.H(4)
    b.depolarizing(0.01)
    return b


# ===================================================================
# Circuit registry for easy access
# ===================================================================

# All circuits for validation (name, builder_func)
ALL_CIRCUITS = [
    ("1q/5ops mixed", build_1q_5ops_mixed),
    ("1q/5ops gates", build_1q_5ops_gates),
    ("2q/5ops mixed", build_2q_5ops_mixed),
    ("2q/7ops mixed", build_2q_7ops_mixed),
    ("2q/10ops mixed", build_2q_10ops_mixed),
    ("2q/5ops gates", build_2q_5ops_gates),
    ("2q/7ops gates", build_2q_7ops_gates),
    ("2q/10ops gates", build_2q_10ops_gates),
    ("3q/5ops mixed", build_3q_5ops_mixed),
    ("3q/10ops mixed", build_3q_10ops_mixed),
    ("3q/15ops mixed", build_3q_15ops_mixed),
    ("3q/20ops mixed", build_3q_20ops_mixed),
    ("3q/25ops mixed", build_3q_25ops_mixed),
    ("4q/15ops mixed", build_4q_15ops_mixed),
    ("4q/20ops mixed", build_4q_20ops_mixed),
    ("4q/25ops mixed", build_4q_25ops_mixed),
    (
        "5q/30ops mixed",
        build_5q_30ops_mixed,
    ),  # Slow with Qiskit but works with AerSimulator
]

# Subset for benchmarking (representative circuits)
BENCHMARK_CIRCUITS = [
    ("1q/5ops mixed", build_1q_5ops_mixed),
    ("1q/5ops gates", build_1q_5ops_gates),
    ("2q/5ops mixed", build_2q_5ops_mixed),
    ("2q/10ops mixed", build_2q_10ops_mixed),
    ("3q/10ops mixed", build_3q_10ops_mixed),
    ("3q/25ops mixed", build_3q_25ops_mixed),
    ("4q/15ops mixed", build_4q_15ops_mixed),
    ("4q/25ops mixed", build_4q_25ops_mixed),
    (
        "5q/30ops mixed",
        build_5q_30ops_mixed,
    ),  # Slow with Qiskit but works with AerSimulator
]

# Grouped by qubit count for organized output
CIRCUITS_BY_QUBITS = {
    1: [
        ("1q/5ops mixed", build_1q_5ops_mixed),
        ("1q/5ops gates", build_1q_5ops_gates),
    ],
    2: [
        ("2q/5ops mixed", build_2q_5ops_mixed),
        ("2q/7ops mixed", build_2q_7ops_mixed),
        ("2q/10ops mixed", build_2q_10ops_mixed),
        ("2q/5ops gates", build_2q_5ops_gates),
        ("2q/7ops gates", build_2q_7ops_gates),
        ("2q/10ops gates", build_2q_10ops_gates),
    ],
    3: [
        ("3q/5ops mixed", build_3q_5ops_mixed),
        ("3q/10ops mixed", build_3q_10ops_mixed),
        ("3q/15ops mixed", build_3q_15ops_mixed),
        ("3q/20ops mixed", build_3q_20ops_mixed),
        ("3q/25ops mixed", build_3q_25ops_mixed),
    ],
    4: [
        ("4q/15ops mixed", build_4q_15ops_mixed),
        ("4q/20ops mixed", build_4q_20ops_mixed),
        ("4q/25ops mixed", build_4q_25ops_mixed),
    ],
    5: [  # Slow with Qiskit but works with AerSimulator
        ("5q/30ops mixed", build_5q_30ops_mixed),
    ],
}
