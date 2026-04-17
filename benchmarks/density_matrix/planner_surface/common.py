from __future__ import annotations

import sys

import numpy as np
import scipy as sp

from squander import Variational_Quantum_Eigensolver


SIGMA_X = sp.sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
SIGMA_Y = sp.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
SIGMA_Z = sp.sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))

DEFAULT_ANSATZ = "HEA"
DEFAULT_LAYERS = 1
DEFAULT_INNER_BLOCKS = 1
PRIMARY_BACKEND = "density_matrix"


def build_open_chain_topology(qbit_num: int) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(qbit_num - 1)]


def _single_pauli_term(
    n_qubits: int, target: int, pauli: sp.sparse.csr_matrix
) -> sp.sparse.csr_matrix:
    result = None
    for qubit in range(n_qubits):
        factor = pauli if qubit == target else sp.sparse.eye(2, format="csr")
        result = factor if result is None else sp.sparse.kron(result, factor, format="csr")
    return result.tocsr()


def _two_pauli_term(
    n_qubits: int,
    first: int,
    second: int,
    pauli: sp.sparse.csr_matrix,
) -> sp.sparse.csr_matrix:
    result = None
    for qubit in range(n_qubits):
        factor = pauli if qubit in {first, second} else sp.sparse.eye(2, format="csr")
        result = factor if result is None else sp.sparse.kron(result, factor, format="csr")
    return result.tocsr()


def build_xxz_hamiltonian(
    n_qubits: int,
    *,
    topology: list[tuple[int, int]] | None = None,
    h: float = 0.5,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
) -> sp.sparse.csr_matrix:
    if topology is None:
        topology = build_open_chain_topology(n_qubits)

    hamiltonian = sp.sparse.csr_matrix(
        (2**n_qubits, 2**n_qubits), dtype=np.complex128
    )

    for control, target in topology:
        hamiltonian += -0.5 * jx * _two_pauli_term(n_qubits, control, target, SIGMA_X)
        hamiltonian += -0.5 * jy * _two_pauli_term(n_qubits, control, target, SIGMA_Y)
        hamiltonian += -0.5 * jz * _two_pauli_term(n_qubits, control, target, SIGMA_Z)

    for qubit in range(n_qubits):
        hamiltonian += -0.5 * h * _single_pauli_term(n_qubits, qubit, SIGMA_Z)

    return hamiltonian.tocsr()


def build_continuity_density_noise() -> list[dict]:
    return [
        {
            "channel": "local_depolarizing",
            "target": 0,
            "after_gate_index": 0,
            "error_rate": 0.1,
        },
        {
            "channel": "amplitude_damping",
            "target": 1,
            "after_gate_index": 2,
            "gamma": 0.05,
        },
        {
            "channel": "phase_damping",
            "target": 0,
            "after_gate_index": 4,
            "lambda": 0.07,
        },
    ]


def build_optimizer_config() -> dict:
    return {
        "max_inner_iterations": 4,
        "max_iterations": 1,
        "convergence_length": 2,
    }


def build_case_metadata(*, qbit_num: int, topology, density_noise) -> dict:
    return {
        "backend": PRIMARY_BACKEND,
        "qbit_num": qbit_num,
        "topology": list(topology),
        "ansatz": DEFAULT_ANSATZ,
        "layers": DEFAULT_LAYERS,
        "inner_blocks": DEFAULT_INNER_BLOCKS,
        "density_noise": [dict(item) for item in density_noise],
    }


def build_software_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": sp.__version__,
    }


def build_phase2_continuity_vqe(
    qbit_num: int,
    *,
    density_noise: list[dict] | None = None,
):
    requested_density_noise = (
        build_continuity_density_noise()
        if density_noise is None
        else [dict(item) for item in density_noise]
    )
    topology = build_open_chain_topology(qbit_num)
    hamiltonian = build_xxz_hamiltonian(qbit_num, topology=topology)
    vqe = Variational_Quantum_Eigensolver(
        hamiltonian,
        qbit_num,
        build_optimizer_config(),
        backend=PRIMARY_BACKEND,
        density_noise=requested_density_noise,
    )
    vqe.set_Ansatz(DEFAULT_ANSATZ)
    vqe.Generate_Circuit(layers=DEFAULT_LAYERS, inner_blocks=DEFAULT_INNER_BLOCKS)
    return vqe, hamiltonian, topology
