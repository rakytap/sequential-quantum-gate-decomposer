#!/usr/bin/env python3
"""Story 2 density-backend validation for anchor VQE cases.

Runs two kinds of evidence:
- fixed-parameter 4-qubit and 6-qubit XXZ anchor comparisons against Qiskit Aer,
- one bounded 4-qubit density-backed optimization trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.VQE.shot_noise_measurement import generate_zz_xx_hamiltonian
from squander import Variational_Quantum_Eigensolver


def build_story2_noise():
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


def build_open_chain_topology(qbit_num: int):
    return [(idx, idx + 1) for idx in range(qbit_num - 1)]


def build_story2_config():
    return {
        "max_inner_iterations": 4,
        "max_iterations": 1,
        "convergence_length": 2,
    }


def build_story2_parameters(param_num: int) -> np.ndarray:
    return np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)


def build_vqe(qbit_num: int, optimizer: str | None = None):
    topology = build_open_chain_topology(qbit_num)
    hamiltonian, _ = generate_zz_xx_hamiltonian(
        n_qubits=qbit_num,
        h=0.5,
        topology=topology,
        Jz=1.0,
        Jx=1.0,
        Jy=1.0,
    )
    vqe = Variational_Quantum_Eigensolver(
        hamiltonian,
        qbit_num,
        build_story2_config(),
        backend="density_matrix",
        density_noise=build_story2_noise(),
    )
    if optimizer is not None:
        vqe.set_Optimizer(optimizer)
    vqe.set_Ansatz("HEA")
    vqe.Generate_Circuit(layers=1, inner_blocks=1)
    return vqe, hamiltonian, topology


def insert_story2_noise(base_circuit: QuantumCircuit, density_noise):
    noisy_circuit = QuantumCircuit(base_circuit.num_qubits)
    noise_by_gate = {}
    for noise_spec in density_noise:
        noise_by_gate.setdefault(noise_spec["after_gate_index"], []).append(noise_spec)

    for gate_index, instruction in enumerate(base_circuit.data):
        qargs = [
            noisy_circuit.qubits[base_circuit.find_bit(qubit).index]
            for qubit in instruction.qubits
        ]
        cargs = [
            noisy_circuit.clbits[base_circuit.find_bit(clbit).index]
            for clbit in instruction.clbits
        ]
        noisy_circuit.append(instruction.operation, qargs, cargs)

        for noise_spec in noise_by_gate.get(gate_index, []):
            target = noise_spec["target"]
            channel = noise_spec["channel"]
            if channel == "local_depolarizing":
                noisy_circuit.append(
                    depolarizing_error(noise_spec["error_rate"], 1),
                    [noisy_circuit.qubits[target]],
                )
            elif channel == "amplitude_damping":
                noisy_circuit.append(
                    amplitude_damping_error(noise_spec["gamma"]),
                    [noisy_circuit.qubits[target]],
                )
            elif channel == "phase_damping":
                noisy_circuit.append(
                    phase_damping_error(noise_spec["lambda"]),
                    [noisy_circuit.qubits[target]],
                )
            else:
                raise ValueError(f"Unsupported Story 2 channel: {channel}")

    noisy_circuit.save_density_matrix()
    return noisy_circuit


def density_energy(hamiltonian, density_matrix):
    energy = np.trace(hamiltonian.dot(density_matrix))
    return float(np.real(energy)), float(np.imag(energy))


def run_fixed_parameter_case(qbit_num: int):
    vqe, hamiltonian, topology = build_vqe(qbit_num)
    params = build_story2_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(params)

    squander_energy = float(vqe.Optimization_Problem(params))

    noisy_qiskit_circuit = insert_story2_noise(
        vqe.get_Qiskit_Circuit(),
        build_story2_noise(),
    )
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(noisy_qiskit_circuit, shots=1).result()
    aer_rho = np.asarray(result.data()["density_matrix"])
    aer_energy_real, aer_energy_imag = density_energy(hamiltonian, aer_rho)

    trace_val = np.trace(aer_rho)
    purity = float(np.real(np.trace(aer_rho @ aer_rho)))
    return {
        "backend": "density_matrix",
        "qbit_num": qbit_num,
        "topology": topology,
        "hamiltonian": {
            "Jx": 1.0,
            "Jy": 1.0,
            "Jz": 1.0,
            "h": 0.5,
        },
        "layers": 1,
        "inner_blocks": 1,
        "density_noise": build_story2_noise(),
        "parameter_vector": params.tolist(),
        "squander_energy": squander_energy,
        "aer_energy_real": aer_energy_real,
        "aer_energy_imag": aer_energy_imag,
        "absolute_energy_error": abs(squander_energy - aer_energy_real),
        "aer_trace_real": float(np.real(trace_val)),
        "aer_trace_imag": float(np.imag(trace_val)),
        "aer_purity": purity,
    }


def run_optimization_trace():
    vqe, _, topology = build_vqe(4, optimizer="COSINE")
    initial_parameters = build_story2_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(initial_parameters)

    initial_energy = float(vqe.Optimization_Problem(initial_parameters))
    vqe.Start_Optimization()
    final_parameters = np.asarray(vqe.get_Optimized_Parameters(), dtype=np.float64)
    final_energy = float(vqe.Optimization_Problem(final_parameters))

    return {
        "backend": "density_matrix",
        "optimizer": "COSINE",
        "qbit_num": 4,
        "topology": topology,
        "layers": 1,
        "inner_blocks": 1,
        "density_noise": build_story2_noise(),
        "initial_parameters": initial_parameters.tolist(),
        "final_parameters": final_parameters.tolist(),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "status": "completed",
    }


def write_json(output_path: Path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON evidence artifacts.",
    )
    args = parser.parse_args()

    fixed_results = [run_fixed_parameter_case(4), run_fixed_parameter_case(6)]
    trace_result = run_optimization_trace()

    for result in fixed_results:
        print(
            "Story 2 fixed case:",
            result["qbit_num"],
            "qubits, |E_sq - E_aer| =",
            f"{result['absolute_energy_error']:.6e}",
        )

    print(
        "Story 2 optimization trace:",
        "initial =", f"{trace_result['initial_energy']:.6e}",
        "final =", f"{trace_result['final_energy']:.6e}",
    )

    if args.output_dir is not None:
        write_json(args.output_dir / "story2_fixed_4q.json", fixed_results[0])
        write_json(args.output_dir / "story2_fixed_6q.json", fixed_results[1])
        write_json(args.output_dir / "story2_trace_4q.json", trace_result)


if __name__ == "__main__":
    main()
