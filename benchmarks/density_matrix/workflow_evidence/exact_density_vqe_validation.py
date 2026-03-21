#!/usr/bin/env python3
"""Exact-density VQE validation for canonical anchor cases.

Runs the foundational exact density-matrix evidence used by the higher-level
Phase 2 bundles:
- fixed-parameter 4-qubit and 6-qubit XXZ anchor comparisons against Qiskit Aer,
- one bounded 4-qubit density-backed optimization trace,
- an exact-regime workflow matrix,
- and a compact validation bundle tying those artifacts together.

Unsupported cases are emitted as structured negative evidence instead of
crashing or silently disappearing.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import state_fidelity

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.VQE.shot_noise_measurement import generate_zz_xx_hamiltonian
from benchmarks.density_matrix.validate_squander_vs_qiskit import (
    ARTIFACT_FILENAME as MICRO_VALIDATION_BUNDLE_FILENAME,
    build_artifact_bundle as build_micro_validation_bundle,
    run_validation as run_micro_validation,
    write_artifact_bundle as write_micro_validation_bundle_file,
)
from benchmarks.density_matrix.noise_support.support_tiers import (
    build_required_case_classification,
    build_support_tier_summary,
    classify_noise_boundary_reason,
)
from squander import Variational_Quantum_Eigensolver
from squander.density_matrix import DensityMatrix, NoisyCircuit

PRIMARY_BACKEND = "density_matrix"
REFERENCE_BACKEND = "qiskit_aer_density_matrix"
DEFAULT_ANSATZ = "HEA"
DEFAULT_LAYERS = 1
DEFAULT_INNER_BLOCKS = 1
FIXED_PARAMETER_QUBITS = (4, 6)
EXACT_REGIME_WORKFLOW_QUBITS = (4, 6, 8, 10)
EXACT_REGIME_PARAMETER_SET_COUNT = 10
EXACT_REGIME_WORKFLOW_ERROR_TOL = 1e-8
EXACT_REGIME_VALIDITY_TOL = 1e-10
EXACT_REGIME_TRACE_TOL = 1e-10
EXACT_REGIME_OBSERVABLE_IMAG_TOL = 1e-10
EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME = "exact_regime_workflow_bundle.json"
OPTIMIZATION_TRACE_ARTIFACT_FILENAME = "optimization_trace_4q.json"
EXACT_REGIME_WORKFLOW_BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "thresholds",
    "software",
    "summary",
    "cases",
)
EXACT_DENSITY_VALIDATION_BUNDLE_FILENAME = "exact_density_validation_bundle.json"
EXACT_DENSITY_VALIDATION_BUNDLE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "software",
    "provenance",
    "summary",
    "artifacts",
)
SUPPORTED_BACKEND_LABELS = {PRIMARY_BACKEND, "state_vector"}
REQUIRED_BRIDGE_GATE_NAMES = {"U3", "CNOT"}
REQUIRED_BRIDGE_NOISE_NAMES = {
    "local_depolarizing",
    "amplitude_damping",
    "phase_damping",
}
ARTIFACT_CORE_FIELDS = (
    "case_name",
    "status",
    "backend",
    "qbit_num",
    "topology",
    "ansatz",
    "layers",
    "inner_blocks",
    "density_noise",
)


def build_reference_noise():
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


def build_optimizer_config():
    return {
        "max_inner_iterations": 4,
        "max_iterations": 1,
        "convergence_length": 2,
    }


def build_initial_parameters(param_num: int) -> np.ndarray:
    return np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)


def build_hamiltonian_metadata():
    return {
        "Jx": 1.0,
        "Jy": 1.0,
        "Jz": 1.0,
        "h": 0.5,
    }


def build_software_metadata():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "qiskit": getattr(qiskit, "__version__", "unknown"),
        "qiskit_aer": getattr(qiskit_aer, "__version__", "unknown"),
    }


def build_exact_regime_threshold_metadata(
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
):
    return {
        "absolute_energy_error": EXACT_REGIME_WORKFLOW_ERROR_TOL,
        "rho_is_valid_tol": EXACT_REGIME_VALIDITY_TOL,
        "trace_deviation": EXACT_REGIME_TRACE_TOL,
        "observable_imag_abs": EXACT_REGIME_OBSERVABLE_IMAG_TOL,
        "required_pass_rate": 1.0,
        "required_workflow_qubits": list(qubit_sizes),
        "fixed_parameter_sets_per_size": parameter_set_count,
    }


def build_exact_regime_parameter_sets(
    param_num: int, count: int = EXACT_REGIME_PARAMETER_SET_COUNT
):
    base = np.linspace(0.05, 0.05 * param_num, param_num, dtype=np.float64)
    parameter_sets = []
    for idx in range(count):
        shift = 0.07 * idx
        scale = 1.0 + 0.025 * idx
        params = np.mod(scale * base + shift, 2.0 * np.pi)
        parameter_sets.append(
            {
                "parameter_set_id": f"set_{idx:02d}",
                "parameter_vector": params,
            }
        )
    return parameter_sets


def get_git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_case_metadata(
    *,
    backend: str,
    qbit_num: int,
    topology,
    density_noise,
    ansatz: str = DEFAULT_ANSATZ,
    layers: int = DEFAULT_LAYERS,
    inner_blocks: int = DEFAULT_INNER_BLOCKS,
    reference_backend: str | None = None,
    hamiltonian: dict | None = None,
    optimizer: str | None = None,
):
    metadata = {
        "backend": backend,
        "qbit_num": qbit_num,
        "topology": topology,
        "ansatz": ansatz,
        "layers": layers,
        "inner_blocks": inner_blocks,
        "density_noise": density_noise,
    }
    if reference_backend is not None:
        metadata["reference_backend"] = reference_backend
    if hamiltonian is not None:
        metadata["hamiltonian"] = hamiltonian
    if optimizer is not None:
        metadata["optimizer"] = optimizer
    return metadata


def validate_artifact_payload(payload):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in payload]
    if missing_fields:
        raise ValueError(
            "Artifact payload is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    if payload["backend"] not in SUPPORTED_BACKEND_LABELS:
        raise ValueError(
            "Artifact payload has unsupported backend label '{}'".format(
                payload["backend"]
            )
        )

    if payload["status"] not in {"completed", "unsupported"}:
        raise ValueError(
            "Artifact payload has unsupported status '{}'".format(payload["status"])
        )


def build_vqe(
    qbit_num: int,
    optimizer: str | None = None,
    density_noise=None,
):
    topology = build_open_chain_topology(qbit_num)
    hamiltonian, _ = generate_zz_xx_hamiltonian(
        n_qubits=qbit_num,
        h=0.5,
        topology=topology,
        Jz=1.0,
        Jx=1.0,
        Jy=1.0,
    )
    requested_density_noise = (
        build_reference_noise() if density_noise is None else [dict(item) for item in density_noise]
    )
    vqe = Variational_Quantum_Eigensolver(
        hamiltonian,
        qbit_num,
        build_optimizer_config(),
        backend=PRIMARY_BACKEND,
        density_noise=requested_density_noise,
    )
    if optimizer is not None:
        vqe.set_Optimizer(optimizer)
    vqe.set_Ansatz(DEFAULT_ANSATZ)
    vqe.Generate_Circuit(layers=DEFAULT_LAYERS, inner_blocks=DEFAULT_INNER_BLOCKS)
    return vqe, hamiltonian, topology


def build_reference_bridge_metadata(vqe):
    bridge = vqe.describe_density_bridge()
    noise_operations = [
        op for op in bridge["operations"] if op["operation_class"] == "NoiseOperation"
    ]
    return {
        "bridge_source_type": bridge["source_type"],
        "bridge_parameter_count": bridge["parameter_count"],
        "bridge_operation_count": bridge["operation_count"],
        "bridge_gate_count": bridge["gate_count"],
        "bridge_noise_count": bridge["noise_count"],
        "bridge_operations": bridge["operations"],
        "bridge_noise_sequence": [op["name"] for op in noise_operations],
        "bridge_noise_targets": [op["target_qbit"] for op in noise_operations],
        "bridge_noise_after_gate_indices": [
            op["source_gate_index"] for op in noise_operations
        ],
        "bridge_noise_fixed_values": [op["fixed_value"] for op in noise_operations],
    }


def classify_bridge_unsupported_reason(reason: str):
    if "unsupported circuit source" in reason or "requires a generated HEA circuit" in reason:
        category = "circuit_source"
        if "custom_gate_structure" in reason:
            first_condition = "custom_gate_structure"
            bridge_source_type = "custom_gate_structure"
        elif "binary_import" in reason:
            first_condition = "binary_import"
            bridge_source_type = "binary_import"
        elif "generated HEA circuit" in reason:
            first_condition = "generated_hea_required"
            bridge_source_type = "unset"
        else:
            first_condition = "circuit_source"
            bridge_source_type = "unset"
    elif (
        "only the HEA ansatz" in reason
        or "first unsupported gate in density backend path" in reason
        or "unsupported gate in density backend path" in reason
    ):
        category = "lowering_path"
        if "only the HEA ansatz" in reason:
            first_condition = "generated_hea_zyz"
            bridge_source_type = "generated_hea_zyz"
        elif "first unsupported gate in density backend path is " in reason:
            first_condition = reason.split(
                "first unsupported gate in density backend path is ", 1
            )[1]
            bridge_source_type = "generated_hea"
        elif "unsupported gate in density backend path: " in reason:
            first_condition = reason.split(
                "unsupported gate in density backend path: ", 1
            )[1]
            bridge_source_type = "generated_hea"
        else:
            first_condition = "lowering_path"
            bridge_source_type = "generated_hea"
    elif "unsupported density-noise insertion" in reason or "after_gate_index" in reason:
        category = "noise_insertion"
        first_condition = "after_gate_index"
        bridge_source_type = "generated_hea"
    elif "Unsupported density-noise channel" in reason or "unsupported density-noise type" in reason:
        category = "noise_type"
        first_condition = "unsupported_density_noise_channel"
        bridge_source_type = "unset"
    else:
        category = "workflow_execution"
        first_condition = "workflow_execution"
        bridge_source_type = "unknown"

    return {
        "unsupported_category": category,
        "first_unsupported_condition": first_condition,
        "bridge_source_type": bridge_source_type,
        "error_match_pass": category != "workflow_execution",
    }


def build_exact_regime_bridge_metadata(vqe, *, execution_ready: bool):
    bridge = vqe.describe_density_bridge()
    gate_sequence = [
        op["name"] for op in bridge["operations"] if op["operation_class"] == "GateOperation"
    ]
    noise_sequence = [
        op["name"] for op in bridge["operations"] if op["operation_class"] == "NoiseOperation"
    ]

    source_pass = bridge["source_type"] == "generated_hea"
    gate_pass = (
        bool(gate_sequence)
        and set(gate_sequence).issubset(REQUIRED_BRIDGE_GATE_NAMES)
        and "U3" in gate_sequence
        and (vqe.get_Qbit_Num() == 1 or "CNOT" in gate_sequence)
    )
    noise_pass = REQUIRED_BRIDGE_NOISE_NAMES.issubset(set(noise_sequence))
    execution_ready = bool(execution_ready)

    return {
        "bridge_source_type": bridge["source_type"],
        "bridge_parameter_count": bridge["parameter_count"],
        "bridge_operation_count": bridge["operation_count"],
        "bridge_gate_count": bridge["gate_count"],
        "bridge_noise_count": bridge["noise_count"],
        "bridge_gate_sequence": gate_sequence,
        "bridge_noise_sequence": noise_sequence,
        "source_pass": source_pass,
        "gate_pass": gate_pass,
        "noise_pass": noise_pass,
        "execution_ready": execution_ready,
        "bridge_supported_pass": source_pass
        and gate_pass
        and noise_pass
        and execution_ready,
    }


def build_unsupported_state_vector_density_noise_vqe(qbit_num: int):
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
        build_optimizer_config(),
        backend="state_vector",
        density_noise=build_reference_noise(),
    )
    vqe.set_Ansatz(DEFAULT_ANSATZ)
    vqe.Generate_Circuit(layers=DEFAULT_LAYERS, inner_blocks=DEFAULT_INNER_BLOCKS)
    return vqe, hamiltonian, topology


def insert_reference_noise(base_circuit: QuantumCircuit, density_noise):
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
                raise ValueError(f"Unsupported fixed-parameter channel: {channel}")

    noisy_circuit.save_density_matrix()
    return noisy_circuit


def density_energy(hamiltonian, density_matrix):
    energy = np.trace(hamiltonian.dot(density_matrix))
    return float(np.real(energy)), float(np.imag(energy))


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    diff = rho1 - rho2
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    return float(0.5 * np.sum(np.sqrt(np.maximum(eigenvalues, 0))))


def build_squander_density_from_qiskit_circuit(base_circuit, density_noise):
    circuit = NoisyCircuit(base_circuit.num_qubits)
    params = []
    noise_by_gate = {}
    for noise_spec in density_noise:
        noise_by_gate.setdefault(noise_spec["after_gate_index"], []).append(noise_spec)

    for gate_index, instruction in enumerate(base_circuit.data):
        qubit_indices = [
            base_circuit.find_bit(qubit).index for qubit in instruction.qubits
        ]
        gate_name = instruction.operation.name

        if gate_name == "u":
            theta, phi, lam = [float(value) for value in instruction.operation.params]
            # Exported Qiskit circuits use the physical U3 theta, while the
            # SQUANDER parametric path stores theta/2 internally.
            circuit.add_U3(qubit_indices[0])
            params.extend([theta / 2.0, phi, lam])
        elif gate_name in {"cx", "cnot"}:
            circuit.add_CNOT(qubit_indices[1], qubit_indices[0])
        else:
            raise ValueError(
                f"Unsupported workflow gate in reconstruction: {gate_name}"
            )

        for noise_spec in noise_by_gate.get(gate_index, []):
            target = noise_spec["target"]
            channel = noise_spec["channel"]
            if channel == "local_depolarizing":
                circuit.add_local_depolarizing(
                    target, error_rate=noise_spec["error_rate"]
                )
            elif channel == "amplitude_damping":
                circuit.add_amplitude_damping(target, gamma=noise_spec["gamma"])
            elif channel == "phase_damping":
                circuit.add_phase_damping(target, lambda_param=noise_spec["lambda"])
            else:
                raise ValueError(
                    f"Unsupported workflow noise channel in reconstruction: {channel}"
                )

    rho = DensityMatrix(base_circuit.num_qubits)
    circuit.apply_to(np.asarray(params, dtype=np.float64), rho)
    return rho


def run_exact_regime_workflow_case(
    qbit_num: int,
    parameter_set_id: str,
    parameter_vector: np.ndarray,
):
    density_noise = build_reference_noise()
    vqe, hamiltonian, topology = build_vqe(qbit_num)
    vqe.set_Optimized_Parameters(parameter_vector)
    bridge_metadata = build_exact_regime_bridge_metadata(vqe, execution_ready=True)

    case_start = time.perf_counter()
    squander_start = time.perf_counter()
    squander_energy = float(vqe.Optimization_Problem(parameter_vector))
    squander_runtime_ms = (time.perf_counter() - squander_start) * 1000.0

    base_qiskit_circuit = vqe.get_Qiskit_Circuit()
    squander_rho = build_squander_density_from_qiskit_circuit(
        base_qiskit_circuit, density_noise
    )
    squander_rho_np = np.asarray(squander_rho.to_numpy())
    squ_trace = squander_rho.trace()
    squ_trace_deviation = float(abs(squ_trace - 1.0))
    squ_density_valid = bool(squander_rho.is_valid(tol=EXACT_REGIME_VALIDITY_TOL))
    squ_purity = float(np.real(np.trace(squander_rho_np @ squander_rho_np)))
    squ_energy_real_from_rho, squ_energy_imag = density_energy(
        hamiltonian, squander_rho_np
    )

    reference_start = time.perf_counter()
    noisy_qiskit_circuit = insert_reference_noise(base_qiskit_circuit, density_noise)
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(noisy_qiskit_circuit, shots=1).result()
    aer_rho = np.asarray(result.data()["density_matrix"])
    reference_runtime_ms = (time.perf_counter() - reference_start) * 1000.0

    aer_energy_real, aer_energy_imag = density_energy(hamiltonian, aer_rho)
    aer_trace = np.trace(aer_rho)
    aer_purity = float(np.real(np.trace(aer_rho @ aer_rho)))
    fidelity = float(
        state_fidelity(
            QiskitDensityMatrix(squander_rho_np), QiskitDensityMatrix(aer_rho)
        )
    )
    density_max_diff = float(np.max(np.abs(squander_rho_np - aer_rho)))
    density_trace_distance = trace_distance(squander_rho_np, aer_rho)

    energy_error = float(abs(squander_energy - aer_energy_real))
    energy_pass = energy_error <= EXACT_REGIME_WORKFLOW_ERROR_TOL
    density_valid_pass = squ_density_valid
    trace_pass = squ_trace_deviation <= EXACT_REGIME_TRACE_TOL
    observable_pass = abs(squ_energy_imag) <= EXACT_REGIME_OBSERVABLE_IMAG_TOL
    workflow_completed = True
    status = (
        "pass"
        if workflow_completed
        and energy_pass
        and density_valid_pass
        and trace_pass
        and observable_pass
        and bridge_metadata["bridge_supported_pass"]
        else "fail"
    )

    artifact = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=qbit_num,
        topology=topology,
        density_noise=density_noise,
        reference_backend=REFERENCE_BACKEND,
        hamiltonian=build_hamiltonian_metadata(),
    )
    artifact.update(
        {
            "case_name": f"exact_regime_{qbit_num}q_{parameter_set_id}",
            "status": status,
            "workflow_completed": workflow_completed,
            "parameter_set_id": parameter_set_id,
            "parameter_vector": parameter_vector.tolist(),
            "squander_energy": squander_energy,
            "squander_energy_from_rho_real": squ_energy_real_from_rho,
            "squander_energy_imag": squ_energy_imag,
            "aer_energy_real": aer_energy_real,
            "aer_energy_imag": aer_energy_imag,
            "absolute_energy_error": energy_error,
            "energy_pass": energy_pass,
            "rho_is_valid": squ_density_valid,
            "rho_is_valid_tol": EXACT_REGIME_VALIDITY_TOL,
            "density_valid_pass": density_valid_pass,
            "squander_trace_real": float(np.real(squ_trace)),
            "squander_trace_imag": float(np.imag(squ_trace)),
            "trace_deviation": squ_trace_deviation,
            "trace_pass": trace_pass,
            "observable_pass": observable_pass,
            "squander_purity": squ_purity,
            "aer_trace_real": float(np.real(aer_trace)),
            "aer_trace_imag": float(np.imag(aer_trace)),
            "aer_purity": aer_purity,
            "state_fidelity": fidelity,
            "density_max_diff": density_max_diff,
            "density_trace_distance": density_trace_distance,
            "squander_runtime_ms": squander_runtime_ms,
            "reference_runtime_ms": reference_runtime_ms,
            "total_case_runtime_ms": (time.perf_counter() - case_start) * 1000.0,
            "process_peak_rss_kb": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            **bridge_metadata,
            **build_required_case_classification(),
        }
    )
    return artifact


def capture_exact_regime_workflow_case(qbit_num: int, parameter_set: dict):
    parameter_set_id = parameter_set["parameter_set_id"]
    parameter_vector = np.asarray(parameter_set["parameter_vector"], dtype=np.float64)
    try:
        return run_exact_regime_workflow_case(qbit_num, parameter_set_id, parameter_vector)
    except Exception as exc:
        unsupported_reason = str(exc)
        unsupported_metadata = classify_bridge_unsupported_reason(unsupported_reason)
        noise_boundary_metadata = classify_noise_boundary_reason(
            unsupported_reason
        )
        if noise_boundary_metadata["unsupported_category"] != "workflow_execution":
            unsupported_metadata["unsupported_category"] = noise_boundary_metadata[
                "unsupported_category"
            ]
            unsupported_metadata["first_unsupported_condition"] = (
                noise_boundary_metadata["first_unsupported_condition"]
            )
        unsupported_metadata["noise_boundary_class"] = noise_boundary_metadata[
            "noise_boundary_class"
        ]
        unsupported_metadata["failure_stage"] = noise_boundary_metadata[
            "failure_stage"
        ]
        return {
            "case_name": f"exact_regime_{qbit_num}q_{parameter_set_id}",
            "status": "unsupported",
            "backend": PRIMARY_BACKEND,
            "reference_backend": REFERENCE_BACKEND,
            "qbit_num": qbit_num,
            "workflow_completed": False,
            "parameter_set_id": parameter_set_id,
            "parameter_vector": parameter_vector.tolist(),
            **unsupported_metadata,
            "unsupported_reason": unsupported_reason,
            "source_pass": False,
            "gate_pass": False,
            "noise_pass": False,
            "execution_ready": False,
            "bridge_supported_pass": False,
            "energy_pass": False,
            "density_valid_pass": False,
            "trace_pass": False,
            "observable_pass": False,
            **build_required_case_classification(),
        }


def run_exact_regime_workflow_matrix(
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
):
    results = []
    for qbit_num in qubit_sizes:
        preview_vqe, _, _ = build_vqe(qbit_num)
        parameter_sets = build_exact_regime_parameter_sets(
            preview_vqe.get_Parameter_Num(), count=parameter_set_count
        )
        for parameter_set in parameter_sets:
            results.append(capture_exact_regime_workflow_case(qbit_num, parameter_set))
    return results


def build_exact_regime_workflow_bundle(
    results,
    qubit_sizes=EXACT_REGIME_WORKFLOW_QUBITS,
    parameter_set_count: int = EXACT_REGIME_PARAMETER_SET_COUNT,
    trace_result=None,
):
    support_tier_summary = build_support_tier_summary(results)
    cases_per_qbit = {
        str(qbit_num): sum(1 for result in results if result["qbit_num"] == qbit_num)
        for qbit_num in qubit_sizes
    }
    passed = sum(1 for result in results if result["status"] == "pass")
    unsupported = sum(1 for result in results if result["status"] == "unsupported")
    total = len(results)
    required_pass_rate = support_tier_summary["required_pass_rate"]
    documented_10q_anchor_present = any(result["qbit_num"] == 10 for result in results)
    documented_10q_anchor_required = 10 in qubit_sizes
    required_counts_present = all(
        cases_per_qbit[str(qbit_num)] >= parameter_set_count for qbit_num in qubit_sizes
    )
    supported_trace_completed = bool(
        trace_result
        and trace_result.get("workflow_completed", False)
        and trace_result.get("bridge_supported_pass", False)
    )
    bridge_supported_cases = sum(
        1 for result in results if result.get("bridge_supported_pass", False)
    )
    bundle_status = (
        "pass"
        if support_tier_summary["mandatory_baseline_completed"]
        and required_counts_present
        and (
            not documented_10q_anchor_required or documented_10q_anchor_present
        )
        and (trace_result is None or supported_trace_completed)
        else "fail"
    )

    bundle = {
        "suite_name": "exact_regime_workflow",
        "status": bundle_status,
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "thresholds": build_exact_regime_threshold_metadata(
            qubit_sizes=qubit_sizes,
            parameter_set_count=parameter_set_count,
        ),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": total - passed,
            "unsupported_cases": unsupported,
            "unsupported_status_cases": unsupported,
            "bridge_supported_cases": bridge_supported_cases,
            "pass_rate": required_pass_rate,
            "required_workflow_qubits": list(qubit_sizes),
            "fixed_parameter_sets_per_size": parameter_set_count,
            "cases_per_qbit": cases_per_qbit,
            "documented_10q_anchor_present": documented_10q_anchor_present,
            "documented_10q_anchor_required": documented_10q_anchor_required,
            "supported_trace_completed": supported_trace_completed,
            "supported_trace_case_name": (
                trace_result.get("case_name", "optimization_trace_4q")
                if trace_result
                else None
            ),
            "required_cases": support_tier_summary["required_cases"],
            "required_passed_cases": support_tier_summary["required_passed_cases"],
            "required_pass_rate": required_pass_rate,
            "mandatory_baseline_case_count": support_tier_summary[
                "mandatory_baseline_case_count"
            ],
            "mandatory_baseline_passed_cases": support_tier_summary[
                "mandatory_baseline_passed_cases"
            ],
            "mandatory_baseline_completed": support_tier_summary[
                "mandatory_baseline_completed"
            ],
            "support_tiers_present": support_tier_summary["support_tiers_present"],
            "optional_cases_count_toward_mandatory_baseline": support_tier_summary[
                "optional_cases_count_toward_mandatory_baseline"
            ],
        },
        "cases": results,
    }
    validate_exact_regime_workflow_bundle(bundle)
    return bundle


def validate_exact_regime_workflow_bundle(bundle):
    missing_fields = [
        field for field in EXACT_REGIME_WORKFLOW_BUNDLE_FIELDS if field not in bundle
    ]
    if missing_fields:
        raise ValueError(
            "Exact-regime workflow bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    required_qubits = bundle["thresholds"]["required_workflow_qubits"]
    required_count = bundle["thresholds"]["fixed_parameter_sets_per_size"]
    cases_per_qbit = bundle["summary"]["cases_per_qbit"]
    for qbit_num in required_qubits:
        if cases_per_qbit.get(str(qbit_num), 0) < required_count:
            raise ValueError(
                f"Exact-regime workflow bundle is missing required cases for {qbit_num} qubits"
            )


def write_exact_regime_workflow_bundle(output_path: Path, bundle):
    validate_exact_regime_workflow_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def print_exact_regime_workflow_summary(bundle):
    print("\n" + "=" * 78)
    print(
        "  Exact-Regime Workflow Bundle [{} vs {}]".format(
            bundle["backend"], bundle["reference_backend"]
        )
    )
    print("=" * 78)
    for qbit_num in bundle["summary"]["required_workflow_qubits"]:
        count = bundle["summary"]["cases_per_qbit"][str(qbit_num)]
        print(f"  {qbit_num} qubits: {count} mandatory fixed-parameter cases recorded")
    print(
        "  Pass rate: {}/{} ({:.0%})".format(
            bundle["summary"]["passed_cases"],
            bundle["summary"]["total_cases"],
            bundle["summary"]["pass_rate"],
        )
    )
    print(
        "  Documented 10-qubit anchor present:",
        bundle["summary"]["documented_10q_anchor_present"],
    )


def _build_validation_bundle_artifact_entry(
    *,
    artifact_id,
    artifact_class,
    mandatory,
    path,
    status,
    expected_statuses,
    purpose,
    generation_command,
    summary=None,
):
    return {
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "mandatory": mandatory,
        "path": path,
        "status": status,
        "expected_statuses": list(expected_statuses),
        "purpose": purpose,
        "generation_command": generation_command,
        "summary": {} if summary is None else dict(summary),
    }


def build_exact_density_validation_bundle(
    output_dir: Path,
    *,
    fixed_results,
    trace_result,
    unsupported_result,
    micro_bundle,
    workflow_bundle,
):
    output_dir = Path(output_dir)
    base_validation_command = (
        f"python benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py "
        f"--output-dir {output_dir}"
    )
    workflow_command = (
        f"python benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py "
        f"--workflow-bundle --output-dir {output_dir}"
    )
    micro_validation_command = (
        f"python benchmarks/density_matrix/validate_squander_vs_qiskit.py "
        f"--output-dir {output_dir}"
    )
    validation_bundle_command = (
        f"python benchmarks/density_matrix/workflow_evidence/exact_density_vqe_validation.py "
        f"--publication-bundle --output-dir {output_dir}"
    )

    artifacts = [
        _build_validation_bundle_artifact_entry(
            artifact_id="micro_validation_bundle",
            artifact_class="micro_validation_bundle",
            mandatory=True,
            path=MICRO_VALIDATION_BUNDLE_FILENAME,
            status=micro_bundle["status"],
            expected_statuses=["pass"],
            purpose="Local exactness and density-validity gate for mandatory 1 to 3 qubit microcases.",
            generation_command=micro_validation_command,
            summary={
                "total_cases": micro_bundle["summary"]["total_cases"],
                "passed_cases": micro_bundle["summary"]["passed_cases"],
                "pass_rate": micro_bundle["summary"]["pass_rate"],
            },
        ),
        _build_validation_bundle_artifact_entry(
            artifact_id="exact_regime_workflow_bundle",
            artifact_class="workflow_exact_regime_bundle",
            mandatory=True,
            path=EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
            status=workflow_bundle["status"],
            expected_statuses=["pass"],
            purpose="Workflow-scale exactness evidence across the mandatory 4/6/8/10 exact regime.",
            generation_command=workflow_command,
            summary={
                "total_cases": workflow_bundle["summary"]["total_cases"],
                "passed_cases": workflow_bundle["summary"]["passed_cases"],
                "pass_rate": workflow_bundle["summary"]["pass_rate"],
                "documented_10q_anchor_present": workflow_bundle["summary"][
                    "documented_10q_anchor_present"
                ],
            },
        ),
        _build_validation_bundle_artifact_entry(
            artifact_id="fixed_parameters_4q",
            artifact_class="fixed_parameter_case",
            mandatory=True,
            path="fixed_parameters_4q.json",
            status=fixed_results[0]["status"],
            expected_statuses=["completed"],
            purpose="Supported fixed-parameter 4-qubit anchor exactness evidence.",
            generation_command=base_validation_command,
            summary={
                "qbit_num": fixed_results[0]["qbit_num"],
                "absolute_energy_error": fixed_results[0].get("absolute_energy_error"),
            },
        ),
        _build_validation_bundle_artifact_entry(
            artifact_id="fixed_parameters_6q",
            artifact_class="fixed_parameter_case",
            mandatory=True,
            path="fixed_parameters_6q.json",
            status=fixed_results[1]["status"],
            expected_statuses=["completed"],
            purpose="Supported fixed-parameter 6-qubit anchor exactness evidence.",
            generation_command=base_validation_command,
            summary={
                "qbit_num": fixed_results[1]["qbit_num"],
                "absolute_energy_error": fixed_results[1].get("absolute_energy_error"),
            },
        ),
        _build_validation_bundle_artifact_entry(
            artifact_id="optimization_trace_4q",
            artifact_class="optimization_trace",
            mandatory=True,
            path="optimization_trace_4q.json",
            status=trace_result["status"],
            expected_statuses=["completed"],
            purpose="Reproducible bounded 4-qubit optimization trace for the exact noisy anchor workflow.",
            generation_command=base_validation_command,
            summary={
                "optimizer": trace_result.get("optimizer"),
                "parameter_count": trace_result.get("parameter_count"),
                "workflow_completed": trace_result.get("workflow_completed"),
                "initial_energy": trace_result.get("initial_energy"),
                "final_energy": trace_result.get("final_energy"),
            },
        ),
        _build_validation_bundle_artifact_entry(
            artifact_id="unsupported_state_vector_density_noise",
            artifact_class="unsupported_case",
            mandatory=True,
            path="unsupported_state_vector_density_noise.json",
            status=unsupported_result["status"],
            expected_statuses=["unsupported"],
            purpose="Structured negative evidence for backend-mismatch and mixed-state-only unsupported behavior.",
            generation_command=base_validation_command,
            summary={
                "unsupported_category": unsupported_result.get("unsupported_category"),
                "unsupported_reason": unsupported_result.get("unsupported_reason"),
            },
        ),
    ]

    mandatory_artifacts = [artifact for artifact in artifacts if artifact["mandatory"]]
    present_count = 0
    status_match_count = 0
    for artifact in mandatory_artifacts:
        if (output_dir / artifact["path"]).exists():
            present_count += 1
        if artifact["status"] in artifact["expected_statuses"]:
            status_match_count += 1

    bundle_status = (
        "pass"
        if present_count == len(mandatory_artifacts)
        and status_match_count == len(mandatory_artifacts)
        else "fail"
    )

    bundle = {
        "suite_name": "exact_density_validation",
        "status": bundle_status,
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "software": build_software_metadata(),
        "provenance": {
            "generation_command": validation_bundle_command,
            "working_directory": str(REPO_ROOT),
            "git_revision": get_git_revision(),
        },
        "summary": {
            "mandatory_artifact_count": len(mandatory_artifacts),
            "present_artifact_count": present_count,
            "status_match_count": status_match_count,
            "missing_artifact_count": len(mandatory_artifacts) - present_count,
            "mismatched_status_count": len(mandatory_artifacts) - status_match_count,
        },
        "artifacts": artifacts,
    }
    validate_exact_density_validation_bundle(bundle, output_dir)
    return bundle


def validate_exact_density_validation_bundle(bundle, bundle_dir: Path):
    missing_fields = [field for field in EXACT_DENSITY_VALIDATION_BUNDLE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Exact-density validation bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    artifact_ids = {artifact["artifact_id"] for artifact in bundle["artifacts"]}
    required_ids = {
        "micro_validation_bundle",
        "exact_regime_workflow_bundle",
        "fixed_parameters_4q",
        "fixed_parameters_6q",
        "optimization_trace_4q",
        "unsupported_state_vector_density_noise",
    }
    if required_ids - artifact_ids:
        raise ValueError(
            "Exact-density validation bundle is missing required artifact IDs: {}".format(
                ", ".join(sorted(required_ids - artifact_ids))
            )
        )

    for artifact in bundle["artifacts"]:
        artifact_path = bundle_dir / artifact["path"]
        if artifact["mandatory"] and not artifact_path.exists():
            raise ValueError(
                f"Exact-density validation bundle is missing artifact file: {artifact['path']}"
            )
        if artifact["status"] not in artifact["expected_statuses"]:
            raise ValueError(
                f"Exact-density validation artifact {artifact['artifact_id']} has unexpected status {artifact['status']}"
            )


def write_exact_density_validation_bundle(output_path: Path, bundle):
    validate_exact_density_validation_bundle(bundle, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def print_exact_density_validation_bundle_summary(bundle):
    print("\n" + "=" * 78)
    print(
        "  Exact-Density Validation Bundle [{} vs {}]".format(
            bundle["backend"], bundle["reference_backend"]
        )
    )
    print("=" * 78)
    print(
        "  Mandatory artifacts present: {}/{}".format(
            bundle["summary"]["present_artifact_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    print(
        "  Status matches: {}/{}".format(
            bundle["summary"]["status_match_count"],
            bundle["summary"]["mandatory_artifact_count"],
        )
    )
    print("  Git revision:", bundle["provenance"]["git_revision"])


def generate_exact_density_validation_bundle(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_results = [
        capture_case(
            f"fixed_parameters_{qbit_num}q",
            lambda qbit_num=qbit_num: run_fixed_parameter_case(qbit_num),
        )
        for qbit_num in FIXED_PARAMETER_QUBITS
    ]
    trace_result = capture_case("optimization_trace_4q", run_optimization_trace)
    unsupported_base_metadata = build_case_metadata(
        backend="state_vector",
        qbit_num=4,
        topology=build_open_chain_topology(4),
        density_noise=build_reference_noise(),
        hamiltonian=build_hamiltonian_metadata(),
    )
    unsupported_result = capture_case(
        "unsupported_state_vector_density_noise",
        run_unsupported_state_vector_density_noise_case,
        base_metadata=unsupported_base_metadata,
    )

    for result in fixed_results:
        write_json(output_dir / f"{result['case_name']}.json", result)
    write_json(output_dir / "optimization_trace_4q.json", trace_result)
    write_json(
        output_dir / "unsupported_state_vector_density_noise.json",
        unsupported_result,
    )

    micro_results = run_micro_validation()
    micro_bundle = build_micro_validation_bundle(micro_results)
    write_micro_validation_bundle_file(output_dir / MICRO_VALIDATION_BUNDLE_FILENAME, micro_bundle)

    workflow_results = run_exact_regime_workflow_matrix()
    workflow_bundle = build_exact_regime_workflow_bundle(workflow_results)
    write_exact_regime_workflow_bundle(
        output_dir / EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME, workflow_bundle
    )

    validation_bundle = build_exact_density_validation_bundle(
        output_dir,
        fixed_results=fixed_results,
        trace_result=trace_result,
        unsupported_result=unsupported_result,
        micro_bundle=micro_bundle,
        workflow_bundle=workflow_bundle,
    )
    write_exact_density_validation_bundle(output_dir / EXACT_DENSITY_VALIDATION_BUNDLE_FILENAME, validation_bundle)
    return validation_bundle


def run_fixed_parameter_case(qbit_num: int):
    vqe, hamiltonian, topology = build_vqe(qbit_num)
    params = build_initial_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(params)
    bridge_metadata = build_reference_bridge_metadata(vqe)

    squander_energy = float(vqe.Optimization_Problem(params))

    noisy_qiskit_circuit = insert_reference_noise(
        vqe.get_Qiskit_Circuit(),
        build_reference_noise(),
    )
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(noisy_qiskit_circuit, shots=1).result()
    aer_rho = np.asarray(result.data()["density_matrix"])
    aer_energy_real, aer_energy_imag = density_energy(hamiltonian, aer_rho)

    trace_val = np.trace(aer_rho)
    purity = float(np.real(np.trace(aer_rho @ aer_rho)))
    artifact = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=qbit_num,
        topology=topology,
        density_noise=build_reference_noise(),
        reference_backend=REFERENCE_BACKEND,
        hamiltonian=build_hamiltonian_metadata(),
    )
    artifact.update(
        {
            "status": "completed",
            "parameter_vector": params.tolist(),
            "squander_energy": squander_energy,
            "aer_energy_real": aer_energy_real,
            "aer_energy_imag": aer_energy_imag,
            "absolute_energy_error": abs(squander_energy - aer_energy_real),
            "aer_trace_real": float(np.real(trace_val)),
            "aer_trace_imag": float(np.imag(trace_val)),
            "aer_purity": purity,
            **bridge_metadata,
        }
    )
    return artifact


def run_optimization_trace():
    vqe, _, topology = build_vqe(4, optimizer="COSINE")
    initial_parameters = build_initial_parameters(vqe.get_Parameter_Num())
    vqe.set_Optimized_Parameters(initial_parameters)
    bridge_metadata = build_exact_regime_bridge_metadata(vqe, execution_ready=True)

    trace_start = time.perf_counter()
    initial_energy = float(vqe.Optimization_Problem(initial_parameters))
    vqe.Start_Optimization()
    final_parameters = np.asarray(vqe.get_Optimized_Parameters(), dtype=np.float64)
    final_energy = float(vqe.Optimization_Problem(final_parameters))

    artifact = build_case_metadata(
        backend=PRIMARY_BACKEND,
        qbit_num=4,
        topology=topology,
        density_noise=build_reference_noise(),
        hamiltonian=build_hamiltonian_metadata(),
        optimizer="COSINE",
    )
    artifact.update(
        {
            "initial_parameters": initial_parameters.tolist(),
            "final_parameters": final_parameters.tolist(),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_improvement": initial_energy - final_energy,
            "optimizer_config": build_optimizer_config(),
            "parameter_count": int(initial_parameters.size),
            "workflow_completed": True,
            "trace_kind": "bounded_optimization_trace",
            "total_trace_runtime_ms": (time.perf_counter() - trace_start) * 1000.0,
            "process_peak_rss_kb": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "status": "completed",
            **bridge_metadata,
            **build_required_case_classification(),
            "counts_toward_mandatory_baseline": False,
            "required_validation_trace": True,
        }
    )
    return artifact


def run_unsupported_state_vector_density_noise_case():
    vqe, _, topology = build_unsupported_state_vector_density_noise_vqe(4)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    artifact = build_case_metadata(
        backend="state_vector",
        qbit_num=4,
        topology=topology,
        density_noise=build_reference_noise(),
        hamiltonian=build_hamiltonian_metadata(),
    )
    artifact.update(
        {
            "parameter_vector": parameters.tolist(),
            "energy": float(vqe.Optimization_Problem(parameters)),
            "status": "completed",
        }
    )
    return artifact


def capture_case(case_name: str, case_callable, base_metadata=None):
    base_metadata = {} if base_metadata is None else dict(base_metadata)
    try:
        result = dict(base_metadata)
        result.update(case_callable())
        result.setdefault("status", "completed")
        result["case_name"] = case_name
    except Exception as exc:
        result = dict(base_metadata)
        result.update(
            {
                "case_name": case_name,
                "status": "unsupported",
                "unsupported_category": "phase2_support_matrix",
                "unsupported_reason": str(exc),
            }
        )

    validate_artifact_payload(result)
    return result


def write_json(output_path: Path, payload):
    validate_artifact_payload(payload)
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
    parser.add_argument(
        "--workflow-bundle",
        action="store_true",
        help="Run the workflow-scale exact-regime validation matrix and emit the bundle.",
    )
    parser.add_argument(
        "--publication-bundle",
        action="store_true",
        help="Run the full exact-density validation bundle workflow.",
    )
    args = parser.parse_args()

    if args.publication_bundle:
        if args.output_dir is None:
            raise ValueError(
                "Exact-density validation bundle generation requires --output-dir"
            )
        validation_bundle = generate_exact_density_validation_bundle(args.output_dir)
        print_exact_density_validation_bundle_summary(validation_bundle)
        if validation_bundle["status"] != "pass":
            raise SystemExit(1)
        return

    if args.workflow_bundle:
        workflow_results = run_exact_regime_workflow_matrix()
        trace_result = capture_case("optimization_trace_4q", run_optimization_trace)
        workflow_bundle = build_exact_regime_workflow_bundle(
            workflow_results, trace_result=trace_result
        )
        print_exact_regime_workflow_summary(workflow_bundle)
        if trace_result["status"] == "completed":
            print(
                "  Workflow supported trace [{}]:".format(trace_result["backend"]),
                trace_result["trace_kind"],
                "bridge_pass =",
                trace_result["bridge_supported_pass"],
            )

        if args.output_dir is not None:
            write_exact_regime_workflow_bundle(
                args.output_dir / EXACT_REGIME_WORKFLOW_BUNDLE_FILENAME,
                workflow_bundle,
            )
            write_json(args.output_dir / OPTIMIZATION_TRACE_ARTIFACT_FILENAME, trace_result)

        if workflow_bundle["status"] != "pass":
            raise SystemExit(1)
        return

    fixed_results = [
        capture_case(
            f"fixed_parameters_{qbit_num}q",
            lambda qbit_num=qbit_num: run_fixed_parameter_case(qbit_num),
        )
        for qbit_num in FIXED_PARAMETER_QUBITS
    ]
    trace_result = capture_case("optimization_trace_4q", run_optimization_trace)
    unsupported_base_metadata = build_case_metadata(
        backend="state_vector",
        qbit_num=4,
        topology=build_open_chain_topology(4),
        density_noise=build_reference_noise(),
        hamiltonian=build_hamiltonian_metadata(),
    )
    unsupported_result = capture_case(
        "unsupported_state_vector_density_noise",
        run_unsupported_state_vector_density_noise_case,
        base_metadata=unsupported_base_metadata,
    )

    for result in fixed_results:
        if result["status"] == "completed":
            print(
                "Fixed-parameter case [{}]:".format(result["backend"]),
                result["qbit_num"],
                "qubits, |E_sq - E_aer| =",
                f"{result['absolute_energy_error']:.6e}",
            )
        else:
            print(
                "Fixed-parameter case [{}]:".format(result["backend"]),
                result["status"],
                result["case_name"],
                result["unsupported_reason"],
            )

    if trace_result["status"] == "completed":
        print(
            "Optimization trace [{}]:".format(trace_result["backend"]),
            "initial =",
            f"{trace_result['initial_energy']:.6e}",
            "final =",
            f"{trace_result['final_energy']:.6e}",
        )
    else:
        print(
            "Optimization trace [{}]:".format(trace_result["backend"]),
            trace_result["status"],
            trace_result["unsupported_reason"],
        )

    print(
        "Unsupported backend-mismatch case [{}]:".format(unsupported_result["backend"]),
        unsupported_result["status"],
        unsupported_result.get("unsupported_reason", ""),
    )

    if args.output_dir is not None:
        for result in fixed_results:
            write_json(args.output_dir / f"{result['case_name']}.json", result)
        write_json(args.output_dir / "optimization_trace_4q.json", trace_result)
        write_json(
            args.output_dir / "unsupported_state_vector_density_noise.json",
            unsupported_result,
        )


if __name__ == "__main__":
    main()
