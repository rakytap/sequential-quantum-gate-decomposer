from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

from benchmarks.density_matrix.partitioned_runtime.common import (
    PHASE3_RUNTIME_DENSITY_TOL,
    PHASE3_RUNTIME_ENERGY_TOL,
    build_density_comparison_metrics,
    density_energy,
    execute_partitioned_with_reference,
)
from benchmarks.density_matrix.planner_calibration.signals import (
    PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME,
    apply_density_scores_and_rankings,
    build_density_signal_record,
)
from benchmarks.density_matrix.planner_calibration.case_selection import (
    PLANNER_CALIBRATION_CASE_KIND_CONTINUITY,
    PLANNER_CALIBRATION_CASE_KIND_MICROCASE,
    iter_planner_calibration_candidate_cases,
)
from squander.partitioning.noisy_runtime import PHASE3_RUNTIME_PATH_BASELINE

PLANNER_CALIBRATION_CALIBRATION_RECORD_SCHEMA_VERSION = "phase3_planner_calibration_record_v1"
PLANNER_CALIBRATION_REFERENCE_BACKEND = "qiskit_aer_density_matrix"
PLANNER_CALIBRATION_EXTERNAL_REFERENCE_CONTINUITY_QUBITS = (4,)


def _flatten_members(descriptor_set):
    return tuple(
        member for partition in descriptor_set.partitions for member in partition.members
    )


def _member_parameter_value(parameters: np.ndarray, member) -> float:
    return float(parameters[member.param_start])


def _apply_member_to_qiskit(qc: QuantumCircuit, member, parameters: np.ndarray) -> None:
    if member.kind == "gate":
        if member.name == "U3":
            theta, phi, lam = parameters[member.param_start : member.param_start + 3]
            qiskit_theta = np.fmod(2.0 * float(theta), 4.0 * np.pi)
            qiskit_phi = np.fmod(float(phi), 2.0 * np.pi)
            qiskit_lambda = np.fmod(float(lam), 2.0 * np.pi)
            qc.u(qiskit_theta, qiskit_phi, qiskit_lambda, member.target_qbit)
            return
        if member.name == "CNOT":
            qc.cx(member.control_qbit, member.target_qbit)
            return
        raise ValueError(
            "Unsupported Qiskit gate '{}' for planner calibration reference".format(member.name)
        )

    if member.kind == "noise":
        value = (
            float(member.fixed_value)
            if member.fixed_value is not None
            else _member_parameter_value(parameters, member)
        )
        if member.name == "local_depolarizing":
            qc.append(depolarizing_error(value, 1), [member.target_qbit])
            return
        if member.name == "amplitude_damping":
            qc.append(amplitude_damping_error(value), [member.target_qbit])
            return
        if member.name == "phase_damping":
            qc.append(phase_damping_error(value), [member.target_qbit])
            return
        raise ValueError(
            "Unsupported Qiskit noise '{}' for planner calibration reference".format(member.name)
        )

    raise ValueError(
        "Unsupported descriptor member kind '{}' for planner calibration reference".format(
            member.kind
        )
    )


def execute_qiskit_density_reference(
    descriptor_set,
    parameters: np.ndarray,
) -> np.ndarray:
    qc = QuantumCircuit(descriptor_set.qbit_num)
    for member in _flatten_members(descriptor_set):
        _apply_member_to_qiskit(qc, member, parameters)
    qc.save_density_matrix()
    simulator = AerSimulator(method="density_matrix")
    result = simulator.run(qc, shots=1).result()
    return np.asarray(result.data()["density_matrix"])


def _requires_external_reference(metadata: dict) -> bool:
    return metadata["case_kind"] == PLANNER_CALIBRATION_CASE_KIND_MICROCASE or (
        metadata["case_kind"] == PLANNER_CALIBRATION_CASE_KIND_CONTINUITY
        and metadata["qbit_num"] in PLANNER_CALIBRATION_EXTERNAL_REFERENCE_CONTINUITY_QUBITS
    )


def build_planner_calibration_calibration_record(
    metadata: dict,
    descriptor_set,
    parameters: np.ndarray,
    *,
    hamiltonian=None,
) -> dict:
    runtime_result, reference_density, density_metrics = execute_partitioned_with_reference(
        descriptor_set, parameters, allow_fusion=False
    )
    record = build_density_signal_record(
        metadata,
        descriptor_set,
        runtime_result,
        density_metrics,
    )
    record["record_schema_version"] = PLANNER_CALIBRATION_CALIBRATION_RECORD_SCHEMA_VERSION
    record["density_aware_objective_name"] = PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME
    record["trace_deviation"] = runtime_result.trace_deviation
    record["rho_is_valid"] = runtime_result.rho_is_valid
    record["rho_is_valid_tol"] = 1e-10
    record["runtime_path"] = PHASE3_RUNTIME_PATH_BASELINE

    internal_density_pass = (
        record["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and record["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.trace_deviation <= PHASE3_RUNTIME_DENSITY_TOL
        and runtime_result.rho_is_valid
    )

    continuity_energy_error = None
    continuity_energy_pass = True
    if hamiltonian is not None:
        runtime_energy_real, _ = density_energy(
            hamiltonian, runtime_result.density_matrix_numpy()
        )
        reference_energy_real, _ = density_energy(
            hamiltonian, np.asarray(reference_density.to_numpy())
        )
        continuity_energy_error = float(abs(runtime_energy_real - reference_energy_real))
        continuity_energy_pass = continuity_energy_error <= PHASE3_RUNTIME_ENERGY_TOL

    record["continuity_energy_error"] = continuity_energy_error
    record["continuity_energy_pass"] = continuity_energy_pass
    record["internal_correctness_pass"] = internal_density_pass and continuity_energy_pass

    external_reference_required = _requires_external_reference(metadata)
    record["external_reference_required"] = external_reference_required
    record["reference_backend"] = (
        PLANNER_CALIBRATION_REFERENCE_BACKEND if external_reference_required else None
    )
    if external_reference_required:
        aer_density = execute_qiskit_density_reference(descriptor_set, parameters)
        external_metrics = build_density_comparison_metrics(
            runtime_result.density_matrix, aer_density
        )
        external_energy_error = None
        if hamiltonian is not None:
            runtime_energy_real, _ = density_energy(
                hamiltonian, runtime_result.density_matrix_numpy()
            )
            aer_energy_real, _ = density_energy(hamiltonian, aer_density)
            external_energy_error = float(abs(runtime_energy_real - aer_energy_real))
        external_reference_pass = (
            external_metrics["frobenius_norm_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and external_metrics["max_abs_diff"] <= PHASE3_RUNTIME_DENSITY_TOL
            and (
                external_energy_error is None
                or external_energy_error <= PHASE3_RUNTIME_ENERGY_TOL
            )
        )
        record["external_frobenius_norm_diff"] = external_metrics["frobenius_norm_diff"]
        record["external_max_abs_diff"] = external_metrics["max_abs_diff"]
        record["external_energy_error"] = external_energy_error
        record["external_reference_pass"] = external_reference_pass
    else:
        record["external_frobenius_norm_diff"] = None
        record["external_max_abs_diff"] = None
        record["external_energy_error"] = None
        record["external_reference_pass"] = True

    record["counted_calibration_case"] = (
        record["internal_correctness_pass"] and record["external_reference_pass"]
    )
    return record


@lru_cache(maxsize=1)
def _build_planner_calibration_calibration_records_cached() -> tuple[dict, ...]:
    records = [
        build_planner_calibration_calibration_record(
            metadata,
            descriptor_set,
            parameters,
            hamiltonian=hamiltonian,
        )
        for metadata, descriptor_set, parameters, hamiltonian in iter_planner_calibration_candidate_cases()
    ]
    return tuple(apply_density_scores_and_rankings(records))


def build_planner_calibration_calibration_records() -> list[dict]:
    return deepcopy(list(_build_planner_calibration_calibration_records_cached()))
