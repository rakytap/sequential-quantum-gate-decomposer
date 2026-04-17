"""Mandatory micro-validation matrix for the exact observable path.

Validates the required 1- to 3-qubit microcases against Qiskit Aer
density-matrix simulation using the frozen exact-observable contract:
`Re Tr(H*rho)`, density validity, trace preservation, and Hermitian-observable
consistency.

Run with: python benchmarks/density_matrix/validate_squander_vs_qiskit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import qiskit
import qiskit_aer
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import state_fidelity

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.circuits import MANDATORY_MICROCASES_BY_QUBITS  # noqa: E402

PRIMARY_BACKEND = "density_matrix"
REFERENCE_BACKEND = "qiskit_aer_density_matrix"
SUITE_NAME = "mandatory_micro_validation"
ENERGY_ERROR_TOL = 1e-10
VALIDITY_TOL = 1e-10
TRACE_TOL = 1e-10
OBSERVABLE_IMAG_TOL = 1e-10
ARTIFACT_FILENAME = "micro_validation_bundle.json"
ARTIFACT_CORE_FIELDS = (
    "suite_name",
    "status",
    "backend",
    "reference_backend",
    "thresholds",
    "summary",
    "software",
    "cases",
)


def parse_microcase_operation(operation: str):
    """Return machine-readable metadata for one microcase builder operation."""
    if operation.startswith("U3("):
        payload = operation[len("U3(") : -1].split(",")
        return {
            "kind": "gate",
            "name": "U3",
            "target_qbit": int(payload[0]),
            "control_qbit": None,
            "value": None,
            "raw_operation": operation,
        }

    if operation.startswith("CNOT("):
        control, target = operation[len("CNOT(") : -1].split("->")
        return {
            "kind": "gate",
            "name": "CNOT",
            "target_qbit": int(target),
            "control_qbit": int(control),
            "value": None,
            "raw_operation": operation,
        }

    if operation.startswith("LocalDepol("):
        target, value = operation[len("LocalDepol(") : -1].split(",")
        return {
            "kind": "noise",
            "name": "local_depolarizing",
            "target_qbit": int(target),
            "control_qbit": None,
            "value": float(value),
            "raw_operation": operation,
        }

    if operation.startswith("AD("):
        target, value = operation[len("AD(") : -1].split(",")
        return {
            "kind": "noise",
            "name": "amplitude_damping",
            "target_qbit": int(target),
            "control_qbit": None,
            "value": float(value),
            "raw_operation": operation,
        }

    if operation.startswith("PD("):
        target, value = operation[len("PD(") : -1].split(",")
        return {
            "kind": "noise",
            "name": "phase_damping",
            "target_qbit": int(target),
            "control_qbit": None,
            "value": float(value),
            "raw_operation": operation,
        }

    if operation.startswith("Depol("):
        value = operation[len("Depol(") : -1]
        return {
            "kind": "noise",
            "name": "depolarizing",
            "target_qbit": None,
            "control_qbit": None,
            "value": float(value),
            "raw_operation": operation,
        }

    for gate_name in ("H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg", "SX"):
        prefix = f"{gate_name}("
        if operation.startswith(prefix):
            target = operation[len(prefix) : -1]
            return {
                "kind": "gate",
                "name": gate_name,
                "target_qbit": int(target),
                "control_qbit": None,
                "value": None,
                "raw_operation": operation,
            }

    if operation.startswith("CZ("):
        control, target = operation[len("CZ(") : -1].split(",")
        return {
            "kind": "gate",
            "name": "CZ",
            "target_qbit": int(target),
            "control_qbit": int(control),
            "value": None,
            "raw_operation": operation,
        }

    raise ValueError(f"Unsupported microcase operation format: {operation}")


def build_microcase_operation_audit(case, operations):
    """Summarize required gate/noise coverage and mixed-sequence order."""
    operation_metadata = [parse_microcase_operation(operation) for operation in operations]
    gate_operations = [
        operation for operation in operation_metadata if operation["kind"] == "gate"
    ]
    noise_operations = [
        operation for operation in operation_metadata if operation["kind"] == "noise"
    ]
    gate_sequence = [operation["name"] for operation in gate_operations]
    noise_sequence = [operation["name"] for operation in noise_operations]
    noise_targets = [operation["target_qbit"] for operation in noise_operations]
    noise_values = [operation["value"] for operation in noise_operations]

    required_gate_coverage_pass = set(case["required_gate_family"]).issubset(
        set(gate_sequence)
    )
    required_noise_model_coverage_pass = set(case["required_noise_models"]).issubset(
        set(noise_sequence)
    )
    noise_sequence_match_pass = noise_sequence == case["required_noise_models"]
    mixed_sequence_order_pass = (
        noise_sequence_match_pass if case["case_kind"] == "mixed_sequence" else None
    )
    operation_audit_pass = (
        required_gate_coverage_pass
        and required_noise_model_coverage_pass
        and noise_sequence_match_pass
    )

    return {
        "operation_metadata": operation_metadata,
        "operation_count": len(operation_metadata),
        "gate_operation_sequence": gate_sequence,
        "noise_operation_sequence": noise_sequence,
        "noise_operation_targets": noise_targets,
        "noise_operation_values": noise_values,
        "required_gate_coverage_pass": required_gate_coverage_pass,
        "required_noise_model_coverage_pass": required_noise_model_coverage_pass,
        "noise_sequence_match_pass": noise_sequence_match_pass,
        "mixed_sequence_order_pass": mixed_sequence_order_pass,
        "operation_audit_pass": operation_audit_pass,
    }


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Compute trace distance: D(ρ,σ) = (1/2)||ρ - σ||_1."""
    diff = rho1 - rho2
    eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
    return float(0.5 * np.sum(np.sqrt(np.maximum(eigenvalues, 0))))


def density_energy(hamiltonian: np.ndarray, density_matrix: np.ndarray):
    """Return the real and imaginary parts of Tr(H*rho)."""
    energy = np.trace(hamiltonian @ density_matrix)
    return float(np.real(energy)), float(np.imag(energy))


def build_threshold_metadata():
    return {
        "absolute_energy_error": ENERGY_ERROR_TOL,
        "rho_is_valid_tol": VALIDITY_TOL,
        "trace_deviation": TRACE_TOL,
        "observable_imag_abs": OBSERVABLE_IMAG_TOL,
        "required_pass_rate": 1.0,
    }


def build_software_metadata():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "qiskit": getattr(qiskit, "__version__", "unknown"),
        "qiskit_aer": getattr(qiskit_aer, "__version__", "unknown"),
    }


def _microcase_base(case):
    return {
        "case_name": case["case_name"],
        "status": "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "qbit_num": case["qbit_num"],
        "case_kind": case["case_kind"],
        "purpose": case["purpose"],
        "required_gate_family": case["required_gate_family"],
        "required_noise_models": case["required_noise_models"],
        "hamiltonian": case["hamiltonian_metadata"],
    }


def validate_microcase(case, verbose=True):
    """Evaluate one mandatory micro-validation microcase."""
    builder = case["builder_fn"]()
    squander_rho = builder.run_squander()
    qiskit_rho = builder.run_qiskit()
    operations = list(builder.ops)
    operation_audit = build_microcase_operation_audit(case, operations)

    squander_arr = np.asarray(squander_rho.to_numpy())
    qiskit_arr = np.asarray(qiskit_rho)

    fidelity = float(
        state_fidelity(
            QiskitDensityMatrix(squander_arr), QiskitDensityMatrix(qiskit_arr)
        )
    )
    max_diff = float(np.max(np.abs(squander_arr - qiskit_arr)))
    rho_trace = squander_rho.trace()
    trace_deviation = float(abs(rho_trace - 1.0))
    density_valid = bool(squander_rho.is_valid(tol=VALIDITY_TOL))
    squander_purity = float(np.real(np.trace(squander_arr @ squander_arr)))
    qiskit_purity = float(np.real(np.trace(qiskit_arr @ qiskit_arr)))
    state_distance = trace_distance(squander_arr, qiskit_arr)

    squander_energy_real, squander_energy_imag = density_energy(
        case["hamiltonian_matrix"], squander_arr
    )
    reference_energy_real, reference_energy_imag = density_energy(
        case["hamiltonian_matrix"], qiskit_arr
    )
    energy_error = float(abs(squander_energy_real - reference_energy_real))

    energy_pass = energy_error <= ENERGY_ERROR_TOL
    density_valid_pass = density_valid
    trace_pass = trace_deviation <= TRACE_TOL
    observable_pass = abs(squander_energy_imag) <= OBSERVABLE_IMAG_TOL
    state_comparison_status = (
        "pass" if fidelity > 0.99999 else ("warn" if fidelity > 0.999 else "fail")
    )
    case_pass = (
        energy_pass
        and density_valid_pass
        and trace_pass
        and observable_pass
        and operation_audit["operation_audit_pass"]
    )

    result = _microcase_base(case)
    result.update(
        {
            "status": "pass" if case_pass else "fail",
            "parameter_vector": builder.get_parameter_vector().tolist(),
            "operations": operations,
            "state_fidelity": fidelity,
            "trace_distance": state_distance,
            "max_diff": max_diff,
            "state_comparison_status": state_comparison_status,
            "squander_purity": squander_purity,
            "reference_purity": qiskit_purity,
            "squander_trace_real": float(np.real(rho_trace)),
            "squander_trace_imag": float(np.imag(rho_trace)),
            "trace_deviation": trace_deviation,
            "rho_is_valid": density_valid,
            "rho_is_valid_tol": VALIDITY_TOL,
            "squander_energy_real": squander_energy_real,
            "squander_energy_imag": squander_energy_imag,
            "reference_energy_real": reference_energy_real,
            "reference_energy_imag": reference_energy_imag,
            "absolute_energy_error": energy_error,
            "energy_pass": energy_pass,
            "density_valid_pass": density_valid_pass,
            "trace_pass": trace_pass,
            "observable_pass": observable_pass,
            **operation_audit,
        }
    )

    if verbose:
        print(
            "  {case_name:<38} [{backend} vs {reference}] "
            "|ΔE|={energy_error:.3e} |Tr-1|={trace_dev:.3e} "
            "|ImE|={imag:.3e} status={status}".format(
                case_name=result["case_name"],
                backend=result["backend"],
                reference=result["reference_backend"],
                energy_error=result["absolute_energy_error"],
                trace_dev=result["trace_deviation"],
                imag=abs(result["squander_energy_imag"]),
                status=result["status"].upper(),
            )
        )

    return result


def capture_microcase(case, verbose=True):
    """Capture validation output for one mandatory micro-validation microcase."""
    try:
        return validate_microcase(case, verbose=verbose)
    except Exception as exc:
        result = _microcase_base(case)
        result.update(
            {
                "status": "fail",
                "error_message": str(exc),
                "energy_pass": False,
                "density_valid_pass": False,
                "trace_pass": False,
                "observable_pass": False,
            }
        )
        if verbose:
            print(
                "  {case_name:<38} [{backend} vs {reference}] ERROR {message}".format(
                    case_name=result["case_name"],
                    backend=result["backend"],
                    reference=result["reference_backend"],
                    message=result["error_message"],
                )
            )
        return result


def run_validation(verbose=True):
    """Run the mandatory micro-validation micro-validation matrix."""
    if verbose:
        print("=" * 78)
        print(
            "  micro-validation Micro-Validation [{} vs {}]".format(
                PRIMARY_BACKEND, REFERENCE_BACKEND
            )
        )
        print("=" * 78)

    results = []
    for qbit_num in sorted(MANDATORY_MICROCASES_BY_QUBITS.keys()):
        if verbose:
            print(f"\n--- {qbit_num}-QUBIT MANDATORY MICROCASES ---", flush=True)
        for case in MANDATORY_MICROCASES_BY_QUBITS[qbit_num]:
            results.append(capture_microcase(case, verbose=verbose))
    return results


def build_artifact_bundle(results):
    passed = sum(1 for result in results if result["status"] == "pass")
    total = len(results)
    pass_rate = 0.0 if total == 0 else passed / total
    bundle = {
        "suite_name": SUITE_NAME,
        "status": "pass" if pass_rate == 1.0 else "fail",
        "backend": PRIMARY_BACKEND,
        "reference_backend": REFERENCE_BACKEND,
        "thresholds": build_threshold_metadata(),
        "software": build_software_metadata(),
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": total - passed,
            "pass_rate": pass_rate,
        },
        "cases": results,
    }
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle):
    missing_fields = [field for field in ARTIFACT_CORE_FIELDS if field not in bundle]
    if missing_fields:
        raise ValueError(
            "Artifact bundle is missing required fields: {}".format(
                ", ".join(missing_fields)
            )
        )


def write_artifact_bundle(output_path: Path, bundle):
    validate_artifact_bundle(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


def print_summary(results):
    """Print micro-validation summary."""
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)

    for result in results:
        details = []
        if not result.get("energy_pass", False):
            details.append("energy")
        if not result.get("density_valid_pass", False):
            details.append("density")
        if not result.get("trace_pass", False):
            details.append("trace")
        if not result.get("observable_pass", False):
            details.append("imag")
        if "error_message" in result:
            details.append("error")

        print(
            "  {case_name:<38} [{backend} vs {reference}] "
            "|ΔE|={energy_error:.3e} status={status}{detail_suffix}".format(
                case_name=result["case_name"],
                backend=result["backend"],
                reference=result["reference_backend"],
                energy_error=result.get("absolute_energy_error", float("nan")),
                status=result["status"].upper(),
                detail_suffix="" if not details else " (" + ",".join(details) + ")",
            )
        )

    print("\n" + "-" * 78)
    total = len(results)
    passed = sum(1 for result in results if result["status"] == "pass")
    print(f"  Total: {passed}/{total} mandatory microcases passed")

    if passed == total:
        print("\n  ALL TESTS PASSED - micro-validation mandatory micro-validation gate is closed.")
    else:
        print("\n  Some mandatory microcases failed - micro-validation is not yet closed.")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for the micro-validation JSON artifact bundle.",
    )
    args = parser.parse_args()

    results = run_validation()
    bundle = build_artifact_bundle(results)
    print_summary(results)

    if args.output_dir is not None:
        write_artifact_bundle(args.output_dir / ARTIFACT_FILENAME, bundle)

    if bundle["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
