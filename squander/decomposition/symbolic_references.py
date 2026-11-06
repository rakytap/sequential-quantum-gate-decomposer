"""Reproducible compiler upper bounds and comparisons for symbolic synthesis.

SDK output is verified at the recorded binding, including global phase. It is
never an exact symbolic identity or an optimality certificate. Unsupported
SDK bases are reported explicitly; a different entangler is never substituted.
"""

from pathlib import Path
import json
import math

import numpy as np
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_bases import BASIS_SETS, get_basis
else:
    import symbolic_gates as gates
    from symbolic_bases import BASIS_SETS, get_basis


QISKIT_BASES = {
    "cx_u": ("cx", "u"),
    "cx_rz_ry": ("cx", "rz", "ry"),
    "ibm_cx": ("cx", "rz", "sx", "x"),
    "ibm_ecr": ("ecr", "rz", "sx", "x"),
    "ibm_cz": ("cz", "rz", "sx", "x"),
    "cz_u": ("cz", "u"),
    "cz_rz_rx": ("cz", "rz", "rx"),
    "iswap_u": ("iswap", "u"),
    "xx_rz_ry": ("rxx", "rz", "ry"),
    "zz_u": ("rzz", "u"),
    "zx_u": ("rzx", "u"),
}


def reference_ledger():
    return json.loads(Path(__file__).with_name("symbolic_reference_ledger.json").read_text())


def catalog_cases():
    """One explicit binding per fixed definition, plus variable-width fixtures.

    The bindings are examples, not generic-angle claims. Trailing defaults
    (notably Cirq global_shift) are left at their declared values.
    """
    for definition in gates.GATE_DEFINITIONS.values():
        if definition.num_qubits is None:
            continue
        count = len(definition.parameters)-len(definition.defaults)
        half_turns = definition.source in {"ionq", "quantinuum"} or "Pow" in definition.name
        scale = 1 if half_turns else sp.pi
        parameters = tuple(scale*sp.Rational(i+1, 7) for i in range(count))
        yield definition.name, parameters
    t = sp.pi/7
    fixtures = [
        ("GP", (t,)), ("Identity", (3,)), ("Controlled", (gates.make_gate("Sx"), 2)),
        ("MCX", (2,)), ("MCPhase", (t, 2)), ("MCMT", (gates.make_gate("H"), 1, 2)),
        ("QFT", (3,)), ("QFTdg", (3,)), ("WalshHadamard", (3,)),
        ("Pauli", ("XYZ",)), ("PauliRotation", ("XYZ", t)),
        ("Diagonal", ((1, sp.I, -1, -sp.I),)),
        ("Unitary", (gates.make_gate("B"),)), ("Permutation", ((2, 0, 1),)),
        ("LinearFunction", (((1, 1), (0, 1)),)),
        ("UC", ((gates.make_gate("H"), gates.make_gate("Sx")),)),
        ("UCRX", ((t, 2*t),)), ("UCRY", ((t, 2*t),)), ("UCRZ", ((t, 2*t),)),
        ("MS", (3, t)), ("RV", (t, 2*t, 3*t)),
    ]
    yield from (case for case in fixtures if gates.gate_definition(case[0]).num_qubits is None)


def _numeric(matrix):
    result = np.asarray(matrix.evalf(17), dtype=complex)
    if not np.isfinite(result).all():
        raise ValueError("Target contains nonfinite values")
    return result


def _sdk_gate(name, parameters, target):
    """Prefer SDK structure only when its actual matrix matches our convention."""
    from qiskit.circuit import library
    from qiskit.quantum_info import Operator
    from qiskit.exceptions import QiskitError
    canonical = gates.gate_definition(name).name
    for class_name, catalog_name in gates._QISKIT_NAMES.items():
        if catalog_name != canonical:
            continue
        try:
            operation = getattr(library, class_name)(*map(float, parameters))
            if np.allclose(Operator(operation).data, target, atol=1e-12, rtol=0):
                return operation, class_name
        except (TypeError, ValueError, AttributeError, QiskitError):
            pass
    return library.UnitaryGate(target), "UnitaryGate"


def _reference_circuit(name, parameters, width):
    from qiskit import QuantumCircuit
    try:
        sequence = gates.decomposition_sequence(name, *parameters)
    except (ValueError, TypeError):
        return None
    circuit = QuantumCircuit(width)
    for operation in sequence:
        if operation[0] == "GP":
            circuit.global_phase += float(operation[1])
            continue
        matrix = _numeric(gates.make_gate(operation[0], *operation[1:-1]))
        sdk_gate, _ = _sdk_gate(operation[0], operation[1:-1], matrix)
        circuit.append(sdk_gate, list(operation[-1]))
    return circuit


def _transpile(circuit, basis, seed):
    from qiskit import QuantumCircuit, transpile
    allowed = QISKIT_BASES[basis]
    # Qiskit's equivalence translator does not always find U -> ZYZ when Ry/Rz
    # are the only locals. Synthesize in U first, then expand that exact Euler
    # identity explicitly, including its scalar phase.
    zyz = basis in {"cx_rz_ry", "xx_rz_ry"}
    compiler_basis = (allowed[0], "u") if zyz else allowed
    result = transpile(circuit, basis_gates=list(compiler_basis), optimization_level=3,
                       approximation_degree=1.0, seed_transpiler=seed)
    if not zyz:
        return result
    expanded = QuantumCircuit(result.num_qubits, global_phase=result.global_phase)
    for instruction in result.data:
        wires = [result.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "u":
            theta, phi, lam = instruction.operation.params
            expanded.rz(lam, wires[0])
            expanded.ry(theta, wires[0])
            expanded.rz(phi, wires[0])
            expanded.global_phase += (phi+lam)/2
        else:
            expanded.append(instruction.operation, wires)
    return expanded


def qiskit_baseline(name, parameters=(), basis="cx_u", *, seed=219, tolerance=2e-10):
    """Best verified upper bound among structured, matrix, and stored inputs.

    Record enough operations and phase to replay the selected circuit. SDK
    parameter slots are a tie breaker only; binary floats are deliberately not
    scored with the symbolic denominator-based cost.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    get_basis(basis)
    row = {"gate": name, "parameters": [str(p) for p in parameters], "basis": basis,
           "seed": seed, "ancillas": 0, "topology": "all_to_all",
           "symbolic_proof": False, "parameter_sensitive_score": None}
    if basis not in QISKIT_BASES:
        return dict(row, status="unsupported_sdk_basis")
    target = _numeric(gates.make_gate(name, *parameters))
    width = target.shape[0].bit_length()-1
    row["num_qubits"] = width
    if width == 0:
        return dict(row, status="verified_numeric", entanglers=0, parameter_slots=0,
                    global_phase=float(np.angle(target[0, 0])), operations=[], max_error=0.0)
    structured = QuantumCircuit(width)
    operation, source = _sdk_gate(name, parameters, target)
    structured.append(operation, range(width))
    matrix_input = QuantumCircuit(width)
    matrix_input.append(UnitaryGate(target), range(width))
    inputs = [(source, structured), ("matrix_synthesis", matrix_input)]
    stored = _reference_circuit(name, parameters, width)
    if stored is not None:
        inputs.append(("stored_exact_reference", stored))
    verified, failures = [], []
    allowed = QISKIT_BASES[basis]
    for source, circuit in inputs:
        try:
            result = _transpile(circuit, basis, seed)
            if result.num_qubits != width:
                raise ValueError("Compiler introduced ancillary qubits")
            if any(instruction.operation.name not in allowed for instruction in result.data):
                raise ValueError("Compiler returned an operation outside the chosen basis")
            actual = Operator(result).data
            error = float(np.max(np.abs(actual-target)))
            phase_correction = 0.0
            if math.isfinite(error) and error > tolerance:
                # Some SDK synthesis paths return a unitary with a missing
                # scalar phase. Repair the delivered circuit explicitly, then
                # verify it again; relative-phase errors still fail.
                phase_correction = float(np.angle(np.vdot(actual, target)))
                result.global_phase += phase_correction
                actual = Operator(result).data
                error = float(np.max(np.abs(actual-target)))
            if not math.isfinite(error) or error > tolerance:
                raise ValueError(f"Phase-sensitive matrix verification failed: {error}")
            operations = [
                {"gate": instruction.operation.name,
                 "wires": [result.find_bit(q).index for q in instruction.qubits],
                 "parameters": [float(p) for p in instruction.operation.params]}
                for instruction in result.data
            ]
            entanglers = sum(len(op["wires"]) == 2 for op in operations)
            slots = sum(len(op["parameters"]) for op in operations)
            verified.append(dict(row, status="verified_numeric", input=source,
                                 entanglers=entanglers, parameter_slots=slots,
                                 global_phase=float(result.global_phase), operations=operations,
                                 max_error=error, explicit_sdk_phase_correction=phase_correction))
        except Exception as error:
            failures.append({"input": source, "error": str(error)})
    if not verified:
        return dict(row, status="compiler_failed", failures=failures)
    best = min(verified, key=lambda r: (r["entanglers"], r["parameter_slots"], len(r["operations"])))
    best["attempts"] = [{k: r[k] for k in ("input", "entanglers", "max_error")} for r in verified]
    if failures:
        best["failures"] = failures
    return best


def verify_symbolic_candidate(name, parameters, basis, sequence, *, global_phase=0,
                              reference_entanglers=None):
    """Reject a false identity before measuring cost or comparing upper bounds.

    A caller-supplied reference bound must describe this same target binding,
    basis, topology, phase convention, and ancilla model.
    """
    selected = get_basis(basis)
    sequence = tuple(sequence)
    count = selected.entangler_count(sequence)
    target = gates.make_gate(name, *parameters)
    width = target.rows.bit_length()-1
    actual = sp.exp(sp.I*global_phase)*gates.compile_gates(width, sequence)
    residuals = [sp.simplify(sp.trigsimp(sp.expand_complex(x), method="fu")) for x in actual-target]
    if any(x != 0 for x in residuals):
        return {"status": "identity_not_proved", "entanglers": count}
    result = {"status": "verified_symbolic", "entanglers": count,
              "within_reference_count": None if reference_entanglers is None
              else count <= reference_entanglers}
    result["comparison"] = ("no_matched_reference" if reference_entanglers is None else
                            "regresses_reference" if count > reference_entanglers else
                            "matches_reference" if count == reference_entanglers else
                            "improves_reference")
    result["accepted"] = result["within_reference_count"] is not False
    # Use the actual unchanged legacy score, without converting SDK floats to
    # invented exact fractions. The score accepts symbolic angle wrappers.
    if __package__:
        from .sympy_param import get_gatecost
    else:
        from sympy_param import get_gatecost
    class ExactAngle:
        def __init__(self, value):
            self.value = sp.sympify(value)
        def to_sympy(self, unused):
            return self.value
    class ExactCircuit:
        gates = [(op[0], op[-1], tuple(ExactAngle(p) for p in op[1:-1])) for op in sequence]
    if sp.simplify(global_phase) != 0:
        ExactCircuit.gates.append(("GP", tuple(range(width)), (ExactAngle(global_phase),)))
    result["parameter_sensitive_score"] = get_gatecost(ExactCircuit(), ())
    return result
