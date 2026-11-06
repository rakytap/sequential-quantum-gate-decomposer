import json

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_minimality import cnot_cost_certificate
from symbolic_references import qiskit_baseline, reference_ledger, verify_symbolic_candidate


@pytest.mark.parametrize("name,count", [
    ("CX", 1), ("CZ", 1), ("ECR", 1), ("CH", 1), ("DCX", 2),
    ("iSWAP", 2), ("SiSWAP", 2), ("SWAP", 3), ("SSWAP", 3),
    ("SYC", 3), ("B", 2), ("XX", 0), ("ZZ90", 1),
])
def test_exact_cnot_minima(name, count):
    assert cnot_cost_certificate(gates.make_gate(name)).minimum == count
    # A nontrivial scalar phase must not affect interaction cost.
    assert cnot_cost_certificate(sp.exp(sp.I*sp.pi/7)*gates.make_gate(name)).minimum == count


def test_special_angles_do_not_inherit_a_generic_minimum():
    theta = sp.Symbol("theta", real=True)
    matrix = gates.make_gate("Rzz", theta)
    certificate = cnot_cost_certificate(matrix)
    assert certificate.minimum is None
    assert certificate.upper_bound == 2
    for angle, count in [(0, 0), (sp.pi/2, 1), (sp.pi, 0), (sp.pi/4, 2)]:
        assert cnot_cost_certificate(matrix.subs(theta, angle)).minimum == count


def test_minimality_rejects_numeric_or_unproved_inputs():
    for matrix in [sp.eye(2), sp.zeros(4), sp.eye(4)*sp.Float(1)]:
        with pytest.raises(ValueError):
            cnot_cost_certificate(matrix)


@pytest.mark.parametrize("name", [name for name in gates.decomposition_sequences
                                  if name not in gates.clifford_decomps and name != "GP"])
def test_all_stored_reference_sequences_match_target_phase(name):
    count = len(gates.gate_descs[name][1])
    parameters = tuple(sp.Rational(i+1, 7)*sp.pi for i in range(count))
    # CZPowGate's historical stored reference takes its original unary exponent.
    if name == "CZPowGate":
        parameters = parameters[:1]
    target = gates.make_gate(name, *parameters)
    actual = gates.compile_gates(target.rows.bit_length()-1,
                                 gates.decomposition_sequence(name, *parameters))
    assert np.max(np.abs(np.asarray((actual-target).evalf(25), dtype=complex))) < 1e-20


def test_xx_minus_yy_reference_is_a_free_parameter_identity():
    theta, beta = sp.symbols("theta beta", real=True)
    difference = gates.make_gate_decomp("xx_minus_yy", theta, beta)-gates.make_gate("xx_minus_yy", theta, beta)
    assert all(sp.simplify(sp.trigsimp(sp.expand_complex(x), method="fu")) == 0 for x in difference)


@pytest.mark.parametrize("width", [0, 1, 2, 3])
def test_reference_global_phase_does_not_scale_with_register_width(width):
    theta = sp.Symbol("theta", real=True)
    assert gates.make_gate_decomp("GP", theta, width) == gates.make_gate("GP", theta, width)


def test_candidate_verification_rejects_a_missing_global_phase():
    pytest.importorskip("deap")
    bad = verify_symbolic_candidate("Rz", (sp.pi/2,), "cx_rz_ry", [("Rz", sp.pi/2, (0,))],
                                    global_phase=sp.pi/4)
    assert bad["status"] == "identity_not_proved"
    good = verify_symbolic_candidate("CZ", (), "cx_u", [
        ("U3", sp.pi/2, 0, sp.pi, (1,)), ("CX", (0, 1)),
        ("U3", sp.pi/2, 0, sp.pi, (1,)),
    ], reference_entanglers=1)
    assert good["status"] == "verified_symbolic"
    assert good["within_reference_count"]
    assert good["parameter_sensitive_score"] > 0


def test_fredkin_reference_stays_at_seven_cx_and_is_not_labeled_proved_minimal():
    row = next(r for r in reference_ledger()["records"] if r["id"] == "fredkin_cx")
    assert row["upper_bound"] == 7 and row["lower_bound"] is None
    assert row["claim"] == "published_upper_bound"
    assert sum(op[0] == "CNOT" for op in gates.decomposition_sequence("CSWAP")) == 7


def test_margolus_optimal_reference_is_exact_and_peres_cancels_a_cnot():
    for name, count in [("Margolus", 3), ("RCCX", 3), ("Peres", 5), ("Toffoli", 6), ("CCY", 6)]:
        sequence = gates.decomposition_sequence(name)
        actual = gates.compile_gates(3, sequence)
        assert all(sp.simplify(x) == 0 for x in actual-gates.make_gate(name))
        assert sum(op[0] == "CNOT" for op in sequence) == count


def test_comparison_inventory_has_a_fixture_for_every_catalog_definition():
    from symbolic_references import catalog_cases
    names = [name for name, _ in catalog_cases()]
    assert set(gates.GATE_DEFINITIONS) == set(names)
    assert len(names) == len(set(names))


def test_verified_identity_is_rejected_when_its_count_regresses_a_reference():
    pytest.importorskip("deap")
    sequence = [("U3", sp.pi/2, 0, sp.pi, (1,)), ("CX", (0,1)),
                ("U3", sp.pi/2, 0, sp.pi, (1,)), ("CX", (0,1)), ("CX", (0,1))]
    result = verify_symbolic_candidate("CZ", (), "cx_u", sequence, reference_entanglers=1)
    assert result["status"] == "verified_symbolic"
    assert result["accepted"] is False and result["comparison"] == "regresses_reference"


def test_separately_tracked_phase_is_included_in_the_original_score():
    pytest.importorskip("deap")
    result = verify_symbolic_candidate("GP", (sp.pi/4,), "cx_u", (), global_phase=sp.pi/4)
    assert result["status"] == "verified_symbolic"
    assert result["parameter_sensitive_score"] > 1


def test_sdk_phase_correction_cannot_hide_a_relative_phase_or_wrong_operator(monkeypatch):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    import symbolic_references
    wrong = QuantumCircuit(2)
    wrong.cx(0, 1)
    monkeypatch.setattr(symbolic_references, "_transpile", lambda *args: wrong.copy())
    assert qiskit_baseline("SWAP")["status"] == "compiler_failed"


@pytest.mark.parametrize("gate,params,basis,bound", [
    ("SWAP", (), "cx_u", 3), ("CCX", (), "cx_u", 6),
    ("CSWAP", (), "cx_u", 7), ("ECR", (), "ibm_ecr", 1),
    ("SYC", (), "cz_u", 3), ("Rzz", (sp.pi/7,), "zz_u", 1),
    ("GP", (sp.pi/7,), "cx_u", 0),
    ("Margolus", (), "cx_u", 3), ("Margolus", (), "ibm_cx", 3),
    ("CCX", (), "cx_rz_ry", 6), ("CCX", (), "xx_rz_ry", 6),
    ("CCXPow", (sp.Rational(1,7),), "ibm_cx", 20),
    ("WalshHadamard", (3,), "cx_u", 0), ("QFT", (3,), "cx_u", 9),
    ("QFTdg", (3,), "cx_u", 9), ("Permutation", ((2,0,1),), "cx_u", 6),
    ("MS", (3, sp.pi/7), "cx_u", 6),
])
def test_sdk_comparisons_replay_with_phase_and_respect_known_upper_bounds(gate, params, basis, bound):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    result = qiskit_baseline(gate, params, basis)
    assert result["status"] == "verified_numeric", result
    assert result["entanglers"] <= bound
    assert result["symbolic_proof"] is False
    assert result["parameter_sensitive_score"] is None
    width = gates.make_gate(gate, *params).rows.bit_length()-1
    circuit = QuantumCircuit(width, global_phase=result["global_phase"])
    for op in result["operations"]:
        getattr(circuit, op["gate"])(*op["parameters"], *op["wires"])
    assert np.allclose(Operator(circuit).data,
                       np.asarray(gates.make_gate(gate, *params).evalf(), dtype=complex), atol=2e-10, rtol=0)
    json.dumps(result, allow_nan=False)


def test_unsupported_basis_is_not_silently_compared_in_cnot_basis():
    pytest.importorskip("qiskit")
    assert qiskit_baseline("SWAP", (), "google_syc")["status"] == "unsupported_sdk_basis"
