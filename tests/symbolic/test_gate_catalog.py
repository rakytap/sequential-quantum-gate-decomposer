import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates


def equal(left, right):
    assert left.shape == right.shape
    assert all(sp.simplify(sp.trigsimp(sp.expand_complex(x), method="fu")) == 0 for x in left-right)


FIXED_GATES = [d for d in gates.GATE_DEFINITIONS.values() if d.num_qubits is not None]


@pytest.mark.parametrize("definition", FIXED_GATES, ids=lambda d: d.name)
def test_entire_fixed_catalog_is_unitary_and_uses_exact_constants(definition):
    arguments = tuple(sp.Rational(i+1, 7)*sp.pi for i in range(len(definition.parameters)))
    matrix = gates.make_gate(definition.name, *arguments)
    assert isinstance(matrix, sp.ImmutableMatrix)
    assert matrix.shape == (1 << definition.num_qubits,)*2
    assert not any(entry.atoms(sp.Float) for entry in matrix)
    numeric = np.asarray(matrix.evalf(), dtype=complex)
    assert np.allclose(numeric.conj().T @ numeric, np.eye(len(numeric)), atol=2e-12, rtol=0)


QISKIT_STANDARD_CLASSES = [name for name in gates._QISKIT_NAMES if name not in {
    "GlobalPhaseGate", "MCXGate", "MCPhaseGate", "MCMTGate", "MSGate", "RVGate",
    "PauliGate", "PauliProductRotationGate", "DiagonalGate", "PermutationGate",
    "UnitaryGate", "UCGate", "UCRXGate", "UCRYGate", "UCRZGate", "QFTGate",
}]


@pytest.mark.parametrize("class_name", QISKIT_STANDARD_CLASSES)
def test_qiskit_standard_matrices_include_exact_global_phase(class_name):
    library = pytest.importorskip("qiskit.circuit.library")
    cls = getattr(library, class_name)
    count = len(gates.gate_definition(class_name).parameters)
    parameters = tuple(sp.Rational(i+1, 7)*sp.pi for i in range(count))
    expected = cls(*map(float, parameters)).to_matrix()
    actual = np.asarray(gates.make_gate(class_name, *parameters).evalf(), dtype=complex)
    assert np.allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", [
    "Rx", "Ry", "Rz", "Rxx", "Ryy", "Rzz", "Rzx", "CP", "CRX", "CRY", "CRZ",
    "fSim", "PhasedFSim", "Canonical", "Barenco", "CZS", "Givens", "xx_plus_yy", "xx_minus_yy",
])
def test_parameterized_unitarity_is_a_symbolic_identity(name):
    parameters = sp.symbols("p:"+str(len(gates.gate_definition(name).parameters)), real=True)
    matrix = gates.make_gate(name, *parameters)
    equal(matrix.H*matrix, sp.eye(matrix.rows))


def test_controlled_global_phase_is_not_discarded():
    theta = sp.Symbol("theta", real=True)
    controlled = gates.controlled_gate(sp.exp(sp.I*theta)*sp.eye(2))
    equal(controlled, sp.diag(1, sp.exp(sp.I*theta), 1, sp.exp(sp.I*theta)))
    equal(gates.controlled_gate(gates.make_gate("X"), ctrl_state=0),
          sp.Matrix([[0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]]))


def test_named_gate_limits_and_control_constructions():
    equal(gates.make_gate("XPow", sp.Rational(1,2)), gates.make_gate("Sx"))
    equal(gates.make_gate("SWAPPow", sp.Rational(1,2)), gates.make_gate("SSWAP"))
    equal(gates.make_gate("ISWAPPow", sp.Rational(1,2)), gates.make_gate("SiSWAP"))
    equal(gates.make_gate("fSim", sp.pi/2, sp.pi/6), gates.make_gate("SYC"))
    equal(gates.make_gate("Canonical", sp.pi/4, sp.pi/8, 0), gates.make_gate("B"))
    equal(gates.make_gate("CCZS", sp.pi/2, 0, 0), gates.make_gate("fFredkin"))
    equal(gates.make_gate("Controlled", gates.make_gate("SWAP")), gates.make_gate("CSWAP"))
    equal(gates.make_gate("C3SX"), gates.make_gate("Controlled", gates.make_gate("Sx"), 3))
    equal(gates.make_gate("MCX", 2), gates.make_gate("CCX"))


def test_legacy_clifford_sx_name_is_not_reassigned_to_qiskit_sqrt_x():
    equal(gates.make_gate("SXGate"), gates.make_gate("Sx"))
    assert gates.make_gate("SX") != gates.make_gate("Sx")


@pytest.mark.parametrize("name", sorted(gates.clifford_decomps))
def test_original_clifford_decompositions_keep_their_declared_phase(name):
    actual = sp.eye(2)
    for operation in gates.clifford_decomps[name]:
        actual = gates.make_gate(operation[0], *operation[1:-1])*actual
    equal(actual, gates.make_gate(name))


def test_complete_wikipedia_named_gate_inventory():
    names = (
        "I GP X Y Z S Sx H CX ACX CZ CCX SWAP iSWAP P T CP CS Rx Ry Rz Rxx Ryy Rzz "
        "Rxy SSWAP SiSWAP SWAPPow CSWAP U3 Barenco B CSX Canonical DB ECR fSim Givens "
        "Magic SYC CZS Deutsch Margolus Peres Toffoli fFredkin"
    ).split()
    assert all(gates.gate_definition(name) for name in names)


def test_generic_families_agree_with_qiskit():
    library = pytest.importorskip("qiskit.circuit.library")
    t = sp.pi/7
    cases = [
        (gates.make_gate("GP", t), library.GlobalPhaseGate(float(t))),
        (gates.make_gate("MCPhase", t, 2), library.MCPhaseGate(float(t), 2)),
        (gates.make_gate("MCX", 3, 2), library.MCXGate(3, ctrl_state=2)),
        (gates.make_gate("MS", 3, t), library.MSGate(3, float(t))),
        (gates.make_gate("RV", t, 2*t, 3*t), library.RVGate(float(t), float(2*t), float(3*t))),
        (gates.make_gate("Pauli", "XYZ"), library.PauliGate("XYZ")),
        (gates.make_gate("Permutation", (2,0,1)), library.PermutationGate([2,0,1])),
        (gates.make_gate("LinearFunction", ((1,1),(0,1))), library.LinearFunction([[1,1],[0,1]])),
        (gates.make_gate("QFT", 2), library.QFTGate(2)),
        (gates.make_gate("UCRY", (t,2*t)), library.UCRYGate([float(t),float(2*t)])),
    ]
    for actual, official in cases:
        from qiskit.quantum_info import Operator
        assert np.allclose(np.asarray(actual.evalf(), dtype=complex), Operator(official).data, atol=1e-12, rtol=0), official


def test_compile_accepts_iterators_and_rejects_invalid_wiring():
    operations = [("H", (0,)), ("CX", (0,1))]
    equal(gates.compile_gates(2, iter(operations)), gates.compile_gates(2, operations))
    with pytest.raises(ValueError, match="distinct"):
        gates.compile_gates(2, [("CX", (0,0))])
    with pytest.raises(ValueError, match="width"):
        gates.compile_gates(2, [("CX", (0,))])


@pytest.mark.parametrize("name,parameters", [
    ("MCX", (-1,)), ("MCX", (2,4)), ("MCX", (True,)),
    ("QFT", (sp.Rational(1,2),)), ("Permutation", ((1,1),)),
    ("LinearFunction", (((1,1),(1,1)),)), ("Diagonal", ((1,2),)),
    ("Unitary", (sp.zeros(2),)),
])
def test_invalid_gate_families_raise(name, parameters):
    with pytest.raises(ValueError):
        gates.make_gate(name, *parameters)


def test_legacy_module_import_runs_no_search_or_diagnostics():
    pytest.importorskip("deap")
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'squander/decomposition'); import sympy_param"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
