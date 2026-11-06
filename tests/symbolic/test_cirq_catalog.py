import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates

cirq = pytest.importorskip("cirq")


def little_endian(matrix):
    n = len(matrix).bit_length()-1
    permutation = [int(format(i, f"0{n}b")[::-1], 2) for i in range(len(matrix))]
    return matrix[np.ix_(permutation, permutation)]


@pytest.mark.parametrize("name,cls", [
    ("XPow",cirq.XPowGate), ("YPow",cirq.YPowGate), ("ZPow",cirq.ZPowGate),
    ("HPow",cirq.HPowGate), ("CXPow",cirq.CXPowGate), ("CZPowGate",cirq.CZPowGate),
    ("SWAPPow",cirq.SwapPowGate), ("ISWAPPow",cirq.ISwapPowGate),
    ("XXPow",cirq.XXPowGate), ("YYPow",cirq.YYPowGate), ("ZZPow",cirq.ZZPowGate),
    ("CCXPow",cirq.CCXPowGate), ("CCZPow",cirq.CCZPowGate),
])
def test_cirq_power_gates_keep_global_shift_and_wire_order(name, cls):
    exponent, shift = sp.Rational(2,7), sp.Rational(1,5)
    actual = np.asarray(gates.make_gate(name, exponent, shift).evalf(), dtype=complex)
    expected = cirq.unitary(cls(exponent=float(exponent), global_shift=float(shift)))
    assert np.allclose(actual, little_endian(expected), atol=1e-12, rtol=0)


def test_google_parameterized_gate_matrices():
    values = [sp.pi*sp.Rational(i+1,13) for i in range(5)]
    a,b,c,d,e = values
    cases = [
        ("fSim", (a,b), cirq.FSimGate(float(a),float(b))),
        ("PhasedFSim", tuple(values), cirq.PhasedFSimGate(theta=float(a),zeta=float(b),chi=float(c),gamma=float(d),phi=float(e))),
        ("PhasedXPow", (a,b,c), cirq.PhasedXPowGate(exponent=float(a),phase_exponent=float(b),global_shift=float(c))),
        ("PhasedXZ", (a,b,c), cirq.PhasedXZGate(x_exponent=float(a),z_exponent=float(b),axis_phase_exponent=float(c))),
        ("PhasedISWAPPow", (a,b,c), cirq.PhasedISwapPowGate(phase_exponent=float(a),exponent=float(b),global_shift=float(c))),
    ]
    for name, params, gate in cases:
        actual = np.asarray(gates.make_gate(name,*params).evalf(), dtype=complex)
        assert np.allclose(actual, little_endian(cirq.unitary(gate)), atol=1e-12, rtol=0), name


def test_ionq_turns_and_asymmetric_wire_phases_against_provider_sdk():
    ionq = pytest.importorskip("cirq_ionq")
    a, b, theta = sp.Rational(1, 7), sp.Rational(2, 9), sp.Rational(1, 8)
    cases = [
        ("GPi", (a,), ionq.GPIGate(phi=float(a))),
        ("GPi2", (b,), ionq.GPI2Gate(phi=float(b))),
        ("IonQMS", (a, b, theta), ionq.MSGate(phi0=float(a), phi1=float(b), theta=float(theta))),
        ("IonQMSMax", (a, b), ionq.MSGate(phi0=float(a), phi1=float(b))),
    ]
    for name, parameters, operation in cases:
        actual = np.asarray(gates.make_gate(name, *parameters).evalf(), dtype=complex)
        assert np.allclose(actual, little_endian(cirq.unitary(operation)), atol=1e-12, rtol=0), name


def test_provider_angle_units_keep_exact_phase():
    a, b = sp.symbols("a b", real=True)
    cases = [
        (gates.make_gate("PhasedX", a, b), gates.make_gate("R", sp.pi*a, sp.pi*b)),
        (gates.make_gate("IonQZZ", a), gates.make_gate("Rzz", 2*sp.pi*a)),
        (gates.make_gate("ZZPhase", a), gates.make_gate("Rzz", sp.pi*a)),
        (gates.make_gate("ZZMax"), gates.make_gate("Rzz", sp.pi/2)),
        (gates.make_gate("IonQMS", 0, 0), gates.make_gate("Rxx", sp.pi/2)),
    ]
    for left, right in cases:
        assert all(sp.simplify(sp.trigsimp(sp.expand_complex(x), method="fu")) == 0 for x in left-right)
