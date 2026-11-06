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
