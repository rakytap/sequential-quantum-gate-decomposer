import copy

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_sdk_lift import lift_sdk_candidate, propose_pi_angle


def sycamore_baseline():
    return dict(status="verified_numeric", basis="ibm_cx", input="test_proposal",
        global_phase=float(-7*sp.pi/24), operations=[
            dict(gate="rz", parameters=[float(-sp.pi/12)], wires=[0]),
            dict(gate="rz", parameters=[float(-sp.pi/12)], wires=[1]),
            dict(gate="cx", parameters=[], wires=[0, 1]),
            dict(gate="rz", parameters=[float(-5*sp.pi/12)], wires=[1]),
            dict(gate="cx", parameters=[], wires=[1, 0]),
            dict(gate="cx", parameters=[], wires=[0, 1])])


def test_numerical_proposals_require_an_exact_target_identity():
    baseline = sycamore_baseline()
    result = lift_sdk_candidate(gates.make_gate("SYC"), baseline, "ibm_cx")
    assert result["status"] == "verified_symbolic"
    assert result["cost"]["single_qubit_gates"] == 3
    assert result["virtual_gp"]["factor"] == "exp(-7*I*pi/24)"
    assert not result["local_cost_optimal"]
    wrong = sp.Matrix(gates.make_gate("SYC"))
    wrong[0, 0] += sp.Rational(1, 10**20)
    assert lift_sdk_candidate(wrong, baseline, "ibm_cx")["status"] == "proposal_not_proved"


@pytest.mark.parametrize("change", ["angle", "wire", "phase", "basis", "primitive"])
def test_bad_sdk_proposals_cannot_acquire_a_certificate(change):
    baseline = copy.deepcopy(sycamore_baseline())
    if change == "angle": baseline["operations"][0]["parameters"] = [float(sp.pi/7)]
    if change == "wire": baseline["operations"][1]["wires"] = [0]
    if change == "phase": baseline["global_phase"] = 0
    if change == "basis": baseline["basis"] = "cx_u"
    if change == "primitive": baseline["operations"][0]["gate"] = "u"
    assert lift_sdk_candidate(gates.make_gate("SYC"), baseline, "ibm_cx")["status"] == "proposal_not_proved"


def test_exact_mode_charges_a_physical_scalar_and_preserves_basis():
    baseline = dict(status="verified_numeric", basis="cx_u", operations=[],
                    global_phase=float(sp.pi/4))
    target = sp.exp(sp.I*sp.pi/4)*sp.eye(2)
    projective = lift_sdk_candidate(target, baseline, "cx_u")
    exact = lift_sdk_candidate(target, baseline, "cx_u", phase_exact=True)
    assert projective["cost"]["single_qubit_gates"] == 0
    assert exact["status"] == "verified_symbolic" and exact["virtual_gp"] is None
    assert exact["cost"]["single_qubit_gates"] > 0
    assert all(op["gate"] == "U3" for op in exact["operations"])
    assert lift_sdk_candidate(gates.make_gate("SYC"), sycamore_baseline(),
                              "ibm_cx", phase_exact=True)["status"] == "proposal_not_proved"


def test_unsupported_angles_and_free_families_are_declined():
    assert propose_pi_angle(float("inf")) is None
    assert propose_pi_angle(0.123456789) is None
    assert propose_pi_angle(float(sp.pi/16)) == sp.pi/16
    theta = sp.Symbol("theta", real=True)
    assert lift_sdk_candidate(gates.make_gate("Rx", theta),
                              sycamore_baseline(), "ibm_cx")["status"] == "proposal_not_proved"
