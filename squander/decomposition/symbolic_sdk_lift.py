"""Use numerical SDK angles only as proposals for exact fixed-target circuits.

Rational multiples of pi are recovered within a bounded proposal alphabet.
Every delivered circuit is then proved against the exact target. This is not
symbolic-family synthesis, and an unsuccessful lift proves no obstruction.
"""

from fractions import Fraction
import math

import sympy as sp

if __package__:
    from .symbolic_bases import get_basis
    from .symbolic_contract import validate_operations, verify_circuit, simplify_exact
    from .symbolic_feasibility import feasibility_obstruction
    from .symbolic_synthesis import _physical_scalar, _verify_phase_candidate
else:
    from symbolic_bases import get_basis
    from symbolic_contract import validate_operations, verify_circuit, simplify_exact
    from symbolic_feasibility import feasibility_obstruction
    from symbolic_synthesis import _physical_scalar, _verify_phase_candidate


NAMES = {"u": "U3", "cx": "CX", "cz": "CZ", "ecr": "ECR",
         "sx": "Sx", "x": "X", "rz": "Rz", "ry": "Ry", "rx": "Rx",
         "iswap": "iSWAP", "rxx": "Rxx", "rzz": "Rzz", "rzx": "Rzx"}


def propose_pi_angle(value, max_denominator=384, tolerance=1e-11):
    """A bounded rational proposal; proximity alone never verifies a circuit."""
    number = float(value)
    if not math.isfinite(number):
        return None
    ratio = Fraction(number / math.pi).limit_denominator(max_denominator)
    if abs(number - math.pi * ratio.numerator / ratio.denominator) > tolerance:
        return None
    return sp.pi * sp.Rational(ratio.numerator, ratio.denominator)


def lift_sdk_candidate(target, baseline, basis, *, phase_exact=False):
    target = sp.ImmutableMatrix(target)
    selected = get_basis(basis)
    declined = dict(status="proposal_not_proved", phase_exact=phase_exact)
    if target.free_symbols or target.atoms(sp.Float):
        return dict(declined, reason="Fixed exact targets only")
    if baseline.get("status") != "verified_numeric" or baseline.get("basis") != selected.name:
        return dict(declined, reason="No verified baseline in the preserved basis")
    if phase_exact and feasibility_obstruction(target, selected, phase_exact=True):
        return dict(declined, reason="Exact phase is obstructed in this preserved basis")
    sequence = []
    for operation in baseline["operations"]:
        name = NAMES.get(operation["gate"])
        if name is None:
            return dict(declined, reason="SDK primitive has no exact lift")
        parameters = tuple(propose_pi_angle(p) for p in operation["parameters"])
        if any(p is None for p in parameters):
            return dict(declined, reason="Angle outside the bounded rational-pi proposal alphabet")
        sequence.append((name, *parameters, tuple(operation["wires"])))
    width = target.rows.bit_length()-1
    try:
        sequence = validate_operations(sequence, selected, width)
    except ValueError as error:
        return dict(declined, reason=str(error))
    angle = propose_pi_angle(baseline.get("global_phase", 0))
    if angle is None:
        return dict(declined, reason="Scalar outside the proposal alphabet")
    factor = sp.exp(sp.I * angle)
    proof = verify_circuit(target, sequence, selected, phase_exact=False, scalar_factor=factor)
    if proof["status"] != "verified_symbolic":
        return dict(declined, reason="Exact target identity failed")
    if phase_exact:
        try:
            if simplify_exact(factor-1) != 0:
                sequence += _physical_scalar(factor, selected, width)
            proof = _verify_phase_candidate(target, sequence, selected, True, factor)
        except ValueError as error:
            return dict(declined, reason=str(error))
    if proof["status"] != "verified_symbolic":
        return dict(declined, reason="Physical phase closure failed")
    proof.update(construction_method="exact_lift_of_fixed_target_SDK_proposal",
                 entangler_optimal=False, local_cost_optimal=False,
                 proposal_provenance=dict(input=baseline.get("input"),
                     numerical_angles_are_not_a_proof=True,
                     max_pi_denominator=384, symbolic_family_claim=False))
    return proof
