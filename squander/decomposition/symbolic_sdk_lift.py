"""Use numerical SDK angles only as proposals for exact fixed-target circuits.

Rational multiples of pi and a small explicit algebraic-angle dictionary are
recovered within a bounded proposal alphabet.
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


def propose_exact_angle(value):
    direct=propose_pi_angle(value)
    if direct is not None:return direct
    # These angles occur in analytical Euler frames for controlled gates.
    # Matching floats proposes an expression; only the subsequent complete
    # exact target identity can accept it. This is no free-gauge assumption.
    for base in (sp.acos(1/sp.sqrt(3)),sp.atan(1/sp.sqrt(2))):
        for coefficient in (1,-1,2,-2,sp.Rational(1,2),sp.Rational(-1,2)):
            offset=propose_pi_angle(float(value)-float(coefficient*base))
            if offset is not None:return coefficient*base+offset
    return None


def _gauge_fixed_proposal(baseline):
    """Propose zero gauges with matched signs/scales and rational-pi offsets.

    Numerical relations only suggest a new exact word. In particular, this
    does not assert that an inferred gauge is free. The complete proposed
    circuit still has to pass the exact fixed-target identity proof.
    """
    if not all(math.isfinite(float(value)) for operation in baseline["operations"]
               for value in operation["parameters"]):return None
    if not math.isfinite(float(baseline.get("global_phase",0))):return None
    representatives=[]
    def angle(value):
        direct=propose_exact_angle(value)
        if direct is not None:return direct
        for old in representatives:
            for coefficient in (1,-1,sp.Rational(1,2),sp.Rational(-1,2),2,-2,
                                sp.Rational(1,4),sp.Rational(-1,4)):
                offset=propose_pi_angle(float(value)-float(coefficient)*old,tolerance=1e-10)
                if offset is not None:return offset
        representatives.append(float(value))
        return sp.S.Zero
    word=[]
    for operation in baseline["operations"]:
        name=NAMES.get(operation["gate"])
        if name is None:return None
        parameters=tuple(angle(value) for value in operation["parameters"])
        if name=="U3" and parameters[0]==0:
            parameters=(sp.S.Zero,sp.S.Zero,sp.simplify(parameters[1]+parameters[2]))
            if parameters[2]==0:continue
        word.append((name,*parameters,tuple(operation["wires"])))
    phase=angle(baseline.get("global_phase",0))
    return tuple(word),phase,len(representatives)


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
    gauge_fixed=False
    for operation in baseline["operations"]:
        name = NAMES.get(operation["gate"])
        if name is None:
            return dict(declined, reason="SDK primitive has no exact lift")
        parameters = tuple(propose_exact_angle(p) for p in operation["parameters"])
        if any(p is None for p in parameters):
            gauge_fixed=True
            break
        sequence.append((name, *parameters, tuple(operation["wires"])))
    proposal=_gauge_fixed_proposal(baseline) if gauge_fixed else None
    if gauge_fixed:
        if proposal is None:return dict(declined,reason="SDK primitive has no exact lift")
        sequence,angle,gauge_count=proposal
    else:
        angle=propose_pi_angle(baseline.get("global_phase",0))
    width = target.rows.bit_length()-1
    try:
        sequence = validate_operations(sequence, selected, width)
    except ValueError as error:
        return dict(declined, reason=str(error))
    if angle is None:
        return dict(declined, reason="Scalar outside the proposal alphabet")
    factor = sp.exp(sp.I * angle)
    # Gauge changes can change a scalar even when all nonlocal content is
    # preserved. Infer that scalar from exact matrices and prove every entry.
    proof = verify_circuit(target, sequence, selected, phase_exact=False,
                           scalar_factor=None if gauge_fixed else factor)
    if proof["status"] != "verified_symbolic":
        return dict(declined, reason="Exact target identity failed")
    if gauge_fixed:
        factor=sp.sympify(proof["virtual_gp"]["factor"])
    if phase_exact:
        try:
            if simplify_exact(factor-1) != 0:
                sequence += _physical_scalar(factor, selected, width)
            proof = _verify_phase_candidate(target, sequence, selected, True, factor)
        except ValueError as error:
            return dict(declined, reason=str(error))
    if proof["status"] != "verified_symbolic":
        return dict(declined, reason="Physical phase closure failed")
    proof.update(construction_method="exact_lift_of_fixed_target_SDK_gauge_proposal" if gauge_fixed else "exact_lift_of_fixed_target_SDK_proposal",
                 entangler_optimal=False, local_cost_optimal=False,
                 proposal_provenance=dict(input=baseline.get("input"),
                     numerical_angles_are_not_a_proof=True,
                     max_pi_denominator=384, symbolic_family_claim=False,
                     algebraic_angle_dictionary=["acos(1/sqrt(3))","atan(1/sqrt(2))"],
                     gauge_fixed=gauge_fixed,proposed_gauges=gauge_count if gauge_fixed else 0))
    return proof
