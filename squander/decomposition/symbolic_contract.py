"""Exact circuits, resource objectives, and the two global-phase policies.

Physical GP operations are forbidden. In projective mode the scalar needed
to recover the target is automatic metadata and has zero physical cost.
"""

from dataclasses import dataclass
from functools import lru_cache
import sympy as sp

if __package__:
    from .symbolic_gates import compile_gates, gate_definition
else:
    from symbolic_gates import compile_gates, gate_definition


def simplify_exact(value):
    if value == 0:
        return sp.S.Zero
    value = sp.simplify(sp.trigsimp(value, method="fu"))
    if value == 0:
        return value
    return sp.simplify(sp.trigsimp(sp.expand_complex(value), method="fu"))


def exact_equal(left, right):
    return left.shape == right.shape and all(simplify_exact(a-b) == 0 for a, b in zip(left, right))


def parameter_score(sequence):
    """Nonnegative-denominator revision of the historical parameter score.

    Clamp only the logarithmic denominator penalty at zero. Ordinary scores
    with denominator magnitude >=1 are unchanged; negative-cost identity
    cycles in the unrestricted historical formula are excluded.
    """
    # Global simplify can combine the logarithms into enormous integer powers
    # (and even hit Python's integer-printing limit). Add their exact values
    # directly; no log-combination is needed for scoring or serialization.
    return sum((_operation_score(tuple(operation)) for operation in sequence),sp.S.Zero)


@lru_cache(maxsize=16384)
def _operation_score(operation):
    total=sp.Integer(1 << len(operation[-1]))
    for parameter in operation[1:-1]:
        total+=_parameter_penalty(parameter)
    return total


@lru_cache(maxsize=32768)
def _parameter_penalty(parameter):
    denominator=sp.simplify(parameter).as_numer_denom()[1]
    return (sp.log(sp.Max(1,sp.Abs(denominator)),2)+sp.Rational(1,100))/5


PARAMETER_SCORE_POLICY=dict(version=2,name="nonnegative_denominator_penalty",
    denominator_term="log2(Max(1, Abs(denominator(simplify(parameter)))))",
    operation_base="2**wire_count",parameter_surcharge="1/500",
    compatibility="Historical scores are identical when all denominator magnitudes are at least one")


@lru_cache(maxsize=16384)
def _operation_weight(operation):
    return float(_operation_score(operation))


def resource_key(sequence):
    return (sum(len(op[-1])==2 for op in sequence),sum(_operation_weight(tuple(op)) for op in sequence),
            sum(len(op[-1])==1 for op in sequence))


@dataclass(frozen=True)
class CircuitCost:
    entanglers: int
    parameter_score: object
    single_qubit_gates: int

    @property
    def key(self):
        return self.entanglers, float(self.parameter_score), self.single_qubit_gates

    def as_dict(self):
        return dict(entanglers=self.entanglers, parameter_score=str(self.parameter_score),
                    single_qubit_gates=self.single_qubit_gates)


def circuit_cost(sequence):
    sequence = tuple(sequence)
    return CircuitCost(sum(len(op[-1]) == 2 for op in sequence), parameter_score(sequence),
                       sum(len(op[-1]) == 1 for op in sequence))


def validate_operations(sequence, basis, width):
    normalized = {"U3": "U", "U1": "P", "CNOT": "CX"}
    canonical = lambda name: normalized.get(gate_definition(name).name, gate_definition(name).name)
    allowed = {canonical(name) for name in basis.gates}
    result = []
    for operation in sequence:
        name = gate_definition(operation[0]).name
        if name == "GP":
            raise ValueError("A physical GP gate is forbidden by the synthesis contract")
        if canonical(name) not in allowed:
            raise ValueError(f"Gate {name!r} is outside basis {basis.name!r}")
        wires = tuple(operation[-1])
        if (len(wires) != gate_definition(name).num_qubits or len(set(wires)) != len(wires)
                or any(not isinstance(q, int) or isinstance(q, bool) or not 0 <= q < width for q in wires)):
            raise ValueError("Invalid operation wires")
        parameters = tuple(sp.sympify(p) for p in operation[1:-1])
        if any(p.atoms(sp.Float) for p in parameters):
            raise ValueError("Symbolic circuits cannot contain floating constants")
        result.append((operation[0], *parameters, wires))
    return tuple(result)


def verify_circuit(target, sequence, basis, *, phase_exact=True, scalar_factor=None):
    """Prove literal equality or infer and prove the unique scalar correction."""
    target = sp.ImmutableMatrix(target)
    if target.atoms(sp.Float):
        raise ValueError("An exact target cannot contain floating constants")
    width = target.rows.bit_length()-1
    if target.shape != (1 << width, 1 << width):
        raise ValueError("Target must have a square power-of-two dimension")
    sequence = validate_operations(sequence, basis, width)
    actual = None
    if phase_exact:
        factor = sp.S.One
    elif scalar_factor is not None:
        factor=sp.sympify(scalar_factor)
    else:
        if __package__:
            from .symbolic_identity import infer_algebraic_circuit_scalar
        else:
            from symbolic_identity import infer_algebraic_circuit_scalar
        inferred=infer_algebraic_circuit_scalar(target,sequence)
        if inferred is False:return dict(status="identity_not_proved",phase_exact=False)
        if inferred is not None:factor=inferred
        else:
            actual = compile_gates(width, sequence)
            pivot = next((i for i, value in enumerate(actual) if value.is_zero is False), None)
            if pivot is None:
                pivot = next((i for i, value in enumerate(actual) if simplify_exact(value) != 0), None)
            if pivot is None:
                return dict(status="identity_not_proved", phase_exact=False)
            factor = simplify_exact(target[pivot]/actual[pivot])
    if __package__:
        from .symbolic_identity import prove_circuit_identity
    else:
        from symbolic_identity import prove_circuit_identity
    proved=prove_circuit_identity(target,sequence,factor)
    if proved is None:
        actual=compile_gates(width,sequence) if actual is None else actual
        proved=exact_equal(target,factor*actual)
    if simplify_exact(factor*sp.conjugate(factor)-1) != 0 or not proved:
        return dict(status="identity_not_proved", phase_exact=phase_exact)
    return dict(status="verified_symbolic", phase_exact=phase_exact,
                cost=circuit_cost(sequence).as_dict(), operations=encode_sequence(sequence),
                virtual_gp=None if phase_exact else dict(factor=str(factor), physical=False, cost=0),
                verification="exact_exponential_polynomial_identity")


def encode_sequence(sequence):
    return [dict(gate=op[0], parameters=[str(p) for p in op[1:-1]], wires=list(op[-1])) for op in sequence]


def verify_phase_closure_certificate(target,result,basis):
    """Replay both exact component identities in a stored physical closure."""
    if result.get("verification")!="composition_of_exact_symbolic_identities" or not result.get("phase_exact"):
        return False
    if result.get("virtual_gp") is not None:return False
    target=sp.ImmutableMatrix(target)
    symbols={str(symbol):symbol for symbol in target.free_symbols}
    try:
        sequence=tuple((op["gate"],*(sp.sympify(p,locals=symbols) for p in op["parameters"]),tuple(op["wires"]))
                       for op in result["operations"])
        sequence=validate_operations(sequence,basis,target.rows.bit_length()-1)
        if circuit_cost(sequence).as_dict()!=result["cost"]:return False
        certificate=result["phase_closure_certificate"]
        split=certificate["prefix_operations"]
        if not isinstance(split,int) or not 0<=split<len(sequence):return False
        factor=sp.sympify(certificate["factor"],locals=symbols)
        source=verify_circuit(target,sequence[:split],basis,phase_exact=False,scalar_factor=factor)
        phase=verify_circuit(factor*sp.eye(target.rows),sequence[split:],basis,phase_exact=True)
        return source["status"]==phase["status"]=="verified_symbolic"
    except (KeyError,TypeError,ValueError):return False


def compare_recorded_costs(left, right):
    """Compare exact recorded objectives, returning None for an unknown score.

    A strict entangler improvement is decisive even when its secondary score
    has not been evaluated. An unknown score never becomes zero, a tie, or a
    reason to use the tertiary local-count objective.
    """
    if left['entanglers'] != right['entanglers']:
        return -1 if left['entanglers'] < right['entanglers'] else 1
    if left.get('parameter_score') is None or right.get('parameter_score') is None:
        return None
    lkey = float(sp.sympify(left['parameter_score'])), left['single_qubit_gates']
    rkey = float(sp.sympify(right['parameter_score'])), right['single_qubit_gates']
    return -1 if lkey < rkey else 1 if lkey > rkey else 0


def retain_phase_results(exact, projective):
    """Prefer phase exact; retain projective only for a strictly better cost."""
    good = lambda row: row is not None and row.get("status") == "verified_symbolic"
    if good(exact):
        comparison = compare_recorded_costs(projective['cost'], exact['cost']) if good(projective) else 1
        keep_projective = good(projective) and (comparison is None or comparison < 0)
        result = dict(preferred="projective" if comparison is not None and comparison < 0 else "phase_exact", phase_exact=exact,
                    projective=projective if keep_projective else None,
                    projective_omitted_as_no_better=good(projective) and not keep_projective)
        if comparison is None:
            result['phase_cost_comparison'] = 'secondary_score_unresolved_both_modes_retained'
        return result
    return dict(preferred="projective" if good(projective) else None, phase_exact=exact,
                projective=projective, projective_omitted_as_no_better=False)
