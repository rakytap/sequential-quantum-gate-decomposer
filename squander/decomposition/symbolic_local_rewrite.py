"""Exact one-wire block rewrites with replayable component identities."""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost, encode_sequence, resource_key, validate_operations, verify_circuit
    from .symbolic_identity import prove_circuit_identity
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost, encode_sequence, resource_key, validate_operations, verify_circuit
    from symbolic_identity import prove_circuit_identity


def _decode(operations):
    return tuple((op["gate"], *(sp.sympify(p) for p in op["parameters"]), tuple(op["wires"]))
                 for op in operations)


def _euler(matrix):
    """Propose a U3 chart; exact matrix multiplication decides acceptance."""
    a, b, c, d = matrix
    if sp.simplify(a) == 0:
        return (sp.pi, sp.arg(c), sp.arg(-b)), sp.S.One
    first = sp.arg(a)
    if sp.simplify(c) == 0:
        return (sp.S.Zero, sp.S.Zero, sp.simplify(sp.arg(d)-first)), sp.exp(sp.I*first)
    aa = sp.simplify(sp.re(a*sp.conjugate(a)))
    cc = sp.simplify(sp.re(c*sp.conjugate(c)))
    return (2*sp.atan2(sp.sqrt(cc), sp.sqrt(aa)), sp.simplify(sp.arg(c)-first),
            sp.simplify(sp.arg(-b)-first)), sp.exp(sp.I*first)


@lru_cache(maxsize=8192)
def _block(sequence, basis):
    if __package__:
        from .symbolic_synthesis import _local, _best_clifford, _cleanup
    else:
        from symbolic_synthesis import _local, _best_clifford, _cleanup
    if len(sequence) < 2:
        return sequence, sp.S.One
    matrix = gates.compile_gates(1, sequence).applyfunc(sp.simplify)
    proposals = [(sequence, sp.S.One)]
    if matrix == sp.eye(2):
        proposals.append(((), sp.S.One))
    clifford = _best_clifford(matrix, basis, False)
    if clifford is not None:
        proposals.append((clifford["_sequence"], clifford["_factor"]))
    try:
        angles, phase = _euler(matrix)
        if "PhasedXZ" in basis.single_qubit_gates:
            theta, phi, lam = angles
            word = (("PhasedXZ", theta/sp.pi, (phi+lam)/sp.pi,
                     sp.Rational(1, 2)-lam/sp.pi, (0,)),)
            proposals.append((word, sp.simplify(phase*sp.exp(-sp.I*theta/2))))
        else:
            word, factor = _local("U3", angles, basis)
            proposals.append((_cleanup(word), sp.simplify(phase*factor)))
    except (ValueError, KeyError, TypeError):
        pass
    for word, factor in sorted(proposals, key=lambda item:resource_key(item[0])):
        if word == sequence and factor == 1:
            return sequence, sp.S.One
        if prove_circuit_identity(matrix, word, factor) is True:
            return word, factor
    return sequence, sp.S.One


def optimize_local_word(sequence, basis, width):
    pending = [[] for _ in range(width)]
    output, records = [], []
    factor = sp.S.One

    def flush(wire):
        nonlocal factor
        if not pending[wire]:
            return
        original = tuple(pending[wire])
        canonical = tuple((*op[:-1], (0,)) for op in original)
        word, scalar = _block(canonical, basis)
        replacement = tuple((*op[:-1], (wire,)) for op in word)
        output.extend(replacement)
        factor *= scalar
        if original != replacement or scalar != 1:
            records.append(dict(wire=wire, original=encode_sequence(original),
                                replacement=encode_sequence(replacement), factor=str(scalar)))
        pending[wire] = []

    for operation in sequence:
        if len(operation[-1]) == 1:
            pending[operation[-1][0]].append(operation)
        else:
            for wire in operation[-1]:
                flush(wire)
            output.append(operation)
    for wire in range(width):
        flush(wire)
    return tuple(output), sp.simplify(factor), records


def optimize_local_result(target, parent, basis):
    """Prove the parent and every changed block; keep only a cheaper result."""
    if parent.get("status") != "verified_symbolic" or target.free_symbols:
        return parent
    if parent.get("local_cost_optimal"):
        return parent
    phase_exact = parent["phase_exact"]
    width = target.rows.bit_length()-1
    sequence = validate_operations(_decode(parent["operations"]), basis, width)
    parent_factor = sp.S.One if phase_exact else sp.sympify(parent["virtual_gp"]["factor"])
    # Status strings are not trusted: the native parent identity is replayed.
    if verify_circuit(target, sequence, basis, phase_exact=phase_exact,
                      scalar_factor=parent_factor)["status"] != "verified_symbolic":
        raise ValueError("Local rewrite parent identity was not proved")
    word, scalar, records = optimize_local_word(sequence, basis, width)
    if not records:
        return parent
    factor = sp.simplify(parent_factor*scalar)
    correction = ()
    if phase_exact and factor != 1:
        if __package__:
            from .symbolic_synthesis import _physical_scalar
        else:
            from symbolic_synthesis import _physical_scalar
        try:
            correction = _physical_scalar(factor, basis, width)
        except ValueError:
            return parent
        if prove_circuit_identity(factor*sp.eye(1 << width), correction) is not True:
            return parent
        word += correction
    if resource_key(word) >= resource_key(sequence):
        return parent
    result = {key:value for key,value in parent.items()}
    result.update(operations=encode_sequence(word), cost=circuit_cost(word).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(factor),physical=False,cost=0),
        verification="exact_one_wire_block_rewrites",
        construction_method="exact_native_one_wire_block_optimization",
        local_cost_optimal=False,
        local_rewrite_certificate=dict(parent_operations=parent["operations"],
            parent_factor=str(parent_factor), parent_phase_exact=phase_exact,
            blocks=records, correction=encode_sequence(correction)))
    return result


def verify_local_rewrite_certificate(target, result, basis):
    if result.get("verification") != "exact_one_wire_block_rewrites":
        return False
    try:
        certificate = result["local_rewrite_certificate"]
        phase_exact = certificate["parent_phase_exact"]
        parent = dict(status="verified_symbolic",phase_exact=phase_exact,
            operations=certificate["parent_operations"],
            virtual_gp=None if phase_exact else dict(factor=certificate["parent_factor"]))
        parent_word = _decode(parent["operations"])
        parent["cost"] = circuit_cost(parent_word).as_dict()
        expected = optimize_local_result(sp.ImmutableMatrix(target), parent, basis)
        return all(result.get(key) == expected.get(key) for key in
                   ("operations","cost","virtual_gp","local_rewrite_certificate","verification"))
    except (ValueError,KeyError,TypeError):
        return False
