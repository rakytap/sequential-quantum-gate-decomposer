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
    wrapped=[]
    wrap_scalar=sp.S.One
    for op in sequence:
        if op[0] in {"Rx","Ry","Rz","P","U1"}:
            turns=sp.simplify(op[1]/(2*sp.pi))
            if turns.is_Rational:
                periods=sp.floor(turns+sp.Rational(1,2))
                angle=sp.simplify(op[1]-2*sp.pi*periods)
                if op[0] in {"Rx","Ry","Rz"}:wrap_scalar*=(-1)**periods
                if angle!=0:wrapped.append((op[0],angle,op[-1]))
                continue
        wrapped.append(op)
    wrapped=_cleanup(tuple(wrapped))
    if len(sequence) < 2 and wrapped==sequence and wrap_scalar==1:
        return sequence, sp.S.One
    matrix = gates.compile_gates(1, sequence).applyfunc(sp.simplify)
    proposals = [(sequence, sp.S.One),(wrapped,wrap_scalar)]
    if matrix == sp.eye(2):
        proposals.append(((), sp.S.One))
    a,b,c,d=matrix
    # A pure native axis can be hidden by phase flips around a non-Clifford
    # rotation. Recover it directly instead of emitting three Euler gates.
    for axis in ("Rx","Ry","Rz"):
        if axis not in basis.single_qubit_gates:continue
        if axis=="Rz":
            if b!=0 or c!=0:continue
            first,last=sp.arg(a),sp.arg(d)
            angle=sp.simplify(last-first)
            scalar=sp.exp(sp.I*(first+last)/2)
        else:
            if sp.simplify(a-d)!=0 or sp.simplify(b-(c if axis=="Rx" else -c))!=0:continue
            first=sp.arg(a) if a!=0 else sp.arg(sp.I*b if axis=="Rx" else c)
            sine=sp.simplify(sp.re((sp.I*b if axis=="Rx" else c)*sp.exp(-sp.I*first)))
            angle=2*sp.atan2(sine,sp.Abs(a))
            scalar=sp.exp(sp.I*first)
        word=_cleanup(((axis,angle,(0,)),))
        proposals.append((word,scalar))
        if scalar==-1 and word:
            # One extra full turn absorbs -I at the same native gate count.
            proposals.append((((axis,angle+2*sp.pi,(0,)),),sp.S.One))
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
    for word, factor in sorted(proposals, key=lambda item:(resource_key(item[0]),item[1]!=1)):
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
    absorption=None
    if phase_exact and factor != 1:
        if __package__:
            from .symbolic_synthesis import _physical_scalar
            from .symbolic_sqrt_iswap_phase import _phase_in_u3,_phase_in_phased_xz
            from .symbolic_native_pauli import _phase
        else:
            from symbolic_synthesis import _physical_scalar
            from symbolic_sqrt_iswap_phase import _phase_in_u3,_phase_in_phased_xz
            from symbolic_native_pauli import _phase
        try:
            correction = _physical_scalar(factor, basis, width)
        except ValueError:
            return parent
        if prove_circuit_identity(factor*sp.eye(1 << width), correction) is not True:
            return parent
        choices=[(word+correction,None)]
        angle=_phase(factor)
        if angle is not None:
            for index,op in enumerate(word):
                if op[0] not in {"U3","U","PhasedXZ"}:continue
                before=("U3",*op[1:]) if op[0]=="U" else op
                close=_phase_in_phased_xz if op[0]=="PhasedXZ" else _phase_in_u3
                replacement=close(before,angle)
                choices.append((word[:index]+replacement+word[index+1:],
                    dict(index=index,before=encode_sequence((op,)),after=encode_sequence(replacement),factor=str(factor))))
        for proposed,record in sorted(choices,key=lambda pair:resource_key(pair[0])):
            if record is not None:
                op=word[record["index"]]
                replacement=tuple((*item[:-1],(0,)) for item in _decode(record["after"]))
                if prove_circuit_identity(factor*gates.make_gate(op[0],*op[1:-1]),replacement) is not True:continue
                correction=()
            word,absorption=proposed,record
            break
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
            blocks=records, correction=encode_sequence(correction),phase_absorption=absorption))
    return result


def verify_local_rewrite_certificate(target, result, basis):
    if result.get("verification") != "exact_one_wire_block_rewrites":
        return False
    try:
        certificate = result["local_rewrite_certificate"]
        phase_exact = certificate["parent_phase_exact"]
        target=sp.ImmutableMatrix(target)
        width=target.rows.bit_length()-1
        result_phase_exact=result.get("phase_exact")
        if not isinstance(result_phase_exact,bool):return False
        parent_word=validate_operations(_decode(certificate["parent_operations"]),basis,width)
        factor=sp.sympify(certificate["parent_factor"])
        if phase_exact and factor!=1:return False
        if verify_circuit(target,parent_word,basis,phase_exact=phase_exact,scalar_factor=factor)["status"]!="verified_symbolic":return False
        pending=[[] for unused in range(width)]
        output=[]
        records=iter(certificate["blocks"])
        next_record=next(records,None)
        def flush(wire):
            nonlocal factor,next_record
            original=tuple(pending[wire])
            if not original:return
            if next_record is not None and next_record["wire"]==wire and next_record["original"]==encode_sequence(original):
                replacement=validate_operations(_decode(next_record["replacement"]),basis,width)
                if any(op[-1]!=(wire,) for op in replacement):raise ValueError("Nonlocal block replacement")
                scalar=sp.sympify(next_record["factor"])
                if sp.simplify(scalar*sp.conjugate(scalar)-1)!=0:raise ValueError("Nonunit scalar")
                before=tuple((*op[:-1],(0,)) for op in original)
                after=tuple((*op[:-1],(0,)) for op in replacement)
                if prove_circuit_identity(gates.compile_gates(1,before),after,scalar) is not True:raise ValueError("Unproved block")
                output.extend(replacement);factor*=scalar
                next_record=next(records,None)
            else:output.extend(original)
            pending[wire]=[]
        for op in parent_word:
            if len(op[-1])==1:pending[op[-1][0]].append(op)
            else:
                for wire in op[-1]:flush(wire)
                output.append(op)
        for wire in range(width):flush(wire)
        if next_record is not None:return False
        word=tuple(output)
        factor=sp.simplify(factor)
        correction=validate_operations(_decode(certificate["correction"]),basis,width)
        absorption=certificate.get("phase_absorption")
        if absorption is not None:
            if not phase_exact or correction:return False
            index=absorption["index"]
            if not isinstance(index,int) or not 0<=index<len(word):return False
            before=word[index]
            if len(before[-1])!=1 or absorption["before"]!=encode_sequence((before,)):return False
            after=validate_operations(_decode(absorption["after"]),basis,width)
            if any(op[-1]!=before[-1] for op in after):return False
            if sp.simplify(sp.sympify(absorption["factor"])-factor)!=0:return False
            local=tuple((*op[:-1],(0,)) for op in after)
            if prove_circuit_identity(factor*gates.make_gate(before[0],*before[1:-1]),local) is not True:return False
            word=word[:index]+after+word[index+1:]
        elif phase_exact:
            if prove_circuit_identity(factor*sp.eye(1<<width),correction) is not True:return False
            word+=correction
        elif correction:return False
        effective_scalar=sp.S.One if phase_exact else factor
        if result_phase_exact and effective_scalar!=1:return False
        phase=None if result_phase_exact else dict(factor=str(effective_scalar),physical=False,cost=0)
        return (result["operations"]==encode_sequence(word) and result["cost"]==circuit_cost(word).as_dict()
                and result["virtual_gp"]==phase)
    except (ValueError,KeyError,TypeError):
        return False
