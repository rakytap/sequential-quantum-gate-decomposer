"""Necessary exact-synthesis obstructions, never a positive universality proof."""

from functools import lru_cache
import math
import sympy as sp

if __package__:
    from .symbolic_gates import gate_definition, make_gate
    from .symbolic_contract import simplify_exact
    from .symbolic_clifford import is_clifford
else:
    from symbolic_gates import gate_definition, make_gate
    from symbolic_contract import simplify_exact
    from symbolic_clifford import is_clifford


@lru_cache(maxsize=128)
def determinant_group(basis, width):
    """Return the finite determinant-root order, or None if not established.

    An m-qubit primitive embedded in n wires contributes det(G)**2**(n-m).
    Parameter-dependent determinants deliberately prevent this finite-group
    obstruction; no unsupported conclusion is inferred from them.
    """
    order = 1
    for name in basis.gates:
        definition = gate_definition(name)
        if definition.num_qubits > width:
            continue
        arguments = sp.symbols("d:"+str(len(definition.parameters)), real=True)
        determinant = sp.simplify(make_gate(name, *arguments).det())
        if determinant.free_symbols:
            return None
        power = 1 << (width-definition.num_qubits)
        determinant = simplify_exact(determinant**power)
        root_order = next((k for k in range(1, 65) if simplify_exact(determinant**k-1) == 0), None)
        if root_order is None:
            return None
        order = order*root_order//math.gcd(order, root_order)
    return order


@lru_cache(maxsize=1)
def _clifford_t_field():
    return sp.QQ.algebraic_field(sp.sqrt(2), sp.I)


@lru_cache(maxsize=2048)
def _determinant(target):
    return sp.simplify(target.det())


def _nonzero_witness(expression):
    """A failed simplification is not a proof of a nonidentity."""
    if expression.is_zero is False:
        return {}
    symbols=sorted(expression.free_symbols,key=str)
    for value in (0,sp.Rational(1,2),sp.pi/2):
        substitution={symbol:value for symbol in symbols}
        witness=simplify_exact(expression.subs(substitution))
        if not witness.has(sp.nan,sp.zoo) and witness.is_zero is False:
            return {str(symbol):str(value) for symbol in symbols}
    # A diagonal probe can land entirely on a special-value locus. Vary one
    # parameter at a time as well: t=1/2, shift=0 detects a phase that vanishes
    # when both parameters are set to 1/2. These are exact nonidentity witnesses,
    # not numerical acceptance tests or a proof for every specialization.
    for symbol in symbols:
        substitution={other:sp.S.Zero for other in symbols}
        substitution[symbol]=sp.Rational(1,2)
        witness=simplify_exact(expression.subs(substitution))
        if not witness.has(sp.nan,sp.zoo) and witness.is_zero is False:
            return {str(other):str(value) for other,value in substitution.items()}
    return None


@lru_cache(maxsize=64)
def _alphabet_in_field(basis):
    from sympy.polys.polyerrors import CoercionFailed,NotAlgebraic
    for name in basis.gates:
        if gate_definition(name).parameters:return False
        for value in dict.fromkeys(make_gate(name)):
            try:_clifford_t_field().from_sympy(value)
            except (CoercionFailed,NotAlgebraic):return False
    return True


@lru_cache(maxsize=64)
def _alphabet_is_clifford(basis):
    return all(not gate_definition(name).parameters and is_clifford(make_gate(name))
               for name in basis.gates)


@lru_cache(maxsize=1024)
def _exact_variation(target,phase_exact):
    symbols=sorted(target.free_symbols,key=str)
    probes=[]
    for value in (0,sp.Rational(1,2),sp.pi/2):
        matrix=target.subs({symbol:value for symbol in symbols}).applyfunc(sp.simplify)
        if not matrix.has(sp.nan,sp.zoo):probes.append(matrix)
    for left,right in zip(probes,probes[1:]):
        if phase_exact:
            if any(simplify_exact(a-b).is_zero is False for a,b in zip(left,right)):return True
        else:
            pivot=next((i for i,x in enumerate(left) if x.is_zero is False),None)
            if pivot is not None and any(simplify_exact(a*right[pivot]-b*left[pivot]).is_zero is False
                                         for a,b in zip(left,right)):return True
    return False


def feasibility_obstruction(target, basis, *, phase_exact=True):
    target = sp.ImmutableMatrix(target)
    width = target.rows.bit_length()-1
    if width == 0:
        if phase_exact and simplify_exact(target[0,0]-1) != 0:
            return dict(status="proved_impossible", obstruction="zero_qubit_no_physical_operation",
                        explanation="A zero-qubit circuit without GP contains only the identity.")
        return None
    if phase_exact:
        order = determinant_group(basis, width)
        if order is not None:
            determinant = _determinant(target)
            residual = simplify_exact(determinant**order-1)
            # For a free family, a nonzero analytic function proves that the
            # entire family is not representable. Specializations may differ.
            witness=_nonzero_witness(residual)
            if witness is not None:
                return dict(status="proved_impossible", obstruction="determinant_group",
                            determinant=str(determinant), determinant_root_order=order,
                            nonzero_condition=str(sp.Ne(residual, 0)),
                            scope="generic_family" if residual.free_symbols else "this_target",
                            exact_nonidentity_witness=witness,
                            explanation=f"Every allowed circuit has det(U)**{order}=1.")
    discrete = all(not gate_definition(name).parameters for name in basis.gates)
    if discrete and target.free_symbols and _exact_variation(target,phase_exact):
        return dict(status="proved_impossible", obstruction="discrete_alphabet_nonconstant_family",
                    scope="generic_family", explanation="A fixed finite gate alphabet cannot express a nonconstant continuous family exactly.",
                    source="https://arxiv.org/abs/1206.5236")
    if basis.universality == "clifford" and _alphabet_is_clifford(basis) and not target.free_symbols and not is_clifford(target):
        return dict(status="proved_impossible", obstruction="not_a_clifford",
                    explanation="Every allowed primitive normalizes the Pauli group; this target does not.")
    if basis.universality == "approximate_discrete" and _alphabet_in_field(basis) and not target.free_symbols:
        # Necessary field membership also applies to multi-qubit circuits.
        # In projective mode ratios remove the arbitrary scalar first.
        values = list(target)
        if not phase_exact:
            pivot = next((x for x in values if x.is_zero is False), None)
            if pivot is None:
                return None
            values = [sp.simplify(x/pivot) for x in values]
        from sympy.polys.polyerrors import CoercionFailed,NotAlgebraic
        for value in dict.fromkeys(values):
            try:
                _clifford_t_field().from_sympy(value)
            except (CoercionFailed,NotAlgebraic):
                return dict(status="proved_impossible", obstruction="outside_clifford_t_number_field",
                            witness_entry_or_ratio=str(value), source="https://arxiv.org/abs/1206.5236",
                            explanation="Finite Clifford+T circuits have entries in Q(sqrt(2), i).")
        if not phase_exact:
            # If the target itself is in K and has a root-of-unity determinant,
            # any projective correction compatible with K must also be a root
            # of unity in K. Its roots of unity have order dividing eight.
            try:
                for value in dict.fromkeys(target):_clifford_t_field().from_sympy(value)
            except (CoercionFailed,NotAlgebraic):
                return None
            determinant=_determinant(target)
            order=determinant_group(basis,width)
            if order is not None and any(simplify_exact(determinant**k-1)==0 for k in range(1,9)):
                residuals=[simplify_exact((sp.exp(sp.I*sp.pi*k*target.rows/4)*determinant)**order-1)
                           for k in range(8)]
                if all(value.is_zero is False for value in residuals):
                    return dict(status="proved_impossible",obstruction="determinant_and_clifford_t_field",
                                explanation="No eighth-root scalar in Q(sqrt(2),i) gives an allowed determinant without ancillas.",
                                source="https://arxiv.org/abs/1206.5236",scope="this_target")
    return None
