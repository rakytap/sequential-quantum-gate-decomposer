"""Exact transfer of a proved native template through one-wire components.

The large parent is checked once. Each local replacement and the diagonal
commutation lemma are proved independently, avoiding re-expansion of the
same large circuit after native Euler lowering.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_bases import get_basis
    from .symbolic_contract import circuit_cost,encode_sequence,validate_operations,resource_key
    from .symbolic_fixed_templates import fixed_template_candidates
    from .symbolic_identity import prove_circuit_identity
else:
    import symbolic_gates as gates
    from symbolic_bases import get_basis
    from symbolic_contract import circuit_cost,encode_sequence,validate_operations,resource_key
    from symbolic_fixed_templates import fixed_template_candidates
    from symbolic_identity import prove_circuit_identity


@lru_cache(maxsize=1)
def verify_diagonal_commutation_lemma():
    angle=sp.Symbol('transfer_angle',real=True)
    cz=('CZ',(0,1))
    return all(prove_circuit_identity(gates.compile_gates(2,(('Rz',angle,(wire,)),cz)),
                  (cz,('Rz',angle,(wire,)))) is True for wire in (0,1))


def commute_diagonal_locals(sequence,width):
    if __package__:
        from .symbolic_synthesis import _cleanup
    else:
        from symbolic_synthesis import _cleanup
    if not verify_diagonal_commutation_lemma():raise ArithmeticError('Diagonal commutator failed')
    pending=[sp.S.Zero]*width
    output=[]
    def flush(wire):
        angle=sp.simplify(pending[wire])
        if angle!=0:output.append(('Rz',angle,(wire,)))
        pending[wire]=sp.S.Zero
    for operation in sequence:
        if operation[0]=='Rz':pending[operation[-1][0]]+=operation[1]
        elif operation[0]=='CZ':output.append(operation)
        else:
            for wire in operation[-1]:flush(wire)
            output.append(operation)
    for wire in range(width):flush(wire)
    return _cleanup(output)


@lru_cache(maxsize=4)
def _parent(name):
    candidates=fixed_template_candidates(name,(),get_basis('cz_u'))
    if not candidates:return None
    word,factor=min(candidates,key=lambda item:resource_key(item[0]))
    target=gates.make_gate(name)
    if prove_circuit_identity(target,word,factor) is not True:raise ArithmeticError('Native parent template failed')
    return word,factor


@lru_cache(maxsize=256)
def _lower_component(operation,basis):
    if __package__:
        from .symbolic_synthesis import _local,_rotation
    else:
        from symbolic_synthesis import _local,_rotation
    name,parameters=operation[0],operation[1:-1]
    if name=='U3' and parameters[1]+parameters[2]==0 and parameters[1] in (sp.pi/2,-sp.pi/2):
        word,factor=_rotation('Rx',-2*parameters[1]*parameters[0]/sp.pi,basis)
    else:word,factor=_local(name,parameters,basis)
    if prove_circuit_identity(gates.make_gate(name,*parameters),word,factor) is not True:
        raise ArithmeticError('Native local component failed')
    return word,factor


def synthesize_template_transfer(name,basis,*,phase_exact):
    if name!='C4X' or basis.entangler!='CZ' or 'Rz' not in basis.single_qubit_gates:return None
    parent=_parent(name)
    if parent is None:return None
    word,factor=parent
    output=[]
    components=[]
    for operation in word:
        if len(operation[-1])==2:output.append(operation);continue
        canonical=(*operation[:-1],(0,))
        lowered,scalar=_lower_component(canonical,basis)
        output.extend((*op[:-1],operation[-1]) for op in lowered)
        factor*=scalar
        components.append(dict(original=encode_sequence((canonical,)),replacement=encode_sequence(lowered),factor=str(scalar)))
    factor=sp.simplify(factor)
    width=gates.make_gate(name).rows.bit_length()-1
    output=commute_diagonal_locals(output,width)
    if phase_exact and factor!=1:
        if __package__:
            from .symbolic_synthesis import _physical_scalar,RuleUnavailable
        else:
            from symbolic_synthesis import _physical_scalar,RuleUnavailable
        try:output+=_physical_scalar(factor,basis,width)
        except RuleUnavailable:return None
    output=validate_operations(output,basis,width)
    return dict(status='verified_symbolic',phase_exact=phase_exact,
        operations=encode_sequence(output),cost=circuit_cost(output).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(factor),physical=False,cost=0),
        verification='exact_native_parent_local_components_and_CZ_commutations',
        construction_method='native_template_basis_transfer',
        basis_transfer_certificate=dict(target=name,parent_basis='cz_u',parent_operations=encode_sequence(word),
            parent_factor=str(parent[1]),components=components,
            commutations='Rz_commutes_with_CZ_on_either_wire'),
        entangler_optimal=False,local_cost_optimal=False,optimality_scope='constructive upper bound')


def verify_basis_transfer_certificate(target,result,basis):
    if result.get('verification')!='exact_native_parent_local_components_and_CZ_commutations':return False
    if not isinstance(result.get('phase_exact'),bool):return False
    name=result.get('basis_transfer_certificate',{}).get('target')
    if name!='C4X' or sp.ImmutableMatrix(target)!=gates.make_gate(name):return False
    expected=synthesize_template_transfer(name,basis,phase_exact=result['phase_exact'])
    return expected is not None and all(result.get(key)==expected.get(key) for key in
        ('status','operations','cost','virtual_gp','basis_transfer_certificate','entangler_optimal','local_cost_optimal'))
