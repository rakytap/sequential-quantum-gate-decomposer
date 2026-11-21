"""Exact polar fusion of constant one-wire blocks, with explicit zero charts.

The verifier uses unitarity, the defining polar identities of arg/atan2 and
column orthogonality. It does not ask a general trigonometric simplifier to
rediscover those branch identities after every algebraic substitution.
"""

from functools import lru_cache
import json
from pathlib import Path
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost,encode_sequence,resource_key,validate_operations
    from .symbolic_controlled_chart import _zero
    from .symbolic_identity import prove_circuit_identity
    from .symbolic_native_fixed import verify_native_lowering_lemmas
    from .symbolic_native_pauli import _phase
    from .symbolic_sqrt_iswap_phase import _phase_in_u3
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost,encode_sequence,resource_key,validate_operations
    from symbolic_controlled_chart import _zero
    from symbolic_identity import prove_circuit_identity
    from symbolic_native_fixed import verify_native_lowering_lemmas
    from symbolic_native_pauli import _phase
    from symbolic_sqrt_iswap_phase import _phase_in_u3


@lru_cache(maxsize=1)
def verify_local_polar_lemma():
    beta,phi,lam,delta=sp.symbols('polar_beta polar_phi polar_lambda polar_delta',real=True)
    r,s=sp.cos(beta),sp.sin(beta)
    target=sp.exp(sp.I*delta)*sp.Matrix([[r,-s*sp.exp(sp.I*lam)],
                              [s*sp.exp(sp.I*phi),r*sp.exp(sp.I*(phi+lam))]])
    if prove_circuit_identity(target,(('U3',2*beta,phi,lam,(0,)),),sp.exp(sp.I*delta)) is not True:return False
    # Generic chart: e^(i arg A)=A/r, e^(i arg C)=C/s,
    # e^(i arg(-B))=-B/s. r,s>0 and r^2+s^2=1 put atan2(s,r)
    # in the first quadrant. The last entry follows from orthogonality.
    A,B,C,D,Ac,Cc,r2,s2=sp.symbols('A B C D Ac Cc r2 s2')
    numerator=r2*B*C+D*s2*A
    relations=B*C*(r2-A*Ac)+D*A*(s2-C*Cc)+A*C*(Ac*B+Cc*D)
    if sp.expand(numerator-relations)!=0:return False
    # The r=0 and s=0 charts use theta=pi and theta=0 respectively;
    # unitarity makes their complementary entries zero and magnitudes one.
    for theta in (0,sp.pi):
        if prove_circuit_identity(gates.make_gate('U3',theta,phi,lam),
                                  (('U3',theta,phi,lam,(0,)),)) is not True:return False
    theta=sp.Symbol('polar_theta',real=True)
    if prove_circuit_identity(gates.make_gate('U3',theta,phi,lam),
            (('Rz',lam,(0,)),('Ry',theta,(0,)),('Rz',phi,(0,))),
            sp.exp(sp.I*(phi+lam)/2)) is not True:return False
    return verify_native_lowering_lemmas()


@lru_cache(maxsize=512)
def _zero_decision(value):
    value=sp.simplify(value)
    if value.is_zero is not None:return bool(value.is_zero)
    if value.is_algebraic is True:
        z=sp.Dummy('polar_root')
        polynomial=sp.Poly(sp.minpoly(value,z),z)
        return polynomial.TC()==0
    raise ValueError('Exact polar zero/nonzero decision unavailable')


@lru_cache(maxsize=1024)
def polar_component(sequence):
    matrix=sp.eye(2)
    for operation in sequence:
        parameters=tuple(map(sp.sympify,operation[1:-1]))
        if (operation[-1]!=(0,) or any(p.free_symbols or p.atoms(sp.Float) or p.is_real is not True for p in parameters)
                or gates.gate_definition(operation[0]).num_qubits!=1):
            raise ValueError('Constant exact real one-wire primitives required')
        if not _primitive_unitarity(operation[0]):raise ValueError('Primitive unitarity not proved')
        matrix=(gates.make_gate(operation[0],*parameters)*matrix).applyfunc(sp.expand_mul)
    matrix=sp.ImmutableMatrix(matrix.applyfunc(sp.simplify))
    if not verify_local_polar_lemma():raise ArithmeticError('Polar chart lemma failed')
    A,B,C,D=matrix
    aa=sp.simplify(A*sp.conjugate(A))
    cc=sp.simplify(C*sp.conjugate(C))
    # The product of the proved primitive unitaries is unitary. Hence these
    # actual squared magnitudes sum to one, |B|^2=cc, and the columns are
    # orthogonal. Re-expanding those same identities into unrelated constant
    # cyclotomic radicals would add no proof and can be much more expensive.
    az,cz=_zero_decision(aa),_zero_decision(cc)
    if az:
        word=(('U3',sp.pi,sp.arg(C),sp.arg(-B),(0,)),);angle=sp.S.Zero;chart='A=0'
    elif cz:
        angle=sp.arg(A);chart='C=0'
        word=() if _zero((A-D,)) else (('U3',0,0,sp.simplify(sp.arg(D)-angle),(0,)),)
    else:
        # Norms of actual complex numbers are nonnegative; exact nonzero
        # decisions above make both positive. No (0,0) atan2 is possible.
        angle=sp.arg(A);chart='A!=0,C!=0'
        word=(('U3',2*sp.atan2(sp.sqrt(cc),sp.sqrt(aa)),
               sp.simplify(sp.arg(C)-angle),sp.simplify(sp.arg(-B)-angle),(0,)),)
    certificate=dict(matrix=[str(value) for value in matrix],chart=chart,
        norms=[str(aa),str(cc)],phase_angle=str(angle),
        theorem='exact_U2_polar_chart_and_column_orthogonality',
        unitarity='composition_of_exact_real_primitive_unitaries')
    return word,angle,certificate


@lru_cache(maxsize=64)
def _primitive_unitarity(name):
    definition=gates.gate_definition(name)
    if name not in {'U3','U','Rx','Ry','Rz','P','U1','H','X','Y','Z','I','S','Sdg','T','Tdg','Sx','Sxdg'}:
        return False
    parameters=sp.symbols(' '.join('primitive_'+p for p in definition.parameters),real=True) if definition.parameters else ()
    if len(definition.parameters)==1:parameters=(parameters,)
    matrix=gates.make_gate(name,*parameters)
    return _zero(matrix.H*matrix-sp.eye(2))


@lru_cache(maxsize=4)
def _parent(name,basis_name):
    data=json.loads(Path(__file__).with_name('symbolic_local_polar_parents.json').read_text())
    row=next((row for row in data['entries'] if (row['gate'],row['basis'])==(name,basis_name)),None)
    if row is None:return None
    word=tuple((op['gate'],*(sp.sympify(p) for p in op['parameters']),tuple(op['wires'])) for op in row['operations'])
    factor=sp.sympify(row['factor'])
    if prove_circuit_identity(gates.make_gate(name),word,factor) is not True:raise ArithmeticError('Polar parent identity failed')
    return word,factor


def synthesize_local_polar(name,basis,*,phase_exact):
    if (name,basis.name) not in {('C3SX','iswap_u'),('C4X','iswap_u'),('Peres','xx_rz_ry')}:return None
    parent=_parent(name,basis.name)
    if parent is None:return None
    source,factor=parent
    angle=_phase(factor)
    if angle is None:raise ValueError('Parent scalar angle not proved')
    width=gates.make_gate(name).rows.bit_length()-1
    pending=[[] for unused in range(width)]
    output=[];records=[]
    def flush(wire):
        nonlocal angle
        if not pending[wire]:return
        original=tuple(pending[wire]);canonical=tuple((*op[:-1],(0,)) for op in original)
        word,phase,certificate=polar_component(canonical)
        if word and 'U3' not in basis.single_qubit_gates:
            theta,phi,lam=word[0][1:-1]
            if set(basis.single_qubit_gates)=={'Rz','Ry'}:
                if __package__:
                    from .symbolic_synthesis import _cleanup
                else:
                    from symbolic_synthesis import _cleanup
                word=_cleanup((('Rz',lam,(0,)),('Ry',theta,(0,)),('Rz',phi,(0,))))
                phase+=(phi+lam)/2
            else:raise ValueError('No proved native polar lowering for this basis')
        if resource_key(word)>=resource_key(canonical):word,phase=canonical,sp.S.Zero
        output.extend((*op[:-1],(wire,)) for op in word)
        angle+=phase
        records.append(dict(wire=wire,original=encode_sequence(original),replacement=encode_sequence(word),proof=certificate))
        pending[wire]=[]
    for operation in source:
        if len(operation[-1])==1:pending[operation[-1][0]].append(operation)
        else:
            for wire in operation[-1]:flush(wire)
            output.append(operation)
    for wire in range(width):flush(wire)
    angle=sp.simplify(angle)
    if phase_exact and angle!=0:
        if 'U3' not in basis.single_qubit_gates:return None
        index=next(index for index,op in enumerate(output) if len(op[-1])==1)
        output=output[:index]+list(_phase_in_u3(output[index],angle))+output[index+1:]
    output=validate_operations(output,basis,width)
    return dict(status='verified_symbolic',phase_exact=phase_exact,
        operations=encode_sequence(output),cost=circuit_cost(output).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*angle)),physical=False,cost=0),
        verification='exact_parent_U2_polar_components_and_phase_absorption',
        construction_method='analytic_native_local_polar_fusion',
        local_polar_certificate=dict(target=name,parent_operations=encode_sequence(source),parent_factor=str(factor),
            blocks=records,total_phase_angle=str(angle)),entangler_optimal=False,local_cost_optimal=False,
        optimality_scope='constructive upper bound; polar charts do not prove a global local minimum')


def verify_local_polar_certificate(target,result,basis):
    if result.get('verification')!='exact_parent_U2_polar_components_and_phase_absorption':return False
    if not isinstance(result.get('phase_exact'),bool):return False
    name=result.get('local_polar_certificate',{}).get('target')
    if name not in {'C3SX','C4X','Peres'} or sp.ImmutableMatrix(target)!=gates.make_gate(name):return False
    try:expected=synthesize_local_polar(name,basis,phase_exact=result['phase_exact'])
    except (ValueError,KeyError,TypeError):return False
    return expected is not None and all(result.get(key)==expected.get(key) for key in
        ('status','operations','cost','virtual_gp','local_polar_certificate','entangler_optimal','local_cost_optimal'))
