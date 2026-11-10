"""Exact two-qubit Clifford interaction search modulo local Clifford frames.

The 20 left-local cosets are indexed by binary Pauli row spaces. This is an
auxiliary discrete search; its entangler minimum is scoped to Clifford local
frames. OSR rank-one factorization recovers the final actual local operators,
including their phase, before the caller verifies the physical circuit.
"""

from collections import deque
from functools import lru_cache
import itertools
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_clifford import pauli_action
    from .symbolic_clifford_search import group_data
    from .symbolic_osr import product_factors
    from .symbolic_osr_search import recognize_local
else:
    import symbolic_gates as gates
    from symbolic_clifford import pauli_action
    from symbolic_clifford_search import group_data
    from symbolic_osr import product_factors
    from symbolic_osr_search import recognize_local


def _columns(action):
    result=[]
    for sign,label in action:
        bits=0
        for q,axis in enumerate(reversed(label)):
            if axis in "XY":bits|=1<<(2*q)
            if axis in "ZY":bits|=1<<(2*q+1)
        result.append(bits)
    return tuple(result)


def _apply(columns,bits):
    result=0
    for bit,column in enumerate(columns):
        if bits&(1<<bit):result^=column
    return result


def _multiply(left,right):return tuple(_apply(left,column) for column in right)


def local_coset_key(columns):
    result=[]
    for q in range(2):
        x=sum(((column>>(2*q))&1)<<j for j,column in enumerate(columns))
        z=sum(((column>>(2*q+1))&1)<<j for j,column in enumerate(columns))
        result.append(tuple(sorted((x,z,x^z))))
    return tuple(result)


@lru_cache(maxsize=1)
def _local_frames():
    names=group_data()[0].representatives
    single=[_columns(pauli_action(gates.make_gate(name))) for name in names]
    return tuple((a,b,(single[a][0],single[a][1],single[b][0]<<2,single[b][1]<<2))
                 for a,b in itertools.product(range(24),repeat=2))


@lru_cache(maxsize=16)
def coset_search(entangler):
    action=pauli_action(gates.make_gate(entangler))
    if action is None:raise ValueError("The chosen primitive is not a fixed Clifford")
    pulse=_columns(action)
    identity=(1,2,4,8)
    root=local_coset_key(identity)
    states={root:identity}
    distances={root:0}
    predecessors={}
    pending=deque([root])
    edges={}
    while pending:
        old=pending.popleft()
        outgoing=[]
        for a,b,frame in _local_frames():
            columns=_multiply(pulse,_multiply(frame,states[old]))
            new=local_coset_key(columns)
            outgoing.append(new)
            if new not in states:
                states[new]=columns
                distances[new]=distances[old]+1
                predecessors[new]=(old,a,b)
                pending.append(new)
        edges[old]=tuple(outgoing)
    # Closure and Bellman checks certify the discrete interaction distance.
    for old,outgoing in edges.items():
        for new in outgoing:
            if new not in distances or distances[new]>distances[old]+1:
                raise ArithmeticError("Invalid native Clifford coset distance")
    return dict(root=root,states=states,distances=distances,predecessors=predecessors,
                native_entangler=entangler,reachable_cosets=len(states),
                proof_scope="two_qubit_circuits_with_Clifford_local_frames_only")


def clifford_source_candidate(matrix,basis):
    """Construct an opaque fixed Clifford target without CNOT-by-CNOT rebasing."""
    if matrix.shape!=(4,4) or matrix.free_symbols:return None
    if basis.entangler not in {"CX","CNOT","CZ","ECR","XX90","ZZMax","iSWAP"}:return None
    action=pauli_action(matrix)
    if action is None:return None
    solution=coset_search(basis.entangler)
    key=local_coset_key(_columns(action))
    if key not in solution["states"]:return None
    names=group_data()[0].representatives
    blocks=[]
    current=key
    while current in solution["predecessors"]:
        old,a,b=solution["predecessors"][current]
        blocks.append(((names[a],(0,)),(names[b],(1,)),(basis.entangler,(0,1))))
        current=old
    word=tuple(op for block in reversed(blocks) for op in block if op[0]!="I")
    residual=(matrix*gates.compile_gates(2,word).H).applyfunc(sp.simplify)
    pair=product_factors(residual,(0,))
    if pair is None:raise ArithmeticError("Pauli local coset failed exact OSR factorization")
    left,right=(recognize_local(factor,()) for factor in pair)
    if left is None or right is None:raise ArithmeticError("Clifford local factors were not recognized")
    word+=left[0]+tuple((*op[:-1],(1,)) for op in right[0])
    factor=sp.simplify(left[1]*right[1])
    return dict(sequence=word,factor=factor,method="native_Clifford_coset_search_with_OSR_local_recovery",
                discrete_minimum=solution["distances"][key],
                reachable_cosets=solution["reachable_cosets"],proof_scope=solution["proof_scope"])
