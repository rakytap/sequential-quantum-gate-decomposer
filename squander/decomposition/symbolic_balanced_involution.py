"""CNOT-cost classification of controlled balanced two-qubit involutions.

For V=V^dagger, V^2=I and tr(V)=0, the minimum is 4 at OSR 4,
3 at OSR 2, and 1/2 for one-/two-wire local products. With unrestricted
two-qubit gates the respective costs are 3,3,1/2. Ancillas are excluded.
The manuscript proves the normal form and low-rank network exclusions;
this checker verifies their polynomial components and exact witnesses.
"""
from functools import lru_cache
import sympy as sp
if __package__:
    from . import symbolic_gates as gates
    from .symbolic_controlled_chart import _zero
    from .symbolic_osr import schmidt_certificate
else:
    import symbolic_gates as gates
    from symbolic_controlled_chart import _zero
    from symbolic_osr import schmidt_certificate


def normal_form(alpha,beta):
    Z=gates.make_gate('Z')
    A=gates.make_gate('Ry',alpha)*Z*gates.make_gate('Ry',-alpha)
    B=gates.make_gate('Ry',beta)*Z*gates.make_gate('Ry',-beta)
    K=gates.make_gate('CX')
    return sp.ImmutableMatrix(K*sp.kronecker_product(B,A)*K)


def _cz(control,target):
    return (('H',(target,)),('CX',(control,target)),('H',(target,)))


def controlled_normal_form_source(alpha,beta):
    """Literal four-CX word for the whole real two-angle normal form."""
    word=[('CX',(1,2))]
    for angle,wire in ((alpha,1),(beta,2)):
        word.extend((('Ry',-angle,(wire,)),*_cz(0,wire),('Ry',angle,(wire,))))
    word.append(('CX',(1,2)))
    return tuple(word)


def boundary_source(edge,angle):
    """Three-CX words for the four fundamental rank-two boundary families."""
    if edge=='alpha_zero':
        before=(('CX',(1,2)),('H',(2,)))
        after=(('H',(2,)),('CX',(1,2)))
        axis,delta,wire='Ry',sp.pi/2-angle,2
    elif edge=='alpha_half_pi':
        before=(('CX',(1,2)),('S',(2,)),('H',(1,)))
        after=(('H',(1,)),('Sdg',(2,)),('CX',(1,2)))
        axis,delta,wire='Rx',angle-sp.pi/2,1
    elif edge=='beta_zero':
        before=(('CX',(2,1)),('Sdg',(1,)))
        after=(('S',(1,)),('CX',(2,1)))
        axis,delta,wire='Rx',-angle,2
    elif edge=='beta_half_pi':
        before=(('CX',(1,2)),('H',(1,)))
        after=(('H',(1,)),('CX',(1,2)))
        axis,delta,wire='Ry',sp.pi/2-angle,1
    else:raise ValueError('Unknown normal-form edge')
    return after+((axis,-delta,(wire,)),)+_cz(0,wire)+((axis,delta,(wire,)),)+before


@lru_cache(maxsize=1)
def verify_normal_form_polynomials():
    """Coefficients of V^2: norm, Tb, T^t a, and ab^t-cof(T)."""
    P=[gates.make_gate(n) for n in 'XYZ'];I=sp.eye(2)
    a=sp.Matrix(sp.symbols('balanced_a:3',real=True))
    b=sp.Matrix(sp.symbols('balanced_b:3',real=True))
    T=sp.Matrix(3,3,sp.symbols('balanced_T:9',real=True))
    V=sum((a[i]*sp.kronecker_product(P[i],I)+b[i]*sp.kronecker_product(I,P[i])
           for i in range(3)),sp.zeros(4))
    V+=sum((T[i,j]*sp.kronecker_product(P[i],P[j]) for i in range(3) for j in range(3)),sp.zeros(4))
    square=sp.expand(V*V)
    if sp.expand(sp.trace(square)/4-a.dot(a)-b.dot(b)-sum(t*t for t in T))!=0:return False
    cofactors=T.cofactor_matrix()
    for i in range(3):
        if sp.expand(sp.trace(square*sp.kronecker_product(P[i],I))/4-2*(T*b)[i])!=0:return False
        if sp.expand(sp.trace(square*sp.kronecker_product(I,P[i]))/4-2*(T.T*a)[i])!=0:return False
        for j in range(3):
            expected=2*(a[i]*b[j]-cofactors[i,j])
            if sp.expand(sp.trace(square*sp.kronecker_product(P[i],P[j]))/4-expected)!=0:return False
    A,B,C,D=sp.symbols('balanced_A balanced_B balanced_C balanced_D',real=True)
    W=A*sp.kronecker_product(P[0],I)+B*sp.kronecker_product(I,P[2])+C*sp.kronecker_product(P[1],P[1])+D*sp.kronecker_product(P[2],P[0])
    residual=sp.expand(W*W-(A*A+B*B+C*C+D*D)*sp.eye(4))
    return residual==2*(A*B+C*D)*sp.kronecker_product(P[0],P[2])


def balanced_controlled_cost(V):
    """Exact fixed-matrix certificate, independent of synthesis failures."""
    V=sp.ImmutableMatrix(V)
    if V.shape!=(4,4) or V.free_symbols or V.atoms(sp.Float):return None
    if not _zero(V-V.H) or not _zero(V*V-sp.eye(4)) or sp.simplify(sp.trace(V))!=0:return None
    rank=schmidt_certificate(V,(0,)).fixed_rank
    if rank not in (1,2,4):return None
    if rank==4:minimum,arbitrary=4,3
    elif rank==2:minimum,arbitrary=3,3
    else:
        partial_a=sp.Matrix(2,2,lambda i,j:sum(V[i+2*k,j+2*k] for k in range(2)))
        partial_b=sp.Matrix(2,2,lambda i,j:sum(V[2*i+k,2*j+k] for k in range(2)))
        minimum=2 if _zero(partial_a) and _zero(partial_b) else 1
        arbitrary=minimum
    if not verify_normal_form_polynomials():raise ArithmeticError('Balanced-involution polynomial lemma failed')
    return dict(method='controlled_balanced_involution_normal_form_classification',
                minimum_cx=minimum,minimum_unrestricted_two_qubit=arbitrary,relative_osr=rank,
                hypotheses='V Hermitian, V^2=I, trace(V)=0; arbitrary one-qubit unitaries; no ancillas; all-to-all',
                scope='global_entangler_count; no local-cost or general-controlled-V classification')
