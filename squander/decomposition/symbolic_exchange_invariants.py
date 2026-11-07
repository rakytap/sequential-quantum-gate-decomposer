"""Exact two-pulse obstructions for square-root exchange primitives.

The SQiSW test uses scalar-invariant inequalities, avoiding numerical Weyl
canonicalization. The sqrt-SWAP test uses its rank-one reflection spectrum.
These are lower bounds only, not symbolic three-pulse constructions.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_entangler_proof import spin_data, _zero, _nonzero_constant
    from .symbolic_minimality import _simplify
else:
    import symbolic_gates as gates
    from symbolic_entangler_proof import spin_data, _zero, _nonzero_constant
    from symbolic_minimality import _simplify


@lru_cache(maxsize=1)
def verify_exchange_obstruction_lemmas():
    # General SU(2) middle gates, with real quaternion coordinates.
    a,b,c,d,e,f,h,k = sp.symbols('exchange_a exchange_b exchange_c exchange_d '
                                'exchange_e exchange_f exchange_h exchange_k', real=True)
    I = sp.eye(2)
    X,Y,Z = (gates.make_gate(name) for name in 'XYZ')
    first = a*I+sp.I*(b*X+c*Y+d*Z)
    second = e*I+sp.I*(f*X+h*Y+k*Z)
    Q = sp.Matrix([[1,0,0,sp.I],[0,sp.I,1,0],
                   [0,sp.I,-1,0],[1,0,0,-sp.I]])/sp.sqrt(2)
    S = gates.make_gate('SiSWAP')
    V = (Q.H*S*sp.kronecker_product(first,second)*S*Q).applyfunc(sp.expand)
    C = V.applyfunc(sp.im).det(method='domain-ge')
    difference = ((a*a+d*d)*(f*f+h*h)-(b*b+c*c)*(e*e+k*k))
    if sp.expand(C-difference**2/4) != 0:
        return False
    # Re tr(V^T V)/4 is itself a sum of squares. For an SU(4)
    # unitary it is A-C, where A=det Re(V), C=det Im(V).
    square_sum = (a*k-d*e)**2+(b*h-c*f)**2
    if sp.expand(sp.re(sp.trace(V.T*V))/4-square_sum) != 0:
        return False
    # Prove the A,C relation and A+C=(g2+1)/4 on a free Cartan
    # diagonal. KAK transports them through determinant-one real frames.
    x,y,z = sp.symbols('exchange_x exchange_y exchange_z', real=True)
    phases = (x-y+z,x+y-z,-x-y-z,-x+y+z)
    A = sp.prod(sp.cos(p) for p in phases)
    C = sp.prod(sp.sin(p) for p in phases)
    real_trace = sp.prod(sp.cos(2*p) for p in (x,y,z))
    g2 = sum(sp.cos(4*p) for p in (x,y,z))
    if not _zero(A-C-real_trace) or not _zero(A+C-(g2+1)/4):
        return False
    # Since g1=(a0+i*b0)^2, a0^2=(Abs(g1)+Re(g1))/2.
    # A,C>=0 implies g2+1>=0 and (g2+1)^2>=16*a0^2.
    a0,b0 = sp.symbols('exchange_real exchange_imaginary', real=True)
    if sp.expand(sp.re((a0+sp.I*b0)**2)-(a0*a0-b0*b0)) != 0:
        return False
    # For sqrt-SWAP, D^2=e^(+/-i*pi/4) J, J=diag(1,1,1,-1).
    # J O^T J O is a product of two real hyperplane reflections.
    vector = sp.Matrix(sp.symbols('exchange_v:4', real=True))
    J = sp.diag(1,1,1,-1)
    product = J*(sp.eye(4)-2*vector*vector.T)
    rho = vector[3]**2
    variables = tuple(vector)
    ideal = sp.groebner((sum(p*p for p in vector)-1,), *variables)
    for polynomial in (sp.trace(product)-4*rho,
                       sp.trace(product*product)-4*(2*rho-1)**2):
        if ideal.reduce(sp.expand(polynomial))[1] != 0:
            return False
    # Consequently g1=-rho^2, g2=1-4rho, eliminating rho gives
    # 16*g1+(g2-1)^2=0 for TWO pulses, independently of local frames.
    if sp.expand(16*(-rho*rho)+((1-4*rho)-1)**2) != 0:
        return False
    return True


def exchange_three_pulse_lower_bound(target, native):
    target = sp.ImmutableMatrix(target)
    if target.shape != (4,4) or target.free_symbols or target.atoms(sp.Float):
        return None
    if native not in {'SiSWAP','SSWAP'}:
        return None
    if not _zero(target.H*target-sp.eye(4)):
        raise ValueError('Target unitarity not proved')
    if not verify_exchange_obstruction_lemmas():
        raise ArithmeticError('Exchange obstruction lemma failed')
    g1,g2 = spin_data(target)[2]
    evidence = None
    if native == 'SiSWAP':
        W = _simplify(sp.re(g2)+1)
        real = _simplify(sp.re(g1))
        magnitude = _simplify(sp.sqrt(sp.simplify(g1*sp.conjugate(g1))))
        gap = _simplify(W*W-8*(magnitude+real))
        if W.is_negative is True or gap.is_negative is True:
            evidence = dict(method='SQiSW_two_pulse_real_imaginary_determinants_nonnegative',
                            W=str(W), squared_gap=str(gap),
                            excluded='W_negative' if W.is_negative is True else 'squared_gap_negative',
                            necessary_conditions='W>=0 and W**2>=8*(Abs(g1)+Re(g1))')
        # Local and single-SQiSW classes have respectively (g1,g2)=(1,3)
        # and (1/4,1); both obey the two inequalities, so violation also
        # excludes the zero/one cases. No cancellability assumption.
    else:
        local_excluded = any(_nonzero_constant(a-b) for a,b in zip((g1,g2),(1,3)))
        one_excluded = any(_nonzero_constant(a-b) for a,b in zip((g1,g2),(-sp.I/4,0)))
        relation = _simplify(16*g1+(g2-1)**2)
        if local_excluded and one_excluded and _nonzero_constant(relation):
            evidence = dict(method='sqrt_SWAP_two_pulse_reflection_invariant',
                            nonzero_relation=str(relation),
                            necessary_relation='16*g1+(g2-1)**2=0',
                            zero_and_one_pulse_classes_separately_excluded=True)
    if evidence is None:
        return None
    return dict(lower_bound=3, native=native, target_invariants=list(map(str,(g1,g2))),
                evidence=evidence, local_resource_scope='arbitrary U(2) locals and scalar phase',
                construction_claim=False, local_cost_claim=False,
                reference='https://arxiv.org/abs/2105.06074' if native=='SiSWAP'
                          else 'https://arxiv.org/abs/quant-ph/0002045')
