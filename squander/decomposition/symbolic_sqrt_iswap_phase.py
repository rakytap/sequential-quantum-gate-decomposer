"""All-real controlled-phase chart with two SQiSW pulses.

The circuit is certified by a polynomial identity and explicit real charts,
including q=0 and q=+/-pi. No floating KAK or sampled acceptance is used.

The CP pulse pattern follows the analytical SQiSW construction also used
by Cirq's two_qubit_to_sqrt_iswap module. Here every constant is exact and
the sign/endpoint charts are proved, without floating Mod or epsilons.
Reference: https://arxiv.org/abs/2105.06074
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost, encode_sequence, validate_operations
    from .symbolic_identity import prove_circuit_identity
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost, encode_sequence, validate_operations
    from symbolic_identity import prove_circuit_identity


def _u3_word(q, phi, xi):
    return (("U3",xi,sp.pi/2,sp.pi/2+q/2,(0,)),
            ("U3",-sp.pi/2,-sp.pi/2,sp.pi/2+q/2,(1,)),
            ("SiSWAP",(0,1)),
            ("U3",-2*phi,-sp.pi/2,3*sp.pi/2,(0,)),
            ("SiSWAP",(0,1)),
            ("U3",xi,-sp.pi/2,sp.pi/2,(0,)),
            ("U3",sp.pi/2,-sp.pi/2,sp.pi/2,(1,)))


def _phase_in_u3(operation, angle):
    _,theta,phi,lam,wire=operation
    return (("U3",sp.pi-theta,0,lam-sp.pi,wire),
            ("U3",sp.pi,angle+phi,sp.pi+angle,wire))


def _phased_xz(operation):
    _,theta,phi,lam,wire=operation
    return ("PhasedXZ",theta/sp.pi,(phi+lam)/sp.pi,
            sp.Rational(1,2)-lam/sp.pi,wire)


def _phase_in_phased_xz(operation, angle):
    _,x,z,a,wire=operation
    theta,phi,lam=sp.pi*x,sp.pi*(z+a)-sp.pi/2,sp.pi/2-sp.pi*a
    # The two replacement U3 angles sum to 2*pi-theta. Include their
    # conversion scalars and U3(theta,...)=exp(-i*theta/2)*PhasedXZ.
    gamma=angle+theta-sp.pi
    return tuple(_phased_xz(op) for op in _phase_in_u3(("U3",theta,phi,lam,wire),gamma))


@lru_cache(maxsize=1)
def verify_phase_chart_lemma():
    u,v,w,A,B=sp.symbols("sq_u sq_v sq_w sq_A sq_B",real=True)
    rx=sp.Matrix([[A,-sp.I*B],[-sp.I*B,A]])
    rz=sp.diag(u-sp.I*v,u+sp.I*v)
    middle=sp.Matrix([[w,sp.I*sp.sqrt(2)*v],[sp.I*sp.sqrt(2)*v,w]])
    source=((rz,(0,)),(rz,(1,)),(rx,(0,)),
            (gates.make_gate("Rx",-sp.pi/2),(1,)),
            (gates.make_gate("Z"),(0,)),(gates.make_gate("SiSWAP"),(0,1)),
            (gates.make_gate("Z"),(0,)),(middle,(0,)),
            (gates.make_gate("SiSWAP"),(0,1)),(rx,(0,)),
            (gates.make_gate("Rx",sp.pi/2),(1,)))
    matrix=sp.eye(4)
    for operation,wires in source:
        matrix=gates.apply_to(matrix,2,operation,wires).applyfunc(sp.expand)
    relations=(u*u+v*v-1,w*w-u*u+v*v,
               2*u*A*A-u-w,2*u*B*B-u+w,2*u*A*B-v)
    ideal=sp.groebner(relations,A,B,w,u,v,extension=sp.I)
    expected=sp.diag(u-sp.I*v,u-sp.I*v,u-sp.I*v,(u+sp.I*v)**3)
    if any(ideal.reduce(sp.expand(value))[1]!=0 for value in matrix-expected):return False
    # Explicit atan2 charts: q in [-pi,pi] gives u>=sqrt(2)/2,
    # w>=0. Thus u+w>0; neither atan2 pair is (0,0).
    # A=(u+w)/d, B=v/d, d^2=(u+w)^2+v^2=2u(u+w).
    circle=sp.groebner((u*u+v*v-1,w*w-u*u+v*v),w,u,v)
    denominator=(u+w)**2+v*v
    identities=(denominator-2*u*(u+w),
                2*u*(u+w)**2-(u+w)*denominator,
                2*u*v*v-(u-w)*denominator,
                2*u*(u+w)*v-v*denominator,
                w*w+2*v*v-1)
    if any(circle.reduce(sp.expand(value))[1]!=0 for value in identities):return False
    q,phi,xi,gamma,t,p,l=sp.symbols("sq_q sq_phi sq_xi sq_gamma sq_t sq_p sq_l",real=True)
    raw=(("Rz",q/2,(0,)),("Rz",q/2,(1,)),("Rx",xi,(0,)),
         ("Rx",-sp.pi/2,(1,)),("Z",(0,)),("SiSWAP",(0,1)),("Z",(0,)),
         ("Rx",-2*phi,(0,)),("SiSWAP",(0,1)),("Rx",xi,(0,)),("Rx",sp.pi/2,(1,)))
    if prove_circuit_identity(gates.compile_gates(2,raw),_u3_word(q,phi,xi),sp.exp(-sp.I*q/2)) is not True:return False
    op=("U3",t,p,l,(0,))
    if prove_circuit_identity(sp.exp(sp.I*gamma)*gates.make_gate("U3",t,p,l),_phase_in_u3(op,gamma)) is not True:return False
    x,z,a=sp.symbols("sq_x sq_z sq_a",real=True)
    pxz=("PhasedXZ",x,z,a,(0,))
    if prove_circuit_identity(sp.exp(sp.I*gamma)*gates.make_gate("PhasedXZ",x,z,a),_phase_in_phased_xz(pxz,gamma)) is not True:return False
    if prove_circuit_identity(gates.make_gate("U3",t,p,l),(_phased_xz(op),),sp.exp(-sp.I*t/2)) is not True:return False
    return True


def phase_chart(theta):
    theta=sp.sympify(theta)
    if theta.is_real is not True:raise ValueError("Controlled-phase angle must be real")
    return _real_phase_chart(theta)


def _real_phase_chart(theta):
    """Internal chart for an angle whose real domain was proved by a lemma."""
    q=sp.atan2(sp.sin(theta),sp.cos(theta))
    u,v=sp.cos(q/4),sp.sin(q/4)
    w=sp.sqrt(sp.cos(q/2))
    phi=sp.atan2(sp.sqrt(2)*v,w)
    xi=2*sp.atan2(v,u+w)
    return q,phi,xi


def phase_word(theta,basis,*,phase_exact):
    """Return a chart-certified native word and its projective scalar angle."""
    if basis.entangler!="SiSWAP":return None
    if not ({"U3","PhasedXZ"}&set(basis.single_qubit_gates)):return None
    if not verify_phase_chart_lemma():raise ArithmeticError("SQiSW phase chart lemma failed")
    q,phi,xi=phase_chart(theta)
    word=_u3_word(q,phi,xi)
    angle=-q/4
    if "PhasedXZ" in basis.single_qubit_gates:
        angle-=sum((op[1]/2 for op in word if op[0]=="U3"),sp.S.Zero)
        word=tuple(_phased_xz(op) if op[0]=="U3" else op for op in word)
        close=_phase_in_phased_xz
    else:close=_phase_in_u3
    if phase_exact and angle!=0:
        word=close(word[0],angle)+word[1:]
        angle=sp.S.Zero
    return validate_operations(word,basis,2),angle


def synthesize_controlled_phase(theta,basis,*,phase_exact):
    value=phase_word(theta,basis,phase_exact=phase_exact)
    if value is None:return None
    word,angle=value
    q,phi,xi=phase_chart(theta)
    return dict(status="verified_symbolic",phase_exact=phase_exact,
        operations=encode_sequence(word),cost=circuit_cost(word).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*angle)),physical=False,cost=0),
        verification="exact_SQiSW_controlled_phase_chart",
        construction_method="two_SQiSW_real_controlled_phase_chart",
        controlled_phase_certificate=dict(theta=str(theta),principal_angle=str(q),
            phi=str(phi),xi=str(xi),domain="all real theta; q in [-pi,pi]; cos(q/4)>0",
            lemma="polynomial_CP_identity_and_nonzero_atan2_charts"),
        entangler_optimal=False,local_cost_optimal=False,
        conditional_entangler_optimality=dict(count=2,condition="Ne(sin(("+str(theta)+")/2),0)",
            source="OSR(target)=2, OSR(SQiSW)=4; local equivalence preserves OSR"),
        optimality_scope="two entanglers generically; local resource score is an upper bound")


def verify_controlled_phase_certificate(target,result,basis):
    if result.get("verification")!="exact_SQiSW_controlled_phase_chart":return False
    target=sp.ImmutableMatrix(target)
    symbols={str(value):value for value in target.free_symbols}
    try:
        theta=sp.sympify(result["controlled_phase_certificate"]["theta"],locals=symbols)
        if target!=gates.make_gate("CP",theta):return False
        expected=synthesize_controlled_phase(theta,basis,phase_exact=result["phase_exact"])
    except (KeyError,ValueError,TypeError):return False
    return all(result.get(key)==expected.get(key) for key in
               ("status","operations","cost","virtual_gp","controlled_phase_certificate"))
