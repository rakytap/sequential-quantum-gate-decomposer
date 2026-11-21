"""Universal exact two-B Cartan chart, with explicit boundary branches.

The outer frames are recovered analytically, rather than numerically KAK
decomposing the pulse skeleton. A six-component Pauli identity and unit-circle
relations certify the complete matrix. Local Y/Z frames handle arbitrary real
angles outside the chart; physical phase closure needs no extra entanglers.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from .symbolic_controlled_chart import _zero
    from .symbolic_identity import prove_circuit_identity,exponential_forms,zero_form
    from .symbolic_native_pauli import bell_coordinates,diagonal_gauge,_phase
    from .symbolic_native_fixed import verify_native_lowering_lemmas
    from .symbolic_sqrt_iswap_phase import _phase_in_u3,_phased_xz,_phase_in_phased_xz
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from symbolic_controlled_chart import _zero
    from symbolic_identity import prove_circuit_identity,exponential_forms,zero_form
    from symbolic_native_pauli import bell_coordinates,diagonal_gauge,_phase
    from symbolic_native_fixed import verify_native_lowering_lemmas
    from symbolic_sqrt_iswap_phase import _phase_in_u3,_phased_xz,_phase_in_phased_xz


def _word(x,pm,pp,a,beta):
    chi=-pm-pp-sp.pi
    return (("U3",sp.pi/2,chi,0,(0,)),
            ("U3",sp.pi+pm-pp,-sp.pi/2,-sp.pi/2,(1,)),("B",(0,1)),
            ("U3",-2*x,0,0,(0,)),("U3",-a,-beta,-beta,(1,)),("B",(0,1)),
            ("U3",-sp.pi/2,0,chi-sp.pi,(0,)),
            ("U3",sp.pi-pm+pp,-sp.pi/2,sp.pi/2,(1,)))


def _pauli_coefficients(u,v,w,Cm,Sm,Cp,Sp):
    root=sp.sqrt(2)
    return {(0,0):(root*Cm*(u-v)+root*Cp*(u+v)-2*w*(Sm+Sp))/4,
        (0,1):-sp.I*(2*w*(Cm+Cp)+root*Sm*(u-v)+root*Sp*(u+v))/4,
        (1,0):sp.I*(2*w*(Cm-Cp)+root*Sm*(u-v)-root*Sp*(u+v))/4,
        (1,1):-(root*Cm*(u-v)-root*Cp*(u+v)-2*w*(Sm-Sp))/4,
        (2,2):sp.I*u/root,(3,3):sp.I*v/root}


@lru_cache(maxsize=1)
def verify_b_cartan_lemma():
    if not verify_native_lowering_lemmas():return False
    pm,pp,a,beta,x=sp.symbols("b_pm b_pp b_a b_beta b_x",real=True)
    u,v,w=sp.cos(a/2)*sp.cos(beta),sp.sin(a/2),sp.cos(a/2)*sp.sin(beta)
    axes=(sp.eye(2),*(gates.make_gate(name) for name in "XYZ"))
    coefficients=_pauli_coefficients(u,v,w,sp.cos(2*pm),sp.sin(2*pm),sp.cos(2*pp),sp.sin(2*pp))
    target=sum((coefficient*sp.kronecker_product(axes[i],axes[j])
                for (i,j),coefficient in coefficients.items()),sp.zeros(4))
    factor=-sp.exp(sp.I*(beta+pm+pp+sp.pi))
    if prove_circuit_identity(target,_word(0,pm,pp,a,beta),factor) is not True:return False
    # The x degree of freedom is transported by B from local Y_0 to
    # -Z_0 X_1; the fixed outer frame maps this to +X_0 X_1.
    B=gates.make_gate("B")
    y0=sp.kronecker_product(sp.eye(2),gates.make_gate("Y"))
    zx=sp.kronecker_product(gates.make_gate("X"),gates.make_gate("Z"))
    if not _zero(B*y0*B.H+zx):return False
    chi=-pm-pp-sp.pi
    left=(("Rz",chi-sp.pi,(0,)),("Ry",-sp.pi/2,(0,)),
          ("Rx",sp.pi-pm+pp,(1,)))
    L=gates.compile_gates(2,left)
    if not _zero(L*zx*L.H+sp.kronecker_product(gates.make_gate("X"),gates.make_gate("X"))):return False
    # Check the phase-normalized spin-flip invariants used by the lower
    # bound against the actual full Cartan matrix. Native B has I1=I2=0;
    # local products have I1=16. Unequal invariants exclude those classes.
    y,z=sp.symbols("b_y b_z",real=True)
    kernel=gates.make_gate("Canonical",x,y,z)
    yy=sp.kronecker_product(gates.make_gate("Y"),gates.make_gate("Y"))
    spin=kernel*yy*kernel.T*yy
    expected1=16*(sp.cos(2*x)*sp.cos(2*y)*sp.cos(2*z)+sp.I*sp.sin(2*x)*sp.sin(2*y)*sp.sin(2*z))**2
    expected2=4*(sp.cos(4*x)*sp.cos(4*y)*sp.cos(4*z)+sp.I*sp.sin(4*x)*sp.sin(4*y)*sp.sin(4*z))
    forms,inverse=exponential_forms((sp.trace(spin)**2-expected1,sp.trace(spin*spin)-expected2))
    if any(not zero_form(value,tuple(inverse)) for value in forms):return False
    native=B*yy*B.T*yy
    if not _zero((sp.trace(native)**2,sp.trace(native*native))):return False
    # On the regular chart Ca,Cb>0. The numerators of all six Pauli
    # coefficients reduce to the target, modulo these exact circle relations.
    cy,sy,cz,sz,r=sp.symbols("b_cy b_sy b_cz b_sz b_r",real=True)
    Ca,Cb=cy*cz-sy*sz,cy*cz+sy*sz
    Sa,Sb=sy*cz+cy*sz,sy*cz-cy*sz
    values=(sp.sqrt(2)*sy*cz,sp.sqrt(2)*cy*sz,r,Sb/Ca,-r/Ca,Sa/Cb,-r/Cb)
    expected={(0,0):cy*cz,(1,1):sy*sz,(2,2):sp.I*sy*cz,(3,3):sp.I*cy*sz}
    ideal=sp.groebner((cy*cy+sy*sy-1,cz*cz+sz*sz-1,
        r*r-(cy*cy-sy*sy)*(cz*cz-sz*sz)),r,cy,sy,cz,sz,extension=[sp.I,sp.sqrt(2)])
    for key,value in _pauli_coefficients(*values).items():
        numerator=sp.cancel(value-expected.get(key,0)).as_numer_denom()[0]
        if ideal.reduce(sp.expand(numerator))[1]!=0:return False
    # The only Ca=0 or Cb=0 points in the folded square are its four
    # corners. The undefined axis is arbitrary because its block is scalar.
    for sy_sign in (-1,1):
        for sz_sign in (-1,1):
            y,z=sy_sign*sp.pi/4,sz_sign*sp.pi/4
            cm=0 if sp.cos(y+z)==0 else sp.sin(y-z)/sp.cos(y+z)
            cp=0 if sp.cos(y-z)==0 else sp.sin(y+z)/sp.cos(y-z)
            # An absent axis uses angle zero, hence cosine one.
            if sp.cos(y+z)==0:cm=1
            if sp.cos(y-z)==0:cp=1
            actual=_pauli_coefficients(sp.sqrt(2)*sp.sin(y)*sp.cos(z),
                sp.sqrt(2)*sp.cos(y)*sp.sin(z),0,cm,0,cp,0)
            wanted={(0,0):sp.cos(y)*sp.cos(z),(1,1):sp.sin(y)*sp.sin(z),
                    (2,2):sp.I*sp.sin(y)*sp.cos(z),(3,3):sp.I*sp.cos(y)*sp.sin(z)}
            if any(sp.simplify(value-wanted.get(key,0))!=0 for key,value in actual.items()):return False
    # Native phase/folding transformations are exact for either parity.
    t,p,l=sp.symbols("b_t b_p b_l",real=True)
    original=gates.make_gate("U3",t,p,l)
    changed=(("U3",sp.pi-t,sp.pi-p,l+sp.pi,(0,)),)
    if prove_circuit_identity(gates.make_gate("Y")*original,changed,sp.exp(sp.I*(p-sp.pi/2))) is not True:return False
    for axis in ("Y","Z"):
        k=sp.Symbol("b_k",integer=True)
        for parity in (0,1):
            n=2*k+parity
            target=gates.make_gate("Ryy" if axis=="Y" else "Rzz",-sp.pi*n)
            product=sp.kronecker_product(gates.make_gate(axis),gates.make_gate(axis)) if parity else sp.eye(4)
            if any(sp.simplify(sp.expand_complex(value))!=0 for value in target-sp.I**n*product):return False
    return True


def _angles(y,z):
    n,m=sp.floor(2*y/sp.pi+sp.Rational(1,2)),sp.floor(2*z/sp.pi+sp.Rational(1,2))
    y,z=y-sp.pi*n/2,z-sp.pi*m/2
    w=sp.sqrt(sp.cos(2*y)*sp.cos(2*z))
    u,v=sp.sqrt(2)*sp.sin(y)*sp.cos(z),sp.sqrt(2)*sp.cos(y)*sp.sin(z)
    minus=sp.Piecewise((0,sp.Eq(sp.cos(y+z),0)),(sp.atan2(-w,sp.sin(y-z)),True))
    plus=sp.Piecewise((0,sp.Eq(sp.cos(y-z),0)),(sp.atan2(-w,sp.sin(y+z)),True))
    radius=sp.sqrt(u*u+w*w)
    beta=sp.Piecewise((0,sp.Eq(radius,0)),(sp.atan2(w,u),True))
    a=2*sp.atan2(v,radius)
    return minus/2,plus/2,a,beta,n,m


def synthesize_b_cartan(target,basis,*,phase_exact):
    if basis.entangler!="B" or not ({"U3","U","PhasedXZ"}&set(basis.single_qubit_gates)):return None
    target=sp.ImmutableMatrix(target)
    before=after=()
    decomposition=bell_coordinates(target)
    if decomposition is None:
        gauged=diagonal_gauge(target)
        if gauged is None:return None
        decomposition,before,after=gauged
    coordinates,scalar=decomposition
    if any(value.is_real is not True for value in coordinates):return None
    width=target.rows.bit_length()-1
    if width!=2:return None
    kernel=gates.make_gate("Canonical",*coordinates)
    left=gates.compile_gates(2,after);right=gates.compile_gates(2,before)
    if not _zero(left*target*right-scalar*kernel):return None
    if not verify_b_cartan_lemma():raise ArithmeticError("Universal two-B chart lemma failed")
    x,y,z=coordinates
    pm,pp,a,beta,n,m=_angles(y,z)
    operations=list(_word(x,pm,pp,a,beta))
    gamma=sp.pi+beta+pm+pp+sp.pi+_phase(scalar)
    p,q=sp.Mod(n,2),sp.Mod(m,2)
    # i^(n+m) Y^p Z^q on each output wire restores the folded coordinates.
    gamma+=sp.pi*(n+m)/2
    for index in (-2,-1):
        op=operations[index];theta,phi,lam=op[1:-1]
        operations[index]=("U3",(1-2*p)*theta+p*sp.pi,
            (1-2*p)*phi+p*sp.pi+q*sp.pi,lam+p*sp.pi,op[-1])
        gamma+=p*(phi-sp.pi/2)
    for op in before:
        index=op[-1][0]
        item=operations[index]
        operations[index]=(*item[:3],item[3]-op[1],item[-1])
        gamma+=op[1]/2
    for op in after:
        index=-2+op[-1][0]
        item=operations[index]
        operations[index]=(item[0],item[1],item[2]-op[1],item[3],item[-1])
        gamma+=op[1]/2
    if "PhasedXZ" in basis.single_qubit_gates:
        gamma-=sum((op[1]/2 for op in operations if op[0]=="U3"),sp.S.Zero)
        operations=[_phased_xz(op) if op[0]=="U3" else op for op in operations]
        close=_phase_in_phased_xz
    else:close=_phase_in_u3
    if phase_exact:
        operations=list(close(operations[0],gamma))+operations[1:]
    operations=validate_operations(operations,basis,2)
    c2=[sp.cos(2*value) for value in coordinates];s2=[sp.sin(2*value) for value in coordinates]
    I1=16*(sp.prod(c2)+sp.I*sp.prod(s2))**2
    I2=4*(sp.prod(sp.cos(4*value) for value in coordinates)+sp.I*sp.prod(sp.sin(4*value) for value in coordinates))
    condition=sp.And(sp.Ne(I1,16),sp.Or(sp.Ne(I1,0),sp.Ne(I2,0)))
    result=dict(status="verified_symbolic",phase_exact=phase_exact,
        operations=encode_sequence(operations),cost=circuit_cost(operations).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*gamma)),physical=False,cost=0),
        verification="exact_two_B_Cartan_chart_and_native_lowering",
        construction_method="universal_two_B_with_exact_outer_frames",
        b_cartan_certificate=dict(coordinates=[str(value) for value in coordinates],scalar=str(scalar),
            before=encode_sequence(before),after=encode_sequence(after),
            folds=[str(n),str(m)],fold_interval="[-pi/4,pi/4)",
            component_proof="six_Pauli_components_unit_circle_reduction_and_corner_charts"),
        entangler_optimal=condition is sp.S.true,local_cost_optimal=False,
        optimality_scope="global entangler minimum subject to the stated nonlocal/non-native condition; local score is an upper bound")
    if condition is not sp.S.false:
        result["conditional_entangler_optimality"]=dict(count=2,condition=str(condition),
            source="complete_Makhlin_local_invariants_and_universal_two_B_construction",
            local_condition=str(sp.Eq(I1,16)),one_B_condition=str(sp.And(sp.Eq(I1,0),sp.Eq(I2,0))),
            references=["https://arxiv.org/abs/quant-ph/0002045","https://arxiv.org/abs/quant-ph/0312193"])
    return result


def verify_b_cartan_certificate(target,result,basis):
    if result.get("verification")!="exact_two_B_Cartan_chart_and_native_lowering":return False
    if not isinstance(result.get("phase_exact"),bool):return False
    try:expected=synthesize_b_cartan(target,basis,phase_exact=result["phase_exact"])
    except (ValueError,KeyError,TypeError):return False
    return expected is not None and all(result.get(key)==expected.get(key) for key in
        ("status","operations","cost","virtual_gp","b_cartan_certificate","entangler_optimal",
         "conditional_entangler_optimality","local_cost_optimal"))
