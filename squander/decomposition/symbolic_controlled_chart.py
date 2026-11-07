"""Exact OSR-two constructions using a real SU(2) axis chart.

A controlled pair of U(2) blocks is reduced to one controlled phase. The
two native pulses and each native one-wire lowering have reusable exact
lemmas. Degenerate quaternion and polar charts are explicit Piecewise
branches, not divisions by a vanishing matrix entry.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from .symbolic_identity import exponential_forms,zero_form,prove_circuit_identity
    from .symbolic_native_pauli import _phase
    from .symbolic_rotation_vector import verify_rotation_vector_lemma
    from .symbolic_sqrt_iswap_phase import (_u3_word,_phase_in_u3,
        _phased_xz,verify_phase_chart_lemma)
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from symbolic_identity import exponential_forms,zero_form,prove_circuit_identity
    from symbolic_native_pauli import _phase
    from symbolic_rotation_vector import verify_rotation_vector_lemma
    from symbolic_sqrt_iswap_phase import (_u3_word,_phase_in_u3,
        _phased_xz,verify_phase_chart_lemma)


def _zero(values):
    forms,inverse=exponential_forms(tuple(values))
    return all(zero_form(value,tuple(inverse)) for value in forms)


def _parts(matrix):
    """Prove U=exp(i*phase)*[[a,-b*],[b,a*]], |a|²+|b|²=1."""
    phase=_phase(matrix.det())
    if phase is None:return None
    phase=phase/2
    normalized=matrix*sp.exp(-sp.I*phase)
    a,b=normalized[0,0],normalized[1,0]
    if not _zero((normalized[0,1]+sp.conjugate(b),normalized[1,1]-sp.conjugate(a),
                  a*sp.conjugate(a)+b*sp.conjugate(b)-1)):return None
    ar,ai,br,bi=(sp.simplify(sp.expand_complex(value)) for value in (sp.re(a),sp.im(a),sp.re(b),sp.im(b)))
    return (ar,ai,br,bi),phase


def _product(left,right):
    ar,ai,br,bi=left
    cr,ci,dr,di=right
    return (ar*cr-ai*ci-br*dr-bi*di,
            ar*ci+ai*cr-br*di+bi*dr,
            br*cr-bi*ci+ar*dr+ai*di,
            br*ci+bi*cr+ar*di-ai*dr)


def _adjoint(parts):
    ar,ai,br,bi=parts
    return ar,-ai,-br,-bi


def _polar(parts):
    ar,ai,br,bi=parts
    aa,bb=sp.sqrt(ar*ar+ai*ai),sp.sqrt(br*br+bi*bi)
    first=sp.Piecewise((0,sp.Eq(aa,0)),(sp.atan2(ai,ar),True))
    last=sp.Piecewise((0,sp.Eq(bb,0)),(sp.atan2(bi,br),True))
    return (2*sp.atan2(bb,aa),last-first,-last-first),first


@lru_cache(maxsize=1)
def verify_controlled_chart_lemma():
    if not verify_rotation_vector_lemma():return False
    if not verify_phase_chart_lemma():return False
    alpha,beta,omega=sp.symbols("controlled_alpha controlled_beta controlled_omega",real=True)
    direction=(sp.sin(beta)*sp.cos(alpha),sp.sin(beta)*sp.sin(alpha),sp.cos(beta))
    pauli=sum((value*gates.make_gate(axis) for value,axis in zip(direction,"XYZ")),sp.zeros(2))
    target=sp.cos(omega)*sp.eye(2)-sp.I*sp.sin(omega)*pauli
    word=(("Rz",-alpha,(0,)),("Ry",-beta,(0,)),("Rz",2*omega,(0,)),
          ("Ry",beta,(0,)),("Rz",alpha,(0,)))
    if prove_circuit_identity(target,word) is not True:return False
    # SU(2) multiplication and norm preservation are polynomial identities.
    first=sp.symbols("controlled_ar controlled_ai controlled_br controlled_bi",real=True)
    second=sp.symbols("controlled_cr controlled_ci controlled_dr controlled_di",real=True)
    def matrix(p):
        ar,ai,br,bi=p
        return sp.Matrix([[ar+sp.I*ai,-br+sp.I*bi],[br+sp.I*bi,ar-sp.I*ai]])
    product=_product(first,second)
    if any(sp.expand(value)!=0 for value in matrix(product)-matrix(first)*matrix(second)):return False
    if sp.expand(sum(v*v for v in product)-sum(v*v for v in first)*sum(v*v for v in second))!=0:return False
    # The spherical chart reconstructs (x,y,z)/r; r=0 is the scalar +/-I
    # branch. The xy=0 poles use alpha=0 and beta=0 or pi explicitly.
    x,y,z,w=sp.symbols("controlled_x controlled_y controlled_z controlled_w",real=True)
    xy=sp.sqrt(x*x+y*y)
    radius=sp.sqrt(x*x+y*y+z*z)
    if sp.simplify(sp.sin(sp.atan2(xy,z))*sp.cos(sp.atan2(y,x))-x/radius)!=0:return False
    if sp.simplify(sp.sin(sp.atan2(xy,z))*sp.sin(sp.atan2(y,x))-y/radius)!=0:return False
    if sp.simplify(sp.cos(sp.atan2(xy,z))-z/radius)!=0:return False
    norm=sp.sqrt(radius*radius+w*w)
    if sp.simplify(sp.cos(sp.atan2(radius,w))-w/norm)!=0:return False
    if sp.simplify(sp.sin(sp.atan2(radius,w))-radius/norm)!=0:return False
    # General controlled identity: CU= P_control(delta-omega) V CP(2omega) V^dagger.
    delta=sp.Symbol("controlled_delta",real=True)
    local=gates.compile_gates(1,(("Ry",beta,(0,)),("Rz",alpha,(0,))))
    controlled=gates.controlled_gate(sp.exp(sp.I*delta)*local*gates.make_gate("Rz",2*omega)*local.H)
    source=(("Rz",-alpha,(1,)),("Ry",-beta,(1,)),("CP",2*omega,(0,1)),
            ("Ry",beta,(1,)),("Rz",alpha,(1,)),("P",delta-omega,(0,)))
    if prove_circuit_identity(controlled,source) is not True:return False
    if __package__:
        from .symbolic_native_echoes import b_pauli_echo
    else:
        from symbolic_native_echoes import b_pauli_echo
    echo,_=b_pauli_echo(("Z","Z"),-omega)
    b_word=(("Rz",omega,(0,)),("Rz",omega,(1,)))+echo
    if prove_circuit_identity(gates.make_gate("CP",2*omega),b_word,sp.exp(sp.I*omega/2)) is not True:return False
    t,p,l=sp.symbols("controlled_t controlled_p controlled_l",real=True)
    parts,phase=_primitive_parts(("U3",t,p,l,(0,)))
    if not _zero(gates.make_gate("U3",t,p,l)-sp.exp(sp.I*phase)*matrix(parts)):return False
    if __package__:
        from .symbolic_osr import realign
    else:
        from symbolic_osr import realign
    return all(realign(gates.make_gate(name),(0,)).rank()==4 for name in ("B","SiSWAP"))


def _primitive_parts(op):
    name=op[0]
    if name=="_matrix":return op[1],op[2]
    if name=="U3":
        theta,phi,lam=op[1:-1]
        c,s=sp.cos(theta/2),sp.sin(theta/2)
        p,m=(phi+lam)/2,(phi-lam)/2
        return (c*sp.cos(p),-c*sp.sin(p),s*sp.cos(m),s*sp.sin(m)),p
    if name in {"Rx","Ry","Rz","P"}:
        theta=op[1]
        c,s=sp.cos(theta/2),sp.sin(theta/2)
        if name=="Rx":return (c,0,0,-s),sp.S.Zero
        if name=="Ry":return (c,0,s,0),sp.S.Zero
        return (c,-s,0,0),theta/2 if name=="P" else sp.S.Zero
    return _parts(gates.make_gate(name,*op[1:-1]))


def _lower_locals(source,phase,basis,phase_exact):
    output=[]
    for op in source:
        if len(op[-1])==2 or op[0]=="U3":output.append(op);continue
        name,wire=op[0],op[-1]
        if name=="_matrix":
            parts,scalar=op[1:3]
            if parts==(1,0,0,0):phase+=scalar;continue
            angles,first=_polar(parts)
            phase+=scalar+first
            output.append(("U3",*angles,wire))
            continue
        if name in {"Rz","Ry","Rx","P"}:
            angle=op[1]
            if angle==0:continue
            if name=="Ry":angles=(angle,0,0)
            elif name=="Rx":angles=(angle,-sp.pi/2,sp.pi/2)
            else:
                angles=(0,0,angle)
                if name=="Rz":phase-=angle/2
            output.append(("U3",*angles,wire))
            continue
        if __package__:
            from .symbolic_synthesis import _local
        else:
            from symbolic_synthesis import _local
        if __package__:
            from .symbolic_bases import get_basis
        else:
            from symbolic_bases import get_basis
        word,scalar=_local(name,op[1:-1],get_basis("cx_u"))
        value=_phase(scalar)
        if value is None:raise ArithmeticError("Unproved local scalar phase")
        phase+=value
        output.extend((*item[:-1],wire) for item in word)
    if "PhasedXZ" in basis.single_qubit_gates:
        phase-=sum((op[1]/2 for op in output if op[0]=="U3"),sp.S.Zero)
        output=[_phased_xz(op) if op[0]=="U3" else op for op in output]
        if phase_exact and phase!=0:
            if __package__:
                from .symbolic_sqrt_iswap_phase import _phase_in_phased_xz
            else:
                from symbolic_sqrt_iswap_phase import _phase_in_phased_xz
            output=list(_phase_in_phased_xz(output[0],phase))+output[1:]
    elif phase_exact and phase!=0:
        output=list(_phase_in_u3(output[0],phase))+output[1:]
    return validate_operations(tuple(output),basis,2),sp.S.Zero if phase_exact else phase


@lru_cache(maxsize=512)
def _controlled_blocks(target,additional_frames=()):
    if target.shape!=(4,4):return None
    # Local Clifford changes of the control axis cover product-Pauli
    # rotations as well as ordinary computational-basis controlled gates.
    frames=((),(("H",),),(("H",),("S",)))+additional_frames
    for frame in additional_frames:
        for op in frame:
            if op[0] not in {"H","S","Rz"}:raise ValueError("Unsupported control-axis frame")
            if op[0]=="Rz" and (len(op)!=2 or op[1].is_real is not True):
                raise ValueError("Control-axis frame must have a real angle")
    for control in (0,1):
        for frame in frames:
            word=tuple((*op,(control,)) for op in frame)
            matrix=gates.compile_gates(2,word)
            candidate=matrix.H*target*matrix
            groups=[tuple(i for i in range(4) if (i>>control)&1==bit) for bit in (0,1)]
            if not _zero(candidate[i,j] for g in (0,1) for i in groups[g] for j in groups[1-g]):continue
            blocks=[sp.ImmutableMatrix(2,2,lambda i,j:candidate[ids[i],ids[j]]) for ids in groups]
            parts=[_parts(block) for block in blocks]
            if any(value is None for value in parts):continue
            return control,word,tuple(parts)
    return None


def synthesize_controlled_chart(target,basis,*,phase_exact,additional_frames=()):
    if basis.entangler not in {"B","SiSWAP"}:return None
    if not ({"U3","PhasedXZ"}&set(basis.single_qubit_gates)):return None
    target=sp.ImmutableMatrix(target)
    blocks=_controlled_blocks(target,additional_frames)
    if blocks is None:return None
    if not verify_controlled_chart_lemma():raise ArithmeticError("Controlled SU(2) chart lemma failed")
    control,frame,((m0,a0),(m1,a1))=blocks
    other=1-control
    relative=_product(m1,_adjoint(m0))
    w,minus_z,y,minus_x=relative
    x,z=-minus_x,-minus_z
    xy=sp.sqrt(x*x+y*y)
    radius=sp.sqrt(x*x+y*y+z*z)
    alpha=sp.Piecewise((0,sp.Eq(xy,0)),(sp.atan2(y,x),True))
    beta=sp.Piecewise((0,sp.Eq(radius,0)),(sp.atan2(xy,z),True))
    omega=sp.atan2(radius,w)
    inverse=tuple(("Rz",-op[1],op[-1]) if op[0]=="Rz" else
                  ("H" if op[0]=="H" else "Sdg",op[-1]) for op in reversed(frame))
    source=inverse+(("_matrix",m0,a0,(other,)),("Rz",-alpha,(other,)),("Ry",-beta,(other,)))
    if basis.entangler=="SiSWAP":
        if not verify_phase_chart_lemma():raise ArithmeticError("SQiSW phase chart lemma failed")
        # omega=atan2(r,w) lies in [0,pi], with w*w+r*r=1.
        # Use half-angle algebra directly; nested sin(atan2(...)) makes
        # otherwise small symbolic cost calculations needlessly expensive.
        sign=sp.Piecewise((1,sp.Ge(w,0)),(-1,True))
        q=sp.Piecewise((2*omega,sp.Ge(w,0)),(2*omega-2*sp.pi,True))
        u=sp.sqrt((1+sp.Abs(w))/2)
        v=sign*sp.sqrt((1-sp.Abs(w))/2)
        root=sp.sqrt(sp.Abs(w))
        phi=sp.atan2(sp.sqrt(2)*v,root)
        xi=2*sp.atan2(v,u+root)
        native=_u3_word(q,phi,xi)
        phase=-q/4
    else:
        if __package__:
            from .symbolic_native_echoes import b_pauli_echo
        else:
            from symbolic_native_echoes import b_pauli_echo
        native,scalar=b_pauli_echo(("Z","Z"),-omega)
        native=(("Rz",omega,(0,)),("Rz",omega,(1,)))+native
        # CP(2omega)=exp(i*omega/2) Rz(omega)^tensor2 exp(i*omega/2 ZZ).
        phase=omega/2
    source+=tuple((*op[:-1],tuple((control,other)[q] for q in op[-1])) for op in native)
    source+=(("Ry",beta,(other,)),("Rz",alpha,(other,)),("P",a1-a0-omega,(control,)))+frame
    operations,angle=_lower_locals(source,phase,basis,phase_exact)
    result=dict(status="verified_symbolic",phase_exact=phase_exact,operations=encode_sequence(operations),
        cost=circuit_cost(operations).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*angle)),physical=False,cost=0),
        verification="exact_controlled_SU2_chart_and_native_phase_lemma",
        construction_method="OSR_two_controlled_blocks_to_two_native_pulses",
        controlled_chart_certificate=dict(control=control,frame=encode_sequence(frame),
            additional_frames=[[[op[0],*map(str,op[1:])] for op in word] for word in additional_frames],
            blocks=[dict(quaternion=list(map(str,p)),phase=str(a)) for p,a in ((m0,a0),(m1,a1))],
            axis=[str(alpha),str(beta)],eigenangle=str(omega),
            lemmas=["exact_SU2_product_and_polar_charts","spherical_axis_including_poles",
                    "controlled_phase_reduction","native_two_pulse_phase_identity"]),
        entangler_optimal=False,local_cost_optimal=False,
        optimality_scope="two entanglers when the relative SU2 block is not scalar; local score is an upper bound")
    condition=sp.Ne(radius,0)
    if condition is sp.S.true:
        result["entangler_optimal"]=True
        result["entangler_optimality_certificate"]=dict(lower_bound=2,
            method="controlled_blocks_OSR_two_excludes_one_native_OSR_four_pulse")
    elif condition is not sp.S.false:
        result["conditional_entangler_optimality"]=dict(count=2,condition=str(condition),
            source="controlled_blocks_OSR_two_excludes_one_native_OSR_four_pulse")
    return result


def verify_controlled_chart_certificate(target,result,basis):
    if result.get("verification")!="exact_controlled_SU2_chart_and_native_phase_lemma":return False
    target=sp.ImmutableMatrix(target)
    symbols={str(symbol):symbol for symbol in target.free_symbols}
    try:
        additional=tuple(tuple((op[0],*(sp.sympify(p,locals=symbols) for p in op[1:])) for op in word)
                         for word in result["controlled_chart_certificate"].get("additional_frames",[]))
        expected=synthesize_controlled_chart(target,basis,phase_exact=result["phase_exact"],additional_frames=additional)
    except (ValueError,KeyError,TypeError):return False
    if expected is None:return False
    certificate=dict(result["controlled_chart_certificate"])
    certificate.setdefault("additional_frames",[])
    return (certificate==expected["controlled_chart_certificate"] and
        all(result.get(key)==expected.get(key) for key in
            ("status","operations","cost","virtual_gp","entangler_optimal",
             "entangler_optimality_certificate","conditional_entangler_optimality")))
