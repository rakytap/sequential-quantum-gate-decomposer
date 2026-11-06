"""Branch-safe real rotation-vector synthesis with replayable exact lemmas."""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from .symbolic_identity import prove_circuit_identity
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost,encode_sequence,validate_operations
    from symbolic_identity import prove_circuit_identity


def _source(alpha,beta,gamma):
    return (("Rz",gamma,(0,)),("Ry",beta,(0,)),("Rz",alpha,(0,)))


@lru_cache(maxsize=1)
def verify_rotation_vector_lemma():
    """Prove the polar SU(2) chart, its zero-amplitude branches and RV norm."""
    first,last,half=sp.symbols("rv_first rv_last rv_half",real=True)
    a=sp.exp(sp.I*first)*sp.cos(half)
    b=sp.exp(sp.I*last)*sp.sin(half)
    polar=sp.Matrix([[a,-sp.conjugate(b)],[b,sp.conjugate(a)]])
    if prove_circuit_identity(polar,_source(last-first,2*half,-last-first)) is not True:return False
    ar,ai,br,bi=sp.symbols("rv_ar rv_ai rv_br rv_bi",real=True)
    abs_a=sp.sqrt(ar*ar+ai*ai)
    abs_b=sp.sqrt(br*br+bi*bi)
    norm=sp.sqrt(ar*ar+ai*ai+br*br+bi*bi)
    # On the regular chart, atan2 reconstructs each complex amplitude and
    # the nonnegative pair of magnitudes. Its normalization is exact.
    for real,imag,magnitude in ((ar,ai,abs_a),(br,bi,abs_b)):
        value=sp.expand_complex(sp.exp(sp.I*sp.atan2(imag,real)))
        if sp.simplify(magnitude*value-real-sp.I*imag)!=0:return False
    if sp.simplify(sp.cos(sp.atan2(abs_b,abs_a))-abs_a/norm)!=0:return False
    if sp.simplify(sp.sin(sp.atan2(abs_b,abs_a))-abs_b/norm)!=0:return False
    # a=0 and b=0 are separate charts: set only the absent amplitude's phase
    # to zero. The remaining magnitude is nonzero since the total norm is 1.
    if polar.subs({first:0,half:sp.pi/2})!=sp.Matrix([[0,-sp.exp(-sp.I*last)],[sp.exp(sp.I*last),0]]):return False
    if polar.subs({last:0,half:0})!=sp.diag(sp.exp(sp.I*first),sp.exp(-sp.I*first)):return False
    x,y,z=sp.symbols("rv_x rv_y rv_z",real=True)
    radius=sp.sqrt(x*x+y*y+z*z)
    target=gates.make_gate("RV",x,y,z)
    scale=sp.sinc(radius/2)/2
    real_a,imag_a=sp.cos(radius/2),-z*scale
    real_b,imag_b=y*scale,-x*scale
    cartesian=sp.Matrix([[real_a+sp.I*imag_a,-real_b+sp.I*imag_b],
                         [real_b+sp.I*imag_b,real_a-sp.I*imag_a]])
    if any(sp.simplify(a-b)!=0 for a,b in zip(target,cartesian)):return False
    unit=real_a**2+imag_a**2+real_b**2+imag_b**2
    regular=unit.xreplace({sp.sinc(radius/2):2*sp.sin(radius/2)/radius})
    if sp.simplify(regular-1)!=0:return False
    # The sinc definition is removable at the origin, which belongs to b=0.
    if gates.make_gate("RV",0,0,0)!=sp.eye(2):return False
    return True


def _angles(parameters):
    x,y,z=map(sp.sympify,parameters)
    if any(value.is_real is not True for value in (x,y,z)):
        raise ValueError("Rotation-vector coordinates must be real")
    radius=sp.sqrt(x*x+y*y+z*z)
    scale=sp.sinc(radius/2)/2
    ar,ai=sp.cos(radius/2),-z*scale
    br,bi=y*scale,-x*scale
    abs_a=sp.sqrt(ar*ar+ai*ai)
    abs_b=sp.sqrt(br*br+bi*bi)
    first=sp.Piecewise((0,sp.Eq(abs_a,0)),(sp.atan2(ai,ar),True))
    last=sp.Piecewise((0,sp.Eq(abs_b,0)),(sp.atan2(bi,br),True))
    return last-first,2*sp.atan2(abs_b,abs_a),-last-first


@lru_cache(maxsize=64)
def _native_template(basis,phase_exact):
    if __package__:
        from .symbolic_synthesis import _local,_cleanup,_physical_scalar,_verify_phase_candidate
    else:
        from symbolic_synthesis import _local,_cleanup,_physical_scalar,_verify_phase_candidate
    symbols=sp.symbols("rv_alpha rv_beta rv_gamma",real=True)
    source=_source(*symbols)
    operations=[]
    factor=sp.S.One
    if "U3" in basis.single_qubit_gates or "U" in basis.single_qubit_gates:
        name="U3" if "U3" in basis.single_qubit_gates else "U"
        alpha,beta,gamma=symbols
        if phase_exact:
            operations=[(name,sp.pi-beta,0,gamma-sp.pi,(0,)),
                        (name,sp.pi,(alpha-gamma)/2,sp.pi-(alpha+gamma)/2,(0,))]
        else:
            operations=[(name,beta,alpha,gamma,(0,))]
            factor=sp.exp(-sp.I*(alpha+gamma)/2)
    else:
        for name,parameter,wires in source:
            word,scalar=_local(name,(parameter,),basis)
            operations.extend(word)
            factor*=scalar
    operations=_cleanup(operations)
    factor=sp.simplify(factor)
    if phase_exact and factor!=1:operations+=_physical_scalar(factor,basis,1)
    target=gates.compile_gates(1,source)
    result=_verify_phase_candidate(target,operations,basis,phase_exact,factor)
    if result["status"]!="verified_symbolic":raise ValueError("Rotation-vector native template was not proved")
    return symbols,operations,factor,result["verification"]


def synthesize_rotation_vector(parameters,basis,*,phase_exact):
    if not verify_rotation_vector_lemma():
        return dict(status="unresolved",phase_exact=phase_exact,reason="Rotation-vector chart lemma was not proved")
    angles=_angles(parameters)
    symbols,template,factor,method=_native_template(basis,phase_exact)
    substitutions=dict(zip(symbols,angles))
    operations=tuple((op[0],*(parameter.subs(substitutions) for parameter in map(sp.sympify,op[1:-1])),op[-1])
                     for op in template)
    operations=validate_operations(operations,basis,1)
    scalar=sp.simplify(factor.subs(substitutions))
    return dict(status="verified_symbolic",phase_exact=phase_exact,
                operations=encode_sequence(operations),cost=circuit_cost(operations).as_dict(),
                virtual_gp=None if phase_exact else dict(factor=str(scalar),physical=False,cost=0),
                verification="exact_rotation_vector_chart_and_native_template",
                rotation_vector_certificate=dict(vector=[str(value) for value in parameters],
                    euler_zyz=[str(value) for value in angles],
                    charts=["a=0,b!=0","b=0,a!=0","a!=0,b!=0"],
                    native_template_verification=method),
                entangler_optimal=True,local_cost_optimal=False,
                optimality_scope="zero_entanglers; local parameter score is an upper bound")


def verify_rotation_vector_certificate(target,result,basis):
    """Replay the exact chart lemma, native template, phase policy and cost."""
    if result.get("verification")!="exact_rotation_vector_chart_and_native_template":return False
    target=sp.ImmutableMatrix(target)
    symbols={str(value):value for value in target.free_symbols}
    try:
        vector=tuple(sp.sympify(value,locals=symbols) for value in result["rotation_vector_certificate"]["vector"])
        if len(vector)!=3 or target!=gates.make_gate("RV",*vector):return False
        expected=synthesize_rotation_vector(vector,basis,phase_exact=result["phase_exact"])
    except (ValueError,KeyError,TypeError):return False
    return all(result.get(key)==expected.get(key) for key in
               ("status","operations","cost","virtual_gp","rotation_vector_certificate"))
