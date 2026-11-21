"""Exact two-pulse charts for number-conserving matchgates.

The even parity block is diagonal and has the same determinant as the odd
block. A proved SU(2) polar chart reduces the odd block to an exchange;
six native local gates suffice for either B or SQiSW. Absent complex
amplitudes use explicit polar branches. All real parameters remain free.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import circuit_cost, encode_sequence
    from .symbolic_controlled_chart import _zero, _polar, _lower_locals
    from .symbolic_identity import prove_circuit_identity
    from .symbolic_native_pauli import _phase
    from .symbolic_native_fixed import verify_native_lowering_lemmas
    from .symbolic_rotation_vector import verify_rotation_vector_lemma
else:
    import symbolic_gates as gates
    from symbolic_contract import circuit_cost, encode_sequence
    from symbolic_controlled_chart import _zero, _polar, _lower_locals
    from symbolic_identity import prove_circuit_identity
    from symbolic_native_pauli import _phase
    from symbolic_native_fixed import verify_native_lowering_lemmas
    from symbolic_rotation_vector import verify_rotation_vector_lemma


def _source(chi0,chi3,first,last,beta,entangler):
    left_difference=last-first+sp.pi/2
    right_difference=-last-first-sp.pi/2
    left_sum=chi3-chi0
    l0,l1=(left_sum-left_difference)/2,(left_sum+left_difference)/2
    r0,r1=-right_difference/2,right_difference/2
    tau=-beta/4
    if entangler=="SiSWAP":
        p=lambda value,wire:("U3",0,0,value,(wire,))
        return (p(r0+sp.pi/2,0),p(r1,1),("SiSWAP",(0,1)),
                p(sp.pi+2*tau,0),p(-2*tau,1),("SiSWAP",(0,1)),
                p(l0+sp.pi/2,0),p(l1,1))
    return (("U3",sp.pi/2,-sp.pi/2,sp.pi+r0,(0,)),
            ("U3",sp.pi/2,-sp.pi/2,sp.pi/2+r1,(1,)),("B",(0,1)),
            ("U3",2*tau,0,sp.pi,(0,)),("U3",2*tau,0,0,(1,)),("B",(0,1)),
            ("U3",sp.pi/2,l0,sp.pi/2,(0,)),("U3",-sp.pi/2,l1-sp.pi/2,sp.pi/2,(1,)))


@lru_cache(maxsize=1)
def verify_matchgate_chart_lemma():
    if not verify_rotation_vector_lemma() or not verify_native_lowering_lemmas():return False
    chi0,chi3,first,last,beta=sp.symbols("mg_c0 mg_c3 mg_first mg_last mg_beta",real=True)
    a=sp.exp(sp.I*first)*sp.cos(beta/2)
    b=sp.exp(sp.I*last)*sp.sin(beta/2)
    phase=sp.exp(sp.I*(chi0+chi3)/2)
    target=sp.diag(sp.exp(sp.I*chi0),1,1,sp.exp(sp.I*chi3))
    target[1,1],target[1,2]=phase*a,-phase*sp.conjugate(b)
    target[2,1],target[2,2]=phase*b,phase*sp.conjugate(a)
    return all(prove_circuit_identity(target,_source(chi0,chi3,first,last,beta,native),
               sp.exp(sp.I*chi0)) is True for native in ("B","SiSWAP"))


@lru_cache(maxsize=256)
def _parts(target):
    if target.shape!=(4,4):return None
    # Cheap structural rejection avoids applying a polar chart to a general
    # two-qubit matrix. Every accepted structural zero is checked exactly.
    allowed={(0,0),(3,3),(1,1),(1,2),(2,1),(2,2)}
    if not _zero(target[i,j] for i in range(4) for j in range(4) if (i,j) not in allowed):return None
    chi0,chi3=_phase(target[0,0]),_phase(target[3,3])
    if chi0 is None or chi3 is None:return None
    odd=target.extract((1,2),(1,2))*sp.exp(-sp.I*(chi0+chi3)/2)
    a,b=odd[0,0],odd[1,0]
    if not _zero((odd[0,1]+sp.conjugate(b),odd[1,1]-sp.conjugate(a),
                  a*sp.conjugate(a)+b*sp.conjugate(b)-1)):return None
    parts=tuple(sp.simplify(sp.expand_complex(value)) for value in (sp.re(a),sp.im(a),sp.re(b),sp.im(b)))
    if any(value.is_real is not True for value in parts):return None
    return chi0,chi3,parts


def synthesize_matchgate_chart(target,basis,*,phase_exact):
    if basis.entangler not in {"B","SiSWAP"} or not ({"U3","U","PhasedXZ"}&set(basis.single_qubit_gates)):return None
    target=sp.ImmutableMatrix(target)
    data=_parts(target)
    if data is None:return None
    if not verify_matchgate_chart_lemma():raise ArithmeticError("Matchgate chart lemma failed")
    chi0,chi3,parts=data
    ar,ai,br,bi=parts
    beta,alpha,gamma=_polar(parts)[0]
    # alpha=last-first, gamma=-last-first, including the absent-amplitude
    # branches. The chart proof covers these branches before substitution.
    first=-(alpha+gamma)/2
    last=(alpha-gamma)/2
    word=_source(chi0,chi3,first,last,beta,basis.entangler)
    word,phase=_lower_locals(word,chi0,basis,phase_exact)
    return dict(status="verified_symbolic",phase_exact=phase_exact,
        operations=encode_sequence(word),cost=circuit_cost(word).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*phase)),physical=False,cost=0),
        verification="exact_matchgate_polar_chart_and_native_template",
        construction_method="number_conserving_matchgate_two_native_pulses",
        matchgate_certificate=dict(even_phases=[str(chi0),str(chi3)],
            odd_su2_parts=[str(value) for value in parts],
            charts=["a=0,b!=0","b=0,a!=0","a!=0,b!=0"],
            determinant_condition="det(odd_block)=even_00*even_11"),
        entangler_optimal=False,local_cost_optimal=False,
        optimality_scope="two-pulse constructive upper bound; no minimum local score claim")


def verify_matchgate_certificate(target,result,basis):
    if result.get("verification")!="exact_matchgate_polar_chart_and_native_template":return False
    try:expected=synthesize_matchgate_chart(target,basis,phase_exact=result["phase_exact"])
    except (ValueError,KeyError,TypeError):return False
    return expected is not None and all(result.get(key)==expected.get(key) for key in
        ("status","operations","cost","virtual_gp","matchgate_certificate","entangler_optimal","local_cost_optimal"))
