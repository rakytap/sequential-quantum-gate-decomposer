"""Replayable fixed native motifs and exact one-wire lowering components.

Numerical analytical decompositions (including Cirq) suggested the pulse
skeletons. Every base motif is proved against the full fixed target here;
neither rounded angles nor sampled fidelities can certify a result.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_bases import make_basis
    from .symbolic_contract import circuit_cost,encode_sequence,validate_operations,verify_circuit
    from .symbolic_identity import prove_circuit_identity
    from .symbolic_native_echoes import native_dcx_source,sqrt_iswap_berkeley_source,sqrt_iswap_sqrt_swap_source
    from .symbolic_sqrt_iswap_phase import _phased_xz,_phase_in_u3,_phase_in_phased_xz
else:
    import symbolic_gates as gates
    from symbolic_bases import make_basis
    from symbolic_contract import circuit_cost,encode_sequence,validate_operations,verify_circuit
    from symbolic_identity import prove_circuit_identity
    from symbolic_native_echoes import native_dcx_source,sqrt_iswap_berkeley_source,sqrt_iswap_sqrt_swap_source
    from symbolic_sqrt_iswap_phase import _phased_xz,_phase_in_u3,_phase_in_phased_xz


@lru_cache(maxsize=1)
def verify_native_lowering_lemmas():
    t,p,l,gamma=sp.symbols("fixed_t fixed_p fixed_l fixed_gamma",real=True)
    op=("U3",t,p,l,(0,))
    if prove_circuit_identity(gates.make_gate("U3",t,p,l),(_phased_xz(op),),sp.exp(-sp.I*t/2)) is not True:return False
    if prove_circuit_identity(sp.exp(sp.I*gamma)*gates.make_gate("U3",t,p,l),_phase_in_u3(op,gamma)) is not True:return False
    x,z,a=sp.symbols("fixed_x fixed_z fixed_a",real=True)
    pxz=("PhasedXZ",x,z,a,(0,))
    return prove_circuit_identity(sp.exp(sp.I*gamma)*gates.make_gate("PhasedXZ",x,z,a),_phase_in_phased_xz(pxz,gamma)) is True


@lru_cache(maxsize=5)
def _proved_base(name):
    if name=="SYC":
        if __package__:
            from .symbolic_sqrt_iswap_face import sqrt_iswap_syc_source
        else:
            from symbolic_sqrt_iswap_face import sqrt_iswap_syc_source
        word,factor=sqrt_iswap_syc_source()
        return word,factor,"exact_parity_block_chart_and_native_local_fusions"
    if name=="B":word,factor=sqrt_iswap_berkeley_source()
    elif name in {"SSWAP","SSWAPdg"}:word,factor=sqrt_iswap_sqrt_swap_source(name=="SSWAPdg")
    elif name=="DCX":word,factor=native_dcx_source("SiSWAP")
    else:return None
    base_basis=make_basis("fixed_native_base",("U3",),"SiSWAP")
    result=verify_circuit(gates.make_gate(name),word,base_basis,phase_exact=False,scalar_factor=factor)
    if result["status"]!="verified_symbolic":raise ArithmeticError("Fixed native motif identity failed")
    return word,factor,result["verification"]


def synthesize_fixed_native(target,basis,*,phase_exact):
    if basis.entangler!="SiSWAP" or not ({"U3","U","PhasedXZ"}&set(basis.single_qubit_gates)):return None
    target=sp.ImmutableMatrix(target)
    if target.shape!=(4,4) or target.free_symbols:return None
    name=next((name for name in ("B","DCX","SSWAP","SSWAPdg","SYC") if target==gates.make_gate(name)),None)
    if name is None:return None
    base,factor,method=_proved_base(name)
    if not verify_native_lowering_lemmas():raise ArithmeticError("Native lowering lemma failed")
    word=base
    if __package__:
        from .symbolic_native_pauli import _phase
    else:
        from symbolic_native_pauli import _phase
    angle=_phase(factor)
    if angle is None:raise ArithmeticError("Fixed motif has no proved real scalar phase")
    if "PhasedXZ" in basis.single_qubit_gates:
        angle-=sum((sp.sympify(op[1])/2 for op in word if op[0]=="U3"),sp.S.Zero)
        word=tuple(_phased_xz(op) if op[0]=="U3" else op for op in word)
        close=_phase_in_phased_xz
    else:close=_phase_in_u3
    angle=sp.simplify(angle)
    closure=None
    if phase_exact and angle!=0:
        index=next(i for i,op in enumerate(word) if len(op[-1])==1)
        before=word[index];after=close(before,angle)
        closure=dict(index=index,before=encode_sequence((before,)),after=encode_sequence(after),angle=str(angle))
        word=word[:index]+after+word[index+1:]
    word=validate_operations(word,basis,2)
    result=dict(status="verified_symbolic",phase_exact=phase_exact,
        operations=encode_sequence(word),cost=circuit_cost(word).as_dict(),
        virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*angle)),physical=False,cost=0),
        verification="composition_of_exact_fixed_native_identities",
        construction_method="fixed_SQiSW_motif_and_exact_native_lowering",
        fixed_native_certificate=dict(target=name,base_operations=encode_sequence(base),base_factor=str(factor),
            base_verification=method,local_lowering="U3_to_PhasedXZ_and_phase_absorption" if "PhasedXZ" in basis.single_qubit_gates else "native_U3",
            closure=closure),entangler_optimal=False,local_cost_optimal=False,
        optimality_scope="entangler certificate only; local score is an upper bound")
    if __package__:
        from .symbolic_synthesis import _attach_osr_certificate
    else:
        from symbolic_synthesis import _attach_osr_certificate
    result=_attach_osr_certificate(result,target,basis)
    if name=="SYC":
        if __package__:
            from .symbolic_minimality import sqrt_iswap_cost_certificate
        else:
            from symbolic_minimality import sqrt_iswap_cost_certificate
        # verify_syc_face_motif proves the local equivalence, including its
        # scalar, to this Cartan point. The relaxed-local Weyl bound applies.
        certificate=sqrt_iswap_cost_certificate(gates.make_gate("Canonical",sp.pi/4,sp.pi/4,-sp.pi/24))
        if certificate is None or certificate["minimum"]!=3:raise ArithmeticError("SYC Weyl bound failed")
        result.update(entangler_optimal=True,native_invariant_lower_bound=3,
            entangler_optimality_certificate=dict(certificate,local_equivalence="proved_SYC_Cartan_local_frames"))
    return result


def verify_fixed_native_certificate(target,result,basis):
    if result.get("verification")!="composition_of_exact_fixed_native_identities":return False
    if not isinstance(result.get("phase_exact"),bool):return False
    try:expected=synthesize_fixed_native(target,basis,phase_exact=result["phase_exact"])
    except (ValueError,KeyError,TypeError):return False
    return expected is not None and all(result.get(key)==expected.get(key) for key in
        ("status","operations","cost","virtual_gp","fixed_native_certificate",
         "entangler_optimal","entangler_optimality_certificate","native_invariant_lower_bound","local_cost_optimal"))
