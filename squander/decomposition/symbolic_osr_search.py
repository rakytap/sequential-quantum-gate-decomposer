"""Matrix-driven symbolic OSR factorization and native interaction peeling.

The search alphabet is explicit and finite. OSR rank-one residuals determine
local factors; complete circuit identities are proved by the caller. Search
exhaustion alone is not a continuous-basis optimality certificate.
"""

from functools import lru_cache
import itertools
import time
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_identity import exponential_forms,zero_form
    from .symbolic_osr import qubit_width,bipartitions,product_from_form
    from .symbolic_clifford_search import group_data
else:
    import symbolic_gates as gates
    from symbolic_identity import exponential_forms,zero_form
    from symbolic_osr import qubit_width,bipartitions,product_from_form
    from symbolic_clifford_search import group_data


def angle_candidates(matrix,limit=32):
    """Exact angle proposals from the matrix's trigonometric/exponential terms."""
    proposals=[]
    def add(angle):
        angle=sp.expand(angle)
        if angle.is_real is not False and not angle.atoms(sp.Float) and angle not in proposals:
            proposals.append(angle)
    symbols=sorted(matrix.free_symbols,key=str)
    if matrix.shape==(4,4) and not symbols:
        yy=sp.kronecker_product(gates.make_gate("Y"),gates.make_gate("Y"))
        gamma=(matrix*yy*matrix.T*yy).applyfunc(sp.simplify)
        eigenvalues=gamma.eigenvals()
        if len(eigenvalues)==2 and set(eigenvalues.values())=={2}:
            first,second=eigenvalues
            phase=sp.simplify(sp.arg(sp.simplify(second/first)))
            if sp.simplify(phase/sp.pi).is_Rational:
                determinant_phase=sp.simplify(sp.arg(matrix.det()))
                if determinant_phase!=0 and sp.simplify(determinant_phase/sp.pi).is_Rational:
                    add(-determinant_phase/2);add(determinant_phase/2)
                add(-sp.Abs(phase)/2);add(sp.Abs(phase)/2)
    if not symbols:
        for entry in matrix:
            if entry==0:continue
            phase=sp.simplify(sp.arg(entry))
            if phase!=0 and sp.simplify(phase/sp.pi).is_Rational:
                for scale in (-sp.Rational(1,2),sp.Rational(1,2),-1,1,-2,2):add(scale*phase)
    for symbol in symbols:
        for scale in (-sp.Rational(1,2),sp.Rational(1,2),-1,1,-2,2):add(scale*symbol)
    for entry in matrix:
        for term in sorted(entry.atoms(sp.sin,sp.cos),key=str):
            for scale in (1,-1,2,-2,sp.Rational(1,2),-sp.Rational(1,2)):add(scale*term.args[0])
        for term in sorted(entry.atoms(sp.exp),key=str):
            angle=sp.simplify(term.args[0]/sp.I)
            if angle.free_symbols:
                for scale in (1,-1,2,-2,sp.Rational(1,2),-sp.Rational(1,2)):add(scale*angle)
    for denominator in (2,4,8,16):
        for sign in (1,-1):add(sign*sp.pi/denominator)
    add(sp.pi);add(0)
    return tuple(proposals[:limit])


def _proportional(left,right,variables):
    pivot=next((i for i,x in enumerate(right) if x!=0 and not zero_form(x,variables)),None)
    if pivot is None:return None
    if all(zero_form(a*right[pivot]-b*left[pivot],variables) for a,b in zip(left,right)):
        return sp.cancel(left[pivot]/right[pivot])
    return None


@lru_cache(maxsize=1)
def _clifford_projective_lookup():
    lookup={}
    for name in group_data()[0].representatives:
        matrix=gates.make_gate(name)
        pivot=next(x for x in matrix if x!=0)
        key=tuple(sp.simplify(x/pivot) for x in matrix)
        lookup[key]=name
    return lookup


@lru_cache(maxsize=2048)
def recognize_local(matrix,angles):
    """Return a symbolic one-wire word and scalar, without an Euler branch cut."""
    matrix=sp.ImmutableMatrix(matrix)
    if matrix.shape!=(2,2):return None
    forms,inverse,convert=exponential_forms(tuple(matrix),return_converter=True)
    target=sp.ImmutableMatrix(2,2,forms[:4])
    pivot=next((x for x in target if x!=0 and not zero_form(x,tuple(inverse))),None)
    key=tuple(sp.cancel(x/pivot).xreplace(inverse) for x in target) if pivot is not None else ()
    key=tuple(sp.simplify(x) if not x.free_symbols else x for x in key)
    name=_clifford_projective_lookup().get(key)
    if name is not None:
        formal=gates.make_gate(name).applyfunc(convert)
        factor=_proportional(target,formal,tuple(inverse))
        if factor is not None:
            sequence=() if name=="I" else ((name,(0,)),)
            return sequence,sp.simplify(factor.xreplace(inverse))
    # An axial rotation lies in span{I,P}. Its two eigenvalues have ratio
    # exp(i*angle), avoiding trigonometric half-angle branch choices.
    identity=sp.eye(2)
    scalar=sp.trace(matrix)/2
    for axis in ("Z","Y","X"):
        pauli=gates.make_gate(axis)
        coefficient=sp.trace(pauli*matrix)/2
        forms,inverse=exponential_forms(tuple(matrix-scalar*identity-coefficient*pauli))
        if not all(zero_form(value,tuple(inverse)) for value in forms):continue
        plus,minus=sp.simplify(scalar+coefficient),sp.simplify(scalar-coefficient)
        for angle in angles:
            forms,inverse=exponential_forms((minus-sp.exp(sp.I*angle)*plus,))
            if zero_form(forms[0],tuple(inverse)):
                factor=sp.simplify(plus*sp.exp(sp.I*angle/2))
                return (("R"+axis,angle,(0,)),),factor
    return None


def _remap(sequence,wires):
    return tuple((*op[:-1],tuple(wires[q] for q in op[-1])) for op in sequence)


def _native_parameters(entangler,angle):
    if entangler in {"Rxx","Rzz","Rzx"}:return (angle,)
    if entangler=="ZZPhase":return (angle/sp.pi,)
    if entangler=="IonQZZ":return (angle/(2*sp.pi),)
    if entangler=="IonQMS":return (0,0,angle/(2*sp.pi))
    return None


def _axis_frame(source_axis,target_axis):
    # W source_axis W† = target_axis. Words are in circuit time order.
    to_z={"Z":(),"X":(("H",(0,)),),"Y":(("Sdg",(0,)),("H",(0,)))}
    inverse={"H":"H","Sdg":"S"}
    first=to_z[source_axis]
    last=tuple((inverse[op[0]],(0,)) for op in reversed(to_z[target_axis]))
    return first+last


def _inverse_clifford_word(sequence):
    inverse={"H":"H","S":"Sdg","Sdg":"S"}
    return tuple((inverse[op[0]],op[-1]) for op in reversed(sequence))


def interaction_proposals(basis,angles):
    """Native pulses and exact local Pauli-axis frames, without CNOT rebasing."""
    entangler=basis.entangler
    base={"Rxx":("X","X"),"Rzz":("Z","Z"),"Rzx":("Z","X"),
          "ZZPhase":("Z","Z"),"IonQZZ":("Z","Z"),"IonQMS":("X","X")}.get(entangler)
    if base is not None:
        for angle in angles:
            if angle==0:continue
            parameters=_native_parameters(entangler,angle)
            if entangler=="IonQMS" and (angle.free_symbols or not 0<=angle<=sp.pi/2):
                # Preserve the actual hardware interaction range. A general
                # signed rotation needs a separate frame/domain construction.
                continue
            pulse=(entangler,*parameters,(0,1))
            for axes in itertools.product("ZXY",repeat=2):
                after=tuple(op for q in range(2) for op in _remap(_axis_frame(base[q],axes[q]),(q,)))
                before=_inverse_clifford_word(after)
                word=before+(pulse,)+after
                matrix=gates.make_gate("PauliRotation","".join(reversed(axes)),angle)
                yield word,matrix
    else:
        definition=gates.gate_definition(entangler)
        if not definition.parameters:
            word=((entangler,(0,1)),)
            yield word,gates.make_gate(entangler)
            # Short native powers can expose a product residual even when a
            # single pulse does not: sqrt(SWAP)^2 and sqrt(iSWAP)^2, for example.
            if entangler in {"SSWAP","SiSWAP"}:
                native=gates.make_gate(entangler)
                for power in (2,3):
                    yield word*power,(native**power).applyfunc(sp.simplify)
            if __package__:
                from .symbolic_rules import native_cx
            else:
                from symbolic_rules import native_cx
            try:frame,factor=native_cx(entangler)
            except KeyError:return
            if sum(len(op[-1])==2 for op in frame)==1 and frame!=word:
                yield frame,gates.make_gate("CX")/factor
        if entangler=="iSWAP":
            # iSWAP conjugates local Paulis to nonlocal Pauli products. Its
            # inverse is ZZ*iSWAP, so an arbitrary Pauli rotation uses two
            # native pulses, with no numerical KAK gauges or GP operation.
            if __package__:
                from .symbolic_clifford import _signed_pauli
            else:
                from symbolic_clifford import _signed_pauli
            native=gates.make_gate("iSWAP")
            local=gates.apply_to(sp.eye(4),2,gates.make_gate("X"),(0,))
            sign,label=_signed_pauli(native*local*native.H)
            original=tuple(reversed(label))
            inverse=(("iSWAP",(0,1)),("Z",(0,)),("Z",(1,)))
            for angle in angles:
                if angle==0:continue
                for axes in itertools.product("ZXY",repeat=2):
                    after=tuple(op for q in range(2) for op in _remap(_axis_frame(original[q],axes[q]),(q,)))
                    word=(_inverse_clifford_word(after)+inverse+(("Rx",sign*angle,(0,)),
                          ("iSWAP",(0,1)))+after)
                    yield word,gates.make_gate("PauliRotation","".join(reversed(axes)),angle)


def _factor_locals(matrix,angles,variables):
    """Recursively test every subsystem cut until only one-wire factors remain."""
    width=qubit_width(matrix)
    if width==1:
        # Inverse substitution is handled by the caller before recognition.
        return None
    for cut in bipartitions(width):
        pair=product_from_form(matrix,cut,variables)
        if pair is not None:return cut,pair
    return None


def osr_source_candidates(matrix,basis,*,seconds=2.0,max_trials=384,max_angles=32,max_depth=None):
    """Discover raw words from a matrix by exact OSR residual factorization."""
    matrix=sp.ImmutableMatrix(matrix)
    if max_depth is None:max_depth=max(2,qubit_width(matrix)-1)
    return _osr_source_candidates(matrix,basis,seconds,max_trials,max_angles,max_depth)


@lru_cache(maxsize=512)
def _osr_source_candidates(matrix,basis,seconds,max_trials,max_angles,max_depth):
    start=time.monotonic()
    width=qubit_width(matrix)
    angles=angle_candidates(matrix,max_angles)
    trials=0
    candidates=[]
    reason="search_alphabet_exhausted"
    forms,inverse,convert=exponential_forms(tuple(matrix)+tuple(sp.exp(sp.I*a/2) for a in angles if a.free_symbols),return_converter=True)
    initial=sp.ImmutableMatrix(matrix.rows,matrix.cols,forms[:len(matrix)])
    variables=tuple(inverse)

    class SearchBudget(Exception):pass
    def check_budget():
        if time.monotonic()-start>seconds:raise SearchBudget("search_wall_clock_budget")
        if trials>=max_trials:raise SearchBudget("search_trial_budget")

    proposals=[]
    proposal_stream=iter(interaction_proposals(basis,angles))
    def pulses():
        index=0
        while True:
            check_budget()
            if index==len(proposals):
                try:word,interaction=next(proposal_stream)
                except StopIteration:return
                proposals.append((word,interaction.H.applyfunc(convert)))
            yield proposals[index]
            index+=1
    def factor_word(operator,depth):
        nonlocal trials
        check_budget()
        local_width=qubit_width(operator)
        if local_width==1:
            restored=operator.applyfunc(lambda x:sp.cancel(x.xreplace(inverse)))
            return recognize_local(restored,angles)
        factored=_factor_locals(operator,angles,variables)
        if factored is not None:
            cut,pair=factored
            other=tuple(q for q in range(local_width) if q not in cut)
            a,b=pair
            left,right=factor_word(a,depth),factor_word(b,depth)
            if left is not None and right is not None:
                return _remap(left[0],cut)+_remap(right[0],other),sp.simplify(left[1]*right[1])
            return None
        if depth==0:return None
        for word,interaction in pulses():
            for wires in itertools.permutations(range(local_width),2):
                check_budget();trials+=1
                full=gates.apply_to(sp.eye(1<<local_width),local_width,interaction,wires)
                residual=operator*full
                # Peeling is guided by a newly separable OSR cut. This limits
                # the present search; failure is never called impossibility.
                if _factor_locals(residual,angles,variables) is None:continue
                result=factor_word(residual,depth-1)
                if result is not None:return _remap(word,wires)+result[0],result[1]
        return None

    try:
        found=factor_word(initial,max_depth)
        if found is not None:
            candidates.append(dict(sequence=found[0],factor=found[1],method="native_interaction_peeling_and_recursive_OSR_factorization"))
            reason="OSR_construction_found"
    except SearchBudget as error:reason=str(error)
    return candidates,dict(trials=trials,termination=reason,elapsed_seconds=time.monotonic()-start,
                           max_native_depth=max_depth,proof_scope="finite_cut_separating_peeling_and_local_recognition_alphabet")
