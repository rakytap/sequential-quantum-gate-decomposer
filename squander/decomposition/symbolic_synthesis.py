"""Phase-aware constructive symbolic synthesis with explicit certificates.

Rules retain free parameters. A construction is an upper bound until a
matching lower bound, or an exhaustive declared search, certifies its cost.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_bases import get_basis
    from .symbolic_clifford_search import group_data, exact_state, solve_clifford_alphabet, clifford_path
    from .symbolic_contract import circuit_cost, resource_key, encode_sequence, simplify_exact, verify_circuit, retain_phase_results
    from .symbolic_feasibility import feasibility_obstruction
else:
    import symbolic_gates as gates
    from symbolic_bases import get_basis
    from symbolic_clifford_search import group_data, exact_state, solve_clifford_alphabet, clifford_path
    from symbolic_contract import circuit_cost, resource_key, encode_sequence, simplify_exact, verify_circuit, retain_phase_results
    from symbolic_feasibility import feasibility_obstruction


class RuleUnavailable(ValueError):
    pass


@lru_cache(maxsize=32768)
def _normalized_angle(angle):
    return sp.simplify(angle)


def _op(name, *parameters, wire=0):
    return (name, *parameters, (wire,))


def _cleanup(sequence):
    """Safe adjacent cancellation and additive rotations; no phase wrapping."""
    result = []
    additive = {"Rx", "Ry", "Rz", "P", "U1"}
    for operation in sequence:
        if operation[0] in additive and _normalized_angle(operation[1]) == 0:
            continue
        if result and operation[0] in additive and result[-1][0] == operation[0] and result[-1][-1] == operation[-1]:
            previous = result.pop()
            angle = _normalized_angle(previous[1]+operation[1])
            if angle != 0:
                result.append((operation[0],angle,operation[-1]))
        elif result and len(operation) == 2 and operation == result[-1] and operation[0] in {"X","Y","Z","H","CX","CNOT","CZ","ECR"}:
            result.pop()
        else:
            result.append(operation)
    return tuple(result)


def _best_clifford(target, basis, phase_exact):
    state = exact_state(target)
    if state is None:
        return None
    solution = solve_clifford_alphabet(basis.single_qubit_gates)
    states = [state] if phase_exact else list(range(state//8*8, state//8*8+8))
    states = [s for s in states if s in solution["distances"]]
    if not states:
        return None
    selected = min(states,key=lambda s:solution["distances"][s])
    operations = clifford_path(solution,selected)
    factor = sp.exp(sp.I*sp.pi*((state % 8-selected % 8) % 8)/4)
    return dict(status="verified_symbolic", phase_exact=phase_exact, operations=encode_sequence(operations),
                cost=circuit_cost(operations).as_dict(),
                virtual_gp=None if phase_exact else dict(factor=str(factor),physical=False,cost=0),
                verification="exact_group_multiplication_table",
                entangler_optimal=True, local_cost_optimal=solution["global_local_optimality"],
                local_cost_certificate_scope=solution["proof_scope"],
                certificate_key="|".join(basis.single_qubit_gates),
                _sequence=operations, _factor=factor)


def _rotation(name, theta, basis):
    """Return (native operations, factor) with R_axis(theta)=factor*circuit."""
    local = basis.single_qubit_gates
    if name in local:
        return (_op(name,theta),), sp.S.One
    if not theta.free_symbols:
        row = _best_clifford(gates.make_gate(name,theta),basis,False)
        if row is not None:
            return row["_sequence"], row["_factor"]
    if "U3" in local or "U" in local:
        u = "U3" if "U3" in local else "U"
        if name == "Rz":
            return (_op(u,0,0,theta),), sp.exp(-sp.I*theta/2)
        return ((_op(u,theta,0,0) if name == "Ry" else _op(u,theta,-sp.pi/2,sp.pi/2)),), sp.S.One
    if "PhasedXZ" in local:
        parameters = (0,theta/sp.pi,0) if name == "Rz" else (theta/sp.pi,0,sp.Rational(1,2) if name == "Ry" else 0)
        return (_op("PhasedXZ",*parameters),), sp.exp(-sp.I*theta/2)
    if "PhasedX" in local and name != "Rz":
        return (_op("PhasedX",theta/sp.pi,sp.Rational(1,2) if name == "Ry" else 0),), sp.S.One
    if "R" in local:
        if name != "Rz":
            return (_op("R",theta,sp.pi/2 if name == "Ry" else 0),), sp.S.One
        return (_op("R",sp.pi/2,sp.pi/2),_op("R",theta,0),_op("R",-sp.pi/2,sp.pi/2)), sp.S.One
    if {"GPi","GPi2"}.issubset(local):
        z = (_op("GPi",0),_op("GPi",theta/(4*sp.pi)))
        if name == "Rz":
            return z, sp.S.One
        before, after = (0,sp.Rational(1,2)) if name == "Ry" else (-sp.Rational(1,4),sp.Rational(1,4))
        return (_op("GPi2",before),)+z+(_op("GPi2",after),), sp.S.One
    if "Rz" in local and "Ry" in local and name == "Rx":
        return (_op("Rz",sp.pi/2),_op("Ry",theta),_op("Rz",-sp.pi/2)),sp.S.One
    if "Rz" in local and "Rx" in local and name == "Ry":
        return (_op("Rz",-sp.pi/2),_op("Rx",theta),_op("Rz",sp.pi/2)),sp.S.One
    if "Rz" in local and "Sx" in local:
        y = (_op("Sx"),_op("Rz",theta+sp.pi),_op("Sx"),_op("Rz",sp.pi))
        if name == "Ry":
            return y,sp.I
        return (_op("Rz",sp.pi/2),)+y+(_op("Rz",-sp.pi/2),),sp.I
    if {"H","P"}.issubset(local):
        if name == "Rz":
            return (_op("P",theta),),sp.exp(-sp.I*theta/2)
        x = (_op("H"),_op("P",theta),_op("H"))
        if name == "Rx":
            return x,sp.exp(-sp.I*theta/2)
        return (_op("P",-sp.pi/2),)+x+(_op("P",sp.pi/2),),sp.exp(-sp.I*theta/2)
    if "H" in local:
        exponent=sp.simplify(theta/(sp.pi/4))
        if exponent.is_Integer:
            options=[(gate,step) for gate,step in (("S",2),("Sdg",-2),("T",1),("Tdg",-1)) if gate in local]
            words={0:()}
            for unused in range(8):
                for old,word in list(words.items()):
                    for gate,step in options:
                        end=(old+step)%8
                        if end not in words or len(word)+1<len(words[end]):
                            words[end]=word+(_op(gate),)
            word=words.get(int(exponent)%8)
            if word is not None:
                phase=sp.exp(-sp.I*theta/2)
                if name=="Rz": return word,phase
                x=(_op("H"),)+word+(_op("H"),)
                if name=="Rx": return x,phase
                before,fb=_local("Sdg",(),basis)
                after,fa=_local("S",(),basis)
                return before+x+after,sp.simplify(phase*fb*fa)
    raise RuleUnavailable(f"No continuous {name} rule for local alphabet {local}")


def _rotation_form(name, parameters):
    """Source rules in Pauli rotations, retaining the exact scalar factor."""
    p = tuple(parameters)
    if name in {"Rx","Ry","Rz"}:
        return (_op(name,p[0]),),sp.S.One
    if name in {"U","U3","U2"}:
        theta,phi,lam = p if name != "U2" else (sp.pi/2,*p)
        return (_op("Rz",lam),_op("Ry",theta),_op("Rz",phi)),sp.exp(sp.I*(phi+lam)/2)
    if name in {"P","U1","S","Sdg","T","Tdg","Z"}:
        angle = p[0] if p else {"S":sp.pi/2,"Sdg":-sp.pi/2,"T":sp.pi/4,"Tdg":-sp.pi/4,"Z":sp.pi}[name]
        return (_op("Rz",angle),),sp.exp(sp.I*angle/2)
    if name in {"X","Y"}:
        return (_op("Rx" if name == "X" else "Ry",sp.pi),),sp.I
    if name == "H":
        return (_op("Rz",sp.pi),_op("Ry",sp.pi/2)),sp.I
    if name in {"Sx","Sxdg","Sy","Sydg"}:
        sign = -1 if name.endswith("dg") else 1
        return (_op("Rx" if name.startswith("Sx") else "Ry",sign*sp.pi/2),),sp.exp(sp.I*sign*sp.pi/4)
    if name in {"R","GPi","GPi2","PhasedX"}:
        if name == "R": theta,phi = p; factor=sp.S.One
        elif name == "PhasedX": theta,phi=(sp.pi*p[0],sp.pi*p[1]);factor=sp.S.One
        else: theta,phi=(sp.pi if name == "GPi" else sp.pi/2,2*sp.pi*p[0]);factor=sp.I if name == "GPi" else sp.S.One
        return (_op("Rz",-phi),_op("Rx",theta),_op("Rz",phi)),factor
    if name in {"XPow","YPow","ZPow"}:
        t,shift = p if len(p)==2 else (*p,0)
        return (_op({"XPow":"Rx","YPow":"Ry","ZPow":"Rz"}[name],sp.pi*t),),sp.exp(sp.I*sp.pi*t*(shift+sp.Rational(1,2)))
    if name in {"HPow","SqrtH","SqrtHdg"}:
        if name == "HPow": t,shift=p if len(p)==2 else (*p,0)
        else: t,shift=(-sp.Rational(1,2) if name.endswith("dg") else sp.Rational(1,2),0)
        return (_op("Ry",-sp.pi/4),_op("Rz",sp.pi*t),_op("Ry",sp.pi/4)),sp.exp(sp.I*sp.pi*t*(shift+sp.Rational(1,2)))
    if name == "PhasedXPow":
        t,phase,*rest=p; shift=rest[0] if rest else 0
        return (_op("Rz",-sp.pi*phase),_op("Rx",sp.pi*t),_op("Rz",sp.pi*phase)),sp.exp(sp.I*sp.pi*t*(shift+sp.Rational(1,2)))
    if name == "PhasedXZ":
        x,z,a=p
        return (_op("Rz",-sp.pi*a),_op("Rx",sp.pi*x),_op("Rz",sp.pi*(z+a))),sp.exp(sp.I*sp.pi*(x+z)/2)
    if name == "I":
        return (),sp.S.One
    raise RuleUnavailable(f"No symbolic rotation rule for {name}")


def _local(name, parameters, basis):
    name = gates.gate_definition(name).name
    parameters = tuple(map(sp.sympify,parameters))
    target = gates.make_gate(name,*parameters)
    if not target.free_symbols:
        row = _best_clifford(target,basis,False)
        if row:
            return row["_sequence"],row["_factor"]
    if name in basis.single_qubit_gates:
        return (_op(name,*parameters),),sp.S.One
    if name in gates.clifford_decomps:
        source = gates.decomposition_sequence(name)
        factor,output=sp.S.One,[]
        for operation in source:
            block,phase=_local(operation[0],operation[1:-1],basis)
            output.extend(block);factor*=phase
        return _cleanup(output),sp.simplify(factor)
    rotations,factor = _rotation_form(name,parameters)
    output=[]
    for operation in rotations:
        block,phase=_rotation(operation[0],operation[1],basis)
        output.extend(block);factor*=phase
    return _cleanup(output),sp.simplify(factor)


def _physical_scalar(factor,basis,width=1):
    # Preserve a root-of-unity exponential before expanding its real/imaginary
    # parts. Otherwise arg(exp(15*i*pi/32)) can become atan(cot(pi/32)), making
    # an elementary exact phase unnecessarily transcendental to the verifier.
    factor = sp.simplify(sp.sympify(factor))
    if factor == 1:
        return ()
    if not factor.free_symbols:
        row=_best_clifford(factor*sp.eye(2),basis,True)
        if row:
            return row["_sequence"]
    if __package__:
        from .symbolic_native_pauli import _phase
    else:
        from symbolic_native_pauli import _phase
    angle=_phase(factor)
    if angle is None:
        constant,exponential=factor.as_independent(sp.exp,as_Add=False)
        angle=(sp.arg(constant)+sp.simplify(exponential.args[0]/sp.I)
               if exponential.func==sp.exp else sp.arg(factor))
    local=basis.single_qubit_gates
    if "U3" in local or "U" in local:
        u="U3" if "U3" in local else "U"
        return (_op(u,sp.pi,angle,angle+sp.pi),_op(u,sp.pi,0,sp.pi))
    if "PhasedXZ" in local:
        return (_op("PhasedXZ",1,2*angle/sp.pi,-angle/sp.pi),_op("PhasedXZ",1,0,0))
    if {"H","P"}.issubset(local):
        x=(_op("H"),_op("P",sp.pi),_op("H"))
        return (_op("P",angle),)+x+(_op("P",angle),)+x
    if width>=2 and not factor.free_symbols:
        state=exact_state(factor*sp.eye(2))
        if state is not None:
            exponent=(state % 8-group_data()[-1] % 8)%8
            gadgets=_scalar_gadgets(basis)
            if exponent in gadgets:return gadgets[exponent]
            if width>=3:
                gadgets=_scalar_gadgets_three(basis)
                if exponent in gadgets:return gadgets[exponent]
    raise RuleUnavailable("Scalar phase has no implemented physical correction in this basis")


@lru_cache(maxsize=8192)
def _local_options(name,parameters,local_gates):
    # The entangler does not affect a one-wire synthesis problem.
    if __package__:
        from .symbolic_bases import make_basis
    else:
        from symbolic_bases import make_basis
    basis=make_basis("local",local_gates,"CX")
    target=gates.make_gate(name,*parameters)
    state=exact_state(target)
    if state is not None:
        solution=solve_clifford_alphabet(local_gates)
        return tuple((clifford_path(solution,s),sp.exp(sp.I*sp.pi*((state%8-s%8)%8)/4))
                     for s in range(state//8*8,state//8*8+8) if s in solution["distances"])
    return (_local(name,parameters,basis),)


@lru_cache(maxsize=512)
def _native_candidates(name,parameters,basis,*,even=False,source_operations=None):
    if __package__:
        from .symbolic_rules import source_rule,native_cx
    else:
        from symbolic_rules import source_rule,native_cx
    tokens=[]
    factor=sp.S.One
    def expand(name,args,wires,depth=0):
        nonlocal factor
        if depth>40:raise RuleUnavailable("Cyclic or excessively deep source construction")
        name=gates.gate_definition(name).name
        if name=="GP":
            factor*=sp.exp(sp.I*args[0]);return
        if len(wires)==1:
            options=_local_options(name,tuple(args),basis.single_qubit_gates)
            if len(options)==1:
                factor*=options[0][1]
                options=((options[0][0],sp.S.One),)
            tokens.append(tuple((tuple((*op[:-1],tuple(wires[q] for q in op[-1])) for op in seq),f) for seq,f in options))
            return
        if len(wires)==2 and (name==basis.entangler or name in {"CNOT","CX"} and basis.entangler in {"CNOT","CX"}):
            native_operation=(basis.entangler,*args,tuple(wires))
            tokens.append((((native_operation,),sp.S.One),))
            return
        if name in {"CNOT","CX"}:
            source,f=native_cx(basis.entangler)
        else:
            proposal=None
            if (len(wires)==2 and basis.entangler in {"iSWAP","SiSWAP","SSWAP","Rxx","Rzz","Rzx","ZZPhase","fSim","TK2"}
                    and name in {"Canonical","fSim","PhasedFSim","Rxx","Ryy","Rzz","SSWAP","SSWAPdg","SiSWAP","SiSWAPdg","SWAPPow","ISWAPPow","PhasedISWAPPow"}):
                if __package__:
                    from .symbolic_native_pauli import native_cartan_candidate
                else:
                    from symbolic_native_pauli import native_cartan_candidate
                proposal=native_cartan_candidate(gates.make_gate(name,*args),basis)
            if proposal is not None:source,f=proposal["sequence"],proposal["factor"]
            else:source,f=source_rule(name,args,even_cartan=even)
        factor*=f
        for operation in source:
            expand(operation[0],operation[1:-1],tuple(wires[q] for q in operation[-1]),depth+1)
    width=gates.make_gate(name,*parameters).rows.bit_length()-1
    if source_operations is None:
        expand(name,parameters,tuple(range(width)))
    else:
        for operation in source_operations:expand(operation[0],operation[1:-1],operation[-1])
    # A frontier of eight scalar phases avoids independently selecting cheap
    # local Cliffords whose accumulated phase is expensive or unreachable.
    frontier={0:((),sp.S.One)}
    for options in tokens:
        updated={}
        for old_sequence,old_factor in frontier.values():
            for sequence,f in options:
                proposal=_cleanup(old_sequence+sequence)
                total=sp.simplify(old_factor*f)
                state=exact_state(total*sp.eye(2))
                key=(state % 8-group_data()[-1] % 8) % 8 if state is not None else None
                # Non-Clifford scalar factors are common to every alternative;
                # retaining them as exact expressions preserves the frontier.
                if key is None:key=total
                if key not in updated or resource_key(proposal)<resource_key(updated[key][0]):
                    updated[key]=(proposal,total)
        frontier=updated
    result=[(sequence,sp.simplify(factor*f)) for sequence,f in frontier.values()]
    if source_operations is None:
        if __package__:
            from .symbolic_fixed_templates import fixed_template_candidates
        else:
            from symbolic_fixed_templates import fixed_template_candidates
        result.extend(fixed_template_candidates(name,parameters,basis))
    return result


@lru_cache(maxsize=32)
def _scalar_gadgets(basis):
    """Physical eighth-root scalars on two wires, including entangler cost.

    Local SU(2) phases alone are insufficient. The two-CNOT Z-parity gadget
    gives iI; the inverse zero-Cartan three-CNOT circuit gives exp(i*pi/4)I.
    Native conversion and an eight-state phase graph account for all phases.
    """
    if __package__:
        from .symbolic_rules import cartan_source,_inverse
    else:
        from symbolic_rules import cartan_source,_inverse
    imaginary=(("Rz",sp.pi,(0,)),("Rz",sp.pi,(1,)),("CX",(0,1)),("Rz",sp.pi,(1,)),("CX",(0,1)))
    eighth,_=cartan_source(0,0,0,force_three=True)
    sources=[(imaginary,2),(_inverse(imaginary),6),(eighth,7),(_inverse(eighth),1)]
    axes={"Rxx":("X","X"),"Rzz":("Z","Z"),"Rzx":("Z","X"),"ZZPhase":("Z","Z")}.get(basis.entangler)
    if axes is not None:
        angle=1 if basis.entangler=="ZZPhase" else sp.pi
        centre=((basis.entangler,angle,(0,1)),)+tuple(("R"+axis,sp.pi,(q,)) for q,axis in enumerate(axes))
        inverse_centre=tuple((op[0],-op[1],op[-1]) for op in reversed(centre))
        sources.extend(((centre,2),(inverse_centre,6)))
    if basis.entangler=="iSWAP":
        centre=(("iSWAP",(0,1)),("Rx",sp.pi,(0,)),("iSWAP",(0,1)),("Rx",-sp.pi,(1,)))
        # Avoid _inverse here: iSWAP's inverse is itself dressed by ZZ.
        sources.extend(((centre,2),(centre*3,6)))
    edges=[]
    solution=solve_clifford_alphabet(basis.single_qubit_gates)
    identity=group_data()[-1]
    for state in range(identity//8*8,identity//8*8+8):
        if state in solution["distances"]:
            exponent=(state%8-identity%8)%8
            if exponent:edges.append((exponent,clifford_path(solution,state)))
    for source,exponent in sources:
        for sequence,factor in _native_candidates("CX",(),basis,source_operations=source):
            scalar=sp.exp(sp.I*sp.pi*exponent/4)/factor
            state=exact_state(scalar*sp.eye(2))
            if state is not None:
                phase=(state%8-identity%8)%8
                if phase:edges.append((phase,sequence))
    paths={0:()}
    for unused in range(8):
        changed=False
        for phase,path in list(paths.items()):
            for shift,sequence in edges:
                end=(phase+shift)%8
                candidate=_cleanup(path+sequence)
                if end not in paths or resource_key(candidate)<resource_key(paths[end]):
                    paths[end]=candidate;changed=True
        if not changed:break
    return paths


def _native_adjoint_word(sequence,basis):
    """Adjoints in the same physical alphabet for the phase-centre motifs."""
    output=[]
    for op in reversed(sequence):
        name,args,wires=op[0],op[1:-1],op[-1]
        if name in {"Rx","Ry","Rz","Rxx","Rzz","Rzx","P","U1","IonQZZ","ZZPhase"}:
            output.append((name,-args[0],wires))
        elif name in {"U","U3"}:
            theta,phi,lam=args
            output.append((name,-theta,-lam,-phi,wires))
        elif name in {"R","PhasedX"}:
            output.append((name,-args[0],args[1],wires))
        elif name=="TK2":output.append((name,*(-value for value in args),wires))
        elif name in {"H","X","Y","Z","CX","CNOT","CZ","ECR","GPi"}:output.append(op)
        elif name=="GPi2":output.append((name,args[0]+sp.Rational(1,2),wires))
        elif name in {"IonQMS","IonQMSMax"}:
            output.append((name,args[0]+sp.Rational(1,2),*args[1:],wires))
        elif name in {"XX90","ZZMax"}:
            frame,_=_rotation("Rz" if name=="XX90" else "Rx",sp.pi,basis)
            frame=tuple((*local[:-1],(wires[0],)) for local in frame)
            output.extend(frame+(op,)+_native_adjoint_word(frame,basis))
        else:raise RuleUnavailable(f"No native adjoint phase motif for {name}")
    return tuple(output)


@lru_cache(maxsize=32)
def _scalar_gadgets_three(basis):
    """Reach three-wire eighth-root phases by squaring an involution circuit.

    If Toffoli=f*C, then C*C=f**(-2) I_8. This genuinely three-wire centre
    resource can exist when the desired scalar is impossible on two wires.
    It is an explicit physical word, not a free phase operation.
    """
    import heapq
    identity=group_data()[-1]
    edges=[]
    for phase,word in _scalar_gadgets(basis).items():
        if phase:edges.append((phase,word))
    for sequence,factor in _native_candidates("CCX",(),basis):
        scalar=simplify_exact(1/factor**2)
        state=exact_state(scalar*sp.eye(2))
        if state is None:continue
        phase=(state%8-identity%8)%8
        if not phase:continue
        word=sequence+sequence
        edges.append((phase,word))
        try:edges.append(((-phase)%8,_native_adjoint_word(word,basis)))
        except RuleUnavailable:pass
    # Additive path costs choose a constructive word; no global phase-gadget
    # optimality is claimed. Safe cleanup is applied to the resulting word.
    distances={0:(0,0.0,0)}
    paths={0:()}
    queue=[((0,0.0,0),0)]
    while queue:
        cost,phase=heapq.heappop(queue)
        if distances[phase]!=cost:continue
        for shift,word in edges:
            weight=resource_key(word)
            proposed=tuple(a+b for a,b in zip(cost,weight))
            end=(phase+shift)%8
            if proposed<distances.get(end,(float("inf"),)*3):
                distances[end]=proposed
                paths[end]=paths[phase]+word
                heapq.heappush(queue,(proposed,end))
    return {phase:_cleanup(word) for phase,word in paths.items()}


def _matrix_sources(target,basis,seconds):
    target=sp.ImmutableMatrix(target)
    if __package__:
        from .symbolic_osr_search import osr_source_candidates
        from .symbolic_clifford_two_qubit import clifford_source_candidate
        from .symbolic_native_pauli import native_cartan_candidate
    else:
        from symbolic_osr_search import osr_source_candidates
        from symbolic_clifford_two_qubit import clifford_source_candidate
        from symbolic_native_pauli import native_cartan_candidate
    if target.shape==(4,4):
        if __package__:
            from .symbolic_osr import product_from_form
            from .symbolic_identity import exponential_forms
        else:
            from symbolic_osr import product_from_form
            from symbolic_identity import exponential_forms
        forms,inverse=exponential_forms(tuple(target))
        if product_from_form(sp.ImmutableMatrix(4,4,forms),(0,),tuple(inverse)) is not None:
            local,search=osr_source_candidates(target,basis,seconds=seconds,max_depth=0)
            if local:return local,search
    clifford=clifford_source_candidate(target,basis)
    if clifford is not None:
        return [clifford],dict(termination="exact_native_Clifford_coset_search",
                               proof_scope=clifford["proof_scope"],discrete_minimum=clifford["discrete_minimum"],
                               reachable_cosets=clifford["reachable_cosets"])
    cartan=native_cartan_candidate(target,basis)
    if cartan is not None:
        return [cartan],dict(termination="native_commuting_Pauli_network_found",
                             proof_scope=cartan["proof_scope"],coordinates=cartan["coordinates"])
    sources,search=osr_source_candidates(target,basis,seconds=seconds)
    if not sources and target.shape==(4,4) and not target.free_symbols:
        # A logical Clifford skeleton is a constructive fallback when the
        # native pulse alphabet is non-Clifford or continuously parameterized.
        # Its discrete CX distance is not a minimum in that native basis.
        logical=clifford_source_candidate(target,get_basis("cx_u"))
        if logical is not None:
            logical["method"]="logical_CNOT_Clifford_skeleton_with_native_rebasing"
            logical["proof_scope"]="logical_Clifford_construction; native optimality is separate"
            sources.append(logical)
    return sources,search


def _lower_matrix_sources(target,basis,phase_exact,seconds):
    sources,search=_matrix_sources(target,basis,seconds)
    prepared=[]
    width=target.rows.bit_length()-1
    for source in sources:
        for sequence,factor in _native_candidates("Identity",(width,),basis,
                                                  source_operations=source["sequence"]):
            factor=sp.simplify(factor*source["factor"])
            try:
                if phase_exact and simplify_exact(factor-1)!=0:
                    sequence=sequence+_physical_scalar(factor,basis,width)
                prepared.append((sequence,factor,source["method"]))
            except RuleUnavailable:continue
    return prepared,search


@lru_cache(maxsize=256)
def _phase_word_proof(factor,sequence,basis,width):
    return verify_circuit(factor*sp.eye(1<<width),sequence,basis,phase_exact=True)


def _verify_phase_candidate(target,sequence,basis,phase_exact,factor):
    """Prove phase closure by exact components, avoiding a redundant product.

    Target=f*C and G=f*I prove G*C=target. Both component identities are
    checked; the physical correction remains in the operation list and cost.
    """
    if phase_exact and simplify_exact(factor-1)!=0:
        width=target.rows.bit_length()-1
        correction=_physical_scalar(factor,basis,width)
        if correction and sequence[-len(correction):]==correction:
            prefix=sequence[:-len(correction)]
            source=verify_circuit(target,prefix,basis,phase_exact=False,scalar_factor=factor)
            phase=_phase_word_proof(factor,correction,basis,width)
            if source["status"]==phase["status"]=="verified_symbolic":
                return dict(status="verified_symbolic",phase_exact=True,cost=circuit_cost(sequence).as_dict(),
                    operations=encode_sequence(sequence),virtual_gp=None,
                    verification="composition_of_exact_symbolic_identities",
                    phase_closure_certificate=dict(prefix_operations=len(prefix),factor=str(factor),
                        prefix_identity="target = factor * prefix",correction_identity="correction = factor * identity",
                        prefix_proof=source["verification"],correction_proof=phase["verification"]))
    return verify_circuit(target,sequence,basis,phase_exact=phase_exact,scalar_factor=None if phase_exact else factor)


@lru_cache(maxsize=256)
def _two_wire_source(name,parameters,basis,seconds):
    """Resynthesize contiguous two-wire source windows before CX rebasing.

    Each replacement has its own exact matrix proof and carries its scalar
    into the enclosing construction. This is a local optimization, not a
    certificate of a globally optimal multi-qubit circuit.
    """
    if __package__:
        from .symbolic_rules import source_rule
        from .symbolic_identity import prove_circuit_identity
    else:
        from symbolic_rules import source_rule
        from symbolic_identity import prove_circuit_identity
    source,factor=source_rule(name,parameters)
    source=tuple(source)
    replacements=[]
    start=0
    output=[]
    while start<len(source):
        operation=source[start]
        if len(operation[-1])!=2:
            output.append(operation);start+=1;continue
        wires=tuple(sorted(operation[-1]))
        end=start+1
        last=start
        while end<len(source) and set(source[end][-1])<=set(wires):
            if len(source[end][-1])==2:last=end
            end+=1
        if last==start:
            output.append(operation);start+=1;continue
        window=source[start:last+1]
        local=tuple((*op[:-1],tuple(wires.index(q) for q in op[-1])) for op in window)
        baseline=min(_native_candidates("Identity",(2,),basis,source_operations=local),
                     key=lambda pair:resource_key(pair[0]))
        target=gates.compile_gates(2,local)
        candidates,_=_lower_matrix_sources(target,basis,False,seconds)
        candidates.sort(key=lambda row:resource_key(row[0]))
        replacement=next((row for row in candidates
                          if resource_key(row[0])<resource_key(baseline[0])
                          and prove_circuit_identity(target,row[0],row[1]) is True),None)
        if replacement is None:
            output.extend(window)
        else:
            word,phase,method=replacement
            output.extend((*op[:-1],tuple(wires[q] for q in op[-1])) for op in word)
            factor*=phase
            replacements.append(dict(wires=list(wires),before_entanglers=resource_key(baseline[0])[0],
                                     after_entanglers=resource_key(word)[0],method=method))
        start=last+1
    return tuple(output),sp.simplify(factor),tuple(replacements)


def _attach_osr_certificate(result,target,basis):
    if __package__:
        from .symbolic_osr import interaction_bound
        from .symbolic_minimality import cnot_cost_certificate,fixed_native_one_pulse_exclusion,sqrt_iswap_cost_certificate
    else:
        from symbolic_osr import interaction_bound
        from symbolic_minimality import cnot_cost_certificate,fixed_native_one_pulse_exclusion,sqrt_iswap_cost_certificate
    bound=interaction_bound(target,basis)
    result["osr_certificate"]=bound
    count=result["cost"]["entanglers"]
    result["entangler_optimal"]=False
    if bound["lower_bound"]==count:
        if bound["condition"]=="True":result["entangler_optimal"]=True
        else:result["conditional_entangler_optimality"]=dict(count=count,condition=bound["condition"],source="exact_OSR_cut_bound")
    if result["phase_exact"] and bound["lower_bound"]==0 and count>0:
        if __package__:
            from .symbolic_feasibility import determinant_group
        else:
            from symbolic_feasibility import determinant_group
        if determinant_group(basis,1)==1:
            # Tensor products of SU(2) operators preserve Y^tensor(n) under
            # transpose congruence. A residual scalar can violate this even
            # though every operator-Schmidt rank is one.
            y=gates.make_gate("Y")
            form=sp.kronecker_product(*([y]*(target.rows.bit_length()-1)))
            differences=(target*form*target.T*form-sp.eye(target.rows)).applyfunc(lambda x:sp.simplify(sp.expand_complex(x)))
            witness=next((value for value in differences if value.is_zero is False),None)
            if witness is not None:
                required=2 if not gates.gate_definition(basis.entangler).parameters else 1
                result["phase_resource_lower_bound"]=dict(lower_bound=required,nonzero_witness=str(witness),
                    method="SU2_local_spin_flip_congruence_and_fixed_pulse_OSR")
                if count==required:result["entangler_optimal"]=True
    if (target.shape==(4,4) and count==2 and bound["lower_bound"]>=1
            and bound["condition"]=="True" and not gates.gate_definition(basis.entangler).parameters):
        exclusion=fixed_native_one_pulse_exclusion(sp.ImmutableMatrix(target),gates.make_gate(basis.entangler))
        if exclusion is not None:
            result["entangler_optimal"]=True
            result["entangler_optimality_certificate"]=exclusion
    if target.shape==(4,4) and not target.free_symbols and basis.entangler in {"CX","CNOT","CZ","ECR","XX90","ZZMax","iSWAP"}:
        exact=cnot_cost_certificate(target)
        result["cnot_invariant_lower_bound"]=exact.lower_bound
        if basis.entangler!="iSWAP":
            if exact.lower_bound==count:result["entangler_optimal"]=True
        elif exact.lower_bound==3 and count==3:
            # Two mirror-DCNOT/iSWAP uses simulate exactly the same nonlocal
            # classes as two CNOTs (Ye--Zhang--Guo, mirror-gate theorem).
            result["entangler_optimal"]=True
            result["entangler_optimality_certificate"]=dict(
                lower_bound=3,method="two_iSWAP_mirror_theorem_and_CNOT_invariant_exclusion",
                sources=["https://arxiv.org/abs/quant-ph/0407108","https://arxiv.org/abs/quant-ph/0308045"])
    result["local_cost_optimal"]=False
    if target.shape==(4,4) and basis.entangler=="SiSWAP":
        certificate=sqrt_iswap_cost_certificate(target)
        if certificate is not None:
            result["native_invariant_lower_bound"]=certificate["minimum"]
            if certificate["minimum"]==count:
                result["entangler_optimal"]=True
                result["entangler_optimality_certificate"]=certificate
    return result


def synthesize_matrix(matrix,basis="cx_u",*,phase_exact=True,seconds=2.0):
    """Synthesize an opaque exact operator by native OSR factorization/search.

    A finite search failure remains open. Every returned physical circuit is
    proved against the entire symbolic matrix, including its phase policy.
    """
    basis=get_basis(basis) if isinstance(basis,str) else basis
    target=sp.ImmutableMatrix(matrix)
    if target.rows<4:raise ValueError("Use this OSR entry point for two or more qubits")
    obstruction=feasibility_obstruction(target,basis,phase_exact=phase_exact)
    if obstruction:return dict(obstruction,phase_exact=phase_exact)
    prepared,search=_lower_matrix_sources(target,basis,phase_exact,seconds)
    for sequence,factor,method in sorted(prepared,key=lambda p:resource_key(p[0])):
        result=_verify_phase_candidate(target,sequence,basis,phase_exact,factor)
        if result["status"]=="verified_symbolic":
            result.update(construction_method=method,osr_search=search,optimality_scope="explicit_lower_bound_only")
            return _attach_osr_certificate(result,target,basis)
    return dict(status="unresolved",phase_exact=phase_exact,reason="No verified OSR construction in this search budget",osr_search=search)


def _synthesize_existing(name, parameters=(), basis="cx_u", *, phase_exact=True,osr_seconds=2.0):
    """Return a verified result, a proved obstruction, or an explicit open case."""
    basis=get_basis(basis) if isinstance(basis,str) else basis
    definition=gates.gate_definition(name)
    parameters=tuple(parameters)
    target=gates.make_gate(name,*parameters)
    obstruction=feasibility_obstruction(target,basis,phase_exact=phase_exact)
    if obstruction:
        return dict(obstruction,phase_exact=phase_exact)
    if target.rows==2:
        if definition.name=="RV":
            if __package__:
                from .symbolic_rotation_vector import synthesize_rotation_vector
            else:
                from symbolic_rotation_vector import synthesize_rotation_vector
            return synthesize_rotation_vector(parameters,basis,phase_exact=phase_exact)
        clifford=_best_clifford(target,basis,phase_exact)
        if clifford:
            return {k:v for k,v in clifford.items() if not k.startswith("_")}
    if not target.free_symbols:
        if all(simplify_exact(a-b)==0 for a,b in zip(target,sp.eye(target.rows))):
            result=verify_circuit(target,(),basis,phase_exact=phase_exact,scalar_factor=1)
            result.update(entangler_optimal=True,local_cost_optimal=True,optimality_scope="empty_circuit")
            return result
    try:
        if definition.name=="GP":
            sequence,factor=(),sp.exp(sp.I*parameters[0])
        elif target.rows==2:
            sequence,factor=_local(name,parameters,basis)
        elif definition.name==basis.entangler:
            sequence,factor=((name,*parameters,(0,1)),),sp.S.One
            if not target.free_symbols:
                alternatives,search=_lower_matrix_sources(target,basis,phase_exact,osr_seconds)
                alternatives.append((sequence,factor,"direct_native_operation"))
                for seq,phase,method in sorted(alternatives,key=lambda row:resource_key(row[0])):
                    result=_verify_phase_candidate(target,seq,basis,phase_exact,phase)
                    if result["status"]=="verified_symbolic":
                        result.update(construction_method=method,osr_search=search,optimality_scope="explicit_lower_bound_only")
                        return _attach_osr_certificate(result,target,basis)
        else:
            candidates=[]
            errors=[]
            cartan_family=definition.name in {"Canonical","fSim","PhasedFSim","SWAPPow","TK2"}
            for even in ((False,True) if phase_exact and cartan_family else (False,)):
                try:candidates.extend(_native_candidates(name,parameters,basis,even=even))
                except (RuleUnavailable,ValueError,KeyError,ImportError,TypeError,AttributeError) as error:
                    errors.append(str(error))
            prepared=[]
            search=None
            window_records=()
            if target.rows>4 and basis.entangler in {"iSWAP","Rxx","Rzz","Rzx","ZZPhase","fSim","TK2"}:
                try:
                    source,phase,window_records=_two_wire_source(name,parameters,basis,osr_seconds)
                    if window_records:
                        for seq,f in _native_candidates("Identity",(target.rows.bit_length()-1,),basis,source_operations=source):
                            total=sp.simplify(phase*f)
                            if phase_exact and simplify_exact(total-1)!=0:
                                seq=seq+_physical_scalar(total,basis,target.rows.bit_length()-1)
                            prepared.append((seq,total,"OSR_two_wire_window_resynthesis"))
                except (RuleUnavailable,ValueError,KeyError):pass
            if target.rows<=16 and not window_records and (not candidates or min(resource_key(p[0])[0] for p in candidates)>1):
                discovered,search=_lower_matrix_sources(target,basis,phase_exact,osr_seconds)
                prepared.extend(discovered)
            best_native=min((resource_key(p[0])[0] for p in prepared),default=float("inf"))
            for seq,phase in candidates:
                if resource_key(seq)[0]>best_native:continue
                try:
                    if phase_exact and simplify_exact(phase-1)!=0:
                        seq=seq+_physical_scalar(phase,basis,target.rows.bit_length()-1)
                    prepared.append((seq,phase,"analytical_source_construction"))
                except RuleUnavailable as error:errors.append(str(error))
            prepared.sort(key=lambda pair:resource_key(pair[0]))
            for seq,phase,method in prepared:
                result=_verify_phase_candidate(target,seq,basis,phase_exact,phase)
                if result["status"]=="verified_symbolic":
                    result.update(entangler_optimal=False,local_cost_optimal=False,optimality_scope="verified_upper_bound",construction_method=method)
                    if search is not None:result["osr_search"]=search
                    if method=="OSR_two_wire_window_resynthesis":result["window_replacements"]=list(window_records)
                    if method in {"analytical_source_construction","OSR_two_wire_window_resynthesis"}:
                        if __package__:
                            from .symbolic_catalog_motifs import motif_schema
                        else:
                            from symbolic_catalog_motifs import motif_schema
                        schema=motif_schema(definition.name)
                        if schema:result["source_motif"]=dict(schema=schema,scope="constructive_template; optimality is separate")
                    if method!="analytical_source_construction" or target.rows<=8 and result["cost"]["entanglers"]<=2:
                        _attach_osr_certificate(result,target,basis)
                    return result
            return dict(status="unresolved",phase_exact=phase_exact,reason="; ".join(dict.fromkeys(errors)) or "Source identity not proved")
        if phase_exact and factor!=1:
            sequence=sequence+_physical_scalar(factor,basis,target.rows.bit_length()-1)
        result=_verify_phase_candidate(target,sequence,basis,phase_exact,factor)
        if result["status"]=="verified_symbolic":
            result.update(entangler_optimal=(target.rows==2),local_cost_optimal=False,
                          optimality_scope="construction_only_except_zero_entangler_lower_bound")
        return result
    except RuleUnavailable as error:
        return dict(status="unresolved",phase_exact=phase_exact,reason=str(error))


def synthesize(name,parameters=(),basis="cx_u",*,phase_exact=True,osr_seconds=2.0):
    """Keep a proved native chart incumbent while optimizing its cost."""
    selected=get_basis(basis) if isinstance(basis,str) else basis
    incumbent=None
    if selected.entangler in {"B","SiSWAP"}:
        target=gates.make_gate(name,*parameters)
        if target.shape==(4,4) and not feasibility_obstruction(target,selected,phase_exact=phase_exact):
            if __package__:
                from .symbolic_controlled_chart import synthesize_controlled_chart
                from .symbolic_sqrt_iswap_phase import synthesize_controlled_phase
                from .symbolic_native_pauli import _phase
            else:
                from symbolic_controlled_chart import synthesize_controlled_chart
                from symbolic_sqrt_iswap_phase import synthesize_controlled_phase
                from symbolic_native_pauli import _phase
            if selected.entangler=="SiSWAP" and target==sp.diag(1,1,1,target[3,3]):
                angle=_phase(target[3,3])
                if angle is not None and angle.is_real is True:
                    incumbent=synthesize_controlled_phase(angle,selected,phase_exact=phase_exact)
            if incumbent is None:
                incumbent=synthesize_controlled_chart(target,selected,phase_exact=phase_exact)
    if __package__:
        from .symbolic_archive import SymbolicDeadline
    else:
        from symbolic_archive import SymbolicDeadline
    try:result=_synthesize_existing(name,parameters,selected,phase_exact=phase_exact,osr_seconds=osr_seconds)
    except SymbolicDeadline:
        if incumbent is None:raise
        return dict(incumbent,additional_search="wall_clock_budget_exhausted_after_proved_incumbent")
    if incumbent is None:return result
    def key(value):
        cost=value["cost"]
        return cost["entanglers"],float(sp.sympify(cost["parameter_score"])),cost["single_qubit_gates"]
    return incumbent if result["status"]!="verified_symbolic" or key(incumbent)<key(result) else result


def synthesize_phase_pair(name,parameters=(),basis="cx_u"):
    return retain_phase_results(synthesize(name,parameters,basis,phase_exact=True),
                                synthesize(name,parameters,basis,phase_exact=False))
