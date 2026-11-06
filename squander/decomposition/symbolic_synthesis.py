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


def _op(name, *parameters, wire=0):
    return (name, *parameters, (wire,))


def _cleanup(sequence):
    """Safe adjacent cancellation and additive rotations; no phase wrapping."""
    result = []
    additive = {"Rx", "Ry", "Rz", "P", "U1"}
    for operation in sequence:
        if operation[0] in additive and sp.simplify(operation[1]) == 0:
            continue
        if result and operation[0] in additive and result[-1][0] == operation[0] and result[-1][-1] == operation[-1]:
            previous = result.pop()
            angle = sp.simplify(previous[1]+operation[1])
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
    factor = simplify_exact(factor)
    if factor == 1:
        return ()
    if not factor.free_symbols:
        row=_best_clifford(factor*sp.eye(2),basis,True)
        if row:
            return row["_sequence"]
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
            source,f=source_rule(name,args,even_cartan=even)
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
    return [(sequence,sp.simplify(factor*f)) for sequence,f in frontier.values()]


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


def synthesize(name, parameters=(), basis="cx_u", *, phase_exact=True):
    """Return a verified result, a proved obstruction, or an explicit open case."""
    basis=get_basis(basis) if isinstance(basis,str) else basis
    definition=gates.gate_definition(name)
    parameters=tuple(parameters)
    target=gates.make_gate(name,*parameters)
    obstruction=feasibility_obstruction(target,basis,phase_exact=phase_exact)
    if obstruction:
        return dict(obstruction,phase_exact=phase_exact)
    if target.rows==2:
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
        else:
            candidates=[]
            errors=[]
            cartan_family=definition.name in {"Canonical","fSim","PhasedFSim","SWAPPow"}
            for even in ((False,True) if phase_exact and cartan_family else (False,)):
                try:candidates.extend(_native_candidates(name,parameters,basis,even=even))
                except (RuleUnavailable,ValueError,KeyError,ImportError,TypeError,AttributeError) as error:
                    errors.append(str(error))
            prepared=[]
            for seq,phase in candidates:
                try:
                    if phase_exact and simplify_exact(phase-1)!=0:
                        seq=_cleanup(seq+_physical_scalar(phase,basis,target.rows.bit_length()-1))
                    prepared.append((seq,phase))
                except RuleUnavailable as error:errors.append(str(error))
            prepared.sort(key=lambda pair:resource_key(pair[0]))
            for seq,phase in prepared:
                result=verify_circuit(target,seq,basis,phase_exact=phase_exact,
                                      scalar_factor=None if phase_exact else phase)
                if result["status"]=="verified_symbolic":
                    result.update(entangler_optimal=False,local_cost_optimal=False,optimality_scope="verified_upper_bound")
                    return result
            return dict(status="unresolved",phase_exact=phase_exact,reason="; ".join(dict.fromkeys(errors)) or "Source identity not proved")
        if phase_exact and factor!=1:
            sequence=_cleanup(sequence+_physical_scalar(factor,basis,target.rows.bit_length()-1))
        result=verify_circuit(target,sequence,basis,phase_exact=phase_exact,
                              scalar_factor=None if phase_exact else factor)
        if result["status"]=="verified_symbolic":
            result.update(entangler_optimal=(target.rows==2),local_cost_optimal=False,
                          optimality_scope="construction_only_except_zero_entangler_lower_bound")
        return result
    except RuleUnavailable as error:
        return dict(status="unresolved",phase_exact=phase_exact,reason=str(error))


def synthesize_phase_pair(name,parameters=(),basis="cx_u"):
    return retain_phase_results(synthesize(name,parameters,basis,phase_exact=True),
                                synthesize(name,parameters,basis,phase_exact=False))
