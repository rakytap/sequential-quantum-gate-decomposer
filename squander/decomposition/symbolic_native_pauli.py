"""Discover native Clifford pulse networks for commuting Pauli interactions.

Binary Pauli transport chooses a two/three-pulse skeleton. Exact OSR recovers
its local inverse, and the full free-parameter circuit is verified elsewhere.
This supplies constructive upper bounds, not an unexplained optimality claim.
"""

from functools import lru_cache
from collections import deque
import itertools
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_clifford import pauli_action,_signed_pauli
    from .symbolic_clifford_search import group_data
    from .symbolic_clifford_search import exact_state
    from .symbolic_clifford_two_qubit import _columns,_local_frames,_multiply,local_coset_key,_apply
    from .symbolic_osr import product_factors
    from .symbolic_osr_search import recognize_local
    from .symbolic_identity import exponential_forms,zero_form
else:
    import symbolic_gates as gates
    from symbolic_clifford import pauli_action,_signed_pauli
    from symbolic_clifford_search import group_data
    from symbolic_clifford_search import exact_state
    from symbolic_clifford_two_qubit import _columns,_local_frames,_multiply,local_coset_key,_apply
    from symbolic_osr import product_factors
    from symbolic_osr_search import recognize_local
    from symbolic_identity import exponential_forms,zero_form


def _images(columns):
    return {columns[0],columns[1],columns[0]^columns[1],columns[2],columns[3],columns[2]^columns[3]}


def _pair(bits):return bits&3,(bits>>2)&3


def _local_word(a,b):
    names=group_data()[0].representatives
    return tuple((names[index],(q,)) for q,index in enumerate((a,b)) if names[index]!="I")


@lru_cache(maxsize=128)
def _skeleton(entangler,axes):
    native=gates.make_gate(entangler)
    action=pauli_action(native)
    if action is None:return None
    pulse=_columns(action)
    wanted={"X":5,"Y":15,"Z":10}
    required={wanted[axis] for axis in axes}
    root=local_coset_key((1,2,4,8))
    frames=_local_frames()
    if len(axes)<=2:
        for a,b,last in frames:
            if not required<=_images(_multiply(last,pulse)):continue
            words=((),(),_local_word(a,b))
            break
        else:return None
        count=2
    else:
        # Before the last pulse, local rotations are transported by E3. At
        # the earlier insertion point they are transported by E3 K2 E2.
        for a,b,middle in frames:
            suffix=_multiply(pulse,_multiply(middle,pulse))
            images={x for x in _images(suffix)|_images(pulse) if all(_pair(x))}
            triples=[tr for tr in itertools.combinations(images,3)
                     if len({_pair(x)[0] for x in tr})==3 and len({_pair(x)[1] for x in tr})==3]
            if not triples:continue
            early=None
            for c,d,first in frames:
                total=_multiply(suffix,_multiply(first,pulse))
                if local_coset_key(total)==root:early=(c,d);break
            if early is None:continue
            last_choice=None
            for e,f,last in frames:
                if any({_apply(last,x) for x in triple}==required for triple in triples):
                    last_choice=(e,f);break
            if last_choice is not None:
                words=(_local_word(*early),_local_word(a,b),_local_word(*last_choice))
                break
        else:return None
        count=3
    first,middle,last=words
    pulse_word=((entangler,(0,1)),)
    if count==2:
        base=pulse_word+pulse_word+last
        late=gates.compile_gates(2,pulse_word+last)
        positions=(late,)
    else:
        base=pulse_word+first+pulse_word+middle+pulse_word+last
        late=gates.compile_gates(2,middle+pulse_word+last)
        early=gates.compile_gates(2,first+pulse_word+middle+pulse_word+last)
        positions=(late,early)
    inverse=gates.compile_gates(2,base).H
    pair=product_factors(inverse,(0,))
    if pair is None:return None
    left,right=(recognize_local(part,()) for part in pair)
    if left is None or right is None:return None
    prefix=left[0]+tuple((*op[:-1],(1,)) for op in right[0])
    factor=sp.simplify(left[1]*right[1])
    insertions={}
    for position,suffix in enumerate(positions):
        for q,axis in itertools.product(range(2),"XYZ"):
            operator=gates.apply_to(sp.eye(4),2,gates.make_gate(axis),(q,))
            image=(suffix*operator*suffix.H).applyfunc(lambda x:sp.simplify(sp.expand_complex(x)))
            signed=_signed_pauli(image)
            if signed is None:return None
            sign,label=signed
            if label in {axis*2 for axis in axes} and label not in insertions:
                insertions[label]=(position,q,axis,sign)
    if not {axis*2 for axis in axes}<=set(insertions):
        raise ArithmeticError("Native skeleton failed exact signed Pauli transport")
    return dict(count=count,prefix=prefix,first=first,middle=middle,last=last,
                factor=factor,insertions=insertions)


def cartan_native_source(a,b,c,entangler):
    """Construct exp(i(aXX+bYY+cZZ)) with native pulses, without rebasing CX."""
    values=dict(zip("XYZ",map(sp.sympify,(a,b,c))))
    axes=tuple(axis for axis in "XYZ" if values[axis]!=0)
    if not axes:return (),sp.S.One
    skeleton=_skeleton(entangler,axes)
    if skeleton is None:return None
    rotations=[[],[]]
    for axis in axes:
        position,q,local,sign=skeleton["insertions"][axis*2]
        rotations[position].append(("R"+local,-2*sign*values[axis],(q,)))
    pulse=((entangler,(0,1)),)
    if skeleton["count"]==2:
        word=skeleton["prefix"]+pulse+tuple(rotations[0])+pulse+skeleton["last"]
    else:
        word=(skeleton["prefix"]+pulse+tuple(rotations[1])+skeleton["first"]+pulse+
              tuple(rotations[0])+skeleton["middle"]+pulse+skeleton["last"])
    return word,skeleton["factor"]


def _phase(expression):
    expression=sp.simplify(sp.expand(expression.rewrite(sp.exp)))
    if not expression.free_symbols:
        magnitude=sp.simplify(sp.expand_complex(expression*sp.conjugate(expression)-1))
        if magnitude.is_zero is False:return None
        phase=sp.arg(expression)
        rational_phase=sp.simplify(phase/sp.pi)
        if rational_phase.is_Rational:return sp.pi*rational_phase
        # Index a possible root of unity, then prove it exactly. Numerical
        # phase estimation proposes a rational angle; it never accepts one.
        # This includes third/sixth roots used by SYC, beyond dyadic roots.
        from fractions import Fraction
        import math
        value=complex(expression.evalf(24))
        rational=Fraction(math.atan2(value.imag,value.real)/math.pi).limit_denominator(192)
        proposed=sp.pi*sp.Rational(rational.numerator,rational.denominator)
        if zero_form(expression-sp.exp(sp.I*proposed)):return proposed
        # Exact root-of-unity recovery. Each proposed phase is checked as an
        # algebraic identity; no angle is rounded from a floating argument.
        for denominator in (4,8,16,32):
            for numerator in range(-denominator,denominator):
                proposed=sp.pi*sp.Rational(numerator,denominator)
                if zero_form(expression-sp.exp(sp.I*proposed)):return proposed
        return phase
    phase=sp.S.Zero
    for term in sp.Mul.make_args(expression):
        if term.func==sp.exp:phase+=sp.simplify(term.args[0]/sp.I)
        elif not term.free_symbols:phase+=sp.arg(term)
        else:return None
    forms,inverse=exponential_forms((expression-sp.exp(sp.I*phase),))
    return sp.simplify(phase) if zero_form(forms[0],tuple(inverse)) else None


def bell_coordinates(matrix):
    return _bell_coordinates(sp.ImmutableMatrix(matrix))


@lru_cache(maxsize=512)
def _bell_coordinates(matrix):
    """Infer a verified commuting Cartan kernel and its actual scalar phase."""
    if matrix.shape!=(4,4):return None
    bell=sp.ImmutableMatrix([[1,1,0,0],[0,0,1,1],[0,0,1,-1],[1,-1,0,0]])/sp.sqrt(2)
    diagonal=bell.T*matrix*bell
    forms,inverse=exponential_forms(tuple(diagonal))
    formal=sp.ImmutableMatrix(4,4,forms)
    if any(not zero_form(formal[i,j],tuple(inverse)) for i in range(4) for j in range(4) if i!=j):return None
    phases=[_phase(diagonal[i,i]) for i in range(4)]
    if None in phases:return None
    p,q,r,s=phases
    coordinates=tuple(sp.simplify(x/4) for x in (p-q+r-s,-p+q+r-s,p+q-r-s))
    return coordinates,sp.exp(sp.I*sp.simplify((p+q+r+s)/4))


@lru_cache(maxsize=512)
def diagonal_gauge(matrix):
    """Infer local Z frames for a parity-preserving operator, exactly.

    Equality of each parity block's paired entries is a linear phase system.
    Ratios are only proposals; cross-multiplied identities and the completed
    free-parameter circuit are proved, including zero-amplitude limits.
    """
    if matrix.shape!=(4,4):return None
    allowed={(0,0),(3,3),(1,1),(2,2),(0,3),(3,0),(1,2),(2,1)}
    forms,inverse=exponential_forms(tuple(matrix))
    if any(not zero_form(forms[4*i+j],tuple(inverse)) for i in range(4) for j in range(4) if (i,j) not in allowed):return None
    def relative(a,b):
        if a==0 and b==0:return None
        if a==0 or b==0:return False
        phase=_phase(sp.cancel(a/b))
        if phase is None:return False
        check,inv=exponential_forms((a-sp.exp(sp.I*phase)*b,))
        return phase if zero_form(check[0],tuple(inv)) else False
    d0=relative(matrix[0,0],matrix[3,3])
    d1=relative(matrix[1,1],matrix[2,2])
    d2=relative(matrix[1,2],matrix[2,1])
    d3=relative(matrix[0,3],matrix[3,0])
    if any(value is False for value in (d0,d1,d2,d3)):return None
    d0=sp.S.Zero if d0 is None else d0
    d1=sp.S.Zero if d1 is None else d1
    d2=sp.S.Zero if d2 is None else d2
    d3=d0 if d3 is None else d3
    lsum,rsum=(d0+d3)/2,(d0-d3)/2
    ldiff,rdiff=-(d1+d2)/2,-(d1-d2)/2
    left=tuple(sp.simplify(x/2) for x in (lsum+ldiff,lsum-ldiff))
    right=tuple(sp.simplify(x/2) for x in (rsum+rdiff,rsum-rdiff))
    before=tuple(("Rz",angle,(q,)) for q,angle in enumerate(right) if angle!=0)
    after=tuple(("Rz",angle,(q,)) for q,angle in enumerate(left) if angle!=0)
    kernel=gates.compile_gates(2,after)*matrix*gates.compile_gates(2,before)
    result=bell_coordinates(kernel)
    if result is None:return None
    return result,before,after


def _variable_cartan_source(coordinates,basis):
    entangler=basis.entangler
    if entangler=="TK2":
        return (("TK2",*(-2*value/sp.pi for value in coordinates),(0,1)),),sp.S.One
    if entangler=="fSim":
        a,b,c=coordinates
        def equal_pair(value,z):
            return (("fSim",-2*value,-4*z,(0,1)),("Rz",-2*z,(0,)),("Rz",-2*z,(1,))),sp.exp(-sp.I*z)
        first,factor=equal_pair((a+b)/2,c)
        if sp.simplify(a-b)==0:return first,factor
        second,_=equal_pair((a-b)/2,0)
        return first+(("X",(0,)),)+second+(("X",(0,)),),factor
    if __package__:
        from .symbolic_osr_search import _axis_frame,_inverse_clifford_word,_native_parameters,_remap
    else:
        from symbolic_osr_search import _axis_frame,_inverse_clifford_word,_native_parameters,_remap
    base={"Rxx":("X","X"),"Rzz":("Z","Z"),"Rzx":("Z","X"),"ZZPhase":("Z","Z")}.get(entangler)
    if base is None:return None
    word=[]
    for axis,value in zip("XYZ",coordinates):
        if value==0:continue
        frame=tuple(op for q in range(2) for op in _remap(_axis_frame(base[q],axis),(q,)))
        word.extend(_inverse_clifford_word(frame))
        word.append((entangler,*_native_parameters(entangler,-2*value),(0,1)))
        word.extend(frame)
    return tuple(word),sp.S.One


def _projective_index(matrix):
    """Numerical proposal index only; the selected word is then proved exactly."""
    values=[complex(value.evalf(16)) for value in matrix]
    pivot=max(range(len(values)),key=lambda i:round(abs(values[i]),9))
    scale=values[pivot]
    return pivot,tuple((round((value/scale).real,9),round((value/scale).imag,9)) for value in values)


@lru_cache(maxsize=16)
def _native_power_frames(entangler):
    frames={}
    pulse=((entangler,(0,1)),)
    for count in (1,2,3):
        for first,second in itertools.product(("I","X","Y","Z"),repeat=2):
            frame=tuple((name,(q,)) for q,name in enumerate((first,second)) if name!="I")
            word=frame+pulse*count+frame
            matrix=sp.ImmutableMatrix(gates.compile_gates(2,word)).applyfunc(sp.simplify)
            key=_projective_index(matrix)
            if key not in frames:frames[key]=(word,matrix)
    return frames


def _fixed_power_source(coordinates,basis):
    if basis.entangler not in {"SiSWAP","SSWAP"} or any(value.free_symbols for value in coordinates):return None
    target=gates.make_gate("Canonical",*coordinates)
    key=_projective_index(target)
    candidate=_native_power_frames(basis.entangler).get(key)
    if candidate is None:return None
    word,matrix=candidate
    pivot=key[0]
    factor=sp.simplify(target[pivot]/matrix[pivot])
    if __package__:
        from .symbolic_identity import prove_circuit_identity
    else:
        from symbolic_identity import prove_circuit_identity
    if prove_circuit_identity(target,word,factor) is True:return word,factor
    return None


@lru_cache(maxsize=3)
def _commuting_fixed_lattice(entangler):
    """Shortest pulse words in a finite commuting Clifford-frame proposal set.

    Coordinates have units pi/8 and are reduced modulo pi, which changes
    only a scalar. Full matrix verification recovers/checks that scalar.
    This search does not certify global optimality over continuous frames.
    """
    coefficients={"SiSWAP":(1,1,0),"SSWAP":(-1,-1,-1),"B":(2,1,0)}[entangler]
    representatives=group_data()[0].representatives
    state_names={state:name for name,state in group_data()[3].items()}
    inverse_names={name:state_names[exact_state(gates.make_gate(name).H)] for name in representatives}
    frames=[]
    paulis=[gates.make_gate(axis) for axis in "XYZ"]
    for name in representatives:
        matrix=gates.make_gate(name)
        images=tuple(_signed_pauli(matrix*p*matrix.H) for p in paulis)
        frames.append((name,images))
    generators={}
    for (a,left),(b,right) in itertools.product(frames,repeat=2):
        if any(x[1]!=y[1] for x,y in zip(left,right)):continue
        vector=[0,0,0]
        for coefficient,x,y in zip(coefficients,left,right):
            vector["XYZ".index(x[1])]+=coefficient*x[0]*y[0]
        vector=tuple(vector)
        frame=tuple((name,(wire,)) for wire,name in enumerate((a,b)) if name!="I")
        inverse=tuple((inverse_names[name],wires) for name,wires in reversed(frame))
        word=inverse+((entangler,(0,1)),)+frame
        if vector not in generators or len(word)<len(generators[vector]):generators[vector]=word
    paths={(0,0,0):()}
    queue=deque([(0,0,0)])
    while queue:
        state=queue.popleft()
        old=paths[state]
        if sum(isinstance(op[0],str) and op[0]==entangler for op in old)>=3:continue
        for vector,word in generators.items():
            following=tuple((a+b)%8 for a,b in zip(state,vector))
            if following not in paths:
                paths[following]=old+word
                queue.append(following)
    return paths


def _fixed_lattice_source(coordinates,basis):
    if basis.entangler not in {"SiSWAP","SSWAP","B"}:return None
    units=tuple(sp.simplify(8*value/sp.pi) for value in coordinates)
    if any(value.is_Integer is not True for value in units):return None
    word=_commuting_fixed_lattice(basis.entangler).get(tuple(int(value)%8 for value in units))
    if word is None:return None
    target=gates.make_gate("Canonical",*coordinates)
    actual=gates.compile_gates(2,word)
    if __package__:
        from .symbolic_identity import prove_circuit_identity,_cyclotomic_values
    else:
        from symbolic_identity import prove_circuit_identity,_cyclotomic_values
    converted=_cyclotomic_values(tuple(dict.fromkeys([*actual,*target])))
    if converted is not None:
        domain,values=converted
        pivot=next((i for i,value in enumerate(actual) if values[value]!=domain.zero),None)
        if pivot is None:return None
        factor=sp.simplify(domain.to_sympy(values[target[pivot]]/values[actual[pivot]]))
    else:
        actual=actual.applyfunc(lambda value:sp.simplify(sp.expand_complex(value)))
        pivot=next((i for i,value in enumerate(actual) if value.is_zero is False),None)
        if pivot is None:return None
        factor=sp.simplify(target[pivot]/actual[pivot])
    if prove_circuit_identity(target,word,factor) is True:return word,factor
    return None


def native_cartan_candidate(matrix,basis):
    result=bell_coordinates(matrix)
    before=after=()
    if result is None:
        gauged=diagonal_gauge(matrix)
        if gauged is None:return None
        result,before,after=gauged
    coordinates,phase=result
    if basis.entangler in {"CX","CNOT","CZ","ECR","XX90","ZZMax","iSWAP"}:
        source=cartan_native_source(*coordinates,basis.entangler)
    else:
        if __package__:
            from .symbolic_native_echoes import b_cartan_source,sqrt_iswap_exchange_source
        else:
            from symbolic_native_echoes import b_cartan_source,sqrt_iswap_exchange_source
        echo=(b_cartan_source(coordinates) if basis.entangler=="B" else
              sqrt_iswap_exchange_source(coordinates) if basis.entangler=="SiSWAP" else None)
        source=(_variable_cartan_source(coordinates,basis) or _fixed_lattice_source(coordinates,basis)
                or echo or _fixed_power_source(coordinates,basis))
    if source is None:return None
    before_inverse=tuple(("Rz",-op[1],op[-1]) for op in reversed(before))
    after_inverse=tuple(("Rz",-op[1],op[-1]) for op in reversed(after))
    return dict(sequence=before_inverse+source[0]+after_inverse,factor=sp.simplify(phase*source[1]),
                method="matrix_Bell_spectrum_and_discovered_native_Pauli_network",
                coordinates=[str(x) for x in coordinates],
                proof_scope="exact_symbolic_construction; lower bound is separate")
