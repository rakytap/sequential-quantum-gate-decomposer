"""Native two-pulse echoes, proved as unrestricted symbolic identities.

B transports a local Y to a Pauli product even though B is not Clifford.
Its two transported Y operators give the full two-coordinate Cartan plane.
The exchange echo similarly retains a free angle without numerical KAK.
"""

import sympy as sp


def inverse(sequence):
    named={"H":"H","X":"X","Y":"Y","Z":"Z","S":"Sdg","Sdg":"S"}
    return tuple((op[0],-op[1],op[-1]) if op[0] in {"Rx","Ry","Rz"}
                 else (named[op[0]],op[-1]) for op in reversed(sequence))


def _shared_frame(first,second):
    words={
        ("X","Y"):(),
        ("X","Z"):(("Rx",sp.pi/2),),
        ("Y","X"):(("Rz",sp.pi/2),),
        ("Y","Z"):(("S",),("H",)),
        ("Z","X"):(("H",),("S",)),
        ("Z","Y"):(("H",),),
    }
    return tuple((*op,(wire,)) for op in words[first,second] for wire in (0,1))


def b_pauli_echo(axes,angle):
    if __package__:
        from .symbolic_osr_search import _axis_frame,_remap,_inverse_clifford_word
    else:
        from symbolic_osr_search import _axis_frame,_remap,_inverse_clifford_word
    frame=tuple(op for q,source in enumerate(("Z","X"))
                for op in _remap(_axis_frame(source,axes[q]),(q,)))
    # B Y_0 B^dagger = -Z_0 X_1; Z_0 B Z_0 = B^dagger.
    word=(_inverse_clifford_word(frame)+(("Z",(0,)),("B",(0,1)),("Z",(0,)),
          ("Ry",-angle,(0,)),("B",(0,1)))+frame)
    return word,sp.S.One


def b_cartan_source(coordinates):
    values=tuple(map(sp.sympify,coordinates))
    active=[i for i,value in enumerate(values) if value!=0]
    if not active:return (),sp.S.One
    if len(active)==1:
        axis="XYZ"[active[0]]
        return b_pauli_echo((axis,axis),-2*values[active[0]])
    if len(active)==2:
        a,b=(values[i] for i in active)
        frame=_shared_frame(*("XYZ"[i] for i in active))
        before=(("Rx",sp.pi/2,(0,)),("H",(0,)),("Rx",sp.pi/2,(1,)))
        after=inverse(before)
        word=(before+(("Z",(0,)),("B",(0,1)),("Z",(0,)),
              ("Ry",2*a,(0,)),("Ry",2*b,(1,)),("B",(0,1)))+after)
        return inverse(frame)+word+frame,sp.S.One
    fixed=[i for i,value in enumerate(values) if value in (sp.pi/4,-sp.pi/4)]
    if len(fixed)<2:return None
    first,second=fixed[:2]
    third=next(i for i in range(3) if i not in (first,second))
    a,b=values[first],values[second]
    sa,sb=1 if a==sp.pi/4 else -1,1 if b==sp.pi/4 else -1
    c=sa*sb*values[third]
    sign={(1,1):(),(1,-1):(("X",(0,)),),(-1,1):(("Y",(0,)),),
          (-1,-1):(("Z",(0,)),)}[sa,sb]
    word=(("H",(0,)),("X",(1,)),("B",(0,1)),("Ry",-sp.pi/2,(0,)),
          ("Ry",2*c-sp.pi,(1,)),("B",(0,1)),("H",(0,)),("Z",(1,)))
    frame=_shared_frame("XYZ"[first],"XYZ"[second])
    return inverse(frame)+sign+word+sign+frame,sp.S.One


def sqrt_iswap_exchange_source(coordinates):
    values=tuple(map(sp.sympify,coordinates))
    active=[i for i,value in enumerate(values) if value!=0]
    if len(active)!=2:return None
    a,b=(values[i] for i in active)
    if sp.simplify(a-b)==0:sign=()
    elif sp.simplify(a+b)==0:sign=(("X",(0,)),)
    else:return None
    frame=_shared_frame(*("XYZ"[i] for i in active))
    word=(("Rz",-sp.pi/2,(0,)),("Z",(0,)),("SiSWAP",(0,1)),("Z",(0,)),
          ("Rz",2*a,(0,)),("Rz",-2*a,(1,)),("SiSWAP",(0,1)),("Rz",sp.pi/2,(0,)))
    return inverse(frame)+sign+word+sign+frame,sp.S.One


def b_sqrt_swap_source(dagger=False):
    """A five-local two-B motif; its full fixed identity is checked on use."""
    beta=sp.acos(1/sp.sqrt(3))
    if dagger:
        word=(("U3",sp.pi/2,sp.pi/2,-sp.pi,(0,)),("B",(0,1)),
              ("U3",-sp.pi/4,0,0,(0,)),("U3",-sp.pi/3,-beta,-beta,(1,)),
              ("B",(0,1)),("U3",sp.pi/2,0,sp.pi/4,(0,)),
              ("U3",sp.pi/4,sp.pi/2,sp.pi/2,(1,)))
        return word,sp.exp(sp.I*(beta-sp.pi/2))
    # Invert the positive-phase circuit and use B^dagger=Z_0 B Z_0.
    # Its four Zs are absorbed exactly into adjacent U3 phase parameters.
    word=(("U3",-sp.pi/4,-sp.pi/2,-sp.pi/2,(1,)),
          ("U3",-sp.pi/2,3*sp.pi/4,0,(0,)),("B",(0,1)),
          ("U3",sp.pi/3,beta,beta,(1,)),("U3",sp.pi/4,sp.pi,sp.pi,(0,)),
          ("B",(0,1)),("U3",-sp.pi/2,sp.pi,sp.pi/2,(0,)))
    return word,sp.exp(-sp.I*(beta-sp.pi/2))


def native_dcx_source(entangler):
    """Clifford frames around iSWAP, keeping its native two-pulse form."""
    before=(("U3",0,0,-sp.pi/2,(1,)),("U3",sp.pi/2,0,0,(0,)))
    after=(("U3",-sp.pi/2,0,-sp.pi/2,(1,)),("U3",0,0,sp.pi,(0,)))
    if entangler=="SiSWAP":middle=(("SiSWAP",(0,1)),)*2
    elif entangler=="B":middle=b_cartan_source((sp.pi/4,sp.pi/4,0))[0]
    elif entangler=="TK2":middle=(("TK2",-sp.Rational(1,2),-sp.Rational(1,2),0,(0,1)),)
    else:return None
    return before+middle+after,sp.S.One


def sqrt_iswap_berkeley_source():
    """Two SQiSWs and three local rotations, with literal SU(4) phase."""
    beta=2*sp.atan2(2**sp.Rational(-1,4),sp.sqrt(2)*sp.sin(sp.pi/8))
    epsilon=sp.pi-beta
    rx=lambda angle:("U3",angle,-sp.pi/2,sp.pi/2,(0,))
    return (rx(epsilon),("SiSWAP",(0,1)),rx(-beta),("SiSWAP",(0,1)),rx(epsilon)),sp.S.One


def _inverse_sqrt_iswap_word(word):
    """Invert native U3/SQiSW words and absorb inverse-pulse Z frames."""
    expanded=[]
    for op in reversed(word):
        if op[0]=="SiSWAP":expanded.extend((("Z",(0,)),op,("Z",(0,))))
        else:
            _,theta,phi,lam,wire=op
            expanded.append(("U3",-theta,-lam,-phi,wire))
    pending=[[],[]];output=[]
    def flush(wire):
        block=pending[wire]
        local=[op for op in block if op[0]!="Z"]
        if not local:
            if len(block)%2:output.append(("U3",0,0,sp.pi,(wire,)))
        else:
            if len(local)!=1:raise ValueError("Unexpected inverse native layer")
            op=local[0];index=block.index(op)
            before=sum(item[0]=="Z" for item in block[:index])%2
            after=sum(item[0]=="Z" for item in block[index+1:])%2
            output.append(("U3",op[1],op[2]+after*sp.pi,op[3]+before*sp.pi,op[-1]))
        pending[wire]=[]
    for op in expanded:
        if len(op[-1])==1:pending[op[-1][0]].append(op)
        else:flush(0);flush(1);output.append(op)
    flush(0);flush(1)
    return tuple(output)


def sqrt_iswap_sqrt_swap_source(dagger=False):
    """Three SQiSWs; all fixed identities are proved before compiler use."""
    beta=2*sp.atan2(2**sp.Rational(-1,4),sp.sqrt(2)*sp.sin(sp.pi/8))
    epsilon=sp.pi-beta
    rx=lambda angle,wire:("U3",angle,-sp.pi/2,sp.pi/2,(wire,))
    word=(("U3",sp.pi,sp.pi/2,sp.pi/2,(0,)),("SiSWAP",(0,1)),
          ("U3",sp.pi/2,sp.pi/2,sp.pi/2,(0,)),rx(sp.pi/2-epsilon,1),
          ("SiSWAP",(0,1)),rx(-sp.pi,0),rx(-epsilon,1),("SiSWAP",(0,1)),
          rx(sp.pi/2,0),rx(sp.pi/2-epsilon,1))
    if dagger:return word,sp.exp(-sp.I*sp.pi/8)
    return _inverse_sqrt_iswap_word(word),sp.exp(sp.I*sp.pi/8)
