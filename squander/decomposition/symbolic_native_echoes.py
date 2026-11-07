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
