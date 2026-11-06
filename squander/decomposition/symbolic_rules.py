"""Constructive exact source rules and native CNOT frames.

These rules supply upper bounds. Optimal interaction counts and optimal local
costs are separate claims and must have their own certificates.
"""

from functools import lru_cache
import itertools
import json
from pathlib import Path
import sympy as sp

if __package__:
    from . import symbolic_gates as g
else:
    import symbolic_gates as g


def _remap(sequence,wires):
    return tuple((*op[:-1],tuple(wires[q] for q in op[-1])) for op in sequence)


def cartan_source(a,b,c,*,even=False,force_three=False):
    if not force_three and (c==0 or even):
        sequence=[("Rx",-sp.pi/2,(0,)),("Rx",-sp.pi/2,(1,)),("CX",(0,1)),
                  ("Rx",-2*a,(0,)),("Rz",-2*b,(1,)),("CX",(0,1)),
                  ("Rx",sp.pi/2,(0,)),("Rx",sp.pi/2,(1,))]
        if c!=0:
            sequence.extend([("CX",(0,1)),("Rz",-2*c,(1,)),("CX",(0,1))])
        return tuple(sequence),sp.S.One
    # Exact three-CNOT Cartan identity. Its fixed scalar is essential: with
    # SU(2)-only locals the odd CNOT determinant prevents literal equality.
    return (("Rx",sp.pi/2,(0,)),("CX",(0,1)),("Rx",sp.pi/2-2*a,(0,)),
            ("Ry",sp.pi/2-2*b,(1,)),("CX",(1,0)),("Rx",-sp.pi/2,(1,)),
            ("Rz",sp.pi/2-2*c,(1,)),("CX",(0,1))),sp.exp(sp.I*sp.pi/4)


@lru_cache(maxsize=1)
def cx_frames():
    return json.loads(Path(__file__).with_name("symbolic_cx_frames.json").read_text())


def native_cx(entangler):
    """CNOT = factor * frame sequence; all frame entries are exact."""
    equivalents={"Rxx":("XX90",("Rxx",sp.pi/2)),"Rzz":("ZZ90",("Rzz",sp.pi/2)),
                 "ZZMax":("ZZ90",("ZZMax",)),"ZZPhase":("ZZ90",("ZZPhase",sp.Rational(1,2))),
                 "IonQZZ":("ZZ90",("IonQZZ",sp.Rational(1,4))),
                 "IonQMS":("XX90",("IonQMS",0,0,sp.Rational(1,4))),
                 "IonQMSMax":("XX90",("IonQMSMax",0,0)),
                 "TK2":("XX90",("TK2",sp.Rational(1,2),0,0)),
                 "fSim":("CZ",("fSim",0,sp.pi))}
    if entangler=="SYC":
        # SYC**6 = Z0 Z1 CZ. This is a constructive fallback, not the known
        # two-SYC optimum; the result is deliberately labelled an upper bound.
        return (("H",(1,)),)+(("SYC",(0,1)),)*6+(("Z",(0,)),("Z",(1,)),("H",(1,))),sp.S.One
    key,replacement=equivalents.get(entangler,(entangler,None))
    frame=cx_frames()[key]
    sequence=[]
    for op in frame["operations"]:
        name=op["gate"]
        parameters=tuple(sp.sympify(x) for x in op["parameters"])
        if replacement is not None and name==key:
            name,*parameters=replacement
        sequence.append((name,*parameters,tuple(op["wires"])))
    return tuple(sequence),sp.sympify(frame["factor"])


def pauli_rotation(label,theta):
    active=[q for q,p in enumerate(reversed(label)) if p!="I"]
    if not active:return (),sp.exp(-sp.I*theta/2)
    before=[];after=[]
    for q,p in enumerate(reversed(label)):
        if p=="X":before.append(("H",(q,)));after.insert(0,("H",(q,)))
        elif p=="Y":
            before.extend([("Sdg",(q,)),("H",(q,))]);after[0:0]=[("H",(q,)),("S",(q,))]
    target=active[-1]
    parity=[("CX",(q,target)) for q in active[:-1]]
    return tuple(before+parity+[("Rz",theta,(target,))]+list(reversed(parity))+after),sp.S.One


def _qft(width):
    sequence=[]
    for target in reversed(range(width)):
        sequence.append(("H",(target,)))
        for control in reversed(range(target)):
            sequence.append(("CP",sp.pi/sp.Integer(2)**(target-control),(control,target)))
    sequence.extend(("SWAP",(q,width-1-q)) for q in range(width//2))
    return tuple(sequence)


def _inverse(sequence):
    result=[]
    pairs={"S":"Sdg","Sdg":"S","T":"Tdg","Tdg":"T","Sx":"Sxdg","Sxdg":"Sx"}
    for op in reversed(sequence):
        name,*parameters=op[:-1]
        if name in {"Rx","Ry","Rz","P","CP","CRX","CRY","CRZ"}:parameters=[-parameters[0]]
        elif name in pairs:name=pairs[name]
        elif name in {"H","X","Y","Z","CX","CNOT","CZ","CCX","SWAP"}:pass
        elif not parameters and name in g.gate_inverses:name=g.gate_inverses[name][0]
        else:raise ValueError(f"No inverse source rule for {name}")
        result.append((name,*parameters,op[-1]))
    return tuple(result)


def source_rule(name,parameters,*,even_cartan=False):
    p=tuple(parameters)
    if name in {"B","DB","iSWAP","SiSWAP","SSWAP"}:
        values={"B":(sp.pi/4,sp.pi/8,0),"DB":(-3*sp.pi/16,-3*sp.pi/16,0),
                "iSWAP":(sp.pi/4,sp.pi/4,0),"SiSWAP":(sp.pi/8,sp.pi/8,0)}
        if name in values:return cartan_source(*values[name],even=even_cartan)
    if name=="Canonical":return cartan_source(*p,even=even_cartan)
    if name=="fSim":
        theta,phi=p
        sequence,factor=cartan_source(-theta/2,-theta/2,-phi/4,even=even_cartan)
        return sequence+(("Rz",-phi/2,(0,)),("Rz",-phi/2,(1,))),factor*sp.exp(-sp.I*phi/4)
    if name=="PhasedFSim":
        theta,zeta,chi,gamma,phi=p
        pre=(("P",(zeta+chi)/2,(0,)),("P",-(zeta+chi)/2,(1,)))
        post=(("P",(zeta-chi)/2-gamma,(0,)),("P",-(zeta-chi)/2-gamma,(1,)))
        return pre+(("fSim",theta,phi,(0,1)),)+post,sp.S.One
    if name=="ECR":
        frame=cx_frames()["ECR"]
        operations=tuple((op["gate"],*(sp.sympify(x) for x in op["parameters"]),tuple(op["wires"])) for op in frame["operations"])
        index=next(i for i,op in enumerate(operations) if op[0]=="ECR")
        return _inverse(operations[:index])+(("CX",(0,1)),)+_inverse(operations[index+1:]),1/sp.sympify(frame["factor"])
    if name in {"Rxx","Ryy","Rzz","Rzx","Rxz"}:
        return pauli_rotation({"Rxx":"XX","Ryy":"YY","Rzz":"ZZ","Rzx":"XZ","Rxz":"ZX"}[name],p[0])
    if name=="PauliRotation":return pauli_rotation(*p)
    if name=="Pauli":
        return tuple((axis,(q,)) for q,axis in enumerate(reversed(p[0])) if axis!="I"),sp.S.One
    if name in {"Identity","WalshHadamard"}:
        return (() if name=="Identity" else tuple(("H",(q,)) for q in range(p[0]))),sp.S.One
    if name in {"QFT","QFTdg"}:
        sequence=_qft(p[0]);return (_inverse(sequence) if name.endswith("dg") else sequence),sp.S.One
    if name=="MS":
        width,angle=p
        return tuple(("Rxx",angle,(a,b)) for a in range(width) for b in range(a+1,width)),sp.S.One
    if name=="Unitary":
        for candidate in ("B","SSWAP","SiSWAP","SYC","CX","ECR"):
            if p[0]==g.make_gate(candidate):return ((candidate,(0,1)),),sp.S.One
        raise ValueError("Arbitrary matrix synthesis has no registered symbolic source rule")
    if name=="FSWAP":
        return (("iSWAP",(0,1)),("Sdg",(0,)),("Sdg",(1,))),sp.S.One
    if name=="SWAPPow":
        t,shift=p if len(p)==2 else (*p,0)
        sequence,factor=cartan_source(-sp.pi*t/4,-sp.pi*t/4,-sp.pi*t/4,even=even_cartan)
        return sequence,factor*sp.exp(sp.I*sp.pi*t*(shift+sp.Rational(1,4)))
    if name in {"ISWAPPow","iSWAP_pow"}:
        t=p[0];shift=p[1] if len(p)>1 else 0
        sequence,factor=cartan_source(sp.pi*t/4,sp.pi*t/4,0,even=even_cartan)
        return sequence,factor*sp.exp(sp.I*sp.pi*t*shift)
    if name in {"XXPow","YYPow","ZZPow"}:
        t,shift=p if len(p)==2 else (*p,0)
        return (({"XXPow":"Rxx","YYPow":"Ryy","ZZPow":"Rzz"}[name],sp.pi*t,(0,1)),),sp.exp(sp.I*sp.pi*t*(shift+sp.Rational(1,2)))
    if name in {"XX","YY","ZZ"}:
        return ((name[0],(0,)),(name[0],(1,))),sp.S.One
    if name in {"CZPow","CZPowGate"}:
        return (("CP",sp.pi*p[0],(0,1)),),sp.exp(sp.I*sp.pi*p[0]*(p[1] if len(p)>1 else 0))
    if name=="CXPow":
        t,shift=p if len(p)==2 else (*p,0)
        return (("CRX",sp.pi*t,(0,1)),("P",sp.pi*t/2,(0,))),sp.exp(sp.I*sp.pi*t*shift)
    if name=="Givens":
        return (("xx_plus_yy",2*p[0],-sp.pi/2,(0,1)),),sp.S.One
    if name in {"CSdg","CT","CTdg"}:
        return (("CP",{"CSdg":-sp.pi/2,"CT":sp.pi/4,"CTdg":-sp.pi/4}[name],(0,1)),),sp.S.One
    if name=="Barenco":
        alpha,phi,theta=p
        return (("CR",2*theta,phi,(0,1)),("P",alpha,(0,))),sp.S.One
    if name=="DCX":return (("CX",(0,1)),("CX",(1,0))),sp.S.One
    if name=="ACX":return (("X",(0,)),("CX",(0,1)),("X",(0,))),sp.S.One
    if name in {"SSWAPdg","SiSWAPdg"}:
        base=name[:-2]
        sequence,factor=source_rule(base,(),even_cartan=even_cartan)
        return _inverse(sequence),sp.conjugate(factor)
    if name in g.decomposition_sequences:
        return g.decomposition_sequence(name,*p),sp.S.One
    if name=="MCX" and p[0]==2 and len(p)==1:return (("CCX",(0,1,2)),),sp.S.One
    if name=="MCPhase" and p[1]==1:return (("CP",p[0],(0,1)),),sp.S.One
    # Analytical SDK definitions retain free Parameters; all recovered
    # constants and the completed circuit are checked exactly by the caller.
    if __package__:
        from .symbolic_qiskit_source import analytical_definition
    else:
        from symbolic_qiskit_source import analytical_definition
    return analytical_definition(name,p)
