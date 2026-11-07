"""A proved SQiSW Weyl-face chart and its three-pulse Sycamore motif.

The two parity blocks are certified separately before substituting inverse
trigonometric expressions. This keeps the proof algebraic and includes the
edges of the chart. The present public constructor is the fixed SYC target;
this does not assert an unrestricted three-pulse Cartan compiler.
"""

from functools import lru_cache
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_controlled_chart import _zero
    from .symbolic_identity import prove_circuit_identity
else:
    import symbolic_gates as gates
    from symbolic_controlled_chart import _zero
    from symbolic_identity import prove_circuit_identity


def _blocks(te,to,ee,eo):
    even=(gates.make_gate("Rz",ee)*gates.make_gate("Rx",sp.pi/4)*
          gates.make_gate("Rz",-2*te)*gates.make_gate("Rx",sp.pi/4)*gates.make_gate("Rz",ee))
    odd=(gates.make_gate("Rz",eo)*gates.make_gate("Rx",-sp.pi/4)*
         gates.make_gate("Rz",2*to)*gates.make_gate("Rx",-sp.pi/4)*gates.make_gate("Rz",eo))
    result=sp.zeros(4)
    for ids,block,phase in (((0,3),even,sp.exp(sp.I*sp.pi/4)),((1,2),odd,sp.exp(-sp.I*sp.pi/4))):
        for i in range(2):
            for j in range(2):result[ids[i],ids[j]]=phase*block[i,j]
    return result


def _face_word(te,to,ee,eo):
    e0,e1=(ee-eo)/2,(ee+eo)/2
    return (("Rx",e0,(0,)),("Rx",e1,(1,)),("SiSWAP",(0,1)),
            ("Rx",-te-to,(0,)),("Rx",-te+to,(1,)),("SiSWAP",(0,1)),
            ("Rx",e0,(0,)),("Rx",e1,(1,)))


@lru_cache(maxsize=1)
def verify_face_lemma():
    te,to,ee,eo=sp.symbols("face_te face_to face_ee face_eo",real=True)
    hh=sp.kronecker_product(gates.make_gate("H"),gates.make_gate("H"))
    if prove_circuit_identity(hh*_blocks(te,to,ee,eo)*hh,_face_word(te,to,ee,eo)) is not True:return False
    # The first and second parity blocks give Rx(2*(y-z)) and
    # Rx(-2*(y+z)). Prove the Cartan link with free real y,z.
    y,z=sp.symbols("face_y face_z",real=True)
    target=sp.zeros(4)
    for ids,block,phase in (((0,3),gates.make_gate("Rx",2*(y-z)),sp.exp(sp.I*sp.pi/4)),
                            ((1,2),gates.make_gate("Rx",-2*(y+z)),sp.exp(-sp.I*sp.pi/4))):
        for i in range(2):
            for j in range(2):target[ids[i],ids[j]]=phase*block[i,j]
    if not _zero(hh*target*hh-gates.make_gate("Canonical",sp.pi/4,y,z)):return False
    # For d=sin(delta), c=cos(delta)>0 and r=sqrt(cos(2delta)),
    # cos(t)=sqrt(2)*d, sin(t)=r and exp(i*e)=(d +/- i*r)/c.
    # These relations give the diagonal c and off-diagonal +/-i*d.
    d,c,r=sp.symbols("face_d face_c face_r",real=True)
    ideal=sp.groebner((c*c+d*d-1,r*r+2*d*d-1),r,c,d,extension=sp.I)
    if any(ideal.reduce(sp.expand(value))[1]!=0 for value in
           (2*d*d+r*r-1,d*d+r*r-c*c,(d+sp.I*r)*(d-sp.I*r)-c*c)):
        return False
    # atan2 reconstruction is independent of the eventual parameter values.
    # The constructor proves its domain, so the norms here never vanish.
    u,v=sp.symbols("face_u face_v",real=True)
    norm=sp.sqrt(u*u+v*v)
    if sp.simplify(sp.cos(sp.atan2(v,u))-u/norm)!=0:return False
    if sp.simplify(sp.sin(sp.atan2(v,u))-v/norm)!=0:return False
    return True


def _face_angles(y,z):
    differences=(y-z,y+z)
    if any(sp.Le(-sp.pi/4,value) is not sp.S.true or sp.Le(value,sp.pi/4) is not sp.S.true
           for value in differences):raise ValueError("Outside the proved Weyl-face chart")
    delta,sigma=differences
    re,ro=sp.sqrt(sp.cos(2*delta)),sp.sqrt(sp.cos(2*sigma))
    return (sp.atan2(re,sp.sqrt(2)*sp.sin(delta)),
            sp.atan2(ro,sp.sqrt(2)*sp.sin(sigma)),
            sp.atan2(re,sp.sin(delta)),sp.atan2(-ro,sp.sin(sigma)))


@lru_cache(maxsize=1)
def verify_syc_face_motif():
    if not verify_face_lemma():return False
    # Conjugating one SQiSW gives L(0,pi/8,-pi/8). It commutes with
    # L(pi/4,pi/8,pi/12), the point used in the two-pulse face chart.
    g0=gates.make_gate("Y")*gates.make_gate("H")*gates.make_gate("S")
    g1=gates.make_gate("H")*gates.make_gate("S")
    frame=sp.kronecker_product(g1,g0)
    if not _zero(frame*gates.make_gate("SiSWAP")*frame.H-gates.make_gate("Canonical",0,sp.pi/8,-sp.pi/8)):return False
    if not _zero(gates.make_gate("Canonical",sp.pi/4,sp.pi/8,sp.pi/12)*
                 gates.make_gate("Canonical",0,sp.pi/8,-sp.pi/8)-
                 gates.make_gate("Canonical",sp.pi/4,sp.pi/4,-sp.pi/24)):return False
    local=gates.make_gate("Rz",-sp.pi/12)*gates.make_gate("Z")
    if not _zero(gates.make_gate("SYC")-sp.exp(-sp.I*sp.pi/24)*
                 sp.kronecker_product(local,local)*gates.make_gate("Canonical",sp.pi/4,sp.pi/4,-sp.pi/24)):return False
    # Prove each native local fusion with unrestricted angles. The scalar
    # factors multiply to exp(i*pi/24+i*(e0-e1)/2), including the target
    # Cartan scalar and both input G^dagger factors.
    e0,e1=sp.symbols("syc_e0 syc_e1",real=True)
    checks=(
        (g0.H,(("U3",sp.pi/2,sp.pi/2,0,(0,)),),sp.I),
        (g1.H,(("U3",-sp.pi/2,sp.pi/2,0,(0,)),),1),
        (gates.make_gate("Rx",e0)*g0,(("U3",-sp.pi/2,0,-sp.pi/2-e0,(0,)),),-sp.I*sp.exp(sp.I*e0/2)),
        (gates.make_gate("Rx",e1)*g1,(("U3",sp.pi/2,0,-sp.pi/2+e1,(0,)),),sp.exp(-sp.I*e1/2)),
        (local*gates.make_gate("Rx",e0),(("U3",e0,5*sp.pi/12,sp.pi/2,(0,)),),sp.exp(sp.I*sp.pi/24)))
    return all(prove_circuit_identity(target,word,factor) is True for target,word,factor in checks)


def sqrt_iswap_syc_source():
    if not verify_syc_face_motif():raise ArithmeticError("SYC face motif identity failed")
    te,to,ee,eo=_face_angles(sp.pi/8,sp.pi/12)
    e0,e1=(ee-eo)/2,(ee+eo)/2
    word=(("U3",sp.pi/2,sp.pi/2,0,(0,)),("U3",-sp.pi/2,sp.pi/2,0,(1,)),("SiSWAP",(0,1)),
          ("U3",-sp.pi/2,0,-sp.pi/2-e0,(0,)),("U3",sp.pi/2,0,-sp.pi/2+e1,(1,)),("SiSWAP",(0,1)),
          ("U3",-te-to,-sp.pi/2,sp.pi/2,(0,)),("U3",-te+to,-sp.pi/2,sp.pi/2,(1,)),("SiSWAP",(0,1)),
          ("U3",e0,5*sp.pi/12,sp.pi/2,(0,)),("U3",e1,5*sp.pi/12,sp.pi/2,(1,)))
    return word,sp.exp(sp.I*sp.pi/24+sp.I*(e0-e1)/2)
