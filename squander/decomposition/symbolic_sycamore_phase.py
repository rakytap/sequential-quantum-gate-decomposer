"""Exact two-Sycamore controlled-phase chart in the catalog's convention.

The echo S Rx(eta)_1 S is block diagonal on wire 1. Its relative SU(2)
axis has a constant azimuth 7*pi/12, enabling an explicit real chart.
Numeric KAK or precomputed decimal Euler angles are not used for acceptance.
"""
from functools import lru_cache
import sympy as sp
if __package__:
    from . import symbolic_gates as gates
else:
    import symbolic_gates as gates


@lru_cache(maxsize=1)
def verify_sycamore_phase_lemma():
    u,v,s=sp.symbols('syc_u syc_v syc_s',real=True)
    C=sp.cos(sp.pi/12);c=u/C;w=sp.exp(-sp.I*sp.pi/6).expand(complex=True)
    M0=sp.Matrix([[c,-s],[-s,-c]])
    M1=sp.Matrix([[-c,-s*w],[-s*w,c*w*w]])
    R=sp.expand(M0*M1/w)
    X,Y,Z=(gates.make_gate(n) for n in 'XYZ')
    axis=-s/(4*C)*X+s*C*Y+u/(4*C*C)*Z
    expected=(v*v-u*u)*sp.eye(2)-2*sp.I*u*axis
    def reduce(value):
        value=sp.expand(value).subs(s*s,1-u*u/(C*C)).subs(v*v,1-u*u)
        return sp.simplify(sp.expand(value))
    if any(reduce(value)!=0 for value in R-expected):return False
    if any(reduce(value)!=0 for value in M0*M0-sp.eye(2)):return False
    norm=s*s/(16*C*C)+s*s*C*C+u*u/(16*C**4)
    if reduce(norm-v*v)!=0:return False
    if reduce(c*c+16*s*s*C*C-16*C*C*v*v)!=0:return False
    b,p=sp.symbols('syc_beta syc_phi',real=True)
    V=gates.make_gate('Rz',p)*gates.make_gate('Ry',b)
    spherical=sp.cos(b)*Z+sp.sin(b)*(sp.cos(p)*X+sp.sin(p)*Y)
    return all(sp.trigsimp(sp.expand_complex(a))==0 for a in V*Z*V.H-spherical)


def sycamore_phase_source(theta):
    """CP(theta)=word literally, for every real theta, with two SYCs.

    q is principal in [-pi,pi], t=|q|. The sign branch is a local X
    conjugation plus a one-wire phase, never an inverse native pulse.
    """
    theta=sp.sympify(theta)
    q=sp.atan2(sp.sin(theta),sp.cos(theta));t=sp.Abs(q)
    sign=sp.Piecewise((1,sp.Lt(q,0)),(0,True))
    C=sp.cos(sp.pi/12);c=sp.sin(t/4)/C;s=sp.sqrt(1-c*c)
    eta=2*sp.acos(c);beta=sp.atan2(4*s*C,c);phi=7*sp.pi/12
    x=('U3',sp.pi*sign,0,sp.pi*sign,(1,))
    source=(x,('Ry',beta,(0,)),('Rz',phi,(0,)),('SYC',(0,1)),
            ('Rx',eta,(1,)),('SYC',(0,1)),('Z',(0,)),('Ry',-eta,(0,)),
            ('Rz',-phi,(0,)),('Ry',-beta,(0,)),('P',t/2+sp.pi/6,(1,)),
            x,('P',-t*sign,(0,)))
    return tuple(op for op in source if not (len(op[-1])==1 and op[0] in ('Ry','Rz','Rx','P','U3') and all(p==0 for p in op[1:-1]))),sp.S.One
