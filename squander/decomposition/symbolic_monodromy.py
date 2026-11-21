"""Exact fixed-depth native reachability via the SU(4) monodromy inequalities.

Peterson--Crooks--Smith, Quantum 4, 247 (2020). The QLR data is attributed
in symbolic_monodromy_qlr.json (Apache-2.0, accompanying license). The two
rho branches and alcove convention follow their maintained monodromy code.
Floating LP solutions are proposals: rational primal/Farkas certificates
are required for every accepted feasibility or exclusion statement.
The result concerns arbitrary U(2) locals and scalar phase, not local cost.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import sympy as sp


def _dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def _fraction(x):return F(str(x)).limit_denominator(1000000)


@lru_cache(maxsize=1)
def qlr_rows():
    data=json.loads(Path(__file__).with_name('symbolic_monodromy_qlr.json').read_text())
    result=[]
    for r,k,a,b,c,degree in data['table']:
        for left,right in ((a,b),) if a==b else ((a,b),(b,a)):
            row=[degree]+[0]*12
            for shape,offset,sign in ((left,0,-1),(right,4,-1),(c,8,1)):
                for i,value in enumerate(shape):row[offset+k+i+1-value]+=sign
            result.append(tuple([row[0]]+[row[i]-row[4] for i in range(1,4)]+
                                [row[i]-row[8] for i in range(5,8)]+[row[i]-row[12] for i in range(9,12)]))
    return tuple(result)


ALCOVE=((1,-2,-1,-1),(0,0,1,1),(0,1,-1,0),(0,0,1,-1))
def _in_alcove(q):return len(q)==3 and all(_dot(row,(F(1),*q))>=0 for row in ALCOVE)


def branch_constraints(native,target,count,branches):
    """b+A*x >= 0, with rational coefficients; each x is a PU(4) alcove."""
    native,target=tuple(map(F,native)),tuple(map(F,target))
    if count<2 or len(branches)!=count-1 or any(b not in (0,1) for b in branches):raise ValueError('Invalid branch word')
    if not _in_alcove(native) or not _in_alcove(target):raise ValueError('Coordinate outside the normalized alcove')
    d=3*(count-2)
    def const(x):return (F(x),)+(F(0),)*d
    def variable(i):return (F(0),)+tuple(F(j==i) for j in range(d))
    def add(a,b):return tuple(x+y for x,y in zip(a,b))
    def scale(a,k):return tuple(k*x for x in a)
    def combine(coef,forms):
        out=const(coef[0])
        for k,f in zip(coef[1:],forms):out=add(out,scale(f,k))
        return out
    N=tuple(const(x) for x in native);T=tuple(const(x) for x in target)
    intermediates=[tuple(variable(3*i+j) for j in range(3)) for i in range(count-2)]
    chain=[N,*intermediates,T];rows=[]
    for i,branch in enumerate(branches):
        A,C=chain[i],chain[i+1]
        if branch:
            C=(add(C[2],const(F(1,2))),add(scale(add(add(C[0],C[1]),C[2]),-1),const(F(1,2))),add(C[0],const(-F(1,2))))
        rows.extend(combine(row,(*A,*N,*C)) for row in qlr_rows())
    for X in intermediates:rows.extend(combine(row,X) for row in ALCOVE)
    return tuple(rows)


def verify_lp_certificate(rows,certificate):
    rows=tuple(tuple(map(F,r)) for r in rows);d=len(rows[0])-1
    if certificate['kind']=='primal':
        x=tuple(map(F,certificate['point']))
        return len(x)==d and all(_dot(r,(F(1),*x))>=0 for r in rows)
    if certificate['kind']=='farkas':
        weighted=[F(0)]*(d+1)
        for index,weight in certificate['weights']:
            weight=F(weight)
            if not isinstance(index,int) or not 0<=index<len(rows) or weight<0:return False
            weighted=[a+weight*b for a,b in zip(weighted,rows[index])]
        return weighted[0]<0 and all(x==0 for x in weighted[1:])
    return False


def _solve(rows):
    import numpy as np
    from scipy.optimize import linprog
    d=len(rows[0])-1
    if d==0:
        bad=next((i for i,r in enumerate(rows) if r[0]<0),None)
        return dict(kind='primal',point=[]) if bad is None else dict(kind='farkas',weights=[[bad,'1']])
    b=np.array([float(r[0]) for r in rows]);A=np.array([[float(x) for x in r[1:]] for r in rows])
    answer=linprog(np.zeros(d),A_ub=-A,b_ub=b,bounds=[(None,None)]*d,method='highs')
    if answer.success:
        x=[_fraction(v) for v in answer.x];out=dict(kind='primal',point=list(map(str,x)))
        if verify_lp_certificate(rows,out):return out
        active=[i for i,s in enumerate(b+A@answer.x) if abs(s)<1e-7]
        variables=sp.symbols('lp_x:'+str(d))
        solutions=sp.linsolve((sp.Matrix([rows[i][1:] for i in active]),sp.Matrix([-rows[i][0] for i in active])),variables)
        for solution in solutions:
            fixed={s:sp.Rational(x[j].numerator,x[j].denominator) for j,s in enumerate(variables)}
            out['point']=[str(t.subs(fixed)) for t in solution]
            if verify_lp_certificate(rows,out):return out
        return None
    if answer.status!=2:return None
    dual=linprog(np.zeros(len(rows)),A_eq=np.vstack((A.T,b)),b_eq=np.r_[np.zeros(d),-1],bounds=(0,None),method='highs')
    if not dual.success:return None
    support=[i for i,y in enumerate(dual.x) if y>1e-9]
    weights=[_fraction(dual.x[i]) for i in support]
    out=dict(kind='farkas',weights=[[i,str(w)] for i,w in zip(support,weights)])
    if verify_lp_certificate(rows,out):return out
    M=sp.Matrix([[rows[i][j] for i in support] for j in range(1,d+1)]+[[rows[i][0] for i in support]])
    rhs=sp.Matrix([0]*d+[-1]);variables=sp.symbols('lp_y:'+str(len(support)))
    for solution in sp.linsolve((M,rhs),variables):
        fixed={s:sp.Rational(w.numerator,w.denominator) for s,w in zip(variables,weights)}
        out['weights']=[[i,str(w.subs(fixed))] for i,w in zip(support,solution)]
        if verify_lp_certificate(rows,out):return out
    return None


def fixed_depth_certificate(native,target,count):
    native,target=tuple(map(F,native)),tuple(map(F,target))
    if not _in_alcove(native) or not _in_alcove(target):raise ValueError('Invalid alcove point')
    if count in (0,1):return dict(count=count,reachable=target==((F(0),)*3 if count==0 else native),branches=[])
    evidence=[]
    for bits in product((0,1),repeat=count-1):
        rows=branch_constraints(native,target,count,bits);certificate=_solve(rows)
        if certificate is None:return dict(count=count,status='exact_certificate_not_reconstructed')
        assert verify_lp_certificate(rows,certificate)
        evidence.append(dict(branches=list(bits),certificate=certificate))
        if certificate['kind']=='primal':return dict(count=count,reachable=True,branches=evidence[-1:])
    return dict(count=count,reachable=False,branches=evidence)


def verify_depth_certificate(native,target,record):
    count=record['count']
    if count in (0,1):return record==fixed_depth_certificate(native,target,count)
    expected=set(product((0,1),repeat=count-1));seen=set()
    for row in record['branches']:
        bits=tuple(row['branches'])
        if bits not in expected or bits in seen:return False
        seen.add(bits);certificate=row['certificate']
        if not verify_lp_certificate(branch_constraints(native,target,count,bits),certificate):return False
        if (certificate['kind']=='primal')!=record['reachable']:return False
    return bool(seen) if record['reachable'] else seen==expected


def coordinate_invariants(coordinate):
    a,b,c=map(sp.Rational,coordinate);points=(a,b,c,-a-b-c)
    roots=[sp.exp(2*sp.pi*sp.I*p) for p in points];trace=sum(roots)
    return trace**2/16,(trace**2-sum(r*r for r in roots))/4


def certify_coordinate(target):
    """Numeric eigenangles propose rationals; exact invariant equality accepts."""
    import numpy as np
    if __package__:
        from .symbolic_entangler_proof import spin_data,_zero
    else:
        from symbolic_entangler_proof import spin_data,_zero
    target=sp.ImmutableMatrix(target)
    if target.shape!=(4,4) or target.free_symbols or target.atoms(sp.Float):return None
    if not _zero(target.H*target-sp.eye(4)):raise ValueError('Unitarity unproved')
    yy=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]],complex)
    U=np.array(target.evalf(20),complex);U*=np.exp(-.25j*np.angle(np.linalg.det(U)))
    values=sorted(np.mod(np.angle(np.linalg.eigvals(U@yy@U.T@yy))/(2*np.pi),1),reverse=True)
    while sum(values)>1e-7:values=values[1:]+[values[0]-1]
    if values[1]+values[2]<-1e-7:values=[values[2]+.5,values[3]+.5,values[0]-.5,values[1]-.5]
    q=tuple(_fraction(v) for v in values[:3])
    if not _in_alcove(q):return None
    actual=spin_data(target)[2];expected=coordinate_invariants(q)
    if not all(_zero(a-b) for a,b in zip(actual,expected)):return None
    return dict(coordinate=list(map(str,q)),method='exact_complete_Makhlin_invariants',scalar_policy='projective local class')


def verify_coordinate(target,record):
    if __package__:
        from .symbolic_entangler_proof import spin_data,_zero
    else:
        from symbolic_entangler_proof import spin_data,_zero
    q=tuple(map(F,record['coordinate']));target=sp.ImmutableMatrix(target)
    return (_in_alcove(q) and _zero(target.H*target-sp.eye(4)) and
            all(_zero(a-b) for a,b in zip(spin_data(target)[2],coordinate_invariants(q))))
