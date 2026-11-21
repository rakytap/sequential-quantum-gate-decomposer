"""Exact operator-Schmidt analysis, factorization and interaction cut bounds.

Realignment rank is computed over exact exponential rational functions. No
floating singular values, numerical rank tolerance or sampled acceptance is
used. Parameter-dependent rank drops remain explicit minor conditions.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools
import sympy as sp

if __package__:
    from .symbolic_identity import exponential_forms,zero_form
    from .symbolic_gates import make_gate
else:
    from symbolic_identity import exponential_forms,zero_form
    from symbolic_gates import make_gate


def qubit_width(matrix):
    matrix=sp.ImmutableMatrix(matrix)
    width=matrix.rows.bit_length()-1
    if matrix.shape!=(1<<width,1<<width) or matrix.atoms(sp.Float):
        raise ValueError("OSR requires an exact square qubit operator")
    return width


def bipartitions(width):
    """All nontrivial cuts once, using local wire 0 as the little-endian bit."""
    for size in range(1,width):
        for cut in itertools.combinations(range(width),size):
            other=tuple(q for q in range(width) if q not in cut)
            if (len(cut),cut)<(len(other),other):yield cut


def _scatter(index,wires):
    return sum(((index>>bit)&1)<<wire for bit,wire in enumerate(wires))


def realign(matrix,cut):
    """Rows index (A_out,A_in), columns (B_out,B_in), in row-major order."""
    matrix=sp.ImmutableMatrix(matrix)
    width=qubit_width(matrix)
    cut=tuple(sorted(cut))
    if not cut or len(set(cut))!=len(cut) or any(q<0 or q>=width for q in cut) or len(cut)==width:
        raise ValueError("Choose a nonempty proper qubit subset")
    other=tuple(q for q in range(width) if q not in cut)
    da,db=1<<len(cut),1<<len(other)
    result=sp.zeros(da*da,db*db)
    for ao,ai,bo,bi in itertools.product(range(da),range(da),range(db),range(db)):
        result[ao*da+ai,bo*db+bi]=matrix[_scatter(ao,cut)|_scatter(bo,other),
                                           _scatter(ai,cut)|_scatter(bi,other)]
    return sp.ImmutableMatrix(result)


def tensor_on_cut(left,right,cut,width):
    """Inverse of an OSR rank-one realignment, retaining the actual phase."""
    cut=tuple(sorted(cut))
    other=tuple(q for q in range(width) if q not in cut)
    result=sp.zeros(1<<width)
    for ao,ai,bo,bi in itertools.product(range(left.rows),range(left.cols),range(right.rows),range(right.cols)):
        result[_scatter(ao,cut)|_scatter(bo,other),_scatter(ai,cut)|_scatter(bi,other)]=left[ao,ai]*right[bo,bi]
    return sp.ImmutableMatrix(result)


@dataclass(frozen=True)
class SchmidtCertificate:
    cut: tuple
    rank_upper_bound: int
    nonzero_minor: object
    minor_rows: tuple
    minor_columns: tuple
    column_factor: object
    row_factor: object

    @property
    def condition(self):return sp.Ne(self.nonzero_minor,0)

    @property
    def fixed_rank(self):
        return self.rank_upper_bound if self.condition==sp.true else None

    def as_dict(self):
        return dict(cut=list(self.cut),rank_upper_bound=self.rank_upper_bound,
                    rank_lower_bound=self.rank_upper_bound,lower_bound_condition=str(self.condition),
                    nonzero_minor=str(self.nonzero_minor),minor_rows=list(self.minor_rows),
                    minor_columns=list(self.minor_columns),
                    proof="exact_realignment_factorization_and_nonzero_minor")


def schmidt_certificate(matrix,cut):
    return _schmidt_certificate(sp.ImmutableMatrix(matrix),tuple(sorted(cut)))


@lru_cache(maxsize=512)
def _schmidt_certificate(matrix,cut):
    original=realign(matrix,cut)
    forms,inverse=exponential_forms(tuple(original))
    formal=sp.ImmutableMatrix(original.rows,original.cols,forms)
    normal=lambda value:sp.cancel(sp.expand(value))
    zero=lambda value:zero_form(value,tuple(inverse))
    _,columns=formal.rref(iszerofunc=zero,simplify=normal)
    rank=len(columns)
    if rank==0:
        return SchmidtCertificate(cut,0,sp.S.One,(),(),sp.zeros(original.rows,0),sp.zeros(0,original.cols))
    left=formal[:,columns]
    _,rows=left.T.rref(iszerofunc=zero,simplify=normal)
    minor=left[list(rows),:]
    witness=normal(minor.det())
    right=(minor.inv()*formal[list(rows),:]).applyfunc(normal)
    if not all(zero(x) for x in formal-left*right):
        raise ArithmeticError("OSR factorization identity was not proved")
    restore=lambda expression:sp.simplify(expression.xreplace(inverse))
    return SchmidtCertificate(cut,rank,restore(witness),tuple(rows),tuple(columns),
                              left.applyfunc(restore),right.applyfunc(restore))


def product_factors(matrix,cut):
    """Return exact tensor factors when rank <=1, without dropping a scalar."""
    certificate=schmidt_certificate(matrix,cut)
    if certificate.rank_upper_bound!=1:return None
    width=qubit_width(matrix)
    da,db=1<<len(cut),1<<(width-len(cut))
    return (sp.ImmutableMatrix(da,da,list(certificate.column_factor[:,0])),
            sp.ImmutableMatrix(db,db,list(certificate.row_factor[0,:])))


def product_from_form(matrix,cut,variables=()):
    """Fast exact rank-one screen inside a shared exponential domain.

    Every 2x2 minor involving a nonzero pivot is checked. Returning a pair
    supplies a factorization witness; a failed screen is not an impossibility
    certificate for the synthesis problem.
    """
    rearranged=realign(matrix,cut)
    pivot=next(((i,j) for i in range(rearranged.rows) for j in range(rearranged.cols)
                if rearranged[i,j]!=0 and not zero_form(rearranged[i,j],variables)),None)
    if pivot is None:return None
    i,j=pivot
    value=rearranged[i,j]
    for row,column in itertools.product(range(rearranged.rows),range(rearranged.cols)):
        if not zero_form(rearranged[row,column]*value-rearranged[row,j]*rearranged[i,column],variables):
            return None
    da,db=1<<len(cut),1<<(qubit_width(matrix)-len(cut))
    left=sp.ImmutableMatrix(da,da,list(rearranged[:,j]))
    right=sp.ImmutableMatrix(db,db,[sp.cancel(x/value) for x in rearranged[i,:]])
    return left,right


def profile(matrix):
    matrix=sp.ImmutableMatrix(matrix)
    return tuple(schmidt_certificate(matrix,cut) for cut in bipartitions(qubit_width(matrix)))


def interaction_bound(matrix,basis):
    """Necessary cut-crossing bound; conditional ranks stay conditional."""
    width=qubit_width(matrix)
    definition_parameters=()
    if __package__:
        from .symbolic_gates import gate_definition
    else:
        from symbolic_gates import gate_definition
    definition=gate_definition(basis.entangler)
    definition_parameters=sp.symbols("native:"+str(len(definition.parameters)),real=True)
    capacity=schmidt_certificate(make_gate(basis.entangler,*definition_parameters),(0,)).rank_upper_bound
    if capacity<=1:raise ValueError("The primitive has no nonlocal operator-Schmidt capacity")
    cuts=profile(matrix)
    demands=[]
    conditions=[]
    singleton_sum=0
    components=[set(range(width))]
    for certificate in cuts:
        rank=certificate.rank_upper_bound
        demand=0
        while capacity**demand<rank:demand+=1
        demands.append(demand)
        conditions.append(certificate.condition)
        if len(certificate.cut)==1:singleton_sum+=demand
        if rank==1:
            cut=set(certificate.cut)
            components=[piece for component in components
                        for piece in (component&cut,component-cut) if piece]
    # An operation crosses at most two singleton cuts. With two wires the
    # unique bipartition is only counted once.
    lower=max(demands+[((singleton_sum+1)//2 if width>2 else 0),width-len(components)])
    if not definition.parameters and any(1<row.rank_upper_bound!=capacity for row in cuts):
        # One fixed pulse has its fixed crossing rank; local pre/post factors
        # cannot turn rank four into rank two. This often strengthens sqrt-
        # SWAP/iSWAP/B bounds for controlled rotations to two native pulses.
        lower=max(lower,2)
    return dict(lower_bound=lower,condition=str(sp.And(*conditions)),native_crossing_rank_capacity=capacity,
                cuts=[certificate.as_dict() for certificate in cuts],
                product_components=[sorted(component) for component in components],
                scope="OSR cut constraints; no claim that this bound is tight")
