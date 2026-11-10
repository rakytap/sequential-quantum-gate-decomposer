"""Four-Ising lower bound for a controlled full-Schmidt-rank two-qubit gate.

The proof relaxes all one-wire gates and scalar phase. In a three-interaction
network the target cuts demand degree at least two and the control degree at
least one. The only surviving edge multisets are a triangle, or two internal
target interactions and one control interaction. The latter has a local
conditional branch and a relative block of Schmidt rank at most two. In a
triangle a nonmonomial middle control frame either gives control rank four,
or reduces a target cut to rank at most two; a monomial control frame leaves
an entangling internal gate in both conditional branches. All contradict
diag(I,V) with OSR(V)=4. See the manuscript's complete normal-form proof.

This is an entangler bound, not a secondary score or local-count minimum.
"""

from functools import lru_cache
import itertools
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_osr import schmidt_certificate,realign
    from .symbolic_controlled_chart import _zero
    from .symbolic_entangler_proof import native_profile
else:
    import symbolic_gates as gates
    from symbolic_osr import schmidt_certificate,realign
    from symbolic_controlled_chart import _zero
    from symbolic_entangler_proof import native_profile


def four_cnot_source():
    p=sp.pi
    return (('Ry',-p/2,(1,)),('CX',(1,2)),
            ('Ry',-p/4,(1,)),('Ry',-3*p/4,(2,)),
            ('CX',(0,1)),('CX',(0,2)),
            ('Ry',p/4,(1,)),('Ry',3*p/4,(2,)),
            ('CX',(1,2)),('Ry',p/2,(1,)))


def three_arbitrary_interaction_source():
    """K, CZ(c,a), K^dagger implements controlled-fSWAP exactly.

    Each K is ONE unrestricted two-qubit operation, not one Ising pulse.
    In little-endian target coordinates K^dagger (I tensor Z) K = F.
    """
    r=sp.sqrt(2)/2
    K=sp.ImmutableMatrix([[1,0,0,0],[0,r,-r,0],[0,r,r,0],[0,0,0,1]])
    return K


def verify_arbitrary_interaction_source():
    K=three_arbitrary_interaction_source()
    local_z=sp.kronecker_product(sp.eye(2),gates.make_gate('Z'))
    F=gates.make_gate('FSWAP')
    if not _zero(K.H*K-sp.eye(4)) or not _zero(K.H*local_z*K-F):return False
    before=gates.apply_to(sp.eye(8),3,K,(1,2))
    after=gates.apply_to(sp.eye(8),3,K.H,(1,2))
    middle=gates.apply_to(sp.eye(8),3,gates.make_gate('CZ'),(0,1))
    return _zero(after*middle*before-gates.make_gate('fFredkin'))


def weyl_class_cost_counterexamples():
    """All V are local, while C(V) has CX costs 0,1,2 respectively.

    Singleton cut ranks 1, (2,1), and (2,2) prove the lower bounds.
    The last circuit needs both target-control edges to reach both targets.
    Thus Weyl(V) alone cannot determine the CNOT cost of controlled-V.
    """
    Z=gates.make_gate('Z');I=sp.eye(2)
    rows=[]
    for name,V,word in [('I',sp.eye(4),()),
                        ('Za',sp.kronecker_product(I,Z),(('CZ',(0,1)),)),
                        ('ZaZb',sp.kronecker_product(Z,Z),(('CZ',(0,1)),('CZ',(0,2))))]:
        target=gates.make_controlled(V,2)
        if not _zero(gates.compile_gates(3,word)-target):raise ArithmeticError('Controlled-local counterexample failed')
        ranks=[schmidt_certificate(target,(q,)).fixed_rank for q in range(3)]
        rows.append(dict(relative_block=name,relative_weyl_class='local',
                         minimum_cx=len(word),singleton_cut_ranks=ranks,
                         proof='exact identity and mandatory nontrivial target cuts'))
    return rows


@lru_cache(maxsize=1)
def verify_controlled_rank_lemmas():
    # Conjugating any one-wire operator by an Ising gate has the other
    # wire's support in span{I,Z}, before and after arbitrary local frames.
    a,b,c,d,C,S=sp.symbols('cr_a cr_b cr_c cr_d cr_C cr_S')
    Z=gates.make_gate('Z');I=sp.eye(2)
    D=C*sp.eye(4)+sp.I*S*sp.kronecker_product(Z,Z)
    M=sp.Matrix([[a,b],[c,d]])
    for source in (sp.kronecker_product(M,I),sp.kronecker_product(I,M)):
        matrix=(C*sp.eye(4)-sp.I*S*sp.kronecker_product(Z,Z))*source*D
        # A controlled diagonal has two B blocks; this exact zero pattern
        # supplies an explicit two-term tensor expansion.
        cut=(0,) if source==sp.kronecker_product(M,I) else (1,)
        aligned=realign(matrix,cut)
        if sum(any(sp.expand(value)!=0 for value in aligned[j,:]) for j in range(aligned.rows))>2:
            return False
    # A two-term control diagonal matrix space cannot be rotated to another
    # such space by nonmonomial left/right unitaries: its rank-one members
    # are exactly the two coordinate projectors.
    u,v=sp.symbols('cr_u cr_v')
    if sp.diag(u,v).det()!=u*v:return False
    # With nonzero control-frame entries the four target blocks span
    # {I,B,A,AB}. Rank two forces A=alpha*I+beta*B, an operator on B alone.
    alpha,beta=sp.symbols('cr_alpha cr_beta')
    B=sp.Matrix(2,2,sp.symbols('cr_B:4'))
    if sp.kronecker_product(I,alpha*I+beta*B)!=alpha*sp.eye(4)+beta*sp.kronecker_product(I,B):
        return False
    # Exhaust all three-interaction edge words surviving the cut demands.
    pairs=((0,1),(0,2),(1,2));counts={'triangle':0,'one_control_crossing':0}
    for word in itertools.product(pairs,repeat=3):
        degrees=[sum(q in edge for edge in word) for q in range(3)]
        if degrees[0]<1 or degrees[1]<2 or degrees[2]<2:continue
        internal=word.count((1,2))
        if internal==1 and all(word.count(pair)==1 for pair in pairs):counts['triangle']+=1
        elif internal==2 and degrees[0]==1:counts['one_control_crossing']+=1
        else:return False
    return counts=={'triangle':6,'one_control_crossing':6}


@lru_cache(maxsize=128)
def controlled_full_rank_lower_bound(matrix,native):
    matrix=sp.ImmutableMatrix(matrix)
    if matrix.shape!=(8,8) or matrix.free_symbols or matrix.atoms(sp.Float):return None
    # This verified Cartan identification includes arbitrary-angle Ising
    # primitives; the proof does not assume every pulse is a maximal CNOT.
    profile=native_profile(native)
    coordinates=profile.get('cartan_representative')
    if profile['crossing_rank_capacity']!=2 or not coordinates or coordinates[1:]!=['0','0']:return None
    if not _zero(matrix.H*matrix-sp.eye(8)):return None
    for control in range(3):
        groups=[tuple(i for i in range(8) if (i>>control)&1==bit) for bit in (0,1)]
        if not _zero(matrix[i,j] for g in (0,1) for i in groups[g] for j in groups[1-g]):continue
        blocks=[sp.ImmutableMatrix(4,4,lambda i,j:matrix[ids[i],ids[j]]) for ids in groups]
        # Both branches must be unitary, and at least one is local. The
        # target is then locally equivalent to diag(I,A^dagger B).
        branch=None
        for i in (0,1):
            if schmidt_certificate(blocks[i],(0,)).fixed_rank==1:branch=i;break
        if branch is None:continue
        relative=(blocks[branch].H*blocks[1-branch]).applyfunc(sp.simplify)
        witness=schmidt_certificate(relative,(0,))
        if witness.fixed_rank!=4:continue
        cuts=[schmidt_certificate(matrix,(q,)).as_dict() for q in range(3)]
        if not verify_controlled_rank_lemmas():raise ArithmeticError('Controlled-rank lemma failed')
        return dict(method='controlled_full_Schmidt_rank_excludes_three_Ising_interactions',
                    lower_bound=4,control=control,local_branch=branch,
                    relative_block=[str(value) for value in relative],relative_osr=witness.as_dict(),
                    cut_witnesses=cuts,native_class=profile,
                    surviving_three_gate_edge_words=12,
                    assumptions='zero ancillas; arbitrary U(2) locals and scalar phase; all-to-all; Ising-class interactions',
                    scope='fixed_target_global_entangler_lower_bound')
    return None
