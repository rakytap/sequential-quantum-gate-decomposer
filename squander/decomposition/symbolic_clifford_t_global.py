"""Exhaustive exact Clifford+T shortest words, including non-Clifford prefixes.

The native alphabet is H,S,T,Tdg, each of score two. Matrices are integer
tuples in Z[sqrt(2),i]/2**k. A meet-in-the-middle ball contains EVERY word
through twice its radius; this is not a Clifford-only intermediate graph.
"""

from collections import deque
from functools import lru_cache
import hashlib
import json
import sympy as sp

if __package__:
    from .symbolic_clifford_search import group_data
else:
    from symbolic_clifford_search import group_data


def _canonical(k, values):
    while k and all(value % 2 == 0 for value in values):
        values = tuple(value//2 for value in values)
        k -= 1
    return (k, *values)


def _root(a):
    return (2*a[1], a[0], 2*a[3], a[2])


def _imaginary(a):
    return (-a[2], -a[3], a[0], a[1])


def _step(state, gate):
    k, values = state[0], state[1:]
    rows = (values[:8], values[8:])
    if gate == 'H':
        new = tuple(a+b for a,b in zip(*rows)) + tuple(a-b for a,b in zip(*rows))
        return _canonical(k+1, tuple(value for index in range(0,16,4) for value in _root(new[index:index+4])))
    if gate == 'S':
        return _canonical(k, rows[0]+tuple(value for index in (0,4) for value in _imaginary(rows[1][index:index+4])))
    sign = 1 if gate == 'T' else -1
    lower = []
    for index in (0,4):
        value = rows[1][index:index+4]
        lower.extend(_root(tuple(a+sign*b for a,b in zip(value,_imaginary(value)))))
    return _canonical(k+1, tuple(2*a for a in rows[0])+tuple(lower))


def _multiply(a,b):
    def product(x,y):
        ar,br,ai,bi=x; cr,dr,ci,di=y
        return (ar*cr+2*br*dr-ai*ci-2*bi*di,
                ar*dr+br*cr-ai*di-bi*ci,
                ar*ci+2*br*di+ai*cr+2*bi*dr,
                ar*di+br*ci+ai*dr+bi*cr)
    values=[]
    for row in range(2):
        for column in range(2):
            terms=[product(a[1+4*(2*row+j):5+4*(2*row+j)],
                           b[1+4*(2*j+column):5+4*(2*j+column)]) for j in range(2)]
            values.extend(x+y for x,y in zip(*terms))
    return _canonical(a[0]+b[0],tuple(values))


def _inverse(a):
    return (a[0],*(value for index in (0,2,1,3)
                   for value in (*a[1+4*index:3+4*index],
                                 *(-x for x in a[3+4*index:5+4*index]))))


def _encode(matrix):
    coefficients=[]
    for entry in matrix:
        for part in (sp.re(entry),sp.im(entry)):
            part=sp.expand(part)
            radical=part.coeff(sp.sqrt(2))
            rational=sp.expand(part-radical*sp.sqrt(2))
            if not rational.is_Rational or not radical.is_Rational:
                raise ValueError('Matrix is outside the exact dyadic ring')
            coefficients.extend((rational,radical))
    denominator=sp.ilcm(*(value.q for value in coefficients))
    k=int(denominator).bit_length()-1
    if denominator!=2**k:raise ValueError('Non-dyadic denominator')
    return _canonical(k,tuple(int(value*denominator) for value in coefficients))


@lru_cache(maxsize=4)
def exact_word_ball(radius=7):
    identity=_encode(sp.eye(2))
    words={identity:()}
    queue=deque((identity,))
    alphabet=('H','S','T','Tdg')
    while queue:
        state=queue.popleft()
        word=words[state]
        if len(word)==radius:continue
        for gate in alphabet:
            end=_step(state,gate)
            if end not in words:
                words[end]=word+(gate,)
                queue.append(end)
    return words


@lru_cache(maxsize=1)
def solve_global_clifford_t():
    radius=7
    ball=exact_word_ball(radius)
    ordered=sorted(ball,key=lambda state:(len(ball[state]),state))
    solutions={}
    for state,matrix in group_data()[1].items():
        target=_encode(matrix)
        best=ball.get(target)
        for first in ordered:
            if best is not None and len(ball[first])>=len(best):break
            second=_multiply(target,_inverse(first))
            last=ball.get(second)
            if last is not None:
                word=ball[first]+last
                if best is None or len(word)<len(best):best=word
        if best is None:raise ArithmeticError('Word ball does not cover the Clifford target upper bound')
        actual=_encode(sp.eye(2))
        for gate in best:actual=_step(actual,gate)
        if actual!=target:raise ArithmeticError('Meet-in-the-middle witness failed')
        solutions[state]=best
    # All resulting witnesses have length <=2r. Any shorter word has a
    # split with both halves <=r, so the exhaustive lookup excludes it.
    if max(map(len,solutions.values()))>2*radius:raise ArithmeticError('Incomplete search radius')
    digest=hashlib.sha256(json.dumps([(state,ball[state]) for state in sorted(ball)],separators=(',',':')).encode()).hexdigest()
    return dict(alphabet=['H','S','T','Tdg'],radius=radius,exact_word_ball_size=len(ball),
                exact_word_ball_sha256=digest,solutions=solutions,
                proof_scope='all_words_in_fixed_H_S_T_Tdg_alphabet_including_nonclifford_intermediates',
                global_local_optimality=True)
