"""Exact two-qubit CNOT-cost certificates, with explicit unresolved conditions.

Shende, Bullock and Markov, PRA 70, 012310 (2004), propositions III.1--3:
https://arxiv.org/abs/quant-ph/0308045 . Arbitrary local unitaries are free;
there are no ancillary qubits and connectivity is unrestricted. This module
certifies an interaction count, not the secondary parameter-sensitive score.
"""

from dataclasses import dataclass
from functools import lru_cache
import sympy as sp


@lru_cache(maxsize=512)
def fixed_native_one_pulse_exclusion(matrix,primitive):
    """A sufficient exact exclusion under arbitrary projective local frames.

    These phase-normalized spin-flip traces are local invariants. Unequal
    invariants exclude one use; equality alone is not an equivalence proof.
    A separate nonlocality proof must exclude zero before concluding a lower
    bound of two interactions.
    """
    if matrix.shape!=(4,4) or matrix.free_symbols:return None
    yy=sp.kronecker_product(sp.Matrix([[0,-sp.I],[sp.I,0]]),sp.Matrix([[0,-sp.I],[sp.I,0]]))
    def invariants(operator):
        gamma=operator*yy*operator.T*yy
        delta=operator.det()
        return sp.trace(gamma)**2/delta,sp.trace(gamma*gamma)/delta
    for index,(target,native) in enumerate(zip(invariants(matrix),invariants(primitive))):
        difference=sp.simplify(sp.expand_complex(target-native))
        if difference.is_zero is False:
            return dict(method="unequal_phase_normalized_spin_flip_invariants",
                        invariant=index,nonzero_difference=str(difference),lower_bound=2,
                        source="https://arxiv.org/abs/quant-ph/0002045")
    return None


def _simplify(expression):
    # Keep common exponential phases factored until they cancel. Expanding a
    # rational phase to trigonometric constants first can make a trivial
    # phase-invariance check needlessly expensive.
    value = sp.simplify(sp.trigsimp(expression, method="fu"))
    roots = {power: sp.exp(sp.I*sp.pi*power.exp) for power in value.atoms(sp.Pow)
             if power.base == -1 and power.exp.is_Rational}
    return sp.simplify(value.xreplace(roots)) if roots else value


def _all_zero(expressions):
    return sp.And(*(sp.Eq(_simplify(value), 0) for value in expressions))


@dataclass(frozen=True)
class CNOTCostCertificate:
    lower_bound: int
    upper_bound: int
    at_most: tuple
    source: str = "https://arxiv.org/abs/quant-ph/0308045"

    @property
    def minimum(self):
        """An integer only when all lower-cost possibilities have been excluded."""
        return self.lower_bound if self.lower_bound == self.upper_bound else None


def cnot_cost_certificate(matrix):
    """Prove a minimum or return exact parameter conditions for costs <=0,1,2.

    No floating tolerance, numerical substitution, or generic-angle assumption
    is used. Inputs must be exact 4x4 matrices with provable unitarity. For a
    free angle a special value can change the minimum, so ``minimum`` may be
    None. Re-run after exact substitution to certify that specialization.

    Instead of choosing a fourth root of det(U), use gamma=U YY U.T YY and
    delta=det(U). The zero/one tests remove that phase algebraically. For the
    two-CNOT test, z=tr(gamma)/sqrt(delta) is real iff z**2 is real and >=0.
    This avoids a symbolic square-root branch without changing the theorem.
    """
    matrix = sp.ImmutableMatrix(matrix)
    if matrix.shape != (4, 4) or matrix.atoms(sp.Float):
        raise ValueError("Provide an exact 4x4 unitary, without floating constants")
    if _all_zero(matrix.H*matrix-sp.eye(4)) != sp.true:
        raise ValueError("Unitarity could not be proved symbolically")
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    yy = sp.kronecker_product(y, y)
    gamma = (matrix*yy*matrix.T*yy).applyfunc(_simplify)
    delta = _simplify(matrix.det())
    scalar = gamma[0, 0]
    zero = _all_zero(gamma-scalar*sp.eye(4)) & sp.Eq(_simplify(scalar**2-delta), 0)
    one = _all_zero(gamma*gamma+delta*sp.eye(4)) & sp.Eq(_simplify(sp.trace(gamma)), 0)
    squared_trace = _simplify(sp.trace(gamma)**2/delta)
    two = sp.And(sp.Eq(_simplify(sp.im(squared_trace)), 0),
                 sp.Ge(_simplify(sp.re(squared_trace)), 0))
    # At most one also includes local gates (two gates can cancel).
    conditions = (zero, sp.Or(zero, one), two)
    lower, upper = 0, 3
    for count, condition in enumerate(conditions):
        if condition == sp.false:
            lower = max(lower, count+1)
        elif condition == sp.true:
            upper = min(upper, count)
    return CNOTCostCertificate(lower, upper, conditions)
