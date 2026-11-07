"""Independent small-count native lower bounds with exact witnesses.

These bounds relax all local gates and the scalar phase. A specialization
can prove the minimum *uniform budget* for a symbolic family; it never proves
that every member of the family has the same minimum. Construction identities
and parameter-sensitive local optimality remain separate responsibilities.
"""

from functools import lru_cache
import itertools
import json
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_identity import exponential_forms, zero_form
    from .symbolic_osr import realign, schmidt_certificate
    from .symbolic_minimality import _simplify
    from .symbolic_feasibility import determinant_group
else:
    import symbolic_gates as gates
    from symbolic_identity import exponential_forms, zero_form
    from symbolic_osr import realign, schmidt_certificate
    from symbolic_minimality import _simplify
    from symbolic_feasibility import determinant_group


MAKHLIN = "https://arxiv.org/abs/quant-ph/0002045"
CNOT = "https://arxiv.org/abs/quant-ph/0308045"


def _zero(value):
    entries = tuple(value) if isinstance(value, sp.MatrixBase) else (value,)
    forms, inverse = exponential_forms(entries)
    return all(zero_form(entry, tuple(inverse)) for entry in forms)


@lru_cache(maxsize=4096)
def spin_data(matrix):
    matrix = sp.ImmutableMatrix(matrix)
    if matrix.shape != (4, 4) or matrix.atoms(sp.Float):
        raise ValueError("An exact two-qubit unitary is required")
    yy = sp.kronecker_product(gates.make_gate('Y'), gates.make_gate('Y'))
    spin = (matrix * yy * matrix.T * yy).applyfunc(_simplify)
    determinant = _simplify(matrix.det())
    trace = _simplify(sp.trace(spin))
    g1 = _simplify(trace**2 / (16 * determinant))
    g2 = _simplify((trace**2 - sp.trace(spin * spin)) / (4 * determinant))
    return spin, determinant, (g1, g2)


@lru_cache(maxsize=1)
def verify_two_pulse_trace_lemmas():
    """Algebraic proofs for every real middle SO(4) frame, not samples.

    For a 2+2 block partition of O in SO(4), the squared Frobenius norms
    are X,Y,Y,X. This follows by subtracting its block row/column norms.
    Two Ising pulses, or two fixed iSWAP-edge pulses, have real spin trace.
    Two sqrt-SWAP pulses have purely imaginary spin trace. Thus g1 is
    respectively nonnegative or nonpositive. The one-pulse class must be
    excluded separately: a pulse need not be cancellable by one more pulse.
    """
    X, Y, Z, W = sp.symbols('trace_X trace_Y trace_Z trace_W', real=True)
    a, b = sp.symbols('trace_a trace_b', real=True)
    trace = (sp.exp(sp.I*(a+b))*X + sp.exp(sp.I*(a-b))*Y
             + sp.exp(sp.I*(-a+b))*Z + sp.exp(-sp.I*(a+b))*W)
    wanted = 2*sp.cos(a+b)*X + 2*sp.cos(a-b)*Y
    if not _zero(trace.subs({Z:Y, W:X})-wanted):
        return False
    # X=W and Y=Z follow from the four block norm equations.
    if sp.expand((X-W)-((X+Y-2)-(Z+W-2)+(X+Z-2)-(Y+W-2))/2) != 0:
        return False
    if sp.expand((Y-Z)-((X+Y-2)-(X+Z-2))) != 0:
        return False
    # Edge pulses have D^2=diag(e^(ia),e^(ia),-e^(-ia),-e^(-ia)).
    edge = (sp.exp(sp.I*(a+b))*X - sp.exp(sp.I*(a-b))*Y
            - sp.exp(sp.I*(-a+b))*Z + sp.exp(-sp.I*(a+b))*W)
    if not _zero(edge.subs({Z:Y, W:X})-(2*sp.cos(a+b)*X-2*sp.cos(a-b)*Y)):
        return False
    # For sqrt-SWAP D^2=e^(+/-i*pi/4) diag(1,1,1,-1), so
    # tr(D^2 O^T D^2 O) is +/-i times a real polynomial.
    o = sp.Matrix(4, 4, sp.symbols('trace_o:16', real=True))
    signs = (1, 1, 1, -1)
    real_polynomial = sum(signs[i]*signs[j]*o[i,j]**2
                          for i,j in itertools.product(range(4), repeat=2))
    D2 = sp.exp(sp.I*sp.pi/4)*sp.diag(*signs)
    if not _zero(sp.trace(D2*o.T*D2*o)-sp.I*real_polynomial):
        return False
    # Changing the fourth root used to normalize det(U) multiplies the
    # spin trace by +/-1. Its square/det(U), hence g1, is unaffected.
    return True


def _native_representative(name, parameters):
    if name in {'CX','CNOT','CZ','ECR','XX90','ZZMax','IonQMSMax'}:
        return (sp.pi/4, 0, 0), 'real_two_pulse_spin_trace'
    if name in {'Rxx','Rzz','Rzx'}:
        return (-parameters[0]/2, 0, 0), 'real_two_pulse_spin_trace'
    if name == 'ZZPhase':
        return (-sp.pi*parameters[0]/2, 0, 0), 'real_two_pulse_spin_trace'
    if name == 'IonQZZ':
        return (-sp.pi*parameters[0], 0, 0), 'real_two_pulse_spin_trace'
    if name == 'IonQMS':
        return (-sp.pi*parameters[2], 0, 0), 'real_two_pulse_spin_trace'
    if name == 'iSWAP':
        return (sp.pi/4, sp.pi/4, 0), 'real_two_pulse_spin_trace'
    if name == 'SYC':
        return (sp.pi/4, sp.pi/4, -sp.pi/24), 'real_two_pulse_spin_trace'
    if name == 'SSWAP':
        return (-sp.pi/8, -sp.pi/8, -sp.pi/8), 'imaginary_two_pulse_spin_trace'
    return None


@lru_cache(maxsize=32)
def native_profile(name):
    definition = gates.gate_definition(name)
    parameters = sp.symbols('native_bound:'+str(len(definition.parameters)), real=True)
    matrix = sp.ImmutableMatrix(gates.make_gate(name, *parameters))
    if not _zero(matrix.H*matrix-sp.eye(4)):
        raise ArithmeticError('Native unitarity lemma failed')
    spin, determinant, invariants = spin_data(matrix)
    rank = schmidt_certificate(matrix, (0,)).rank_upper_bound
    profile = dict(native=name, crossing_rank_capacity=rank,
                   invariants=tuple(map(str, invariants)),
                   parameterized=bool(parameters),
                   fixed_local_class=not any(value.free_symbols for value in invariants))
    representative = _native_representative(name, parameters)
    if representative is not None:
        coordinates, lemma = representative
        expected = spin_data(sp.ImmutableMatrix(gates.make_gate('Canonical', *coordinates)))[2]
        if not all(_zero(a-b) for a,b in zip(invariants, expected)):
            raise ArithmeticError('Native Cartan class identification failed: '+name)
        if not verify_two_pulse_trace_lemmas():
            raise ArithmeticError('Two-pulse trace lemma failed')
        profile.update(two_pulse_lemma=lemma, cartan_representative=list(map(str, coordinates)),
                       class_identification='exact_phase_normalized_Makhlin_invariants')
    if name == 'fSim':
        theta, phi = parameters
        identity = sp.eye(4)
        # The repeated eigenvalue exp(-i*phi) and the odd-parity pair
        # exp(+/-2i*theta) give this MONIC cubic for every real parameter,
        # including coincident eigenvalues. No generic-rank assumption.
        annihilator = ((spin-sp.exp(-sp.I*phi)*identity)
                       *(spin*spin-2*sp.cos(2*theta)*spin+identity))
        if not _zero(annihilator):
            raise ArithmeticError('fSim monic cubic annihilator failed')
        profile['spin_minimal_polynomial_degree_cap'] = 3
        profile['spin_annihilator'] = '(t-exp(-I*phi))*(t**2-2*cos(2*theta)*t+1)'
    return profile


def _nonzero_constant(value):
    value = _simplify(value)
    if value.free_symbols or value.atoms(sp.Float):
        return False
    if value.is_zero is not None:
        return value.is_zero is False
    if value.is_algebraic is True:
        variable = sp.Dummy('native_bound_root')
        return sp.Poly(sp.minpoly(value, variable), variable).TC() != 0
    return False


@lru_cache(maxsize=4096)
def _fixed_lower_bound(matrix, name):
    matrix = sp.ImmutableMatrix(matrix)
    if matrix.shape != (4,4) or matrix.free_symbols:
        raise ValueError('A fixed exact two-qubit matrix is required')
    if not _zero(matrix.H*matrix-sp.eye(4)):
        raise ValueError('Target unitarity was not proved')
    profile = native_profile(name)
    schmidt = schmidt_certificate(matrix, (0,))
    if not _nonzero_constant(schmidt.nonzero_minor):
        return dict(lower_bound=0, evidence=[])
    rank = schmidt.rank_upper_bound
    capacity = profile['crossing_rank_capacity']
    lower = 0
    while capacity**lower < rank:
        lower += 1
    evidence = [dict(method='exact_realignment_rank', rank=rank,
                     minor=schmidt.as_dict(), native_crossing_rank_cap=capacity)]
    one_excluded = lower >= 2
    spin, determinant, invariants = spin_data(matrix)
    if rank > 1 and profile['fixed_local_class']:
        native = tuple(sp.sympify(x) for x in profile['invariants'])
        for index, (target_value, native_value) in enumerate(zip(invariants, native)):
            difference = _simplify(target_value-native_value)
            if _nonzero_constant(difference):
                one_excluded = True
                evidence.append(dict(method='unequal_Makhlin_invariant', invariant=index+1,
                                     nonzero_difference=str(difference), reference=MAKHLIN))
                break
        if rank != capacity:
            one_excluded = True
            evidence.append(dict(method='unequal_fixed_native_OSR', target_rank=rank, native_rank=capacity))
    if rank > 1 and profile.get('spin_minimal_polynomial_degree_cap') == 3:
        powers = [sp.eye(4), spin, spin*spin, spin**3]
        krylov = sp.Matrix.hstack(*(sp.Matrix(16,1,list(power)) for power in powers))
        # Propose one minor by exact row reduction, then prove that actual
        # restored determinant nonzero. An undecided coefficient cannot
        # produce a lower-bound certificate just by being a formal pivot.
        forms, inverse = exponential_forms(tuple(krylov))
        formal = sp.Matrix(16, 4, forms)
        _, rows = formal.T.rref(iszerofunc=lambda value:zero_form(value,tuple(inverse)),
                                simplify=lambda value:sp.cancel(sp.expand(value)))
        if len(rows) == 4:
            value = _simplify(formal[list(rows),:].det().xreplace(inverse))
            if _nonzero_constant(value):
                one_excluded = True
                evidence.append(dict(method='spin_Krylov_degree_exceeds_native_cap',
                                     rows=list(rows), nonzero_minor=str(value),
                                     native_degree_cap=3))
    if one_excluded:
        lower = max(lower, 2)
        g1 = invariants[0]
        imaginary = _simplify(sp.im(g1))
        real = _simplify(sp.re(g1))
        nonreal = _nonzero_constant(imaginary)
        lemma = profile.get('two_pulse_lemma')
        wrong_sign = ((real.is_negative is True if lemma == 'real_two_pulse_spin_trace'
                       else real.is_positive is True) if imaginary == 0 else False)
        if lemma and (nonreal or wrong_sign):
            lower = max(lower, 3)
            evidence.append(dict(method=lemma, g1=str(g1), real_part=str(real),
                                 imaginary_part=str(imaginary),
                                 excluded='nonreal' if nonreal else 'wrong_sign',
                                 theorem='two_native_pulse_spin_trace_sign',
                                 local_resource_scope='arbitrary U(2) locals and scalar phase'))
    return dict(lower_bound=lower, evidence=evidence)


def _specializations(target):
    symbols = tuple(sorted(target.free_symbols, key=str))
    if not symbols:
        yield {}
        return
    # These are exact target instances used only as LOWER-BOUND witnesses.
    # A witness proves that a uniform budget must handle this hard member;
    # it does not certify any construction by sampled agreement.
    for values in (tuple(sp.Rational(1,2) for _ in symbols),
                   tuple(sp.pi/4 for _ in symbols),
                   tuple((i+1)*sp.pi/7 for i in range(len(symbols))),
                   tuple(sp.Rational(i+1, 5) for i in range(len(symbols)))):
        yield dict(zip(symbols, values))


def small_count_certificate(target, basis, entanglers, *, phase_exact=False):
    """Certify a supplied 1--4 pulse upper bound, or decline it honestly."""
    target = sp.ImmutableMatrix(target)
    if target.atoms(sp.Float):
        raise ValueError('Floating matrices cannot certify an exact lower bound')
    if (target.shape != (4,4) or not isinstance(entanglers, int)
            or isinstance(entanglers, bool) or not 1 <= entanglers <= 4):
        return None
    for substitutions in _specializations(target):
        specialized = sp.ImmutableMatrix(target.subs(substitutions))
        proof = _fixed_lower_bound(specialized, basis.entangler)
        lower = proof['lower_bound']
        parity = None
        if phase_exact and lower == 3 and determinant_group(basis,1) == 1:
            definition = gates.gate_definition(basis.entangler)
            if (not definition.parameters and _zero(specialized.det()-1)
                    and _zero(gates.make_gate(basis.entangler).det()+1)):
                lower = 4
                parity = dict(target_determinant='1', local_determinant='1',
                              native_determinant='-1', native_count_must_be_even=True)
        if lower > entanglers:
            raise ArithmeticError('A native lower bound contradicts the supplied construction')
        if lower != entanglers:
            continue
        return dict(method='exact_native_cut_and_spin_invariant_lower_bound',
                    lower_bound=lower, matched_construction_entanglers=entanglers,
                    native=basis.entangler, phase_exact=bool(phase_exact),
                    scope='minimum_uniform_family_budget' if substitutions else 'fixed_target_minimum',
                    substitutions={str(p):str(v) for p,v in substitutions.items()},
                    specialization_caveat=('Other parameter values can use fewer pulses' if substitutions else None),
                    witness=proof, exact_determinant_parity=parity,
                    assumes_verified_upper_bound=True, local_cost_optimality=False,
                    native_profile=native_profile(basis.entangler))
    return None


def verify_small_count_certificate(target, basis, certificate):
    if not isinstance(certificate, dict):
        return False
    try:
        expected = small_count_certificate(target, basis, certificate['matched_construction_entanglers'],
                                           phase_exact=certificate['phase_exact'])
    except (ValueError, KeyError, TypeError, ArithmeticError):
        return False
    # JSON represents tuple-valued profile fields as lists. Compare their
    # canonical interchange forms so an archived proof replays exactly as
    # the in-memory certificate does.
    return (expected is not None
            and json.dumps(certificate,sort_keys=True)==json.dumps(expected,sort_keys=True))
