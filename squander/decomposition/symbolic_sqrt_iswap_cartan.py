"""Exact three-SQiSW Cartan construction with shared symbolic outer frames.

The published middle-pulse chart is completed using real 3x3 Pauli tensors.
Every frame is arithmetic, square roots, and explicit Piecewise branches.
There is no numerical KAK step, numerical acceptance, or physical GP gate.

References: https://arxiv.org/abs/2105.06074 (middle chart and pulse budget),
https://arxiv.org/abs/quant-ph/0002045 (complete local invariants).
"""

from functools import lru_cache
import itertools
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_contract import encode_sequence, validate_operations
    from .symbolic_expression_program import ExpressionProgram, evaluate_program
    from .symbolic_controlled_chart import _zero
    from .symbolic_native_pauli import bell_coordinates, diagonal_gauge, _phase
    from .symbolic_sqrt_iswap_phase import (_phase_in_u3, _phased_xz, _phase_in_phased_xz,
                                           verify_phase_chart_lemma)
    from .symbolic_local_polar import verify_local_polar_lemma
    from .symbolic_minimality import sqrt_iswap_cost_certificate
else:
    import symbolic_gates as gates
    from symbolic_contract import encode_sequence, validate_operations
    from symbolic_expression_program import ExpressionProgram, evaluate_program
    from symbolic_controlled_chart import _zero
    from symbolic_native_pauli import bell_coordinates, diagonal_gauge, _phase
    from symbolic_sqrt_iswap_phase import (_phase_in_u3, _phased_xz, _phase_in_phased_xz,
                                         verify_phase_chart_lemma)
    from symbolic_local_polar import verify_local_polar_lemma
    from symbolic_minimality import sqrt_iswap_cost_certificate


def _qmul(a, b):
    av, bv = sp.Matrix(a[1:]), sp.Matrix(b[1:])
    return sp.Matrix([a[0]*b[0]-av.dot(bv), *(a[0]*bv+b[0]*av+av.cross(bv))])


def _qadj(q):
    return sp.Matrix([q[0], -q[1], -q[2], -q[3]])


def _qmatrix(q):
    a, x, y, z = q
    return sp.Matrix([[a-sp.I*z, -y-sp.I*x], [y-sp.I*x, a+sp.I*z]])


def _rotation(q):
    a, x, y, z = q
    return sp.Matrix([[a*a+x*x-y*y-z*z, 2*(x*y-a*z), 2*(x*z+a*y)],
                      [2*(x*y+a*z), a*a-x*x+y*y-z*z, 2*(y*z-a*x)],
                      [2*(x*z-a*y), 2*(y*z+a*x), a*a-x*x-y*y+z*z]])


def _largest_conditions(values):
    return [sp.And(*(sp.Ge(values[i], value, evaluate=False) for value in values),
                   evaluate=False) for i in range(len(values)-1)] + [sp.true]


def _projected_column(p, projector):
    # For an orthogonal projector ||P e_j||^2=P_jj. The largest diagonal
    # is at least rank(P)/3, including repeated-eigenvalue charts.
    diagonal = list(projector.diagonal())
    vectors = [p.matrix(projector[:, j]/p.bind(sp.sqrt(diagonal[j]))) for j in range(3)]
    return p.vector_choice(vectors, _largest_conditions(diagonal))


def _lift_rotation(p, R):
    squares = [p.bind((1+sp.trace(R))/4)] + [
        p.bind((1+2*R[j, j]-sp.trace(R))/4) for j in range(3)]
    anti = [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]
    choices = []
    for pivot in range(4):
        q = [None]*4
        q[pivot] = p.bind(sp.sqrt(squares[pivot]))
        denominator = 4*q[pivot]
        if pivot == 0:
            q[1:] = [p.bind(value/denominator) for value in anti]
        else:
            j = pivot-1
            q[0] = p.bind(anti[j]/denominator)
            for k in range(3):
                if j != k:
                    q[k+1] = p.bind((R[j, k]+R[k, j])/denominator)
        choices.append(sp.Matrix(q))
    return p.vector_choice(choices, _largest_conditions(squares))


def _pauli_tensor(c, t, u, v, w):
    r, I = sp.sqrt(2), sp.I
    return sp.Matrix([
        [-I*(c*c*u*u-c*c*v*v-u*u),
         2*w*(I*c*c*u+c*t*v-I*u/2), -r*I*w*(I*c*c*v+c*t*u-I*v)],
        [-I*u*w, I*c*c*u*u+I*c*c*v*v-I*c*c+2*c*t*u*v-I*u*u-I*v*v+I,
         -r*I*(2*I*c*c*u*v+c*t*u*u+c*t*v*v-c*t-I*u*v)],
        [r*c*w*(I*c*v+t*u), r*c*t*(u*u-v*v), c*c*u*u-c*c*v*v+v*v]])


def _spin_coefficients(coordinates):
    c = [sp.cos(2*value) for value in coordinates]
    s = [sp.sin(2*value) for value in coordinates]
    return [c[i]*s[(i+1)%3]*s[(i+2)%3]+sp.I*s[i]*c[(i+1)%3]*c[(i+2)%3]
            for i in range(3)]


def _left_frames(p, T, coefficients):
    real = p.matrix(T.applyfunc(sp.re))
    imag = p.matrix(T.applyfunc(sp.im))
    H = p.matrix(real.T*real-imag.T*imag)
    a = [p.bind(sp.re(value)) for value in coefficients]
    b = [p.bind(sp.im(value)) for value in coefficients]
    h = [p.bind(a[i]**2-b[i]**2) for i in range(3)]
    identity = sp.eye(3)

    def projector(i):
        result = identity
        for j in range(3):
            if j == i:
                continue
            candidate = p.matrix((H-h[j]*identity)/(h[i]-h[j]))
            factor = sp.Matrix(3, 3, lambda r, c: p.choose(
                identity[r, c], sp.Eq(h[i], h[j], evaluate=False), candidate[r, c]))
            result = p.matrix(result*factor)
        return result

    v0 = _projected_column(p, projector(0))
    P1 = projector(1)
    P1 = p.matrix(P1-sp.Matrix(3, 3, lambda r, c: p.choose(
        v0[r]*v0[c], sp.Eq(h[0], h[1], evaluate=False), 0)))
    v1 = _projected_column(p, P1)
    V = p.matrix(sp.Matrix.hstack(v0, v1, p.matrix(v0.cross(v1))))
    norm = [p.bind(a[i]**2+b[i]**2) for i in range(3)]
    nonzero = [sp.Ne(value, 0, evaluate=False) for value in norm]
    columns = [p.matrix((a[i]*real+b[i]*imag)*V[:, i]/norm[i]) for i in range(3)]
    choices, conditions = [], []
    for mask in range(8):
        nz = [i for i in range(3) if mask >> i & 1]
        W = columns.copy()
        if len(nz) == 2:
            i = next(i for i in range(3) if i not in nz)
            W[i] = p.matrix(W[(i+1)%3].cross(W[(i+2)%3]))
        elif len(nz) == 1:
            i = nz[0]
            candidates = [p.matrix(identity[:, j]-W[i]*W[i][j]) for j in range(3)]
            lengths = [p.bind(vector.dot(vector)) for vector in candidates]
            vectors = [p.matrix(vector/p.bind(sp.sqrt(length)))
                       for vector, length in zip(candidates, lengths)]
            W[(i+1)%3] = p.vector_choice(vectors, _largest_conditions(lengths))
            W[(i+2)%3] = p.matrix(W[i].cross(W[(i+1)%3]))
        elif not nz:
            W = [identity[:, i] for i in range(3)]
        choices.append(p.matrix(sp.Matrix.hstack(*W)))
        conditions.append(sp.And(*(nonzero[i] if i in nz else sp.Not(nonzero[i], evaluate=False)
                                   for i in range(3)), evaluate=False))
    W = sp.Matrix(3, 3, lambda r, c: p.bind(sp.Piecewise(
        *((matrix[r, c], condition) for matrix, condition in zip(choices, conditions)),
        evaluate=False)))
    return _lift_rotation(p, W), _lift_rotation(p, V)


def _local_quaternions(p, matrix):
    axes = [sp.eye(2)] + [-sp.I*gates.make_gate(name) for name in "XYZ"]
    F = sp.Matrix(4, 4, lambda i, j: p.bind(sp.re(
        sp.trace(sp.kronecker_product(axes[i], axes[j]).H*matrix)/4)))
    lengths = [p.bind(F[:, j].dot(F[:, j])) for j in range(4)]
    candidates = [p.matrix(F[:, j]/p.bind(sp.sqrt(lengths[j]))) for j in range(4)]
    first = p.vector_choice(candidates, _largest_conditions(lengths))
    second = p.matrix(F.T*first)
    return first, second


def _two_pulse(p, coordinates):
    x, y, z = coordinates
    X, Y, Z = [p.bind(sp.cos(2*value)) for value in coordinates]
    radical = p.bind(sp.sqrt(X*X+Y*Y+Z*Z-2*X*Y*Z-1))
    d = p.bind(X-Y+Z)
    c = p.bind(sp.sqrt((1+d-radical)/2))
    t = p.bind(sp.sqrt((1-d+radical)/2))
    v = p.bind(sp.sqrt((1-d-radical)/2))
    D = p.bind((1+X)*(1+Z)*(1-Y)/2)
    sign = p.choose(-1, sp.Lt(z, 0, evaluate=False), 1)
    u = p.choose(0, sp.Eq(c, 0, evaluate=False), sign*sp.sqrt(D)/c)
    w = p.choose(1, sp.Eq(c, 0, evaluate=False), sp.sqrt(X*Y*Z)/c)
    middle = [sp.Matrix([u, -v, 0, -w]), sp.Matrix([c, -t, 0, 0])]
    left = _left_frames(p, _pauli_tensor(c, t, u, v, w), _spin_coefficients(coordinates))
    S = gates.make_gate("SiSWAP")
    U = p.matrix(S*sp.kronecker_product(*map(_qmatrix, middle))*S, real=False)
    L = p.matrix(sp.kronecker_product(*map(_qmatrix, left)), real=False)
    A = gates.make_gate("Canonical", *coordinates)
    right_matrix = p.matrix(A.H*L.H*U, real=False)
    right = _local_quaternions(p, right_matrix)
    return [tuple(map(_qadj, right)), tuple(middle), tuple(map(_qadj, left))]


@lru_cache(maxsize=1)
def _permutation_frames():
    one = sp.Matrix([1, 0, 0, 0])
    root = sp.sqrt(2)/2
    generators = [sp.Matrix([root, root, 0, 0]), sp.Matrix([root, 0, root, 0])]
    queue, seen, result = [one], set(), {}
    while queue and len(result) < 6:
        q = queue.pop(0)
        R = _rotation(q).applyfunc(sp.simplify)
        key = tuple(R)
        if key in seen:
            continue
        seen.add(key)
        permutation = tuple(next(i for i in range(3) if R[i, j] != 0) for j in range(3))
        result.setdefault(permutation, q)
        queue.extend(_qmul(g, q).applyfunc(sp.simplify) for g in generators)
    if len(result) != 6:
        raise ArithmeticError("Incomplete Clifford axis permutation table")
    return result


def _canonicalize(p, coordinates):
    folds = [p.bind(sp.floor(2*value/sp.pi+sp.Rational(1, 2))) for value in coordinates]
    folded = [p.bind(value-sp.pi*n/2) for value, n in zip(coordinates, folds)]
    permutations = list(itertools.permutations(range(3)))
    conditions = [sp.And(sp.Ge(sp.Abs(folded[a]), sp.Abs(folded[b]), evaluate=False),
                         sp.Ge(sp.Abs(folded[b]), sp.Abs(folded[c]), evaluate=False),
                         evaluate=False) for a, b, c in permutations[:-1]] + [sp.true]
    ordered = p.vector_choice([sp.Matrix([folded[i] for i in perm]) for perm in permutations], conditions)
    C = p.vector_choice([_permutation_frames()[perm] for perm in permutations], conditions)
    sx = p.choose(-1, sp.Lt(ordered[0], 0, evaluate=False), 1)
    sy = p.choose(-1, sp.Lt(ordered[1], 0, evaluate=False), 1)
    signed = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    signs = [sp.And(sp.Eq(sx, a, evaluate=False), sp.Eq(sy, b, evaluate=False), evaluate=False)
             for a, b in signed[:-1]] + [sp.true]
    D = p.vector_choice([sp.Matrix(q) for q in ((1, 0, 0, 0), (0, 1, 0, 0),
                                               (0, 0, 1, 0), (0, 0, 0, 1))], signs)
    F = sp.Matrix([1, 0, 0, 0])
    for i, n in enumerate(folds):
        q = [p.bind(sp.cos(sp.pi*n/2)), 0, 0, 0]
        q[i+1] = p.bind(-sp.sin(sp.pi*n/2))
        F = p.matrix(_qmul(q, F))
    left = (p.matrix(_qmul(_qmul(F, C), D)), p.matrix(_qmul(F, C)))
    right = (p.matrix(_qmul(_qadj(D), _qadj(C))), _qadj(C))
    canonical = (p.bind(sx*ordered[0]), p.bind(sy*ordered[1]), p.bind(sx*sy*ordered[2]))
    return canonical, left, right, p.bind(-sp.pi*sum(folds)/2)


def _three_pulse(p, coordinates):
    canonical, left, right, phase = _canonicalize(p, coordinates)
    x, y, z = canonical
    # These three commuting one-pulse kernels leave a two-pulse residual.
    choices = ((0, sp.pi/8, -sp.pi/8), (0, sp.pi/8, sp.pi/8), (-sp.pi/8, sp.pi/8, 0))
    conditions = [sp.And(sp.Gt(y, sp.pi/8, evaluate=False), sp.Lt(z, 0, evaluate=False), evaluate=False),
                  sp.Gt(y, sp.pi/8, evaluate=False), sp.true]
    one = p.vector_choice([sp.Matrix(choice) for choice in choices], conditions)
    one_frames = []
    for choice in choices:
        # Constant choices simplify completely and have no fold scalar.
        # Evaluate their finite frame programs, rather than assuming aliases
        # produced by a different program are constant values.
        cp = ExpressionProgram()
        _, a, b, phi = _canonicalize(cp, choice)
        values = evaluate_program(cp.encode(), (), [*a[0], *a[1], *b[0], *b[1], phi], {})
        if sp.simplify(values[-1]) != 0:
            raise ArithmeticError("Unexpected scalar in one-SQiSW frame")
        one_frames.append(tuple(sp.Matrix(values[i:i+4]) for i in range(0, 16, 4)))
    frames = [p.vector_choice([value[i] for value in one_frames], conditions) for i in range(4)]
    residual, rleft, rright, rphase = _canonicalize(p, tuple(canonical[i]-one[i] for i in range(3)))
    two = _two_pulse(p, residual)
    # A(original)=left*A(residual raw)*A(one)*right, all frames are SU(2).
    # Matrix order is reversed in the time-ordered local layers below.
    before = tuple(p.matrix(_qmul(frames[2+i], right[i])) for i in range(2))
    middle1 = tuple(p.matrix(_qmul(_qmul(two[0][i], rright[i]), frames[i])) for i in range(2))
    after = tuple(p.matrix(_qmul(_qmul(left[i], rleft[i]), two[2][i])) for i in range(2))
    return [before, middle1, two[1], after], p.bind(phase+rphase)


def _polar(p, q):
    a, x, y, z = q
    first = p.choose(0, sp.Eq(a*a+z*z, 0, evaluate=False), sp.atan2(-z, a))
    last = p.choose(0, sp.Eq(x*x+y*y, 0, evaluate=False), sp.atan2(-x, y))
    theta = p.bind(2*sp.atan2(sp.sqrt(x*x+y*y), sp.sqrt(a*a+z*z)))
    return (theta, p.bind(last-first), p.bind(-last-first)), first


@lru_cache(maxsize=32)
def _uniform_budget_witness(coordinates, parameters):
    if not parameters:
        return None
    # A linear specialization into the SWAP class excludes every uniform
    # budget below three. Unused gauge/phase parameters are set to zero.
    try:
        if any(sp.Poly(value, *parameters).total_degree() > 1 for value in coordinates):
            return None
    except sp.PolynomialError:
        return None
    solutions = sp.solve([value-sp.pi/4 for value in coordinates], parameters, dict=True)
    for solution in solutions:
        unused = {value: 0 for value in parameters if value not in solution}
        assignment = {value: sp.sympify(solution.get(value, 0)).subs(unused) for value in parameters}
        specialized = tuple(sp.simplify(value.subs(assignment)) for value in coordinates)
        if specialized != (sp.pi/4,)*3:
            continue
        lower = sqrt_iswap_cost_certificate(gates.make_gate('Canonical', *specialized))
        if lower is not None and lower['minimum'] == 3:
            return dict(scope='minimum_uniform_family_budget', minimum=3,
                        substitutions={str(key): str(value) for key, value in assignment.items()},
                        specialized_coordinates=list(map(str, specialized)), lower_bound=lower,
                        pointwise_minimum_claim=False)
    return None


def synthesize_sqrt_iswap_cartan(target, basis, *, phase_exact):
    """Return a shared-expression construction; local score remains unscored.

    ``parameter_score=None`` is intentional. It must never be treated as a
    score of zero or as proof of local optimality. The exact entangler budget
    and the full scalar expression program are independently available.
    """
    if basis.entangler != "SiSWAP" or not ({"U3", "U", "PhasedXZ"} & set(basis.single_qubit_gates)):
        return None
    target = sp.ImmutableMatrix(target)
    if target.shape != (4, 4):
        return None
    if target.atoms(sp.Float):
        raise ValueError('Supply an exact two-qubit target without floating constants')
    before = after = ()
    decomposition = bell_coordinates(target)
    if decomposition is None:
        gauged = diagonal_gauge(target)
        if gauged is None:
            return None
        decomposition, before, after = gauged
    coordinates, scalar = decomposition
    if any(value.is_real is not True for value in coordinates):
        return None
    scalar_phase = _phase(scalar)
    if scalar_phase is None or scalar_phase.is_real is not True:
        return None
    if not _zero(gates.compile_gates(2, after)*target*gates.compile_gates(2, before)
                 -scalar*gates.make_gate("Canonical", *coordinates)):
        return None
    if not verify_sqrt_iswap_cartan_lemmas():
        raise ArithmeticError("Three-SQiSW component proof failed")
    p = ExpressionProgram(target.free_symbols)
    layers, phase = _three_pulse(p, coordinates)
    phase = p.bind(phase+scalar_phase)
    for layer, frames in ((0, before), (-1, after)):
        for op in frames:
            angle, wire = op[1], op[-1][0]
            q = sp.Matrix([sp.cos(angle/2), 0, 0, -sp.sin(angle/2)])
            index = 1-wire
            layers[layer] = tuple(p.matrix(_qmul(layers[layer][i], q) if layer == 0 else
                                           _qmul(q, layers[layer][i])) if i == index else layers[layer][i]
                                  for i in range(2))
    operations = []
    for index, layer in enumerate(layers):
        if index:
            operations.append(("SiSWAP", (0, 1)))
        for high, q in enumerate(layer):
            angles, delta = _polar(p, q)
            phase = p.bind(phase+delta)
            operations.append(("U3", *angles, (1-high,)))
    if "PhasedXZ" in basis.single_qubit_gates:
        phase = p.bind(phase-sum(op[1]/2 for op in operations if op[0] == "U3"))
        operations = [_phased_xz(op) if op[0] == "U3" else op for op in operations]
        close = _phase_in_phased_xz
    else:
        close = _phase_in_u3
    if phase_exact:
        operations = list(close(operations[0], phase))+operations[1:]
    operations = validate_operations(operations, basis, 2)
    result = dict(status="verified_symbolic", phase_exact=phase_exact,
                operations=encode_sequence(operations), expression_program=p.encode(),
                cost=dict(entanglers=3, single_qubit_gates=sum(len(op[-1]) == 1 for op in operations),
                          parameter_score=None),
                local_score_status="not_evaluated_for_expanded_shared_expressions",
                virtual_gp=None if phase_exact else dict(factor=str(sp.exp(sp.I*phase)), physical=False, cost=0),
                verification="exact_three_SQiSW_Pauli_tensor_chart",
                sq_cartan_certificate=dict(coordinates=list(map(str, coordinates)), scalar=str(scalar),
                    before=encode_sequence(before), after=encode_sequence(after),
                    component_proof="spin_characteristic_polynomial_real_projectors_and_SU2_lifts"),
                entangler_optimal=False, local_cost_optimal=False,
                optimality_scope="three-pulse upper bound; matching lower bounds are separate")
    witness = _uniform_budget_witness(tuple(coordinates), p.inputs)
    if witness is not None:
        result['uniform_entangler_budget_certificate'] = witness
        result['optimality_scope'] = 'minimum uniform family budget three; no pointwise or local-score minimum claim'
    return result


def verify_sqrt_iswap_cartan_certificate(target, result, basis):
    if result.get("verification") != "exact_three_SQiSW_Pauli_tensor_chart":
        return False
    if not isinstance(result.get("phase_exact"), bool):
        return False
    expected = synthesize_sqrt_iswap_cartan(target, basis, phase_exact=result["phase_exact"])
    return expected is not None and all(result.get(key) == expected.get(key) for key in (
        "status", "operations", "expression_program", "cost", "virtual_gp", "sq_cartan_certificate",
        "local_score_status", "entangler_optimal", "local_cost_optimal", "optimality_scope",
        "uniform_entangler_budget_certificate"))


def instantiate_sqrt_iswap_cartan(result, substitutions, *, digits=None):
    """Evaluate the original parameters, preserving the explicit phase policy.

    With ``digits=None`` all expressions remain exact. Numerical output is
    intended for interoperability and regression checks, never certification.
    The second return value is the virtual scalar (one in phase-exact mode).
    """
    inputs = tuple(sp.Symbol(name, real=True) for name in result["expression_program"]["inputs"])
    outputs = [parameter for op in result["operations"] for parameter in op["parameters"]]
    if not result["phase_exact"]:
        outputs.append(result["virtual_gp"]["factor"])
    values = iter(evaluate_program(result["expression_program"], inputs, outputs, substitutions, digits=digits))
    operations = tuple((op["gate"], *(next(values) for _ in op["parameters"]), tuple(op["wires"]))
                       for op in result["operations"])
    return operations, sp.S.One if result["phase_exact"] else next(values)


@lru_cache(maxsize=1)
def verify_sqrt_iswap_cartan_lemmas():
    return (verify_phase_chart_lemma() and verify_local_polar_lemma()
            and _verify_middle_lemma() and _verify_frame_lemmas())


def _verify_middle_lemma():
    c, t, u, v, w = sp.symbols("sq_c sq_t sq_u sq_v sq_w", real=True)
    axes = [sp.eye(2)] + [gates.make_gate(name) for name in "XYZ"]
    S = gates.make_gate("SiSWAP")
    U = S*sp.kronecker_product(u*axes[0]+sp.I*v*axes[1]+sp.I*w*axes[3],
                              c*axes[0]+sp.I*t*axes[1])*S
    yy = sp.kronecker_product(axes[2], axes[2])
    spin = U*yy*U.T*yy
    ideal = sp.groebner((c*c+t*t-1, u*u+v*v+w*w-1), t, w, c, u, v,
                       extension=[sp.I, sp.sqrt(2)])

    def zero(value):
        return ideal.reduce(sp.expand(value))[1] == 0

    if any(not zero(value) for value in U.H*U-sp.eye(4)):
        return False
    T = _pauli_tensor(c, t, u, v, w)
    wanted = (c*c*w*w+2*sp.I*c*t*u*v)*sp.eye(4) + sum((
        T[i, j]*sp.kronecker_product(axes[i+1], axes[j+1])
        for i, j in itertools.product(range(3), repeat=2)), sp.zeros(4))
    if any(not zero(value) for value in spin-wanted):
        return False
    if S.det() != 1:
        return False
    real, imag = T.applyfunc(sp.re), T.applyfunc(sp.im)
    if any(not zero(value) for value in real.T*imag-2*c**3*t*u*v*w*w*sp.eye(3)):
        return False
    second = 4*c**4-8*c*c*u*u+4*v**4-8*v*v+2
    if not zero((sp.trace(spin)**2-sp.trace(spin*spin))/2-second):
        return False
    # Equality of the full characteristic polynomials, not just |tr|^2.
    # det=1 follows from the SU(2) factors and det(S)=1; for a unitary
    # 4x4 matrix the third coefficient is the conjugate of its trace.
    X, Y, Z, r = sp.symbols("sq_X sq_Y sq_Z sq_r", real=True)
    d = X-Y+Z
    C, V = (1+d-r)/2, (1-d-r)/2
    D = (1+X)*(1+Z)*(1-Y)/2
    relation = sp.groebner((r*r-X*X-Y*Y-Z*Z+2*X*Y*Z+1,), r, X, Y, Z, domain=sp.QQ)
    identities = (C*(1-V)-D-X*Y*Z,
                  4*C*C-8*D+4*V*V-8*V+2-2*(2*(X*X+Y*Y+Z*Z)-3),
                  4*(1-C)*V*D-(1-X*X)*(1-Y*Y)*(1-Z*Z))
    if any(relation.reduce(sp.expand(value))[1] != 0 for value in identities):
        return False
    # In the two-pulse chamber 0<=X<=Y,Z<=1 and x>=y+|z|.
    # Delta=(X-YZ)^2-(1-Y^2)(1-Z^2)>=0. The following two
    # nonnegative factorizations put d+/-sqrt(Delta) in [-1,1].
    delta = X*X+Y*Y+Z*Z-2*X*Y*Z-1
    if sp.expand((1-d)**2-delta-2*(1-X)*(1+Y)*(1-Z)) != 0:
        return False
    if sp.expand((1+d)**2-delta-4*(D+X*Y*Z)) != 0:
        return False
    # Exact target spin tensor and second characteristic coefficient.
    cs = sp.symbols("sq_C:3", real=True)
    ss = sp.symbols("sq_S:3", real=True)
    PP = [sp.kronecker_product(axis, axis) for axis in axes[1:]]
    target_spin = sp.eye(4)
    for Cj, Sj, pauli in zip(cs, ss, PP):
        target_spin = target_spin*(Cj*sp.eye(4)+sp.I*Sj*pauli)
    k = [cs[i]*ss[(i+1)%3]*ss[(i+2)%3]+sp.I*ss[i]*cs[(i+1)%3]*cs[(i+2)%3]
         for i in range(3)]
    scalar = sp.prod(cs)+sp.I*sp.prod(ss)
    if any(sp.expand(value) != 0 for value in target_spin-scalar*sp.eye(4)-sum(
            (coefficient*pauli for coefficient, pauli in zip(k, PP)), sp.zeros(4))):
        return False
    circles = sp.groebner([Cj*Cj+Sj*Sj-1 for Cj, Sj in zip(cs, ss)], *ss, *cs,
                         extension=sp.I)
    target_second = (sp.trace(target_spin)**2-sp.trace(target_spin*target_spin))/2
    if circles.reduce(sp.expand(target_second-4*sum(Cj*Cj for Cj in cs)+6))[1] != 0:
        return False
    # c=0 is precisely the CNOT corner in this chamber; choose u=v=0,w=t=1.
    corner = {c: 0, t: 1, u: 0, v: 0, w: 1}
    if sp.simplify(sp.trace(spin.subs(corner))) != 0 or second.subs(corner) != 2:
        return False
    return True


def _verify_frame_lemmas():
    q = sp.Matrix(sp.symbols("sq_q:4", real=True))
    r = sp.Matrix(sp.symbols("sq_r:4", real=True))
    if any(sp.expand(value) != 0 for value in _qmatrix(_qmul(q, r))-_qmatrix(q)*_qmatrix(r)):
        return False
    R = _rotation(q)
    axes = [gates.make_gate(name) for name in "XYZ"]
    ideal = sp.groebner((sum(value*value for value in q)-1,), *q, extension=sp.I)
    for j, axis in enumerate(axes):
        wanted = sum((R[i, j]*axes[i] for i in range(3)), sp.zeros(2))
        if any(ideal.reduce(sp.expand(value))[1] != 0 for value in
               _qmatrix(q)*axis*_qmatrix(q).H-wanted):
            return False
    squares = [(1+sp.trace(R))/4]+[(1+2*R[j, j]-sp.trace(R))/4 for j in range(3)]
    if any(ideal.reduce(sp.expand(value-q[i]**2))[1] != 0 for i, value in enumerate(squares)):
        return False
    anti = [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]
    if any(sp.expand(anti[j]-4*q[0]*q[j+1]) != 0 for j in range(3)):
        return False
    for i, j in itertools.combinations(range(3), 2):
        if sp.expand(R[i, j]+R[j, i]-4*q[i+1]*q[j+1]) != 0:
            return False
    # Real Pauli tensor simultaneous SVD: Re(k_i)*Im(k_i) is independent
    # of i. Hence H=Re(T^T T) has eigenvalues Re(k_i^2), and collisions
    # imply k_i=+/-k_j. Projectors may then choose any orthonormal basis.
    x, y, z = sp.symbols("sq_x sq_y sq_z", real=True)
    coefficients = _spin_coefficients((x, y, z))
    products = [sp.re(k)*sp.im(k) for k in coefficients]
    if any(sp.expand(value-products[0]) != 0 for value in products[1:]):
        return False
    # Rank-one quaternion coefficients factor a local SU(2) tensor exactly.
    F = q*r.T
    if any(sp.expand(value) != 0 for value in F.T*q-r*sum(value*value for value in q)):
        return False
    for perm, frame in _permutation_frames().items():
        rotation = _rotation(frame)
        if any(rotation[perm[j], j]**2 != 1 for j in range(3)):
            return False
    # Integer coordinate folds have period four in n. Each one is a local
    # SU(2) tensor times its explicit scalar; the scalar is never discarded.
    for axis, pauli in zip("XYZ", axes):
        PP = sp.kronecker_product(pauli, pauli)
        for n in range(4):
            local = gates.make_gate("R"+axis.lower(), -sp.pi*n)
            folded = sp.exp(-sp.I*sp.pi*n/2)*sp.kronecker_product(local, local)
            if not _zero(folded-sp.I**n*(PP if n % 2 else sp.eye(4))):
                return False
    # All five eigenvalue multiplicity patterns. Repeating a nonmatching
    # factor squares an indicator projector and leaves it unchanged.
    lambdas = sp.symbols("sq_lambda:3", real=True)
    for pattern in ((0, 1, 2), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 0, 0)):
        spectrum = [lambdas[index] for index in pattern]
        for i in range(3):
            for eigenvalue in set(spectrum):
                indicator = sp.prod((eigenvalue-spectrum[j])/(spectrum[i]-spectrum[j])
                                    for j in range(3) if spectrum[j] != spectrum[i])
                if sp.cancel(indicator-int(eigenvalue == spectrum[i])) != 0:
                    return False
    # Residual-chamber inequalities, in units pi/8, for 0<=c<=b<=a<=2.
    # The low-b split is a<1 / a>=1; the high-b split is c<=1 / c>=1.
    # Each displayed sum consists of nonnegative terms on that branch.
    a, b, c = sp.symbols("sq_domain_a sq_domain_b sq_domain_c", real=True)
    differences = ((a+1)-(1-b)-c-(a+(b-c)),
                   (3-a)-(1-b)-c-((2-a)+(b-c)),
                   a-(b-1)-(1-c)-((a-b)+c),
                   a-(b-1)-(c-1)-((a-b)+(2-c)))
    if any(sp.expand(value) != 0 for value in differences):
        return False
    return True
