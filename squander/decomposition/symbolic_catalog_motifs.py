"""Exact structural motifs for native conventions and register constructors.

These constructive proposals retain every parameter and physical scalar.
They supply upper bounds. Each enclosing circuit still requires an exact
identity proof and a separate entangler/local-cost optimality certificate.
"""

import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_native_pauli import _phase
    from .symbolic_clifford_search import exact_state, solve_clifford_alphabet, clifford_path
else:
    import symbolic_gates as gates
    from symbolic_native_pauli import _phase
    from symbolic_clifford_search import exact_state, solve_clifford_alphabet, clifford_path


MOTIF_SCHEMAS = {
    "native_interaction_frame": ("XX90", "YY90", "ZZ90", "ZZMax", "IonQZZ", "ZZPhase", "IonQMS", "IonQMSMax", "TK2"),
    "controlled_phase_or_rotation": ("CSXdg", "Deutsch", "CCP", "CCZPow", "CCXPow", "Controlled", "MCMT", "UC"),
    "excitation_exchange_echo": ("CZS", "CCZS", "fFredkin"),
    "fermionic_or_phased_exchange_power": ("FSWAPPow", "PhasedISWAPPow"),
    "cyclic_Gray_code_multiplexor": ("UCRX", "UCRY", "UCRZ"),
    "Walsh_phase_polynomial": ("Diagonal",),
    "wire_permutation": ("Permutation",),
    "GF2_Gaussian_network": ("LinearFunction",),
}


def motif_schema(name):
    return next((schema for schema, names in MOTIF_SCHEMAS.items() if name in names), None)


def _phase_on_wires(angle, wires):
    if angle == 0:
        return ()
    if len(wires) == 1:
        return (("P", angle, tuple(wires)),)
    if angle == sp.pi and len(wires) in (2,3):
        return (("CZ" if len(wires) == 2 else "CCZ", tuple(wires)),)
    return (("MCPhase", angle, len(wires)-1, tuple(wires)),)


def _uniform_rotations(axis, angles):
    """Cyclic Gray-code parity motif for any power-of-two multiplexor."""
    angles = tuple(map(sp.sympify, angles))
    count = len(angles)
    if count == 0 or count & (count-1):
        raise ValueError("A uniform rotation needs a power-of-two angle array")
    if count == 1:
        return (("R"+axis.lower(), angles[0], (0,)),)
    gray = [index ^ (index >> 1) for index in range(count)]
    sequence = []
    for index, pattern in enumerate(gray):
        theta = sp.expand(sum((-1)**(bin(state & pattern).count("1"))*angle
                              for state, angle in enumerate(angles))/count)
        if theta != 0:
            sequence.append(("R"+axis.lower(), theta, (0,)))
        changed = pattern ^ gray[(index+1) % count]
        control = changed.bit_length()
        sequence.append(("CZ" if axis == "X" else "CX", (control, 0)))
    return tuple(sequence)


def _controlled_local(matrix, controls, target):
    """Control an exact Clifford matrix without numeric Euler fitting."""
    matrix = sp.ImmutableMatrix(matrix)
    controls = tuple(controls)
    for name, angle in (("X", sp.pi), ("Sx", sp.pi/2), ("Sxdg", -sp.pi/2)):
        if matrix == gates.make_gate(name):
            return (("H", (target,)),)+_phase_on_wires(angle, controls+(target,))+(("H", (target,)),)
    if matrix == gates.make_gate("H"):
        return (("Ry", -sp.pi/4, (target,)),)+_phase_on_wires(sp.pi, controls+(target,))+(("Ry", sp.pi/4, (target,)),)
    state = exact_state(matrix)
    if state is None:
        raise ValueError("No exact controlled-local motif for this matrix")
    solution = solve_clifford_alphabet(("U3",))
    choices = [other for other in solution["distances"] if other//8 == state//8]
    selected = min(choices, key=lambda other: solution["distances"][other])
    word = clifford_path(solution, selected)
    phase = sp.pi*sp.Rational((state % 8-selected % 8) % 8, 4)
    if not word:
        return _phase_on_wires(phase, controls)
    if len(word) != 1 or word[0][0] not in {"U", "U3"} or len(controls) != 1:
        raise ValueError("Controlled Clifford requires a supported one-U3 chart")
    return (("CU3", *word[0][1:-1], (controls[0], target)),)+_phase_on_wires(phase, controls)


def diagonal_motif(entries):
    """Exact Walsh phase-polynomial synthesis, with the actual scalar phase."""
    count = len(entries)
    if count == 0 or count & (count-1):
        raise ValueError("A diagonal needs a power-of-two entry count")
    phases = [_phase(sp.sympify(entry)) for entry in entries]
    if None in phases:
        raise ValueError("Diagonal phases could not be recovered exactly")
    sequence = []
    for pattern in range(1, count):
        coefficient = sp.simplify(sum((-1)**(bin(state & pattern).count("1"))*phase
                                      for state, phase in enumerate(phases))/count)
        if coefficient == 0:
            continue
        wires = [bit for bit in range(count.bit_length()-1) if pattern & (1 << bit)]
        parity = tuple(("CX", (wire, wires[-1])) for wire in wires[:-1])
        sequence.extend(parity+(("Rz", -2*coefficient, (wires[-1],)),)+tuple(reversed(parity)))
    return tuple(sequence), sp.exp(sp.I*sp.simplify(sum(phases)/count))


def catalog_motif(name, parameters):
    """Return a structural source proposal, or None when no motif applies."""
    p = tuple(parameters)
    if name in {"XX90", "YY90", "ZZ90", "ZZMax"}:
        axis = {"XX90":"Rxx", "YY90":"Ryy", "ZZ90":"Rzz", "ZZMax":"Rzz"}[name]
        return ((axis, sp.pi/2, (0,1)),), sp.S.One
    if name in {"IonQZZ", "ZZPhase"}:
        return (("Rzz", (2*sp.pi if name == "IonQZZ" else sp.pi)*p[0], (0,1)),), sp.S.One
    if name in {"IonQMS", "IonQMSMax"}:
        phi0, phi1 = p[:2]
        theta = p[2] if name == "IonQMS" else sp.Rational(1,4)
        before = tuple(("Rz", -2*sp.pi*phase, (wire,)) for wire, phase in enumerate((phi0,phi1)))
        after = tuple(("Rz", 2*sp.pi*phase, (wire,)) for wire, phase in enumerate((phi0,phi1)))
        return before+(("Rxx", 2*sp.pi*theta, (0,1)),)+after, sp.S.One
    if name == "TK2":
        return (("Canonical", *(-sp.pi*value/2 for value in p), (0,1)),), sp.S.One
    if name == "CSXdg":
        return (("H", (1,)), ("CP", -sp.pi/2, (0,1)), ("H", (1,))), sp.S.One
    if name == "fFredkin":
        return (("CCZS", sp.pi/2, 0, 0, (0,1,2)),), sp.S.One
    if name == "Deutsch":
        theta = p[0]
        rotation = _uniform_rotations("X", (0,0,0,2*theta))
        wires = (2,0,1)
        return (("CP", sp.pi/2, (0,1)),)+tuple((*op[:-1],tuple(wires[q] for q in op[-1])) for op in rotation), sp.S.One
    if name in {"CCP", "CCZPow", "CCXPow"}:
        angle = p[0] if name == "CCP" else sp.pi*p[0]
        factor = sp.S.One if name == "CCP" else sp.exp(sp.I*sp.pi*p[0]*(p[1] if len(p)>1 else 0))
        middle = (("MCPhase", angle, 2, (0,1,2)),)
        if name == "CCXPow":
            middle = (("H", (2,)),)+middle+(("H", (2,)),)
        return middle, factor
    if name == "FSWAPPow":
        t, shift = p if len(p) == 2 else (*p, 0)
        return (("fSim", sp.pi*t/2, 0, (0,1)), ("P", sp.pi*t/2, (0,)),
                ("P", sp.pi*t/2, (1,))), sp.exp(sp.I*sp.pi*t*shift)
    if name == "PhasedISWAPPow":
        phase, t, shift = p if len(p) == 3 else (*p, 0)
        return (("PhasedFSim", -sp.pi*t/2, 0, 2*sp.pi*phase, 0, 0, (0,1)),), sp.exp(sp.I*sp.pi*t*shift)
    if name in {"CZS", "CCZS"}:
        theta, phi, gamma = p
        chi = -phi-sp.pi/2
        wires = (0,1) if name == "CZS" else (1,2)
        middle = _phase_on_wires(gamma+sp.pi, (0,) if name == "CZS" else (0,1))
        return (("PhasedFSim", -theta/2, 0, chi, 0, 0, wires),)+middle+(
                ("PhasedFSim", theta/2, 0, chi, 0, 0, wires),), sp.S.One
    if name == "Controlled":
        matrix, count = p[:2]
        if len(p) > 2 and p[2] is not None:
            raise ValueError("No open-control register motif implemented")
        return _controlled_local(matrix, tuple(range(count)), count), sp.S.One
    if name == "MCMT":
        matrix, controls, targets = p
        return tuple(op for target in range(controls, controls+targets)
                     for op in _controlled_local(matrix, tuple(range(controls)), target)), sp.S.One
    if name.startswith("UCR") and name in {"UCRX", "UCRY", "UCRZ"}:
        return _uniform_rotations(name[-1], p[0]), sp.S.One
    if name == "UC" and len(p[0]) == 2:
        first, last = map(sp.ImmutableMatrix, p[0])
        state = exact_state(first)
        if state is None:
            raise ValueError("Uniform Clifford motif needs a Clifford first branch")
        solution = solve_clifford_alphabet(("U3",))
        word = clifford_path(solution, state)
        return word+_controlled_local(last*first.H, (1,), 0), sp.S.One
    if name == "Diagonal":
        return diagonal_motif(p[0])
    if name == "Permutation":
        pattern = list(p[0])
        current = list(range(len(pattern)))
        word = []
        for wire, wanted in enumerate(pattern):
            other = current.index(wanted)
            if other != wire:
                current[wire], current[other] = current[other], current[wire]
                word.append(("SWAP", (wire,other)))
        return tuple(word), sp.S.One
    if name == "LinearFunction":
        matrix = [list(map(int, row)) for row in sp.Matrix(p[0]).tolist()]
        word = []
        for column in range(len(matrix)):
            pivot = next(row for row in range(column,len(matrix)) if matrix[row][column])
            if pivot != column:
                matrix[pivot], matrix[column] = matrix[column], matrix[pivot]
                word.append(("SWAP", (pivot,column)))
            for row in range(len(matrix)):
                if row != column and matrix[row][column]:
                    matrix[row] = [a ^ b for a,b in zip(matrix[row],matrix[column])]
                    word.append(("CX", (column,row)))
        return tuple(reversed(word)), sp.S.One
    return None
