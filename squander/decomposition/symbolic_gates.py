"""Exact symbolic qubit gate matrices and the original Clifford catalog.

Local wire 0 is the least-significant bit. Gate sequences are in circuit order.
This module has no numerical optimizer or import-time search dependency.
"""
from __future__ import annotations

import functools
import itertools
import operator
from dataclasses import dataclass
import sympy

if __package__:
    from .sympy_gen import clifford_decomps, clifford_descs, gate_inverses_clifford, aliases
else:
    from sympy_gen import clifford_decomps, clifford_descs, gate_inverses_clifford, aliases

little_endian = True
def bit_reverse(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)
def endian_swap_matrix(U):
    dim = U.shape[0]
    n = dim.bit_length() - 1
    assert 2**n == dim

    perm = [bit_reverse(i, n) for i in range(dim)]
    P = sympy.eye(dim)[perm, :]

    return P * U * P

def find_control_qubits(psi, num_qubits):
    """
    Identify control-qubit sets that make a unitary ‘pure-controlled’ or ‘sparse-controlled’.

    A subset S of qubits is:
        - pure-controlled if, when control pattern S is not fully satisfied in basis index j,
        row j of the unitary equals the corresponding computational basis row (identity row).
        - sparse-controlled if, when control pattern S is not fully satisfied in j,
        row j has zeros everywhere except possibly in columns k that satisfy S.

    Args:
        psi (sympy.Matrix): 2**num_qubits × 2**num_qubits unitary matrix.
        num_qubits (int): Total number of qubits.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            pure_controls:  list of control index lists (little-endian by default) making psi pure-controlled.
            sparsity_controls: list of control index lists making psi sparse-controlled.
    """
    pure_controls = []
    sparsity_controls = []
    for i in range(1, 1<<num_qubits):
        is_pure, is_sparse = True, True
        for j in range(1<<num_qubits):
            if i & j != i:
                if not all(psi[j, k] == (1 if j == k else 0) for k in range(1<<num_qubits)):
                    is_pure = False
                if not all(psi[j, k] == 0 for k in range(1<<num_qubits) if i & k == i):
                    is_sparse = False
        if is_pure: pure_controls.append([j if little_endian else num_qubits-1-j for j in range(num_qubits) if i & (1<<j)])
        if is_sparse: sparsity_controls.append([j if little_endian else num_qubits-1-j for j in range(num_qubits) if i & (1<<j)])
    return pure_controls, sparsity_controls
def make_gateprep_dict(): #primitive roots of unity
    d = {}
    for i1 in (3, 4, 6, 8, 9, 15, 16, 18, 20, 21): #0
        for sgn in (1, -1):
            z = sgn*sympy.exp(sympy.I*sympy.pi*i1/24)
            d[sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z)))] = z
        for i2 in (i1, (i1+12)%24, 24-i1):
            for sgn0, sgn1 in itertools.product((1, -1), repeat=2):
                z = sgn0*sympy.exp(sympy.I*sympy.pi*i1/24)/2+sgn1*sympy.exp(-sympy.I*sympy.pi*i2/24)/2
                #print(i1, i2, sgn0, sgn1, sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z))))
                d[sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z)))] = z
    return d
gateprep_dict = {}
#print(gateprep_dict); assert False
def has_radical(expr):
    return len(expr.atoms(sympy.Pow)) > 0
def separate_exp(expr):
    expexprs, nonexpexprs = [], []
    for x in sympy.Mul.make_args(expr):
        if x.atoms(sympy.exp):
            expexprs.append(x)
        else:
            nonexpexprs.append(x)
    return sympy.Mul(*expexprs), sympy.Mul(*nonexpexprs)
def gateprep(x):
    if not gateprep_dict:
        gateprep_dict.update(make_gateprep_dict())
    x = sympy.factor_terms(x)
    if x == 0: return x
    x = sympy.Mul(*[gateprep_dict[z] if z in gateprep_dict else z for y in sympy.separatevars(x).as_independent(*x.free_symbols) for z in separate_exp(y)])
    return quantsimp(x)
def quantsimp(x):
    #x = sympy.sympify(x)
    #if isinstance(x, sympy.Expr) and x.is_zero:
    #    return sympy.S.Zero if not x.is_Number else x
    x = x.rewrite(sympy.exp)#.rewrite(sympy.sqrt)
    #x = sympy.expand_power_exp(x)
    #x = sympy.powdenest(x)
    #x = sympy.sqrtdenest(x)
    #x = sympy.nsimplify(x, rational=True)
    #x = x.replace(lambda z: z.is_constant(), lambda z: sympy.simplify(sympy.expand_complex(z)))
    x = sympy.powsimp(x, combine='base')
    x = sympy.together(x)
    x = sympy.cancel(x)
    #x = sympy.powsimp(x, combine="base")
    x = sympy.expand_mul(x)
    #x = sympy.together(x)
    #x = sympy.cancel(x)
    #x = sympy.expand_complex(x)
    return x.doit()
def textbook_simp(x):
    #print(x)
    x = quantsimp(x).rewrite(sympy.sqrt)
    #x = sympy.expand_complex(x).simplify()
    x = sympy.factor_terms(x, radical=True)
    cs = [(sympy.Wild('c'+str(i), exclude=[z]), sympy.Wild('s'+str(i), exclude=list(x.free_symbols-{z}))) for i, z in enumerate(x.free_symbols)]
    for c, t in cs:
        x = x.replace(c*sympy.exp(sympy.I*t)+c*sympy.exp(-sympy.I*t), 2*c*sympy.cos(t)) #cosine definition
        x = x.replace(c*sympy.exp(sympy.I*t)-c*sympy.exp(-sympy.I*t), 2*c*sympy.I*sympy.sin(t)) #sine definition
    x = sympy.powsimp(x)
    for exprsn in (sympy.exp(sympy.I*sympy.pi*z/24) for z in range(-24, 24+1) if z % 12 != 0):
        if x == exprsn.rewrite(sympy.sqrt): x = exprsn #x = x.replace(exprsn.rewrite(sympy.sqrt), exprsn)
    x = sympy.expand_mul(x)
    x = sympy.factor_terms(x, radical=True) #for factoring 1/2
    x = sympy.factor_terms(x, radical=True) #for sign=True or -1 factoring
    #t = sympy.Wild('t', properties=[lambda k: k.is_Rational])
    #x = x.replace(-(-1)**t, sympy.exp(sympy.I*sympy.pi*(t+1)))
    #x = x.replace((-1)**t, sympy.exp(sympy.I*sympy.pi*t))
    #x = x.replace(sympy.I**t, sympy.exp(sympy.I*sympy.pi*t/2))
    #x = x.replace((-sympy.I)**t, sympy.exp(-sympy.I*sympy.pi*t/2))
    return x

def apply_to(psi, num_qubits, gate, gate_qubits):
    """
    Left-apply a k-qubit gate to designated positions inside an n-qubit operator.

    Embeds the k-qubit `gate` on `gate_qubits` (indices in [0..n-1]) under the
    current endianness, and returns (gate ⊗ I_rest) · psi with the correct qubit
    wiring. Works for psi as a full 2^n×2^n operator (not a statevector).

    Args:
        psi (sympy.Matrix): 2**num_qubits × 2**num_qubits operator to transform.
        num_qubits (int): Total number of qubits n.
        gate (sympy.Matrix): 2**k × 2**k unitary to embed.
        gate_qubits (Iterable[int]): The k target qubit indices where `gate` acts.

    Returns:
        sympy.Matrix: New 2**num_qubits × 2**num_qubits matrix after applying `gate`.
    """
    num_qubits = _integer(num_qubits, "num_qubits")
    gate_qubits = tuple(_integer(q, "wire") for q in gate_qubits)
    if len(set(gate_qubits)) != len(gate_qubits) or any(q >= num_qubits for q in gate_qubits):
        raise ValueError("Gate wires must be distinct and inside the register")
    if psi.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError("Input operator width does not match num_qubits")
    if gate.shape != (1 << len(gate_qubits), 1 << len(gate_qubits)):
        raise ValueError("Gate matrix width does not match its wires")
    pos = [(q if little_endian else num_qubits-1-q) for q in gate_qubits]
    k = len(gate_qubits)
    deltas, combo = [1<<p for p in pos], []
    for u in range (1 << k):
        m = 0; x = u; b = 0
        while x:
            if x & 1: m ^= deltas[b]
            x >>= 1; b += 1
        combo.append(m)
    pos_set = set(pos)
    rest = [p for p in range(num_qubits) if not p in pos_set]
    output = sympy.zeros(1<<num_qubits, 1<<num_qubits)
    for rest_bits in itertools.product((0, 1), repeat=len(rest)):
        base = 0
        for bit, p in zip(rest_bits, rest):
            if bit: base |= 1<<p
        mat, idx = sympy.zeros(0, 1<<num_qubits), []
        for u in range(1 << k):
            idx.append(base ^ combo[u])
            mat = mat.col_join(psi[idx[u],:])
        prod = gate @ mat
        for u in range(1 << k):
            output[idx[u],:] = prod[u,:].applyfunc(quantsimp)
    return output
def compile_gates(num_qubits, gates):
    """
    Compose a list of embedded gates into a full n-qubit unitary.

    Args:
        num_qubits (int): Total number of qubits n.
        gates (Iterable[tuple[sympy.Matrix, Iterable[int]]]):
            Sequence of (gate, gate_qubits) pairs. Each `gate` is 2**k×2**k,
            applied to the provided `gate_qubits`.

    Returns:
        sympy.Matrix: The resulting 2**n × 2**n unitary product.
    """
    num_qubits = _integer(num_qubits, "num_qubits")
    gates = list(gates)
    Umtx = sympy.eye(2**num_qubits)
    for gate in gates:
        Umtx = apply_to(Umtx, num_qubits, make_gate(*gate[:-1], prep=True) if isinstance(gate[0], str) else gate[0], gate[-1])
    return Umtx #.applyfunc(quantsimp)
def make_controlled(gate, gate_qubits, gateother=None): #control is first qubit, gate qubits come after
    """
    Build a 1-control controlled version of `gate` (control is the first qubit by convention).

    Constructs block-diagonal diag(gateother or I, gate) and applies a permutation so that,
    under little-endian, the control is the least significant qubit.

    Args:
        gate (sympy.Matrix): 2**m × 2**m target unitary.
        gate_qubits (int): Number of target qubits m (excluding the control qubit).
        gateother (sympy.Matrix | None): Optional block for the control-off subspace.
            If None, identity of size 2**m is used.

    Returns:
        sympy.Matrix: A 2**(m+1) × 2**(m+1) controlled-gate matrix.
    """
    res = sympy.diag(gateother if not gateother is None else sympy.eye(1<<gate_qubits), gate)
    P = sympy.eye(1<<(gate_qubits+1))[:, [2*x+y for y in (0, 1) for x in range(1<<gate_qubits)]]
    return P * res * P.T if little_endian else res

#H = U3(π/2, 0, π)
#U2(ϕ,λ)=U(π/2,ϕ,λ)
#S = U3(0, 0, π/2)
#Sdg = U3(0, 0, −π/2)
#T = U3(0, 0, π/4)
#Tdg = U3(0, 0, −π/4)
@functools.lru_cache(maxsize=None)
def _legacy_make_gate(name, *params, textbook=True, prep=False):
    res = gate_descs[name][3 if textbook and len(gate_descs[name]) > 3 else 2](*params)
    return res.applyfunc(gateprep) if prep else res
def make_gate_decomp(name, *params): return decomp_dict[name](*params)
def make_inverse(g): return g**-1
def dagger_gate(Utarget):
    Umtx = Utarget.conjugate().transpose()
    t = sympy.Wild('t')
    return Umtx.applyfunc(lambda x: x.replace(sympy.conjugate(sympy.I**t), (-sympy.I)**t).replace(sympy.conjugate((-sympy.I)**t), sympy.I**t))
def make_sqrt(g): return (g**sympy.Rational(1, 2)).applyfunc(lambda x: x.rewrite(sympy.exp).replace(lambda e: e.is_Float, lambda e: sympy.nsimplify(e, rational=True)))

vardict = {
    "theta": sympy.Symbol("θ", real=True),
    "phi": sympy.Symbol("ϕ", real=True),
    "lbda": sympy.Symbol("λ", real=True),
    "gamma": sympy.Symbol("γ", real=True),
    "alpha": sympy.Symbol("α", real=True),
    "beta": sympy.Symbol("β", real=True),
    "t": sympy.Symbol("t", real=True),
}

gate_descs = { #(num_qubits, num_params, sympy_generator_function)
    "GP": (0, [(1, "theta")], lambda theta, qbits: sympy.exp(theta*sympy.I)*sympy.eye(1<<qbits)),
    "I": (1, [], lambda: sympy.eye(2),
        lambda: sympy.Matrix([[1, 0], [0, 1]])),
    "H": (1, [], lambda: sympy.Matrix([[1, 1], [1, -1]])/sympy.sqrt(2),
        lambda: sympy.Matrix([[1/sympy.sqrt(2), 1/sympy.sqrt(2)], [1/sympy.sqrt(2), -1/sympy.sqrt(2)]])),
    "S": (1, [], lambda: sympy.Matrix([[1, 0], [0, sympy.I]])),
    "Sdg": (1, [], lambda: make_inverse(make_gate("S")),
        lambda: sympy.Matrix([[1, 0], [0, -sympy.I]])),
    "T": (1, [], lambda: sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*sympy.pi/4)]])),
    "Tdg": (1, [], lambda: make_inverse(make_gate("T")),
            lambda: sympy.Matrix([[1, 0], [0, sympy.exp(-sympy.I*sympy.pi/4)]])),
    "Sx": (1, [], lambda: make_sqrt(make_gate("X")).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[(1+sympy.I)/2, (1-sympy.I)/2], [(1-sympy.I)/2, (1+sympy.I)/2]]).applyfunc(sympy.factor_terms)),
    "Sxdg": (1, [], lambda: make_inverse(make_gate("Sx")).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[(1-sympy.I)/2, (1+sympy.I)/2], [(1+sympy.I)/2, (1-sympy.I)/2]]).applyfunc(sympy.factor_terms)), #sympy.Matrix([[1, -sympy.I], [-sympy.I, 1]])/2),
    "X": (1, [], lambda: sympy.Matrix([[0, 1], [1, 0]])),
    "Y": (1, [], lambda: sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])),
    "Z": (1, [], lambda: sympy.Matrix([[1, 0], [0, -1]])),

    "Rx": (1, [(2, "theta")], lambda theta: sympy.exp(-sympy.I*theta/2*make_gate("X")).applyfunc(textbook_simp),
        lambda theta: sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2)], [-sympy.I*sympy.sin(theta/2), sympy.cos(theta/2)]])),
    "Ry": (1, [(2, "theta")], lambda theta: sympy.exp(-sympy.I*theta/2*make_gate("Y")).applyfunc(textbook_simp),
        lambda theta: sympy.Matrix([[sympy.cos(theta/2), -sympy.sin(theta/2)], [sympy.sin(theta/2), sympy.cos(theta/2)]])),
    "Rz": (1, [(2, "phi")], lambda phi: sympy.exp(-sympy.I*phi/2*make_gate("Z")).applyfunc(textbook_simp),
        lambda phi: sympy.Matrix([[sympy.exp(-sympy.I*phi/2), 0], [0, sympy.exp(sympy.I*phi/2)]])),
    "P": (1, [(1, "theta")], lambda theta: sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*theta)]])),
    "U1": (1, [(1, "theta")], lambda theta: make_gate("P", theta)),
    "U": (1, [(2, "theta"), (1, "phi"), (1, "lbda")], lambda theta, phi, lbda: compile_gates(1, [("P", lbda, [0]), ("Ry", theta, [0]), ("P", phi, [0])]).applyfunc(textbook_simp),
        lambda theta, phi, lbda: sympy.Matrix([[sympy.cos(theta/2), -sympy.exp(sympy.I*lbda)*sympy.sin(theta/2)], [sympy.exp(sympy.I*phi)*sympy.sin(theta/2), sympy.exp(sympy.I*(phi+lbda))*sympy.cos(theta/2)]])),
    "U2": (1, [(1, "phi"), (1, "lbda")], lambda phi, lbda: make_gate("U", sympy.pi/2, phi, lbda),
        lambda phi, lbda: sympy.Matrix([[1/sympy.sqrt(2), -sympy.exp(sympy.I*lbda)/sympy.sqrt(2)], [sympy.exp(sympy.I*phi)/sympy.sqrt(2), sympy.exp(sympy.I*(lbda+phi))/sympy.sqrt(2)]])),
    "U3": (1, [(2, "theta"), (1, "phi"), (1, "lbda")], lambda theta, phi, lbda: make_gate("U", theta, phi, lbda)),
    "R": (1, [(2, "theta"), (1, "phi")], lambda theta, phi: make_gate("U", theta, phi-sympy.pi/2, -phi+sympy.pi/2).applyfunc(textbook_simp),
        lambda theta, phi: sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.exp(-sympy.I*phi)*sympy.sin(theta/2)], [-sympy.I*sympy.exp(sympy.I*phi)*sympy.sin(theta/2), sympy.cos(theta/2)]])),
    "CX": (2, [], lambda: sympy.Matrix(make_controlled(make_gate("X"), 1)),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])),
    "CNOT": (2, [], lambda: make_gate("CX")),
    "CY": (2, [], lambda: sympy.Matrix(make_controlled(make_gate("Y"), 1)),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 0, 0, -sympy.I], [0, 0, 1, 0], [0, sympy.I, 0, 0]])),
    "CZ": (2, [], lambda: sympy.Matrix(make_controlled(make_gate("Z"), 1)),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])),
    "CRX": (2, [(2, "theta")], lambda theta: make_controlled(make_gate("Rx", theta), 1),
        lambda theta: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(theta/2), 0, -sympy.I*sympy.sin(theta/2)], [0, 0, 1, 0], [0, -sympy.I*sympy.sin(theta/2), 0, sympy.cos(theta/2)]])),
    "CRY": (2, [(2, "theta")], lambda theta: make_controlled(make_gate("Ry", theta), 1),
        lambda theta: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(theta/2), 0, -sympy.sin(theta/2)], [0, 0, 1, 0], [0, sympy.sin(theta/2), 0, sympy.cos(theta/2)]])),
    "CRZ": (2, [(2, "theta")], lambda theta: make_controlled(make_gate("Rz", theta), 1),
        lambda theta: sympy.Matrix([[1, 0, 0, 0], [0, sympy.exp(-sympy.I*theta/2), 0, 0], [0, 0, 1, 0], [0, 0, 0, sympy.exp(sympy.I*theta/2)]])),
    "CSX": (2, [], lambda: make_controlled(make_gate("Sx"), 1),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, (1+sympy.I)/2, 0, (1-sympy.I)/2], [0, 0, 1, 0], [0, (1-sympy.I)/2, 0, (1+sympy.I)/2]]).applyfunc(sympy.factor_terms)),
    "CS": (2, [], lambda: make_controlled(make_gate("S"), 1),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sympy.I]])),
    "CH": (2, [], lambda: make_controlled(make_gate("H").applyfunc(gateprep), 1).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 1/sympy.sqrt(2), 0, 1/sympy.sqrt(2)], [0, 0, 1, 0], [0, 1/sympy.sqrt(2), 0, -1/sympy.sqrt(2)]])),
    "CP": (2, [(1, "phi")], lambda phi: make_controlled(make_gate("P", phi), 1),
        lambda phi: sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sympy.exp(sympy.I*phi)]])), #CR in some frameworks
    "CU1": (2, [(1, "theta")], lambda theta: make_gate("CP", theta)),
    "CU": (2, [(2, "theta"), (1, "phi"), (1, "lbda"), (1, "gamma")], lambda theta, phi, lbda, gamma: make_controlled(compile_gates(1, [("U", theta, phi, lbda, [0]), ("GP", gamma, 1, [0])]), 1).applyfunc(textbook_simp),
        lambda theta, phi, lbda, gamma: sympy.Matrix([[1, 0, 0, 0], [0, sympy.exp(sympy.I*gamma)*sympy.cos(theta/2), 0, -sympy.exp(sympy.I*(gamma+lbda))*sympy.sin(theta/2)], [0, 0, 1, 0], [0, sympy.exp(sympy.I*(gamma+phi))*sympy.sin(theta/2), 0, sympy.exp(sympy.I*(gamma+phi+lbda))*sympy.cos(theta/2)]])),
    "CU3": (2, [(2, "theta"), (1, "phi"), (1, "lbda")], lambda theta, phi, lbda: make_gate("CU", theta, phi, lbda, 0),
        lambda theta, phi, lbda: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(theta/2), 0, -sympy.exp(sympy.I*lbda)*sympy.sin(theta/2)], [0, 0, 1, 0], [0, sympy.exp(sympy.I*phi)*sympy.sin(theta/2), 0, sympy.exp(sympy.I*(phi+lbda))*sympy.cos(theta/2)]])),
    "CR": (2, [(2, "theta"), (1, "phi")], lambda theta, phi: make_controlled(make_gate("R", theta, phi), 1),
        lambda theta, phi: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(theta/2), 0, -sympy.I*sympy.exp(-sympy.I*phi)*sympy.sin(theta/2)], [0, 0, 1, 0], [0, -sympy.I*sympy.exp(sympy.I*phi)*sympy.sin(theta/2), 0, sympy.cos(theta/2)]])),
    "CROT": (2, [(2, "theta"), (1, "phi")], lambda theta, phi: make_controlled(make_gate("R", theta, phi), 1, make_gate("R", -theta, phi)),
        lambda theta, phi: sympy.Matrix([[sympy.cos(theta/2), 0, sympy.I*sympy.exp(-sympy.I*phi)*sympy.sin(theta/2), 0], [0, sympy.cos(theta/2), 0, -sympy.I*sympy.exp(-sympy.I*phi)*sympy.sin(theta/2)], [sympy.I*sympy.exp(sympy.I*phi)*sympy.sin(theta/2), 0, sympy.cos(theta/2), 0], [0, -sympy.I*sympy.exp(sympy.I*phi)*sympy.sin(theta/2), 0, sympy.cos(theta/2)]])),
    "Rxx": (2, [(2, "theta")], lambda theta: sympy.exp(-sympy.I*theta/2*compile_gates(2, [("X", [0]), ("X", [1])])).applyfunc(textbook_simp), #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_X(), [0]), (gen_X(), [1])])),
        lambda theta: sympy.Matrix([[sympy.cos(theta/2), 0, 0, -sympy.I*sympy.sin(theta/2)], [0, sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2), 0], [0, -sympy.I*sympy.sin(theta/2), sympy.cos(theta/2), 0], [-sympy.I*sympy.sin(theta/2), 0, 0, sympy.cos(theta/2)]])),
    "Ryy": (2, [(2, "theta")], lambda theta: sympy.exp(-sympy.I*theta/2*compile_gates(2, [("Y", [0]), ("Y", [1])])).applyfunc(textbook_simp), #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Y(), [0]), (gen_Y(), [1])])),
        lambda theta: sympy.Matrix([[sympy.cos(theta/2), 0, 0, sympy.I*sympy.sin(theta/2)], [0, sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2), 0], [0, -sympy.I*sympy.sin(theta/2), sympy.cos(theta/2), 0], [sympy.I*sympy.sin(theta/2), 0, 0, sympy.cos(theta/2)]])),
    "Rzz": (2, [(2, "theta")], lambda theta: sympy.exp(-sympy.I*theta/2*compile_gates(2, [("Z", [0]), ("Z", [1])])).applyfunc(textbook_simp), #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Z(), [0]), (gen_Z(), [1])])),
        lambda theta: sympy.Matrix([[sympy.exp(-sympy.I*theta/2), 0, 0, 0], [0, sympy.exp(sympy.I*theta/2), 0, 0], [0, 0, sympy.exp(sympy.I*theta/2), 0], [0, 0, 0, sympy.exp(-sympy.I*theta/2)]])),
    "Rxy": (2, [(2, "phi")], lambda phi: sympy.exp(-sympy.I*phi/4*(compile_gates(2, [("X", [0]), ("X", [1])]) + compile_gates(2, [("Y", [0]), ("Y", [1])]))).applyfunc(textbook_simp),
        lambda phi: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(phi/2), -sympy.I*sympy.sin(phi/2), 0], [0, -sympy.I*sympy.sin(phi/2), sympy.cos(phi/2), 0], [0, 0, 0, 1]])),
    "Rxmy": (2, [(2, "phi")], lambda phi: sympy.exp(-sympy.I*phi/4*(compile_gates(2, [("X", [0]), ("X", [1])]) - compile_gates(2, [("Y", [0]), ("Y", [1])]))).applyfunc(textbook_simp),
        lambda phi: sympy.Matrix([[sympy.cos(phi/2), 0, 0, -sympy.I*sympy.sin(phi/2)], [0, 1, 0, 0], [0, 0, 1, 0], [-sympy.I*sympy.sin(phi/2), 0, 0, sympy.cos(phi/2)]])),
    "xx_plus_yy": (2, [(2, "phi"), (1, "beta")], lambda phi, beta: compile_gates(2, [("Rz", beta, [0]), ("Rxy", phi, [0, 1]), ("Rz", -beta, [0])]).applyfunc(textbook_simp),
        lambda phi, beta: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(phi/2), -sympy.I*sympy.exp(-sympy.I*beta)*sympy.sin(phi/2), 0], [0, -sympy.I*sympy.exp(sympy.I*beta)*sympy.sin(phi/2), sympy.cos(phi/2), 0], [0, 0, 0, 1]])),
    "xx_minus_yy": (2, [(2, "phi"), (1, "beta")], lambda phi, beta: compile_gates(2, [("Rz", -beta, [0]), ("Rxmy", phi, [0, 1]), ("Rz", beta, [0])]).applyfunc(textbook_simp),
        lambda phi, beta: sympy.Matrix([[sympy.cos(phi/2), 0, 0, -sympy.I*sympy.exp(-sympy.I*beta)*sympy.sin(phi/2)], [0, 1, 0, 0], [0, 0, 1, 0], [-sympy.I*sympy.exp(sympy.I*beta)*sympy.sin(phi/2), 0, 0, sympy.cos(phi/2)]])),
    "iSWAP_pow": (2, [(2/sympy.pi, "alpha")], lambda alpha: (make_gate("iSWAP")**alpha).applyfunc(textbook_simp),
        lambda alpha: sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(sympy.pi*alpha/2), sympy.I*sympy.sin(sympy.pi*alpha/2), 0], [0, sympy.I*sympy.sin(sympy.pi*alpha/2), sympy.cos(sympy.pi*alpha/2), 0], [0, 0, 0, 1]])),
    "SYC": (2, [], lambda: make_gate("fSim", sympy.pi/2, sympy.pi/6).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 0, -sympy.I, 0], [0, -sympy.I, 0, 0], [0, 0, 0, sympy.exp(-sympy.I*sympy.pi/6)]])),
    "CCZ": (3, [], lambda: make_controlled(make_gate("CZ"), 2),
        lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])),
    "CCX": (3, [], lambda: make_controlled(make_gate("CX"), 2),
        lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]])),
    "Toffoli": (3, [], lambda: make_gate("CCX")),
    "CSWAP": (3, [], lambda: make_controlled(make_gate("SWAP"), 2),
        lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])),
    "SWAP": (2, [], lambda: functools.reduce(operator.add, (compile_gates(2, [(gen, [0]), (gen, [1])]) for gen in ("I", "X", "Y", "Z"))) / 2,
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
    "SSWAP": (2, [], lambda: make_sqrt(make_gate("SWAP")).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, (1 + sympy.I)/2, (1 - sympy.I)/2, 0], [0, (1 - sympy.I)/2, (1 + sympy.I)/2, 0], [0, 0, 0, 1]]).applyfunc(sympy.factor_terms)),
    "iSWAP": (2, [], lambda: make_gate("Rxy", -sympy.pi),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 0, sympy.I, 0], [0, sympy.I, 0, 0], [0, 0, 0, 1]])),
    "SiSWAP": (2, [], lambda: make_sqrt(make_gate("iSWAP")).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[1, 0, 0, 0], [0, 1/sympy.sqrt(2), sympy.I/sympy.sqrt(2), 0], [0, sympy.I/sympy.sqrt(2), 1/sympy.sqrt(2), 0], [0, 0, 0, 1]])),
    "CZPowGate": (2, [(1/sympy.pi, "t")], lambda t: make_gate("CP", sympy.pi*t)),
    "fSim": (2, [(1, "theta"), (1, "phi")], lambda theta, phi: compile_gates(2, [("iSWAP_pow", -2*theta/sympy.pi, [0, 1]), ("CZPowGate", -phi/sympy.pi, [0, 1])]).applyfunc(textbook_simp)),
    "fFredkin": (3, [], lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])),
    "Peres": (3, [], lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0]])),
    "Margolus": (3, [], lambda: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0]])),
    "Deutsch": (3, [(1, "theta")], lambda theta: sympy.Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, sympy.I*sympy.cos(theta), 0, 0, 0, sympy.sin(theta)], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, sympy.sin(theta), 0, 0, 0, sympy.I*sympy.cos(theta)]]))
}

gate_descs.update(clifford_descs)

#T=Rz(π/4)GP(pi/4), S=Rz(π/2)GP(pi/2), Z=Rz(π)GP(pi)
identities = ( #(gate1, gate2, result_gate)
    ("S", "S", "Z"),
    ("T", "T", "S"),
    ("Z", "S", "Sdg"),
    ("Sdg", "T", "Tdg"),
    ("Rx", "Rx", "Rx"),
    ("Ry", "Ry", "Ry"),
    ("Rz", "Rz", "Rz"),
)

#also single parameter angle inverses
identity_pairs = ('T', 'Tdg'), ('Tdg', 'T')
identity_angle_pairs = ('Rx', 'Rx', lambda x: (-x,)), ('Ry', 'Ry', lambda x: (-x,)), ('Rz', 'Rz', lambda x: (-x,)), ('P', 'P', lambda x: (-x,)), ('U2', 'U2', lambda x, y: (-y-sympy.pi, -x+sympy.pi)), ('U3', 'U3', lambda x, y, z: (-x, -z, -y))
gate_inverses = {x[0]: x[1:] for x in gate_inverses_clifford + identity_pairs + identity_angle_pairs}
basic_pauli = ('I', 'X', 'Y', 'Z')
t_gates = ('T', 'Tdg')
basic_clifford = basic_pauli + ('H', 'S', "Sdg", "Sx", "Sxdg")
extended_clifford = basic_clifford+("HX", "HY", "HZ", "SX", "SY", "HS", "HSX", "HSY", "HSdg", "SH", "SHX", "SHY", "SHZ", "SxY", "SxZ")
single_gate_clifford = tuple(x[0] for x in gate_inverses_clifford if x[0] not in clifford_decomps or len(clifford_decomps[x[0]]) == 1)
full_clifford = tuple(min([z for z in single_gate_clifford if (z == x or z.startswith(x + "_")) and (gate_inverses[z][0] == z or z[-1] == gate_inverses[z][0][-1] or not gate_inverses[z][0].startswith(z + "_") and (z[-1] if '_' in z else False, gate_inverses[z][0][-1] if '_' in gate_inverses[z][0] else False) in ((False, False), ('2', False), (False, '2'), ('6', False), (False, '6')))],
                          key=lambda z: 0 if z not in clifford_decomps else len(clifford_decomps[z]), default=None) for x in extended_clifford)
#print(full_clifford)
assert all(gate_inverses[x][0] in full_clifford for x in full_clifford), [x for x in full_clifford if gate_inverses[x][0] not in full_clifford]
assert len(full_clifford) == 24
complete_clifford = tuple(x[0] for x in gate_inverses_clifford)
cnot_full_clifford_t = ('CNOT',) + full_clifford + t_gates

qasm_standard_gate_library = (
    "P", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "Sx", "Rx", "Ry", "Rz",
    "CNOT", "CY", "CZ", "CP", "CRX", "CRY", "CRZ", "CH", "CU", "SWAP",
    "CCX", "CSWAP", "U", "I", "U1", "U2", "U3"
)

decomp_dict = {
    "U3": lambda theta, phi, lbda: compile_gates(1, [('P', lbda, (0,)), ('Ry', theta, (0,)), ('P', phi, (0,))]).applyfunc(textbook_simp),
    'GP': lambda theta, qbits: compile_gates(qbits, [gate for qbit in range(qbits) for gate in (("P", theta, [qbit]), ("X", [qbit]), ("P", theta, [qbit]), ("X", [qbit]))]),
    'CZ': lambda: compile_gates(2, [("H", [1]), ("CNOT", [0, 1]), ("H", [1])]),
    'CY': lambda: compile_gates(2, [("Sdg", [1]), ("CNOT", [0, 1]), ("S", [1])]),
    'CH': lambda: compile_gates(2, [("Ry", sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Ry", -sympy.pi/4, [1])]).applyfunc(textbook_simp),
    'CP': lambda phi: compile_gates(2, [("P", phi/2, [0]), ("CNOT", [0, 1]), ("P", -phi/2, [1]), ("CNOT", [0, 1]), ("P", phi/2, [1])]),
    'CRX': lambda theta: compile_gates(2, [("S", [1]), ("Ry", theta/2, [1]), ("CNOT", [0, 1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Sdg", [1])]).applyfunc(textbook_simp),
    'CRY': lambda theta: compile_gates(2, [("Ry", theta/2, [1]), ("CNOT", [0, 1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1])]).applyfunc(textbook_simp),
    'CRZ': lambda theta: compile_gates(2, [("Rz", theta/2, [1]), ("CNOT", [0, 1]), ("Rz", -theta/2, [1]), ("CNOT", [0, 1])]),
    'CSX': lambda: compile_gates(2, [("Sdg", [0]), ("H", [1]), ("Y", [1]), ("Tdg", [0]), ("CNOT", [0, 1]), ("Y", [1]), ("Tdg", [1]), ("CNOT", [0, 1]), ("T", [1]), ("H", [1])]).applyfunc(textbook_simp),
    'CS': lambda: compile_gates(2, [("T", [0]), ("T", [1]), ("CNOT", [0, 1]), ("Tdg", [1]), ("CNOT", [0, 1])]).applyfunc(textbook_simp), #compile_gates(2, [("Rz", sympy.pi/4, [0]), ("CNOT", [0, 1]), ("Rz", -sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Rz", sympy.pi/4, [1]), (gen_GP(sympy.pi/8, 2), [0, 1])]).applyfunc(textbook_simp)
    'CR': lambda theta, phi: compile_gates(2, [("Rz", -phi+sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Ry", theta/2, [1]), ("Rz", phi-sympy.pi/2, [1])]).applyfunc(textbook_simp),
    'CROT': lambda theta, phi: compile_gates(2, [("Rz", -phi, [1]), ("Ry", sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Rz", theta, [1]), ("CNOT", [0, 1]), ("Ry", -sympy.pi/2, [1]), ("Rz", phi, [1])]).applyfunc(textbook_simp),
    'Rxx': lambda theta: compile_gates(2, [("CNOT", [0, 1]), ("Rx", theta, [0]), ("CNOT", [0, 1])]).applyfunc(textbook_simp),
    'Ryy': lambda theta: compile_gates(2, [("Rx", sympy.pi/2, [0]), ("Rx", sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Rz", theta, [1]), ("CNOT", [0, 1]), ("Rx", -sympy.pi/2, [0]), ("Rx", -sympy.pi/2, [1])]).applyfunc(textbook_simp),
    'Rzz': lambda theta: compile_gates(2, [("CNOT", [0, 1]), ("Rz", theta, [1]), ("CNOT", [0, 1])]),
    'Rxy': lambda phi: compile_gates(2, [("Sdg", [1]), ("S", [0]), ("Sx", [1]), ("S", [1]), ("CNOT", [1, 0]), ("Ry", -phi/2, [0]), ("Ry", -phi/2, [1]), ("CNOT", [1, 0]), ("Sdg", [0]), ("Sdg", [1]), ("Sxdg", [1]), ("S", [1])]).applyfunc(textbook_simp), #compile_gates(2, [(gen_Rxx_decomp(phi/2), [0,1]), (gen_Ryy_decomp(phi/2), [0,1])]).applyfunc(textbook_simp)
    'Rxmy': lambda phi: compile_gates(2, [("Sdg", [0]), ("S", [1]), ("Sx", [0]), ("S", [0]), ("CNOT", [0, 1]), ("Ry", phi/2, [0]), ("Ry", -phi/2, [1]), ("CNOT", [0, 1]), ("Sdg", [0]), ("Sdg", [1]), ("Sxdg", [0]), ("S", [0])]).applyfunc(textbook_simp),
    'xx_plus_yy': lambda phi, beta: compile_gates(2, [("Rz", beta, [0]), ("Sdg", [1]), ("S", [0]), ("Sx", [1]), ("S", [1]), ("CNOT", [1, 0]), ("Ry", -phi/2, [0]), ("Ry", -phi/2, [1]), ("CNOT", [1, 0]), ("Sdg", [0]), ("Sdg", [1]), ("Sxdg", [1]), ("S", [1]), ("Rz", -beta, [0])]).applyfunc(textbook_simp),
    'xx_minus_yy': lambda phi, beta: compile_gates(2, [("Rz", beta, [1]), ("Sdg", [0]), ("S", [1]), ("Sx", [0]), ("S", [0]), ("CNOT", [0, 1]), ("Ry", phi/2, [0]), ("Ry", -phi/2, [1]), ("CNOT", [0, 1]), ("Sdg", [0]), ("Sdg", [1]), ("Sxdg", [0]), ("S", [0]), ("Rz", -beta, [1])]).applyfunc(textbook_simp),
    'CZPowGate': lambda t: make_gate_decomp("CP", sympy.pi*t),
    'iSWAP_pow': lambda alpha: make_gate_decomp("Rxy", -sympy.pi*alpha), #compile_gates(2, [("H", [0]), ("H", [1]), ("CNOT", [0,1]), ("Rz", -(alpha*sympy.pi)/2, [1]), ("CNOT", [0,1]), ("H", [0]), ("H", [1]), ("Sdg", [0]), ("Sdg", [1]), ("H", [0]), ("H", [1]), ("CNOT", [0,1]), ("Rz", -(alpha*sympy.pi)/2, [1]), ("CNOT", [0,1]), ("H", [0]), ("H", [1]), ("S", [0]), ("S", [1])]).applyfunc(textbook_simp)
    'fSim': lambda theta, phi: compile_gates(2, [(make_gate_decomp("iSWAP_pow", -2*theta/sympy.pi), [0, 1]), (make_gate_decomp("CZPowGate", -phi/sympy.pi), [0, 1])]).applyfunc(textbook_simp),
    #3 CNOT SYC: https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00248-8
    #'SYC': lambda: compile_gates(2, [("Rz", -3*sympy.pi/4, [0]), ("Rz", sympy.pi/4, [1]), ("Sx", [0]), ("Sx", [1]), ("Rz", -sympy.pi, [0]), ("Rz", sympy.pi, [1]), ("Sx", [1]), ("Rz", 5*sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Sx", [0]), ("Rz", -3*sympy.pi/4, [1]), ("Sx", [1]), ("Rz", sympy.pi, [1]), ("Sx", [1]), ("Rz", 9*sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Rz", sympy.pi/2, [0]), ("Sx", [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("Rz", 11*sympy.pi/12, [0]), ("Sx", [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("Rz", sympy.pi/6, [0]), ("Sx", [1]), ("Rz", -sympy.pi/3, [1]), ("GP", 17*sympy.pi/24, 2, [0, 1])]).applyfunc(textbook_simp), #gen_fSim_decomp(sympy.pi/2, sympy.pi/6)
    'SYC': lambda: compile_gates(2, [('P', -sympy.pi/12, (0,)), ('P', -sympy.pi/12, (1,)), ('CNOT', [0, 1]), ('P', -5*sympy.pi/12, (1,)), ('CNOT', [1, 0]), ('CNOT', [0, 1])]).applyfunc(textbook_simp),
    'CU': lambda theta, phi, lbda, gamma: compile_gates(2, [("Rz", (lbda-phi)/2, [1]), ("CNOT", [0, 1]), ("Rz", -(phi+lbda)/2, [1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Ry", theta/2, [1]), ("Rz", phi, [1]), ("P", (lbda+phi)/2+gamma, [0])]).applyfunc(textbook_simp),
    'CU3': lambda theta, phi, lbda: compile_gates(2, [("Rz", (lbda-phi)/2, [1]), ("CNOT", [0, 1]), ("Rz", -(phi+lbda)/2, [1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Ry", theta/2, [1]), ("Rz", phi, [1]), ("P", (lbda+phi)/2, [0])]).applyfunc(textbook_simp),
    'CCZ': lambda: compile_gates(3, [("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1])]),
    'CCX': lambda: compile_gates(3, [("H", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("H", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1])]),
    'SWAP': lambda: compile_gates(2, [("CNOT", [0, 1]), ("CNOT", [1, 0]), ("CNOT", [0, 1])]),
    #7 CNOT CSWAP: https://arxiv.org/pdf/2305.18128
    #'CSWAP': lambda: compile_gates(3, [("S", [1]), ("CNOT", [2, 1]), ("Sdg", [1]), ("Sx", [2]), ("T", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("T", [1]), ("Tdg", [2]), ("CNOT", [0, 2]), ("CNOT", [0, 1]), ("T", [2]), ("T", [0]), ("Tdg", [1]), ("H", [2]), ("CNOT", [0, 1]), ("CNOT", [2, 1]), ("GP", -sympy.pi/4, 3, [0, 1, 2])]).applyfunc(textbook_simp), #compile_gates(3, [("CNOT", [2, 1]), ("H", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("H", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1]), ("CNOT", [2, 1])])
    #'CSWAP': lambda: compile_gates(3, [('Tdg', (1,)), ('T', (2,)), ('CNOT', [2, 1]), ('T', (1,)), ('Rx', sympy.pi/2, (2,)), ('T', (2,)), ('CNOT', [0, 2]), ('T', (2,)), ('CNOT', [1, 2]), ('Tdg', (2,)), ('CNOT', [0, 2]), ('T', (2,)), ('Rx', -sympy.pi/4, (2,)), ('Ry', -sympy.pi/2, (2,)), ('CNOT', [0, 1]), ('T', (1,)), ('CNOT', [0, 1]), ('CNOT', [2, 1]), ('Tdg', (0,)), ('Tdg', (1,)), ('GP', -sympy.pi/8, 3, [0, 1, 2])]),
    'CSWAP': lambda: compile_gates(3, [('Tdg', (1,)), ('T', (2,)), ('CNOT', [2, 1]), ('T', (1,)), ('T', (2,)), ('U2', -sympy.pi/4, sympy.pi/4, (2,)), ('CNOT', [0, 2]), ('T', (2,)), ('CNOT', [1, 2]), ('Tdg', (2,)), ('CNOT', [0, 2]), ('U2', 3*sympy.pi/4, -3*sympy.pi/4, (2,)), ('CNOT', [0, 1]), ('T', (1,)), ('CNOT', [0, 1]), ('CNOT', [2, 1]), ('Tdg', (0,)), ('Tdg', (1,))]),
    'iSWAP': lambda: compile_gates(2, [("S", [0]), ("S", [1]), ("H", [0]), ("CNOT", [0, 1]), ("CNOT", [1, 0]), ("H", [1])]),
    'SSWAP': lambda: compile_gates(2, [("CNOT", [0, 1]), ("H", [0]), ("T", [0]), ("Tdg", [1]), ("H", [0]), ("H", [1]), ("CNOT", [0, 1]), ("H", [0]), ("H", [1]), ("Tdg", [0]), ("H", [0]), ("CNOT", [0, 1]), ("Sdg", [0]), ("S", [1])]).applyfunc(textbook_simp),
    #'SiSWAP': lambda: compile_gates(2, [("Sx", [0]), ("Rz", sympy.pi/2, [0]), ("CNOT", [0, 1]), ("Sx", [0]), ("Sx", [1]), ("Rz", sympy.pi*7/4, [0]), ("Rz", sympy.pi*7/4, [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [0]), ("CNOT", [0, 1]), ("Sx", [0]), ("Sx", [1])]).applyfunc(textbook_simp),
    'SiSWAP': lambda: compile_gates(2, [("Ry", sympy.pi/2, [0]), ("S", [1]), ("CNOT", [0, 1]), ("Ry", sympy.pi/4, [0]), ("Ry", sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Ry", -sympy.pi/2, [0]), ("Sdg", [1])]).applyfunc(textbook_simp),
}

decomp_dict.update({gen: lambda gen=gen: compile_gates(1, clifford_decomps[gen]).applyfunc(textbook_simp) for gen in clifford_decomps})

def test_decomp():
    for gatepair in identity_pairs + gate_inverses_clifford:
        assert compile_gates(1, [(gatepair[0], [0]), (gatepair[1], [0])]) == sympy.eye(2), gatepair
    for inv in gate_inverses:
        if len(gate_inverses[inv]) == 1:
            assert compile_gates(1, [(inv, [0]), (gate_inverses[inv][0], [0])]) == sympy.eye(2), inv
        else:
            params = [vardict[param[1]] for param in gate_descs[gate_inverses[inv][0]][1]]
            assert compile_gates(1, [(inv, *params, [0]), (gate_inverses[inv][0], *gate_inverses[inv][1](*params), [0])]).applyfunc(textbook_simp) == sympy.eye(2), inv
    for gate in gate_descs:
        print(gate)
        if gate != 'CSWAP': continue
        params = [vardict[param[1]] for param in gate_descs[gate][1]]
        if gate == "GP": params.append(1)
        textbook = make_gate(gate, *params)
        for i in range(len(gate_descs[gate][1])):
            print(i, params[i], gate_descs[gate][1][i][0], 2*sympy.pi)
            testperiod = make_gate(gate, *params[:i], params[i]+gate_descs[gate][1][i][0]*2*sympy.pi, *params[i+1:], prep=True)
            testperiod = testperiod.applyfunc(textbook_simp)
            assert textbook == testperiod, (textbook, testperiod, i)
        if len(gate_descs[gate]) == 4:
            res = make_gate(gate, *params, textbook=False)
            assert textbook == res, (textbook, res)
        #else: print(res)
        if gate in decomp_dict:
            check = make_gate_decomp(gate, *params)
            assert textbook == check, (textbook, check)
    #reverse CNOT is HxH-CNOT-HxH on both qubits
    assert compile_gates(2, [("CNOT", [1, 0])]) == compile_gates(2, [("H", [0]), ("H", [1]), ("CNOT", [0, 1]), ("H", [0]), ("H", [1]),])
    theta, phi, lbda = vardict["theta"], vardict["phi"], vardict["lbda"]
    theta2 = sympy.Symbol("θ2", real=True)
    print(find_control_qubits(make_gate("U3", theta, phi, lbda), 1), find_control_qubits(make_gate("CRY", theta), 2), find_control_qubits(make_gate("CCX"), 3))
    for i in range(3): #this proves any single qubit chain removes all purity, and converts aligning control to target
        print(f"U3({i})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [("U3", theta, phi, lbda, [i]), ("CRY", theta2, (0, 1))]), 3))
    for i in range(3):
        for j in range(3):
            if i == j : continue
            print(f"CRY({i}, {j})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [("CRY", theta, [i, j]), ("CRY", theta2, (0, 1))]), 3))


# Catalog conventions and sources are recorded alongside the public definitions.
CATALOG_SOURCES = {
    "wikipedia": "https://en.wikipedia.org/wiki/List_of_quantum_logic_gates",
    "qiskit": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library",
    "cirq": "https://quantumai.google/cirq/gatezoo",
}


@dataclass(frozen=True)
class GateDefinition:
    """A named unitary or constructor; continuous parameters are real angles.

    ``num_qubits=None`` denotes a constructor whose arguments determine its
    width. ``defaults`` contains defaults for trailing arguments. Provider names
    are aliases, never a license to discard a gate's exact global phase.
    """

    name: str
    num_qubits: object
    parameters: tuple
    defaults: tuple = ()
    source: str = "wikipedia"
    convention: str = "little-endian; circuit-order composition"


GATE_DEFINITIONS = {
    name: GateDefinition(name, desc[0], tuple(p[1] for p in desc[1]))
    for name, desc in gate_descs.items()
}
GATE_ALIASES = {}
_CONSTRUCTORS = {}


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _register(name, width, parameters, builder, *, defaults=(), source="wikipedia",
              convention="little-endian; circuit-order composition", constructor=False):
    GATE_DEFINITIONS[name] = GateDefinition(
        name, width, tuple(parameters), tuple(defaults), source, convention
    )
    if constructor:
        _CONSTRUCTORS[name] = builder
    else:
        # Keep the original experimental search's descriptor interface usable.
        gate_descs[name] = (width, [(None, p) for p in parameters], builder)


def _alias(alias, canonical):
    if alias in gate_descs and alias != canonical:
        # In particular, legacy SX/SY mean S*X/S*Y, not sqrt(X)/sqrt(Y).
        raise ValueError(f"Refusing to overwrite legacy gate name {alias!r}")
    GATE_ALIASES[alias] = canonical


def gate_definition(name):
    """Return metadata after resolving an explicitly registered alias."""
    try:
        return GATE_DEFINITIONS[GATE_ALIASES.get(name, name)]
    except KeyError as error:
        raise ValueError(f"Unknown symbolic gate {name!r}") from error


def _matrix_qubits(matrix):
    matrix = sympy.ImmutableMatrix(matrix)
    if matrix.rows != matrix.cols or matrix.rows < 1 or matrix.rows & (matrix.rows - 1):
        raise ValueError("A qubit gate must be square with power-of-two dimension")
    return matrix, matrix.rows.bit_length() - 1


def controlled_gate(matrix, num_controls=1, ctrl_state=None):
    """Control any exact gate, with controls on the low-index local wires.

    Control bit i is local wire i; target wires follow the controls. A zero
    control state implements open controls. The base gate's phase is retained.
    """
    matrix, targets = _matrix_qubits(matrix)
    controls = _integer(num_controls, "num_controls")
    state = (1 << controls) - 1 if ctrl_state is None else _integer(ctrl_state, "ctrl_state")
    if state >= 1 << controls:
        raise ValueError("ctrl_state does not fit num_controls")
    result = sympy.eye(1 << (targets + controls))
    for row in range(matrix.rows):
        for col in range(matrix.cols):
            result[(row << controls) | state, (col << controls) | state] = matrix[row, col]
    return sympy.ImmutableMatrix(result)


def adjoint_gate(name, *parameters):
    return make_gate(name, *parameters).H


def _pauli(label):
    if not isinstance(label, str) or not label or any(p not in "IXYZ" for p in label):
        raise ValueError("A Pauli label must be a nonempty string over I, X, Y, Z")
    # Qiskit labels list the most-significant wire first.
    return functools.reduce(sympy.kronecker_product, (make_gate(p) for p in label))


def _pauli_rotation(label, theta):
    pauli = _pauli(label)
    return sympy.cos(theta / 2) * sympy.eye(pauli.rows) - sympy.I * sympy.sin(theta / 2) * pauli


def _involution_power(matrix, exponent, global_shift=0):
    identity = sympy.eye(matrix.rows)
    return sympy.exp(sympy.I * sympy.pi * exponent * global_shift) * (
        (identity + matrix) / 2
        + sympy.exp(sympy.I * sympy.pi * exponent) * (identity - matrix) / 2
    )


def _cartan(a, b, c):
    """Wikipedia N(a,b,c) = exp(i(a XX + b YY + c ZZ))."""
    return sympy.Matrix([
        [sympy.exp(sympy.I*c)*sympy.cos(a-b), 0, 0, sympy.I*sympy.exp(sympy.I*c)*sympy.sin(a-b)],
        [0, sympy.exp(-sympy.I*c)*sympy.cos(a+b), sympy.I*sympy.exp(-sympy.I*c)*sympy.sin(a+b), 0],
        [0, sympy.I*sympy.exp(-sympy.I*c)*sympy.sin(a+b), sympy.exp(-sympy.I*c)*sympy.cos(a+b), 0],
        [sympy.I*sympy.exp(sympy.I*c)*sympy.sin(a-b), 0, 0, sympy.exp(sympy.I*c)*sympy.cos(a-b)],
    ])


def _fsim(theta, phi):
    return sympy.Matrix([
        [1, 0, 0, 0], [0, sympy.cos(theta), -sympy.I*sympy.sin(theta), 0],
        [0, -sympy.I*sympy.sin(theta), sympy.cos(theta), 0],
        [0, 0, 0, sympy.exp(-sympy.I*phi)],
    ])


def _phased_fsim(theta, zeta, chi, gamma, phi):
    # Cirq's first local wire is the high-order wire; convert explicitly.
    return endian_swap_matrix(sympy.Matrix([
        [1, 0, 0, 0],
        [0, sympy.exp(-sympy.I*(gamma+zeta))*sympy.cos(theta), -sympy.I*sympy.exp(sympy.I*(chi-gamma))*sympy.sin(theta), 0],
        [0, -sympy.I*sympy.exp(-sympy.I*(gamma+chi))*sympy.sin(theta), sympy.exp(sympy.I*(zeta-gamma))*sympy.cos(theta), 0],
        [0, 0, 0, sympy.exp(-sympy.I*(2*gamma+phi))],
    ]))


def _czswap(theta, phi, gamma):
    e = sympy.exp(sympy.I*gamma)
    c, s = sympy.cos(theta/2), sympy.sin(theta/2)
    return endian_swap_matrix(sympy.Matrix([
        [1, 0, 0, 0],
        [0, c*c-e*s*s, (1+e)*sympy.exp(-sympy.I*phi)*sympy.sin(theta)/2, 0],
        [0, (1+e)*sympy.exp(sympy.I*phi)*sympy.sin(theta)/2, s*s-e*c*c, 0],
        [0, 0, 0, -e],
    ]))


def _relative_controlled_x(kind):
    if kind == "RCCX":
        matrix = sympy.eye(8)
        matrix[3, 3] = matrix[7, 7] = 0
        matrix[7, 3], matrix[3, 7], matrix[5, 5] = sympy.I, -sympy.I, -1
    else:
        matrix = sympy.eye(16)
        matrix[7, 7] = matrix[15, 15] = 0
        matrix[15, 7], matrix[7, 15] = -1, 1
        matrix[3, 3], matrix[11, 11] = sympy.I, -sympy.I
    return matrix


# Fixed-width named gates absent from the original catalog.
_register("Sy", 1, (), lambda: _involution_power(make_gate("Y"), sympy.Rational(1, 2)))
_register("Sydg", 1, (), lambda: make_gate("Sy").H)
_register("SqrtH", 1, (), lambda: _involution_power(make_gate("H"), sympy.Rational(1, 2)))
_register("SqrtHdg", 1, (), lambda: make_gate("SqrtH").H)
_register("CSdg", 2, (), lambda: controlled_gate(make_gate("Sdg")), source="qiskit")
_register("CSXdg", 2, (), lambda: controlled_gate(make_gate("Sxdg")))
_register("CT", 2, (), lambda: controlled_gate(make_gate("T")))
_register("CTdg", 2, (), lambda: controlled_gate(make_gate("Tdg")))
_register("ACX", 2, (), lambda: controlled_gate(make_gate("X"), ctrl_state=0))
_register("DCX", 2, (), lambda: compile_gates(2, [("CX", (0, 1)), ("CX", (1, 0))]), source="qiskit")
_register("ECR", 2, (), lambda: sympy.Matrix([
    [0, 1, 0, sympy.I], [1, 0, -sympy.I, 0],
    [0, sympy.I, 0, 1], [-sympy.I, 0, 1, 0],
]) / sympy.sqrt(2), source="qiskit")
_register("Rzx", 2, ("theta",), lambda theta: _pauli_rotation("XZ", theta), source="qiskit")
_register("Rxz", 2, ("theta",), lambda theta: _pauli_rotation("ZX", theta))
_register("FSWAP", 2, (), lambda: sympy.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,-1]]))
_register("SSWAPdg", 2, (), lambda: make_gate("SSWAP").H)
_register("SiSWAPdg", 2, (), lambda: make_gate("SiSWAP").H)
_register("Canonical", 2, ("a", "b", "c"), _cartan,
          convention="exp(i(a XX + b YY + c ZZ)); angles in radians")
_register("B", 2, (), lambda: _cartan(sympy.pi/4, sympy.pi/8, 0))
_register("DB", 2, (), lambda: _fsim(3*sympy.pi/8, 0))
_register("Barenco", 2, ("alpha", "phi", "theta"),
          lambda alpha, phi, theta: controlled_gate(sympy.exp(sympy.I*alpha)*make_gate("R", 2*theta, phi)))
_register("Magic", 2, (), lambda: endian_swap_matrix(sympy.Matrix([
    [1,sympy.I,0,0], [0,0,sympy.I,1], [0,0,sympy.I,-1], [1,-sympy.I,0,0]
]) / sympy.sqrt(2)))
_register("Givens", 2, ("theta",), lambda theta: endian_swap_matrix(sympy.Matrix([
    [1,0,0,0], [0,sympy.cos(theta),-sympy.sin(theta),0],
    [0,sympy.sin(theta),sympy.cos(theta),0], [0,0,0,1],
])))
_register("CZS", 2, ("theta", "phi", "gamma"), _czswap)
_register("CCZS", 3, ("theta", "phi", "gamma"), lambda theta, phi, gamma: controlled_gate(_czswap(theta, phi, gamma)))
_register("RCCX", 3, (), lambda: _relative_controlled_x("RCCX"), source="qiskit")
_register("RC3X", 4, (), lambda: _relative_controlled_x("RC3X"), source="qiskit")
_register("C3X", 4, (), lambda: controlled_gate(make_gate("X"), 3), source="qiskit")
_register("C4X", 5, (), lambda: controlled_gate(make_gate("X"), 4), source="qiskit")
_register("C3SX", 4, (), lambda: controlled_gate(make_gate("Sx"), 3), source="qiskit")
_register("CCY", 3, (), lambda: controlled_gate(make_gate("Y"), 2))
_register("CCP", 3, ("theta",), lambda theta: controlled_gate(make_gate("P", theta), 2))
_register("XX90", 2, (), lambda: _pauli_rotation("XX", sympy.pi/2))
_register("YY90", 2, (), lambda: _pauli_rotation("YY", sympy.pi/2))
_register("ZZ90", 2, (), lambda: _pauli_rotation("ZZ", sympy.pi/2))
for _label in ("XX", "YY", "ZZ"):
    _register(_label, 2, (), lambda label=_label: _pauli(label), source="cirq")
_register("fSim", 2, ("theta", "phi"), _fsim, source="cirq")
_register("PhasedFSim", 2, ("theta", "zeta", "chi", "gamma", "phi"), _phased_fsim, source="cirq")

for _name, _base in [("XPow", "X"), ("YPow", "Y"), ("ZPow", "Z"), ("HPow", "H"),
                     ("CXPow", "CX"), ("CZPow", "CZ"), ("SWAPPow", "SWAP"),
                     ("CCXPow", "CCX"), ("CCZPow", "CCZ"), ("FSWAPPow", "FSWAP")]:
    _register(_name, gate_descs[_base][0], ("t", "global_shift"),
              lambda t, global_shift, base=_base: _involution_power(make_gate(base), t, global_shift),
              defaults=(0,), source="cirq")
for _name, _pauli_label in [("XXPow", "XX"), ("YYPow", "YY"), ("ZZPow", "ZZ")]:
    _register(_name, 2, ("t", "global_shift"),
              lambda t, global_shift, label=_pauli_label: _involution_power(_pauli(label), t, global_shift),
              defaults=(0,), source="cirq")
_register("ISWAPPow", 2, ("t", "global_shift"), lambda t, shift: sympy.exp(sympy.I*sympy.pi*t*shift)*sympy.Matrix([
    [1,0,0,0], [0,sympy.cos(sympy.pi*t/2),sympy.I*sympy.sin(sympy.pi*t/2),0],
    [0,sympy.I*sympy.sin(sympy.pi*t/2),sympy.cos(sympy.pi*t/2),0], [0,0,0,1],
]), defaults=(0,), source="cirq")
_register("PhasedXPow", 1, ("t", "phase_exponent", "global_shift"),
          lambda t, p, s: make_gate("ZPow", p)*make_gate("XPow", t, s)*make_gate("ZPow", -p), defaults=(0,), source="cirq")
_register("PhasedXZ", 1, ("x_exponent", "z_exponent", "axis_phase_exponent"),
          lambda x, z, a: make_gate("ZPow", z+a)*make_gate("XPow", x)*make_gate("ZPow", -a), source="cirq")
_register("PhasedISWAPPow", 2, ("phase_exponent", "t", "global_shift"),
          lambda p, t, shift: sympy.exp(sympy.I*sympy.pi*t*shift)*endian_swap_matrix(sympy.Matrix([
              [1,0,0,0],
              [0,sympy.cos(sympy.pi*t/2),sympy.I*sympy.sin(sympy.pi*t/2)*sympy.exp(2*sympy.pi*sympy.I*p),0],
              [0,sympy.I*sympy.sin(sympy.pi*t/2)*sympy.exp(-2*sympy.pi*sympy.I*p),sympy.cos(sympy.pi*t/2),0],
              [0,0,0,1],
          ])), defaults=(0,), source="cirq")


def _qft(num_qubits):
    width = _integer(num_qubits, "num_qubits")
    size = 1 << width
    return sympy.Matrix(size, size, lambda row, col: sympy.exp(2*sympy.pi*sympy.I*sympy.Rational(row*col, size)) / sympy.sqrt(size))


def _diagonal(entries):
    entries = tuple(map(sympy.sympify, entries))
    matrix, _ = _matrix_qubits(sympy.diag(*entries))
    if any(sympy.simplify(e*sympy.conjugate(e)-1) != 0 for e in entries):
        raise ValueError("Diagonal entries must have provable unit magnitude")
    return matrix


def _unitary(matrix):
    matrix, _ = _matrix_qubits(matrix)
    if any(sympy.simplify(e) != 0 for e in matrix.H*matrix-sympy.eye(matrix.rows)):
        raise ValueError("Matrix is not provably unitary")
    return matrix


def _permutation(pattern):
    pattern = tuple(_integer(p, "wire") for p in pattern)
    if sorted(pattern) != list(range(len(pattern))):
        raise ValueError("Wire permutation must contain each wire exactly once")
    matrix = sympy.zeros(1 << len(pattern))
    for col in range(matrix.cols):
        row = sum(((col >> old) & 1) << new for new, old in enumerate(pattern))
        matrix[row, col] = 1
    return matrix


def _linear_function(linear):
    linear = sympy.Matrix(linear)
    if linear.rows != linear.cols or any(x not in (0, 1) for x in linear):
        raise ValueError("Linear function must be a square binary matrix")
    if int(linear.det()) % 2 == 0:
        raise ValueError("Linear function must be invertible over GF(2)")
    result = sympy.zeros(1 << linear.rows)
    for col in range(result.cols):
        bits = linear*sympy.Matrix([(col >> q) & 1 for q in range(linear.rows)])
        row = sum((int(bit) % 2) << q for q, bit in enumerate(bits))
        result[row, col] = 1
    return result


def _multiplexor(gates):
    gates = tuple(_unitary(g) for g in gates)
    if not gates or len(gates) & (len(gates)-1) or any(g.shape != (2,2) for g in gates):
        raise ValueError("Multiplexor needs a power-of-two number of one-qubit unitaries")
    return sympy.diag(*gates)  # target is local wire 0, then the control wires


def _ms(num_qubits, theta):
    width = _integer(num_qubits, "num_qubits", 1)
    result = sympy.eye(1 << width)
    for a, b in itertools.combinations(range(width), 2):
        result = apply_to(result, width, make_gate("Rxx", theta), (a,b))
    return result


def _mcmt(matrix, num_controls, num_targets):
    matrix, width = _matrix_qubits(matrix)
    if width != 1:
        raise ValueError("MCMT's base gate must have one target qubit")
    targets = _integer(num_targets, "num_targets", 1)
    return controlled_gate(functools.reduce(sympy.kronecker_product, [matrix]*targets), num_controls)


def _rotation_vector(vx, vy, vz):
    radius = sympy.sqrt(vx*vx+vy*vy+vz*vz)
    return sympy.cos(radius/2)*sympy.eye(2) - sympy.I*sympy.sinc(radius/2)*(vx*make_gate("X")+vy*make_gate("Y")+vz*make_gate("Z"))/2


_register("GP", None, ("theta", "num_qubits"), lambda theta, n: sympy.exp(sympy.I*theta)*sympy.eye(1 << _integer(n, "num_qubits")), defaults=(0,), constructor=True)
_register("Identity", None, ("num_qubits",), lambda n: sympy.eye(1 << _integer(n, "num_qubits")), constructor=True)
_register("Controlled", None, ("matrix", "num_controls", "ctrl_state"), controlled_gate, defaults=(1,None), constructor=True)
_register("MCX", None, ("num_controls", "ctrl_state"), lambda n, s: controlled_gate(make_gate("X"), n, s), defaults=(None,), source="qiskit", constructor=True)
_register("MCPhase", None, ("theta", "num_controls", "ctrl_state"), lambda theta, n, s: controlled_gate(make_gate("P", theta), n, s), defaults=(None,), source="qiskit", constructor=True)
_register("MCMT", None, ("matrix", "num_controls", "num_targets"), _mcmt, source="qiskit", constructor=True)
_register("QFT", None, ("num_qubits",), _qft, constructor=True)
_register("QFTdg", None, ("num_qubits",), lambda n: _qft(n).H, constructor=True)
_register("WalshHadamard", None, ("num_qubits",), lambda n: functools.reduce(sympy.kronecker_product, [make_gate("H")]*_integer(n,"num_qubits",1)), constructor=True)
_register("Pauli", None, ("label",), _pauli, source="qiskit", constructor=True)
_register("PauliRotation", None, ("label", "theta"), _pauli_rotation, source="qiskit", constructor=True)
_register("Diagonal", None, ("entries",), _diagonal, source="qiskit", constructor=True)
_register("Unitary", None, ("matrix",), _unitary, source="qiskit", constructor=True)
_register("Permutation", None, ("pattern",), _permutation, source="qiskit", constructor=True)
_register("LinearFunction", None, ("matrix",), _linear_function, source="qiskit", constructor=True)
_register("UC", None, ("gates",), _multiplexor, source="qiskit", constructor=True)
for _axis in "XYZ":
    _register("UCR"+_axis, None, ("angles",), lambda angles, axis=_axis: _multiplexor([make_gate("R"+axis.lower(), theta) for theta in angles]), source="qiskit", constructor=True)
_register("MS", None, ("num_qubits", "theta"), _ms, source="qiskit", constructor=True)
_register("RV", 1, ("vx", "vy", "vz"), _rotation_vector, source="qiskit")

_QISKIT_NAMES = {
    "IGate":"I", "HGate":"H", "XGate":"X", "YGate":"Y", "ZGate":"Z",
    "SGate":"S", "SdgGate":"Sdg", "SXGate":"Sx", "SXdgGate":"Sxdg",
    "TGate":"T", "TdgGate":"Tdg", "PhaseGate":"P", "RGate":"R",
    "RXGate":"Rx", "RYGate":"Ry", "RZGate":"Rz", "UGate":"U",
    "U1Gate":"U1", "U2Gate":"U2", "U3Gate":"U3", "CXGate":"CX",
    "CYGate":"CY", "CZGate":"CZ", "CHGate":"CH", "CSGate":"CS",
    "CSdgGate":"CSdg", "CSXGate":"CSX", "CPhaseGate":"CP", "CRXGate":"CRX",
    "CRYGate":"CRY", "CRZGate":"CRZ", "CUGate":"CU", "CU1Gate":"CU1",
    "CU3Gate":"CU3", "DCXGate":"DCX", "ECRGate":"ECR", "iSwapGate":"iSWAP",
    "RXXGate":"Rxx", "RYYGate":"Ryy", "RZZGate":"Rzz", "RZXGate":"Rzx",
    "SwapGate":"SWAP", "XXPlusYYGate":"xx_plus_yy", "XXMinusYYGate":"xx_minus_yy",
    "CCXGate":"CCX", "CCZGate":"CCZ", "CSwapGate":"CSWAP", "RCCXGate":"RCCX",
    "RC3XGate":"RC3X", "C3XGate":"C3X", "C4XGate":"C4X", "C3SXGate":"C3SX",
    "GlobalPhaseGate":"GP", "MCXGate":"MCX", "MCPhaseGate":"MCPhase",
    "MCMTGate":"MCMT", "MSGate":"MS", "RVGate":"RV", "PauliGate":"Pauli",
    "PauliProductRotationGate":"PauliRotation", "DiagonalGate":"Diagonal",
    "PermutationGate":"Permutation", "UnitaryGate":"Unitary", "UCGate":"UC",
    "UCRXGate":"UCRX", "UCRYGate":"UCRY", "UCRZGate":"UCRZ", "QFTGate":"QFT",
}
for _alias_name, _canonical_name in _QISKIT_NAMES.items():
    _alias(_alias_name, _canonical_name)

# The original unary constructor is exactly Cirq's global_shift=0 case.
# Retain that call while accepting Cirq's optional phase parameter as well.
GATE_ALIASES["CZPowGate"] = "CZPow"
for _alias_name, _canonical_name in {
    "NOT":"X", "Feynman":"CX", "AntiCNOT":"ACX", "OpenCX":"ACX", "Fredkin":"CSWAP",
    "V":"Sx", "Vdg":"Sxdg", "SqrtX":"Sx", "SqrtXdg":"Sxdg", "SqrtY":"Sy", "SqrtZ":"S",
    "CV":"CSX", "SqrtSWAP":"SSWAP", "SqrtISWAP":"SiSWAP", "fSWAP":"FSWAP",
    "Berkeley":"B", "DagwoodBumstead":"DB", "N":"Canonical", "Can":"Canonical",
    "CPhase":"CP", "Sycamore":"SYC", "FSimGate":"fSim", "PhasedFSimGate":"PhasedFSim",
    "PhasedXPowGate":"PhasedXPow", "PhasedXZGate":"PhasedXZ", "SwapPowGate":"SWAPPow",
    "ISwapPowGate":"ISWAPPow", "CXPowGate":"CXPow", "XPowGate":"XPow", "YPowGate":"YPow",
    "ZPowGate":"ZPow", "HPowGate":"HPow", "CCXPowGate":"CCXPow", "CCZPowGate":"CCZPow",
    "XXPowGate":"XXPow", "YYPowGate":"YYPow", "ZZPowGate":"ZZPow", "PhasedISwapPowGate":"PhasedISWAPPow",
    "RX":"Rx", "RY":"Ry", "RZ":"Rz", "RXX":"Rxx", "RYY":"Ryy", "RZZ":"Rzz", "RZX":"Rzx",
    "id":"I", "sx":"Sx", "sxdg":"Sxdg", "cx":"CX", "ecr":"ECR", "global_phase":"GP",
}.items():
    _alias(_alias_name, _canonical_name)


def make_gate(name, *parameters, textbook=True, prep=False):
    """Construct an exact immutable matrix, preserving global phase.

    Real SymPy symbols and rational multiples of pi remain exact. Constructor
    arguments such as control counts, matrices, and Pauli labels are documented
    by ``gate_definition(name).parameters``. ``GP(theta)`` is a scalar 1x1 gate;
    use ``GP(theta, n)`` for a global phase on n qubits, as in the legacy tool.
    """
    definition = gate_definition(name)
    name = definition.name
    count = len(definition.parameters)
    required = count-len(definition.defaults)
    if not required <= len(parameters) <= count:
        raise ValueError(f"{name} expects {required}..{count} arguments: {definition.parameters}")
    if len(parameters) < count:
        parameters += definition.defaults[len(parameters)-required:]
    if name in _CONSTRUCTORS:
        matrix = _CONSTRUCTORS[name](*parameters)
    else:
        parameters = tuple(sympy.sympify(p) for p in parameters)
        matrix = _legacy_make_gate(name, *parameters, textbook=textbook, prep=prep)
    return sympy.ImmutableMatrix(matrix)


def list_gates(*, include_aliases=False):
    names = set(GATE_DEFINITIONS)
    if include_aliases:
        names.update(GATE_ALIASES)
    return tuple(sorted(names))
