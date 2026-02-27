from deap import base, creator, tools, gp, algorithms
import numpy as np
import functools, itertools, operator, random
import sympy
import multiprocessing
from squander import Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import N_Qubit_Decomposition_Guided_Tree

#A Symbolic Search Framework for Exact Multi-Basis Quantum Gate Decomposition

def determine_CNOT_structure(Umtx, params):
    allU = [] #2k+1 samples needed per Fourier analysis
    num_samples = 4 #this should be an even number to avoid sampling only at pi/2**x
    paramspace = list(itertools.product(*[np.linspace(0, np.pi*2*x[0], num=num_samples) for x in params]))
    for pos in paramspace:
        allU.append(np.array(Umtx.subs({x[1]: y for x, y in zip(params, pos)}).evalf()).astype(np.complex128))
    config = {'tree_level_max': 4 if len(Umtx) < 8 else 8, 'stop_first_solution': True, 'tolerance': 1e-10}
    optim = N_Qubit_Decomposition_Guided_Tree(allU, config, 0, None, paramspace=paramspace, paramscale=[x[0] for x in params])
    optim.set_Optimizer("BFGS2")
    optim.set_Verbose(0)
    optim.Start_Decomposition()
    cnot_structure = [(gate.get_Target_Qbit(), gate.get_Control_Qbit()) for gate in optim.get_Circuit().get_Gates() if gate.get_Name() == "CNOT"]
    print("CNOT structure:", cnot_structure, "num_samples:", num_samples)
    return cnot_structure

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
    for i1 in (3, 4, 6, 9, 15, 18, 20, 21): #0
        for sgn in (1, -1):
            z = sgn*sympy.exp(sympy.I*sympy.pi*i1/24)
            d[sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z)))] = z
        for i2 in (i1, (i1+12)%24):
            for sgn0, sgn1 in itertools.product((1, -1), repeat=2):
                z = sgn0*sympy.exp(sympy.I*sympy.pi*i1/24)/2+sgn1*sympy.exp(-sympy.I*sympy.pi*i2/24)/2
                #print(i1, i2, sgn0, sgn1, sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z))))
                d[sympy.factor_terms(sympy.expand_mul(sympy.expand_complex(z)))] = z
    return d
gateprep_dict = make_gateprep_dict()
def gateprep(x):
    x = sympy.factor_terms(x)
    if x in gateprep_dict: x = gateprep_dict[x]
    return quantsimp(x)
def quantsimp(x):
    #x = sympy.sympify(x)
    #if isinstance(x, sympy.Expr) and x.is_zero:
    #    return sympy.S.Zero if not x.is_Number else x
    #assert "sqrt" not in str(x), x
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
    if len(gates) and gates[0][-1] == list(range(num_qubits)): #shortcut for global gates
        Umtx = make_gate(*gates[0][:-1], prep=True) if isinstance(gates[0][0], str) else gates[0][0]
        gates = gates[1:]
    else:
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
def make_gate(name, *params, textbook=True, prep=False):
    res = gate_descs[name][3 if textbook and len(gate_descs[name]) > 3 else 2](*params)
    return res.applyfunc(gateprep) if prep else res 
def make_gate_decomp(name, *params): return decomp_dict[name](*params)
def make_inverse(g): return g**-1
def dagger_gate(Utarget):
    Umtx = Utarget.conjugate().transpose()
    t = sympy.Wild('t')
    return Umtx.applyfunc(lambda x: x.replace(sympy.conjugate(sympy.I**t), (-sympy.I)**t).replace(sympy.conjugate((-sympy.I)**t), sympy.I**t))
def make_sqrt(g): return (g**(1/2)).applyfunc(lambda x: x.rewrite(sympy.exp).replace(lambda e: e.is_Float, lambda e: sympy.nsimplify(e, rational=True)))

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
    "xx_minus_yy": (2, [(2, "phi"), (1, "beta")], lambda phi, beta: compile_gates(2, [("Rz", beta, [0]), ("Rxmy", phi, [0, 1]), ("Rz", -beta, [0])]).applyfunc(textbook_simp),
        lambda phi, beta: sympy.Matrix([[sympy.cos(phi/2), 0, 0, -sympy.I*sympy.exp(sympy.I*beta)*sympy.sin(phi/2)], [0, 1, 0, 0], [0, 0, 1, 0], [-sympy.I*sympy.exp(-sympy.I*beta)*sympy.sin(phi/2), 0, 0, sympy.cos(phi/2)]])),
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

clifford_descs = {
    #Clifford additions at all global phases
    #"I": (1, [], lambda: compile_gates(1, []),
    #    lambda: sympy.Matrix([[1, 0], [0, 1]])),
    "I_q1": (1, [], lambda: compile_gates(1, [('S', (0,)), ('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "I_q2": (1, [], lambda: compile_gates(1, [('Z', (0,)), ('Rz', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[sympy.I, 0], [0, sympy.I]])),
    "I_q3": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('Rz', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I - 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I - 1)/2]])),
    "I_q4": (1, [], lambda: compile_gates(1, [('Rx', -2*sympy.pi, (0,))]),
        lambda: sympy.Matrix([[-1, 0], [0, -1]])),
    "I_q5": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi, (0,)), ('S', (0,)), ('Rz', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, -sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "I_q6": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('Z', (0,)), ('Rx', -2*sympy.pi, (0,)), ('T', (0,)), ('Rz', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[-sympy.I, 0], [0, -sympy.I]])),
    "I_q7": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi, (0,)), ('H', (0,)), ('S', (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(1 - sympy.I)/2, 0], [0, sympy.sqrt(2)*(1 - sympy.I)/2]])),
    #"X": (1, [], lambda: compile_gates(1, [('X', (0,))]),
    #    lambda: sympy.Matrix([[0, 1], [1, 0]])),
    "X_q1": (1, [], lambda: compile_gates(1, [('T', (0,)), ('X', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "X_q2": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, sympy.I], [sympy.I, 0]])),
    "X_q3": (1, [], lambda: compile_gates(1, [('T', (0,)), ('Rx', -sympy.pi, (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I - 1)/2], [sympy.sqrt(2)*(sympy.I - 1)/2, 0]])),
    "X_q4": (1, [], lambda: compile_gates(1, [('Z', (0,)), ('Ry', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, -1], [-1, 0]])),
    "X_q5": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('Rx', sympy.pi, (0,)), ('Tdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, -sympy.sqrt(2)*(sympy.I + 1)/2], [-sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "X_q6": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, -sympy.I], [-sympy.I, 0]])),
    "X_q7": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('X', (0,)), ('Tdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(1 - sympy.I)/2], [sympy.sqrt(2)*(1 - sympy.I)/2, 0]])),
    #"Y": (1, [], lambda: compile_gates(1, [('Y', (0,))]),
    #    lambda: sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])),
    "Y_q1": (1, [], lambda: compile_gates(1, [('T', (0,)), ('Y', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(1 - sympy.I)/2], [sympy.sqrt(2)*(sympy.I - 1)/2, 0]])),
    "Y_q2": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, 1], [-1, 0]])),
    "Y_q3": (1, [], lambda: compile_gates(1, [('T', (0,)), ('Ry', -sympy.pi, (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I + 1)/2], [-sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "Y_q4": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[0, sympy.I], [-sympy.I, 0]])),
    "Y_q5": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('Ry', sympy.pi, (0,)), ('Tdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I - 1)/2], [sympy.sqrt(2)*(1 - sympy.I)/2, 0]])),
    "Y_q6": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, -1], [1, 0]])),
    "Y_q7": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('Y', (0,)), ('Tdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, -sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    #"Z": (1, [], lambda: compile_gates(1, [('Z', (0,))]),
    #    lambda: sympy.Matrix([[1, 0], [0, -1]])),
    "Z_q1": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, -sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "Z_q2": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[sympy.I, 0], [0, -sympy.I]])),
    "Z_q3": (1, [], lambda: compile_gates(1, [('S', (0,)), ('Rz', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I - 1)/2, 0], [0, sympy.sqrt(2)*(1 - sympy.I)/2]])),
    "Z_q4": (1, [], lambda: compile_gates(1, [('Tdg', (0,)), ('Z', (0,)), ('Rx', -2*sympy.pi, (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[-1, 0], [0, 1]])),
    "Z_q5": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi, (0,)), ('H', (0,)), ('S', (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Rz', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "Z_q6": (1, [], lambda: compile_gates(1, [('Rz', sympy.pi, (0,))]),
        lambda: sympy.Matrix([[-sympy.I, 0], [0, sympy.I]])),
    "Z_q7": (1, [], lambda: compile_gates(1, [('Rz', sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(1 - sympy.I)/2, 0], [0, sympy.sqrt(2)*(sympy.I - 1)/2]])),
    #"H": (1, [], lambda: compile_gates(1, [('H', (0,))]).applyfunc(textbook_simp),
    #    lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "H_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('H', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I + 1)/2], [(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "H_q2": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Ry', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "H_q3": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I - 1)/2], [(sympy.I - 1)/2, (1 - sympy.I)/2]])),
    "H_q4": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Ry', -sympy.pi/2, (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "H_q5": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Ry', -sympy.pi/2, (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, -(sympy.I + 1)/2], [-(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "H_q6": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Rx', -sympy.pi, (0,)), ('H', (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "H_q7": (1, [], lambda: compile_gates(1, [('H', (0,)), ('Tdg', (0,)), ('Tdg', (0,)), ('Rz', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (1 - sympy.I)/2], [(1 - sympy.I)/2, (sympy.I - 1)/2]])),
    "HX": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HX_q1": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Ry', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, -(sympy.I + 1)/2], [(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HX_q2": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HX_q3": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (1 - sympy.I)/2], [(sympy.I - 1)/2, (sympy.I - 1)/2]])),
    "HX_q4": (1, [], lambda: compile_gates(1, [('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HX_q5": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I + 1)/2], [-(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HX_q6": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Ry', sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HX_q7": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', sympy.pi/2, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I - 1)/2], [(1 - sympy.I)/2, (1 - sympy.I)/2]])),
    "HY": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi, (0,)), ('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HY_q1": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Rx', sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I - 1)/2], [(sympy.I - 1)/2, (sympy.I - 1)/2]])),
    "HY_q2": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HY_q3": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, -(sympy.I + 1)/2], [-(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HY_q4": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HY_q5": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (1 - sympy.I)/2], [(1 - sympy.I)/2, (1 - sympy.I)/2]])),
    "HY_q6": (1, [], lambda: compile_gates(1, [('Ry', -3*sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)/2], [sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HY_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I + 1)/2], [(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HZ": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HZ_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I + 1)/2], [-(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HZ_q2": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Rx', -sympy.pi, (0,)), ('Ry', 3*sympy.pi/2, (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HZ_q3": (1, [], lambda: compile_gates(1, [('Ry', 3*sympy.pi/2, (0,)), ('Tdg', (0,)), ('Tdg', (0,)), ('Rz', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I - 1)/2], [(1 - sympy.I)/2, (sympy.I - 1)/2]])),
    "HZ_q4": (1, [], lambda: compile_gates(1, [('Ry', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HZ_q5": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', 3*sympy.pi/2, (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, -(sympy.I + 1)/2], [(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HZ_q6": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Rx', -sympy.pi, (0,)), ('Ry', -sympy.pi/2, (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HZ_q7": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi/2, (0,)), ('Tdg', (0,)), ('Tdg', (0,)), ('Rz', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (1 - sympy.I)/2], [(sympy.I - 1)/2, (1 - sympy.I)/2]])),
    #"S": (1, [], lambda: compile_gates(1, [('S', (0,))]),
    #    lambda: sympy.Matrix([[1, 0], [0, sympy.I]])),
    "S_q1": (1, [], lambda: compile_gates(1, [('S', (0,)), ('Rz', -sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I - 1)/2]])),
    "S_q2": (1, [], lambda: compile_gates(1, [('Z', (0,)), ('T', (0,)), ('Rz', -sympy.pi/2, (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,)), ('Rx', -2*sympy.pi, (0,))]),
        lambda: sympy.Matrix([[sympy.I, 0], [0, -1]])),
    "S_q3": (1, [], lambda: compile_gates(1, [('Rz', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I - 1)/2, 0], [0, -sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "S_q4": (1, [], lambda: compile_gates(1, [('S', (0,)), ('Rx', -2*sympy.pi, (0,))]),
        lambda: sympy.Matrix([[-1, 0], [0, -sympy.I]])),
    "S_q5": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Sx', (0,)), ('Rz', -sympy.pi/2, (0,)), ('S', (0,)), ('Rz', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(1 - sympy.I)/2]])),
    "S_q6": (1, [], lambda: compile_gates(1, [('Z', (0,)), ('T', (0,)), ('Rz', -sympy.pi/2, (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-sympy.I, 0], [0, 1]])),
    "S_q7": (1, [], lambda: compile_gates(1, [('Rz', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(1 - sympy.I)/2, 0], [0, sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "SX": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[0, sympy.I], [1, 0]])),
    "SX_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I - 1)/2], [sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "SX_q2": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('Y', (0,))]),
        lambda: sympy.Matrix([[0, -1], [sympy.I, 0]])),
    "SX_q3": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,)), ('Y', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, -sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(sympy.I - 1)/2, 0]])),
    "SX_q4": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi, (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[0, -sympy.I], [-1, 0]])),
    "SX_q5": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi, (0,)), ('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(1 - sympy.I)/2], [-sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "SX_q6": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[0, 1], [-sympy.I, 0]])),
    "SX_q7": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(1 - sympy.I)/2, 0]])),
    "SY": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('Rx', -sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, 1], [sympy.I, 0]])),
    "SY_q1": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,)), ('Rx', -sympy.pi, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(sympy.I - 1)/2, 0]])),
    "SY_q2": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[0, sympy.I], [-1, 0]])),
    "SY_q3": (1, [], lambda: compile_gates(1, [('T', (0,)), ('Rx', -sympy.pi, (0,)), ('T', (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(sympy.I - 1)/2], [-sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "SY_q4": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('Rx', sympy.pi, (0,))]),
        lambda: sympy.Matrix([[0, -1], [-sympy.I, 0]])),
    "SY_q5": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,)), ('Rx', sympy.pi, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, -sympy.sqrt(2)*(sympy.I + 1)/2], [sympy.sqrt(2)*(1 - sympy.I)/2, 0]])),
    "SY_q6": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('X', (0,))]),
        lambda: sympy.Matrix([[0, -sympy.I], [1, 0]])),
    "SY_q7": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,)), ('X', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[0, sympy.sqrt(2)*(1 - sympy.I)/2], [sympy.sqrt(2)*(sympy.I + 1)/2, 0]])),
    "SZ": (1, [], lambda: compile_gates(1, [('Sdg', (0,))]),
        lambda: sympy.Matrix([[1, 0], [0, -sympy.I]])),
    "SZ_q1": (1, [], lambda: compile_gates(1, [('Rz', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(1 - sympy.I)/2]])),
    "SZ_q2": (1, [], lambda: compile_gates(1, [('T', (0,)), ('Rz', -sympy.pi, (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[sympy.I, 0], [0, 1]])),
    "SZ_q3": (1, [], lambda: compile_gates(1, [('S', (0,)), ('Rz', -sympy.pi/2, (0,)), ('T', (0,)), ('Rz', -sympy.pi, (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(sympy.I - 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "SZ_q4": (1, [], lambda: compile_gates(1, [('Sdg', (0,)), ('Rx', -2*sympy.pi, (0,))]),
        lambda: sympy.Matrix([[-1, 0], [0, sympy.I]])),
    "SZ_q5": (1, [], lambda: compile_gates(1, [('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*(sympy.I + 1)/2, 0], [0, sympy.sqrt(2)*(sympy.I - 1)/2]])),
    "SZ_q6": (1, [], lambda: compile_gates(1, [('Z', (0,)), ('T', (0,)), ('Rz', -sympy.pi/2, (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[-sympy.I, 0], [0, -1]])),
    "SZ_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Sxdg', (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*(1 - sympy.I)/2, 0], [0, -sympy.sqrt(2)*(sympy.I + 1)/2]])),
    "HS": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HS_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I + 1)/2], [(sympy.I - 1)/2, (1 - sympy.I)/2]])),
    "HS_q2": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi, (0,)), ('Ry', -sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HS_q3": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I - 1)/2], [-(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HS_q4": (1, [], lambda: compile_gates(1, [('Ry', 3*sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HS_q5": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, -(sympy.I + 1)/2], [(1 - sympy.I)/2, (sympy.I - 1)/2]])),
    "HS_q6": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -sympy.pi/2, (0,)), ('H', (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HS_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('H', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (1 - sympy.I)/2], [(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HSX": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Ry', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HSX_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (1 - sympy.I)/2], [(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HSX_q2": (1, [], lambda: compile_gates(1, [('Ry', -3*sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HSX_q3": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I + 1)/2], [(sympy.I - 1)/2, (sympy.I - 1)/2]])),
    "HSX_q4": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HSX_q5": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I - 1)/2], [-(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HSX_q6": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HSX_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Ry', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, -(sympy.I + 1)/2], [(1 - sympy.I)/2, (1 - sympy.I)/2]])),
    "HSY": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HSY_q1": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, -(sympy.I + 1)/2], [(sympy.I - 1)/2, (sympy.I - 1)/2]])),
    "HSY_q2": (1, [], lambda: compile_gates(1, [('Sxdg', (0,)), ('Ry', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HSY_q3": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (1 - sympy.I)/2], [-(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HSY_q4": (1, [], lambda: compile_gates(1, [('Ry', -3*sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HSY_q5": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Ry', -3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I + 1)/2], [(1 - sympy.I)/2, (1 - sympy.I)/2]])),
    "HSY_q6": (1, [], lambda: compile_gates(1, [('Sxdg', (0,)), ('Ry', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HSY_q7": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I - 1)/2], [(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "HSZ": (1, [], lambda: compile_gates(1, [('H', (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2]])),
    "HSZ_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('H', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I + 1)/2], [(1 - sympy.I)/2, (sympy.I - 1)/2]])),
    "HSZ_q2": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Ry', 3*sympy.pi/2, (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)/2]])),
    "HSZ_q3": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Ry', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I - 1)/2], [(sympy.I + 1)/2, -(sympy.I + 1)/2]])),
    "HSZ_q4": (1, [], lambda: compile_gates(1, [('Ry', 3*sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2]])),
    "HSZ_q5": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi/2, (0,)), ('Sdg', (0,)), ('Tdg', (0,)), ('Tdg', (0,)), ('Rz', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, -(sympy.I + 1)/2], [(sympy.I - 1)/2, (1 - sympy.I)/2]])),
    "HSZ_q6": (1, [], lambda: compile_gates(1, [('X', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)/2]])),
    "HSZ_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Ry', -sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (1 - sympy.I)/2], [-(sympy.I + 1)/2, (sympy.I + 1)/2]])),
    "SH": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SH_q1": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Y', (0,)), ('Sxdg', (0,)), ('Z', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I - 1)/2], [(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SH_q2": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Y', (0,)), ('Rx', -sympy.pi/2, (0,)), ('Z', (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SH_q3": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Sxdg', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, -(sympy.I + 1)/2], [(sympy.I - 1)/2, (sympy.I + 1)/2]])),
    "SH_q4": (1, [], lambda: compile_gates(1, [('Rx', 3*sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SH_q5": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sxdg', (0,)), ('Z', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (1 - sympy.I)/2], [-(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SH_q6": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Y', (0,)), ('Rx', 3*sympy.pi/2, (0,)), ('Z', (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SH_q7": (1, [], lambda: compile_gates(1, [('Sxdg', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I + 1)/2], [(1 - sympy.I)/2, -(sympy.I + 1)/2]])),
    "SHX": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SHX_q1": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (1 - sympy.I)/2], [(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SHX_q2": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Ry', -sympy.pi/2, (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SHX_q3": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Ry', -sympy.pi/2, (0,)), ('Rx', -2*sympy.pi, (0,)), ('Sx', (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I + 1)/2], [(sympy.I - 1)/2, -(sympy.I + 1)/2]])),
    "SHX_q4": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SHX_q5": (1, [], lambda: compile_gates(1, [('Rx', -2*sympy.pi, (0,)), ('Sx', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I - 1)/2], [-(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SHX_q6": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Ry', -sympy.pi/2, (0,)), ('Rx', sympy.pi/2, (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SHX_q7": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Ry', -sympy.pi/2, (0,)), ('Sx', (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, -(sympy.I + 1)/2], [(1 - sympy.I)/2, (sympy.I + 1)/2]])),
    "SHY": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SHY_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Ry', 3*sympy.pi/2, (0,)), ('S', (0,)), ('Sx', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, -(sympy.I + 1)/2], [(sympy.I - 1)/2, -(sympy.I + 1)/2]])),
    "SHY_q2": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SHY_q3": (1, [], lambda: compile_gates(1, [('Sx', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (1 - sympy.I)/2], [-(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SHY_q4": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SHY_q5": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sx', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I + 1)/2], [(1 - sympy.I)/2, (sympy.I + 1)/2]])),
    "SHY_q6": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('Sdg', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SHY_q7": (1, [], lambda: compile_gates(1, [('Rx', -2*sympy.pi, (0,)), ('Sx', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I - 1)/2], [(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SHZ": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SHZ_q1": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', -2*sympy.pi, (0,)), ('Sxdg', (0,)), ('Z', (0,)), ('T', (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I - 1)/2], [-(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SHZ_q2": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', 3*sympy.pi/2, (0,)), ('Z', (0,)), ('T', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SHZ_q3": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Sxdg', (0,)), ('Sdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, -(sympy.I + 1)/2], [(1 - sympy.I)/2, -(sympy.I + 1)/2]])),
    "SHZ_q4": (1, [], lambda: compile_gates(1, [('Rx', 3*sympy.pi/2, (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SHZ_q5": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sxdg', (0,)), ('Z', (0,)), ('T', (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (1 - sympy.I)/2], [(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SHZ_q6": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', -sympy.pi/2, (0,)), ('Z', (0,)), ('T', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SHZ_q7": (1, [], lambda: compile_gates(1, [('Sxdg', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I + 1)/2], [(sympy.I - 1)/2, (sympy.I + 1)/2]])),
    #"Sx": (1, [], lambda: compile_gates(1, [('Sx', (0,))]),
    #    lambda: sympy.Matrix([[(sympy.I + 1)/2, (1 - sympy.I)/2], [(1 - sympy.I)/2, (sympy.I + 1)/2]])),
    "Sx_q1": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi/2, (0,)), ('Sx', (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "Sx_q2": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,)), ('Tdg', (0,)), ('Tdg', (0,)), ('Rz', sympy.pi/2, (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I + 1)/2], [(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "Sx_q3": (1, [], lambda: compile_gates(1, [('Rx', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "Sx_q4": (1, [], lambda: compile_gates(1, [('Rx', -2*sympy.pi, (0,)), ('Sx', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I - 1)/2], [(sympy.I - 1)/2, -(sympy.I + 1)/2]])),
    "Sx_q5": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Ry', -sympy.pi, (0,)), ('Rx', sympy.pi/2, (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "Sx_q6": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Ry', -sympy.pi, (0,)), ('Sx', (0,)), ('Tdg', (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, -(sympy.I + 1)/2], [-(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "Sx_q7": (1, [], lambda: compile_gates(1, [('Rx', sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SxX": (1, [], lambda: compile_gates(1, [('Sxdg', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I + 1)/2], [(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SxX_q1": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SxX_q2": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Y', (0,)), ('Ry', -sympy.pi/2, (0,)), ('Rx', -sympy.pi/2, (0,)), ('Z', (0,)), ('S', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I - 1)/2], [(sympy.I - 1)/2, (sympy.I + 1)/2]])),
    "SxX_q3": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', 3*sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SxX_q4": (1, [], lambda: compile_gates(1, [('Rx', -2*sympy.pi, (0,)), ('Sxdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, -(sympy.I + 1)/2], [-(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SxX_q5": (1, [], lambda: compile_gates(1, [('Rx', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SxX_q6": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sxdg', (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (1 - sympy.I)/2], [(1 - sympy.I)/2, -(sympy.I + 1)/2]])),
    "SxX_q7": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', -sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SxY": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sxdg', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (1 - sympy.I)/2], [(sympy.I - 1)/2, (sympy.I + 1)/2]])),
    "SxY_q1": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Rx', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SxY_q2": (1, [], lambda: compile_gates(1, [('Sxdg', (0,)), ('Z', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, (sympy.I + 1)/2], [-(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SxY_q3": (1, [], lambda: compile_gates(1, [('Rx', -sympy.pi/2, (0,)), ('Z', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
    "SxY_q4": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi, (0,)), ('Sxdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (sympy.I - 1)/2], [(1 - sympy.I)/2, -(sympy.I + 1)/2]])),
    "SxY_q5": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi, (0,)), ('Rx', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SxY_q6": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Sxdg', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, -(sympy.I + 1)/2], [(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SxY_q7": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SxZ": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Sx', (0,))]),
        lambda: sympy.Matrix([[(sympy.I + 1)/2, (1 - sympy.I)/2], [(sympy.I - 1)/2, -(sympy.I + 1)/2]])),
    "SxZ_q1": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('X', (0,)), ('Ry', -3*sympy.pi/2, (0,)), ('Rx', -sympy.pi, (0,)), ('Tdg', (0,)), ('T', (0,)), ('Sx', (0,)), ('S', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2], [-sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2]])),
    "SxZ_q2": (1, [], lambda: compile_gates(1, [('Ry', -sympy.pi, (0,)), ('Sx', (0,))]),
        lambda: sympy.Matrix([[(sympy.I - 1)/2, (sympy.I + 1)/2], [-(sympy.I + 1)/2, (1 - sympy.I)/2]])),
    "SxZ_q3": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2], [-sympy.sqrt(2)*sympy.I/2, sympy.sqrt(2)/2]])),
    "SxZ_q4": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Sx', (0,)), ('Tdg', (0,)), ('T', (0,))]),
        lambda: sympy.Matrix([[-(sympy.I + 1)/2, (sympy.I - 1)/2], [(1 - sympy.I)/2, (sympy.I + 1)/2]])),
    "SxZ_q5": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Sx', (0,)), ('T', (0,)), ('T', (0,)), ('Rz', 3*sympy.pi/2, (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[-sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2], [sympy.sqrt(2)/2, sympy.sqrt(2)*sympy.I/2]])),
    "SxZ_q6": (1, [], lambda: compile_gates(1, [('Ry', sympy.pi, (0,)), ('Sx', (0,))]),
        lambda: sympy.Matrix([[(1 - sympy.I)/2, -(sympy.I + 1)/2], [(sympy.I + 1)/2, (sympy.I - 1)/2]])),
    "SxZ_q7": (1, [], lambda: compile_gates(1, [('Y', (0,)), ('Rx', -2*sympy.pi, (0,)), ('Rx', -3*sympy.pi/2, (0,)), ('Tdg', (0,)), ('T', (0,))]).applyfunc(textbook_simp),
        lambda: sympy.Matrix([[sympy.sqrt(2)/2, -sympy.sqrt(2)*sympy.I/2], [sympy.sqrt(2)*sympy.I/2, -sympy.sqrt(2)/2]])),
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

gate_inverses_clifford = (
    ('I', 'I'), ('I_q1', 'I_q7'), ('I_q2', 'I_q6'), ('I_q3', 'I_q5'), ('I_q4', 'I_q4'), ('I_q5', 'I_q3'), ('I_q6', 'I_q2'), ('I_q7', 'I_q1'), 
    ('X', 'X'), ('X_q1', 'X_q7'), ('X_q2', 'X_q6'), ('X_q3', 'X_q5'), ('X_q4', 'X_q4'), ('X_q5', 'X_q3'), ('X_q6', 'X_q2'), ('X_q7', 'X_q1'), 
    ('Y', 'Y'), ('Y_q1', 'Y_q7'), ('Y_q2', 'Y_q6'), ('Y_q3', 'Y_q5'), ('Y_q4', 'Y_q4'), ('Y_q5', 'Y_q3'), ('Y_q6', 'Y_q2'), ('Y_q7', 'Y_q1'), 
    ('Z', 'Z'), ('Z_q1', 'Z_q7'), ('Z_q2', 'Z_q6'), ('Z_q3', 'Z_q5'), ('Z_q4', 'Z_q4'), ('Z_q5', 'Z_q3'), ('Z_q6', 'Z_q2'), ('Z_q7', 'Z_q1'), 
    ('H', 'H'), ('H_q1', 'H_q7'), ('H_q2', 'H_q6'), ('H_q3', 'H_q5'), ('H_q4', 'H_q4'), ('H_q5', 'H_q3'), ('H_q6', 'H_q2'), ('H_q7', 'H_q1'), 
    ('HX', 'HZ'), ('HX_q1', 'HZ_q7'), ('HX_q2', 'HZ_q6'), ('HX_q3', 'HZ_q5'), ('HX_q4', 'HZ_q4'), ('HX_q5', 'HZ_q3'), ('HX_q6', 'HZ_q2'), ('HX_q7', 'HZ_q1'), 
    ('HY', 'HY_q4'), ('HY_q1', 'HY_q3'), ('HY_q2', 'HY_q2'), ('HY_q3', 'HY_q1'), ('HY_q4', 'HY'), ('HY_q5', 'HY_q7'), ('HY_q6', 'HY_q6'), ('HY_q7', 'HY_q5'), 
    ('HZ', 'HX'), ('HZ_q1', 'HX_q7'), ('HZ_q2', 'HX_q6'), ('HZ_q3', 'HX_q5'), ('HZ_q4', 'HX_q4'), ('HZ_q5', 'HX_q3'), ('HZ_q6', 'HX_q2'), ('HZ_q7', 'HX_q1'), 
    ('S', 'SZ'), ('S_q1', 'SZ_q7'), ('S_q2', 'SZ_q6'), ('S_q3', 'SZ_q5'), ('S_q4', 'SZ_q4'), ('S_q5', 'SZ_q3'), ('S_q6', 'SZ_q2'), ('S_q7', 'SZ_q1'), 
    ('SX', 'SX_q6'), ('SX_q1', 'SX_q5'), ('SX_q2', 'SX_q4'), ('SX_q3', 'SX_q3'), ('SX_q4', 'SX_q2'), ('SX_q5', 'SX_q1'), ('SX_q6', 'SX'), ('SX_q7', 'SX_q7'), 
    ('SY', 'SY_q6'), ('SY_q1', 'SY_q5'), ('SY_q2', 'SY_q4'), ('SY_q3', 'SY_q3'), ('SY_q4', 'SY_q2'), ('SY_q5', 'SY_q1'), ('SY_q6', 'SY'), ('SY_q7', 'SY_q7'), 
    ('SZ', 'S'), ('SZ_q1', 'S_q7'), ('SZ_q2', 'S_q6'), ('SZ_q3', 'S_q5'), ('SZ_q4', 'S_q4'), ('SZ_q5', 'S_q3'), ('SZ_q6', 'S_q2'), ('SZ_q7', 'S_q1'), 
    ('HS', 'SHX'), ('HS_q1', 'SHX_q7'), ('HS_q2', 'SHX_q6'), ('HS_q3', 'SHX_q5'), ('HS_q4', 'SHX_q4'), ('HS_q5', 'SHX_q3'), ('HS_q6', 'SHX_q2'), ('HS_q7', 'SHX_q1'), 
    ('HSX', 'SHZ_q6'), ('HSX_q1', 'SHZ_q5'), ('HSX_q2', 'SHZ_q4'), ('HSX_q3', 'SHZ_q3'), ('HSX_q4', 'SHZ_q2'), ('HSX_q5', 'SHZ_q1'), ('HSX_q6', 'SHZ'), ('HSX_q7', 'SHZ_q7'), 
    ('HSY', 'SHY_q2'), ('HSY_q1', 'SHY_q1'), ('HSY_q2', 'SHY'), ('HSY_q3', 'SHY_q7'), ('HSY_q4', 'SHY_q6'), ('HSY_q5', 'SHY_q5'), ('HSY_q6', 'SHY_q4'), ('HSY_q7', 'SHY_q3'), 
    ('HSZ', 'SH'), ('HSZ_q1', 'SH_q7'), ('HSZ_q2', 'SH_q6'), ('HSZ_q3', 'SH_q5'), ('HSZ_q4', 'SH_q4'), ('HSZ_q5', 'SH_q3'), ('HSZ_q6', 'SH_q2'), ('HSZ_q7', 'SH_q1'), 
    ('SH', 'HSZ'), ('SH_q1', 'HSZ_q7'), ('SH_q2', 'HSZ_q6'), ('SH_q3', 'HSZ_q5'), ('SH_q4', 'HSZ_q4'), ('SH_q5', 'HSZ_q3'), ('SH_q6', 'HSZ_q2'), ('SH_q7', 'HSZ_q1'), 
    ('SHX', 'HS'), ('SHX_q1', 'HS_q7'), ('SHX_q2', 'HS_q6'), ('SHX_q3', 'HS_q5'), ('SHX_q4', 'HS_q4'), ('SHX_q5', 'HS_q3'), ('SHX_q6', 'HS_q2'), ('SHX_q7', 'HS_q1'), 
    ('SHY', 'HSY_q2'), ('SHY_q1', 'HSY_q1'), ('SHY_q2', 'HSY'), ('SHY_q3', 'HSY_q7'), ('SHY_q4', 'HSY_q6'), ('SHY_q5', 'HSY_q5'), ('SHY_q6', 'HSY_q4'), ('SHY_q7', 'HSY_q3'), 
    ('SHZ', 'HSX_q6'), ('SHZ_q1', 'HSX_q5'), ('SHZ_q2', 'HSX_q4'), ('SHZ_q3', 'HSX_q3'), ('SHZ_q4', 'HSX_q2'), ('SHZ_q5', 'HSX_q1'), ('SHZ_q6', 'HSX'), ('SHZ_q7', 'HSX_q7'), 
    ('Sx', 'SxX'), ('Sx_q1', 'SxX_q7'), ('Sx_q2', 'SxX_q6'), ('Sx_q3', 'SxX_q5'), ('Sx_q4', 'SxX_q4'), ('Sx_q5', 'SxX_q3'), ('Sx_q6', 'SxX_q2'), ('Sx_q7', 'SxX_q1'), 
    ('SxX', 'Sx'), ('SxX_q1', 'Sx_q7'), ('SxX_q2', 'Sx_q6'), ('SxX_q3', 'Sx_q5'), ('SxX_q4', 'Sx_q4'), ('SxX_q5', 'Sx_q3'), ('SxX_q6', 'Sx_q2'), ('SxX_q7', 'Sx_q1'), 
    ('SxY', 'SxY_q6'), ('SxY_q1', 'SxY_q5'), ('SxY_q2', 'SxY_q4'), ('SxY_q3', 'SxY_q3'), ('SxY_q4', 'SxY_q2'), ('SxY_q5', 'SxY_q1'), ('SxY_q6', 'SxY'), ('SxY_q7', 'SxY_q7'), 
    ('SxZ', 'SxZ_q6'), ('SxZ_q1', 'SxZ_q5'), ('SxZ_q2', 'SxZ_q4'), ('SxZ_q3', 'SxZ_q3'), ('SxZ_q4', 'SxZ_q2'), ('SxZ_q5', 'SxZ_q1'), ('SxZ_q6', 'SxZ'), ('SxZ_q7', 'SxZ_q7'),
)

#also single parameter angle inverses
identity_pairs = ('T', 'Tdg'), ('Tdg', 'T')
identity_angle_pairs = ('Rx', 'Rx'), ('Ry', 'Ry'), ('Rz', 'Rz')
gate_inverses = gate_inverses_clifford + identity_pairs + identity_angle_pairs
basic_pauli = ('I', 'X', 'Y', 'Z')
basic_clifford = basic_pauli + ('H', "HX", "HY", "HZ", 'S', "SX", "SY", "SZ", "HS", "HSX", "HSY", "HSZ", "SH", "SHX", "SHY", "SHZ", "Sx", "SxX", "SxY", "SxZ")
full_clifford = tuple(x[0] for x in gate_inverses_clifford if x[0] == x[1] and x[0][-1] not in ('4', '6', '7') or x[0] != x[1] and x[0][-1]==x[1][-1] and x[0][-1] not in ('5', '7'))
complete_clifford = tuple(x[0] for x in gate_inverses_clifford)

qasm_standard_gate_library = [
    "P", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "Sx", "Rx", "Ry", "Rz",
    "CNOT", "CY", "CZ", "CP", "CRX", "CRY", "CRZ", "CH", "CU", "SWAP",
    "CCX", "CSWAP"
]

decomp_dict = {
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
    'SYC': lambda: compile_gates(2, [("Rz", -3*sympy.pi/4, [0]), ("Rz", sympy.pi/4, [1]), ("Sx", [0]), ("Sx", [1]), ("Rz", -sympy.pi, [0]), ("Rz", sympy.pi, [1]), ("Sx", [1]), ("Rz", 5*sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Sx", [0]), ("Rz", -3*sympy.pi/4, [1]), ("Sx", [1]), ("Rz", sympy.pi, [1]), ("Sx", [1]), ("Rz", 9*sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Rz", sympy.pi/2, [0]), ("Sx", [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("Rz", 11*sympy.pi/12, [0]), ("Sx", [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("CNOT", [0, 1]), ("Sx", [0]), ("Rz", sympy.pi/2, [1]), ("Rz", sympy.pi/6, [0]), ("Sx", [1]), ("Rz", -sympy.pi/3, [1]), ("GP", 17*sympy.pi/24, 2, [0, 1])]).applyfunc(textbook_simp), #gen_fSim_decomp(sympy.pi/2, sympy.pi/6)
    'CU': lambda theta, phi, lbda, gamma: compile_gates(2, [("Rz", (lbda-phi)/2, [1]), ("CNOT", [0, 1]), ("Rz", -(phi+lbda)/2, [1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Ry", theta/2, [1]), ("Rz", phi, [1]), ("P", (lbda+phi)/2+gamma, [0])]).applyfunc(textbook_simp),
    'CU3': lambda theta, phi, lbda: compile_gates(2, [("Rz", (lbda-phi)/2, [1]), ("CNOT", [0, 1]), ("Rz", -(phi+lbda)/2, [1]), ("Ry", -theta/2, [1]), ("CNOT", [0, 1]), ("Ry", theta/2, [1]), ("Rz", phi, [1]), ("P", (lbda+phi)/2, [0])]).applyfunc(textbook_simp),
    'CCZ': lambda: compile_gates(3, [("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1])]),
    'CCX': lambda: compile_gates(3, [("H", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("H", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1])]),
    'SWAP': lambda: compile_gates(2, [("CNOT", [0, 1]), ("CNOT", [1, 0]), ("CNOT", [0, 1])]),
    #7 CNOT CSWAP: https://arxiv.org/pdf/2305.18128
    'CSWAP': lambda: compile_gates(3, [("S", [1]), ("CNOT", [2, 1]), ("Sdg", [1]), ("Sx", [2]), ("T", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("T", [1]), ("Tdg", [2]), ("CNOT", [0, 2]), ("CNOT", [0, 1]), ("T", [2]), ("T", [0]), ("Tdg", [1]), ("H", [2]), ("CNOT", [0, 1]), ("CNOT", [2, 1]), ("GP", -sympy.pi/4, 3, [0, 1, 2])]).applyfunc(textbook_simp), #compile_gates(3, [("CNOT", [2, 1]), ("H", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [2]), ("CNOT", [1, 2]), ("Tdg", [2]), ("CNOT", [0, 2]), ("T", [1]), ("T", [2]), ("H", [2]), ("CNOT", [0, 1]), ("T", [0]), ("Tdg", [1]), ("CNOT", [0, 1]), ("CNOT", [2, 1])])
    'iSWAP': lambda: compile_gates(2, [("S", [0]), ("S", [1]), ("H", [0]), ("CNOT", [0, 1]), ("CNOT", [1, 0]), ("H", [1])]),
    'SSWAP': lambda: compile_gates(2, [("CNOT", [0, 1]), ("H", [0]), ("T", [0]), ("Tdg", [1]), ("H", [0]), ("H", [1]), ("CNOT", [0, 1]), ("H", [0]), ("H", [1]), ("Tdg", [0]), ("H", [0]), ("CNOT", [0, 1]), ("Sdg", [0]), ("S", [1])]).applyfunc(textbook_simp),
    #'SiSWAP': lambda: compile_gates(2, [("Sx", [0]), ("Rz", sympy.pi/2, [0]), ("CNOT", [0, 1]), ("Sx", [0]), ("Sx", [1]), ("Rz", sympy.pi*7/4, [0]), ("Rz", sympy.pi*7/4, [1]), ("Sx", [0]), ("Rz", sympy.pi/2, [0]), ("CNOT", [0, 1]), ("Sx", [0]), ("Sx", [1])]).applyfunc(textbook_simp),
    'SiSWAP': lambda: compile_gates(2, [("Ry", sympy.pi/2, [0]), ("S", [1]), ("CNOT", [0, 1]), ("Ry", sympy.pi/4, [0]), ("Ry", sympy.pi/4, [1]), ("CNOT", [0, 1]), ("Ry", -sympy.pi/2, [0]), ("Sdg", [1])]).applyfunc(textbook_simp),
}

def test_decomp():
    #for gatepair in identity_pairs + gate_inverses_clifford:
    #    assert compile_gates(1, [(gatepair[0], [0]), (gatepair[1], [0])]) == sympy.eye(2), gatepair
    for gate in gate_descs:
        print(gate)
        params = [vardict[param[1]] for param in gate_descs[gate][1]]
        if gate == "GP": params.append(1)
        textbook = make_gate(gate, *params)
        for i in range(len(gate_descs[gate][1])):
            if gate in ("U", "U2"): continue
            print(i, params[i], gate_descs[gate][1][i][0], 2*sympy.pi)
            testperiod = make_gate(gate, *params[:i], params[i]+gate_descs[gate][1][i][0]*2*sympy.pi, *params[i+1:])
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
#test_decomp(); assert False

QUBIT = int #np.int32

def sympy_to_gp(gates):
    pass
def gp_to_sympy(individual):
    pass
class ParamIndex:
    def __init__(self, index):
        self.index = index
    def __repr__(self): return f"ParamIndex({self.index})"
class AngleScale:
    def __init__(self, num, scale):
        self.num = num
        self.scale = scale
    def __repr__(self): return f"AngleScale_{'m' if self.num < 0 else ''}{abs(self.num)}_{self.scale}"
class ParamIndexSum:
    def __init__(self):
        self.params = []
    def add_param(self, param):
        self.params.append(param)
        return self
    def add_params(self, params):
        self.params.extend(params)
        return self
    def to_sympy(self, params):
        return sum([(2*sympy.pi if index.index == -1 else params[index.index]) * scale.num / scale.scale for index, scale in self.params])
    def to_qiskit(self, params):
        return sum([(2*np.pi if index.index == -1 else params[index.index]) * scale.num / scale.scale for index, scale in self.params])
    def __lt__(self, other): return len(self.params) < len(other.params)
    def __repr__(self): return f"ParamIndexSum({self.params})"
    def __str__(self):
        return " + ".join([f"{'2π' if index.index == -1 else 'ARG'+str(index.index)}*{scale.num}/{scale.scale}" for index, scale in self.params])
def make_angle(index : ParamIndex, scale : AngleScale):
    return ParamIndexSum().add_param((index, scale))
def angle_sum(angle1, angle2):
    return ParamIndexSum().add_params(angle1.params).add_params(angle2.params)
ANGLE = ParamIndexSum
class CircuitBuilder:
    def __init__(self):
        self.gates = []
    def add_gate(self, gate):
        self.gates.append(gate)
        return self
    def add_circuit(self, circuit):
        self.gates.extend(circuit.gates)
        return self
    def to_sympy(self, params, num_qubits):
        gate_ops = []
        for gate in self.gates:
            gate_name, qubits, gate_params = gate
            nqbits = gate_descs[gate_name][0]
            if nqbits == 0:
                gate_ops.append( (gate_name, *[param.to_sympy(params) for param in gate_params], len(qubits), qubits) )
            else: gate_ops.append( (gate_name, *[param.to_sympy(params) for param in gate_params], qubits) )
        return gate_ops
    def to_squander(self, params, num_qubits):
        circ = Circuit(num_qubits)
        params = []
        for gate in self.gates:
            gate_name, qubits, gate_params = gate
            nqbits, nparams = gate_descs[gate_name][:2]
            gen_func = "add_" + gate_name.upper()
            getattr(circ, gen_func)(*qubits)
            for param in gate_params:
                params.append(param.to_sympy(params))
        return circ, params
    def __repr__(self): return f"CircuitBuilder({self.gates})"

def gen_gate_qubits(name, qubits, *args):
    return CircuitBuilder().add_gate((name, qubits, args))
def gen_gate(name, *args):
    l = len(gate_descs[name][1])
    return CircuitBuilder().add_gate((name, args[l:], args[:l]))
def sequence(circuit1, circuit2):
    return CircuitBuilder().add_circuit(circuit1).add_circuit(circuit2)

def pass_through(x): return x
def randfloat_in_range(lo, hi): return ((hi-lo)*np.random.rand()-lo)

COST_METHOD_TRACE, COST_METHOD_FROBENIUS, COST_METHOD_HILBERT_SCHMIDT = 0, 1, 2
def expr_tree_size(expr: sympy.Expr) -> int:
    """Total number of nodes in the expression tree."""
    return sum(1 for _ in sympy.preorder_traversal(expr))

def expr_tree_depth(expr: sympy.Expr) -> int:
    """Maximum depth of the expression tree."""
    if not expr.args:
        return 1
    return 1 + max(expr_tree_depth(arg) for arg in expr.args)

def expr_leaf_count(expr: sympy.Expr) -> int:
    """Number of leaf nodes (atoms) in the expression tree."""
    if not expr.args:
        return 1
    return sum(expr_leaf_count(arg) for arg in expr.args)

def expr_structural_cost(expr: sympy.Expr,
                         weight_ops: float = 1.0,
                         weight_depth: float = 0.5,
                         weight_size: float = 0.2,
                         weight_leaves: float = 0.1) -> float:
    """
    Structural 'simplicity' cost for a SymPy expression.
    Lower is 'simpler'. Zero expression gets cost 0.
    """
    # Fast exact zero check
    if expr == 0:
        return 0.0
    terms = sympy.Add.make_args(expr)
    num_terms = len(terms)
    num_factors = sum(len(sympy.Mul.make_args(term)) for term in terms)
    return 16.0 + num_terms * 1.0 + num_factors * 0.5
    #ops = sympy.count_ops(expr, visual=False)
    #depth = expr_tree_depth(expr)
    #size = expr_tree_size(expr)
    #leaves = expr_leaf_count(expr)
    #return 1.0 + (weight_ops * ops +
    #        weight_depth * depth +
    #        weight_size * size +
    #        weight_leaves * leaves)
def extract_exp_i_phase(expr):
    """
    Try to write expr = rest * exp(I*t) where exp(I*t) is a multiplicative factor.
    Returns t if found, else None.
    Conservative: only succeeds if an exp(...) factor exists.
    """
    expr = sympy.sympify(expr)
    if expr.is_zero:
        return None

    # Pull out exp(...) factors multiplicatively
    rest, exp_part = expr.as_independent(sympy.exp, as_Add=False)

    # exp_part is product of exp(...) terms (or 1)
    if exp_part == 1:
        return None

    # Collapse product of exp(a)*exp(b) -> exp(a+b)
    exp_part = sympy.powsimp(exp_part, force=True)

    # Now accept only exp(I*t) exactly (or exp(-I*t))
    if exp_part.func is sympy.exp:
        arg = sympy.simplify(exp_part.args[0])
        # Check if arg is I*t with t real-ish symbolically
        t = sympy.simplify(arg / sympy.I)
        # Require that arg == I*t exactly (no remainder)
        if sympy.simplify(arg - sympy.I*t) == 0:
            return t

    return None
def dephase_matrix_by_first_clean_exp(U):
    """
    If we find a pivot entry with a clean multiplicative exp(I*t) factor,
    multiply entire U by exp(-I*t). Otherwise return U unchanged.
    """
    if any(U[i,i] == 1 for i in range(U.shape[0])): return U, None
    for e in list(U):  # row-major
        t = extract_exp_i_phase(e)
        if t is not None:
            phase = sympy.exp(-sympy.I*t)
            return (phase * U).applyfunc(quantsimp), t
    return U, None
def get_eval_circ(unitary, params, compiled_circuit):
    num_qubits = unitary.shape[0].bit_length() - 1
    param_syms_scaled = [vardict[x[1]]*x[0] for x in params]
    spcirc = compiled_circuit.to_sympy(param_syms_scaled, num_qubits)
    rescirc = compile_gates(num_qubits, [(unitary, list(range(num_qubits)))] + spcirc)
    return num_qubits, rescirc
def eval_symcost(num_qubits, rescirc, allow_global_phase=False):
    if allow_global_phase:
        if all(rescirc[i,j] == (rescirc[0,0] if i==j else 0) for i in range(1<<num_qubits) for j in range(1<<num_qubits)):
            return 0.0
        #diag = sympy.Trace(rescirc).rewrite(sympy.Sum)
        #gamma = diag / sympy.Abs(diag) if diag != 0 else 1 #canonical phase
        #rescirc = (rescirc / gamma).applyfunc(quantsimp)
    rescirc = rescirc - sympy.eye(1<<num_qubits)
    #rescirc = sympy.simplify(rescirc)
    #print(rescirc)
    symcost = sum(expr_structural_cost(expr) for expr in rescirc) / len(rescirc)
    return symcost
def eval_circ(unitary, params, compiled_circuit, symonly=False, allow_global_phase=False):
    num_qubits, rescirc = get_eval_circ(unitary, params, compiled_circuit)
    param_syms = [vardict[x[1]] for x in params]
    #rescirc = dephase_matrix_by_first_clean_exp(rescirc)[0]    
    if not symonly:
        tr = sympy.Trace(rescirc).rewrite(sympy.Sum)
        cost = 0.0
        num_samples = 4
        for pos in itertools.product(*[np.linspace(0, np.pi*2*x[0], num=num_samples) for x in params]):
            #.rewrite(sympy.exp)
            cost += 1.0 - complex(tr.evalf(subs={x: y for x, y in zip(param_syms, pos)})).real / (1<<num_qubits)
        cost = cost / (num_samples**len(params))
    else: cost = 0.0
    symcost = eval_symcost(num_qubits, rescirc, allow_global_phase=allow_global_phase)
    return symcost, cost, rescirc
def evaluate(individual, pset, hof, unitary, params, num_qubits, ansatz, cost_func_method):
    #print(individual)
    compiled_circuit = CircuitBuilder()
    count = 0
    for part in ansatz:
        if part is not None:
            angles = [gp.compile(expr=individual[count+i], pset=pset) for i in range(part[0])]
            if len(params) > 0: angles = [angle(*[ParamIndex(i) for i in range(len(params))]) for angle in angles]
            count += part[0]
            compiled_circuit.add_circuit(part[1](*angles))
        else:
            cc = gp.compile(expr=individual[count], pset=pset)
            if len(params) > 0: cc = cc(*[ParamIndex(i) for i in range(len(params))])
            compiled_circuit.add_circuit(cc)
            count += 1
    #print(compiled_circuit)
    symcost, cost, rescirc = eval_circ(unitary, params, compiled_circuit, allow_global_phase=True)
    if cost == 0.0 or symcost == 0:
        hof.clear()
        hof.insert(individual)
        print("Found exact solution:", [str(x) for x in individual], cost, symcost, len(individual))
    import math
    if math.isnan(cost) or math.isnan(symcost):
        print("NaN cost detected:", [str(x) for x in individual], cost, symcost, rescirc, unitary)
    #print(cost)
    #if cost < 0.0: print(individual, "Negative cost!", cost, tr, rescirc, unitary)
    return float(symcost), float(cost), len(individual)

def find_best_orientation(cnot_structure, Umtx, param_info):
    results = []
    for x in itertools.product(*[(x, tuple(reversed(x))) for x in cnot_structure]):
        compiled_circuit = CircuitBuilder()
        for qbits in x:
            compiled_circuit.add_circuit(gen_gate("CNOT", *qbits))
        results.append((eval_circ(Umtx, param_info, compiled_circuit, symonly=True), x))
    return min(results, key=lambda x: x[0][:2])[1]

def gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=False, gp_gate=False):
    if isinstance(gate, str):
        param_info = gate_descs[gate][1]
        Utarget = make_gate(gate, *[vardict[x[1]] for x in param_info], *((1,) if gate == "GP" else ()))
    else:
        param_info = gate[1]
        Utarget = gate[2](*[vardict[x[1]] for x in param_info])
    #Umtx = make_inverse(Utarget).applyfunc(gateprep)
    Umtx = dagger_gate(Utarget).applyfunc(gateprep)
    print("Target:", Utarget, "Dagger:", Umtx)
    num_qubits = Umtx.shape[0].bit_length() - 1
    assert 'CNOT' in basis
    print(gate, "num_qubits:", num_qubits, "scale_max:", scale_max, "num_params:", len(param_info), "layers:", layers)
    if gate in ("GP",): cnot_structure = []
    elif gate in ("CH", "CZ"): cnot_structure = [(0, 1)]
    elif gate in ("Rxy", "CRX", "CRY", "CRZ", "CP", "CS", "CSX", "CU3", "CU", "Rxx", "Ryy", "Rzz", "iSWAP", "SiSWAP"): cnot_structure = [(0, 1), (0, 1)]
    elif gate in ("SYC", "SWAP", "SSWAP",): cnot_structure = [(0, 1), (0, 1), (0, 1)]
    elif gate in ("Peres",): cnot_structure = [(0, 2), (0, 2), (1, 2), (0, 1), (1, 2)]
    elif gate in ("CCZ", "CCX"): cnot_structure = [(1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1)]
    elif gate in ("CSWAP",): cnot_structure = [(2, 1), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1), (2, 1)]
    elif num_qubits > 1: cnot_structure = determine_CNOT_structure(Umtx, param_info)
    else: cnot_structure = []
    if len(cnot_structure):
        cnot_structure = find_best_orientation(cnot_structure, Umtx, param_info)
        print("CNOT structure:", cnot_structure)
    if u3_ansatz:
        ansatz, regions = [], []
        for x, y in cnot_structure:
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [x])))
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [y])))
            regions.extend([False]*6)
            ansatz.append((0, functools.partial(gen_gate_qubits, 'CNOT', [x, y])))
        for i in range(num_qubits):
            ansatz.append((3, functools.partial(gen_gate_qubits, 'U3', [i])))
            regions.extend([False]*3)
    else:
        ansatz, regions = [None], [True]
        for x, y in cnot_structure:
            ansatz.extend(((0, functools.partial(gen_gate_qubits, 'CNOT', [x, y])), None))
            regions.append(True)
    if gp_gate:
        ansatz.append((1, functools.partial(gen_gate_qubits, 'GP', tuple(range(num_qubits)))))
        regions.append(False)
    return ansatz, regions, num_qubits, Umtx, param_info
def decompose_unitary(gate, scale_max, layers=4, basis=('CNOT', 'S', 'Sdg', 'T', 'Tdg', 'Ry', 'Rz', 'H')):
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=False)

    pset = gp.PrimitiveSetTyped("MAIN", [ParamIndex for _ in param_info], CircuitBuilder)
    psetangle = gp.PrimitiveSetTyped("MAIN", [ParamIndex for _ in param_info], ANGLE)
    for name in basis:
        if gate_descs[name][0] == 1:
            pset.addPrimitive(functools.partial(gen_gate, name), [ANGLE]*len(gate_descs[name][1])+[QUBIT]*gate_descs[name][0], CircuitBuilder, name=name)
    #pset.addPrimitive(CNOT_gate, [QUBIT, QUBIT], CircuitBuilder, name="CNOT")
    #for qubit1 in range(num_qubits):
    #    for qubit2 in range(num_qubits):
    #        if qubit1 == qubit2: continue
    #        pset.addPrimitive(functools.partial(CNOT_gate, control_qubit=qubit1, target_qubit=qubit2), [], CircuitBuilder, name=f"CNOT_{qubit1}_{qubit2}")
    #pset.addPrimitive(CRY_gate, [ANGLE, QUBIT, QUBIT], CircuitBuilder, name="CRY")
    pset.addPrimitive(sequence, [CircuitBuilder, CircuitBuilder], CircuitBuilder, name="sequence")
    pset.addEphemeralConstant("qubit_index", functools.partial(np.random.randint, 0, num_qubits), ret_type=QUBIT)

    for ps in (pset, psetangle):
        ps.addPrimitive(make_angle, [ParamIndex, AngleScale], ANGLE, name="make_angle")
        ps.addPrimitive(angle_sum, [ANGLE, ANGLE], ANGLE, name="angle_sum")
        
        ps.addTerminal(ParamIndexSum().add_param((ParamIndex(-1), AngleScale(scale_max, scale_max))), ANGLE, name="twopi_angle")
        ps.addTerminal(ParamIndex(-1), ParamIndex, name="twopi")
    
        # Avoid generating zero scale to prevent division by zero in angle construction
        for i in range(-scale_max, scale_max + 1):
            if i == 0: continue
            ps.addTerminal(AngleScale(i, scale_max), AngleScale, name=f"AngleScale_{'m' if i < 0 else ''}{abs(i)}")

    import operator
    # Set up the DEAP framework
    def init_individual(regions):
        return creator.Individual([gp.PrimitiveTree((toolbox if is_circ else toolboxangle).expr()) for is_circ in regions])
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.1, -0.01))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    tree_depth = num_qubits * (num_qubits - 1) * layers
    # Register the genetic operators
    toolbox = base.Toolbox()
    toolboxangle = base.Toolbox()
    def bounded_expr(pset, min_, max_, max_height=tree_depth):
        while True:
            expr = gp.genHalfAndHalf(pset=pset, min_=min_, max_=max_)
            # Temporarily build an individual to measure height
            ind = gp.PrimitiveTree(expr)
            if ind.height <= max_height:
                return expr
    def mutate_multi(ind, expr, pset):
        i = random.randrange(len(ind))
        ind[i], = gp.mutUniform(ind[i], expr=expr, pset=pset)
        return (ind,)    
    for tb, ps in ((toolbox, pset), (toolboxangle, psetangle)):
        tb.register("compile", gp.compile, pset=ps)
        tb.register("expr", bounded_expr, pset=ps, min_=0, max_=layers)
        tb.register("expr_mut", gp.genHalfAndHalf, pset=ps, min_=0, max_=layers)
        tb.register("mutate", mutate_multi, expr=tb.expr_mut, pset=ps)
    toolbox.register("individual", init_individual, regions=regions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    hof = tools.HallOfFame(1)
    #toolbox/pset contains wrapper functions
    cost_func_method = COST_METHOD_TRACE
    toolbox.register("evaluate", functools.partial(evaluate, pset=pset, hof=hof, unitary=Umtx, params=param_info, num_qubits=num_qubits, ansatz=ansatz, cost_func_method=cost_func_method))
    # Prefer parsimony to control tree bloat
    #toolbox.register("select", tools.selDoubleTournament, fitness_size=7, parsimony_size=1.4, fitness_first=True)
    toolbox.register("select", tools.selNSGA2)
    def mate_multi(ind1, ind2, termpb=0.9):
        i = random.randrange(len(ind1))
        ind1[i], ind2[i] = gp.cxOnePointLeafBiased(ind1[i], ind2[i], termpb=termpb)
        return ind1, ind2

    toolbox.register("mate", mate_multi, termpb=0.9)
    npopulation = 32*32
    def individual_height(ind):
        return max(tree.height for tree in ind)
    toolbox.decorate("mate", gp.staticLimit(key=individual_height, max_value=tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=individual_height, max_value=tree_depth))
    toolbox.register("generate", toolbox.population, n=npopulation)
    pop = toolbox.generate()
    #pop.insert(0, creator.Individual([gp.PrimitiveTree.from_string("Ry(make_angle(twopi, AngleScale_8), 1)", pset), gp.PrimitiveTree.from_string("Ry(make_angle(twopi, AngleScale_m8), 1)", pset), gp.PrimitiveTree.from_string("twopi_angle", pset)]))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(H(1), CNOT_0_1), H(1))", pset)))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(H(0), H(1)), sequence(sequence(CNOT_0_1, Rz(make_angle(ARG0, AngleScale_1), 1)), sequence(CNOT_0_1, sequence(H(0), H(1)))))", pset)))
    #pop.insert(0, creator.Individual(gp.PrimitiveTree.from_string("sequence(sequence(sequence(Rz(make_angle(twopi, AngleScale_m4), 1), CNOT_0_1), Rz(make_angle(twopi, AngleScale_m4), 1)), sequence(sequence(Ry(make_angle(twopi, AngleScale_8), 1), CNOT_0_1), sequence(Ry(make_angle(twopi, AngleScale_m8), 1), sequence(Rz(make_angle(twopi, AngleScale_2), 1), P(make_angle(twopi, AngleScale_4), 0)))))", pset)))
    #for x in pop: print(x)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=1000, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaMuCommaLambda(pop, toolbox, mu=100, lambda_=1000, cxpb=0.6, mutpb=0.3, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    #algorithms.eaGenerateUpdate(toolbox, ngen=100000, stats=stats, halloffame=hof, verbose=True)
    print("Best individual:", hof[0])
    print("Best individual fitness:", hof[0].fitness.values)

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return list(reversed(total_path))
def A_star(starts, isGoal, h, getNeighbors, d):
    openSet, cameFrom, gScore, fScore = set(starts), {}, {start: 0 for start in starts}, {start: h(start) for start in starts}
    while openSet:
        current = min(openSet, key=fScore.__getitem__)
        if isGoal(current): return reconstruct_path(cameFrom, current)
        openSet.remove(current)
        for neighbor in getNeighbors(current):
            tentative_gScore = gScore[current] + d(current, neighbor)
            if neighbor not in gScore or tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + h(neighbor)
                if neighbor not in openSet: openSet.add(neighbor)
    return None
def key_conv(x):
    return x
    re, im = x.as_real_imag()
    #return sympy.Trace(sympy.Abs(re)+sympy.Abs(im)*sympy.I).rewrite(sympy.Sum)
    return sympy.Add(*(sympy.Abs(re)+sympy.Abs(im)*sympy.I))
    #return sympy.Abs(sympy.Add(*(re+im)))
def best_first(starts, is_goal, neighbors, eval_E, canonicalize):
    import heapq
    p = multiprocessing.Pool()
    # starts: iterable of nodes (can be just [start_node])
    pq = []
    best_E_by_key = {}   # dominance pruning per canonical state

    for s in starts:
        E, k = eval_E(s)
        k = key_conv(k)
        best_E_by_key[k] = E
        heapq.heappush(pq, (E, s))
    while pq:
        E, node = heapq.heappop(pq)

        if is_goal(node, E):
            return node

        if any(x in best_E_by_key and best_E_by_key[x] <= E for x in canonicalize(node)):
            continue

        _, k = eval_E(node)
        knb = key_conv(k)
        if knb in best_E_by_key and E > best_E_by_key[knb]:
            continue
        print(E, node, k, knb)

        nbrs = neighbors(node)
        
        for nb, (Enb, knb) in zip(nbrs, p.map(eval_E, nbrs)):
        #for nb in nbrs:
        #    Enb, knb = eval_E(nb)
            knb = key_conv(knb)

            if knb in best_E_by_key and Enb >= best_E_by_key[knb]:
                continue

            best_E_by_key[knb] = Enb
            heapq.heappush(pq, (Enb, nb))

    return None
def make_circ(ansatz, node, num_qubits):
    compiled_circuit = CircuitBuilder()
    count = 0
    for part in ansatz:
        if part is not None:
            angles = node[count][:part[0]]
            #if len(param_info) > 0: angles = [angle(*[ParamIndex(i) for i in range(len(param_info))]) for angle in angles]
            count += part[0]
            compiled_circuit.add_circuit(part[1](*angles))
        else:
            for _ in range(num_qubits):
                cc = CircuitBuilder()
                for x in node[count]:
                    cc.add_circuit(gen_gate(*x))
                #if len(param_info) > 0: cc = cc(*[ParamIndex(i) for i in range(len(param_info))])
                compiled_circuit.add_circuit(cc)
                count += 1
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(1, 8)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(-1), AngleScale(-1, 8)), 1)) #CH
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-1, 2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-1, 2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(1, 2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("CNOT", 0, 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(2), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Ry", make_angle(ParamIndex(0), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0))
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(2), AngleScale(1, 1)), 0))#.add_circuit(gen_gate("Rz", make_angle(ParamIndex(1), AngleScale(1, 1)), 0))
    return compiled_circuit

#@lru_cache(None)
def eval(Umtx, param_info, ansatz, allow_global_phase, node):
    compiled_circuit = make_circ(ansatz, node, Umtx.shape[0].bit_length()-1)
    symcost, cost, U = eval_circ(Umtx, param_info, compiled_circuit, False, allow_global_phase=allow_global_phase)
    #print(compiled_circuit, symcost, cost, U)
    if cost == 0.0 or symcost == 0.0: print("Found exact solution:", node, cost, symcost, U)
    return (symcost, cost, sum(sum(1+len(gate_descs[z[0]][1])*0.75 for z in x) for x in node)), sympy.ImmutableDenseMatrix(U)
def gen_clifford():
    all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('Sx',)), (('I',), ('X',), ('Y',), ('Z',)))]
    #all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('H', 'S', 'H')), (('I',), ('H', 'S', 'S', 'H'), ('H', 'S', 'S', 'H', 'S', 'S'), ('S', 'S')))]
    clifford_gates = {}
    phases = tuple(make_angle(ParamIndex(-1), AngleScale(i, 8)) for i in range(8))
    print("Phases:", phases)
    for cliff in all_clifford:
        for i in range(len(phases)):
            cb = CircuitBuilder()
            for gate in (gen_gate(x, 0) for x in cliff):
                cb.add_circuit(gate)
            cb.add_circuit(gen_gate_qubits("GP", [0], phases[i]))
            mat = sympy.ImmutableMatrix(compile_gates(1, cb.to_sympy([], 1)).applyfunc(textbook_simp))
            res = decompose_unitary_search((1, [], lambda: mat), 4, basis=('CNOT', 'H', 'S', 'Sdg', 'Sx', 'Sxdg', 'X', 'Y', 'Z', 'T', 'Tdg', 'P', 'Rx', 'Ry', 'Rz'), allow_global_phase=False)            
            name = "".join(cliff).replace('I', '')
            if name == "": name = "I"
            clifford_gates[mat] = (name+(f"_q{i}" if i!=0 else ""), res)
            print(mat)
        #if len(clifford_gates) == 24: break
    subs = {sympy.I: sympy.Symbol("sympy.I", real=True),
            sympy.pi: sympy.Symbol("sympy.pi", real=True)}
    for gate in clifford_gates:
        strmat = str(gate.subs(subs).applyfunc(sympy.factor_terms)) #.replace(sympy.sqrt(2)/2, 1/sympy.sqrt(2))
        print(f'    \"{clifford_gates[gate][0]}\": (1, [], lambda: compile_gates(1, [{", ".join("("+repr(x[0])+", "+", ".join(str(y) for y in x[1:-1])+(", " if len(x[1:-1]) else "")+str(x[-1])+")" for x in clifford_gates[gate][1])}]){".applyfunc(textbook_simp)" if "sqrt" in strmat else ""},\n        lambda: sympy.{strmat.replace("sqrt", "sympy.sqrt")}),')
    s, count = "", 0
    for gate in clifford_gates:
        rev = sympy.ImmutableMatrix(compile_gates(1, [(dagger_gate(gate), [0])]).applyfunc(textbook_simp))
        s += f'({repr(clifford_gates[gate][0])}, {repr(clifford_gates[rev][0])}), '
        count += 1
        if count == len(phases):
            print(s)
            s, count = "", 0

#not maximally entangling: CS, CT
#maximally entangling: CNOT, CZ, CY, CH, iSWAP, SiSWAP (partial), XX(pi/4), exp(i pi/4 (X⊗X + Y⊗Y)), SSWAP (partial)
#maximally entangling only at specific angles: CRX, CRY, CRZ, CP, Rxy, iSWAP_pow
def decompose_unitary_search(gate, scale_max, layers=4, basis=('CNOT',)+full_clifford+('T', 'Tdg', 'P', 'Rx', 'Ry', 'Rz'), allow_global_phase=True):
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(gate, scale_max, layers, basis)
    num_regions = len(regions)*num_qubits
    all_angles = [
        make_angle(ParamIndex(i), AngleScale(scale, scale_max)) for i in range(-1, len(param_info)) for scale in range(-scale_max, scale_max+1) if scale != 0
    ]
    #comp_angles = {make_angle(ParamIndex(i), AngleScale(scale, scale_max)): make_angle(ParamIndex(i), AngleScale(-scale, scale_max)) for i in range(-1, len(param_info)) for scale in range(-scale_max, scale_max+1) if scale != 0}
    all_gates = [tuple(((gate, *angles, qbit),) if i==region*num_qubits+qbit else () for i in range(num_regions))
                 for gate in basis if gate_descs[gate][0] == 1
                 for region in range(len(regions)) if regions[region]
                 for qbit in range(num_qubits)
                 for angles in itertools.product(*[all_angles for _ in gate_descs[gate][1]])
             ]
    if num_qubits > 1:
        rev_angles = [
            make_angle(ParamIndex(i), AngleScale(scale, scale_max)) for i in range(-1, len(param_info)) for scale in range(scale_max, -scale_max-1, -1) if scale != 0
        ]
        all_gates.extend([tuple(((gate, *angles, qbit1),) if i==region*num_qubits+qbit1 else ((gate, *angles, qbit2),) if i==region*num_qubits+qbit2 else () for i in range(num_regions))
                        for gate in basis if gate_descs[gate][0] == 1
                        for region in range(len(regions)) if regions[region]
                        for qbit1 in range(num_qubits) for qbit2 in range(qbit1+1, num_qubits)
                        for angles in itertools.product(*[all_angles for _ in gate_descs[gate][1]])
                ])
        all_gates.extend([tuple(((gate1, *[x[0] for x in angles], qbit),) if i==region1*num_qubits+qbit else ((gate2, *[x[1] for x in angles], qbit),) if i==region2*num_qubits+qbit else () for i in range(num_regions))
                        for gate1, gate2 in gate_inverses if gate1 in basis and gate2 in basis
                        for region1 in range(len(regions)) if regions[region1]
                        #for region2 in (next(iter(region2 for region2 in range(region1+1, len(regions)) if regions[region2]), None),) if region2 is not None
                        for region2 in range(region1+1, len(regions)) if regions[region2]
                        for qbit in range(num_qubits)
                        for angles in itertools.product(*[list(zip(all_angles, rev_angles)) for _ in gate_descs[gate1][1]])
                ])
    #print(all_gates)

    #startcost = eval_circ(Umtx, param_info, CircuitBuilder())[0]
    starts = list(itertools.product(*[(((),) if region else ((x,) for x in all_angles)) for region in regions for qbit in range(num_qubits if region else 1)]))
    #print(starts)
    #res = A_star(starts, lambda node: eval(node) == 0, lambda node: eval(node),
    #       lambda node: [tuple(x+((gate[1:],) if gate[0] == i else ()) for i, x in enumerate(node)) for gate in gates],
    #       lambda node1, node2: abs(eval(node2) - eval(node1)))
    def get_neighbors(node):
        return [tuple(x[:i]+y+x[i:] for x, y, i in zip(node, gates, pos)) for gates in all_gates
                for pos in itertools.product(*[range((len(x) if len(gates[i]) > 0 else 0)+1) for i, x in enumerate(node)])]
    if num_qubits == 1:
        pauli_dressings = [
            CircuitBuilder().add_circuit(gen_gate(g, qubit)) for g in basic_pauli
            for qubit in range(num_qubits)
        ]
    elif num_qubits > 1:
        pauli_dressings = [
            CircuitBuilder().add_circuit(gen_gate(g1, qubit1)).add_circuit(gen_gate(g2, qubit2)) for g1, g2 in itertools.product(basic_pauli, repeat=2)
            for qubit1 in range(num_qubits) for qubit2 in range(qubit1+1, num_qubits)
        ]

    params = [vardict[x[1]]*x[0] for x in param_info]
    def canonicalize(node):
        return
        _, _, U = eval(Umtx, param_info, ansatz, allow_global_phase, node)  # populate cache
        preUs = [compile_gates(num_qubits, pauli_dressing_pre.to_sympy(params, num_qubits) + [(U, list(range(num_qubits)))]) for pauli_dressing_pre in pauli_dressings]
        yield from (mat for pauli_dressing in pauli_dressings for preU in preUs for mat in (sympy.ImmutableMatrix(eval_circ(preU, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)
        #yield from (mat for pauli_dressing in pauli_dressings for mat in (sympy.ImmutableMatrix(eval_circ(U, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)

    res = best_first(starts, lambda node, E: E[0] == 0.0 or E[1] == 0.0, get_neighbors,
                     functools.partial(eval, Umtx, param_info, ansatz, allow_global_phase), canonicalize)
    invparams = {vardict[x[1]]: x[1] for x in param_info}
    invparams.update({sympy.pi: sympy.Symbol("sympy.pi", real=True),
                      sympy.I: sympy.Symbol("sympy.I", real=True)})
    def circ_to_code(circ):
        return ", ".join(f"({gate}({', '.join(str(x.to_sympy(params)) for x in angles)}), ({', '.join(str(q) for q in qubits)},))" for gate, qubits, angles in circ.gates)
    def circ_to_compile(circ):
        return tuple((gate, *(str(x.to_sympy(params).subs(invparams)) for x in angles), qubits) for gate, qubits, angles in circ.gates)
    def circ_to_qiskit(circ):
        from qiskit.circuit import QuantumCircuit, Parameter
        qc = QuantumCircuit(num_qubits)
        namedict = {"CNOT": "cx", "H": "h", "S": "s", "Sdg": "sdg", "Sx": "sx", "Sxdg": "sxdg", "T": "t", "Tdg": "tdg", "X": "x", "Y": "y", "Z": "z", "P": "p", "Rx": "rx", "Ry": "ry", "Rz": "rz",
                    "HX": ("h", "x"), "HY": ("h", "y"), "HZ": ("h", "z"), "SX": ("s", "x"), "SY": ("s", "y"), "SZ": ("s", "z"), "HS": ("h", "s"), "HSX": ("h", "s", "x"), "HSY": ("h", "s", "y"), 
                    "HSZ": ("h", "sdg"), "SH": ("s", "h"), "SHX": ("s", "h", "x"), "SHY": ("s", "h", "y"), "SHZ": ("s", "h", "z"), "SxX": ("sx", "s"), "SxY": ("sx", "y"), "SxZ": ("sx", "z")
                    }
        p = [Parameter(str(x)) for x in params]
        for gate, qubits, angles in circ.gates:
            if gate in namedict:
                if isinstance(namedict[gate], tuple):
                    for g in namedict[gate]:
                        getattr(qc, g)(*(x.to_qiskit(p) for x in angles), *qubits)
                else:
                    getattr(qc, namedict[gate])(*(x.to_qiskit(p) for x in angles), *qubits)
        return qc
    fincirc = make_circ(ansatz, res, num_qubits)
    #print(circ_to_code(fincirc))
    compres = circ_to_compile(fincirc)
    print(compres)
    print(circ_to_qiskit(fincirc).draw())

    return compres

#decompose_unitary("CS", 16, 2)
#decompose_unitary("SYC", 48, 2)
#decompose_unitary("CCX", 1, 2)
#decompose_unitary("Rxy", 8, 2)
#decompose_unitary("SSWAP", 1, 2)
#decompose_unitary("SiSWAP", 8, 2)
#decompose_unitary("CRY", 2, 2)
#decompose_unitary("iSWAP_pow", 8, 2)
#decompose_unitary("Rxx", 2, 2)
#decompose_unitary("CZ", 2, 2)
#decompose_unitary("CH", 8, 2)
#decompose_unitary_search("CY", 1)
#decompose_unitary_search("CZ", 1)
#decompose_unitary_search("CH", 8)
#decompose_unitary_search("CP", 8)
#decompose_unitary_search("Rxx", 1)
#decompose_unitary_search("Ryy", 1)
#decompose_unitary_search("Rzz", 1)
#decompose_unitary_search("CS", 1)
#decompose_unitary_search("Rxy", 8)
#decompose_unitary_search("SWAP", 1)
#decompose_unitary_search("SSWAP", 8)
#decompose_unitary_search("CSX", 8)
#decompose_unitary_search("iSWAP", 8)
decompose_unitary_search("SiSWAP", 8)
#decompose_unitary_search("CRX", 2)
#decompose_unitary_search("CRY", 2)
#decompose_unitary_search("CRZ", 2)
#decompose_unitary_search("SYC", 2)
#decompose_unitary_search("CSWAP", 2)
#decompose_unitary_search("CU3", 1)
#decompose_unitary_search("U3", 1)
#decompose_unitary_search("GP", 1)
#decompose_unitary_search("P", 1)
#decompose_unitary_search("Deutsch", 1)
#decompose_unitary_search("Peres", 1)
#decompose_unitary_search("CCX", 1)
#decompose_unitary_search((1, [], lambda: compile_gates(1, [("Rx", -3*sympy.pi/2, [0]), ("Rz", -sympy.pi/4, [0]), ("H", [0]), ("X", [0]), ("Rz", -3*sympy.pi/2, [0])])), 8)
#gen_clifford()
