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
    num_samples = 2*(len(params)+1) + 1
    paramspace = list(itertools.product(*[np.linspace(0, np.pi*2*x[0], num=num_samples) for x in params]))
    for pos in paramspace:
        allU.append(np.array(Umtx.subs({x[1]: y for x, y in zip(params, pos)}).evalf()).astype(np.complex128))
    config = {'tree_level_max': 4, 'stop_first_solution': True, 'tolerance': 1e-10}
    optim = N_Qubit_Decomposition_Guided_Tree(allU, config, 0, None, paramspace=paramspace, paramscale=[x[0] for x in params])
    optim.set_Optimizer("BFGS2")
    optim.set_Verbose(0)
    optim.Start_Decomposition()
    cnot_structure = [(gate.get_Target_Qbit(), gate.get_Control_Qbit()) for gate in optim.get_Circuit().get_Gates() if gate.get_Name() == "CNOT"]
    print("CNOT structure:", cnot_structure)
    return cnot_structure

little_endian = True
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
def quantsimp(x):
    x = sympy.sympify(x)
    if isinstance(x, sympy.Expr) and x.is_zero:
        return sympy.S.Zero if not x.is_Number else x
    x = x.rewrite(sympy.exp)
    x = sympy.expand_power_exp(x)
    x = sympy.powdenest(x)
    #x = sympy.sqrtdenest(x)
    #x = sympy.nsimplify(x, rational=True)
    x = x.replace(lambda e: e.is_Float, lambda e: sympy.nsimplify(e, rational=True))
    x = x.replace(lambda z: z.is_constant(), lambda z: sympy.simplify(sympy.expand_complex(z)))
    x = sympy.powsimp(x)
    x = sympy.together(x)
    x = sympy.cancel(x)
    #x = sympy.powdenest(x)
    x = sympy.expand_mul(x)
    return x.doit()
def textbook_simp(x):
    x = quantsimp(x)
    x = sympy.factor_terms(x, radical=True)
    cs = [(sympy.Wild('c'+str(i), exclude=[z]), sympy.Wild('s'+str(i), exclude=list(x.free_symbols-{z}))) for i, z in enumerate(x.free_symbols)]
    for c, t in cs:
        x = x.replace(c*sympy.exp(sympy.I*t)+c*sympy.exp(-sympy.I*t), 2*c*sympy.cos(t)) #cosine definition
        x = x.replace(c*sympy.exp(sympy.I*t)-c*sympy.exp(-sympy.I*t), 2*c*sympy.I*sympy.sin(t)) #sine definition
    x = sympy.powsimp(x)
    x = sympy.factor_terms(x, radical=True)
    t = sympy.Wild('t', properties=[lambda k: k.is_Rational])
    x = x.replace(-(-1)**t, sympy.exp(sympy.I*sympy.pi*(t+1)))
    x = x.replace((-1)**t, sympy.exp(sympy.I*sympy.pi*t))
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
    if len(gates) and gates[0][1] == list(range(num_qubits)): #shortcut for global gates
        Umtx = gates[0][0]
        gates = gates[1:]
    else:
        Umtx = sympy.eye(2**num_qubits)
    for gate in gates:
        Umtx = apply_to(Umtx, num_qubits, *gate)
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
def make_inverse(g): return g**-1
def make_sqrt(g): return g**(1/2)
@functools.lru_cache(None)
def gen_I(): return sympy.eye(2)
@functools.lru_cache(None)
def gen_Rx(theta): return sympy.exp(-sympy.I*theta/2*gen_X()).applyfunc(textbook_simp)
@functools.lru_cache(None)
def gen_Ry(theta): return sympy.exp(-sympy.I*theta/2*gen_Y()).applyfunc(textbook_simp)
@functools.lru_cache(None)
def gen_Rz(phi): return sympy.exp(-sympy.I*phi/2*gen_Z()).applyfunc(textbook_simp)
@functools.lru_cache(None)
def gen_GP(theta, qbits): return sympy.exp(theta*sympy.I)*sympy.eye(1<<qbits)
@functools.lru_cache(None)
def gen_H(): return sympy.Matrix([[1, 1], [1, -1]])/sympy.sqrt(2)
@functools.lru_cache(None)
def gen_X(): return sympy.Matrix([[0, 1], [1, 0]])
@functools.lru_cache(None)
def gen_Y(): return sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])
@functools.lru_cache(None)
def gen_Z(): return sympy.Matrix([[1, 0], [0, -1]])
@functools.lru_cache(None)
def gen_S(): return sympy.Matrix([[1, 0], [0, sympy.I]])
@functools.lru_cache(None)
def gen_Sdg(): return make_inverse(gen_S()) #sympy.Matrix([[1, 0], [0, -sympy.I]])
@functools.lru_cache(None)
def gen_Sx(): return make_sqrt(gen_X()).applyfunc(textbook_simp)
@functools.lru_cache(None)
def gen_Sxdg(): return make_inverse(gen_Sx()).applyfunc(textbook_simp) #sympy.Matrix([[1, -sympy.I], [-sympy.I, 1]])/2

@functools.lru_cache(None)
def gen_HX(): return compile_gates(1, [(gen_H(), [0]), (gen_X(), [0])])
@functools.lru_cache(None)
def gen_HY(): return compile_gates(1, [(gen_H(), [0]), (gen_Y(), [0])])
@functools.lru_cache(None)
def gen_HZ(): return compile_gates(1, [(gen_H(), [0]), (gen_Z(), [0])])
@functools.lru_cache(None)
def gen_SX(): return compile_gates(1, [(gen_S(), [0]), (gen_H(), [0])])
@functools.lru_cache(None)
def gen_SY(): return compile_gates(1, [(gen_S(), [0]), (gen_Y(), [0])])
@functools.lru_cache(None)
def gen_HS(): return compile_gates(1, [(gen_H(), [0]), (gen_S(), [0])])
@functools.lru_cache(None)
def gen_HSX(): return compile_gates(1, [(gen_H(), [0]), (gen_S(), [0]), (gen_X(), [0])])
@functools.lru_cache(None)
def gen_HSY(): return compile_gates(1, [(gen_H(), [0]), (gen_S(), [0]), (gen_Y(), [0])])
@functools.lru_cache(None)
def gen_HSdg(): return compile_gates(1, [(gen_H(), [0]), (gen_Sdg(), [0])])
@functools.lru_cache(None)
def gen_SH(): return compile_gates(1, [(gen_S(), [0]), (gen_H(), [0])])
@functools.lru_cache(None)
def gen_SHX(): return compile_gates(1, [(gen_S(), [0]), (gen_H(), [0]), (gen_X(), [0])])
@functools.lru_cache(None)
def gen_SHY(): return compile_gates(1, [(gen_S(), [0]), (gen_H(), [0]), (gen_Y(), [0])])
@functools.lru_cache(None)
def gen_SHZ(): return compile_gates(1, [(gen_S(), [0]), (gen_H(), [0]), (gen_Z(), [0])])
@functools.lru_cache(None)
def gen_SxY(): return compile_gates(1, [(gen_Sx(), [0]), (gen_Y(), [0])])
@functools.lru_cache(None)
def gen_SxZ(): return compile_gates(1, [(gen_Sx(), [0]), (gen_Z(), [0])])

def gen_CZPowGate(t): return gen_CP(sympy.pi*t)
def gen_fSim(theta, phi): return compile_gates(2, [(gen_iSWAP_pow(-2*theta/sympy.pi), [0, 1]), (gen_CZPowGate(-phi/sympy.pi), [0, 1])])
def gen_SYC(): return gen_fSim(sympy.pi/2, sympy.pi/6)
@functools.lru_cache(None)
def gen_T(): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*sympy.pi/4)]])
@functools.lru_cache(None)
def gen_Tdg(): return make_inverse(gen_T())
def gen_P(theta): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*theta)]])
def gen_U1(theta): return gen_P(theta)
def gen_CH(): return sympy.Matrix(make_controlled(gen_H(), 1))
def gen_CX(): return sympy.Matrix(make_controlled(gen_X(), 1))
def gen_CNOT(): return gen_CX()
def gen_CY(): return sympy.Matrix(make_controlled(gen_Y(), 1))
def gen_CZ(): return sympy.Matrix(make_controlled(gen_Z(), 1))
def gen_U(theta, phi, lbda): return compile_gates(1, [(gen_Rz(lbda), [0]), (gen_Ry(theta), [0]), (gen_Rz(phi), [0]), (gen_GP((phi+lbda)/2, 1), [0])]).applyfunc(textbook_simp)
def gen_U2(phi, lbda): return gen_U(sympy.pi/2, phi, lbda)
def gen_U3(theta, phi, lbda): return gen_U(theta, phi, lbda)
def gen_R(theta, phi): return gen_U(theta, phi-sympy.pi/2, -phi+sympy.pi/2).applyfunc(textbook_simp)
def gen_CR(theta, phi): return make_controlled(gen_R(theta, phi), 1)
def gen_CROT(theta, phi): return make_controlled(gen_R(theta, phi), 1, gen_R(-theta, phi))
def gen_CRX(theta): return make_controlled(gen_Rx(theta), 1)
def gen_CRY(theta): return make_controlled(gen_Ry(theta), 1)
def gen_CRZ(theta): return make_controlled(gen_Rz(theta), 1)
def gen_CSX(): return make_controlled(gen_Sx(), 1)
def gen_CCX(): return make_controlled(gen_CNOT(), 2)
def gen_Toffoli(): return gen_CCX()
def gen_CCZ(): return make_controlled(gen_CZ(), 2)
def gen_SWAP(): return functools.reduce(operator.add, (compile_gates(2, [(gen(), [0]), (gen(), [1])]) for gen in (gen_I, gen_X, gen_Y, gen_Z))) / 2
def gen_CSWAP(): return make_controlled(gen_SWAP(), 2)
def gen_iSWAP(): return gen_Rxy(-sympy.pi)
def gen_iSWAP_pow(alpha): return (gen_iSWAP()**alpha).applyfunc(textbook_simp)
def gen_CP(phi): return make_controlled(gen_P(phi), 1)
#def gen_CR(phi): return gen_CP(phi)
def gen_CS(): return make_controlled(gen_S(), 1)
def gen_CU1(theta): return gen_CP(theta)
def gen_CU(theta, phi, lbda, gamma): return make_controlled(compile_gates(1, [(gen_U(theta, phi, lbda), [0]), (gen_GP(gamma, 1), [0])]), 1).applyfunc(textbook_simp)
def gen_CU3(theta, phi, lbda): return gen_CU(theta, phi, lbda, 0)
def gen_Rxx(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])])).applyfunc(textbook_simp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_X(), [0]), (gen_X(), [1])])
def gen_Ryy(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])])).applyfunc(textbook_simp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Y(), [0]), (gen_Y(), [1])])
def gen_Rzz(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Z(), [0]), (gen_Z(), [1])])).applyfunc(textbook_simp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Z(), [0]), (gen_Z(), [1])])
def gen_xx_plus_yy(phi, beta): return compile_gates(2, [(gen_Rz(beta), [0]), (gen_Rxy(phi), [0, 1]), (gen_Rz(-beta), [0])]).applyfunc(textbook_simp)
def gen_xx_minus_yy(phi, beta): return compile_gates(2, [(gen_Rz(beta), [0]), (gen_Rxmy(phi), [0, 1]), (gen_Rz(-beta), [0])]).applyfunc(textbook_simp)
def gen_Rxy(phi): return sympy.exp(-sympy.I*phi/4*(compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])]) + compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])]))).applyfunc(textbook_simp)
def gen_Rxmy(phi): return sympy.exp(-sympy.I*phi/4*(compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])]) - compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])]))).applyfunc(textbook_simp)
def gen_SSWAP(): return make_sqrt(gen_SWAP()).applyfunc(textbook_simp)
def gen_SiSWAP(): return make_sqrt(gen_iSWAP()).applyfunc(textbook_simp)
def gen_CZ_decomp(): return compile_gates(2, [(gen_H(), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [1])])
def gen_CY_decomp(): return compile_gates(2, [(gen_Sdg(), [1]), (gen_CNOT(), [0, 1]), (gen_S(), [1])])
def gen_CH_decomp(): return compile_gates(2, [(gen_Ry(sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-sympy.pi/4), [1])]).applyfunc(textbook_simp)
def gen_CP_decomp(phi): return compile_gates(2, [(gen_Rz(phi/2), [0]), (gen_CNOT(), [0, 1]), (gen_Rz(-phi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(phi/2), [1]), (gen_GP(phi/4, 2), [0, 1])])
def gen_CRX_decomp(theta): return compile_gates(2, [(gen_S(), [1]), (gen_Ry(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Sdg(), [1])]).applyfunc(textbook_simp)
def gen_CRY_decomp(theta): return compile_gates(2, [(gen_Ry(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1])]).applyfunc(textbook_simp)
def gen_CRZ_decomp(theta): return compile_gates(2, [(gen_Rz(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-theta/2), [1]), (gen_CNOT(), [0, 1])])
def gen_CSX_decomp(): return compile_gates(2, [(gen_Sdg(), [0]), (gen_H(), [1]), (gen_Y(), [1]), (gen_Tdg(), [0]), (gen_CNOT(), [0, 1]), (gen_Y(), [1]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1]), (gen_T(), [1]), (gen_H(), [1])]).applyfunc(textbook_simp)
#def gen_CS_decomp(): return compile_gates(2, [(gen_Rz(sympy.pi/4), [0]), (gen_CNOT(), [0, 1]), (gen_Rz(-sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(sympy.pi/4), [1]), (gen_GP(sympy.pi/8, 2), [0, 1])]).applyfunc(textbook_simp)
def gen_CS_decomp(): return compile_gates(2, [(gen_Sdg(), [0]), (gen_Y(), [1]), (gen_Tdg(), [0]), (gen_CNOT(), [0, 1]), (gen_Y(), [1]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1]), (gen_T(), [1])]).applyfunc(textbook_simp)
def gen_CR_decomp(theta, phi): return compile_gates(2, [(gen_Rz(-phi+sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(phi-sympy.pi/2), [1])]).applyfunc(textbook_simp)
def gen_CROT_decomp(theta, phi): return compile_gates(2, [(gen_Rz(-phi), [1]), (gen_Ry(sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-sympy.pi/2), [1]), (gen_Rz(phi), [1])]).applyfunc(textbook_simp)
def gen_Rxx_decomp(theta): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_Rx(theta), [0]), (gen_CNOT(), [0, 1])]).applyfunc(textbook_simp)
def gen_Ryy_decomp(theta): return compile_gates(2, [(gen_Rx(sympy.pi/2), [0]), (gen_Rx(sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1]), (gen_Rx(-sympy.pi/2), [0]), (gen_Rx(-sympy.pi/2), [1])]).applyfunc(textbook_simp)
def gen_Rzz_decomp(theta): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1])])
#def gen_Rxy_decomp(phi): return compile_gates(2, [(gen_Rxx_decomp(phi/2), [0,1]), (gen_Ryy_decomp(phi/2), [0,1])]).applyfunc(textbook_simp)
def gen_Rxy_decomp(phi): return compile_gates(2, [(gen_Sdg(), [1]), (gen_S(), [0]), (gen_Sx(), [1]), (gen_S(), [1]), (gen_CNOT(), [1, 0]), (gen_Ry(-phi/2), [0]), (gen_Ry(-phi/2), [1]), (gen_CNOT(), [1, 0]), (gen_Sdg(), [0]), (gen_Sdg(), [1]), (gen_Sxdg(), [1]), (gen_S(), [1])]).applyfunc(textbook_simp)
def gen_CZPowGate_decomp(t): return gen_CP_decomp(sympy.pi*t)
#def gen_iSWAP_pow_decomp(alpha): return compile_gates(2, [(gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(-(alpha*sympy.pi)/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_Sdg(), [0]), (gen_Sdg(), [1]), (gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(-(alpha*sympy.pi)/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_S(), [0]), (gen_S(), [1])]).applyfunc(textbook_simp)
def gen_iSWAP_pow_decomp(alpha): return gen_Rxy_decomp(-sympy.pi*alpha)
#print(gen_Rxy(sympy.Symbol("ϕ", real=True)))
#print(gen_Rxy_decomp(sympy.Symbol("ϕ", real=True)))
#print(gen_iSWAP_pow(sympy.Symbol("α", real=True)))
#print(gen_iSWAP_pow_decomp(sympy.Symbol("α", real=True)))
#print(gen_iSWAP_pow(0), gen_iSWAP_pow(1), gen_iSWAP_pow(2), gen_iSWAP_pow(3)); assert False
def gen_fSim_decomp(theta, phi): return compile_gates(2, [(gen_iSWAP_pow_decomp(-2*theta/sympy.pi), [0, 1]), (gen_CZPowGate_decomp(-phi/sympy.pi), [0, 1])])
#def gen_SYC_decomp(): return gen_fSim_decomp(sympy.pi/2, sympy.pi/6)
#3 CNOT SYC: https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00248-8
def gen_SYC_decomp(): return compile_gates(2, [(gen_Rz(-3*sympy.pi/4), [0]), (gen_Rz(sympy.pi/4), [1]), (gen_Sx(), [0]), (gen_Sx(), [1]), (gen_Rz(-sympy.pi), [0]), (gen_Rz(sympy.pi), [1]), (gen_Sx(), [1]), (gen_Rz(5*sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Sx(), [0]), (gen_Rz(-3*sympy.pi/4), [1]), (gen_Sx(), [1]), (gen_Rz(sympy.pi), [1]), (gen_Sx(), [1]), (gen_Rz(9*sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(sympy.pi/2), [0]), (gen_Sx(), [1]), (gen_Sx(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_Rz(11*sympy.pi/12), [0]), (gen_Sx(), [1]), (gen_Sx(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Sx(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_Rz(sympy.pi/6), [0]), (gen_Sx(), [1]), (gen_Rz(-sympy.pi/3), [1]), (gen_GP(17*sympy.pi/24, 2), [0, 1])]).applyfunc(textbook_simp)
def gen_CU_decomp(theta, phi, lbda, gamma): return compile_gates(2, [(gen_Rz((lbda-phi)/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-(phi+lbda)/2), [1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(phi), [1]), (gen_P((lbda+phi)/2+gamma), [0])]).applyfunc(textbook_simp)
def gen_CU3_decomp(theta, phi, lbda): return compile_gates(2, [(gen_Rz((lbda-phi)/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-(phi+lbda)/2), [1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(phi), [1]), (gen_P((lbda+phi)/2), [0])]).applyfunc(textbook_simp)
def gen_CCZ_decomp(): return compile_gates(3, [(gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1])])
def gen_CCX_decomp(): return compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1])])
def gen_SWAP_decomp(): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_CNOT(), [0, 1])])
#7 CNOT CSWAP: https://arxiv.org/pdf/2305.18128
def gen_CSWAP_decomp(): return compile_gates(3, [(gen_S(), [1]), (gen_CNOT(), [2, 1]), (gen_Sdg(), [1]), (gen_Sx(), [2]), (gen_T(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_T(), [1]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_CNOT(), [0, 1]), (gen_T(), [2]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [2, 1]), (gen_GP(-sympy.pi/4, 3), [0, 1, 2])]).applyfunc(textbook_simp)
#def gen_CSWAP_decomp(): return compile_gates(3, [(gen_CNOT(), [2, 1]), (gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [2, 1])])
def gen_iSWAP_decomp(): return compile_gates(2, [(gen_S(), [0]), (gen_S(), [1]), (gen_H(), [0]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_H(), [1])])
def gen_SSWAP_decomp(): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_Sdg(), [1]), (gen_T(), [1]), (gen_Rx(sympy.pi/4), [0]), (gen_H(), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [1]), (gen_Rx(-sympy.pi/4), [0]), (gen_Sdg(), [0]), (gen_CNOT(), [0, 1]), (gen_S(), [1])]).applyfunc(textbook_simp)
def gen_Rx_test(theta): return sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2)], [-sympy.I*sympy.sin(theta/2), sympy.cos(theta/2)]])
def gen_Ry_test(theta): return sympy.Matrix([[sympy.cos(theta/2), -sympy.sin(theta/2)], [sympy.sin(theta/2), sympy.cos(theta/2)]])
def gen_Rz_test(theta): return sympy.Matrix([[sympy.exp(-sympy.I*theta/2), 0], [0, sympy.exp(sympy.I*theta/2)]])
def gen_U_test(theta, phi, lbda): return sympy.Matrix([[sympy.cos(theta/2), -sympy.exp(sympy.I*lbda)*sympy.sin(theta/2)], [sympy.exp(sympy.I*phi)*sympy.sin(theta/2), sympy.exp(sympy.I*(phi+lbda))*sympy.cos(theta/2)]])
def gen_S_test(): return sympy.Matrix([[1, 0], [0, sympy.I]])
def gen_Sdg_test(): return sympy.Matrix([[1, 0], [0, -sympy.I]])
def gen_Sx_test(): return sympy.Matrix([[(1+sympy.I)/2, (1-sympy.I)/2], [(1-sympy.I)/2, (1+sympy.I)/2]]).applyfunc(textbook_simp)
def gen_Sxdg_test(): return sympy.Matrix([[(1-sympy.I)/2, (1+sympy.I)/2], [(1+sympy.I)/2, (1-sympy.I)/2]]).applyfunc(textbook_simp)
def gen_SYC_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 0, -sympy.I, 0], [0, -sympy.I, 0, 0], [0, 0, 0, sympy.exp(-sympy.I*sympy.pi/6)]])
def gen_CX_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
def gen_CZ_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
def gen_CY_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 0, 0, -sympy.I], [0, 0, 1, 0], [0, sympy.I, 0, 0]])
def gen_R_test(theta, phi): return sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.exp(-sympy.I*phi)*sympy.sin(theta/2)], [-sympy.I*sympy.exp(sympy.I*phi)*sympy.sin(theta/2), sympy.cos(theta/2)]])
def gen_Rxx_test(theta): return sympy.Matrix([[sympy.cos(theta/2), 0, 0, -sympy.I*sympy.sin(theta/2)], [0, sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2), 0], [0, -sympy.I*sympy.sin(theta/2), sympy.cos(theta/2), 0], [-sympy.I*sympy.sin(theta/2), 0, 0, sympy.cos(theta/2)]])
def gen_Ryy_test(theta): return sympy.Matrix([[sympy.cos(theta/2), 0, 0, sympy.I*sympy.sin(theta/2)], [0, sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2), 0], [0, -sympy.I*sympy.sin(theta/2), sympy.cos(theta/2), 0], [sympy.I*sympy.sin(theta/2), 0, 0, sympy.cos(theta/2)]])
def gen_Rzz_test(theta): return sympy.Matrix([[sympy.exp(-sympy.I*theta/2), 0, 0, 0], [0, sympy.exp(sympy.I*theta/2), 0, 0], [0, 0, sympy.exp(sympy.I*theta/2), 0], [0, 0, 0, sympy.exp(-sympy.I*theta/2)]])
def gen_Rxy_test(phi): return sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(phi/2), -sympy.I*sympy.sin(phi/2), 0], [0, -sympy.I*sympy.sin(phi/2), sympy.cos(phi/2), 0], [0, 0, 0, 1]])
def gen_CU_test(theta, phi, lbda, gamma): return sympy.Matrix([[1, 0, 0, 0], [0, sympy.exp(sympy.I*gamma)*sympy.cos(theta/2), 0, -sympy.exp(sympy.I*(gamma+lbda))*sympy.sin(theta/2)], [0, 0, 1, 0], [0, sympy.exp(sympy.I*(gamma+phi))*sympy.sin(theta/2), 0, sympy.exp(sympy.I*(gamma+phi+lbda))*sympy.cos(theta/2)]])
def gen_CU3_test(theta, phi, lbda): return sympy.Matrix([[1, 0, 0, 0], [0, sympy.cos(theta/2), 0, -sympy.exp(sympy.I*lbda)*sympy.sin(theta/2)], [0, 0, 1, 0], [0, sympy.exp(sympy.I*phi)*sympy.sin(theta/2), 0, sympy.exp(sympy.I*(phi+lbda))*sympy.cos(theta/2)]])
def test_decomp():
    theta, phi, lbda, gamma = sympy.Symbol("θ", real=True), sympy.Symbol("ϕ", real=True), sympy.Symbol("λ", real=True), sympy.Symbol("γ", real=True)
    alpha, beta, theta2 = sympy.Symbol("α", real=True), sympy.Symbol("β", real=True), sympy.Symbol("θ2", real=True)
    #print(compile_gates(2, [(gen_Rz(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-theta/2), [1]), (gen_CNOT(), [0, 1]), (make_inverse(gen_CRZ(theta)), [0, 1])]))
    #print(make_inverse(gen_CRZ(theta)), [0, 1])
    #print(compile_gates(2, [(make_inverse(gen_CRZ(theta)), [0, 1]), (gen_Rz(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-theta/2), [1]), (gen_CNOT(), [0, 1])]))
    #print(compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2])]))
    print(gen_Rxmy(theta))
    print(gen_xx_plus_yy(theta, beta))
    print(gen_xx_minus_yy(theta, beta))
    assert gen_U(theta, phi, lbda) == gen_U_test(theta, phi, lbda), (gen_U(theta, phi, lbda), gen_U_test(theta, phi, lbda))
    assert gen_Rx(theta) == gen_Rx_test(theta), (gen_Rx(theta), gen_Rx_test(theta))
    assert gen_Ry(theta) == gen_Ry_test(theta), (gen_Ry(theta), gen_Ry_test(theta))
    assert gen_Rz(theta) == gen_Rz_test(theta), (gen_Rz(theta), gen_Rz_test(theta))
    assert gen_S() == gen_S_test(), (gen_S(), gen_S_test())
    assert gen_Sdg() == gen_Sdg_test(), (gen_Sdg(), gen_Sdg_test())
    assert gen_Sx() == gen_Sx_test(), (gen_Sx(), gen_Sx_test())
    assert gen_Sxdg() == gen_Sxdg_test(), (gen_Sxdg(), gen_Sxdg_test())
    assert gen_SYC() == gen_SYC_test(), (gen_SYC(), gen_SYC_test())
    assert gen_CX() == gen_CX_test(), (gen_CX(), gen_CX_test())
    assert gen_CY() == gen_CY_test(), (gen_CY(), gen_CY_test())
    assert gen_CZ() == gen_CZ_test(), (gen_CZ(), gen_CZ_test())
    assert gen_R(theta, phi) == gen_R_test(theta, phi), (gen_R(theta, phi), gen_R_test(theta, phi))
    assert gen_Rxx(theta) == gen_Rxx_test(theta), (gen_Rxx(theta), gen_Rxx_test(theta))
    assert gen_Ryy(theta) == gen_Ryy_test(theta), (gen_Ryy(theta), gen_Ryy_test(theta))
    assert gen_Rzz(theta) == gen_Rzz_test(theta), (gen_Rzz(theta), gen_Rzz_test(theta))
    assert gen_Rxy(phi) == gen_Rxy_test(phi), (gen_Rxy(phi), gen_Rxy_test(phi))
    assert gen_CU(theta, phi, lbda, gamma) == gen_CU_test(theta, phi, lbda, gamma), (gen_CU(theta, phi, lbda, gamma), gen_CU_test(theta, phi, lbda, gamma))
    assert gen_CU3(theta, phi, lbda) == gen_CU3_test(theta, phi, lbda), (gen_CU3(theta, phi, lbda), gen_CU3_test(theta, phi, lbda))
    assert gen_CZ() == gen_CZ_decomp()
    assert gen_CY() == gen_CY_decomp()
    assert gen_CRX(theta) == gen_CRX_decomp(theta), (gen_CRX(theta), gen_CRX_decomp(theta))
    assert gen_CRY(theta) == gen_CRY_decomp(theta), (gen_CRY(theta), gen_CRY_decomp(theta))
    assert gen_CRZ(theta) == gen_CRZ_decomp(theta)
    assert gen_CSX() == gen_CSX_decomp(), (gen_CSX(), gen_CSX_decomp())
    assert gen_CCZ() == gen_CCZ_decomp()
    assert gen_CCX() == gen_CCX_decomp()
    assert gen_CP(phi) == gen_CP_decomp(phi)
    assert gen_CS() == gen_CS_decomp(), (gen_CS(), gen_CS_decomp())
    assert gen_CH() == gen_CH_decomp(), (gen_CH(), gen_CH_decomp())
    assert gen_CR(theta, phi) == gen_CR_decomp(theta, phi), (gen_CR(theta, phi), gen_CR_decomp(theta, phi))
    assert gen_Rxx(theta) == gen_Rxx_decomp(theta), (gen_Rxx(theta), gen_Rxx_decomp(theta))
    assert gen_Ryy(theta) == gen_Ryy_decomp(theta), (gen_Ryy(theta), gen_Ryy_decomp(theta))
    assert gen_Rzz(theta) == gen_Rzz_decomp(theta), (gen_Rzz(theta), gen_Rzz_decomp(theta))
    assert gen_Rxy(phi) == gen_Rxy_decomp(phi), (gen_Rxy(phi), gen_Rxy_decomp(phi)) #not correct!!!
    assert gen_CROT(theta, phi) == gen_CROT_decomp(theta, phi), (gen_CROT(theta, phi), gen_CROT_decomp(theta, phi))
    assert gen_CZPowGate(alpha) == gen_CZPowGate_decomp(alpha)
    assert gen_iSWAP_pow(alpha) == gen_iSWAP_pow_decomp(alpha), (gen_iSWAP_pow(alpha), gen_iSWAP_pow_decomp(alpha)) #not correct!!!
    assert gen_fSim(theta, phi) == gen_fSim_decomp(theta, phi), (gen_fSim(theta, phi), gen_fSim_decomp(theta, phi)) #not correct!!!
    assert gen_SSWAP() == gen_SSWAP_decomp(), (gen_SSWAP(), gen_SSWAP_decomp())
    assert gen_SYC() == gen_SYC_decomp(), (gen_SYC(), gen_SYC_decomp())
    assert gen_CU3(theta, phi, lbda) == gen_CU3_decomp(theta, phi, lbda), (gen_CU3(theta, phi, lbda), gen_CU3_decomp(theta, phi, lbda))
    assert gen_CU(theta, phi, lbda, gamma) == gen_CU_decomp(theta, phi, lbda, gamma), (gen_CU(theta, phi, lbda, gamma), gen_CU_decomp(theta, phi, lbda, gamma))
    assert gen_SWAP() == gen_SWAP_decomp()
    assert gen_CSWAP() == gen_CSWAP_decomp()
    assert gen_iSWAP() == gen_iSWAP_decomp()
    #reverse CNOT is H-CNOT-H on both qubits
    assert compile_gates(2, [(gen_CNOT(), [1, 0])]) == compile_gates(2, [(gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [0]), (gen_H(), [1]),])
    print(find_control_qubits(gen_U3(theta, phi, lbda), 1), find_control_qubits(gen_CRY(theta), 2), find_control_qubits(gen_CCX(), 3))
    for i in range(3): #this proves any single qubit chain removes all purity, and converts aligning control to target
        print(f"U3({i})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [(gen_U3(theta, phi, lbda), [i]), (gen_CRY(theta2), (0, 1))]), 3))
    for i in range(3):
        for j in range(3):
            if i == j : continue
            print(f"CRY({i}, {j})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [(gen_CRY(theta), [i, j]), (gen_CRY(theta2), (0, 1))]), 3))
#test_decomp(); assert False

QUBIT = int #np.int32

theta, phi, lbda, gamma = sympy.Symbol("θ", real=True), sympy.Symbol("ϕ", real=True), sympy.Symbol("λ", real=True), sympy.Symbol("γ", real=True)
alpha, beta = sympy.Symbol("α", real=True), sympy.Symbol("β", real=True)
gate_descs = { #(num_qubits, num_params, sympy_generator_function)
    "GP": (0, [(1, theta)], gen_GP, "add_GP"),
    "I": (1, [], gen_I, "add_I"),
    "H": (1, [], gen_H, "add_H"),
    "S": (1, [], gen_S, "add_S"),
    "Sdg": (1, [], gen_Sdg, "add_Sdg"),
    "T": (1, [], gen_T, "add_T"),
    "Tdg": (1, [], gen_Tdg, "add_Tdg"),
    "Sx": (1, [], gen_Sx, "add_SX"),
    "Sxdg": (1, [], gen_Sxdg, "add_SXdg"),
    "X": (1, [], gen_X, "add_X"),
    "Y": (1, [], gen_Y, "add_Y"),
    "Z": (1, [], gen_Z, "add_Z"),

    "HX": (1, [], gen_HX, "add_HX"),
    "HY": (1, [], gen_HY, "add_HY"),
    "HZ": (1, [], gen_HZ, "add_HZ"),
    "SX": (1, [], gen_SX, "add_SX"),
    "SY": (1, [], gen_SY, "add_SY"),
    "HS": (1, [], gen_HS, "add_HS"),
    "HSX": (1, [], gen_HSX, "add_HSX"),
    "HSY": (1, [], gen_HSY, "add_HSY"),
    "HSdg": (1, [], gen_HSdg, "add_HSdg"),
    "SH": (1, [], gen_SH, "add_SH"),
    "SHX": (1, [], gen_SHX, "add_SHX"),
    "SHY": (1, [], gen_SHY, "add_SHY"),
    "SHZ": (1, [], gen_SHZ, "add_SHZ"),
    "SxY": (1, [], gen_SxY, "add_SxY"),
    "SxZ": (1, [], gen_SxZ, "add_SxZ"),

    "Rx": (1, [(2, theta)], gen_Rx, "add_Rx"),
    "Ry": (1, [(2, theta)], gen_Ry, "add_Ry"),
    "Rz": (1, [(2, phi)], gen_Rz, "add_Rz"),
    "U1": (1, [(1, theta)], gen_U1, "add_U1"),
    "P": (1, [(1, theta)], gen_P, "add_P"),
    "U2": (1, [(1, phi), (1, lbda)], gen_U2, "add_U2"),
    "U3": (1, [(2, theta), (1, phi), (1, lbda)], gen_U3, "add_U3"),
    "CNOT": (2, [], gen_CNOT, "add_CNOT"),
    "CY": (2, [], gen_CY, "add_CY"),
    "CZ": (2, [], gen_CZ, "add_CZ"),
    "CRX": (2, [(2, theta)], gen_CRX, "add_CRX"),
    "CRY": (2, [(2, theta)], gen_CRY, "add_CRY"),
    "CRZ": (2, [(2, theta)], gen_CRZ, "add_CRZ"),
    "CSX": (2, [], gen_CSX, "add_CSX"),
    "CS": (2, [], gen_CS, "add_CS"),
    "CH": (2, [], gen_CH, "add_CH"),
    "CP": (2, [(1, phi)], gen_CP, "add_CP"),
    "Rxx": (2, [(2, theta)], gen_Rxx, "add_Rxx"),
    "Ryy": (2, [(2, theta)], gen_Ryy, "add_Ryy"),
    "Rzz": (2, [(2, theta)], gen_Rzz, "add_Rzz"),
    "Rxy": (2, [(2, phi)], gen_Rxy, "add_Rxy"),
    "Rxmy": (2, [(2, phi)], gen_Rxmy, "add_Rxmy"),
    "xx_plus_yy": (2, [(2, phi), (1, beta)], gen_xx_plus_yy, "add_xx_plus_yy"),
    "xx_minus_yy": (2, [(2, phi), (1, beta)], gen_xx_minus_yy, "add_xx_minus_yy"),
    "iSWAP_pow": (2, [(2/sympy.pi, alpha)], gen_iSWAP_pow, "add_iSWAP_pow"),
    "SYC": (2, [], gen_SYC, "add_SYC"),
    "CCZ": (3, [], gen_CCZ, "add_CCZ"),
    "CCX": (3, [], gen_CCX, "add_CCX"),
    "CSWAP": (3, [], gen_CSWAP, "add_CSWAP"),
    "SSWAP": (2, [], gen_SSWAP, "add_SSWAP"),
    "SiSWAP": (2, [], gen_SiSWAP, "add_SiSWAP"),
}
def sympy_to_gp(gates):
    pass
def gp_to_sympy(individual):
    pass
class ParamIndex:
    def __init__(self, index):
        self.index = index
    def __repr__(self): return f"ParamIndex({self.index})"
class AngleScale:
    def __init__(self, scale):
        self.scale = scale
    def __repr__(self): return f"AngleScale_{'m' if self.scale < 0 else ''}{abs(self.scale)}"
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
        return sum([(2*sympy.pi if index.index == -1 else params[index.index]) / scale.scale for index, scale in self.params])
    def to_qiskit(self, params):
        return sum([(2*np.pi if index.index == -1 else params[index.index]) / scale.scale for index, scale in self.params])
    def __lt__(self, other): return len(self.params) < len(other.params)
    def __repr__(self): return f"ParamIndexSum({self.params})"
    def __str__(self):
        return " + ".join([f"2π/{scale.scale}" if index.index == -1 else f"ARG{index.index}/{scale.scale}" for index, scale in self.params])
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
            nqbits, nparams, gen_func, _ = gate_descs[gate_name]
            if nqbits == 0:
                gate_ops.append( (gen_func(*[param.to_sympy(params) for param in gate_params], len(qubits)), qubits) )
            else: gate_ops.append( (gen_func(*[param.to_sympy(params) for param in gate_params]), qubits) )
        return gate_ops
    def to_squander(self, params, num_qubits):
        circ = Circuit(num_qubits)
        params = []
        for gate in self.gates:
            gate_name, qubits, gate_params = gate
            nqbits, nparams, _, gen_func = gate_descs[gate_name]
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

    ops = sympy.count_ops(expr, visual=False)
    depth = expr_tree_depth(expr)
    size = expr_tree_size(expr)
    leaves = expr_leaf_count(expr)

    return 1.0 + (weight_ops * ops +
            weight_depth * depth +
            weight_size * size +
            weight_leaves * leaves)
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
    param_syms = [x[1] for x in params]
    spcirc = compiled_circuit.to_sympy(param_syms, num_qubits)
    rescirc = compile_gates(num_qubits, [(unitary, list(range(num_qubits)))] + spcirc)
    return num_qubits, param_syms, rescirc
def eval_symcost(num_qubits, rescirc, allow_global_phase=False):
    if allow_global_phase:
        if all(rescirc[i,j] == (rescirc[0,0] if i==j else 0) for i in range(1<<num_qubits) for j in range(1<<num_qubits)):
            return 0.0
    rescirc = rescirc - sympy.eye(1<<num_qubits)
    rescirc = sympy.simplify(rescirc)
    #print(rescirc)
    symcost = sum(expr_structural_cost(expr) for expr in rescirc) / len(rescirc)
    return symcost
def eval_circ(unitary, params, compiled_circuit, symonly=False, allow_global_phase=False):
    num_qubits, param_syms, rescirc = get_eval_circ(unitary, params, compiled_circuit)
    rescirc = dephase_matrix_by_first_clean_exp(rescirc)[0]    
    if not symonly:
        tr = sympy.Trace(rescirc).rewrite(sympy.Sum)
        cost = 0.0
        num_samples = 2*(len(params)+1) + 1
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
    symcost, cost, rescirc = eval_circ(unitary, params, compiled_circuit)
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

def gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=False, gp_gate=False):
    param_info = gate_descs[gate][1]
    Umtx = gate_descs[gate][2](*[x[1] for x in param_info]).conjugate().transpose()
    t = sympy.Wild('t')
    Umtx = Umtx.applyfunc(lambda x: x.replace(sympy.conjugate(sympy.I**t), (-sympy.I)**t).replace(sympy.conjugate((-sympy.I)**t), sympy.I**t))
    print(gate_descs[gate][2](*[x[1] for x in param_info]), Umtx)
    num_qubits = Umtx.shape[0].bit_length() - 1
    assert 'CNOT' in basis
    print(gate, "num_qubits:", num_qubits, "scale_max:", scale_max, "num_params:", len(param_info), "layers:", layers)
    if gate in ("Rxx", "Ryy", "Rzz"): cnot_structure = [(0, 1), (0, 1)] #falsely shown as needing 0 CNOTs with OSR?
    elif gate in ("CH", "CZ"): cnot_structure = [(0, 1)]
    elif gate in ("Rxy", "SiSWAP", "CRX", "CRY", "CRZ", "CP", "CS", "CSX"): cnot_structure = [(0, 1), (0, 1)]
    elif gate in ("SYC", "SSWAP",): cnot_structure = [(0, 1), (0, 1), (0, 1)]
    elif gate in ("CCZ", "CCX"): cnot_structure = [(1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1)]
    elif gate in ("CSWAP",): cnot_structure = [(2, 1), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1), (2, 1)]
    elif num_qubits > 1: cnot_structure = determine_CNOT_structure(Umtx, param_info)
    else: cnot_structure = []
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
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(gate, scale_max, layers, basis, u3_ansatz=True)

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
        
        ps.addTerminal(ParamIndexSum().add_param((ParamIndex(-1), AngleScale(1))), ANGLE, name="twopi_angle")
        ps.addTerminal(ParamIndex(-1), ParamIndex, name="twopi")
    
        # Avoid generating zero scale to prevent division by zero in angle construction
        for i in range(-scale_max, scale_max + 1):
            if i == 0: continue
            ps.addTerminal(AngleScale(i), AngleScale, name=f"AngleScale_{'m' if i < 0 else ''}{abs(i)}")

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
def best_first(starts, is_goal, neighbors, eval_E, canonicalize):
    import heapq
    p = multiprocessing.Pool()
    # starts: iterable of nodes (can be just [start_node])
    pq = []
    best_E_by_key = {}   # dominance pruning per canonical state

    for s in starts:
        E, tb, k = eval_E(s)
        k = sum(sympy.Abs(k))
        best_E_by_key[k] = E
        heapq.heappush(pq, (E, tb, s))
    while pq:
        E, _, node = heapq.heappop(pq)

        if is_goal(node, E):
            return node

        if any(x in best_E_by_key and best_E_by_key[x] <= E for x in canonicalize(node)):
            continue

        print(E, node, eval_E(node))

        nbrs = neighbors(node)
        
        for nb, (Enb, tb, knb) in zip(nbrs, p.map(eval_E, nbrs)):
        #for nb in nbrs:
        #    Enb, tb, knb = eval_E(nb)
            knb = sum(sympy.Abs(knb))

            if knb in best_E_by_key and Enb >= best_E_by_key[knb]:
                continue

            best_E_by_key[knb] = Enb
            heapq.heappush(pq, (Enb, tb, nb))

    return None
def make_circ(ansatz, node):
    compiled_circuit = CircuitBuilder()
    count = 0
    for part in ansatz:
        if part is not None:
            angles = node[count][:part[0]]
            #if len(param_info) > 0: angles = [angle(*[ParamIndex(i) for i in range(len(param_info))]) for angle in angles]
            count += part[0]
            compiled_circuit.add_circuit(part[1](*angles))
        else:
            cc = CircuitBuilder()
            for x in node[count]:
                cc.add_circuit(gen_gate(*x))
            #if len(param_info) > 0: cc = cc(*[ParamIndex(i) for i in range(len(param_info))])
            compiled_circuit.add_circuit(cc)
            count += 1
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(-2)), 1)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(2)), 1)) #CP
    #compiled_circuit = CircuitBuilder().add_circuit(gen_gate("Rz", make_angle(ParamIndex(0), AngleScale(2)), 0)).add_circuit(gen_gate("CNOT", 0, 1)).add_circuit(gen_gate("CNOT", 0, 1)) #CP
    return compiled_circuit

#@lru_cache(None)
def eval(Umtx, param_info, ansatz, allow_global_phase, node):
    compiled_circuit = make_circ(ansatz, node)
    symcost, cost, U = eval_circ(Umtx, param_info, compiled_circuit, True, allow_global_phase=allow_global_phase)
    #print(compiled_circuit, symcost, cost, U)
    if symcost == 0.0: print("Found exact solution:", node, cost, symcost)
    return symcost, cost, sympy.ImmutableDenseMatrix(U)
def gen_clifford():
    all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('SX',)), (('I',), ('X',), ('Y',), ('Z',)))]
    #all_clifford = [x+y for x, y in itertools.product((('I',), ('H',), ('S',), ('H', 'S'), ('S', 'H'), ('H', 'S', 'H')), (('I',), ('H', 'S', 'S', 'H'), ('H', 'S', 'S', 'H', 'S', 'S'), ('S', 'S')))]
    clifford_gates = {}
    for cliff in all_clifford:
        cb = CircuitBuilder()
        for gate in (gen_gate(x, 0) for x in cliff):
            cb.add_circuit(gate)
        mat = sympy.ImmutableMatrix(compile_gates(1, cb.to_sympy([], 1)))
        if not any(sympy.expand(mat*phase) in clifford_gates for phase in (1, -1, sympy.I, -sympy.I)):
            clifford_gates[mat] = cliff
    for gate in clifford_gates:
        #print(gate, clifford_gates[gate])
        for phase in (1, -1, sympy.I, -sympy.I):
            rev = sympy.expand(make_inverse(gate)*phase)
            if rev in clifford_gates:
                print(gate, phase, clifford_gates[gate], clifford_gates[rev])
#not maximally entangling: CS, CT
#maximally entangling: CNOT, CZ, CY, CH, iSWAP, SiSWAP (partial), XX(pi/4), exp(i pi/4 (X⊗X + Y⊗Y)), SSWAP (partial)
#maximally entangling only at specific angles: CRX, CRY, CRZ, CP, Rxy, iSWAP_pow
def decompose_unitary_search(gate, scale_max, layers=4, basis=('CNOT', 'H', 'S', 'Sdg', 'Sx', 'Sxdg', 'X', 'Y', 'Z', "HX", "HY", "HZ", "SX", "SY", "HS", "HSX", "HSY", "HSdg", "SH", "SHX", "SHY", "SHZ", "SxY", "SxZ", 'T', 'Tdg', 'Rx', 'Ry', 'Rz'), allow_global_phase=False):
    ansatz, regions, num_qubits, Umtx, param_info = gen_ansatz(gate, scale_max, layers, basis)
    num_regions = len(regions)
    all_angles = [
        make_angle(ParamIndex(i), AngleScale(scale)) for i in range(-1, len(param_info)) for scale in range(-scale_max, scale_max+1) if scale != 0
    ]
    #comp_angles = {make_angle(ParamIndex(i), AngleScale(scale)): make_angle(ParamIndex(i), AngleScale(-scale)) for i in range(-1, len(param_info)) for scale in range(-scale_max, scale_max+1) if scale != 0}
    all_gates = [tuple(((gate, *angles, qbit),) if i==region else () for i in range(num_regions))
                 for gate in basis if gate_descs[gate][0] == 1
                 for region in range(num_regions) if regions[region]
                 for qbit in range(num_qubits)
                 for angles in itertools.product(*[all_angles for _ in gate_descs[gate][1]])
             ]
    identity_pairs = ('T', 'Tdg'), ('Tdg', 'T')
    identity_clifford_pairs = ('I', 'I'), ('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('H', 'H'), ('HX', 'HZ'), ('HY', 'HY'), ('HZ', 'HX'), ('S', 'Sdg'), ('SX', 'SX'), ('SY', 'SY'), ('Sdg', 'S'), ('HS', 'SHX'), ('HSX', 'SHZ'), ('HSY', 'SHY'), ('HSdg', 'SH'), ('SH', 'HSdg'), ('SHX', 'HS'), ('SHY', 'HSY'), ('SHZ', 'HSX'), ('Sx', 'Sxdg'), ('Sxdg', 'Sx'), ('SxY', 'SxY'), ('SxZ', 'SxZ')
    identity_angle_pairs = ('Rx', 'Rx'), ('Ry', 'Ry'), ('Rz', 'Rz')
    if num_qubits > 1:
        rev_angles = [
            make_angle(ParamIndex(i), AngleScale(scale)) for i in range(-1, len(param_info)) for scale in range(scale_max, -scale_max-1, -1) if scale != 0
        ]
        all_gates.extend([tuple(((gate, *angles, qbit1),) if i==region1 else ((gate, *angles, qbit2),) if i==region2 else () for i in range(num_regions))
                        for gate in identity_pairs+identity_clifford_pairs+identity_angle_pairs if gate in basis
                        for region1 in range(num_regions) if regions[region1]
                        #for region2 in (next(iter(region2 for region2 in range(region1+1, num_regions) if regions[region2]), None),) if region2 is not None
                        for region2 in range(region1+1, num_regions) if regions[region2]
                        for qbit1 in range(num_qubits) for qbit2 in range(qbit1+1, num_qubits)
                        for angles in itertools.product(*[all_angles for _ in gate_descs[gate][1]])
                ])
        all_gates.extend([tuple(((gate1, *[x[0] for x in angles], qbit),) if i==region1 else ((gate2, *[x[1] for x in angles], qbit),) if i==region2 else () for i in range(num_regions))
                        for gate1, gate2 in identity_pairs+identity_clifford_pairs+identity_angle_pairs if gate1 in basis and gate2 in basis
                        for region1 in range(num_regions) if regions[region1]
                        #for region2 in (next(iter(region2 for region2 in range(region1+1, num_regions) if regions[region2]), None),) if region2 is not None
                        for region2 in range(region1+1, num_regions) if regions[region2]
                        for qbit in range(num_qubits)
                        for angles in itertools.product(*[list(zip(all_angles, rev_angles)) for _ in gate_descs[gate1][1]])
                ])
    #print(all_gates)

    #startcost = eval_circ(Umtx, param_info, CircuitBuilder())[0]
    starts = list(itertools.product(*[(((),) if region else ((x,) for x in all_angles)) for region in regions]))
    #print(starts)
    #res = A_star(starts, lambda node: eval(node) == 0, lambda node: eval(node),
    #       lambda node: [tuple(x+((gate[1:],) if gate[0] == i else ()) for i, x in enumerate(node)) for gate in gates],
    #       lambda node1, node2: abs(eval(node2) - eval(node1)))
    def get_neighbors(node):
        return [tuple(x[:i]+y+x[i:] for x, y, i in zip(node, gates, pos)) for gates in all_gates
                for pos in itertools.product(*[range((len(x) if len(gates[i]) > 0 else 0)+1) for i, x in enumerate(node)])]
    basic_pauli = ('I', 'X', 'Y', 'Z')
    full_clifford = ('I', 'H', 'S', 'Sdg', 'Sx', 'Sxdg', 'X', 'Y', 'Z', "HX", "HY", "HZ", "SX", "SY", "HS", "HSX", "HSY", "HSdg", "SH", "SHX", "SHY", "SHZ", "SxY", "SxZ")
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

    params = [x[1] for x in param_info]
    def canonicalize(node):
        return
        _, _, U = eval(Umtx, param_info, ansatz, allow_global_phase, node)  # populate cache
        preUs = [compile_gates(num_qubits, pauli_dressing_pre.to_sympy(params, num_qubits) + [(U, list(range(num_qubits)))]) for pauli_dressing_pre in pauli_dressings]
        yield from (mat for pauli_dressing in pauli_dressings for preU in preUs for mat in (sympy.ImmutableMatrix(eval_circ(preU, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)
        #yield from (mat for pauli_dressing in pauli_dressings for mat in (sympy.ImmutableMatrix(eval_circ(U, param_info, pauli_dressing, True, allow_global_phase=allow_global_phase)[2]),) if mat != U)

    res = best_first(starts, lambda node, E: E == 0, get_neighbors,
                     functools.partial(eval, Umtx, param_info, ansatz, allow_global_phase), canonicalize)
    def circ_to_code(circ):
        return ", ".join(f"(gen_{gate}({', '.join(str(x.to_sympy(params)) for x in angles)}), ({', '.join(str(q) for q in qubits)},))" for gate, qubits, angles in circ.gates)
    def circ_to_qiskit(circ):
        from qiskit.circuit import QuantumCircuit, Parameter
        qc = QuantumCircuit(num_qubits)
        namedict = {"CNOT": "cx", "H": "h", "S": "s", "Sdg": "sdg", "Sx": "sx", "Sxdg": "sxdg", "T": "t", "Tdg": "tdg", "X": "x", "Y": "y", "Z": "z", "Rx": "rx", "Ry": "ry", "Rz": "rz",
                    "HX": ("h", "x"), "HY": ("h", "y"), "HZ": ("h", "z"), "SX": ("s", "x"), "SY": ("s", "y"), "HS": ("h", "s"), "HSX": ("h", "s", "x"), "HSY": ("h", "s", "y"), 
                    "HSdg": ("h", "sdg"), "SH": ("s", "h"), "SHX": ("s", "h", "x"), "SHY": ("s", "h", "y"), "SHZ": ("s", "h", "z"), "SxY": ("sx", "y"), "SxZ": ("sx", "z")
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
    print(circ_to_code(make_circ(ansatz, res)))
    print(circ_to_qiskit(make_circ(ansatz, res)).draw())

    return res

#decompose_unitary("CS", 16, 2)
#decompose_unitary("SYC", 48, 2)
#decompose_unitary("CCZ", 2, 2)
#decompose_unitary("Rxy", 8, 2)
#decompose_unitary("SSWAP", 1, 2)
#decompose_unitary("SiSWAP", 1, 2)
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
decompose_unitary_search("Rxy", 8)
#decompose_unitary_search("SSWAP", 8)
#decompose_unitary_search("CSX", 8)
#decompose_unitary_search("SiSWAP", 8)
#decompose_unitary_search("CRX", 2)
#decompose_unitary_search("CRY", 2)
#decompose_unitary_search("CRZ", 2)
#decompose_unitary_search("SYC", 2)
#decompose_unitary_search("CSWAP", 2)
#decompose_unitary_search("U3", 8)
