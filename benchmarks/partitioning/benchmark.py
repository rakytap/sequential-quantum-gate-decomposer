import itertools
from unittest import result
from squander.partitioning.partition import PartitionCircuitQasm
from qiskit import QuantumCircuit
from squander.partitioning.tools import qiskit_to_squander_name, get_qubits, total_float_ops
from squander import utils

import timeit
import glob
import os

USE_ILP = True

MAX_GATES_ALLOWED = 1024**2 

METHOD_NAMES = [
    "kahn", 
    "tdag",
    "gtqcp",
    "qiskit",
    "qiskit-fusion",
    "bqskit-Quick",
    # "bqskit-Greedy", 
    # "bqskit-Scan",
    # "bqskit-Cluster", 
] + (["ilp", "ilp-fusion", #"ilp-fusion-ca"
      ] if USE_ILP else [])

from squander.gates import gates_Wrapper as gate
SUPPORTED_GATES = {x for n in dir(gate) for x in (getattr(gate, n),) if not n.startswith("_") and issubclass(x, gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES = {n for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}

def purity_analysis():
    """
    Run a small symbolic experiment to study “purity” and “sparsity” of control sets.

    This routine builds a set of helper functions and 1q/2q/3q gate factories with SymPy,
    then:
      1) Checks which qubit subsets act as pure/sparse controls for some canonical gates
         (e.g., U3 on 1q, CRY on 2q, CCX on 3q).
      2) Composes short circuits (e.g., U3 on qubit i followed by CRY(0,1)) and reports
         which control subsets remain pure/sparse after composition.
      3) Prints the identified “pure” and “sparsity” control sets for each experiment.

    Notes:
      - Endianness is governed by the local variable `little_endian` (default: True).
      - `apply_to` operates on full operators (2^n × 2^n), not statevectors.
      - Requires SymPy. The helper `apply_to` also uses `itertools.product`.

    Args:
        None

    Returns:
        None
            Results are printed to stdout; the function is intended as a diagnostic/demo.
    """
    import sympy, functools, operator
    from sympy.combinatorics import Permutation
    theta, phi, lbda, gamma = sympy.Symbol("θ"), sympy.Symbol("ϕ"), sympy.Symbol("λ"), sympy.Symbol("γ")
    alpha, theta2 = sympy.Symbol("α"), sympy.Symbol("θ2")
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
    def force_euler_pairs(e):
        # replace 1 - exp(iθ) and exp(iθ) - 1 with ±2i exp(iθ/2) sin(θ/2)
        a = sympy.Wild('a', properties=[lambda k: k.is_real is not False])
        e = e.replace(1 - sympy.exp(sympy.I*a), -2*sympy.I*sympy.exp(sympy.I*a/2)*sympy.sin(a/2))
        e = e.replace(sympy.exp(sympy.I*a) - 1,  2*sympy.I*sympy.exp(sympy.I*a/2)*sympy.sin(a/2))
        return e
    def quantsimp(x, rewrite=False):
        x = force_euler_pairs(sympy.sympify(x).rewrite(sympy.exp) if rewrite else sympy.sympify(x))
        return sympy.simplify(sympy.trigsimp(sympy.exptrigsimp(sympy.together(sympy.expand_power_exp(sympy.powsimp(sympy.nsimplify(x, rational=True), force=True, deep=True)))), recursive=True)).doit()
    def quantsimprw(x): return quantsimp(x, rewrite=True)
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
        Umtx = sympy.eye(2**num_qubits)
        for gate in gates: Umtx = apply_to(Umtx, num_qubits, *gate)
        return Umtx
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
    def make_inverse(g): return g**-1
    def make_sqrt(g): return g**(1/2)
    def gen_I(): return sympy.eye(2)
    def gen_Rx(theta): return sympy.exp(-sympy.I*theta/2*gen_X()).applyfunc(quantsimp)
    def gen_Ry(theta): return sympy.exp(-sympy.I*theta/2*gen_Y()).applyfunc(quantsimp)
    def gen_Rz(phi): return sympy.exp(-sympy.I*phi/2*gen_Z()).applyfunc(quantsimp)
    def gen_GP(theta, qbits): return sympy.exp(theta*sympy.I)*sympy.eye(1<<qbits)
    def gen_H(): return sympy.Matrix([[1, 1], [1, -1]])/sympy.sqrt(2)
    def gen_X(): return sympy.Matrix([[0, 1], [1, 0]])
    def gen_Y(): return sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])
    def gen_Z(): return sympy.Matrix([[1, 0], [0, -1]])
    def gen_S(): return sympy.Matrix([[1, 0], [0, sympy.I]])
    def gen_Sdg(): return make_inverse(gen_S()) #sympy.Matrix([[1, 0], [0, -sympy.I]])
    def gen_SX(): return make_sqrt(gen_X()).applyfunc(quantsimp)
    def gen_CZPowGate(t): return gen_CP(sympy.pi*t)
    def gen_fSim(theta, phi): return compile_gates(2, [(gen_iSWAP_pow(-2*theta/sympy.pi), [0, 1]), (gen_CZPowGate(-phi/sympy.pi), [0, 1])])
    def gen_SYC(): return gen_fSim(sympy.pi/2, sympy.pi/6)
    def gen_T(): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*sympy.pi/4)]])
    def gen_Tdg(): return make_inverse(gen_T())
    def gen_P(theta): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*theta)]])
    def gen_U1(theta): return gen_P(theta)
    def gen_CH(): return sympy.Matrix(make_controlled(gen_H(), 1))
    def gen_CX(): return sympy.Matrix(make_controlled(gen_X(), 1))
    def gen_CNOT(): return gen_CX()
    def gen_CY(): return sympy.Matrix(make_controlled(gen_Y(), 1))
    def gen_CZ(): return sympy.Matrix(make_controlled(gen_Z(), 1))
    def gen_U(theta, phi, lbda): return compile_gates(1, [(gen_Rz(phi), [0]), (gen_Ry(theta), [0]), (gen_Rz(lbda), [0]), (gen_GP((phi+lbda)/2, 1), [0])])
    def gen_U2(phi, lbda): return gen_U(sympy.pi/2, phi, lbda)
    def gen_U3(theta, phi, lbda): return gen_U(theta, phi, lbda)
    def gen_R(theta, phi): return gen_U(theta, phi-sympy.pi/2, -phi+sympy.pi/2)
    def gen_CR(theta, phi): return make_controlled(gen_R(theta, phi), 1)
    def gen_CROT(theta, phi): return make_controlled(gen_R(theta, phi), 1, gen_R(-theta, phi))
    def gen_CRX(theta): return make_controlled(gen_Rx(theta), 1)
    def gen_CRY(theta): return make_controlled(gen_Ry(theta), 1)
    def gen_CRZ(theta): return make_controlled(gen_Rz(theta), 1)
    def gen_CCX(): return make_controlled(gen_CNOT(), 2)
    def gen_Toffoli(): return gen_CCX()
    def gen_CCZ(): return make_controlled(gen_CZ(), 2)
    def gen_SWAP(): return functools.reduce(operator.add, (compile_gates(2, [(gen(), [0]), (gen(), [1])]) for gen in (gen_I, gen_X, gen_Y, gen_Z))) / 2
    def gen_CSWAP(): return make_controlled(gen_SWAP(), 2)
    def gen_iSWAP(): return gen_Rxy(-sympy.pi)
    def gen_iSWAP_pow(alpha): return (gen_iSWAP()**alpha).applyfunc(quantsimprw)#.rewrite(sympy.exp)
    def gen_CP(phi): return make_controlled(gen_P(phi), 1)
    #def gen_CR(phi): return gen_CP(phi)
    def gen_CS(): return make_controlled(gen_S(), 1)
    def gen_CU1(theta): return gen_CP(theta)
    def gen_CU(theta, phi, lbda, gamma): return make_controlled(compile_gates(1, [(gen_U(theta, phi, lbda), [0]), (gen_GP(gamma, 1), [0])]), 1)
    def gen_CU3(theta, phi, lbda): return gen_CU(theta, phi, lbda, 0)
    def gen_Rxx(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])])).applyfunc(quantsimp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_X(), [0]), (gen_X(), [1])])
    def gen_Ryy(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])])).applyfunc(quantsimp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Y(), [0]), (gen_Y(), [1])])
    def gen_Rzz(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Z(), [0]), (gen_Z(), [1])])).applyfunc(quantsimp) #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Z(), [0]), (gen_Z(), [1])])
    def gen_Rxy(phi): return sympy.exp(-sympy.I*phi/4*(compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])]) + compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])]))).applyfunc(quantsimprw)
    def gen_CZ_decomp(): return compile_gates(2, [(gen_H(), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [1])])
    def gen_CY_decomp(): return compile_gates(2, [(gen_H(), [1]), (gen_S(), [1]), (gen_CNOT(), [0, 1]), (gen_Sdg(), [1]), (gen_H(), [1])])
    def gen_CH_decomp(): return compile_gates(2, [(gen_Rz(-sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-sympy.pi/2), [1]), (gen_Ry(sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-sympy.pi/4), [1]), (gen_Rz(sympy.pi), [1]), (gen_P(sympy.pi/2), [0])])
    def gen_CP_decomp(phi): return compile_gates(2, [(gen_Rz(phi/2), [0]), (gen_CNOT(), [0, 1]), (gen_Rz(-phi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(phi/2), [1]), (gen_GP(phi/4, 2), [0, 1])])# @ 
    def gen_CRX_decomp(theta): return compile_gates(2, [(gen_H(), [1]), (gen_Rz(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [1])])
    def gen_CRY_decomp(theta): return compile_gates(2, [(gen_Ry(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1])])
    def gen_CRZ_decomp(theta): return compile_gates(2, [(gen_Rz(theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-theta/2), [1]), (gen_CNOT(), [0, 1])])
    def gen_CS_decomp(): return compile_gates(2, [(gen_Rz(sympy.pi/4), [0]), (gen_CNOT(), [0, 1]), (gen_Rz(-sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(sympy.pi/4), [1]), (gen_GP(sympy.pi/8, 2), [0, 1])])
    def gen_CR_decomp(theta, phi): return compile_gates(2, [(gen_Rz(phi-sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(-phi+sympy.pi/2), [1])])
    def gen_CROT_decomp(theta, phi): return compile_gates(2, [(gen_Rz(phi), [1]), (gen_Ry(-sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(sympy.pi/2), [1]), (gen_Rz(-phi), [1])])
    def gen_Rxx_decomp(theta): return compile_gates(2, [(gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1]), (gen_H(), [0]), (gen_H(), [1])])
    def gen_Ryy_decomp(theta): return compile_gates(2, [(gen_Rx(sympy.pi/2), [0]), (gen_Rx(sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1]), (gen_Rx(-sympy.pi/2), [0]), (gen_Rx(-sympy.pi/2), [1])])
    def gen_Rzz_decomp(theta): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_Rz(theta), [1]), (gen_CNOT(), [0, 1])])
    def gen_Rxy_decomp(phi): return compile_gates(2, [(gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(phi/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_Sdg(), [0]), (gen_Sdg(), [1]), (gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(phi/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_S(), [0]), (gen_S(), [1])]).applyfunc(quantsimprw)
    def gen_CZPowGate_decomp(t): return gen_CP_decomp(sympy.pi*t)
    def gen_iSWAP_pow_decomp(alpha): return compile_gates(2, [(gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(-(alpha*sympy.pi)/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_Sdg(), [0]), (gen_Sdg(), [1]), (gen_H(), [0]), (gen_H(), [1]), (gen_CNOT(), [0,1]), (gen_Rz(-(alpha*sympy.pi)/2), [1]), (gen_CNOT(), [0,1]), (gen_H(), [0]), (gen_H(), [1]), (gen_S(), [0]), (gen_S(), [1])]).applyfunc(quantsimprw)
    def gen_fSim_decomp(theta, phi): return compile_gates(2, [(gen_iSWAP_pow_decomp(-2*theta/sympy.pi), [0, 1]), (gen_CZPowGate_decomp(-phi/sympy.pi), [0, 1])])
    def gen_SYC_decomp(): return gen_fSim_decomp(sympy.pi/2, sympy.pi/6) #compile_gates(2, [(gen_Rz(-3*sympy.pi/4), [0]), (gen_Rz(sympy.pi/4), [1]), (gen_SX(), [0]), (gen_SX(), [1]), (gen_Rz(-sympy.pi), [0]), (gen_Rz(sympy.pi), [1]), (gen_SX(), [1]), (gen_Rz(5*sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_SX(), [0]), (gen_Rz(-3*sympy.pi/4), [1]), (gen_SX(), [1]), (gen_Rz(sympy.pi), [1]), (gen_SX(), [1]), (gen_Rz(9*sympy.pi/4), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(sympy.pi/2), [0]), (gen_SX(), [1]), (gen_SX(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_Rz(11*sympy.pi/12), [0]), (gen_SX(), [1]), (gen_SX(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_CNOT(), [0, 1]), (gen_SX(), [0]), (gen_Rz(sympy.pi/2), [1]), (gen_Rz(sympy.pi/6), [0]), (gen_SX(), [1]), (gen_Rz(-sympy.pi/3), [1])])
    def gen_CU_decomp(theta, phi, lbda, gamma): return compile_gates(2, [(gen_Rz((phi-lbda)/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-(phi+lbda)/2), [1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(lbda), [1]), (gen_P((lbda+phi)/2+gamma), [0])])
    def gen_CU3_decomp(theta, phi, lbda): return compile_gates(2, [(gen_Rz((phi-lbda)/2), [1]), (gen_CNOT(), [0, 1]), (gen_Rz(-(phi+lbda)/2), [1]), (gen_Ry(-theta/2), [1]), (gen_CNOT(), [0, 1]), (gen_Ry(theta/2), [1]), (gen_Rz(lbda), [1]), (gen_P((lbda+phi)/2), [0])])
    def gen_CCZ_decomp(): return compile_gates(3, [(gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1])])
    def gen_CCX_decomp(): return compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1])])
    def gen_SWAP_decomp(): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_CNOT(), [0, 1])])
    def gen_CSWAP_decomp(): return compile_gates(3, [(gen_CNOT(), [2, 1]), (gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tdg(), [1]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [2, 1])])
    def gen_iSWAP_decomp(): return compile_gates(2, [(gen_S(), [0]), (gen_S(), [1]), (gen_H(), [0]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_H(), [1])])
    def gen_SX_test(): return sympy.Matrix([[1+sympy.I, 1-sympy.I], [1-sympy.I, 1+sympy.I]])/2
    def gen_Rx_test(theta): return sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2)], [-sympy.I*sympy.sin(theta/2), sympy.cos(theta/2)]])
    def gen_SYC_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 0, -sympy.I, 0], [0, -sympy.I, 0, 0], [0, 0, 0, sympy.exp(-sympy.I*sympy.pi/6)]])
    #print(compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tdg(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2])]))
    """
    assert gen_Rx(theta) == gen_Rx_test(theta)
    assert gen_SX() == gen_SX_test()
    assert gen_SYC() == gen_SYC_test()
    assert gen_CZ() == gen_CZ_decomp()
    assert gen_CY() == gen_CY_decomp()
    assert gen_CRX(theta) == gen_CRX_decomp(theta)
    assert gen_CRY(theta) == gen_CRY_decomp(theta)
    assert gen_CRZ(theta) == gen_CRZ_decomp(theta)
    assert gen_CCZ() == gen_CCZ_decomp()
    assert gen_CCX() == gen_CCX_decomp()
    assert gen_CP(phi) == gen_CP_decomp(phi)
    assert gen_CS() == gen_CS_decomp()
    assert gen_CH() == gen_CH_decomp()
    assert gen_CR(theta, phi) == gen_CR_decomp(theta, phi)
    assert gen_Rxx(theta) == gen_Rxx_decomp(theta)
    assert gen_Ryy(theta) == gen_Ryy_decomp(theta)
    assert gen_Rzz(theta) == gen_Rzz_decomp(theta)
    assert gen_Rxy(phi) == gen_Rxy_decomp(phi)
    assert gen_CROT(theta, phi) == gen_CROT_decomp(theta, phi)
    assert gen_CZPowGate(alpha) == gen_CZPowGate_decomp(alpha)
    """
    assert gen_iSWAP_pow(alpha) == gen_iSWAP_pow_decomp(alpha), (gen_iSWAP_pow(alpha), gen_iSWAP_pow_decomp(alpha))
    assert gen_fSim(theta, phi) == gen_fSim_decomp(theta, phi), (gen_fSim(theta, phi), gen_fSim_decomp(theta, phi))
    assert gen_SYC() == gen_SYC_decomp(), (gen_SYC(), gen_SYC_decomp())
    assert gen_CU3(theta, phi, lbda) == gen_CU3_decomp(theta, phi, lbda)
    assert gen_CU(theta, phi, lbda, gamma) == gen_CU_decomp(theta, phi, lbda, gamma)
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

purity_analysis(); assert False

def projectq_import_qasm(filename, eng, initial_state=None):
    import re, math
    from projectq.ops import H, X, Y, Z, S, SqrtX, Sdag, T, Tdag, R, Rx, Ry, Rz, CNOT, CZ
    def _eval_angle(expr):
        # supports e.g. "pi/2", "3*pi/4", numeric literals
        expr = expr.strip().lower().replace("pi", f"({math.pi})")
        return eval(expr, {"__builtins__": {}}, {})
    # Minimal OpenQASM-2 parser for common qelib1 gates.
    # Ignores: creg/measure/barrier/includes/custom-gates. Extend as needed.
    _rx = re.compile(r"rx\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _ry = re.compile(r"ry\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _rz = re.compile(r"rz\(([^)]+)\)\s+q\[(\d+)\];", re.I)
    _u3 = re.compile(r"u3\(([^,]+),([^,]+),([^)]+)\)\s+q\[(\d+)\];", re.I)
    _one = re.compile(r"(h|x|y|z|s|t|sdg|tdg|sx|r)\s+q\[(\d+)\];", re.I)
    _cx  = re.compile(r"cx\s+q\[(\d+)\],\s*q\[(\d+)\];", re.I)
    _cz  = re.compile(r"cz\s+q\[(\d+)\],\s*q\[(\d+)\];", re.I)
    _meas   = re.compile(r"measure\s+q\[(\d+)\]\s*->\s*([A-Za-z_]\w*)\[(\d+)\];", re.I)
    _creg   = re.compile(r"creg\s+([A-Za-z_]\w*)\[(\d+)\];", re.I)
    _qreg = re.compile(r"qreg\s+q\[(\d+)\];", re.I)
    with open(filename, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("//")]
    nq = None
    for ln in lines:
        m = _qreg.search(ln)
        if m:
            nq = int(m.group(1)); break
    qureg = eng.allocate_qureg(nq)
    if not initial_state is None:
        eng.flush()
        eng.backend.set_wavefunction(initial_state.tolist(), qureg)
    replay = []
    for ln in lines:
        if ln.lower().startswith(("openqasm", "include", "qreg", "creg", "barrier", "measure")):
            continue
        m = _one.match(ln)
        if m:
            gate, i = m.group(1).lower(), int(m.group(2))
            replay.append(({ "h":H, "x":X, "y":Y, "z":Z, "s":S, "t":T, "sdg":Sdag, "tdg":Tdag, "sx":SqrtX, "r":R }[gate], qureg[i]))
            continue
        m = _rx.match(ln)
        if m: replay.append((Rx(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _ry.match(ln)
        if m: replay.append((Ry(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _rz.match(ln)
        if m: replay.append((Rz(_eval_angle(m.group(1))), qureg[int(m.group(2))])); continue
        m = _u3.match(ln)
        if m:
            theta, phi, lam, i = (_eval_angle(m.group(1)), _eval_angle(m.group(2)), _eval_angle(m.group(3)), int(m.group(4)))
            # Decompose U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
            replay.append((Rz(phi), qureg[i])); replay.append((Ry(theta), qureg[i])); replay.append((Rz(lam), qureg[i]))
            continue
        m = _cx.match(ln)
        if m: replay.append((CNOT, (qureg[int(m.group(1))], qureg[int(m.group(2))]))); continue
        m = _cz.match(ln)
        if m: replay.append((CZ, (qureg[int(m.group(1))], qureg[int(m.group(2))]))); continue
        # Unknown/unsupported line types are silently skipped. Add more patterns as needed.
    return qureg, replay
def preprocess_qasm_angles(qasm: str) -> str:
    import re, ast, math

    _ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
                    ast.Name, ast.Constant, ast.Call)
    _ALLOWED_NAMES = {"pi": math.pi}
    # If you also want 'tau' or 'e': add {"tau": 2*math.pi, "e": math.e}

    def _safe_eval(expr: str) -> float:
        """Evaluate a simple arithmetic expression with 'pi' safely."""
        # normalize unicode minus, strip spaces
        expr = expr.replace("−", "-").strip()
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, _ALLOWED_NODES):
                raise ValueError(f"Disallowed token in angle expression: {type(n).__name__}")
            if isinstance(n, ast.Name) and n.id not in _ALLOWED_NAMES:
                raise ValueError(f"Unknown name in angle expression: {n.id}")
            if isinstance(n, ast.Call):  # disallow function calls
                raise ValueError("Function calls not allowed in angle expressions")
        return float(eval(compile(node, "<angle>", "eval"), {"__builtins__": {}}, _ALLOWED_NAMES))

    _angle_gate = r"(u1|u2|u3|rx|ry|rz|p|cp|r|crx|cry|crz)"
    # Matches gate(params) with any content until the closing ')'
    _param_pat = re.compile(rf"\b{_angle_gate}\s*\(([^)]+)\)", re.IGNORECASE)

    def _format_float(x: float) -> str:
        # 17 digits -> round-trip for double; avoid scientific noise where possible
        return f"{x:.17g}"
    """Replace symbolic angle expressions (with 'pi') by numeric literals."""
    def _repl(m: re.Match) -> str:
        gate = m.group(1)
        params = m.group(2)
        # split on commas that are not nested (these are flat anyway)
        parts = [p.strip() for p in params.split(",")]
        new_parts = [_format_float(_safe_eval(p)) for p in parts]
        return f"{gate}(" + ",".join(new_parts) + ")"
    return _param_pat.sub(_repl, qasm)
import numpy as np
def normalize_state(state):
  return state / np.linalg.norm(state)
def state_vector_equivalence(psi, phi):
  """Checks if two quantum state vectors are equal up to a global phase."""
  # Normalize to prevent issues if vectors are not exactly normalized
  psi = normalize_state(psi)
  phi = normalize_state(phi)
  
  # Compute relative phase using the first nonzero entry
  phase_factor = np.vdot(psi, phi)  # Inner product
  phase = np.angle(phase_factor)
  
  # Compare after removing global phase
  return np.allclose(psi * np.exp(-1j * phase), phi)
def test_simulation(max_qubits = 4, random_initial_state=True):
    import numpy as np
    SVSIM_METHOD_NAMES = ["SQUANDER", "Qiskit", "qulacs", "Cirq", "ProjectQ"]
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print(f"Total QASM: {len(files)}, max qubits: {max_qubits}")
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
        num_gates = len(qc.data)
        fname = os.path.basename(filename)
        if num_gates > MAX_GATES_ALLOWED or not qc_gates_names.issubset(SUPPORTED_GATES_NAMES) or filename.endswith("_qsearch.qasm") or filename.endswith("_squander.qasm"):
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {qc_gates_names} num_gates {num_gates}")
            continue
        qbit_num = qc.num_qubits
        matrix_size = 1 << qbit_num
        print(f"Testing {fname}; qubits {qbit_num} num_gates {num_gates} matrix_size {matrix_size}")
        if (random_initial_state ) :
            initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
            initial_state = initial_state_real + initial_state_imag*1j
            initial_state = normalize_state(initial_state)
        else:
            initial_state = np.zeros( (matrix_size), dtype=np.complex128 )
            initial_state[0] = 1.0 + 0j
        res = {}
        for method in SVSIM_METHOD_NAMES:
            output = [None, None]
            if method == "SQUANDER":
                def f():
                    transformed_state = initial_state.copy()
                    circ, params, _ = PartitionCircuitQasm( filename, max_qubits, strategy="ilp-fusion" )
                    circ.set_min_fusion(14)
                    def run():
                        circ.apply_to(params, transformed_state)
                        output[0] = transformed_state
                    output[1] = run
            elif method == "Qiskit":
                import qiskit_aer as Aer
                from qiskit import transpile
                def f():
                    circuit_qiskit = QuantumCircuit.from_qasm_file(filename)
                    init = QuantumCircuit(qbit_num)
                    init.initialize( initial_state )
                    circuit_qiskit = circuit_qiskit.compose(init, front=True)
                    circuit_qiskit.save_statevector()
                    backend = Aer.AerSimulator(method='statevector', fusion_enable=True, fusion_max_qubit=max_qubits)
                    compiled_circuit = transpile(circuit_qiskit, backend)
                    def run():
                        result = backend.run(compiled_circuit).result()
                        transformed_state = result.get_statevector(compiled_circuit)
                        output[0] = np.array(transformed_state)
                    output[1] = run
            elif method == "qulacs":
                from qulacs import QuantumState, converter
                import qulacs
                def f():
                    state = QuantumState(qbit_num)
                    state.load( initial_state )
                    with open(filename, 'r') as file:
                        circuit_qulacs = converter.convert_QASM_to_qulacs_circuit([preprocess_qasm_angles(x) for x in file.readlines() if not x.startswith("creg")])
                    def run():
                        circuit_qulacs.update_quantum_state( state )
                        transformed_state = state.get_vector()
                        output[0] = transformed_state
                    output[1] = run
            elif method == "Cirq":
                import cirq #pip install cirq, ply, qsimcirq
                from cirq.contrib.qasm_import import circuit_from_qasm
                import qsimcirq
                def f():
                    with open(filename, 'r') as file:
                        circuit_cirq = circuit_from_qasm(file.read())
                    #simulator = cirq.Simulator()
                    #initial_state_cirq = cirq.StateVectorSimulationState(initial_state=initial_state, qubits=[cirq.NamedQubit(f"q_{i}") for i in range(qbit_num-1,-1,-1)])
                    options = qsimcirq.QSimOptions(max_fused_gate_size=max_qubits, cpu_threads=os.cpu_count())
                    qs = qsimcirq.QSimSimulator(options)
                    state = initial_state.astype(np.complex64)
                    def run():
                        result = qs.simulate(circuit_cirq, qubit_order=[cirq.NamedQubit(f"q_{i}") for i in range(qbit_num-1,-1,-1)], initial_state=state)
                        #result = simulator.simulate(circuit_cirq, initial_state=initial_state_cirq)
                        transformed_state = result.final_state_vector
                        output[0] = transformed_state
                    output[1] = run
            elif method == "ProjectQ":
                from projectq import MainEngine
                from projectq.backends import Simulator
                from projectq.ops import Measure, All
                def f():
                    eng = MainEngine(Simulator(gate_fusion=True))
                    qureg, replay = projectq_import_qasm(filename, eng, initial_state)
                    def run():
                        for op in replay: op[0] | op[1]
                        eng.flush() #Execute; ProjectQ performs fusion inside the C++ backend
                        _, psi = eng.backend.cheat()
                        transformed_state = np.array(psi, dtype=complex)
                        output[0] = transformed_state
                        All(Measure) | qureg
                    output[1] = run
            print(method)
            tpart = timeit.timeit(f, number=1)
            trun = timeit.timeit(output[1], number=1)
            res[method] = (tpart, trun, output[0])
            #assert state_vector_equivalence(res["SQUANDER"][2], res[method][2])
            overlap = np.abs(np.dot(res["SQUANDER"][2], np.conj(res[method][2])))
            assert np.isclose(overlap, 1.0, atol=1e-3), overlap
            #assert np.linalg.norm(res["SQUANDER"][2] - res[method][2]) < 1e-8
        print({x: res[x][:2] for x in res})

def test_partitions(max_qubits = 4):
    """
    Benchmark partitioning strategies on QASM circuits
    Args:
        
        max_qubits: Max qubits per partition
    """
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print(f"Total QASM: {len(files)}, max qubits: {max_qubits}")
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
        num_gates = len(qc.data)
        fname = os.path.basename(filename)
        if num_gates > MAX_GATES_ALLOWED or not qc_gates_names.issubset(SUPPORTED_GATES_NAMES) or filename.endswith("_qsearch.qasm") or filename.endswith("_squander.qasm"):
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {qc_gates_names} num_gates {num_gates}")
            continue
        circ, _ = utils.qasm_to_squander_circuit(filename)
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        gate_to_qubit = { i: get_qubits(g) for i, g in gate_dict.items() }
        gate_to_tqubit = { i: g.get_Target_Qbit() for i, g in gate_dict.items() }
        print(fname, qc.num_qubits, num_gates, f"k={max_qubits}",
              "Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, [[i] for i in gate_dict]),
              "Control-aware Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, [[i] for i in gate_dict]))
        res = {}
        for method in METHOD_NAMES:
            print(method)
            ls = []
            def f():
                ls.extend(PartitionCircuitQasm( filename, max_qubits, method ))
            t = timeit.timeit(f, number=1)
            partitioned_circuit, parameters, L = ls
            res[method] = len(partitioned_circuit.get_Gates()), t, total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, L), total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, L)
        allfiles[fname] = (qc.num_qubits, num_gates, res)
        print(fname, allfiles[fname])
    import json
    print(json.dumps(allfiles, indent=2))    
    import matplotlib.pyplot as plt
    sorted_items = sorted(allfiles.items(), key=lambda item: (item[1][0], item[1][1]))
    circuits_sorted = [name for name, _ in sorted_items]
    circuits_labels = [name.replace(".qasm", f" ({x[0]}, {x[1]})") for name, x in sorted_items]
    print("Circuit Name & Qubit Count & Gate Count & " + " & ".join(METHOD_NAMES) + r"\\")
    if USE_ILP:
        ilpbest = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] != min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        ilptie = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] == min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilpbest for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilptie for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    else:
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in circuits_sorted for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    markers = ["o", "*", "D", "s", "+", "<", ">", "v", "^"]
    for j in range(4):
        title = ["Total Partitions", "Runtime Performance", "Total Float Ops for Fusion", "Total Float Ops for Control-Aware Fusion"][j]
        y_label = ["Partition Count", "Time in seconds", "Number of Float Ops", "Number of Float Ops"][j]
        for i, strat in enumerate(METHOD_NAMES):
            y = [allfiles[name][2][strat][j] for name in circuits_sorted]
            plt.plot(circuits_labels, y, marker=markers[i], label=strat)
        plt.xlabel("Circuit (sorted by qubits, gates)")
        plt.ylabel(y_label, fontsize=6)
        plt.title(f"{y_label} - Maximum k={max_qubits} Qubits per Partition")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}-{max_qubits}-max_qubit.svg", format="svg", transparent=True)
        plt.clf()

if __name__ == "__main__":
    for max_qubits in range(3, 6):
        pass #test_partitions(max_qubits)
    test_simulation(5)
