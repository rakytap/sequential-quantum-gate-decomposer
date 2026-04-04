"""
Implementation to optimize wide circuits (i.e. circuits with many qubits) by    partitioning the circuit into smaller partitions and redecompose the smaller partitions
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os, contextlib, collections, time


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import translate_param_order, build_dependency
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology

CNOT_COUNT_DICT = {'CNOT': 1, 'CH': 1, 'CZ': 1, 'SYC': 3, 'CRY': 2, 'CU': 2, 'CR': 2, 'CROT': 2, 'CRX': 2, 'CRZ': 2, 'CP': 2, 'CCX': 6, 'CSWAP': 7, 'SWAP': 3}
def CNOTGateCount( circ: Circuit, max_gates: int = 0 ) -> int :
    """
    Call to get the number of CNOT gates in the circuit

    
    Args:

        circ (Circuit) A squander circuit representation


    Return:

        Returns with the CNOT gate count

    
    """ 

    if not isinstance(circ, Circuit ):
        raise Exception("The input parameters should be an instance of Squander Circuit")

    gate_counts = circ.get_Gate_Nums()    
    num_cnots = sum(CNOT_COUNT_DICT.get(gate, 0) * count for gate, count in gate_counts.items())

    if max_gates > 0: return num_cnots*max_gates + sum(y for x, y in gate_counts.items() if x not in CNOT_COUNT_DICT)
    return num_cnots



class N_Qubit_Decomposition_Guided_Tree(N_Qubit_Decomposition_custom):
    def __init__(self, Umtx, config, accelerator_num, topology, paramspace=None, paramscale=None):
        super().__init__(Umtx[0] if isinstance(Umtx, list) else Umtx, config=config, accelerator_num=accelerator_num)
        self.Umtx = Umtx if isinstance(Umtx, list) else [Umtx] #already conjugate transposed
        self.qbit_num = self.Umtx[0].shape[0].bit_length() -1
        self.config = config
        self.accelerator_num = accelerator_num
        self.paramspace = paramspace
        self.paramscale = () if paramscale is None else paramscale
        #self.set_Cost_Function_Variant( 0 )	 #0 is Frobenius, 3 is HS, 10 is OSR
        if topology is None:
            topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        self.topology = topology
    def enumerate_unordered_cnot_BFS(n: int, topology=None, use_gl=True):
        # Precompute unordered pairs
        topology = [(i, j) for i in range(n) for j in range(i+1, n)] if topology is None else topology
        prior_level_info = None
        while True:
            visited, seq_pairs_of, seq_dir_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(n, topology, prior_level_info, use_gl=use_gl)
            if not res: break
            yield res
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res)))
    def canonical_prefix_ok(seq):
        m = len(seq)
        if m <= 1: return -1
        succ = {}
        indeg = {}
        last_on = {}
        for k in range(m):
            for q in seq[k]:
                if q in last_on:
                    p = last_on[q]
                    succ.setdefault(p, []).append(k)
                    indeg[k] = indeg.get(k, 0) + 1
                last_on[q] = k
        import heapq
        pq = [(seq[x], x) for x in range(m) if indeg.get(x, 0) == 0]
        heapq.heapify(pq)
        for pos in range(m):
            # Kahn's algorithm
            if len(pq) == 0: return pos #malformed (shouldn't happen)
            u = heapq.heappop(pq)
            if u[1] != pos: return pos  #deviation: not canonical
            for v in succ.get(u[1], ()):
                indeg[v] -= 1
                if indeg[v] == 0: heapq.heappush(pq, (seq[v], v))
        return -1
    def enumerate_unordered_cnot_BFS_level(n: int, topology=None, prior_level_info=None, use_gl=True):
        """
        Enumerate GL(n,2) states in increasing CNOT depth.
        Moves are *recorded* as unordered pairs (for your "structure" view)
        but each expansion tries both directions internally.

        Yields: (depth, A, seq_pairs, seq_directed)
        - depth: minimal number of CNOTs
        - A: packed matrix (tuple of n bit-rows)
        - seq_pairs: minimal-length sequence of unordered pairs that reaches A
        - seq_directed: a matching directed-move realization of seq_pairs
        """
        if prior_level_info is None:
            # Initial state
            start_key = tuple(1 << i for i in range(n))

            # Visited: we only need to mark states once (minimal depth)
            visited = {start_key}

            # We also keep *one* representative sequence per state (unordered + directed)
            seq_pairs_of = {start_key: []}
            seq_dir_of = {start_key: []}

            # Yield the root
            return visited, seq_pairs_of, seq_dir_of, [(start_key, [], [])]
        else:
            visited, seq_pairs_of, seq_dir_of, q = prior_level_info
        res = []
        new_seq_pairs_of = {}
        new_seq_dir_of = {}

        while q:
            A = q.pop()
            last_pairs = seq_pairs_of[A]
            last_dirs = seq_dir_of[A]
            for p in topology:
                if not use_gl:
                    if len(last_pairs) >= 3 and all(p==x for x in last_pairs[-3:]): continue # avoid more than 3 repeated CNOTs
                    if N_Qubit_Decomposition_Guided_Tree.canonical_prefix_ok(last_pairs + [p]) >= 0: continue  # not canonical prefix
                # Try both directions, but record the *same* unordered step 'p'
                for mv in (p, (p[1], p[0])) if use_gl else (p,):
                    #CNOT left
                    if use_gl:
                        if mv[0] == mv[1]: B = A
                        else: B = list(A); B[mv[1]] ^= B[mv[0]]; B = tuple(B)
        
                        if B in visited: continue  # already discovered at minimal depth
                    else: B = tuple(last_dirs + [p])

                    visited.add(B)
                    new_seq_pairs_of[B] = last_pairs + [p]
                    new_seq_dir_of[B] = last_dirs + [mv]

                    # Emit as soon as we discover the state (BFS → minimal depth)
                    res.append((B, new_seq_pairs_of[B], new_seq_dir_of[B]))
        return visited, new_seq_pairs_of, new_seq_dir_of, res
    def build_sequence(stop=5, ordered=True, use_gl=True):
        #https://oeis.org/A002884
        #unordered sequence: 1, 1, 4, 88, 9556, 4526605
        #unordered at 5 qubits: {0: 1, 1: 10, 2: 85, 3: 650, 4: 4475, 5: 27375, 6: 142499, 7: 580482, 8: 1501297, 9: 1738232, 10: 517884, 11: 13591, 12: 24} 
        for i in range(2, stop+1):
            d = {}
            for z in N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(i, use_gl=use_gl):
                for x in (list if ordered else set)(tuple(x[1]) for x in z):
                    d[len(x)] = d.get(len(x), 0) + 1
                if not use_gl and len(d) > 5: break
            print({x: d[x] for x in sorted(d)}, sum(d.values()))
    def extract_bits(x, pos):
        return sum(((x >> p) & 1) << i for i, p in enumerate(pos))
    def build_osr_matrix(U, n, A):
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        dA = 1 << len(A)
        dB = 1 << len(B)
        return U.reshape([2]*(2*n)).transpose(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)).reshape(dA*dA, dB*dB)
    def accumulate_grad_for_cut(U, G, Umat, VTmat, n, A): # qubits on A
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        mat = np.array(G) * Umat @ VTmat  # reconstruct U from its dyadic decomposition
        revmap = [None]*(2*n)
        for i, x in enumerate(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)):
            revmap[x] = i
        U += mat.reshape([2]*(2*n)).transpose(tuple(revmap)).reshape(*U.shape)
        return U
    def trace_out_qubits(U, n, A):
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A)
        M = np.linalg.svd(M, compute_uv=True, full_matrices=False)[0][:,0].reshape(1<<len(A), 1<<len(A))
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(M)
    def numerical_rank_osr(M, Fnorm, tol=1e-10):
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False) / Fnorm
        #print(s)
        return int(np.sum(s >= s[0]*tol)), s
    def operator_schmidt_rank(U, n, A, Fnorm, tol=1e-10):
        return N_Qubit_Decomposition_Guided_Tree.numerical_rank_osr(N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A), Fnorm, tol)
    def unique_cuts(n):
        import itertools
        """All nontrivial unordered bipartitions (no complements)."""
        qubits = tuple(range(n))
        for r in range(1, n//2 + 1):  # only up to half
            for S in itertools.combinations(qubits, r):
                if r < n - r:
                    yield S
                else:  # r == n-r (only possible when n even): tie-break
                    comp = tuple(q for q in qubits if q not in S)
                    if S < comp:      # lexicographically smaller tuple wins
                        yield S
    def get_circuit_from_pairs(self, pairs, finalizing=True):
        circ = Circuit(self.qbit_num)
        for pair in pairs:
            circ.add_U3(pair[0])
            circ.add_U3(pair[1])
            circ.add_CNOT(pair[0], pair[1])
        if finalizing:
            for qbit in range(self.qbit_num):
                circ.add_U3(qbit)
        return circ
    def ceil_log2(x): return 0 if x == 0 else (x-1).bit_length()
    def logsumexp_smoothmax(Lc, tau=1e-2):
        if not Lc: return 0.0
        if tau <= 0.0: raise RuntimeError("tau must be > 0")
        m = max(Lc)
        acc = 0.0
        for v in Lc: acc += np.exp((v - m)/tau)
        return tau * np.log(acc) + m
    def dyadic_loss(S, max_dyadic, rho=0.9, tol=1e-4):
        tot_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(len(S))
        w = 1.0
        acc = 0.0
        for k in range(max_dyadic-1, -1, -1):
            if k < tot_dyadic:
                val = S[1 << k] - S[0] * tol    
                acc += w * val * val
            w *= rho
        return acc
    def avg_loss(cuts_S, rho=0.9):
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.dyadic_loss(S, max_dyadic, rho)
        return total_loss / len(cuts_S)
    # Aggregated cost over cuts: softmax (log-sum-exp) of per-cut dyadic losses
    def cuts_softmax_dyadic_cost(cuts_S, rho=0.1, tau=1e-2):
        if tau <= 0.0: raise RuntimeError("tau must be > 0")
        Lc = []
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.dyadic_loss(S, max_dyadic, rho))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    # Gradient w.r.t. the singular values (diagonal of dL/dΣ):
    def dyadic_loss_grad_diag(S, max_dyadic, Fnorm, rho=0.1, tol=1e-4):
        n = len(S)
        # c_k = rho^k / Mk  for k=1..n-1, then prefix sum C_j = sum_{k=1}^j c_k
        tot_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(n)
        grad = [0.0] * tot_dyadic
        w = 1.0
        for k in range(max_dyadic-1, -1, -1):
            if k < tot_dyadic:
                idx = 1 << k
                grad[k] = 2.0 * w * S[idx] * (1.0-tol) / Fnorm  #1-tol not needed if using stop-grad
            w *= rho                         # w = rho^k
        return grad
    def cuts_avg_dyadic_grad(cuts_S, Fnorm, rho=0.1):
        C = len(cuts_S)
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        Lc = []
        for c in range(C):
            Lc.append(N_Qubit_Decomposition_Guided_Tree.dyadic_loss_grad_diag(cuts_S[c], max_dyadic, Fnorm * C, rho))
        return Lc
    # Gradient w.r.t. singular values (same length as S).
    # Only dyadic positions (1,2,4,...) get nonzero entries; others are 0.
    def cuts_softmax_tail_grad(cuts_S, Fnorm, rho=0.1, tau=1e-2):
        C = len(cuts_S)
        if C == 0: return []
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        # 1) per-cut losses
        Lc = [N_Qubit_Decomposition_Guided_Tree.dyadic_loss(cuts_S[c], max_dyadic, rho) for c in range(C)]

        # 2) softmax weights w_c = exp((Lc - m)/tau) / Z
        m = max(Lc)
        w = [np.exp((Lc[c] - m)/tau) for c in range(C)]
        Z = np.sum(w)
        for c in range(C): w[c] /= (Z if Z > 0.0 else 1.0)

        # 3) dL/dS^{(c)} = w_c * dL_c/dS^{(c)}
        return [[v * w[c] for v in N_Qubit_Decomposition_Guided_Tree.dyadic_loss_grad_diag(cuts_S[c], max_dyadic, Fnorm, rho)] for c in range(C)]        
    def loss_for_rank(S, rank):
        start = 1 << rank
        if start >= len(S): return 0.0
        return sum(x*x for x in S[start:])
    def avg_loss_for_rank(cuts_S, rank):
        if not cuts_S: return 0.0
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank)
        return total_loss / len(cuts_S)
    # Aggregated cost over cuts: softmax (log-sum-exp) of per-cut dyadic losses
    def cuts_softmax_rank_cost(cuts_S, rank, tau=1e-2):
        Lc = []
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    # Gradient w.r.t. the singular values (diagonal of dL/dΣ):
    def loss_for_rank_grad_diag(S, rank, Fnorm):
        """
        Gradient of a single-cut tail loss with respect to the RAW singular values,
        assuming S is already normalized and Fnorm is treated as constant.

        If S = sigma / Fnorm, then d/dsigma_i sum_{j>=r} S_j^2 = 2*S_i/Fnorm on tail.
        """
        n = len(S)
        start = 1 << rank
        grad = [0.0] * n
        if start >= n:
            return grad
        invF = 1.0 / Fnorm
        for i in range(start, n):
            grad[i] = 2.0 * S[i] * invF
        return grad
    def cuts_avg_rank_grad(cuts_S, rank, Fnorm):
        """
        Gradient of average tail loss across cuts.
        Returns one gradient vector per cut, same length as that cut's S.
        """
        C = len(cuts_S)
        if C == 0:
            return []
        scale = 1.0 / C
        out = []
        for S in cuts_S:
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(S, rank, Fnorm)
            out.append([scale * v for v in g])
        return out
    # Gradient w.r.t. singular values (same length as S).
    def cuts_softmax_rank_grad(cuts_S, rank, Fnorm, tau=1e-2):
        """
        Gradient of smooth-max across cuts:
            L = tau * log(sum_c exp(L_c / tau))
        so
            dL = sum_c softmax_c * dL_c
        """
        C = len(cuts_S)
        if C == 0:
            return []
        if tau <= 0.0:
            raise RuntimeError("tau must be > 0")

        Lc = [
            N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank)
            for S in cuts_S
        ]

        m = max(Lc)
        w = [np.exp((v - m) / tau) for v in Lc]
        Z = np.sum(w)
        if Z <= 0.0:
            Z = 1.0
        w = [x / Z for x in w]

        out = []
        for c, S in enumerate(cuts_S):
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(S, rank, Fnorm)
            out.append([w[c] * v for v in g])
        return out
    # Build M with build_osr_matrix, then SVD (econ) and grab top triplet.
    def top_k_triplet_for_cut(U, # (N x N), row-major, N = 1<<q
        q,                  # number of qubits
        A,  # qubits on side A
        Fnorm            # e.g., sqrt(N)
        ):
        # 1) Build M for this cut    
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, q, A)
        k = min(M.shape)

        # 2) SVD: M = U * diag(S) * VT  (VT = V^H)
        # Row-major API handles leading dims as col counts.
        res = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        return res.S / Fnorm, res.U, res.Vh # normalized singular value
    def get_deriv_osr_entanglement(matrix, use_cuts, rank, use_softmax):
        qbit_num = len(matrix).bit_length()-1
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(qbit_num)) if len(use_cuts) == 0 else use_cuts
        Fnorm = np.sqrt(len(matrix))
        deriv = np.zeros(matrix.shape, dtype=complex)
        # Compute the derivative of the OSR entanglement cost function
        triplets = []
        allS = []
        for cut in cuts:
            # 1) top k triplet on the normalized reshape M_c
            S, Umat, VTmat = N_Qubit_Decomposition_Guided_Tree.top_k_triplet_for_cut(matrix, qbit_num, cut, Fnorm)
            triplets.append(([], Umat, VTmat))
            allS.append(S)
        if use_softmax: allS = N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_grad(allS, rank, Fnorm)
        else: allS = N_Qubit_Decomposition_Guided_Tree.cuts_avg_rank_grad(allS, rank, Fnorm)
        for i in range(len(cuts)):
            triplets[i] = (allS[i], triplets[i][1], triplets[i][2])
        for i in range(len(cuts)):
            G, Umat, VTmat = triplets[i]
            N_Qubit_Decomposition_Guided_Tree.accumulate_grad_for_cut(deriv, G, Umat, VTmat, qbit_num, cuts[i])
        return deriv
    # Compute grad component = Re Tr( A^† B ) for A = dL/dU, B = dU/dθ
    # A and B are (rows x cols) with row-major leading dimension.
    def real_trace_conj_dot(A, B):
        return np.sum(A.real * B.real + A.imag * B.imag) # Re Tr(A^† B)
    def param_derivs(circ, Umtx, x):
        n = len(x)
        derivs = [None]*n
        for i in range(n):
            kind = i % 3
            if kind == 0: # d/dt:  ∂U/∂t = U(t+π/2, φ, λ)
                x_shift = x.copy()
                x_shift[i] += np.pi/2
                Ui = Umtx.copy()
                circ.apply_to(x_shift, Ui)
                derivs[i] = Ui
            else: # d/dφ or d/dλ: ∂U/∂p = 0.5*(U(p+π/2) - U(p-π/2))
                xp = x.copy(); xp[i] += np.pi/2
                xm = x.copy(); xm[i] -= np.pi/2
                Up = Umtx.copy()
                Um = Umtx.copy()
                circ.apply_to(xp, Up)
                circ.apply_to(xm, Um)
                derivs[i] = 0.5 * (Up - Um)
        return derivs
    
    def _global_phase_fix(U):
        return U / (np.linalg.det(U)**(1/len(U)))
    def _polar_unitary(X):
        U,_,Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh

    def su2_to_u3_zyz(U):
        """
        Decompose a 2x2 unitary (det=1) into Qiskit U3: Rz(phi) @ Ry(theta) @ Rz(lam).
        Returns (theta, phi, lam) in radians.
        """
        U = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U)
        # Handle numeric edge cases robustly
        a = U[0,0]; b = U[0,1]; c = U[1,0]; d = U[1,1]
        # Prefer arccos for theta; it's stable when |a| is not tiny
        ca = np.clip(np.abs(a), 0.0, 1.0)
        theta = 2.0 * np.arccos(ca)
        # If sin(theta/2) ~ 0, collapse to Z rotations
        eps = 1e-12
        if abs(np.sin(theta/2)) < eps:
            # Then c≈0, b≈0; only Z phases matter: U ≈ e^{iα} Rz(phi+lam)
            # Choose phi=0, lam = arg(d) - arg(a)
            phi = 0.0
            lam = (np.angle(d) - np.angle(a))
            # Normalize to [-pi,pi)
            lam = (lam + np.pi)%(2*np.pi) - np.pi
            return float(theta), float(phi), float(lam)

        # Otherwise, phases from elements and normalize
        phi = (np.angle(c)-np.angle(a)); phi=(phi+np.pi)%(2*np.pi)-np.pi
        lam = (np.angle(b)-np.angle(a)-np.pi); lam=(lam+np.pi)%(2*np.pi)-np.pi
        return float(theta),float(phi),float(lam)

    def _A_from_c(c1,c2,c3):
        X = np.array([[0,1],[1,0]], complex)
        Y = np.array([[0,-1j],[1j,0]], complex)
        Z = np.array([[1,0],[0,-1]], complex)
        XX = np.kron(X,X); YY = np.kron(Y,Y); ZZ = np.kron(Z,Z)
        H = c1*XX + c2*YY + c3*ZZ
        # use exp via eig (4x4) for robustness
        ew, EV = np.linalg.eig(1j*H)
        A = EV @ np.diag(np.exp(ew)) @ np.linalg.inv(EV)
        # project back to unitary (remove numeric drift)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A)
    # Factor K1, K2 → (2x2 ⊗ 2x2)
    def factor_local(K):
        # reshape to (2,2,2,2), SVD the (a,c ; b,d) unfolding
        M = K.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
        U,_,Vh = np.linalg.svd(M, full_matrices=False)
        A = U[:,0].reshape(2,2); B = Vh.conj().T[:,0].reshape(2,2)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A), N_Qubit_Decomposition_Guided_Tree._polar_unitary(B)
    def _magic_basis_plusYY():
        # Complex magic basis (matches A(c)=exp(-i/2*(c1 XX + c2 YY + c3 ZZ)) below)
        # Columns are (|Φ+>, i|Φ->, i|Ψ+>, |Ψ->) up to harmless phases
        return (1/np.sqrt(2))*np.array([
            [1, 0, 0,  1j],
            [0, 1j,1,  0 ],
            [0, 1j,-1, 0 ],
            [1j,0, 0, -1 ]
        ], dtype=complex)

    def _project_to_SO4(O):
        # nearest real orthogonal with det=+1
        O = np.real_if_close(O, tol=1e5)
        U, _, Vt = np.linalg.svd(O)
        O = U @ Vt
        if np.linalg.det(O) < 0:
            O[:,0] *= -1
        return O

    def _clean_col_phases(W):
        Wc = W.copy()
        for j in range(Wc.shape[1]):
            col = Wc[:, j]
            k = np.argmax(np.abs(col))
            if np.abs(col[k]) > 1e-14:
                Wc[:, j] *= np.exp(-1j * np.angle(col[k]))
        return Wc

    def closest_local_product(W4):
        A, B = N_Qubit_Decomposition_Guided_Tree.factor_local(W4)
        return N_Qubit_Decomposition_Guided_Tree._global_phase_fix(A), N_Qubit_Decomposition_Guided_Tree._global_phase_fix(B)
    def kak_u3s_around_cx(U, n, c, t, iters=3):
        U4 = N_Qubit_Decomposition_Guided_Tree.trace_out_qubits(U, n, (c, t))
        U4 = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U4)
        from qiskit.synthesis import TwoQubitWeylDecomposition
        twd = TwoQubitWeylDecomposition(U4)
        c1, c2, c3 = twd.a, twd.b, twd.c
        K1A, K1B, K2A, K2B = twd.K1l, twd.K1r, twd.K2l, twd.K2r
        A = N_Qubit_Decomposition_Guided_Tree._A_from_c(c1,c2,c3)
        U_rec = np.kron(K1A,K1B) @ A @ np.kron(K2A,K2B)
        z = np.trace(U_rec.conj().T @ U4)
        U_rec *= np.exp(1j * np.angle(z))
        print("Frob err:", np.linalg.norm(U_rec - U4), c1, c2, c3)
        thA_pre, phA_pre, laA_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K2A.conj().T)
        thB_pre, phB_pre, laB_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K2B.conj().T)
        thA_post,phA_post,laA_post= N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K1A.conj().T)  # left-apply ⇒ take dagger on outputs
        thB_post,phB_post,laB_post= N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K1B.conj().T)
        return {
            "c": (c1,c2,c3),
            "pre":  { "A": (thA_pre/2, phA_pre, laA_pre),
                    "B": (thB_pre/2, phB_pre, laB_pre) },
            "post": { "A": (thA_post/2, phA_post, laA_post),
                    "B": (thB_post/2, phB_post, laB_post) }
        }
    def params_to_mat(self, params):
        allU = []
        for U, pspace in zip(self.Umtx, [None] if self.paramspace is None else self.paramspace):
            U = U.copy()
            scaled_params = np.sum(params.reshape(-1, 1+len(pspace)) * np.array((1.0,) + pspace), axis=1) if pspace is not None else params
            self.get_Circuit().apply_to(scaled_params if pspace is not None else params, U)
            allU.append(U)
        return allU
    def OSR_with_local_alignment(self, pairs, cuts, Fnorm, tol, rank, use_softmax, method="dual_annealing"):
        if len(pairs) != 0:
            self.set_Cost_Function_Variant( 10 )
            #self.Run_Decomposition(pairs, False)
            self.set_Gate_Structure(self.get_circuit_from_pairs(pairs, False))
            import scipy
            param_bound = np.array(([2*np.pi] + [1/x for x in self.paramscale])*self.get_Parameter_Num())
            def cost(x):
                allU = self.params_to_mat(x)
                S = [N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol)[1] for U in allU for cut in cuts]
                if use_softmax: return N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_cost(S, rank)
                else: return N_Qubit_Decomposition_Guided_Tree.avg_loss_for_rank(S, rank)
            def jacobian(x):
                allU = self.params_to_mat(x)
                grad = np.zeros(len(x), dtype=float)
                for Ubase, U, pspace in zip(self.Umtx, allU, [None] if self.paramspace is None else self.paramspace):
                    dL = N_Qubit_Decomposition_Guided_Tree.get_deriv_osr_entanglement(U, cuts, rank, use_softmax)
                    basevec = np.array((1.0,) if pspace is None else (1.0,) + pspace)
                    scaled_params = np.sum(x.reshape(-1, 1+len(pspace)) * basevec, axis=1) if pspace is not None else x                    
                    derivs = N_Qubit_Decomposition_Guided_Tree.param_derivs(self.get_Circuit(), Ubase, scaled_params)
                    newgrad = np.array([N_Qubit_Decomposition_Guided_Tree.real_trace_conj_dot(dL, deriv) for deriv in derivs])
                    if pspace is not None: newgrad = (np.array(newgrad)[:,np.newaxis] * basevec).reshape(-1)
                    grad += newgrad
                return grad / len(self.Umtx)
            if method == "differential_evolution":
                best = scipy.optimize.differential_evolution(cost, [ (0, x) for x in param_bound ], maxiter=100, polish=False)
                best = scipy.optimize.minimize(cost, best.x, method='BFGS', jac=jacobian, options={'maxiter': 200})
            elif method == "dual_annealing":
                best = None
                for seed in range(20):
                    res = scipy.optimize.dual_annealing(cost, [ (0, x) for x in param_bound ], maxiter=100)#, minimizer_kwargs={'jac': jacobian})
                    if best is None or res.fun < best.fun: best = res
            elif method == "basinhopping":
                best = scipy.optimize.basinhopping(cost, np.random.rand(len(param_bound))*param_bound, niter=50, stepsize=np.pi/2, minimizer_kwargs={'jac': jacobian})
            else:
                best = min([scipy.optimize.minimize(cost, np.random.rand(len(param_bound))*param_bound, method='BFGS', jac=jacobian, options={'maxiter': 200}) for _ in range(20)], key=lambda r: r.fun)
            #print(best)
            self.set_Cost_Function_Variant( 3 )
            allU = self.params_to_mat(best.x)
        else: allU = self.Umtx
        return [(N_Qubit_Decomposition_Guided_Tree.ceil_log2(rank), s) for U in allU for cut in cuts for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol),)]
    def Run_Decomposition(self, pairs, finalizing=True):
        circ = self.get_circuit_from_pairs(pairs, finalizing)
        self.set_Gate_Structure(circ)
        self.set_Optimized_Parameters(np.random.rand(self.get_Parameter_Num())*(2*np.pi))
        super().Start_Decomposition()
        if finalizing:
            params = self.get_Optimized_Parameters()
            self.err = self.Optimization_Problem(params)
            return self.err < self.config.get('tolerance', 1e-8)
    def generate_insertions(curpath, topology, num_cnot):
        import itertools
        n = len(curpath)
        nslots = n + 1
        for places in itertools.combinations_with_replacement(range(nslots), num_cnot):
            for pairs in itertools.product(topology, repeat=num_cnot):
                out = []
                j = 0  # index into inserted pairs
                for slot in range(nslots):
                    while j < num_cnot and places[j] == slot:
                        out.append(pairs[j])
                        j += 1
                    if slot < n:
                        out.append(curpath[slot])
                yield tuple(out)
    def Start_Decomposition(self):
        import heapq, itertools
        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        #because we have U already conjugate transposed, must use prefix order
        B = self.config.get('beam', None)#8*len(self.topology))
        max_depth = self.config.get('tree_level_max', 14)
        tol = 1e-3
        Fnorm = np.sqrt(1<<self.qbit_num)
        best = []
        visited = set()
        all_ranks = list(range(min(2, self.qbit_num-1)))
        def get_osr_stats(path, rank, use_softmax):
            h = self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol, rank=rank, use_softmax=use_softmax, method="basin_hopping")
            min_cnots = max((x[0] for x in h), default=0)
            ranktot = sum(x[0] for x in h)
            kappa = sum(sum(y*y for y in x[1][1:]) for x in h)
            return min_cnots, ranktot+kappa, h
        def add_to_heap(path, parent_stats):
            if len(path) > max_depth: return False
            if path in visited: return False
            visited.add(path)
            if self.qbit_num > 1:
                min_cnots, rankkappa = min(get_osr_stats(path, rank, use_sm)[:2] for (rank, use_sm) in itertools.product(all_ranks, (False,))) #(False, True)
            else: min_cnots, rankkappa = 0, 0.0
            if parent_stats is not None and (min_cnots, rankkappa) >= parent_stats: return False
            heapq.heappush(best, (min_cnots, rankkappa, path))
            return True
        add_to_heap((), None)
        while best:
            #print(best[0])
            min_cnots, rankkappa, curpath = heapq.heappop(best)
            if min_cnots == 0:
                #print(path)
                for i in range(10):
                    if self.Run_Decomposition(curpath):
                        self.all_solutions.append((self.get_Circuit(), self.get_Optimized_Parameters()))
                        if stop_first_solution: return
                        break
                    #print("Looping", h)
            num_cnot = 1
            while True:
                any_added = False
                for newpath in N_Qubit_Decomposition_Guided_Tree.generate_insertions(curpath, self.topology, num_cnot):
                    if add_to_heap(newpath, (min_cnots, rankkappa)): any_added = True
                if any_added: break
                num_cnot += 1
                if len(curpath) + num_cnot > max_depth: break
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        #print("No decomposition found within the given CNOT limit.")
    """
    def Start_Decomposition(self):
        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        if self.topology is None:
            self.topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        pair_affects = {
            pair: [i for i,A in enumerate(cuts) if (pair[0] in A) ^ (pair[1] in A)]
            for pair in self.topology
        }
        #because we have U already conjugate transposed, must use prefix order
        B = self.config.get('beam', None)#8*len(self.topology))
        max_depth = self.config.get('tree_level_max', 14)
        tol = 1e-3
        Fnorm = np.sqrt(1<<self.qbit_num)
        prior_level_info = None
        for depth in range(max_depth+1):
            remaining = max_depth - depth
            visited, seq_pairs_of, seq_dir_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(self.qbit_num, self.topology, prior_level_info, use_gl=False)
            nextprefixes = []
            for path in set(tuple(x[1]) for x in res):
                curh = None if len(path)==0 else prefixes[path[:-1]]
                check_cuts = pair_affects[tuple(sorted(path[-1]))] if not curh is None else range(len(cuts))
                #samples = [max(x[0] for x in self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol)) for _ in range(5)]
                #if len(set(samples)) != 1: print(samples)
                h = self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol, use_softmax=False, method="dual_annealing")
                min_cnots = max((x[0] for x in h), default=0)
                print(path, h, N_Qubit_Decomposition_Guided_Tree.avg_loss([x[1] for x in h]), remaining, min_cnots)
                if min_cnots == 0:
                    #print(path)
                    for i in range(10):
                        if self.Run_Decomposition(path):
                            self.all_solutions.append((self.get_Circuit(), self.get_Optimized_Parameters()))
                            if stop_first_solution: return
                            break
                        #print("Looping", h)
                if min_cnots > remaining: continue
                if not curh is None:
                    #print(path, [(h[i], curh[i]) for i in check_cuts])
                    #if any(h[i][0] > curh[i][0] for i in check_cuts): continue
                    if max((x[0] for x in curh), default=0) < min_cnots: continue
                nextprefixes.append((path, h))
            nextprefixes.sort(key=lambda t: (max((x[0] for x in t[1]), default=0), sum(x[0] for x in t[1]), N_Qubit_Decomposition_Guided_Tree.avg_loss([x[1] for x in t[1]])))
            prefixes = {x[0]: x[1] for x in nextprefixes[:B]}
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res) if tuple(x[1]) in prefixes))
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        #print("No decomposition found within the given CNOT limit.")
    """
    def get_Decomposition_Error(self): return self.err
    def compositions(total, parts):
        """
        All nonnegative integer tuples of length `parts` summing to `total`.
        """
        if parts == 1:
            yield (total,)
            return
        for x in range(total + 1):
            for rest in N_Qubit_Decomposition_Guided_Tree.compositions(total - x, parts - 1):
                yield (x,) + rest
    def solve_best_min_cnots(num_qubits, cuts, rank_kappa, topology, use_surplus=True):
        m = len(topology)
        cut_to_edges = [[i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)] for cut in cuts]
        total = 0
        best_kappa = None
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(sum(edge_counts[j] for j in cut_to_edge)>=cut_bound[0] for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa)):
                    new_kappa = 0.0
                    for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa):
                        coverage = sum(edge_counts[j] for j in cut_to_edge)
                        if use_surplus:
                            new_kappa += cut_bound[1] * (coverage - cut_bound[0])
                        else: new_kappa += cut_bound[1] * coverage
                    best_kappa = new_kappa if best_kappa is None else max(best_kappa, new_kappa)
            if best_kappa is not None: break
            total += 1
        return total, best_kappa
    def solve_min_cnots(num_qubits, cuts, cut_bounds, topology):
        m = len(topology)
        cut_to_edges = [[i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)] for cut in cuts]
        total = 0
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(sum(edge_counts[j] for j in cut_to_edge)>=cut_bound for cut_to_edge, cut_bound in zip(cut_to_edges, cut_bounds)):
                    return total
            total += 1
    def gen_all_min_cnots(num_qbits, topology=None): #OSR tells min CNOTs at most for 3 qubits 3, 4 qubits 6, 5 qubits 7
        import itertools
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(num_qbits))
        min_cnot_bounds = [2*min(cut_size, num_qbits - cut_size) for cut_size in (len(cut) for cut in cuts)]
        if topology is None:
            topology = [(i, j) for i in range(num_qbits) for j in range(i+1, num_qbits)]
        for cnot_bounds in itertools.product(*(range(bound+1) for bound in min_cnot_bounds)):
            #if tuple(sorted(cnot_bounds)) != cnot_bounds: continue
            print(cnot_bounds, N_Qubit_Decomposition_Guided_Tree.solve_min_cnots(num_qbits, cuts, cnot_bounds, topology))
#N_Qubit_Decomposition_Guided_Tree.gen_all_min_cnots(3); assert False
#N_Qubit_Decomposition_Guided_Tree.build_sequence(); assert False
#print(len(list(N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(3, [(0,1),(1,2),])))); assert False
class qgd_Wide_Circuit_Optimization:
    """
    Class implementing the optimization of wide circuits (i.e. circuits with many qubits) by
    partitioning the circuit into smaller partitions and redecompose the smaller partitions

    """
    
    def __init__( self, config ):

        config.setdefault('strategy', 'TreeSearch')
        config.setdefault('parallel', 0 )
        config.setdefault('verbosity', 0 )
        config.setdefault('tolerance', 1e-8 )
        config.setdefault('test_subcircuits', False )
        config.setdefault('test_final_circuit', True )
        config.setdefault('max_partition_size', 3 )
        config.setdefault('topology', None)
        config.setdefault('partition_strategy','ilp')
        
        #testing the fields of config 
        strategy = config[ 'strategy' ]
        allowed_startegies = ['TreeSearch', 'TabuSearch', 'Adaptive', 'TreeGuided' ]
        if not strategy in allowed_startegies :
            raise Exception(f"The decomposition startegy should be either of {allowed_startegies}, got {strategy}.")


        parallel = config[  'parallel' ]
        allowed_parallel = [0, 1, 2 ]
        if not parallel in allowed_parallel :
            raise Exception(f"The parallel configuration should be either of {allowed_parallel}, got {parallel}.")


        verbosity = config[ 'verbosity' ]
        if not isinstance( verbosity, int) :
            raise Exception(f"The verbosity parameter should be an integer.")


        tolerance = config[ 'tolerance' ]
        if not isinstance( tolerance, float) :
            raise Exception(f"The tolerance parameter should be a float.")


        test_subcircuits = config[ 'test_subcircuits' ]
        if not isinstance( test_subcircuits, bool) :
            raise Exception(f"The test_subcircuits parameter should be a bool.")


        test_final_circuit = config[ 'test_final_circuit' ]
        if not isinstance( test_final_circuit, bool) :
            raise Exception(f"The test_final_circuit parameter should be a bool.")

 

        max_partition_size = config[ 'max_partition_size' ]
        if not isinstance( max_partition_size, int) :
            raise Exception(f"The max_partition_size parameter should be an integer.")

        self.config = config


        self.max_partition_size = max_partition_size



    def ConstructCircuitFromPartitions( self, circs: List[Circuit], parameter_arrs: List[List[np.ndarray]] ) -> Tuple[Circuit, np.ndarray]:
        """
        Call to construct the wide quantum circuit from the partitions.

    
        Args:

            circs ( List[Circuit] ) A list of Squander circuits to be compared

            parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

        Return:

            Returns with the constructed circuit and the corresponding parameter array

    
        """ 

        if not isinstance( circs, list ):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance( parameter_arrs, list ):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs) :
            raise Exception("The first two arguments should be of the same length")


        qbit_num = circs[0].get_Qbit_Num()



        wide_parameters = np.concatenate( parameter_arrs, axis=0 ) 


        wide_circuit = Circuit( qbit_num )

        for circ in circs:
            wide_circuit.add_Circuit( circ )


        assert wide_circuit.get_Parameter_Num() == wide_parameters.size, \
                f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"



        return wide_circuit, wide_parameters


    @staticmethod
    def DecomposePartition( Umtx: np.ndarray, config: dict, mini_topology = None, structure = None ) -> Circuit:
        """
        Call to run the decomposition of a given unitary Umtx, typically associated with the circuit 
        partition to be optimized

    
        Args:

            Umtx (np.ndarray) A complex typed unitary to be decomposed


        Return:

            Returns with the the decoposed circuit structure and with the corresponding gate parameters

    
        """ 
        strategy = config["strategy"]
        if strategy == "TreeSearch":
            cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
        elif strategy == "TabuSearch":
            cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology )
        elif strategy == "Adaptive":
            cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=1, topology=mini_topology )
        elif strategy == "TreeGuided":
            cDecompose = N_Qubit_Decomposition_Guided_Tree( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology )
        elif strategy == "Custom":
            cDecompose = N_Qubit_Decomposition_custom( Umtx.conj().T, config=config, accelerator_num=0 )
            assert structure is not None, "Custom decomposition strategy requires a gate structure to be provided."
            cDecompose.set_Gate_Structure( structure )
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")


        tolerance = config["tolerance"]
        cDecompose.set_Verbose( config["verbosity"] )
        cDecompose.set_Cost_Function_Variant( 3 )	
        cDecompose.set_Optimization_Tolerance( tolerance )
    

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer( "BFGS" )

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e: 
            #print(e)
            raise e
            #return []
        if not config.get("stop_first_solution", True): return cDecompose.all_solutions
        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
        assert parameters is not None


        if strategy == "Custom":
            err = cDecompose.Optimization_Problem(parameters)
            it = 0
            while err > tolerance and it < 20:
                cDecompose.set_Optimized_Parameters(np.random.rand(cDecompose.get_Parameter_Num())*(2*np.pi))
                cDecompose.Start_Decomposition()
                parameters       = cDecompose.get_Optimized_Parameters()
                err = cDecompose.Optimization_Problem(parameters)
                it += 1
            if err > tolerance or it != 0: print( "Decomposition error: ", err, it )
        else: err = cDecompose.get_Decomposition_Error()
        #print( "Decomposition error: ", err )
        if tolerance < err:
            #raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []


        return [(squander_circuit, parameters)]



    @staticmethod
    def CompareAndPickCircuits( circs: List[Circuit], parameter_arrs: List[List[np.ndarray]], metric : Callable[ [Circuit], int ] = CNOTGateCount ) -> Circuit:
        """
        Call to pick the most optimal circuit corresponding a specific metric. Looks for the circuit
        with the minimal metric value.

    
        Args:

            circs ( List[Circuit] ) A list of Squander circuits to be compared

            parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

            metric (optional) The metric function to decide which input circuit is better.


        Return:

            Returns with the chosen circuit and the corresponding parameter array

    
        """ 

        if not isinstance( circs, list ):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance( parameter_arrs, list ):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs) :
            raise Exception("The first two arguments should be of the same length")

        metrics = [metric( circ ) for circ in circs]

        metrics = np.array( metrics )

        min_idx = np.argmin( metrics )

        return circs[ min_idx ], parameter_arrs[ min_idx ]



    @staticmethod
    def PartitionDecompositionProcess( subcircuit: Circuit, subcircuit_parameters: np.ndarray, config: dict, structure=None ) -> Tuple[Circuit, np.ndarray]:
        """
        Implements an asynchronous process to decompose a unitary associated with a partition in a large 
        quantum circuit

    
        Args:

            circ ( Circuit ) A subcircuit representing a partition

            parameters ( np.ndarray ) A parameter array associated with the input circuit

        
        """             

        qbit_num_orig_circuit = subcircuit.get_Qbit_Num()

        involved_qbits = subcircuit.get_Qbits()

        qbit_num = len( involved_qbits )

        # create qbit map:
        qbit_map = {}
        for idx in range( len(involved_qbits) ):
            qbit_map[ involved_qbits[idx] ] = idx
        mini_topology = None 
        if config["topology"] is not None:
            mini_topology = extract_subtopology(involved_qbits, qbit_map, config)
        # remap the subcircuit to a smaller qubit register
        remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )

        if qbit_num > 3 and structure is None and config.get("strategy", "") == "TreeGuided":
            circo = Circuit(qbit_num)
            for gate in remapped_subcircuit.get_Gates(): circo.add_Gate(gate)
            remapped_subcircuit = circo
            partitioned_circuit, params, recombine_info = qgd_Wide_Circuit_Optimization.make_all_partition_circuit(remapped_subcircuit, subcircuit_parameters, 3)
            optimized_circuits = []
            subcircs = partitioned_circuit.get_Gates()
            #first find the optimal CNOT decomposition
            for innercirc in subcircs:
                start_idx = innercirc.get_Parameter_Start_Index()
                innercirc_parameters = params[ start_idx:start_idx+innercirc.get_Parameter_Num() ]
                callback_fnc = lambda  x : qgd_Wide_Circuit_Optimization.CompareAndPickCircuits( [innercirc, *(z[0] for z in x)], [innercirc_parameters, *(z[1] for z in x)] )
                optimized_circuits.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": True, 'tree_level_max': max(0, CNOTGateCount(subcircuit, 0)-1)}, structure=None)))
            parts, struct_idxs = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(remapped_subcircuit, [x[0] for x in optimized_circuits], params, recombine_info)
            #enumerate all solutions for each subcircuit in the optimal
            all_sol_for_idx = []
            for idx in struct_idxs:
                innercirc = subcircs[idx]
                start_idx = innercirc.get_Parameter_Start_Index()
                innercirc_parameters = params[ start_idx:start_idx+innercirc.get_Parameter_Num() ]
                callback_fnc = lambda  x : x + [(innercirc, innercirc_parameters)]
                all_sol_for_idx.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": False, 'tree_level_max': max(0, CNOTGateCount(subcircuit, 0))}, structure=None)))
            all_decomposed = []
            import itertools
            opt = qgd_Wide_Circuit_Optimization({**config, "max_partition_size": 3})
            if np.prod([len(x) for x in all_sol_for_idx]) > 32:
                import random
                trycombs = [[random.choice(x) for x in all_sol_for_idx] for _ in range(32)]
            else: trycombs = itertools.product(*all_sol_for_idx)
            for combination in trycombs:
                structures = [qgd_Wide_Circuit_Optimization.copy_circuit_structure(x[0]) for x in combination]
                optcirc, optparams = opt._OptimizeWideCircuit(remapped_subcircuit, subcircuit_parameters, False, parts, structures)
                reoptcirc, reoptparams = opt._OptimizeWideCircuit(optcirc.get_Flat_Circuit(), optparams)
                all_decomposed.append((reoptcirc.get_Flat_Circuit(), reoptparams))
        else:
            if not structure is None:
                structure = structure.Remap_Qbits( qbit_map, qbit_num )

            # get the unitary representing the circuit
            unitary = remapped_subcircuit.get_Matrix( subcircuit_parameters )

            # decompose a small unitary into a new circuit
            all_decomposed = qgd_Wide_Circuit_Optimization.DecomposePartition( unitary, config, mini_topology, structure=structure )
        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[ value ] = key
        result = []
        for decomposed_circuit, decomposed_parameters in all_decomposed:

            # remap the decomposed circuit in order to insert it into a large circuit
            new_subcircuit = decomposed_circuit.Remap_Qbits( inverse_qbit_map, qbit_num_orig_circuit )


            if config["test_subcircuits"]:
                CompareCircuits( subcircuit, subcircuit_parameters, new_subcircuit, decomposed_parameters, parallel=config["parallel"] )
            


            new_subcircuit = new_subcircuit.get_Flat_Circuit()
            result.append((new_subcircuit, decomposed_parameters))
        return result
    def build_partition_topo_deps(allparts):
        gate_to_parts = {}
        for i, part in enumerate(allparts):
            for gate in part:
                gate_to_parts.setdefault(gate, set()).add(i)
        g = {i: set() for i in range(len(allparts))}
        rg = {i: set() for i in range(len(allparts))}
        for i, part in enumerate(allparts):
            for gate in part:
                for other_part in gate_to_parts[gate]:
                    if other_part != i and (len(part & allparts[other_part]) > 0 and (len(part) < len(allparts[other_part])) or part < allparts[other_part]):
                        g[i].add(other_part)
                        rg[other_part].add(i)
        rg_ret = {i: set(rg[i]) for i in range(len(allparts))}        
        S = collections.deque(m for m in rg if len(rg[m]) == 0)
        L = []
        while S:
            n = S.popleft()
            L.append(n)
            for m in set(g[n]):
                g[n].remove(m)
                rg[m].remove(n)
                if len(rg[m]) == 0:
                    S.append(m)
        if len(L) != len(allparts):
            raise ValueError("Dependency graph is not a DAG")
        neworder = {old: new for new, old in enumerate(L)}
        rg_ret = {neworder[i]: set(neworder[j] for j in rg_ret[i]) for i in range(len(allparts))}
        return [allparts[i] for i in L], rg_ret #return partitions in dependency order and dependencies
    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, max_partition_size)
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}
        partitioned_circuit = Circuit( qbit_num_orig_circuit )
        params = []
        allparts, part_deps = qgd_Wide_Circuit_Optimization.build_partition_topo_deps(allparts)
        for part in allparts:
            surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded_chains))
            #topo sort part + surrounded chains
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates}, {x: rgo[x] & gates for x in gates}, gate_to_qubit):
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in chain:
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return partitioned_circuit, parameters, (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit), part_deps
    def strip_single_qubit_head_tails(circ, params):
        gate_dict, g, rg, gate_to_qubit, S = build_dependency(circ)
        newcirc = Circuit(circ.get_Qbit_Num())
        new_params = []
        for i in gate_dict:
            gate = gate_dict[i]
            if len(gate_to_qubit[i]) == 1 and (len(g[i]) == 0 or len(rg[i]) == 0):
                continue
            newcirc.add_Gate(gate)
            start_idx = gate.get_Parameter_Start_Index()
            new_params.append(params[start_idx:start_idx + gate.get_Parameter_Num()])
        return newcirc, np.empty((0,), dtype=np.float64) if len(new_params) == 0 else np.concatenate(new_params, axis=0)
    def get_fingerprint(circ, params):
        return tuple((gate.get_Name(), tuple(gate.get_Involved_Qbits())) for gate in circ.get_Gates()) + tuple(params)
    def recombine_all_partition_circuit(circ, optimized_subcircuits, optimized_parameter_list, recombine_info ):
        from squander.partitioning.ilp import topo_sort_partitions, ilp_global_optimal, recombine_single_qubit_chains
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = recombine_info
        max_gates = sum(sum(y for x, y in c.get_Gate_Nums().items() if x not in CNOT_COUNT_DICT) for c in optimized_subcircuits[:len(allparts)])        
        weights = [CNOTGateCount(circ, max_gates) for circ in optimized_subcircuits[:len(allparts)]]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L], fusion_info, surrounded_only=True)
        single_qubit_chain_idx = {frozenset(chain): idx + len(allparts) for idx, chain in enumerate(single_qubit_chains)}
        for extrapart in parts[len(struct_idxs):]:
            struct_idxs.append(single_qubit_chain_idx[extrapart])
        L = topo_sort_partitions(circ, parts)
        return [optimized_subcircuits[struct_idxs[i]] for i in L], [optimized_parameter_list[struct_idxs[i]] for i in L]

    def OptimizeWideCircuit( self, circ: Circuit, parameters: np.ndarray ) -> Tuple[Circuit, np.ndarray]:
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(circ, self.config['topology']):
            topo = self.config['topology']
            self.config['topology'] = None
            circ, parameters = self.OptimizeWideCircuit( circ, parameters )
            self.config['all_to_all_optimization_time'] = self.config['optimization_time']
            self.config['all_to_all_circuit'] = circ
            self.config['all_to_all_parameters'] = parameters
            self.config['topology'] = topo
            start_time = time.time()
            circ, parameters = self.route_circuit(circ, parameters)
            self.config['routing_time'] = time.time() - start_time
            self.config['routed_circuit'] = circ
            self.config['routed_parameters'] = parameters
        start_time = time.time()
        part_size_start = self.max_partition_size
        part_size_end = self.max_partition_size
        if self.config.get("use_osr", False) or self.config.get("use_graph_search", False): part_size_end = min(4, circ.get_Qbit_Num())
        count = CNOTGateCount(circ, 0)
        fingerprint_dict = {}
        for max_part_size in range(part_size_start, part_size_end + 1):
            # instantiate the object for optimizing wide circuits
            wide_circuit_optimizer = qgd_Wide_Circuit_Optimization( {**self.config, 'max_partition_size': max_part_size} )
            while True:
                # run circuit optimization
                circ_flat, parameters = wide_circuit_optimizer.InnerOptimizeWideCircuit( circ, parameters, fingerprint_dict=fingerprint_dict )
                circ = circ_flat.get_Flat_Circuit()
                newcount = CNOTGateCount(circ, 0)
                no_improve = newcount >= count
                count = newcount
                if no_improve: break
        self.config['optimization_time'] = time.time() - start_time
        return circ, parameters
    def InnerOptimizeWideCircuit( self, circ: Circuit, orig_parameters: np.ndarray, fingerprint_dict=None ) -> Tuple[Circuit, np.ndarray]:
        """
        Call to optimize a wide circuit (i.e. circuits with many qubits) by
        partitioning the circuit into smaller partitions and redecompose the smaller partitions


        Args: 

            circ ( Circuit ) A circuit to be partitioned

            orig_parameters ( np.ndarray ) A parameter array associated with the input circuit

        Return:

            Returns with the optimized circuit and the corresponding parameter array

        """
        from squander.utils import circuit_to_CNOT_basis
        circ, orig_parameters = circuit_to_CNOT_basis(circ, orig_parameters)
        max_gates = sum(y for x, y in circ.get_Gate_Nums().items() if x not in CNOT_COUNT_DICT)
        
        global_min = self.config.get("global_min", True)
        if global_min:
            partitioned_circuit, parameters, recombine_info, part_deps = qgd_Wide_Circuit_Optimization.make_all_partition_circuit(circ, orig_parameters, self.max_partition_size)            

        else:
            partitioned_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy=self.config['partition_strategy'] )
            part_deps = None

        subcircuits = partitioned_circuit.get_Gates()

        #subcircuits = subcircuits[9:10]

        if parent_process() is None: print(len(subcircuits), "partitions found to optimize")


        # the list of optimized subcircuits
        optimized_subcircuits = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list = [None] * len(subcircuits)

        # list of AsyncResult objects
        async_results = [None] * len(subcircuits)

        total_opt = [0]

        def process_result(partition_idx):
            if optimized_subcircuits[partition_idx] is not None: return
            subcircuit = subcircuits[ partition_idx ]
            # callback function done on the master process to compare the new decomposed and the original suncircuit
            start_idx = subcircuit.get_Parameter_Start_Index()
            subcircuit_parameters = parameters[ start_idx:start_idx + subcircuit.get_Parameter_Num() ]
            fingerprint = None if fingerprint_dict is None else qgd_Wide_Circuit_Optimization.get_fingerprint(subcircuit, subcircuit_parameters)
            callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, *(z[0] for z in x)], [subcircuit_parameters, *(z[1] for z in x)], lambda c: CNOTGateCount(c, max_gates) )
            if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
            else:
                new_subcircuit, new_parameters = callback_fnc(async_results[partition_idx][0](*async_results[partition_idx][1])
                    if parent_process() is not None else async_results[partition_idx].get( timeout = None ))

                if subcircuit != new_subcircuit:
                    print( "original subcircuit:    ", subcircuit.get_Gate_Nums(), partition_idx) 
                    print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
                if fingerprint_dict is not None:
                    fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                    fingerprint_dict[qgd_Wide_Circuit_Optimization.get_fingerprint(new_subcircuit, new_parameters)] = (new_subcircuit, new_parameters)
                    trim_subcirc, trim_parameters = qgd_Wide_Circuit_Optimization.strip_single_qubit_head_tails(new_subcircuit, new_parameters)
                    fingerprint_dict[qgd_Wide_Circuit_Optimization.get_fingerprint(trim_subcirc, trim_parameters)] = (trim_subcirc, trim_parameters)
            if total_opt[0] % 100 == 99: print(total_opt[0]+1, "partitions optimized")
            total_opt[0] += 1
            optimized_subcircuits[ partition_idx ] = new_subcircuit
            optimized_parameter_list[ partition_idx ] = new_parameters
        with (contextlib.nullcontext() if parent_process() is not None else Pool(processes=mp.cpu_count())) as pool:
            remaining = list(range(len(subcircuits)))
            while remaining:
                still_remaining = []
                #  code for iterate over partitions and optimize them
                for partition_idx in remaining:
                    subcircuit = subcircuits[partition_idx]

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx   = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[ start_idx:end_idx ]

                    fingerprint = None if fingerprint_dict is None else qgd_Wide_Circuit_Optimization.get_fingerprint(subcircuit, subcircuit_parameters)
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                        optimized_subcircuits[ partition_idx ], optimized_parameter_list[ partition_idx ] = fingerprint_dict[fingerprint]
                        continue
                    if part_deps is not None and partition_idx in part_deps:
                        any_optimized, any_remaining = False, False
                        for dep_idx in part_deps[partition_idx]:
                            if optimized_subcircuits[dep_idx] is None and (async_results[dep_idx] is None or not async_results[dep_idx].ready()):
                                any_remaining = True
                                continue
                            elif optimized_subcircuits[dep_idx] is None:
                                process_result(dep_idx)
                            if CNOTGateCount(optimized_subcircuits[dep_idx]) < CNOTGateCount(subcircuits[dep_idx]): #if the dependency partition was optimized, skip
                                any_optimized = True
                                break
                        if any_optimized:
                            optimized_subcircuits[ partition_idx ] = subcircuit
                            optimized_parameter_list[ partition_idx ] = subcircuit_parameters
                            continue
                        if any_remaining:
                            still_remaining.append(partition_idx)
                            continue
                    # call a process to decompose a subcircuit
                    config = {**self.config, 'tree_level_max': max(0, CNOTGateCount(subcircuit, 0)-1)}
                    fargs = (self.PartitionDecompositionProcess, (subcircuit, subcircuit_parameters, config, None))
                    #print("Dispatching", subcircuit.get_Involved_Qubits(), "qubits with", CNOGateCount(subcircuit, 0), "CNOT gates, partition ", partition_idx)
                    async_results[partition_idx] = fargs if parent_process() is not None else pool.apply_async(*fargs)
                if len(remaining) == len(still_remaining):
                    time.sleep(0.1)
                remaining = still_remaining
            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx in range( len(subcircuits ) ):
                process_result(partition_idx)


        # construct the wide circuit from the optimized suncircuits
        if global_min:
            optimized_subcircuits, optimized_parameter_list = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(circ, optimized_subcircuits, optimized_parameter_list, recombine_info)
        
        wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )

        if parent_process() is None:
            print( "original circuit:    ", circ.get_Gate_Nums()) 
            print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 

        qgd_Wide_Circuit_Optimization.check_valid_routing(wide_circuit, self.config["topology"])

        self.check_compare_circuits( circ, orig_parameters, wide_circuit, wide_parameters )

        return wide_circuit, wide_parameters

    def all_to_all_topology(num_qubits):
        return [(i, j) for i in range(num_qubits) for j in range(i+1, num_qubits)]
    def linear_topology(num_qubits):
        return [(i, i+1) for i in range(num_qubits - 1)]
    def star_topology(num_qubits):
        return [(0, i) for i in range(1, num_qubits)]
    def ring_topology(num_qubits):
        return [(i, (i+1) % num_qubits) for i in range(num_qubits)]
    def lattice_topology(x_qbits, y_qbits):
        return [(i*x_qbits + j, i*x_qbits + (j+1)) for i in range(y_qbits) for j in range(x_qbits - 1)] + [(i*x_qbits + j, (i+1)*x_qbits + j) for i in range(y_qbits - 1) for j in range(x_qbits)]
    def heavy_hexagonal_topology(rows, cols):
        """
        Finite heavy-hex patch.

        rows, cols describe the underlying honeycomb 'brick-wall' patch.
        The first rows*cols qubits are the original honeycomb vertices.
        Every original edge gets one inserted degree-2 qubit.

        Returns:
            list[(u, v)]  undirected couplers
        """

        def vid(r, c):
            return r * cols + c

        # Underlying honeycomb / brick-wall edges
        base_edges = []

        for r in range(rows):
            for c in range(cols):
                # Vertical brick-wall edges
                if r + 1 < rows:
                    base_edges.append((vid(r, c), vid(r + 1, c)))

                # Alternating horizontal edges
                if c + 1 < cols and ((r + c) % 2 == 0):
                    base_edges.append((vid(r, c), vid(r, c + 1)))

        # Subdivide every honeycomb edge by inserting a qubit
        next_id = rows * cols
        heavy_edges = []

        for u, v in base_edges:
            w = next_id
            next_id += 1
            heavy_edges.append((u, w))
            heavy_edges.append((w, v))

        return heavy_edges
    def sycamore_topology(): return qgd_Wide_Circuit_Optimization.lattice_topology(6, 9) #there is a defective qubit at (0, 3) in the sycamore chip, but we ignore it here for simplicity
    def is_valid_routing(wide_circuit, topo):
        if topo is None: return True
        import itertools
        topo_set = {frozenset(edge) for edge in topo}
        def qubits_connected(qubits):
            if len(qubits) <= 1: return True
            edges = {frozenset((q1, q2)) for q1, q2 in itertools.combinations(qubits, 2) if frozenset((q1, q2)) in topo_set}
            if len(edges) == 0: return False
            cur_set = set(edges.pop())
            while edges:
                next_edge = next((e for e in edges if len(e & cur_set) > 0), None)
                if next_edge is None: return False
                cur_set |= next_edge
                edges.remove(next_edge)
            return True
        return all(qubits_connected(gate.get_Involved_Qbits())
            for gate in wide_circuit.get_Flat_Circuit().get_Gates()
            if len(gate.get_Involved_Qbits()) > 1)
    def check_valid_routing(wide_circuit, topo):
        assert qgd_Wide_Circuit_Optimization.is_valid_routing(wide_circuit, topo), "Final circuit contains gates that do not respect the routing constraints."
    def check_compare_circuits(self, circ, orig_parameters, wide_circuit, wide_parameters, routing=False):
        if self.config["test_final_circuit"]:
            if routing:
                CompareCircuits(circ, orig_parameters, wide_circuit, wide_parameters,
                    initial_mapping=self.config["initial_mapping"], final_mapping=self.config["final_mapping"], parallel=0 )
            else:
                CompareCircuits(circ, orig_parameters, wide_circuit, wide_parameters)
    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):
        if self.config.get("use_bqskit_routing", False):
            from squander import Qiskit_IO
            from bqskit import Circuit as BQSKitCircuit, compile
            from bqskit.compiler import Compiler
            from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow
            from bqskit.compiler.basepass import BasePass
            class SquanderPartitioner(BasePass):
                def __init__(self, max_partition_size):
                    super().__init__()
                    self.max_partition_size = max_partition_size
                async def run(self, circuit: BQSKitCircuit, data=None):
                    from squander import Qiskit_IO
                    from squander.partitioning.partition import PartitionCircuit
                    circ_qiskit = QuantumCircuit.from_qasm_str(OPENQASM2Language().encode(circuit))
                    circ, orig_parameters = Qiskit_IO.convert_Qiskit_to_Squander(circ_qiskit)
                    partitioned_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy="ilp" )
                    partitioned_circuit_qiskit = Qiskit_IO.get_Qiskit_Circuit(partitioned_circuit, parameters)
                    partitioned_circuit_bqskit = OPENQASM2Language().decode(qasm2.dumps(partitioned_circuit_qiskit))
                    circuit.become(partitioned_circuit_bqskit, False)
            from bqskit.passes import (
                GeneralizedSabreLayoutPass,
                GeneralizedSabreRoutingPass,
                SetModelPass,
                IfThenElsePass,
                QuickPartitioner
            )
            from bqskit.ir.gates import CNOTGate  # example; extend as needed
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            # Build BQSKit machine model from your topology
            model        = MachineModel(circ.get_Qbit_Num(), self.config["topology"])

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo        = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)
            
            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))
            # Customizable knobs

            # Routing-only pass pipeline — NO optimization passes
            mainflow = build_seqpam_mapping_optimization_workflow(block_size=self.config['max_partition_size'])
            for curpass in mainflow._passes:
                if isinstance(curpass, IfThenElsePass):
                    for i in range(len(curpass.on_true._passes)):
                        if isinstance(curpass.on_true._passes[i], QuickPartitioner):
                            curpass.on_true._passes[i] = SquanderPartitioner(self.config['max_partition_size'])
            
            routing_workflow = [
                SetModelPass(model),                          # attach hardware model to circuit
                build_seqpam_mapping_optimization_workflow()
                #GeneralizedSabreLayoutPass(),                # SABRE-style layout
                #GeneralizedSabreRoutingPass(),               # SABRE-style routing
            ]

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(bqskit_circ, routing_workflow, True)

            # Convert back: BQSKit → Qiskit → Squander
            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(OPENQASM2Language().encode(routed_bqskit_circ))
            Squander_remapped_circuit, parameters_remapped_circuit = \
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            Squander_remapped_circuit = Squander_remapped_circuit.Remap_Qbits({i: j for i, j in enumerate(pass_data.placement)})
            self.config["initial_mapping"] = list(pass_data.placement[x] for x in pass_data.initial_mapping)
            self.config["final_mapping"] = list(pass_data.placement[x] for x in pass_data.final_mapping)
        elif self.config.get("use_qiskit_sabre", False):
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.transpiler.passes import SabreLayout, SabreSwap
            from qiskit.transpiler import PassManager, CouplingMap
            from squander.gates import gates_Wrapper as gate
            #SUPPORTED_GATES_NAMES = {n.lower().replace("cnot", "cx") for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")}
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)
            coupling_map = [[i,j] for i,j in self.config['topology']]
            #circuit_qiskit_sabre = transpile(circo, basis_gates=SUPPORTED_GATES_NAMES, coupling_map=coupling_map, optimization_level=0)
            coupling_map = CouplingMap(coupling_map)
            # Customizable SABRE parameters
            sabre_seed     = self.config.get("sabre_seed", 42)
            sabre_trials   = self.config.get("sabre_trials", 5)   # layout trials
            swap_trials    = self.config.get("sabre_swap_trials", sabre_trials)
            heuristic      = self.config.get("sabre_heuristic", "decay")  # "basic" | "lookahead" | "decay"

            layout_pass = SabreLayout(
                coupling_map,
                seed=sabre_seed,
                max_iterations=sabre_trials,
                swap_trials=swap_trials,
            )
            swap_pass = SabreSwap(
                coupling_map,
                heuristic=heuristic,
                seed=sabre_seed,
                trials=swap_trials,
            )

            pm = PassManager([
                layout_pass,          # find initial qubit mapping via SABRE
                swap_pass,            # insert SWAP gates for routing
            ])            
            circuit_qiskit_sabre = pm.run(circo)
            Squander_remapped_circuit, parameters_remapped_circuit = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_sabre)
            self.config["initial_mapping"] = circuit_qiskit_sabre.layout.initial_index_layout()
            self.config["final_mapping"] = circuit_qiskit_sabre.layout.final_index_layout()
        else:
            sabre = SABRE(circ, self.config["topology"])
            Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(orig_parameters)
            self.config["initial_mapping"] = pi
            self.config["final_mapping"] = final_pi
        qgd_Wide_Circuit_Optimization.check_valid_routing(Squander_remapped_circuit, self.config["topology"])
        self.check_compare_circuits(circ, orig_parameters, Squander_remapped_circuit, parameters_remapped_circuit, routing=True)
        return Squander_remapped_circuit, parameters_remapped_circuit
