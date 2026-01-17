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

from typing import List, Callable

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config ):
    """
    Extract a subtopology from the full topology based on involved qubits.
    
    Args:
        involved_qbits: Set or list of qubit indices that are involved in the partition
        qbit_map: Dictionary mapping original qubit indices to remapped indices
        config: Configuration dictionary containing the "topology" key with full topology edges
        
    Returns:
        List of tuples representing edges in the subtopology, with qubits remapped according to qbit_map
    """
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology

def CNOTGateCount( circ: Circuit ) -> int :
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

    return gate_counts.get('CNOT', 0) #+  3*gate_counts.get('SWAP', 0)



class N_Qubit_Decomposition_Guided_Tree(N_Qubit_Decomposition_custom):
    """
    Guided tree search decomposition for quantum gate synthesis.
    
    This class implements a tree search algorithm for decomposing unitaries into
    CNOT gates, using operator Schmidt rank (OSR) entanglement as a guiding heuristic.
    """
    def __init__(self, Umtx, config, accelerator_num, topology):
        """
        Initialize the guided tree decomposition.
        
        Args:
            Umtx: Unitary matrix to decompose (already conjugate transposed)
            config: Configuration dictionary for decomposition parameters
            accelerator_num: Number of accelerator devices to use
            topology: List of tuples representing allowed CNOT connections.
                     If None, creates a fully connected topology.
        """
        super().__init__(Umtx, config=config, accelerator_num=accelerator_num)
        self.Umtx = Umtx #already conjugate transposed
        self.qbit_num = Umtx.shape[0].bit_length() -1
        self.config = config
        self.accelerator_num = accelerator_num
        #self.set_Cost_Function_Variant( 0 )	 #0 is Frobenius, 3 is HS, 10 is OSR
        if topology is None:
            topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        self.topology = topology
    @staticmethod
    def enumerate_unordered_cnot_BFS(n: int, topology=None, use_gl=True):
        """
        Enumerate all reachable states in GL(n,2) using breadth-first search.
        
        This generator yields states in increasing CNOT depth, exploring the space
        of all possible CNOT gate sequences.
        
        Args:
            n: Number of qubits
            topology: List of allowed CNOT pairs. If None, uses fully connected topology.
            use_gl: If True, uses GL(n,2) group structure. If False, uses explicit sequences.
            
        Yields:
            Tuples of (visited states, sequence pairs, directed sequences, results)
            where results contain (state, unordered_pairs, directed_pairs) for each discovered state.
        """
        # Precompute unordered pairs
        topology = [(i, j) for i in range(n) for j in range(i+1, n)] if topology is None else topology
        prior_level_info = None
        while True:
            visited, seq_pairs_of, seq_dir_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(n, topology, prior_level_info, use_gl=use_gl)
            if not res: break
            yield res
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res)))
    @staticmethod
    def canonical_prefix_ok(seq):
        """
        Check if a sequence of CNOT pairs forms a canonical prefix.
        
        Uses Kahn's algorithm to verify topological ordering. A canonical prefix
        means the sequence can be topologically sorted in its natural order.
        
        Args:
            seq: List of unordered CNOT pairs (tuples of qubit indices)
            
        Returns:
            -1 if the sequence is canonical, otherwise returns the position
            where the canonical ordering is violated.
        """
        m = len(seq)
        if m <= 1: return -1
        succ = {}
        indeg = {}
        last_on = {}
        for k in range(m):
            for q in seq[k]:
                if q in last_on:
                    p = last_on[q]
                    succ.setdefault(p, set()).add(k)
                    indeg[k] = indeg.get(k, 0) + 1
                last_on[q] = k
        import heapq
        pq = [(seq[x], x) for x in range(m) if indeg.get(x, 0) == 0]
        heapq.heapify(pq)
        for pos in range(m):
            # Kahn's algorithm
            if len(pq) == 0: return pos #malformed (shouldn't happen)
            u = heapq.heappop(pq)
            if u[1] != pos: return pos #deviation: not canonical
            for v in succ.get(u[1], ()):
                indeg[v] -= 1
                if indeg[v] == 0: heapq.heappush(pq, (seq[v], v))
        return -1
    @staticmethod
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

            assert topology is not None, "Topology is required at this point"
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
    @staticmethod
    def build_sequence(stop=5, ordered=True, use_gl=True):
        """
        Build and print statistics about CNOT sequences for different qubit counts.
        
        This is a utility function for analyzing the number of distinct CNOT sequences
        at different depths. See OEIS sequence A002884 for reference.
        
        Args:
            stop: Maximum number of qubits to analyze (default: 5)
            ordered: If True, counts ordered sequences. If False, counts unordered sets.
            use_gl: If True, uses GL(n,2) group structure. If False, uses explicit sequences.
        """
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
    @staticmethod
    def extract_bits(x, pos):
        """
        Extract specific bit positions from an integer and pack them into a new integer.
        
        Args:
            x: Integer to extract bits from
            pos: List of bit positions to extract
            
        Returns:
            Integer with extracted bits packed into positions 0, 1, 2, ...
        """
        return sum(((x >> p) & 1) << i for i, p in enumerate(pos))
    @staticmethod
    def build_osr_matrix(U, n, A):
        """
        Build the operator Schmidt rank (OSR) matrix for a given bipartition.
        
        Reshapes the unitary matrix U according to the bipartition A|B to compute
        the operator Schmidt decomposition.
        
        Args:
            U: Unitary matrix of shape (2^n, 2^n)
            n: Number of qubits
            A: List of qubit indices on side A of the bipartition
            
        Returns:
            Reshaped matrix of shape (2^|A| * 2^|A|, 2^|B| * 2^|B|) for OSR analysis
        """
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        dA = 1 << len(A)
        dB = 1 << len(B)
        return U.reshape([2]*(2*n)).transpose(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)).reshape(dA*dA, dB*dB)
    @staticmethod
    def accumulate_grad_for_cut(U, G, Umat, VTmat, n, A):
        """
        Accumulate gradient contributions for a specific bipartition cut.
        
        Reconstructs the gradient contribution from the dyadic decomposition
        and accumulates it into the full gradient matrix U.
        
        Args:
            U: Gradient matrix to accumulate into (modified in place)
            G: Gradient coefficients for dyadic components
            Umat: Left singular vectors from SVD
            VTmat: Right singular vectors (conjugate transpose) from SVD
            n: Number of qubits
            A: List of qubit indices on side A of the bipartition
        """
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        dyadic_idxs = [1 << k for k in range(len(G))]
        mat = np.array(G) * Umat[:,dyadic_idxs] @ VTmat[dyadic_idxs,:]  # reconstruct U from its dyadic decomposition
        revmap = [None]*(2*n)
        for i, x in enumerate(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)):
            revmap[x] = i
        U += mat.reshape([2]*(2*n)).transpose(tuple(revmap)).reshape(*U.shape)
        return U
    @staticmethod
    def trace_out_qubits(U, n, A):
        """
        Trace out qubits not in set A, returning the reduced unitary.
        
        Args:
            U: Unitary matrix of shape (2^n, 2^n)
            n: Number of qubits
            A: List of qubit indices to keep (others are traced out)
            
        Returns:
            Reduced unitary matrix of shape (2^|A|, 2^|A|)
        """
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A)
        M = np.linalg.svd(M, compute_uv=True, full_matrices=False)[0][:,0].reshape(1<<len(A), 1<<len(A))
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(M)
    @staticmethod
    def numerical_rank_osr(M, Fnorm, tol=1e-10):
        """
        Compute the numerical rank of an operator Schmidt rank matrix.
        
        Args:
            M: Matrix to analyze
            Fnorm: Frobenius norm for normalization
            tol: Tolerance for determining rank (relative to largest singular value)
            
        Returns:
            Tuple of (rank, singular_values) where rank is the number of
            singular values above the tolerance threshold.
        """
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False) / Fnorm
        #print(s)
        return int(np.sum(s >= s[0]*tol)), s
    @staticmethod
    def operator_schmidt_rank(U, n, A, Fnorm, tol=1e-10):
        """
        Compute the operator Schmidt rank for a bipartition of a unitary.
        
        Args:
            U: Unitary matrix of shape (2^n, 2^n)
            n: Number of qubits
            A: List of qubit indices on side A of the bipartition
            Fnorm: Frobenius norm for normalization
            tol: Tolerance for determining rank
            
        Returns:
            Tuple of (rank, singular_values) from the operator Schmidt decomposition
        """
        return N_Qubit_Decomposition_Guided_Tree.numerical_rank_osr(N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A), Fnorm, tol)
    @staticmethod
    def unique_cuts(n):
        """
        Generate all unique nontrivial bipartitions of n qubits.
        
        Yields unordered bipartitions, avoiding duplicates where a cut and its
        complement would be the same. For even n, uses lexicographic ordering
        to break ties.
        
        Args:
            n: Number of qubits
            
        Yields:
            Tuples of qubit indices representing one side of each unique bipartition
        """
        import itertools
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
        """
        Construct a circuit from a sequence of CNOT pairs.
        
        Each pair is implemented as U3 gates on both qubits followed by a CNOT.
        Optionally adds finalizing U3 gates on all qubits.
        
        Args:
            pairs: List of tuples (q1, q2) representing CNOT gates
            finalizing: If True, adds U3 gates on all qubits at the end
            
        Returns:
            Circuit object representing the gate sequence
        """
        circ = Circuit(self.qbit_num)
        for pair in pairs:
            circ.add_U3(pair[0])
            circ.add_U3(pair[1])
            circ.add_CNOT(pair[0], pair[1])
        if finalizing:
            for qbit in range(self.qbit_num):
                circ.add_U3(qbit)
        return circ
    @staticmethod
    def ceil_log2(x): return 0 if x == 0 else (x-1).bit_length()
    @staticmethod
    def logsumexp_smoothmax(Lc, tau=1e-2):
        """
        Compute a smooth maximum using log-sum-exp with temperature parameter.
        
        This is a differentiable approximation to the maximum function.
        
        Args:
            Lc: List of values to take the smooth maximum of
            tau: Temperature parameter (smaller = sharper, closer to true max)
            
        Returns:
            Smooth maximum value: tau * log(sum(exp((Lc - m)/tau))) + m
        """
        m = max(Lc)
        sum = 0.0
        for v in Lc: sum += np.exp((v - m)/tau)
        return tau * np.log(sum) + m
    @staticmethod
    def dyadic_loss(S, max_dyadic, rho=0.9, tol=1e-4):
        """
        Compute dyadic loss for singular values, emphasizing lower-rank structure.
        
        The loss function penalizes deviations from ideal dyadic (power-of-2) structure
        in the singular values, with exponentially decreasing weights.
        
        Args:
            S: Array of singular values
            max_dyadic: Maximum dyadic level to consider
            rho: Exponential decay factor for weights (default: 0.9)
            tol: Tolerance threshold for deviations
            
        Returns:
            Scalar loss value
        """
        tot_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(len(S))
        w = 1.0
        acc = 0.0
        for k in range(max_dyadic-1, -1, -1):
            if k < tot_dyadic:
                val = S[1 << k] - S[0] * tol    
                acc += w * val * val
            w *= rho
        return acc
    @staticmethod
    def avg_loss(cuts_S, rho=0.9):
        """
        Compute average dyadic loss across multiple cuts.
        
        Args:
            cuts_S: List of singular value arrays, one per cut
            rho: Exponential decay factor for dyadic loss weights
            
        Returns:
            Average loss across all cuts
        """
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.dyadic_loss(S, max_dyadic, rho)
        return total_loss / len(cuts_S)
    @staticmethod
    def cuts_softmax_dyadic_cost(cuts_S, rho=0.1, tau=1e-2):
        """
        Compute aggregated cost over cuts using softmax of per-cut dyadic losses.
        
        Uses log-sum-exp (smooth maximum) to aggregate losses across cuts,
        emphasizing the worst-performing cuts.
        
        Args:
            cuts_S: List of singular value arrays, one per cut
            rho: Exponential decay factor for dyadic loss weights
            tau: Temperature parameter for softmax (must be > 0)
            
        Returns:
            Aggregated cost value
        """
        if tau <= 0.0: raise RuntimeError("tau must be > 0")
        Lc = []
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.dyadic_loss(S, max_dyadic, rho))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    @staticmethod
    def dyadic_loss_grad_diag(S, max_dyadic, Fnorm, rho=0.1, tol=1e-4):
        """
        Compute gradient of dyadic loss with respect to singular values.
        
        Only dyadic positions (1, 2, 4, 8, ...) receive nonzero gradient entries.
        
        Args:
            S: Array of singular values
            max_dyadic: Maximum dyadic level to consider
            Fnorm: Frobenius norm for normalization
            rho: Exponential decay factor for weights
            tol: Tolerance threshold
            
        Returns:
            List of gradient values (only dyadic positions are nonzero)
        """
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
    @staticmethod
    def cuts_avg_dyadic_grad(cuts_S, Fnorm, rho=0.1):
        """
        Compute average gradient across cuts for dyadic loss.
        
        Args:
            cuts_S: List of singular value arrays, one per cut
            Fnorm: Frobenius norm for normalization
            rho: Exponential decay factor for weights
            
        Returns:
            List of gradient arrays, one per cut
        """
        C = len(cuts_S)
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(max(len(S) for S in cuts_S))
        Lc = []
        for c in range(C):
            Lc.append(N_Qubit_Decomposition_Guided_Tree.dyadic_loss_grad_diag(cuts_S[c], max_dyadic, Fnorm * C, rho))
        return Lc
    @staticmethod
    def cuts_softmax_tail_grad(cuts_S, Fnorm, rho=0.1, tau=1e-2):
        """
        Compute gradient with softmax weighting across cuts.
        
        Combines per-cut gradients with softmax weights, emphasizing cuts
        with higher losses. Only dyadic positions receive nonzero gradients.
        
        Args:
            cuts_S: List of singular value arrays, one per cut
            Fnorm: Frobenius norm for normalization
            rho: Exponential decay factor for dyadic loss weights
            tau: Temperature parameter for softmax
            
        Returns:
            List of weighted gradient arrays, one per cut
        """
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

    @staticmethod
    def top_k_triplet_for_cut(U, q, A, Fnorm):
        """
        Compute top-k SVD triplet for operator Schmidt rank matrix of a cut.
        
        Builds the OSR matrix for bipartition A|B, performs SVD, and returns
        the normalized singular values and corresponding singular vectors.
        
        Args:
            U: Unitary matrix of shape (2^q, 2^q), row-major
            q: Number of qubits
            A: List of qubit indices on side A of the bipartition
            Fnorm: Frobenius norm for normalization (typically sqrt(2^q))
            
        Returns:
            Tuple of (normalized_singular_values, U_matrix, Vh_matrix) from SVD
        """
        # 1) Build M for this cut    
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, q, A)
        k = min(M.shape)

        # 2) SVD: M = U * diag(S) * VT  (VT = V^H)
        # Row-major API handles leading dims as col counts.
        res = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        return res.S / Fnorm, res.U, res.Vh # normalized singular value
    @staticmethod
    def get_deriv_osr_entanglement(matrix, use_cuts, use_softmax):
        """
        Compute derivative of operator Schmidt rank entanglement cost function.
        
        Computes gradients with respect to the unitary matrix by aggregating
        contributions from all bipartition cuts.
        
        Args:
            matrix: Unitary matrix to compute derivative for
            use_cuts: List of specific cuts to use. If empty, uses all unique cuts.
            use_softmax: If True, uses softmax aggregation. If False, uses average.
            
        Returns:
            Gradient matrix of same shape as input matrix
        """
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
        if use_softmax: allS = N_Qubit_Decomposition_Guided_Tree.cuts_softmax_tail_grad(allS, Fnorm, 1.0)
        else: allS = N_Qubit_Decomposition_Guided_Tree.cuts_avg_dyadic_grad(allS, Fnorm, 0.9)
        for i in range(len(cuts)):
            triplets[i] = (allS[i], triplets[i][1], triplets[i][2])
        for i in range(len(cuts)):
            G, Umat, VTmat = triplets[i]
            N_Qubit_Decomposition_Guided_Tree.accumulate_grad_for_cut(deriv, G, Umat, VTmat, qbit_num, cuts[i])
        return deriv
    @staticmethod
    def real_trace_conj_dot(A, B):
        """
        Compute real part of trace of conjugate transpose dot product.
        
        Computes Re(Tr(A^† B)) = Re(Tr(A^H B)) for gradient computation.
        Equivalent to sum of element-wise real parts of A^† * B.
        
        Args:
            A: Matrix A (dL/dU in gradient context)
            B: Matrix B (dU/dθ in gradient context)
            
        Returns:
            Real scalar value: Re(Tr(A^† B))
        """
        return np.sum(A.real * B.real + A.imag * B.imag) # Re Tr(A^† B)
    @staticmethod
    def param_derivs(circ, Umtx, x):
        """
        Compute parameter derivatives of a circuit with respect to gate parameters.
        
        Uses finite differences: π/2 shifts for rotation angles, symmetric differences
        for phase parameters.
        
        Args:
            circ: Circuit object
            Umtx: Base unitary matrix
            x: Parameter array
            
        Returns:
            List of derivative matrices, one per parameter
        """
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
    @staticmethod
    def get_all_clifford(qbit_num, qbits):
        """
        Generate all Clifford circuits on specified qubits.
        
        Constructs all possible Clifford circuits by combining H and S gates
        in all possible ways on the specified qubits.
        
        Args:
            qbit_num: Total number of qubits in the circuit
            qbits: List of qubit indices to apply Clifford gates to
            
        Returns:
            List of Circuit objects, each representing a different Clifford circuit
        """
        import itertools
        circuits = []
        all_clifford = [x+y for x, y in itertools.product(((), (True,), (False, True), (False, False, True), (False, False, False, True), (True, False, True)), ((), (False,), (False, False), (False, False)))]
        for clifford_idxs in itertools.product(range(len(all_clifford)), repeat=len(qbits)):
            circ = Circuit(qbit_num)
            for clifford_idx, qbit in zip(clifford_idxs, qbits):
                for sh in all_clifford[clifford_idx]:
                    if sh:
                        circ.add_H(qbit)
                    else:
                        circ.add_S(qbit)
            circuits.append(circ)
        return circuits
    
    @staticmethod
    def _global_phase_fix(U):
        """
        Remove global phase from a unitary matrix.
        
        Normalizes the matrix so that det(U) = 1 by dividing by the
        n-th root of the determinant.
        
        Args:
            U: Unitary matrix
            
        Returns:
            Phase-normalized unitary matrix with det = 1
        """
        return U / (np.linalg.det(U)**(1/len(U)))
    @staticmethod
    def _polar_unitary(X):
        """
        Compute the closest unitary matrix to X using polar decomposition.
        
        Uses SVD to find U such that U is unitary and closest to X in Frobenius norm.
        
        Args:
            X: Input matrix
            
        Returns:
            Unitary matrix U from polar decomposition: X = U * P where P is positive semidefinite
        """
        U,_,Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh

    @staticmethod
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

    @staticmethod
    def _A_from_c(c1,c2,c3):
        """
        Construct a 2-qubit unitary from Weyl chamber coordinates.
        
        Builds exp(-i/2 * (c1*XX + c2*YY + c3*ZZ)) where XX, YY, ZZ are
        tensor products of Pauli matrices.
        
        Args:
            c1: Coefficient for XX term
            c2: Coefficient for YY term
            c3: Coefficient for ZZ term
            
        Returns:
            4x4 unitary matrix representing the 2-qubit gate
        """
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
    def factor_local(K):
        """
        Factor a 2-qubit gate into a tensor product of 1-qubit gates.
        
        Uses SVD to decompose a 4x4 matrix into K1 ⊗ K2 where K1 and K2
        are 2x2 matrices (up to global phase).
        
        Args:
            K: 4x4 matrix representing a 2-qubit gate
            
        Returns:
            Tuple of (K1, K2) where each is a 2x2 unitary matrix
        """
        # reshape to (2,2,2,2), SVD the (a,c ; b,d) unfolding
        M = K.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
        U,_,Vh = np.linalg.svd(M, full_matrices=False)
        A = U[:,0].reshape(2,2); B = Vh.conj().T[:,0].reshape(2,2)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A), N_Qubit_Decomposition_Guided_Tree._polar_unitary(B)
    def _magic_basis_plusYY():
        """
        Return the complex magic basis matrix.
        
        This is a specific basis used in 2-qubit gate decomposition that
        diagonalizes the Weyl chamber representation.
        
        Returns:
            4x4 complex matrix representing the magic basis
        """
        # Complex magic basis (matches A(c)=exp(-i/2*(c1 XX + c2 YY + c3 ZZ)) below)
        # Columns are (|Φ+>, i|Φ->, i|Ψ+>, |Ψ->) up to harmless phases
        return (1/np.sqrt(2))*np.array([
            [1, 0, 0,  1j],
            [0, 1j,1,  0 ],
            [0, 1j,-1, 0 ],
            [1j,0, 0, -1 ]
        ], dtype=complex)

    def _project_to_SO4(O):
        """
        Project a matrix to the nearest special orthogonal matrix (SO(4)).
        
        Finds the closest real orthogonal matrix with determinant +1.
        
        Args:
            O: Input matrix (should be close to orthogonal)
            
        Returns:
            Real orthogonal matrix with det = +1, closest to O in Frobenius norm
        """
        # nearest real orthogonal with det=+1
        O = np.real_if_close(O, tol=1e5)
        U, _, Vt = np.linalg.svd(O)
        O = U @ Vt
        if np.linalg.det(O) < 0:
            O[:,0] *= -1
        return O

    def _clean_col_phases(W):
        """
        Remove arbitrary phases from columns of a matrix.
        
        Normalizes each column by removing the phase of its largest-magnitude element,
        making the matrix representation more canonical.
        
        Args:
            W: Input matrix
            
        Returns:
            Matrix with cleaned column phases
        """
        Wc = W.copy()
        for j in range(Wc.shape[1]):
            col = Wc[:, j]
            k = np.argmax(np.abs(col))
            if np.abs(col[k]) > 1e-14:
                Wc[:, j] *= np.exp(-1j * np.angle(col[k]))
        return Wc

    def closest_local_product(W4):
        """
        Find the closest tensor product of 1-qubit gates to a 2-qubit gate.
        
        Args:
            W4: 4x4 matrix representing a 2-qubit gate
            
        Returns:
            Tuple of (A, B) where A ⊗ B is the closest local product to W4
        """
        A, B = N_Qubit_Decomposition_Guided_Tree.factor_local(W4)
        return N_Qubit_Decomposition_Guided_Tree._global_phase_fix(A), N_Qubit_Decomposition_Guided_Tree._global_phase_fix(B)
    def kak_u3s_around_cx(U, n, c, t, iters=3):
        """
        Decompose a 2-qubit gate using KAK decomposition with U3 gates around CNOT.
        
        Uses Qiskit's TwoQubitWeylDecomposition to extract Weyl chamber coordinates,
        then converts to U3 gate parameters.
        
        Args:
            U: 2^n x 2^n unitary matrix
            n: Total number of qubits
            c: Control qubit index
            t: Target qubit index
            iters: Number of iterations (currently unused)
            
        Returns:
            Dictionary with keys:
            - "c": (c1, c2, c3) Weyl chamber coordinates
            - "pre": {"A": (theta, phi, lam), "B": (theta, phi, lam)} pre-CNOT U3 parameters
            - "post": {"A": (theta, phi, lam), "B": (theta, phi, lam)} post-CNOT U3 parameters
        """
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
        """
        Convert circuit parameters to unitary matrix.
        
        Applies the circuit with given parameters to the base unitary matrix.
        
        Args:
            params: Parameter array for the circuit gates
            
        Returns:
            Unitary matrix resulting from applying the circuit with these parameters
        """
        U = self.Umtx.copy()
        self.get_Circuit().apply_to(params, U)
        return U
    def OSR_with_local_alignment(self, pairs, cuts, Fnorm, tol):
        """
        Optimize circuit parameters to minimize operator Schmidt rank entanglement.
        
        Uses the OSR cost function (variant 10) to align local gates and minimize
        entanglement across all specified cuts. Performs optimization using
        basinhopping algorithm.
        
        Args:
            pairs: List of CNOT pairs defining the gate structure
            cuts: List of bipartitions to optimize over
            Fnorm: Frobenius norm for normalization
            tol: Tolerance for numerical rank computation
            
        Returns:
            List of tuples (rank, singular_values) for each cut, where rank is
            the ceiling of log2 of the numerical rank
        """
        def ceil_log2(x): return 0 if x == 0 else (x-1).bit_length()
        if len(pairs) != 0:
            self.set_Cost_Function_Variant( 10 )
            #self.Run_Decomposition(pairs, False)
            self.set_Gate_Structure(self.get_circuit_from_pairs(pairs, False))
            import scipy
            #def cost(x):
            #    U = self.Umtx.copy()
            #    self.get_Circuit().apply_to(x, U)
            #    S = [(ceil_log2(rank), s) for cut in cuts for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol),)]
            #    return sum(1-s[0]*s[0] for r,s in S) / len(S)
            """
            bestparams = []
            for cut in cuts:
                def cost(x):
                    U = self.Umtx.copy()
                    self.get_Circuit().apply_to(x, U)
                    rank, s = N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol)
                    acc, w = 0.0, 1.0
                    for i in range(len(s).bit_length()-1):
                    #for i in range(len(s).bit_length()-1-1, -1, -1):
                        val = s[1<<i] - tol/10 * s[0]
                        acc += val*val * w
                        w *= 0.1
                    return acc
                #self.set_Optimized_Parameters(np.random.rand(self.get_Parameter_Num())*(2*np.pi))
                #bestparams.append(scipy.optimize.basinhopping(cost, self.get_Optimized_Parameters(), niter=50, stepsize=2*np.pi/100).x)
                #bestparams.append(scipy.optimize.dual_annealing(cost, [ (0, 2*np.pi) for _ in range(self.get_Parameter_Num()) ], maxiter=100).x)
                bestparams.append(scipy.optimize.minimize(cost, self.get_Optimized_Parameters(), method='BFGS', options={'maxiter': 1000, 'disp': False}).x)
                #print([(ceil_log2(rank), s) for cut in cuts for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(self.params_to_mat(bestparams[-1]), self.qbit_num, cut, Fnorm, tol),)])
            return [(ceil_log2(rank), s) for cut, params in zip(cuts, bestparams) for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(self.params_to_mat(params), self.qbit_num, cut, Fnorm, tol),)]
            """
            #best = scipy.optimize.differential_evolution(self.Optimization_Problem, [ (0, 2*np.pi) for _ in range(self.get_Parameter_Num()) ], maxiter=100, polish=True)
            #best = scipy.optimize.dual_annealing(self.Optimization_Problem, [ (0, 2*np.pi) for _ in range(self.get_Parameter_Num()) ], maxiter=100)
            best = scipy.optimize.basinhopping(self.Optimization_Problem, np.random.rand(self.get_Parameter_Num())*(2*np.pi), niter=50, stepsize=2*np.pi/100)
            #print(best)
            self.set_Cost_Function_Variant( 3 )
            self.set_Optimized_Parameters(best.x)
            U = self.Umtx.copy()
            self.get_Circuit().apply_to(self.get_Optimized_Parameters(), U)
        else: U = self.Umtx
        return [(ceil_log2(rank), s) for cut in cuts for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol),)]
    def Run_Decomposition(self, pairs, finalizing=True):
        """
        Run decomposition for a given sequence of CNOT pairs.
        
        Constructs the circuit, sets random initial parameters, and runs
        the decomposition optimization.
        
        Args:
            pairs: List of CNOT pairs (tuples of qubit indices)
            finalizing: If True, adds finalizing U3 gates on all qubits
            
        Returns:
            True if decomposition error is below tolerance, False otherwise
        """
        circ = self.get_circuit_from_pairs(pairs, finalizing)
        self.set_Gate_Structure(circ)
        self.set_Optimized_Parameters(np.random.rand(self.get_Parameter_Num())*(2*np.pi))
        super().Start_Decomposition()
        if finalizing:
            params = self.get_Optimized_Parameters()
            self.err = self.Optimization_Problem(params)
            return self.err < self.config.get('tolerance', 1e-8)
    def Start_Decomposition(self):
        """
        Start the guided tree search decomposition algorithm.
        
        Performs breadth-first search over CNOT sequences, using operator
        Schmidt rank as a heuristic to guide the search. Explores sequences
        in increasing depth, pruning based on remaining depth and entanglement.
        """
        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
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
                h = self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol)
                min_cnots = max((x[0] for x in h), default=0)
                #print(path, h, remaining, min_cnots)
                if min_cnots == 0:
                    #print(path)
                    for i in range(10):
                        if self.Run_Decomposition(path):
                            self.all_solutions.append((self.get_Circuit(), self.get_Optimized_Parameters()))
                            if stop_first_solution or len(path) <= 1: return
                            break
                        #print("Looping", h)
                if min_cnots > remaining: continue
                if not curh is None:
                    #print(path, [(h[i], curh[i]) for i in check_cuts])
                    #if any(h[i][0] > curh[i][0] for i in check_cuts): continue
                    if max((x[0] for x in curh), default=0) < min_cnots: continue
                nextprefixes.append((path, h))
            nextprefixes.sort(key=lambda t: (max((x[0] for x in t[1]), default=0), sum(x[0] for x in t[1]), sum(1-x[1][0]*x[1][0] for x in t[1])))
            prefixes = {x[0]: x[1] for x in nextprefixes[:B]}
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res) if tuple(x[1]) in prefixes))
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        #print("No decomposition found within the given CNOT limit.")
    def get_Decomposition_Error(self):
        """
        Get the decomposition error from the last optimization.
        
        Returns:
            Decomposition error (Frobenius norm of difference)
        """
        return self.err
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
        config.setdefault('routed', False)
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



    def ConstructCircuitFromPartitions( self, circs: List[Circuit], parameter_arrs: [List[np.ndarray]] ) -> (Circuit, np.ndarray):
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
        cDecompose.set_Optimizer( "BFGS2" )

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e: 
            raise e
            return []
        if not config.get("stop_first_solution", True): return cDecompose.all_solutions
        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
        assert parameters is not None


        if strategy == "Custom": err = cDecompose.Optimization_Problem(parameters)
        else: err = cDecompose.get_Decomposition_Error()
        #print( "Decomposition error: ", err )
        if tolerance < err:
            #raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []


        return [(squander_circuit, parameters)]



    @staticmethod
    def CompareAndPickCircuits( circs: List[Circuit], parameter_arrs: [List[np.ndarray]], metric : Callable[ [Circuit], int ] = CNOTGateCount ) -> Circuit:
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
    def PartitionDecompositionProcess( subcircuit: Circuit, subcircuit_parameters: np.ndarray, config: dict, structure=None ) -> (Circuit, np.ndarray):
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
        if config["topology"] != None:
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
                optimized_circuits.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": True, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}, structure=None)))
            parts, struct_idxs = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(remapped_subcircuit, 3, [x[0] for x in optimized_circuits], recombine_info)
            #enumerate all solutions for each subcircuit in the optimal
            all_sol_for_idx = []
            for idx in struct_idxs:
                innercirc = subcircs[idx]
                start_idx = innercirc.get_Parameter_Start_Index()
                innercirc_parameters = params[ start_idx:start_idx+innercirc.get_Parameter_Num() ]
                callback_fnc = lambda  x : x + [(innercirc, innercirc_parameters)]
                all_sol_for_idx.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": False, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0))}, structure=None)))
            all_decomposed = []
            opt = qgd_Wide_Circuit_Optimization({**config, "max_partition_size": 3})
            if np.prod([len(x) for x in all_sol_for_idx]) > 32:
                import random
                trycombs = [[random.choice(x) for x in all_sol_for_idx] for _ in range(32)]
            else: trycombs = itertools.product(*all_sol_for_idx)
            for combination in trycombs:
                structures = [qgd_Wide_Circuit_Optimization.copy_circuit_structure(x[0]) for x in combination]
                optcirc, optparams = opt.OptimizeWideCircuit(remapped_subcircuit, subcircuit_parameters, False, parts, structures)
                reoptcirc, reoptparams = opt.OptimizeWideCircuit(optcirc.get_Flat_Circuit(), optparams)
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

    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        """
        Partition a circuit into all possible subcircuits up to max_partition_size.
        
        Uses integer linear programming to find optimal partitions and handles
        single-qubit gate chains separately.
        
        Args:
            circ: Circuit to partition
            orig_parameters: Original parameter array
            max_partition_size: Maximum number of qubits per partition
            
        Returns:
            Tuple of (partitioned_circuit, parameters, recombine_info) where
            recombine_info contains metadata needed for recombination
        """
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, max_partition_size)
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}
        partitined_circuit = Circuit( qbit_num_orig_circuit )
        params = []
        for part in allparts:
            surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded_chains))
            #topo sort part + surrounded chains
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates}, {x: rgo[x] & gates for x in gates}):
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitined_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in chain:
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitined_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return partitined_circuit, parameters, (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit)
    def copy_circuit_structure(structure):
        """
        Copy a circuit structure, converting CNOT gates to U3-CNOT-U3 sequences.
        
        Creates a new circuit with the same structure but with U3 gates
        added around each CNOT and finalizing U3 gates on all qubits.
        
        Args:
            structure: Circuit to copy
            
        Returns:
            New circuit with U3 gates added around CNOTs
        """
        from squander.gates.qgd_Circuit import CNOT
        newcirc = Circuit(structure.get_Qbit_Num())
        for gate in structure.get_Gates():
            if isinstance(gate, CNOT):
                newcirc.add_U3(gate.get_Target_Qbit())
                newcirc.add_U3(gate.get_Control_Qbit())
                newcirc.add_Gate(gate)
        for qbit in structure.get_Qbits():
            newcirc.add_U3(qbit)
        return newcirc
    def recombine_all_partition_circuit(circ, max_partition_size, optimized_subcircuits, recombine_info):
        """
        Recombine optimized subcircuits back into a full circuit.
        
        Uses ILP to find optimal combination of subcircuits and recombines
        single-qubit gate chains. Returns the topologically sorted result.
        
        Args:
            circ: Original circuit
            max_partition_size: Maximum partition size used
            optimized_subcircuits: List of optimized subcircuits
            recombine_info: Metadata from make_all_partition_circuit
            
        Returns:
            Tuple of (parts, struct_idxs) where parts are the recombined partitions
            and struct_idxs are indices into optimized_subcircuits
        """
        from squander.partitioning.ilp import topo_sort_partitions, ilp_global_optimal, recombine_single_qubit_chains
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = recombine_info
        max_gates = max(sum(y for x, y in c.get_Gate_Nums().items() if x !='CNOT') for c in optimized_subcircuits)
        def to_cost(c): return CNOTGateCount(c)*max_gates + sum(y for x, y in c.get_Gate_Nums().items() if x !='CNOT')
        weights = [to_cost(circ) for circ in optimized_subcircuits[:len(allparts)]]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L], fusion_info)
        single_qubit_chain_idx = {frozenset(chain): idx + len(allparts) for idx, chain in enumerate(single_qubit_chains)}
        for extrapart in parts[len(struct_idxs):]:
            struct_idxs.append(single_qubit_chain_idx[extrapart])
        L = topo_sort_partitions(circ, max_partition_size, parts)
        return [parts[i] for i in L], [struct_idxs[i] for i in L]

    def OptimizeWideCircuit( self, circ: Circuit, orig_parameters: np.ndarray, global_min=True, prepartitioning=None, structures=None ) -> (Circuit, np.ndarray):
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
        if self.config["topology"] != None and self.config["routed"]==False:
            circ, orig_parameters = self.route_circuit(circ,orig_parameters)

        if global_min:
            partitined_circuit, parameters, recombine_info = qgd_Wide_Circuit_Optimization.make_all_partition_circuit(circ, orig_parameters, self.max_partition_size)

        elif prepartitioning is not None:
            from squander.partitioning.kahn import kahn_partition_preparts
            from squander.partitioning.tools import translate_param_order
            partitined_circuit, param_order, _ = kahn_partition_preparts(circ, self.max_partition_size, prepartitioning)
            parameters = translate_param_order(orig_parameters, param_order)
        else:
            partitined_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy=self.config['partition_strategy'] )

        qbit_num_orig_circuit = circ.get_Qbit_Num()


        subcircuits = partitined_circuit.get_Gates()

        #subcircuits = subcircuits[9:10]

        if parent_process() is None: print(len(subcircuits), "partitions found to optimize")


        # the list of optimized subcircuits
        optimized_subcircuits = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list = [None] * len(subcircuits)

        if parent_process() is not None:
            #  code for iterate over partitions and optimize them
            for partition_idx, subcircuit in enumerate( subcircuits ):
        

                # isolate the parameters corresponding to the given sub-circuit
                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = start_idx + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
    
        
            
                # callback function done on the master process to compare the new decomposed and the original suncircuit
                callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, *(z[0] for z in x)], [subcircuit_parameters, *(z[1] for z in x)] )

                # call a process to decompose a subcircuit
                config = {**self.config, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}
                config = config if structures is None or partition_idx >= len(structures) else {**config, 'strategy': 'Custom', 'max_inner_iterations': 10000, 'max_iteration_loops': 4}
                new_subcircuit, new_parameters = callback_fnc(self.PartitionDecompositionProcess( subcircuit, subcircuit_parameters, config,
                                                                                     None if structures is None or partition_idx >= len(structures) else structures[partition_idx] ))
                if subcircuit != new_subcircuit:

                    print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
                    print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 

                if partition_idx % 100 == 99: print(partition_idx+1, "partitions optimized")
                optimized_subcircuits[ partition_idx ] = new_subcircuit
                optimized_parameter_list[ partition_idx ] = new_parameters
        else:
            # list of AsyncResult objects
            async_results = [None] * len(subcircuits)
            with Pool(processes=mp.cpu_count()) as pool:

                #  code for iterate over partitions and optimize them
                for partition_idx, subcircuit in enumerate( subcircuits ):
            

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx   = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[ start_idx:end_idx ]
    
        
                
                    # call a process to decompose a subcircuit
                    config = {**self.config, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}
                    config = config if structures is None or partition_idx >= len(structures) else {**config, 'strategy': 'Custom', 'max_inner_iterations': 10000, 'max_iteration_loops': 4}
                    async_results[partition_idx]  = pool.apply_async( self.PartitionDecompositionProcess, (subcircuit, subcircuit_parameters, config,
                                                                                                        None if structures is None or partition_idx >= len(structures) else structures[partition_idx]))
                #  code for iterate over async results and retrieve the new subcircuits
                for partition_idx, subcircuit in enumerate( subcircuits ):
                    # callback function done on the master process to compare the new decomposed and the original suncircuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    subcircuit_parameters = parameters[ start_idx:start_idx + subcircuit.get_Parameter_Num() ]
                    callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, *(z[0] for z in x)], [subcircuit_parameters, *(z[1] for z in x)] )
                    new_subcircuit, new_parameters = callback_fnc(async_results[partition_idx].get( timeout = None ))

                    if subcircuit != new_subcircuit:

                        print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
                        print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
                    if partition_idx % 100 == 99: print(partition_idx+1, "partitions optimized")
                    optimized_subcircuits[ partition_idx ] = new_subcircuit
                    optimized_parameter_list[ partition_idx ] = new_parameters


        # construct the wide circuit from the optimized suncircuits
        if global_min:
             parts, struct_idxs = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(circ, self.max_partition_size, optimized_subcircuits, recombine_info)
             structures = [qgd_Wide_Circuit_Optimization.copy_circuit_structure(optimized_subcircuits[x]) for x in struct_idxs]
             return self.OptimizeWideCircuit(circ, orig_parameters, global_min=False, prepartitioning=parts, structures=structures)
        else:
            wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )

        if parent_process() is None:
            print( "original circuit:    ", circ.get_Gate_Nums()) 
            print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 


        if self.config["test_final_circuit"]:
            CompareCircuits( partitined_circuit, parameters, wide_circuit, wide_parameters )

        
        return wide_circuit, wide_parameters

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):
        """
        Route a circuit to match hardware topology using SABRE algorithm.
        
        Maps logical qubits to physical qubits and inserts SWAP gates as needed
        to satisfy connectivity constraints.
        
        Args:
            circ: Circuit to route
            orig_parameters: Original parameter array
            
        Returns:
            Tuple of (routed_circuit, routed_parameters) with qubits mapped
            to physical topology
        """
        sabre = SABRE(circ, self.config["topology"])
        Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(orig_parameters)
        self.config.setdefault("initial_mapping",pi)
        self.config.setdefault("final_mapping",final_pi)
        self.config["routed"] = True
        return Squander_remapped_circuit, parameters_remapped_circuit
