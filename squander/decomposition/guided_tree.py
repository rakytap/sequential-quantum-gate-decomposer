"""Experimental guided tree search for symbolic decomposition benchmarks.

This module is intentionally kept under benchmarks/partitioning rather than
production decomposition code. It supports sympy_gen/sympy_param experiments.
"""

from squander import N_Qubit_Decomposition_custom
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

import numpy as np

from typing import Any, Dict, List, Optional, Set, Tuple, Union


SQUANDER_FLOAT64_TOLERANCE = 1e-10
SQUANDER_FLOAT32_TOLERANCE = 1e-5


def _config_uses_float32(config):
    return bool(config.get("use_float", False))


def _default_squander_tolerance(config):
    return (
        SQUANDER_FLOAT32_TOLERANCE
        if _config_uses_float32(config)
        else SQUANDER_FLOAT64_TOLERANCE
    )


class N_Qubit_Decomposition_Guided_Tree(N_Qubit_Decomposition_custom):
    """Tree-guided multi-qubit decomposition using operator Schmidt rank (OSR) style costs."""

    def __init__(
        self, Umtx, config, accelerator_num, topology, paramspace=None, paramscale=None
    ):
        """Initialize guided tree search over a unitary (or list of unitaries) and hardware topology.

        Args:
            Umtx: Complex unitary matrix, or list of such matrices (already conjugate-transposed per caller).
            config: Decomposition / search configuration dict.
            accelerator_num: Number of accelerators for the base decomposer.
            topology: List of undirected coupler pairs ``(i, j)``; default is all-to-all.
            paramspace: Optional per-parameter affine scaling space for ``params_to_mat``.
            paramscale: Optional scaling denominators paired with ``paramspace``.
        """
        super().__init__(
            Umtx[0] if isinstance(Umtx, list) else Umtx,
            config=config,
            accelerator_num=accelerator_num,
        )
        self.Umtx = (
            Umtx if isinstance(Umtx, list) else [Umtx]
        )  # already conjugate transposed
        self.qbit_num = self.Umtx[0].shape[0].bit_length() - 1
        self.config = config
        self.accelerator_num = accelerator_num
        self.paramspace = paramspace
        self.paramscale = () if paramscale is None else paramscale
        # self.set_Cost_Function_Variant( 0 )	 #0 is Frobenius, 3 is HS, 10 is OSR
        if topology is None:
            topology = [
                (i, j)
                for i in range(self.qbit_num)
                for j in range(i + 1, self.qbit_num)
            ]
        self.topology = topology

    @staticmethod
    def enumerate_unordered_cnot_BFS(n: int, topology=None, use_gl=True):
        """Yield successive BFS levels of CNOT-reachable GL(n,2) states (see ``enumerate_unordered_cnot_BFS_level``).

        Args:
            n: Number of qubits.
            topology: Allowed unordered CNOT pairs; default all pairs.
            use_gl: If True, use GL-style column updates; else restricted enumeration.

        Yields:
            Each level's list of ``(state_key, seq_pairs, seq_directed)`` discoveries.
        """
        # Precompute unordered pairs
        topology = (
            [(i, j) for i in range(n) for j in range(i + 1, n)]
            if topology is None
            else topology
        )
        prior_level_info: Union[tuple[Any, Any, Any, Any], None] = None
        while True:
            visited, seq_pairs_of, seq_dir_of, res = (
                N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(
                    n, topology, prior_level_info, use_gl=use_gl
                )
            )
            if not res:
                break
            yield res
            prior_level_info = (
                visited,
                seq_pairs_of,
                seq_dir_of,
                list(x[0] for x in reversed(res)),
            )

    @staticmethod
    def canonical_prefix_ok(seq):
        """Check whether a sequence of unordered pair steps has a canonical topological order.

        Returns:
            ``-1`` if the prefix is OK; otherwise the first index where canonical order fails.
        """
        m = len(seq)
        if m <= 1:
            return -1
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
            if len(pq) == 0:
                return pos  # malformed (shouldn't happen)
            u = heapq.heappop(pq)
            if u[1] != pos:
                return pos  # deviation: not canonical
            for v in succ.get(u[1], ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(pq, (seq[v], v))
        return -1

    @staticmethod
    def enumerate_unordered_cnot_BFS_level(
        n: int,
        topology: Optional[List[Tuple[int, int]]] = None,
        prior_level_info: Optional[
            Tuple[
                Set[Tuple[int, ...]],
                Dict[Tuple[int, ...], List[Tuple[int, int]]],
                Dict[Tuple[int, ...], List[Tuple[int, int]]],
                List[
                    Tuple[Tuple[int, ...], List[Tuple[int, int]], List[Tuple[int, int]]]
                ],
            ]
        ] = None,
        use_gl=True,
    ):
        """Enumerate GL(n,2) states at the next BFS depth from ``prior_level_info``.

        Moves are *recorded* as unordered pairs (structure view); each expansion
        may try both CNOT directions internally when ``use_gl`` is True.

        Returns:
            Tuple ``(visited, seq_pairs_of, seq_dir_of, res)`` where ``res`` is a
            list of ``(A, seq_pairs, seq_directed)`` for newly discovered states
            ``A``: ``seq_pairs`` is the unordered-pair history; ``seq_directed`` is
            a consistent directed realization. On the first call, pass
            ``prior_level_info=None`` to obtain the root state only.
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
            assert topology is not None
            for p in topology:
                if not use_gl:
                    if len(last_pairs) >= 3 and all(p == x for x in last_pairs[-3:]):
                        continue  # avoid more than 3 repeated CNOTs
                    if (
                        N_Qubit_Decomposition_Guided_Tree.canonical_prefix_ok(
                            last_pairs + [p]
                        )
                        >= 0
                    ):
                        continue  # not canonical prefix
                # Try both directions, but record the *same* unordered step 'p'
                for mv in (p, (p[1], p[0])) if use_gl else (p,):
                    # CNOT left
                    if use_gl:
                        if mv[0] == mv[1]:
                            B = A
                        else:
                            B = list(A)
                            B[mv[1]] ^= B[mv[0]]
                            B = tuple(B)

                        if B in visited:
                            continue  # already discovered at minimal depth
                    else:
                        B = tuple(last_dirs + [p])

                    visited.add(B)
                    new_seq_pairs_of[B] = last_pairs + [p]
                    new_seq_dir_of[B] = last_dirs + [mv]

                    # Emit as soon as we discover the state (BFS → minimal depth)
                    res.append((B, new_seq_pairs_of[B], new_seq_dir_of[B]))
        return visited, new_seq_pairs_of, new_seq_dir_of, res

    @staticmethod
    def build_sequence(stop: int = 5, ordered: bool = True, use_gl: bool = True):
        """Debug helper: print distribution of minimal CNOT sequence lengths by qubit count (up to ``stop``).

        See OEIS A002884 for related enumeration context. Not used in production optimization paths.
        """
        # https://oeis.org/A002884
        # unordered sequence: 1, 1, 4, 88, 9556, 4526605
        # unordered at 5 qubits: {0: 1, 1: 10, 2: 85, 3: 650, 4: 4475, 5: 27375, 6: 142499, 7: 580482, 8: 1501297, 9: 1738232, 10: 517884, 11: 13591, 12: 24}
        for i in range(2, stop + 1):
            d = {}
            for z in N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(
                i, use_gl=use_gl
            ):
                for x in (list if ordered else set)(tuple(x[1]) for x in z):
                    d[len(x)] = d.get(len(x), 0) + 1
                if not use_gl and len(d) > 5:
                    break
            print({x: d[x] for x in sorted(d)}, sum(d.values()))

    @staticmethod
    def extract_bits(x, pos):
        """Pack bits of integer ``x`` at positions ``pos`` into a smaller integer (LSB-first order)."""
        return sum(((x >> p) & 1) << i for i, p in enumerate(pos))

    @staticmethod
    def build_osr_matrix(U, n, A):
        """Reshape unitary ``U`` (size ``2^n``) into the OSR matrix for bipartition ``A`` vs complement.

        Args:
            U: Flattened ``2^n x 2^n`` unitary (row-major).
            n: Qubit count.
            A: Tuple of qubit indices on subsystem A.

        Returns:
            Matrix of shape ``(2^{|A|})^2 x (2^{|B|})^2`` for Schmidt analysis.
        """
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n - 1 - q for q in A], [n - 1 - q for q in B]
        dA = 1 << len(A)
        dB = 1 << len(B)
        return (
            U.reshape([2] * (2 * n))
            .transpose(
                tuple(A) + tuple(t + n for t in A) + tuple(B) + tuple(t + n for t in B)
            )
            .reshape(dA * dA, dB * dB)
        )

    @staticmethod
    def accumulate_grad_for_cut(U, G, Umat, VTmat, n, A):  # qubits on A
        """Accumulate gradient ``G * Umat @ VTmat`` from an SVD triplet back into full ``U`` layout for cut ``A``."""
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n - 1 - q for q in A], [n - 1 - q for q in B]
        mat = np.array(G) * Umat @ VTmat  # reconstruct U from its dyadic decomposition
        revmap = [None] * (2 * n)
        for i, x in enumerate(
            tuple(A) + tuple(t + n for t in A) + tuple(B) + tuple(t + n for t in B)
        ):
            revmap[x] = i
        U += mat.reshape([2] * (2 * n)).transpose(tuple(revmap)).reshape(*U.shape)
        return U

    @staticmethod
    def trace_out_qubits(U, n, A):
        """Trace out complement of subsystem ``A`` and return a unitary polar factor on ``A`` (2^{|A|} x 2^{|A|})."""
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A)
        M = np.linalg.svd(M, compute_uv=True, full_matrices=False)[0][:, 0].reshape(
            1 << len(A), 1 << len(A)
        )
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(M)

    @staticmethod
    def numerical_rank_osr(M, Fnorm, tol=1e-10):
        """Count singular values of ``M/Fnorm`` above ``tol`` relative to the largest; returns ``(rank, s)``."""
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False) / Fnorm
        # print(s)
        return int(np.sum(s >= s[0] * tol)), s

    @staticmethod
    def operator_schmidt_rank(U, n, A, Fnorm, tol=1e-10):
        """Operator Schmidt rank of ``U`` across cut ``A`` (via OSR matrix), using ``numerical_rank_osr``."""
        return N_Qubit_Decomposition_Guided_Tree.numerical_rank_osr(
            N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A), Fnorm, tol
        )

    @staticmethod
    def unique_cuts(n):
        """Yield all nontrivial unordered bipartitions of ``n`` qubits (each complement pair once)."""
        import itertools

        qubits = tuple(range(n))
        for r in range(1, n // 2 + 1):  # only up to half
            for S in itertools.combinations(qubits, r):
                if r < n - r:
                    yield S
                else:  # r == n-r (only possible when n even): tie-break
                    comp = tuple(q for q in qubits if q not in S)
                    if S < comp:  # lexicographically smaller tuple wins
                        yield S

    def get_circuit_from_pairs(self, pairs, finalizing=True):
        """Build a layer of U3–U3–CNOT per pair, optionally followed by trailing U3 on every qubit."""
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
    def ceil_log2(x):
        """Ceiling of log2 for nonnegative integer ``x``; ``0`` maps to ``0``."""
        return 0 if x == 0 else (x - 1).bit_length()

    @staticmethod
    def logsumexp_smoothmax(Lc, tau=1e-2):
        """Smooth maximum of list ``Lc``: ``tau * log(sum exp(v/tau)) + max``, stable implementation."""
        if not Lc:
            return 0.0
        if tau <= 0.0:
            raise RuntimeError("tau must be > 0")
        m = max(Lc)
        acc = 0.0
        for v in Lc:
            acc += np.exp((v - m) / tau)
        return tau * np.log(acc) + m

    @staticmethod
    def dyadic_loss(S, max_dyadic, rho=0.9, tol=1e-4):
        """Weighted loss on dyadic singular-value indices (powers of two) of normalized spectrum ``S``."""
        tot_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(len(S))
        w = 1.0
        acc = 0.0
        for k in range(max_dyadic - 1, -1, -1):
            if k < tot_dyadic:
                val = S[1 << k] - S[0] * tol
                acc += w * val * val
            w *= rho
        return acc

    @staticmethod
    def avg_loss(cuts_S, rho=0.9):
        """Average ``dyadic_loss`` over a list of singular-value spectra ``cuts_S``."""
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(
            max(len(S) for S in cuts_S)
        )
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.dyadic_loss(
                S, max_dyadic, rho
            )
        return total_loss / len(cuts_S)

    # Aggregated cost over cuts: softmax (log-sum-exp) of per-cut dyadic losses
    @staticmethod
    def cuts_softmax_dyadic_cost(cuts_S, rho=0.1, tau=1e-2):
        """Log-sum-exp aggregate of per-cut dyadic losses (temperature ``tau``)."""
        if tau <= 0.0:
            raise RuntimeError("tau must be > 0")
        Lc = []
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(
            max(len(S) for S in cuts_S)
        )
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.dyadic_loss(S, max_dyadic, rho))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    # Gradient w.r.t. the singular values (diagonal of dL/dΣ):
    @staticmethod
    def dyadic_loss_grad_diag(S, max_dyadic, Fnorm, rho=0.1, tol=1e-4):
        """Diagonal gradient of ``dyadic_loss`` w.r.t. singular values (dyadic indices only)."""
        n = len(S)
        # c_k = rho^k / Mk  for k=1..n-1, then prefix sum C_j = sum_{k=1}^j c_k
        tot_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(n)
        grad = [0.0] * tot_dyadic
        w = 1.0
        for k in range(max_dyadic - 1, -1, -1):
            if k < tot_dyadic:
                idx = 1 << k
                grad[k] = (
                    2.0 * w * S[idx] * (1.0 - tol) / Fnorm
                )  # 1-tol not needed if using stop-grad
            w *= rho  # w = rho^k
        return grad

    @staticmethod
    def cuts_avg_dyadic_grad(cuts_S, Fnorm, rho=0.1):
        """Per-cut gradients for the average dyadic loss (list parallel to ``cuts_S``)."""
        C = len(cuts_S)
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(
            max(len(S) for S in cuts_S)
        )
        Lc = []
        for c in range(C):
            Lc.append(
                N_Qubit_Decomposition_Guided_Tree.dyadic_loss_grad_diag(
                    cuts_S[c], max_dyadic, Fnorm * C, rho
                )
            )
        return Lc

    # Gradient w.r.t. singular values (same length as S).
    # Only dyadic positions (1,2,4,...) get nonzero entries; others are 0.
    @staticmethod
    def cuts_softmax_tail_grad(cuts_S, Fnorm, rho=0.1, tau=1e-2):
        """Gradient of softmax-of-dyadic-losses w.r.t. each cut's singular values."""
        C = len(cuts_S)
        if C == 0:
            return []
        max_dyadic = N_Qubit_Decomposition_Guided_Tree.ceil_log2(
            max(len(S) for S in cuts_S)
        )
        # 1) per-cut losses
        Lc = [
            N_Qubit_Decomposition_Guided_Tree.dyadic_loss(cuts_S[c], max_dyadic, rho)
            for c in range(C)
        ]

        # 2) softmax weights w_c = exp((Lc - m)/tau) / Z
        m = max(Lc)
        w = [np.exp((Lc[c] - m) / tau) for c in range(C)]
        Z = np.sum(w)
        for c in range(C):
            w[c] /= Z if Z > 0.0 else 1.0

        # 3) dL/dS^{(c)} = w_c * dL_c/dS^{(c)}
        return [
            [
                v * w[c]
                for v in N_Qubit_Decomposition_Guided_Tree.dyadic_loss_grad_diag(
                    cuts_S[c], max_dyadic, Fnorm, rho
                )
            ]
            for c in range(C)
        ]

    @staticmethod
    def loss_for_rank(S, rank):
        """Sum of squares of singular values from index ``2**rank`` onward (tail beyond target rank)."""
        start = 1 << rank
        if start >= len(S):
            return 0.0
        return sum(x * x for x in S[start:])

    @staticmethod
    def avg_loss_for_rank(cuts_S, rank):
        """Average ``loss_for_rank`` over cuts."""
        if not cuts_S:
            return 0.0
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank)
        return total_loss / len(cuts_S)

    # Aggregated cost over cuts: softmax (log-sum-exp) of per-cut dyadic losses
    @staticmethod
    def cuts_softmax_rank_cost(cuts_S, rank, tau=1e-2):
        """Softmax aggregate of per-cut ``loss_for_rank`` (temperature ``tau``)."""
        Lc = []
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    # Gradient w.r.t. the singular values (diagonal of dL/dΣ):
    @staticmethod
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

    @staticmethod
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
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(
                S, rank, Fnorm
            )
            out.append([scale * v for v in g])
        return out

    # Gradient w.r.t. singular values (same length as S).
    @staticmethod
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

        Lc = [N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank) for S in cuts_S]

        m = max(Lc)
        w = [np.exp((v - m) / tau) for v in Lc]
        Z = np.sum(w)
        if Z <= 0.0:
            Z = 1.0
        w = [x / Z for x in w]

        out = []
        for c, S in enumerate(cuts_S):
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(
                S, rank, Fnorm
            )
            out.append([w[c] * v for v in g])
        return out

    # Build M with build_osr_matrix, then SVD (econ) and grab top triplet.
    @staticmethod
    def top_k_triplet_for_cut(
        U,  # (N x N), row-major, N = 1<<q
        q,  # number of qubits
        A,  # qubits on side A
        Fnorm,  # e.g., sqrt(N)
    ):
        """SVD of OSR matrix for cut ``A``: returns normalized singular values and ``U``, ``Vh``."""
        # 1) Build M for this cut
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, q, A)
        k = min(M.shape)

        # 2) SVD: M = U * diag(S) * VT  (VT = V^H)
        # Row-major API handles leading dims as col counts.
        res = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        return res.S / Fnorm, res.U, res.Vh  # normalized singular value

    @staticmethod
    def get_deriv_osr_entanglement(matrix, use_cuts, rank, use_softmax):
        """Gradient of rank / softmax-rank entanglement cost w.r.t. unitary ``matrix`` entries."""
        qbit_num = len(matrix).bit_length() - 1
        cuts = (
            list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(qbit_num))
            if len(use_cuts) == 0
            else use_cuts
        )
        Fnorm = np.sqrt(len(matrix))
        deriv = np.zeros(matrix.shape, dtype=complex)
        # Compute the derivative of the OSR entanglement cost function
        triplets = []
        allS = []
        for cut in cuts:
            # 1) top k triplet on the normalized reshape M_c
            S, Umat, VTmat = N_Qubit_Decomposition_Guided_Tree.top_k_triplet_for_cut(
                matrix, qbit_num, cut, Fnorm
            )
            triplets.append(([], Umat, VTmat))
            allS.append(S)
        if use_softmax:
            allS = N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_grad(
                allS, rank, Fnorm
            )
        else:
            allS = N_Qubit_Decomposition_Guided_Tree.cuts_avg_rank_grad(
                allS, rank, Fnorm
            )
        for i in range(len(cuts)):
            triplets[i] = (allS[i], triplets[i][1], triplets[i][2])
        for i in range(len(cuts)):
            G, Umat, VTmat = triplets[i]
            N_Qubit_Decomposition_Guided_Tree.accumulate_grad_for_cut(
                deriv, G, Umat, VTmat, qbit_num, cuts[i]
            )
        return deriv

    # Compute grad component = Re Tr( A^† B ) for A = dL/dU, B = dU/dθ
    # A and B are (rows x cols) with row-major leading dimension.
    @staticmethod
    def real_trace_conj_dot(A, B):
        """Return ``Re Tr(A† B)`` for complex matrices ``A``, ``B`` (row-major storage)."""
        return np.sum(A.real * B.real + A.imag * B.imag)  # Re Tr(A^† B)

    @staticmethod
    def param_derivs(circ, Umtx, x):
        """Finite-difference / shift-style partial derivatives ``∂U/∂θ_i`` for each gate parameter in ``x``."""
        n = len(x)
        derivs = [None] * n
        for i in range(n):
            kind = i % 3
            if kind == 0:  # d/dt:  ∂U/∂t = U(t+π/2, φ, λ)
                x_shift = x.copy()
                x_shift[i] += np.pi / 2
                Ui = Umtx.copy()
                circ.apply_to(x_shift, Ui)
                derivs[i] = Ui
            else:  # d/dφ or d/dλ: ∂U/∂p = 0.5*(U(p+π/2) - U(p-π/2))
                xp = x.copy()
                xp[i] += np.pi / 2
                xm = x.copy()
                xm[i] -= np.pi / 2
                Up = Umtx.copy()
                Um = Umtx.copy()
                circ.apply_to(xp, Up)
                circ.apply_to(xm, Um)
                derivs[i] = 0.5 * (Up - Um)
        return derivs

    @staticmethod
    def _global_phase_fix(U):
        """Remove global phase from square unitary ``U`` using determinant normalization."""
        return U / (np.linalg.det(U) ** (1 / len(U)))

    @staticmethod
    def _polar_unitary(X):
        """Nearest unitary to ``X`` via polar decomposition (SVD)."""
        U, _, Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh

    @staticmethod
    def su2_to_u3_zyz(U):
        """
        Decompose a 2x2 unitary (det=1) into Qiskit U3: Rz(phi) @ Ry(theta) @ Rz(lam).
        Returns (theta, phi, lam) in radians.
        """
        U = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U)
        # Handle numeric edge cases robustly
        a = U[0, 0]
        b = U[0, 1]
        c = U[1, 0]
        d = U[1, 1]
        # Prefer arccos for theta; it's stable when |a| is not tiny
        ca = np.clip(np.abs(a), 0.0, 1.0)
        theta = 2.0 * np.arccos(ca)
        # If sin(theta/2) ~ 0, collapse to Z rotations
        eps = 1e-12
        if abs(np.sin(theta / 2)) < eps:
            # Then c≈0, b≈0; only Z phases matter: U ≈ e^{iα} Rz(phi+lam)
            # Choose phi=0, lam = arg(d) - arg(a)
            phi = 0.0
            lam = np.angle(d) - np.angle(a)
            # Normalize to [-pi,pi)
            lam = (lam + np.pi) % (2 * np.pi) - np.pi
            return float(theta), float(phi), float(lam)

        # Otherwise, phases from elements and normalize
        phi = np.angle(c) - np.angle(a)
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        lam = np.angle(b) - np.angle(a) - np.pi
        lam = (lam + np.pi) % (2 * np.pi) - np.pi
        return float(theta), float(phi), float(lam)

    @staticmethod
    def _A_from_c(c1, c2, c3):
        """Two-qubit canonical interaction ``exp(-i/2 * (c1 XX + c2 YY + c3 ZZ))`` as a unitary."""
        X = np.array([[0, 1], [1, 0]], complex)
        Y = np.array([[0, -1j], [1j, 0]], complex)
        Z = np.array([[1, 0], [0, -1]], complex)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)
        H = c1 * XX + c2 * YY + c3 * ZZ
        # use exp via eig (4x4) for robustness
        ew, EV = np.linalg.eig(1j * H)
        A = EV @ np.diag(np.exp(ew)) @ np.linalg.inv(EV)
        # project back to unitary (remove numeric drift)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A)

    # Factor K1, K2 → (2x2 ⊗ 2x2)
    @staticmethod
    def factor_local(K):
        """Factor 4x4 unitary ``K`` into Kronecker product of two 2x2 unitaries (SVD on reshaped tensor)."""
        # reshape to (2,2,2,2), SVD the (a,c ; b,d) unfolding
        M = K.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        U, _, Vh = np.linalg.svd(M, full_matrices=False)
        A = U[:, 0].reshape(2, 2)
        B = Vh.conj().T[:, 0].reshape(2, 2)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(
            A
        ), N_Qubit_Decomposition_Guided_Tree._polar_unitary(B)

    @staticmethod
    def _magic_basis_plusYY():
        """Magic basis matrix for two-qubit canonical form (Bell-like columns)."""
        # Complex magic basis (matches A(c)=exp(-i/2*(c1 XX + c2 YY + c3 ZZ)) below)
        # Columns are (|Φ+>, i|Φ->, i|Ψ+>, |Ψ->) up to harmless phases
        return (1 / np.sqrt(2)) * np.array(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1j, 0, 0, -1]],
            dtype=complex,
        )

    @staticmethod
    def _project_to_SO4(O):
        """Nearest proper SO(4) rotation to real matrix ``O`` (SVD with det fix)."""
        # nearest real orthogonal with det=+1
        O = np.real_if_close(O, tol=1e5)
        U, _, Vt = np.linalg.svd(O)
        O = U @ Vt
        if np.linalg.det(O) < 0:
            O[:, 0] *= -1
        return O

    @staticmethod
    def _clean_col_phases(W):
        """Remove column-wise global phases from matrix ``W`` (largest-magnitude entry per column)."""
        Wc = W.copy()
        for j in range(Wc.shape[1]):
            col = Wc[:, j]
            k = np.argmax(np.abs(col))
            if np.abs(col[k]) > 1e-14:
                Wc[:, j] *= np.exp(-1j * np.angle(col[k]))
        return Wc

    @staticmethod
    def closest_local_product(W4):
        """Best product of single-qubit unitaries approximating 4x4 ``W4`` (via ``factor_local``)."""
        A, B = N_Qubit_Decomposition_Guided_Tree.factor_local(W4)
        return N_Qubit_Decomposition_Guided_Tree._global_phase_fix(
            A
        ), N_Qubit_Decomposition_Guided_Tree._global_phase_fix(B)

    @staticmethod
    def kak_u3s_around_cx(U, n, c, t, iters=3):
        """KAK-style two-qubit block on control ``c`` and target ``t``: Weyl angles and U3 params (debug helper)."""
        U4 = N_Qubit_Decomposition_Guided_Tree.trace_out_qubits(U, n, (c, t))
        U4 = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U4)
        from qiskit.synthesis import TwoQubitWeylDecomposition

        twd = TwoQubitWeylDecomposition(U4)
        c1, c2, c3 = twd.a, twd.b, twd.c
        K1A, K1B, K2A, K2B = twd.K1l, twd.K1r, twd.K2l, twd.K2r
        A = N_Qubit_Decomposition_Guided_Tree._A_from_c(c1, c2, c3)
        U_rec = np.kron(K1A, K1B) @ A @ np.kron(K2A, K2B)
        z = np.trace(U_rec.conj().T @ U4)
        U_rec *= np.exp(1j * np.angle(z))
        print("Frob err:", np.linalg.norm(U_rec - U4), c1, c2, c3)
        thA_pre, phA_pre, laA_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(
            K2A.conj().T
        )
        thB_pre, phB_pre, laB_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(
            K2B.conj().T
        )
        thA_post, phA_post, laA_post = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(
            K1A.conj().T
        )  # left-apply ⇒ take dagger on outputs
        thB_post, phB_post, laB_post = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(
            K1B.conj().T
        )
        return {
            "c": (c1, c2, c3),
            "pre": {
                "A": (thA_pre / 2, phA_pre, laA_pre),
                "B": (thB_pre / 2, phB_pre, laB_pre),
            },
            "post": {
                "A": (thA_post / 2, phA_post, laA_post),
                "B": (thB_post / 2, phB_post, laB_post),
            },
        }

    def params_to_mat(self, params):
        """Apply current gate structure to each target unitary with (optional) affine parameter scaling."""
        allU = []
        for U, pspace in zip(
            self.Umtx, [None] if self.paramspace is None else self.paramspace
        ):
            U = U.copy()
            scaled_params = (
                np.sum(
                    params.reshape(-1, 1 + len(pspace)) * np.array((1.0,) + pspace),
                    axis=1,
                )
                if pspace is not None
                else params
            )
            self.get_Circuit().apply_to(
                scaled_params if pspace is not None else params, U
            )
            allU.append(U)
        return allU

    def OSR_with_local_alignment(
        self, pairs, cuts, Fnorm, tol, rank, use_softmax, method="dual_annealing"
    ):
        """Optimize gate parameters to reduce OSR-based entanglement across ``cuts`` (optionally softmax-aggregated).

        Uses cost variant 10 during optimization, then restores variant 3. Returns list of
        ``(ceil_log2(rank), singular_spectrum)``-style entries per unitary and cut.
        """
        if len(pairs) != 0:
            self.set_Cost_Function_Variant(10)
            # self.Run_Decomposition(pairs, False)
            self.set_Gate_Structure(self.get_circuit_from_pairs(pairs, False))
            import scipy

            param_bound = np.array(
                ([2 * np.pi] + [1 / x for x in self.paramscale])
                * self.get_Parameter_Num()
            )

            def cost(x):
                allU = self.params_to_mat(x)
                S = [
                    N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(
                        U, self.qbit_num, cut, Fnorm, tol
                    )[1]
                    for U in allU
                    for cut in cuts
                ]
                if use_softmax:
                    return N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_cost(
                        S, rank
                    )
                else:
                    return N_Qubit_Decomposition_Guided_Tree.avg_loss_for_rank(S, rank)

            def jacobian(x):
                allU = self.params_to_mat(x)
                grad = np.zeros(len(x), dtype=float)
                for Ubase, U, pspace in zip(
                    self.Umtx,
                    allU,
                    [None] if self.paramspace is None else self.paramspace,
                ):
                    dL = N_Qubit_Decomposition_Guided_Tree.get_deriv_osr_entanglement(
                        U, cuts, rank, use_softmax
                    )
                    basevec = np.array((1.0,) if pspace is None else (1.0,) + pspace)
                    scaled_params = (
                        np.sum(x.reshape(-1, 1 + len(pspace)) * basevec, axis=1)
                        if pspace is not None
                        else x
                    )
                    derivs = N_Qubit_Decomposition_Guided_Tree.param_derivs(
                        self.get_Circuit(), Ubase, scaled_params
                    )
                    newgrad = np.array(
                        [
                            N_Qubit_Decomposition_Guided_Tree.real_trace_conj_dot(
                                dL, deriv
                            )
                            for deriv in derivs
                        ]
                    )
                    if pspace is not None:
                        newgrad = (np.array(newgrad)[:, np.newaxis] * basevec).reshape(
                            -1
                        )
                    grad += newgrad
                return grad / len(self.Umtx)

            if method == "differential_evolution":
                best = scipy.optimize.differential_evolution(
                    cost, [(0, x) for x in param_bound], maxiter=100, polish=False
                )
                best = scipy.optimize.minimize(
                    cost, best.x, method="BFGS", jac=jacobian, options={"maxiter": 200}
                )
            elif method == "dual_annealing":
                best = None
                for seed in range(20):
                    res = scipy.optimize.dual_annealing(
                        cost, [(0, x) for x in param_bound], maxiter=100
                    )  # , minimizer_kwargs={'jac': jacobian})
                    if best is None or res.fun < best.fun:
                        best = res
            elif method == "basinhopping":
                best = scipy.optimize.basinhopping(
                    cost,
                    np.random.rand(len(param_bound)) * param_bound,
                    niter=50,
                    stepsize=np.pi / 2,
                    minimizer_kwargs={"jac": jacobian},
                )
            else:
                best = min(
                    [
                        scipy.optimize.minimize(
                            cost,
                            np.random.rand(len(param_bound)) * param_bound,
                            method="BFGS",
                            jac=jacobian,
                            options={"maxiter": 200},
                        )
                        for _ in range(20)
                    ],
                    key=lambda r: r.fun,
                )
            # print(best)
            self.set_Cost_Function_Variant(3)
            assert best is not None
            allU = self.params_to_mat(best.x)
        else:
            allU = self.Umtx
        return [
            (N_Qubit_Decomposition_Guided_Tree.ceil_log2(rank), s)
            for U in allU
            for cut in cuts
            for rank, s in (
                N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(
                    U, self.qbit_num, cut, Fnorm, tol
                ),
            )
        ]

    def Run_Decomposition(self, pairs, finalizing=True):
        """Run BFGS decomposition for CNOT structure ``pairs``; set ``self.err`` and return success vs tolerance."""
        circ = self.get_circuit_from_pairs(pairs, finalizing)
        self.set_Gate_Structure(circ)
        self.set_Optimized_Parameters(
            np.random.rand(self.get_Parameter_Num()) * (2 * np.pi)
        )
        super().Start_Decomposition()
        if finalizing:
            params = self.get_Optimized_Parameters()
            self.err = self.Optimization_Problem(params)
            return self.err < self.config.get(
                "tolerance",
                _default_squander_tolerance(self.config),
            )

    @staticmethod
    def generate_insertions(curpath, topology, num_cnot):
        """Yield CNOT insertion patterns: insert ``num_cnot`` topology pairs into sequence ``curpath``."""
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
        """Beam-style search over CNOT prefixes guided by OSR stats; collects solutions in ``self.all_solutions``."""
        import heapq, itertools

        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        # because we have U already conjugate transposed, must use prefix order
        B = self.config.get("beam", None)  # 8*len(self.topology))
        max_depth = self.config.get("tree_level_max", 14)
        tol = 1e-3
        Fnorm = np.sqrt(1 << self.qbit_num)
        best = []
        visited = set()
        all_ranks = list(range(min(2, self.qbit_num - 1)))

        def get_osr_stats(path, rank, use_softmax):
            """Return ``(min_cnots, rank_kappa_metric, raw_osr_list)`` for prefix ``path``."""
            h = self.OSR_with_local_alignment(
                path,
                cuts,
                Fnorm,
                tol=tol,
                rank=rank,
                use_softmax=use_softmax,
                method="basin_hopping",
            )
            min_cnots = max((x[0] for x in h), default=0)
            ranktot = sum(x[0] for x in h)
            kappa = sum(sum(y * y for y in x[1][1:]) for x in h)
            return min_cnots, ranktot + kappa, h

        def add_to_heap(path, parent_stats):
            """Push ``path`` onto search heap if within depth and OSR bounds improve on ``parent_stats``."""
            if len(path) > max_depth:
                return False
            if path in visited:
                return False
            visited.add(path)
            if self.qbit_num > 1:
                min_cnots, rankkappa = min(
                    get_osr_stats(path, rank, use_sm)[:2]
                    for (rank, use_sm) in itertools.product(all_ranks, (False,))
                )  # (False, True)
            else:
                min_cnots, rankkappa = 0, 0.0
            if parent_stats is not None and (min_cnots, rankkappa) >= parent_stats:
                return False
            heapq.heappush(best, (min_cnots, rankkappa, path))
            return True

        add_to_heap((), None)
        while best:
            # print(best[0])
            min_cnots, rankkappa, curpath = heapq.heappop(best)
            if min_cnots == 0:
                # print(path)
                for i in range(10):
                    if self.Run_Decomposition(curpath):
                        self.all_solutions.append(
                            (self.get_Circuit(), self.get_Optimized_Parameters())
                        )
                        if stop_first_solution:
                            return
                        break
                    # print("Looping", h)
            num_cnot = 1
            while True:
                any_added = False
                for newpath in N_Qubit_Decomposition_Guided_Tree.generate_insertions(
                    curpath, self.topology, num_cnot
                ):
                    if add_to_heap(newpath, (min_cnots, rankkappa)):
                        any_added = True
                if any_added:
                    break
                num_cnot += 1
                if len(curpath) + num_cnot > max_depth:
                    break
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        # print("No decomposition found within the given CNOT limit.")

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

    def get_Decomposition_Error(self):
        """Last decomposition error (Frobenius / cost) from guided search or ``Run_Decomposition``."""
        return self.err

    @staticmethod
    def compositions(total, parts):
        """
        All nonnegative integer tuples of length `parts` summing to `total`.
        """
        if parts == 1:
            yield (total,)
            return
        for x in range(total + 1):
            for rest in N_Qubit_Decomposition_Guided_Tree.compositions(
                total - x, parts - 1
            ):
                yield (x,) + rest

    @staticmethod
    def solve_best_min_cnots(num_qubits, cuts, rank_kappa, topology, use_surplus=True):
        """Minimize total CNOT count subject to per-cut edge coverage vs ``rank_kappa`` bounds; return best kappa."""
        m = len(topology)
        cut_to_edges = [
            [i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)]
            for cut in cuts
        ]
        total = 0
        best_kappa = None
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(
                    sum(edge_counts[j] for j in cut_to_edge) >= cut_bound[0]
                    for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa)
                ):
                    new_kappa = 0.0
                    for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa):
                        coverage = sum(edge_counts[j] for j in cut_to_edge)
                        if use_surplus:
                            new_kappa += cut_bound[1] * (coverage - cut_bound[0])
                        else:
                            new_kappa += cut_bound[1] * coverage
                    best_kappa = (
                        new_kappa if best_kappa is None else max(best_kappa, new_kappa)
                    )
            if best_kappa is not None:
                break
            total += 1
        return total, best_kappa

    @staticmethod
    def solve_min_cnots(num_qubits, cuts, cut_bounds, topology):
        """Smallest total CNOT budget such that each cut's crossing edges meet ``cut_bounds``."""
        m = len(topology)
        cut_to_edges = [
            [i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)]
            for cut in cuts
        ]
        total = 0
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(
                    sum(edge_counts[j] for j in cut_to_edge) >= cut_bound
                    for cut_to_edge, cut_bound in zip(cut_to_edges, cut_bounds)
                ):
                    return total
            total += 1

    @staticmethod
    def gen_all_min_cnots(
        num_qbits, topology=None
    ):  # OSR tells min CNOTs at most for 3 qubits 3, 4 qubits 6, 5 qubits 7
        """Debug: print min CNOT solutions for all combinations of per-cut bounds (see ``solve_min_cnots``)."""
        import itertools

        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(num_qbits))
        min_cnot_bounds = [
            2 * min(cut_size, num_qbits - cut_size)
            for cut_size in (len(cut) for cut in cuts)
        ]
        if topology is None:
            topology = [
                (i, j) for i in range(num_qbits) for j in range(i + 1, num_qbits)
            ]
        for cnot_bounds in itertools.product(
            *(range(bound + 1) for bound in min_cnot_bounds)
        ):
            # if tuple(sorted(cnot_bounds)) != cnot_bounds: continue
            print(
                cnot_bounds,
                N_Qubit_Decomposition_Guided_Tree.solve_min_cnots(
                    num_qbits, cuts, cnot_bounds, topology
                ),
            )


# N_Qubit_Decomposition_Guided_Tree.gen_all_min_cnots(3); assert False
# N_Qubit_Decomposition_Guided_Tree.build_sequence(); assert False
# print(len(list(N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(3, [(0,1),(1,2),])))); assert False
