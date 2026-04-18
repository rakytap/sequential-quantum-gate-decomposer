"""
This is an implementation of Partition Aware Mapping.
"""
import logging
import multiprocessing as mp
import time
from collections import deque, defaultdict
from itertools import permutations
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.ilp import (
    get_all_partitions,
    _get_topo_order,
    topo_sort_partitions,
    ilp_global_optimal,
)
# Module-level globals for pool workers (set via Pool initializer)
_worker_config = None

def _init_decompose_worker(config):
    global _worker_config
    _worker_config = config

def _decompose_one(Umtx, mini_topology):
    """Pool worker function. Uses config set once by initializer instead of
    pickling it per task."""
    from squander.synthesis.PartAM import qgd_Partition_Aware_Mapping
    return qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(
        Umtx, _worker_config, mini_topology
    )

from squander.synthesis.PartAM_utils import (
    get_subtopologies_of_type,
    get_unique_subtopologies,
    get_canonical_form,
    get_node_mapping,
    compute_automorphisms,
    derive_result_from_automorphism,
    SingleQubitPartitionResult,
    PartitionSynthesisResult,
    PartitionCandidate,
    PartitionScoreData,
    check_circuit_compatibility,
    construct_swap_circuit,
)


# ============================================================================
# Main Class: qgd_Partition_Aware_Mapping
# ============================================================================

class qgd_Partition_Aware_Mapping:

    # ------------------------------------------------------------------------
    # Initialization & Configuration
    # ------------------------------------------------------------------------

    def __init__(self, config):
        self.topology = config['topology']
        self.config = config
        self.config.setdefault('strategy', 'TreeSearch')
        self.config.setdefault('parallel', 0 )
        self.config.setdefault('verbosity', 0 )
        self.config.setdefault('tolerance', 1e-8 )
        self.config.setdefault('test_subcircuits', False )
        self.config.setdefault('test_final_circuit', True )
        self.config.setdefault('max_partition_size', 3 )
        self.config.setdefault('topology', None)
        self.config.setdefault('routed', False)
        self.config.setdefault('partition_strategy','ilp')
        self.config.setdefault('optimizer', 'BFGS2')
        self.config.setdefault('use_basin_hopping', 1)
        self.config.setdefault('bh_T', 1.0)
        self.config.setdefault('bh_stepsize', 0.5)
        self.config.setdefault('bh_interval', 50)
        self.config.setdefault('bh_target_accept_rate', 0.5)
        self.config.setdefault('bh_stepwise_factor', 0.9)
        self.config.setdefault('use_osr', 0)
        self.config.setdefault('n_layout_trials', 1)
        self.config.setdefault('score_tolerance', 0.05)
        self.config.setdefault('random_seed', 42)
        self.config.setdefault('cleanup', True)
        self.config.setdefault('prefilter_top_k', 50)
        strategy = self.config['strategy']
        allowed_strategies = ['TreeSearch', 'TabuSearch', 'Adaptive']
        if not strategy in allowed_strategies:
            raise Exception(f"The strategy should be either of {allowed_strategies}, got {strategy}.")
        
        # Initialize caches for performance optimization
        self._topology_cache = {}  # {frozenset(edges): [topology_candidates]}
        self._swap_cache = {}     # {(pi_tuple, qbit_map_frozen): (swaps, output_perm)}
        self._adj = None          # Precomputed adjacency list (built by compute_distances_bfs)

    # ------------------------------------------------------------------------
    # Caching Methods
    # ------------------------------------------------------------------------

    def _get_subtopologies_of_type_cached(self, mini_topology):
        """
        Cached version of get_subtopologies_of_type.
        Uses canonical form of mini_topology as cache key.
        """
        
        # Create canonical form key
        target_qubits = set()
        for u, v in mini_topology:
            target_qubits.add(u)
            target_qubits.add(v)
        if not target_qubits:
            return []
        
        # Use canonical form as cache key
        canonical_key = get_canonical_form(target_qubits, mini_topology)
        
        if canonical_key not in self._topology_cache:
            self._topology_cache[canonical_key] = get_subtopologies_of_type(self.topology, mini_topology)
        
        return self._topology_cache[canonical_key]

    # ------------------------------------------------------------------------
    # Static Synthesis Helpers (extracted from SynthesizeWideCircuit)
    # ------------------------------------------------------------------------

    @staticmethod
    def _topo_key(mini_topology):
        return tuple(sorted(tuple(sorted(e)) for e in mini_topology))

    @staticmethod
    def _cache_key(Umtx, mini_topology):
        topo_key = tuple(sorted(tuple(sorted(e)) for e in mini_topology))
        return (np.round(Umtx, decimals=10).tobytes(), topo_key)

    @staticmethod
    def _get_auts(mini_topo, aut_cache):
        key = tuple(sorted(tuple(sorted(e)) for e in mini_topo))
        if key not in aut_cache:
            aut_cache[key] = compute_automorphisms(mini_topo)
        return aut_cache[key]

    @staticmethod
    def _build_permuted_unitary(meta, P_i, P_o):
        N = meta['N']
        circ_tmp = Circuit(N)
        circ_tmp.add_Permutation(list(P_i))
        circ_tmp.add_Circuit(meta['circuit'])
        circ_tmp.add_Permutation(list(P_o))
        return circ_tmp.get_Matrix(meta['params'])

    @staticmethod
    def _add_result_with_auts(result, perm_pair, synth_circuit, synth_params,
                              topology_idx, N, mini_topology, known_pairs, pair_key,
                              use_auts, aut_cache):
        """Add a synthesis result and derive automorphism equivalents."""
        result.add_result(perm_pair, synth_circuit, synth_params, topology_idx)
        if use_auts:
            if pair_key not in known_pairs:
                known_pairs[pair_key] = set()
            known_pairs[pair_key].add(perm_pair)
            P_i, P_o = perm_pair
            auts = qgd_Partition_Aware_Mapping._get_auts(mini_topology, aut_cache)
            identity = tuple(range(N))
            for sigma in auts:
                if sigma == identity:
                    continue
                new_P_i, new_P_o, new_circ, new_params = derive_result_from_automorphism(
                    sigma, P_i, P_o, synth_circuit, synth_params, N
                )
                if (new_P_i, new_P_o) not in known_pairs[pair_key]:
                    result.add_result((new_P_i, new_P_o), new_circ, new_params, topology_idx)
                    known_pairs[pair_key].add((new_P_i, new_P_o))

    def _build_scoring_partitions(self, optimized_partitions) -> List[Optional[PartitionScoreData]]:
        """
        Create lightweight, picklable views of partitions that contain only the
        data required during heuristic scoring.
        """
        scoring_partitions: List[Optional[PartitionScoreData]] = []
        for partition in optimized_partitions:
            if isinstance(partition, SingleQubitPartitionResult):
                scoring_partitions.append(None)
                continue

            mini_topologies = tuple(
                tuple(tuple(edge) for edge in mini_topology)
                for mini_topology in partition.mini_topologies
            )

            topology_candidates = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                if hasattr(partition, "get_topology_candidates"):
                    candidates = partition.get_topology_candidates(tdx)
                else:
                    candidates = self._get_subtopologies_of_type_cached(mini_topology)
                topology_candidates.append(
                    tuple(tuple(edge) for edge in candidates)
                )

            permutations_pairs = tuple(
                tuple((tuple(P_i), tuple(P_o)) for (P_i, P_o) in partition.permutations_pairs[tdx])
                for tdx in range(len(partition.mini_topologies))
            )

            circuit_structures = tuple(
                tuple(tuple(struct) for struct in partition.circuit_structures[tdx])
                for tdx in range(len(partition.mini_topologies))
            )

            scoring_partitions.append(
                PartitionScoreData(
                    mini_topologies=mini_topologies,
                    topology_candidates=tuple(topology_candidates),
                    permutations_pairs=permutations_pairs,
                    circuit_structures=circuit_structures,
                    qubit_map=dict(partition.qubit_map),
                    involved_qbits=tuple(partition.involved_qbits),
                )
            )
        return scoring_partitions

    # ------------------------------------------------------------------------
    # Partition Decomposition Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def DecomposePartition_and_Perm(Umtx: np.ndarray, config: dict, mini_topology = None, max_retries: int = 5) -> Circuit:
        """
        Call to decompose a partition. Retries up to max_retries times if the
        decomposition error exceeds the configured tolerance.  Returns the
        best-error attempt across all retries and logs a warning when no
        attempt reaches ``config["tolerance"]``.
        """
        tolerance = config["tolerance"]
        strategy = config["strategy"]

        best_err = float('inf')
        best_circuit = None
        best_params = None

        for attempt in range(max_retries):
            if strategy == "TreeSearch":
                cDecompose = N_Qubit_Decomposition_Tree_Search(Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
            elif strategy == "TabuSearch":
                cDecompose = N_Qubit_Decomposition_Tabu_Search(Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
            elif strategy == "Adaptive":
                cDecompose = N_Qubit_Decomposition_adaptive(Umtx.conj().T, level_limit_max=5, level_limit_min=1, topology=mini_topology)
            else:
                raise Exception(f"Unsupported decomposition type: {strategy}")
            cDecompose.set_Verbose(config["verbosity"])
            cDecompose.set_Cost_Function_Variant(3)
            cDecompose.set_Optimization_Tolerance(tolerance)
            cDecompose.set_Optimizer(config["optimizer"])
            cDecompose.Start_Decomposition()

            err = cDecompose.get_Decomposition_Error()
            if err < best_err:
                best_err = err
                best_circuit = cDecompose.get_Circuit()
                best_params = cDecompose.get_Optimized_Parameters()

            if best_err <= tolerance:
                break

        if best_err > tolerance:
            N = int(np.log2(Umtx.shape[0]))
            logging.warning(
                "DecomposePartition_and_Perm: %d-qubit partition on topology %s "
                "did not reach tolerance %.2e after %d retries "
                "(best error %.2e). Returning best attempt; final circuit error "
                "may be elevated.",
                N, list(mini_topology) if mini_topology else None,
                tolerance, max_retries, best_err,
            )

        return best_circuit, best_params

    # ------------------------------------------------------------------------
    # Circuit Synthesis
    # ------------------------------------------------------------------------

    def SynthesizeWideCircuit(self, circ, orig_parameters):
        """
        Partition and synthesize a full circuit.

        Flow:
            1) Enumerate candidate partitions.
            2) ILP-select the minimum-count non-overlapping cover (uniform weights).
            3) Synthesize only the selected partitions via SeqPAM (two-stage P_i/P_o
               sweep over mini_topologies, executed by _run_parallel_synthesis).

        Args:
            circ: The full quantum circuit (must be flat — no subcircuit blocks)
            orig_parameters: Parameters for circ

        Returns:
            optimized_partitions: List of PartitionSynthesisResult /
                SingleQubitPartitionResult, in topological order.
        """
        working_circ = circ
        working_parameters = orig_parameters
        qbit_num = circ.get_Qbit_Num()

        # ---- Phase 0: Compute distance matrix ----
        D = self.compute_distances_bfs(qbit_num)

        # ---- Phase 1: Partition enumeration ----
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(working_circ, self.config["max_partition_size"])
        qbit_num_orig_circuit = working_circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(working_circ.get_Gates())}

        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}

        # ---- Phase 2: ILP partition selection ----
        # 2-qubit partitions are free (weight 0) since they are trivially
        # synthesized as themselves; 3+ qubit partitions cost 1.
        weights = [
            0 if len({q for gate in part for q in gate_to_qubit[gate]}) == 2 else 1
            for part in allparts
        ]
        L_parts, _ = ilp_global_optimal(allparts, g, weights=weights)

        # ---- Phase 3: Build gate sets for selected partitions (+ standalone chains) ----
        selected_surrounded_starts = set()
        selected_parts_gates = []
        for i in L_parts:
            part = allparts[i]
            surrounded = {t for s in part for t in go[s]
                          if t in single_qubit_chains_prepost
                          and go[single_qubit_chains_prepost[t][-1]]
                          and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded))
            selected_parts_gates.append(gates)
            selected_surrounded_starts.update(surrounded)

        standalone_chains = []
        for chain in single_qubit_chains:
            if chain[0] not in selected_surrounded_starts:
                selected_parts_gates.append(frozenset(chain))
                standalone_chains.append(chain)

        n_multi = len(L_parts)

        # ---- Phase 4: Assemble partitioned circuit from selected partitions only ----
        partitioned_circuit = Circuit(qbit_num_orig_circuit)
        params = []

        for gates in selected_parts_gates[:n_multi]:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates},
                                            {x: rgo[x] & gates for x in gates},
                                            gate_to_qubit):
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)

        for chain in standalone_chains:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in chain:
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)

        parameters = np.concatenate(params, axis=0)

        # ---- Phase 5: SeqPAM synthesis on selected partitions only ----
        subcircuits = partitioned_circuit.get_Gates()
        optimized_results = [None] * len(subcircuits)
        partition_meta = []
        for partition_idx, subcircuit in enumerate(subcircuits):
            start_idx = subcircuit.get_Parameter_Start_Index()
            end_idx = start_idx + subcircuit.get_Parameter_Num()
            subcircuit_parameters = parameters[start_idx:end_idx]
            involved_qbits = subcircuit.get_Qbits()
            qbit_num_sub = len(involved_qbits)
            qbit_map = {involved_qbits[idx]: idx for idx in range(len(involved_qbits))}
            remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num_sub)

            if qbit_num_sub == 1:
                optimized_results[partition_idx] = SingleQubitPartitionResult(
                    remapped_subcircuit, subcircuit_parameters,
                    original_qubits=list(involved_qbits)
                )
                partition_meta.append(None)
            else:
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num_sub)
                partition_meta.append({
                    'N': qbit_num_sub,
                    'circuit': remapped_subcircuit,
                    'params': subcircuit_parameters,
                    'mini_topologies': mini_topologies,
                    'involved_qbits': involved_qbits,
                    'qbit_map': qbit_map,
                })

        results_map = self._run_parallel_synthesis(partition_meta)
        for partition_idx, result in results_map.items():
            optimized_results[partition_idx] = result

        # ---- Phase 6: Topologically order selected partitions ----
        L = topo_sort_partitions(working_circ, selected_parts_gates)
        return [optimized_results[idx] for idx in L]

    def _run_parallel_synthesis(self, partition_meta):
        """Phase 2: Run parallel synthesis for all multi-qubit partitions.

        Args:
            partition_meta: List of per-partition dicts (None for single-qubit partitions).

        Returns:
            results_map: Dict mapping partition_idx to PartitionSynthesisResult.
        """
        n_cpus = mp.cpu_count()
        use_auts = self.config.get('use_automorphisms', True)
        disable_pbar = self.config.get('progressbar', 0) == False
        aut_cache = {}
        decomp_cache = {}

        with Pool(processes=n_cpus, initializer=_init_decompose_worker,
                  initargs=(self.config,)) as pool:
            # Initialize PartitionSynthesisResult for each multi-qubit partition
            results_map = {}
            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                results_map[partition_idx] = PartitionSynthesisResult(
                    meta['N'], meta['mini_topologies'], meta['involved_qbits'],
                    meta['qbit_map'],
                )

            # ---- Stage 1: fix random P_o, sweep all P_i ----
            stage1_futures = []
            stage1_cached = []
            stage1_P_o = {}
            known_pairs = {}

            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                N = meta['N']
                perms_all = list(permutations(range(N)))
                for topology_idx, mini_topology in enumerate(meta['mini_topologies']):
                    P_o_initial = perms_all[np.random.choice(len(perms_all))]
                    stage1_P_o[(partition_idx, topology_idx)] = P_o_initial
                    for P_i in perms_all:
                        Umtx = self._build_permuted_unitary(meta, P_i, P_o_initial)
                        ck = self._cache_key(Umtx, mini_topology)
                        if ck in decomp_cache:
                            stage1_cached.append((partition_idx, topology_idx, P_i, ck))
                        else:
                            future = pool.apply_async(
                                _decompose_one, (Umtx, mini_topology)
                            )
                            stage1_futures.append((partition_idx, topology_idx, P_i, ck, future))

            # Process Stage 1 cache hits immediately
            for partition_idx, topology_idx, P_i, ck in stage1_cached:
                meta = partition_meta[partition_idx]
                N = meta['N']
                P_o_initial = stage1_P_o[(partition_idx, topology_idx)]
                mini_topology = meta['mini_topologies'][topology_idx]
                synth_circuit, synth_params = decomp_cache[ck]
                pair_key = (partition_idx, topology_idx)
                self._add_result_with_auts(
                    results_map[partition_idx], (P_i, P_o_initial),
                    synth_circuit, synth_params, topology_idx,
                    N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                )

            # Collect Stage 1 pool results
            cache_hits_s1 = len(stage1_cached)
            for partition_idx, topology_idx, P_i, ck, future in tqdm(
                stage1_futures, desc=f"Stage 1 Synthesis ({cache_hits_s1} cached)",
                disable=disable_pbar
            ):
                synth_circuit, synth_params = future.get()
                decomp_cache[ck] = (synth_circuit, synth_params)
                meta = partition_meta[partition_idx]
                N = meta['N']
                P_o_initial = stage1_P_o[(partition_idx, topology_idx)]
                mini_topology = meta['mini_topologies'][topology_idx]
                pair_key = (partition_idx, topology_idx)
                self._add_result_with_auts(
                    results_map[partition_idx], (P_i, P_o_initial),
                    synth_circuit, synth_params, topology_idx,
                    N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                )

            # ---- Stage 2: fix top-k P_i from Stage 1, sweep all P_o ----
            top_k_pi = self.config.get('top_k_pi', 1)
            stage2_futures = []
            stage2_cached = []

            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                N = meta['N']
                perms_all = list(permutations(range(N)))
                result = results_map[partition_idx]
                for topology_idx, mini_topology in enumerate(meta['mini_topologies']):
                    pair_key = (partition_idx, topology_idx)
                    kp = known_pairs.get(pair_key, set()) if use_auts else set()
                    for P_i_cand in result.get_top_k_results(topology_idx, top_k_pi):
                        for P_o in perms_all:
                            if use_auts and (tuple(P_i_cand), P_o) in kp:
                                continue
                            Umtx = self._build_permuted_unitary(meta, P_i_cand, P_o)
                            ck = self._cache_key(Umtx, mini_topology)
                            if ck in decomp_cache:
                                stage2_cached.append((partition_idx, topology_idx, P_i_cand, P_o, ck))
                            else:
                                future = pool.apply_async(
                                    _decompose_one, (Umtx, mini_topology)
                                )
                                stage2_futures.append((partition_idx, topology_idx, P_i_cand, P_o, ck, future))

            # Process Stage 2 cache hits
            for partition_idx, topology_idx, P_i_cand, P_o, ck in stage2_cached:
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                synth_circuit, synth_params = decomp_cache[ck]
                pair_key = (partition_idx, topology_idx)
                self._add_result_with_auts(
                    results_map[partition_idx], (tuple(P_i_cand), P_o),
                    synth_circuit, synth_params, topology_idx,
                    N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                )

            # Collect Stage 2 pool results
            cache_hits_s2 = len(stage2_cached)
            for partition_idx, topology_idx, P_i_cand, P_o, ck, future in tqdm(
                stage2_futures, desc=f"Stage 2 Synthesis ({cache_hits_s2} cached)",
                disable=disable_pbar
            ):
                synth_circuit, synth_params = future.get()
                decomp_cache[ck] = (synth_circuit, synth_params)
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                pair_key = (partition_idx, topology_idx)
                self._add_result_with_auts(
                    results_map[partition_idx], (tuple(P_i_cand), P_o),
                    synth_circuit, synth_params, topology_idx,
                    N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                )

        return results_map

    # ------------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------------

    def Partition_Aware_Mapping(self, circ: Circuit, orig_parameters: np.ndarray):
        N = circ.get_Qbit_Num()


        optimized_partitions = self.SynthesizeWideCircuit(circ, orig_parameters)

        for partition in optimized_partitions:
            if isinstance(partition, PartitionSynthesisResult):
                partition._topology = self.topology
                partition._topology_cache = self._topology_cache

        DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)

        D = self.compute_distances_bfs(N)
        scoring_partitions = self._build_scoring_partitions(optimized_partitions)

        n_iterations = self.config.get('sabre_iterations', 1)
        n_trials = self.config.get('n_layout_trials', 1)
        random_seed = self.config.get('random_seed', 42)
        do_cleanup = self.config.get('cleanup', True)
        routing_start = time.time()
        if n_iterations == 0:
            # Single forward pass from identity layout
            F = self.get_initial_layer(IDAG, N, optimized_partitions)
            partition_order, pi, pi_initial = self.Heuristic_Search(
                F, pi=np.arange(N), DAG=DAG, IDAG=IDAG,
                optimized_partitions=optimized_partitions,
                scoring_partitions=scoring_partitions, D=D,
            )
        else:
            if do_cleanup:
                from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization
                cleanup_config = dict(self.config)
                cleanup_config['topology'] = self.topology
                cleanup_config['routed'] = True
                cleanup_config['test_subcircuits'] = False
                cleanup_config['test_final_circuit'] = False
                cleanup_config['global_min'] = False
                wco = qgd_Wide_Circuit_Optimization(cleanup_config)

                # Save single-qubit partition circuits before trial loop
                saved_sq_circuits = {i: p.circuit for i, p in enumerate(optimized_partitions)
                                     if isinstance(p, SingleQubitPartitionResult)}

                best_circuit = best_params = best_pi_init = best_pi = None
                best_cost = float('inf')

                for trial in range(max(1, n_trials)):
                    rng = np.random.RandomState(random_seed + trial) if n_trials > 1 else None
                    pi = np.arange(N)

                    for iteration in range(n_iterations):
                        # Reverse pass: walk DAG backwards (swap DAG↔IDAG)
                        F_rev = self.get_final_layer(DAG, N, optimized_partitions)
                        pi, _ = self._heuristic_search_layout_only(
                            F_rev, pi, IDAG, DAG, optimized_partitions, scoring_partitions, D,
                            rng=rng,
                            reverse=True,
                        )

                        # Forward layout-only pass (skip on last iteration)
                        if iteration < n_iterations - 1:
                            F_fwd = self.get_initial_layer(IDAG, N, optimized_partitions)
                            pi, _ = self._heuristic_search_layout_only(
                                F_fwd, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
                                rng=rng,
                            )

                    # Restore single-qubit partition circuits before full forward pass
                    for i, orig in saved_sq_circuits.items():
                        optimized_partitions[i].circuit = orig.copy()

                    # Full forward pass
                    F_trial = self.get_initial_layer(IDAG, N, optimized_partitions)
                    partition_order, pi_out, pi_init = self.Heuristic_Search(
                        F_trial, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
                    )

                    # Build circuit + cleanup
                    trial_circuit, trial_params = self.Construct_circuit_from_HS(
                        partition_order, optimized_partitions, N,
                    )
                    pre_cleanup_cnots = trial_circuit.get_Gate_Nums().get('CNOT', 0)
                    trial_circuit, trial_params = wco.OptimizeWideCircuit(
                        trial_circuit.get_Flat_Circuit(), trial_params
                    )

                    cost = trial_circuit.get_Gate_Nums().get('CNOT', 0)

                    if cost < best_cost:
                        best_cost = cost
                        best_pre_cleanup = pre_cleanup_cnots
                        best_circuit, best_params = trial_circuit, trial_params
                        best_pi_init, best_pi = pi_init, pi_out

                final_circuit, final_parameters = best_circuit, best_params
                pi_initial, pi = best_pi_init, best_pi

            else:
                best_pi = None
                best_cost = float('inf')

                for trial in range(max(1, n_trials)):
                    rng = np.random.RandomState(random_seed + trial) if n_trials > 1 else None
                    pi = np.arange(N)

                    for iteration in range(n_iterations):
                        # Reverse pass: walk DAG backwards (swap DAG↔IDAG)
                        F_rev = self.get_final_layer(DAG, N, optimized_partitions)
                        pi, _ = self._heuristic_search_layout_only(
                            F_rev, pi, IDAG, DAG, optimized_partitions, scoring_partitions, D,
                            rng=rng,
                            reverse=True,
                        )

                        # Forward layout-only pass (skip on last iteration — real pass follows)
                        if iteration < n_iterations - 1:
                            F_fwd = self.get_initial_layer(IDAG, N, optimized_partitions)
                            pi, _ = self._heuristic_search_layout_only(
                                F_fwd, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
                                rng=rng,
                            )

                    # Score this trial: deterministic forward layout-only pass
                    F_eval = self.get_initial_layer(IDAG, N, optimized_partitions)
                    _, cost = self._heuristic_search_layout_only(
                        F_eval, pi.copy(), DAG, IDAG, optimized_partitions, scoring_partitions, D,
                        rng=None,
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_pi = pi.copy()

                # Final forward pass — builds actual circuits
                F = self.get_initial_layer(IDAG, N, optimized_partitions)
                partition_order, pi, pi_initial = self.Heuristic_Search(
                    F, best_pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
                )

        if do_cleanup and n_iterations > 0:
            # Cleanup already done per-trial
            self._routing_time = time.time() - routing_start
            self._cnot_pre_cleanup = best_pre_cleanup
        else:
            final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order, optimized_partitions, N)
            self._routing_time = time.time() - routing_start
            self._cnot_pre_cleanup = final_circuit.get_Gate_Nums().get('CNOT', 0)

            if self.config.get('cleanup', True):
                from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization
                cleanup_config = dict(self.config)
                cleanup_config['topology'] = self.topology
                cleanup_config['routed'] = True
                cleanup_config['test_subcircuits'] = False
                cleanup_config['test_final_circuit'] = False
                cleanup_config['global_min'] = False
                wco = qgd_Wide_Circuit_Optimization(cleanup_config)
                final_circuit, final_parameters = wco.OptimizeWideCircuit(
                    final_circuit.get_Flat_Circuit(), final_parameters
                )

        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def _select_best_candidate(self, partition_candidates, scores, rng=None):
        """Select best candidate, with optional stochastic tie-breaking."""
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        tolerance = self.config.get('score_tolerance', 0.05)

        if rng is not None and min_score > 0:
            threshold = min_score * (1 + tolerance)
            close_indices = np.where(scores_array <= threshold)[0]
            if len(close_indices) > 1:
                return partition_candidates[rng.choice(close_indices)]
            return partition_candidates[close_indices[0]]
        else:
            return partition_candidates[np.argmin(scores_array)]

    def _prefilter_candidates(self, partition_candidates, pi, D, top_k, reverse=False):
        """Pre-filter candidates using cheap swap-count estimate before full A* scoring."""
        if len(partition_candidates) <= top_k:
            return partition_candidates
        local_cost_weight = self.config.get('local_cost_weight', 0.1)
        estimates = np.array([
            pc.estimate_swap_count(pi, D, reverse=reverse) * 3 + local_cost_weight * pc.cnot_count
            for pc in partition_candidates
        ])
        top_k_indices = np.argpartition(estimates, top_k)[:top_k]
        return [partition_candidates[i] for i in top_k_indices]

    def _bfs_shortest_path(self, src, dst):
        """BFS shortest path on self._adj. Returns list of physical nodes
        from src to dst (inclusive); empty list if unreachable."""
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            node = q.popleft()
            for nb in self._adj[node]:
                if nb in parent:
                    continue
                parent[nb] = node
                if nb == dst:
                    path = [dst]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(nb)
        return []

    @staticmethod
    def _apply_swaps_to_pi(pi, swaps):
        """Return a new pi after applying a list of (phys_a, phys_b) swaps."""
        pi_new = [int(x) for x in pi]
        n = len(pi_new)
        p2v = [0] * n
        for q in range(n):
            p2v[pi_new[q]] = q
        for P1, P2 in swaps:
            q1, q2 = p2v[P1], p2v[P2]
            p2v[P1], p2v[P2] = q2, q1
            pi_new[q1], pi_new[q2] = P2, P1
        return pi_new

    def _release_valve(self, F, pi, D, canonical_data):
        """Force progress on the easiest F partition's hardest pair.

        Picks the F partition whose worst-pair distance under pi is smallest
        (cheapest to bridge). BFS-routes that pair along the shortest path,
        applying swaps from both ends toward the middle — LightSABRE §II.7.

        Returns (swap_list, pi_new). Empty swap list if everything is already
        adjacent or no eligible partition exists.
        """
        best = None
        for p_idx in F:
            entry = canonical_data.get(p_idx)
            if entry is None or entry['edges_u'] is None:
                continue
            eu, ev = entry['edges_u'], entry['edges_v']
            worst_d = 0
            worst_pair = None
            for i in range(len(eu)):
                u, v = int(eu[i]), int(ev[i])
                d = D[int(pi[u])][int(pi[v])]
                if d > worst_d:
                    worst_d = d
                    worst_pair = (u, v)
            if worst_d <= 1 or worst_pair is None:
                continue
            if best is None or worst_d < best[0] or (worst_d == best[0] and p_idx < best[1]):
                best = (worst_d, p_idx, worst_pair[0], worst_pair[1])

        if best is None:
            return [], list(pi)

        _, _, u, v = best
        path = self._bfs_shortest_path(int(pi[u]), int(pi[v]))
        if len(path) < 2:
            return [], list(pi)

        k = len(path) - 1
        m = k // 2
        swaps = []
        for i in range(m):
            swaps.append((path[i], path[i + 1]))
        for i in range(k, m + 1, -1):
            swaps.append((path[i], path[i - 1]))

        pi_new = self._apply_swaps_to_pi(pi, swaps)
        return swaps, pi_new

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D):
        pi_initial = pi.copy()

        resolved_partitions = [False] * len(DAG)
        partition_order = []
        step = 0

        # Drain initial single-qubit partitions from F, recursively resolving
        # any single-qubit descendants that become ready.  Children that are
        # multi-qubit are pushed into F for the main search loop.
        queue = [p for p in F if isinstance(optimized_partitions[p], SingleQubitPartitionResult)]
        while queue:
            partition_idx = queue.pop()
            if resolved_partitions[partition_idx]:
                continue
            if partition_idx in F:
                F.remove(partition_idx)
            single_qubit_part = optimized_partitions[partition_idx]
            original_qubit = int(single_qubit_part.involved_qbits[0])
            circuit_qubit = int(single_qubit_part.circuit.get_Qbits()[0])
            single_qubit_part.circuit = single_qubit_part.circuit.Remap_Qbits({circuit_qubit: int(pi[original_qubit])}, max(D.shape))
            partition_order.append(single_qubit_part)
            resolved_partitions[partition_idx] = True
            for child in DAG[partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            queue.append(child)
                        else:
                            F.append(child)

        # Initialize progress bar
        total_partitions = len(DAG)
        pbar = tqdm(total=total_partitions, desc="Heuristic Search",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} resolved',
                   disable=self.config.get('progressbar', 0) == False)

        max_E_size = self.config.get('max_E_size', 20)
        max_lookahead = self.config.get('max_lookahead', 4)
        E_W = self.config.get('E_weight', 0.5)
        E_alpha = self.config.get('E_alpha', 0.9)

        canonical_data = self._build_canonical_neighbor_data(scoring_partitions, reverse=False)

        valve_enabled = self.config.get('release_valve_enabled', True)
        valve_threshold = self.config.get('release_valve_threshold', 20)
        swaps_since_clean = 0

        while len(F) != 0:
                if valve_enabled and swaps_since_clean > valve_threshold:
                    valve_swaps, pi_bridged = self._release_valve(F, pi, D, canonical_data)
                    if valve_swaps:
                        partition_order.append(construct_swap_circuit(valve_swaps, len(pi)))
                        pi = np.asarray(pi_bridged)
                    swaps_since_clean = 0
                    continue

                partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
                if len(partition_candidates) == 0:
                    break

                top_k = self.config.get('prefilter_top_k', 50)
                partition_candidates = self._prefilter_candidates(partition_candidates, pi, D, top_k)

                F_snapshot = tuple(F)

                E = self.generate_extended_set(
                    F, DAG, IDAG, resolved_partitions, optimized_partitions,
                    max_E_size=max_E_size, max_lookahead=max_lookahead
                )

                scores = [
                        self.score_partition_candidate(
                            partition_candidate,
                            F_snapshot,
                            pi,
                            scoring_partitions,
                            D,
                            self._swap_cache,
                            E=E,
                            W=E_W,
                            alpha=E_alpha,
                            canonical_data=canonical_data,
                            adj=self._adj,
                            local_cost_weight=self.config.get('local_cost_weight', 0.1),
                        )
                        for partition_candidate in partition_candidates
                    ]
                min_partition_candidate = self._select_best_candidate(partition_candidates, scores)

                F.remove(min_partition_candidate.partition_idx)
                resolved_partitions[min_partition_candidate.partition_idx] = True
                resolved_count = sum(resolved_partitions)
                pbar.n = resolved_count
                pbar.refresh()

                swap_order, pi = min_partition_candidate.transform_pi(
                    pi, D, self._swap_cache, adj=self._adj,
                )
                if len(swap_order) != 0:
                    partition_order.append(construct_swap_circuit(swap_order, len(pi)))
                    swaps_since_clean += len(swap_order)
                else:
                    swaps_since_clean = 0

                partition_order.append(min_partition_candidate)
                children = list(DAG[min_partition_candidate.partition_idx])
                step += 1
                while len(children) != 0:
                    child = children.pop(0)
                    parents_resolved = True
                    for parent in IDAG[child]:
                        parents_resolved *= resolved_partitions[parent]
                    if (not resolved_partitions[child] and child not in F) and parents_resolved:
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            child_partition = optimized_partitions[child]
                            original_qubit = int(child_partition.involved_qbits[0])
                            circuit_qubit = int(child_partition.circuit.get_Qbits()[0])
                            child_partition.circuit = child_partition.circuit.Remap_Qbits({circuit_qubit: int(pi[original_qubit])},max(D.shape))
                            partition_order.append(child_partition)
                            resolved_partitions[child] = True
                            resolved_count = sum(resolved_partitions)
                            pbar.n = resolved_count
                            pbar.refresh()
                            children.extend(DAG[child])
                        else:
                            F.append(child)

        pbar.close()
        return partition_order, pi, pi_initial

    def _heuristic_search_layout_only(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, rng=None, reverse=False):
        """Run heuristic search but only track layout (pi). No circuit modification.

        Args:
            reverse: When True, swap P_i/P_o roles in scoring and layout
                     updates (used for backward passes in SABRE iterations).

        Returns:
            (pi, total_swaps): final layout and total number of SWAPs accumulated.
        """
        resolved_partitions = [False] * len(DAG)
        total_swaps = 0

        # Resolve initial single-qubit partitions, recursively draining any
        # single-qubit descendants.  Multi-qubit descendants go into F.
        queue = [p for p in F if isinstance(optimized_partitions[p], SingleQubitPartitionResult)]
        while queue:
            partition_idx = queue.pop()
            if resolved_partitions[partition_idx]:
                continue
            if partition_idx in F:
                F.remove(partition_idx)
            resolved_partitions[partition_idx] = True
            for child in DAG[partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            queue.append(child)
                        else:
                            F.append(child)

        max_E_size = self.config.get('max_E_size', 20)
        max_lookahead = self.config.get('max_lookahead', 4)
        E_W = self.config.get('E_weight', 0.5)
        E_alpha = self.config.get('E_alpha', 0.9)

        canonical_data = self._build_canonical_neighbor_data(scoring_partitions, reverse=reverse)

        while F:
            partition_candidates = self.obtain_partition_candidates(F, optimized_partitions)
            if not partition_candidates:
                break

            top_k = self.config.get('prefilter_top_k', 50)
            partition_candidates = self._prefilter_candidates(partition_candidates, pi, D, top_k, reverse=reverse)

            F_snapshot = tuple(F)

            E = self.generate_extended_set(
                F, DAG, IDAG, resolved_partitions, optimized_partitions,
                max_E_size=max_E_size, max_lookahead=max_lookahead
            )

            scores = [
                self.score_partition_candidate(
                    pc, F_snapshot, pi, scoring_partitions, D,
                    self._swap_cache,
                    E=E, W=E_W, alpha=E_alpha,
                    reverse=reverse,
                    canonical_data=canonical_data,
                    adj=self._adj,
                    local_cost_weight=self.config.get('local_cost_weight', 0.1),
                )
                for pc in partition_candidates
            ]

            best = self._select_best_candidate(partition_candidates, scores, rng=rng)
            F.remove(best.partition_idx)
            resolved_partitions[best.partition_idx] = True

            swaps, pi = best.transform_pi(
                pi, D, self._swap_cache, reverse=reverse, adj=self._adj,
            )
            total_swaps += len(swaps)

            # Promote children
            for child in DAG[best.partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            resolved_partitions[child] = True
                            stack = list(DAG[child])
                            while stack:
                                gc = stack.pop()
                                if not resolved_partitions[gc] and gc not in F:
                                    if all(resolved_partitions[p] for p in IDAG[gc]):
                                        if isinstance(optimized_partitions[gc], SingleQubitPartitionResult):
                                            resolved_partitions[gc] = True
                                            stack.extend(DAG[gc])
                                        else:
                                            F.append(gc)
                        else:
                            F.append(child)

        return pi, total_swaps

    # ------------------------------------------------------------------------
    # Circuit Construction
    # ------------------------------------------------------------------------

    def Construct_circuit_from_HS(self, partition_order, optimized_partitions,N):
        final_circuit = Circuit(N)
        final_parameters = []
        perm_count = 0
        partition_count = 0
        
        for part in partition_order:
            if isinstance(part, Circuit):
                final_circuit.add_Circuit(part)
                perm_count += 1
            elif isinstance(part, SingleQubitPartitionResult):
                final_circuit.add_Circuit(part.circuit)
                final_parameters.append(part.parameters)
                partition_count += 1
            else:
                part_circ, part_parameters = part.get_final_circuit(optimized_partitions,N)
                final_circuit.add_Circuit(part_circ)
                final_parameters.append(part_parameters)
                partition_count += 1
        
        if final_parameters:
            final_parameters = np.concatenate([np.atleast_1d(p).ravel() for p in final_parameters], axis=0)
        else:
            final_parameters = np.array([])
        if not check_circuit_compatibility(final_circuit,self.topology):
            print("ERROR: Final circuit is not compatible with device topology!")
        return final_circuit, final_parameters
    
    # ------------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------------

    @staticmethod
    def _build_canonical_neighbor_data(scoring_partitions, reverse=False):
        """Per partition, keep only the virtual-qubit edges of the lowest-CNOT
        (mini_topology, P_i, P_o) combo — LightSABRE-style: assume each F/E
        partition will be scheduled with its best combo.

        Returns dict {partition_idx: {'edges_u': np.intp[n_edges],
                                       'edges_v': np.intp[n_edges],
                                       'cnot': int}}.
        Partitions with no mini-topology edges have edges_u = edges_v = None.
        """
        data = {}
        for idx, partition in enumerate(scoring_partitions):
            if partition is None:
                continue
            qbit_map_inv = {v: q for q, v in partition.qubit_map.items()}
            best_cnot = None
            best_edges = None
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                for pdx, (P_i, P_o) in enumerate(partition.permutations_pairs[tdx]):
                    cnot = len(partition.circuit_structures[tdx][pdx])
                    if best_cnot is not None and cnot >= best_cnot:
                        continue
                    P_route = P_o if reverse else P_i
                    if mini_topology:
                        edges = [(qbit_map_inv[P_route[u]], qbit_map_inv[P_route[v]])
                                 for u, v in mini_topology]
                    else:
                        edges = []
                    best_cnot = cnot
                    best_edges = edges
            if best_cnot is None:
                continue
            if best_edges:
                eu = np.array([e[0] for e in best_edges], dtype=np.intp)
                ev = np.array([e[1] for e in best_edges], dtype=np.intp)
            else:
                eu = ev = None
            data[idx] = {'edges_u': eu, 'edges_v': ev, 'cnot': best_cnot}
        return data

    @staticmethod
    def score_partition_candidate(partition_candidate, F, pi, scoring_partitions, D, swap_cache,
                                  E=None, W=0.5, alpha=0.9, reverse=False,
                                  canonical_data=None, adj=None,
                                  local_cost_weight=0.1):
        """LightSABRE-style relative scoring (arXiv:2409.08368, eq. 1).

        H = 3 * |swaps|
          + local_cost_weight * cand.cnot_count
          + (1/|F'|) * average routing cost over F \\ {cand}
          + (W/|E|)  * alpha^d-decayed routing cost over E
        """
        swaps, output_perm = partition_candidate.transform_pi(
            pi, D, swap_cache, reverse=reverse, adj=adj, neighbor_info=None,
        )
        score = 3.0 * len(swaps)
        score += local_cost_weight * partition_candidate.cnot_count

        if canonical_data is None:
            return score

        output_perm_arr = np.asarray(output_perm, dtype=np.intp)
        D_arr = np.asarray(D)
        cand_idx = partition_candidate.partition_idx

        # Basic component: average dist over F \ {cand}
        f_sum = 0.0
        n_other = 0
        for partition_idx in F:
            if partition_idx == cand_idx:
                continue
            entry = canonical_data.get(partition_idx)
            if entry is None:
                continue
            n_other += 1
            eu = entry['edges_u']
            if eu is None:
                continue
            phys_u = output_perm_arr[eu]
            phys_v = output_perm_arr[entry['edges_v']]
            f_sum += 3.0 * np.maximum(0, D_arr[phys_u, phys_v] - 1).sum()
        if n_other > 0:
            score += f_sum / n_other

        # Lookahead component: alpha^depth-decayed average over E
        if E:
            e_sum = 0.0
            for partition_idx, depth in E:
                if partition_idx == cand_idx:
                    continue
                entry = canonical_data.get(partition_idx)
                if entry is None:
                    continue
                eu = entry['edges_u']
                if eu is None:
                    continue
                phys_u = output_perm_arr[eu]
                phys_v = output_perm_arr[entry['edges_v']]
                d_cost = 3.0 * np.maximum(0, D_arr[phys_u, phys_v] - 1).sum()
                e_sum += (alpha ** depth) * d_cost
            score += W * e_sum / len(E)

        return score

    # ------------------------------------------------------------------------
    # Extended Set
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_extended_set(F, DAG, IDAG, resolved_partitions, optimized_partitions,
                              max_E_size=20, max_lookahead=4):
        """
        Generate SABRE-style extended set: multi-qubit partitions near the
        front layer, up to ``max_lookahead`` levels deep and ``max_E_size``
        entries.  Returns list of (partition_idx, depth) tuples.
        """
        E = []
        E_set = set()
        F_set = set(F)

        for front_idx in F:
            if len(E) >= max_E_size:
                break

            # BFS from front_idx through DAG children
            queue = []  # (child_idx, depth)
            for child in DAG[front_idx]:
                queue.append((child, 1))

            while queue and len(E) < max_E_size:
                child_idx, depth = queue.pop(0)
                if depth > max_lookahead:
                    continue
                if child_idx in E_set or child_idx in F_set:
                    continue
                if resolved_partitions[child_idx]:
                    continue

                # Check all parents resolved (except those still in F)
                parents_resolved = all(
                    resolved_partitions[p] or p in F_set
                    for p in IDAG[child_idx]
                )
                if not parents_resolved:
                    continue

                # Skip single-qubit partitions — follow through them
                if isinstance(optimized_partitions[child_idx], SingleQubitPartitionResult):
                    for grandchild in DAG[child_idx]:
                        queue.append((grandchild, depth))
                    continue

                E.append((child_idx, depth))
                E_set.add(child_idx)

                if depth < max_lookahead:
                    for grandchild in DAG[child_idx]:
                        queue.append((grandchild, depth + 1))

        return E

    # ------------------------------------------------------------------------
    # Candidate Generation
    # ------------------------------------------------------------------------

    def obtain_partition_candidates(self, F, optimized_partitions):
        partition_candidates = []
        for partition_idx in F:
            partition = optimized_partitions[partition_idx]
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                # Use pre-computed topology candidates if available, otherwise compute and cache
                if hasattr(partition, 'get_topology_candidates'):
                    topology_candidates = partition.get_topology_candidates(tdx)
                else:
                    topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(partition.permutations_pairs[tdx]):
                        partition_candidates.append(PartitionCandidate(partition_idx,tdx,pdx,partition.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition.qubit_map,partition.involved_qbits,cnot_count=partition.cnot_counts[tdx][pdx]))
        return partition_candidates

    # ------------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------------
        
    def get_initial_layer(self, IDAG, N, optimized_partitions):
        initial_layer = []
        active_qbits = list(range(N))
        for idx in range(len(IDAG)):
            if len(IDAG[idx]) == 0:
                initial_layer.append(idx)
                for qbit in optimized_partitions[idx].involved_qbits:
                    active_qbits.remove(qbit)
            if len(active_qbits) == 0:
                break
        return initial_layer

    def get_final_layer(self, DAG, N, optimized_partitions):
        final_layer = []
        active_qbits = list(range(N))
        for idx in range(len(DAG) - 1, -1, -1):
            if len(DAG[idx]) == 0:
                final_layer.append(idx)
                for qbit in optimized_partitions[idx].involved_qbits:
                    if qbit in active_qbits:
                        active_qbits.remove(qbit)
            if len(active_qbits) == 0:
                break
        return final_layer
            
    def construct_DAG_and_IDAG(self, optimized_partitions):
        DAG = []
        IDAG = []
        for idx in range(len(optimized_partitions)):
            parents = []
            children = []
            if idx != len(optimized_partitions)-1:
                involved_qbits_current = optimized_partitions[idx].involved_qbits.copy()
                for next_idx in range(idx+1, len(optimized_partitions)):
                    involved_qbits_next = optimized_partitions[next_idx].involved_qbits
                    intersection = [i for i in involved_qbits_current if i in involved_qbits_next]
                    if len(intersection) > 0:
                        children.append(next_idx)
                    for intersection_qbit in intersection:
                        involved_qbits_current.remove(intersection_qbit)
                    if len(involved_qbits_current) == 0:
                        break
            if idx != 0:
                involved_qbits_current = optimized_partitions[idx].involved_qbits.copy()
                for prev_idx in range(idx-1, -1, -1):
                    involved_qbits_prev = optimized_partitions[prev_idx].involved_qbits
                    intersection = [i for i in involved_qbits_current if i in involved_qbits_prev]
                    if len(intersection) > 0:
                        parents.append(prev_idx)
                    for intersection_qbit in intersection:
                        involved_qbits_current.remove(intersection_qbit)
                    if len(involved_qbits_current) == 0:
                        break
            DAG.append(children)
            IDAG.append(parents)
        return DAG, IDAG
    
    # ------------------------------------------------------------------------
    # Distance & Layout
    # ------------------------------------------------------------------------

    def compute_distances_bfs(self, N):
        """BFS distance computation - faster than Floyd-Warshall."""
        D = np.ones((N, N)) * np.inf
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in self.config['topology']:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(N):
            D[start][start] = 0
            queue = deque([(start, 0)])
            visited = {start}
            
            while queue:
                node, dist = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        D[start][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        
        # Store adjacency list for reuse by A* routing
        self._adj = [list(adj[i]) for i in range(N)]

        return D


    def generate_DAG_levels(self, circuit):
        """
        Generate DAG levels - groups gates by their topological level.
        
        Args:
            circuit: The quantum circuit to analyze
            
        Returns:
            List of lists, where each inner list contains gate indices at the same DAG level.
            Level 0 contains gates with no parents, level 1 contains gates whose parents
            are all at level 0, etc.
        """ 
        gates = circuit.get_Gates()
        num_gates = len(gates)
        
        # Build parent count for each gate
        parent_counts = [0] * num_gates
        children_map = [[] for _ in range(num_gates)]
        
        for gate_idx in range(num_gates):
            gate = gates[gate_idx]
            parents = circuit.get_Parents(gate)
            parent_counts[gate_idx] = len(parents)
            
            # Build children map
            children = circuit.get_Children(gate)
            for child_idx in children:
                children_map[gate_idx].append(child_idx)
        
        # Initialize level 0 with gates that have no parents
        levels = []
        current_level = []
        processed = [False] * num_gates
        
        # Find gates with no parents (level 0)
        for gate_idx in range(num_gates):
            if parent_counts[gate_idx] == 0:
                current_level.append(gate_idx)
                processed[gate_idx] = True
        
        # Process levels using BFS
        while current_level:
            levels.append(current_level)
            next_level = []
            
            # Process all gates in current level
            for gate_idx in current_level:
                # Decrement parent counts for children
                for child_idx in children_map[gate_idx]:
                    parent_counts[child_idx] -= 1
                    # If all parents are processed, add to next level
                    if parent_counts[child_idx] == 0 and not processed[child_idx]:
                        next_level.append(child_idx)
                        processed[child_idx] = True
            
            current_level = next_level
        
        return levels

