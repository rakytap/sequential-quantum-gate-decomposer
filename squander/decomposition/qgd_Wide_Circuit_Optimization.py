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
from multiprocessing import Process, Pool
import os


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config ):
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

    return gate_counts.get('CNOT', 0) +  3*gate_counts.get('SWAP', 0)



class N_Qubit_Decomposition_Guided_Tree(N_Qubit_Decomposition_custom):
    def __init__(self, Umtx, config, accelerator_num, topology):
        super().__init__(Umtx, config=config, accelerator_num=accelerator_num)
        self.Umtx = Umtx #already conjugate transposed
        self.qbit_num = Umtx.shape[0].bit_length() -1
        self.config = config
        self.accelerator_num = accelerator_num
        if topology is None:
            topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        self.topology = topology
    def enumerate_unordered_cnot_BFS(n: int, topology=None):
        # Precompute unordered pairs
        topology = [(i, j) for i in range(n) for j in range(i+1, n)] if topology is None else topology
        prior_level_info = None
        while True:
            visited, seq_pairs_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(n, topology, prior_level_info)
            if not res: break
            for item in res: yield item
            prior_level_info = (visited, seq_pairs_of, list(x[0] for x in reversed(res)))

    def enumerate_unordered_cnot_BFS_level(n: int, topology=None, prior_level_info=None):
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

            # Yield the root
            return visited, seq_pairs_of, [(start_key, [], [])]
        else:
            visited, seq_pairs_of, q = prior_level_info
        res = []
        new_seq_pairs_of = {}

        while q:
            A = q.pop()
            last_pairs = seq_pairs_of[A]
            for p in topology:
                # Try both directions, but record the *same* unordered step 'p'
                for mv in (p, (p[1], p[0])):
                    #CNOT left
                    if mv[0] == mv[1]: B = A
                    else: B = list(A); B[mv[1]] ^= B[mv[0]]; B = tuple(B)
    
                    if B in visited: continue  # already discovered at minimal depth

                    visited.add(B)
                    new_seq_pairs_of[B] = last_pairs + [p]

                    # Emit as soon as we discover the state (BFS → minimal depth)
                    res.append((B, new_seq_pairs_of[B]))
        return visited, new_seq_pairs_of, res
    def build_sequence():
        #https://oeis.org/A002884
        #1, 1, 4, 88, 9556 4526605 {0: 1, 1: 1, 2: 1, 3: 1} {0: 1, 1: 3, 2: 9, 3: 22, 4: 33, 5: 18, 6: 2} {0: 1, 1: 6, 2: 33, 3: 160, 4: 647, 5: 2005, 6: 3665, 7: 2588, 8: 445, 9: 6}
        #{0: 1, 1: 10, 2: 85, 3: 650, 4: 4475, 5: 27375, 6: 142499, 7: 580482, 8: 1501297, 9: 1738232, 10: 517884, 11: 13591, 12: 24} 
        for i in range(2, 6):
            d = {}
            for x in set(tuple(x[1]) for x in N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(i)):
                d[len(x)] = d.get(len(x), 0) + 1
            print({x: d[x] for x in sorted(d)}, sum(d.values()))
    def extract_bits(x, pos):
        return sum(((x >> p) & 1) << i for i, p in enumerate(pos))
    def build_osr_matrix(U, n, A):
        A_sorted = list(sorted(A))
        B = list(sorted(set(range(n)) - set(A_sorted)))
        N = 1 << n
        dA = 1 << len(A)
        dB = 1 << len(B)
        m_rows = dA * dA
        m_cols = dB * dB
        """
        T = U.reshape([2]* (2*n))
        i_axes_A = A_sorted
        o_axes_A = [q + n for q in A_sorted]
        i_axes_B = B
        o_axes_B = [q + n for q in B]
        perm = i_axes_A + o_axes_A + i_axes_B + o_axes_B
        return T.transpose(perm).reshape((m_rows, m_cols))
        """
        M = np.zeros((m_rows, m_cols), dtype=U.dtype)
        # Row-major indexing: U[in + out*N] is element (in, out)
        for in_ in range(N):
            a = N_Qubit_Decomposition_Guided_Tree.extract_bits(in_, A_sorted) * dA
            b = N_Qubit_Decomposition_Guided_Tree.extract_bits(in_, B) * dB
            for out in range(N):
                ap = N_Qubit_Decomposition_Guided_Tree.extract_bits(out, A_sorted)
                bp = N_Qubit_Decomposition_Guided_Tree.extract_bits(out, B)
                r = a + ap  # row in M
                c = b + bp  # col in M
                M[r, c] = U[in_, out]
        return M
    def numerical_rank_osr(M, tol=1e-10):
        s = np.linalg.svd(M, compute_uv=False)
        #print(s)
        return int(np.sum(s > tol)), s[0]#/s[0]
    def operator_schmidt_rank(U, n, A, tol=1e-10):
        return N_Qubit_Decomposition_Guided_Tree.numerical_rank_osr(N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A), tol)
    def osr_cnot_rank_second_singular(U, n, cuts, tol=1e-10):
        def ceil_log2(x): return 0 if x == 0 else (x-1).bit_length()
        return [(ceil_log2(x), y) for A in cuts for x, y in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, n, A, tol),)]
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
    def Run_Decomposition(self, pairs):
        circ = Circuit(self.qbit_num)
        for pair in pairs:
            circ.add_U3(pair[0])
            circ.add_U3(pair[1])
            circ.add_CNOT(pair[0], pair[1])
        for qbit in range(self.qbit_num):
            circ.add_RZ(qbit)
            circ.add_RY(qbit)
            circ.add_RZ(qbit)
        self.set_Gate_Structure(circ)
        for _ in range(self.config.get('restarts', 1)):
            self.set_Optimized_Parameters(np.random.rand(self.get_Parameter_Num())*2*np.pi)
            super().Start_Decomposition()
            params = self.get_Optimized_Parameters()
            self.err = self.Optimization_Problem(params)
            if self.err < self.config.get('tolerance', 1e-8): return True
        return False
    def Start_Decomposition(self):
        self.err = 1.0
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        pair_affects = {
            pair: [i for i,A in enumerate(cuts) if (pair[0] in A) ^ (pair[1] in A)]
            for pair in self.topology
        }
        #because we have U already conjugate transposed, must use prefix order
        B = self.config.get('beam', None)#8*len(self.topology))
        #from squander.gates import S, H
        #compact_clifford_gates = [(), (S, ), (H, ), (S, H), (H, S)] #full set is [I, H, SH, SSH, SSSH, HSH] X [I, S, SS, SSS]
        max_depth = self.config.get('tree_level_max', 14)
        prior_level_info = None
        for depth in range(max_depth+1):
            remaining = max_depth - depth
            visited, seq_pairs_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(self.qbit_num, self.topology, prior_level_info)
            nextprefixes = []
            for path in set(tuple(x[1]) for x in res):
                #if max(x[0] for x in curh) > remaining + 1: continue
                curh = None if len(path)==0 else prefixes[path[:-1]]                
                allh = []
                revpath = tuple(reversed(path))
                check_cuts = pair_affects[path[-1]] if not curh is None else range(len(cuts))
                for _ in range(1):                    
                    for p in (path,) if path==revpath else (path, revpath):
                        if self.Run_Decomposition(p): return
                        U = self.Umtx.copy()
                        self.get_Circuit().apply_to(self.get_Optimized_Parameters(), U)
                        h = N_Qubit_Decomposition_Guided_Tree.osr_cnot_rank_second_singular(U, self.qbit_num, cuts, tol=1e-10)
                        allh.append(h)
                h = min(allh, key=lambda x: (max(x[i][0] for i in check_cuts), sum(x[i][0] for i in check_cuts), sum(x[i][1] for i in check_cuts)))
                #print(new_prefix, curh, h)
                if not curh is None:
                    #print(path, [([x[i] for x in allh], curh[i]) for i in check_cuts])
                    if all(h[i][0] > curh[i][0] for i in check_cuts): continue
                    if all(h[i][0] == curh[i][0] for i in check_cuts) and any(h[i][1] > curh[i][1] for i in check_cuts): continue
                    #if all(h[i][1] > curh[i][1] for i in check_cuts): continue
                    #if sum(h) > sum(curh): continue
                nextprefixes.append((path, h))
            nextprefixes.sort(key=lambda t: (max(x[0] for x in t[1]), sum(x[0] for x in t[1]), sum(x[1] for x in t[1])))
            prefixes = {x[0]: x[1] for x in nextprefixes[:B]}
            prior_level_info = (visited, seq_pairs_of, list(x[0] for x in reversed(res) if tuple(x[1]) in prefixes))
            #print(len(nextprefixes), depth)
    def get_Decomposition_Error(self): return self.err
#N_Qubit_Decomposition_Guided_Tree.build_sequence(); assert False

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
            from squander.gates.qgd_Circuit import CNOT
            newcirc = Circuit(structure.get_Qbit_Num())
            for gate in structure.get_Gates():
                if isinstance(gate, CNOT):
                    newcirc.add_U3(gate.get_Target_Qbit())
                    newcirc.add_U3(gate.get_Control_Qbit())
                    newcirc.add_Gate(gate)
            for qbit in structure.get_Qbits():
                newcirc.add_RZ(qbit)
                newcirc.add_RY(qbit)
                newcirc.add_RZ(qbit)
            cDecompose.set_Gate_Structure( newcirc )
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")


        tolerance = config["tolerance"]
        cDecompose.set_Verbose( config["verbosity"] )
        cDecompose.set_Cost_Function_Variant( 3 )	
        cDecompose.set_Optimization_Tolerance( tolerance )
    

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer( "BFGS" )

        # starting the decomposition
        cDecompose.Start_Decomposition()
            
        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()


        if strategy == "Custom": err = cDecompose.Optimization_Problem(parameters)
        else: err = cDecompose.get_Decomposition_Error()
        #print( "Decomposition error: ", err )
        if tolerance < err:
            return None, None


        return squander_circuit, parameters



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
        if not structure is None:
            structure = structure.Remap_Qbits( qbit_map, qbit_num )

        # get the unitary representing the circuit
        unitary = remapped_subcircuit.get_Matrix( subcircuit_parameters )

        # decompose a small unitary into a new circuit
        decomposed_circuit, decomposed_parameters = qgd_Wide_Circuit_Optimization.DecomposePartition( unitary, config, mini_topology, structure=structure )

        if decomposed_circuit is None:
            return subcircuit, subcircuit_parameters #remaining code will fail, just return original circuit

        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[ value ] = key

        # remap the decomposed circuit in order to insert it into a large circuit
        new_subcircuit = decomposed_circuit.Remap_Qbits( inverse_qbit_map, qbit_num_orig_circuit )


        if config["test_subcircuits"]:
            CompareCircuits( subcircuit, subcircuit_parameters, new_subcircuit, decomposed_parameters, parallel=config["parallel"] )
        


        new_subcircuit = new_subcircuit.get_Flat_Circuit()

        return new_subcircuit, decomposed_parameters


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
        if self.config["topology"] != None and self.config["routed"]==False:
            circ, orig_parameters = self.route_circuit(circ,orig_parameters)

        if global_min:
            from squander.partitioning.ilp import get_all_partitions, _get_topo_order, topo_sort_partitions, ilp_global_optimal, recombine_single_qubit_chains
            allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, self.max_partition_size)
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
        elif prepartitioning is not None:
            from squander.partitioning.kahn import kahn_partition_preparts
            from squander.partitioning.tools import translate_param_order
            partitined_circuit, param_order, _ = kahn_partition_preparts(circ, self.max_partition_size, prepartitioning)
            parameters = translate_param_order(orig_parameters, param_order)
        else:
            partitined_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy=self.config['partition_strategy'] )

        qbit_num_orig_circuit = circ.get_Qbit_Num()


        subcircuits = partitined_circuit.get_Gates()

        #subcircuits = [subcircuits[9]]

        print(len(subcircuits), "partitions found to optimize")


        # the list of optimized subcircuits
        optimized_subcircuits = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list = [None] * len(subcircuits)

        # list of AsyncResult objects
        async_results = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:

            #  code for iterate over partitions and optimize them
            for partition_idx, subcircuit in enumerate( subcircuits ):
        

                # isolate the parameters corresponding to the given sub-circuit
                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
   
    
            
                # callback function done on the master process to compare the new decomposed and the original suncircuit
                callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, x[0]], [subcircuit_parameters, x[1]] ) 

                # call a process to decompose a subcircuit
                config = self.config if not global_min or subcircuit.get_Qbit_Num() < 3 else {**self.config, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}
                config = self.config if structures is None or partition_idx >= len(structures) else {**self.config, 'strategy': 'Custom', 'max_inner_iterations': 10000, 'max_iteration_loops': 4}
                async_results[partition_idx]  = pool.apply_async( self.PartitionDecompositionProcess, (subcircuit, subcircuit_parameters, config,
                                                                                                       None if structures is None or partition_idx >= len(structures) else structures[partition_idx]), 
                                                                 callback=callback_fnc )
            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx, subcircuit in enumerate( subcircuits ):
    
                new_subcircuit, new_parameters = async_results[partition_idx].get( timeout = None )

                '''
                if subcircuit != new_subcircuit:

                    print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
                    print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
                '''
                if partition_idx % 100 == 99: print(partition_idx+1, "partitions optimized")
                #if new_subcircuit.get_Gate_Nums().get('CNOT', 0) < subcircuit.get_Gate_Nums().get('CNOT', 0):
                #    for gate in subcircuit.get_Gates(): print(gate)
                #print(partition_idx, new_subcircuit.get_Gate_Nums().get('CNOT', 0), subcircuit.get_Gate_Nums().get('CNOT', 0))
                #if new_subcircuit.get_Gate_Nums().get('CNOT', 0) == 0 and subcircuit.get_Gate_Nums().get('CNOT', 0) > 0:
                    #for gate in new_subcircuit.get_Gates(): print(gate, gate.get_Target_Qbit(), gate.get_Control_Qbit())
                    #print("----")
                    #for gate in subcircuit.get_Gates(): print(gate, gate.get_Target_Qbit(), gate.get_Control_Qbit())
                if new_subcircuit.get_Gate_Nums().get('CNOT', 0) < subcircuit.get_Gate_Nums().get('CNOT', 0):
                    optimized_subcircuits[ partition_idx ] = new_subcircuit
                    optimized_parameter_list[ partition_idx ] = new_parameters
                else:
                    optimized_subcircuits[ partition_idx ] = subcircuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[ start_idx:end_idx ]
                    optimized_parameter_list[ partition_idx ] = subcircuit_parameters


        # construct the wide circuit from the optimized suncircuits
        if global_min:
            max_gates = max(len(c.get_Gates()) for c in optimized_subcircuits)
            def to_cost(d): return d.get('CNOT', 0)*max_gates + sum(d[x] for x in d if x != 'CNOT')
            weights = [to_cost(circ.get_Gate_Nums()) for circ in optimized_subcircuits[:len(allparts)]]
            L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
            structures = [optimized_subcircuits[i] for i in L]
            parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L], fusion_info)
            L = topo_sort_partitions(circ, self.max_partition_size, parts)
            return self.OptimizeWideCircuit(circ, orig_parameters, global_min=False, prepartitioning=[parts[i] for i in L], structures=[structures[i] for i in L])
        else:
            wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )

        print( "original circuit:    ", circ.get_Gate_Nums()) 
        print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 


        if self.config["test_final_circuit"]:
            CompareCircuits( partitined_circuit, parameters, wide_circuit, wide_parameters )

        
        return wide_circuit, wide_parameters

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):

        sabre = SABRE(circ, self.config["topology"])
        Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(orig_parameters)
        self.config.setdefault("initial_mapping",pi)
        self.config.setdefault("final_mapping",final_pi)
        self.config["routed"] = True
        return Squander_remapped_circuit, parameters_remapped_circuit
