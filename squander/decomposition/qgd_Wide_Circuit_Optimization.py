"""
Implementation to optimize wide circuits (i.e. circuits with many qubits) by    partitioning the circuit into smaller partitions and redecompose the smaller partitions
"""

from squander.decomposition.qgd_N_Qubit_Decomposition_Tree_Search import qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search
from squander.decomposition.qgd_N_Qubit_Decomposition_Tabu_Search import qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search
from squander import N_Qubit_Decomposition_adaptive
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

    return gate_counts.get('CNOT', 0)




    



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
        
        #testing the fields of config 
        strategy = config[ 'strategy' ]
        allowed_startegies = ['TreeSearch', 'TabuSearch', 'Adaptive' ]
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
    def DecomposePartition( Umtx: np.ndarray, config: dict, mini_topology = None ) -> Circuit:
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


        print( "Decomposition error: ", cDecompose.get_Decomposition_Error() )

        if tolerance < cDecompose.get_Decomposition_Error():
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
    def PartitionDecompositionProcess( subcircuit: Circuit, subcircuit_parameters: np.ndarray, config: dict ) -> (Circuit, np.ndarray):
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

        # get the unitary representing the circuit
        unitary = remapped_subcircuit.get_Matrix( subcircuit_parameters )

        # decompose a small unitary into a new circuit
        decomposed_circuit, decomposed_parameters = qgd_Wide_Circuit_Optimization.DecomposePartition( unitary, config, mini_topology )

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




    def OptimizeWideCircuit( self, circ: Circuit, orig_parameters: np.ndarray, global_min=False, prepartitioning=None ) -> (Circuit, np.ndarray):
        """
        Call to optimize a wide circuit (i.e. circuits with many qubits) by
        partitioning the circuit into smaller partitions and redecompose the smaller partitions


        Args: 

            circ ( Circuit ) A circuit to be partitioned

            orig_parameters ( np.ndarray ) A parameter array associated with the input circuit

        Return:

            Returns with the optimized circuit and the corresponding parameter array

        """
        if self.config["topology"] != None:
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
            partitined_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy="kahn" )

        qbit_num_orig_circuit = circ.get_Qbit_Num()


        subcircuits = partitined_circuit.get_Gates()




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
                callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, x[0]], [subcircuit_parameters, x[1]] ) if config["topology"]==None else lambda x : (x[0],x[1])


                # call a process to decompose a subcircuit
                config = self.config if not global_min or len(subcircuit.get_Qbits()) < 4 else {**self.config, 'strategy': "Adaptive"}
                async_results[partition_idx]  = pool.apply_async( self.PartitionDecompositionProcess, (subcircuit, subcircuit_parameters, config), callback=callback_fnc )



            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx, subcircuit in enumerate( subcircuits ):
    
                new_subcircuit, new_parameters = async_results[partition_idx].get( timeout = None )

                '''
                if subcircuit != new_subcircuit:

                    print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
                    print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
                '''

                optimized_subcircuits[ partition_idx ] = new_subcircuit
                optimized_parameter_list[ partition_idx ] = new_parameters


        # construct the wide circuit from the optimized suncircuits
        if global_min:
            max_gates = max(len(c.get_Gates()) for c in optimized_subcircuits)
            def to_cost(d): return d.get('CNOT', 0)*max_gates + sum(d[x] for x in d if x != 'CNOT')
            weights = [to_cost(circ.get_Gate_Nums()) for circ in optimized_subcircuits[:len(allparts)]]
            L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
            parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L], fusion_info)
            L = topo_sort_partitions(circ, self.max_partition_size, parts)
            return self.OptimizeWideCircuit(circ, orig_parameters, global_min=False, prepartitioning=[parts[i] for i in L])
            """
            Lgate = [set(allparts[i]) for i in L]
            for part in Lgate:
                surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
                part.update(*(single_qubit_chains_prepost[v] for v in surrounded_chains))
            gates = set.union(*Lgate)
            chain_idx = {}
            for i, chain in enumerate(single_qubit_chains):
                if chain[0] in gates: continue
                chain_idx[len(Lgate)] = len(allparts) + i
                Lgate.append(set(chain))
            Lidx = topo_sort_partitions(circ, self.max_partition_size, Lgate)
            wide_circuit = Circuit( qbit_num_orig_circuit )
            params = []
            for i in Lidx:
                c = Circuit( qbit_num_orig_circuit )
                idx = L[i] if i < len(L) else chain_idx[i]
                wide_circuit.add_Circuit(optimized_subcircuits[idx])
                params.append(optimized_parameter_list[idx])
            wide_parameters = np.concatenate(params, axis=0)
            """
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
        self.config["routed"] = False
        return Squander_remapped_circuit, parameters_remapped_circuit