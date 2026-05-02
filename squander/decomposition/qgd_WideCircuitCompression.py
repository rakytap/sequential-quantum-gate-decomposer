"""
Wide-circuit compression: partition large circuits into subcircuits and run
OSR-guided gate-structure compression on each partition.
"""

from squander import N_Qubit_Decomposition_OSR_Compression
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

from squander.partitioning.partition import PartitionCircuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    CNOT_COUNT_DICT,
    CNOTGateCount,
    extract_subtopology,
    qgd_Wide_Circuit_Optimization,
)

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, parent_process
import os, contextlib, time

from typing import List, Tuple, Optional, cast


class qgd_WideCircuitCompression:
    """Optimize wide circuits by partitioning and per-partition OSR compression.

    Each partition is treated as a fixed gate structure. The OSR compression
    decomposer attempts to remove entangling gates while still reproducing the
    partition's unitary within the configured tolerance. If compression fails,
    the original partition is kept.
    """

    def __init__(self, config):
        config.setdefault("parallel", 0)
        config.setdefault("verbosity", 0)
        config.setdefault("tolerance", 1e-8)
        config.setdefault("test_subcircuits", False)
        config.setdefault("test_final_circuit", True)
        config.setdefault("max_partition_size", 3)
        config.setdefault("partition_strategy", "ilp")
        config.setdefault("topology", None)

        if config["parallel"] not in (0, 1, 2):
            raise Exception(
                f"The parallel configuration should be either of [0, 1, 2], got {config['parallel']}."
            )
        if not isinstance(config["verbosity"], int):
            raise Exception("The verbosity parameter should be an integer.")
        if not isinstance(config["tolerance"], float):
            raise Exception("The tolerance parameter should be a float.")
        if not isinstance(config["test_subcircuits"], bool):
            raise Exception("The test_subcircuits parameter should be a bool.")
        if not isinstance(config["test_final_circuit"], bool):
            raise Exception("The test_final_circuit parameter should be a bool.")
        if not isinstance(config["max_partition_size"], int):
            raise Exception("The max_partition_size parameter should be an integer.")

        self.config = config
        self.max_partition_size = config["max_partition_size"]

    @staticmethod
    def CompressPartition(
        subcircuit: Circuit,
        subcircuit_parameters: np.ndarray,
        config: dict,
    ) -> Tuple[Circuit, np.ndarray]:
        """Run OSR compression on a single partition subcircuit.

        Returns the compressed circuit (remapped to the original wide register)
        and its parameters. Falls back to the original subcircuit on failure.
        """
        qbit_num_orig = subcircuit.get_Qbit_Num()
        involved = subcircuit.get_Qbits()
        qbit_num = len(involved)

        qbit_map = {q: i for i, q in enumerate(involved)}
        remapped = subcircuit.Remap_Qbits(qbit_map, qbit_num)

        # restrict OSR mutations to topology edges that survive the partition
        local_config = dict(config)
        if config.get("topology") is not None:
            mini_topology = extract_subtopology(involved, qbit_map, config)
            local_config.setdefault("osr_compression_mutate_full_topology", 0)
        else:
            mini_topology = None

        # the partition unitary is the OSR target
        unitary = remapped.get_Matrix(subcircuit_parameters)

        cDecompose = N_Qubit_Decomposition_OSR_Compression(
            unitary.conj().T,
            qbit_num=qbit_num,
            config=local_config,
            accelerator_num=0,
            topology=mini_topology,
        )
        cDecompose.set_Verbose(config["verbosity"])
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimization_Tolerance(config["tolerance"])
        cDecompose.set_Optimizer("BFGS")

        # supply the existing structure + warm-start parameters
        cDecompose.set_Gate_Structure(remapped)
        cDecompose.set_Optimized_Parameters(subcircuit_parameters)

        try:
            cDecompose.Start_Decomposition()
        except Exception:
            return subcircuit, subcircuit_parameters

        new_circ = cDecompose.get_Circuit()
        new_params = cDecompose.get_Optimized_Parameters()
        err = cDecompose.get_Decomposition_Error()

        if err > config["tolerance"]:
            return subcircuit, subcircuit_parameters

        inverse_map = {v: k for k, v in qbit_map.items()}
        new_circ = new_circ.Remap_Qbits(inverse_map, qbit_num_orig).get_Flat_Circuit()

        if config["test_subcircuits"]:
            CompareCircuits(
                subcircuit,
                subcircuit_parameters,
                new_circ,
                new_params,
                parallel=config["parallel"],
            )

        return new_circ, new_params

    def InnerCompressWideCircuit(
        self, circ: Circuit, parameters: np.ndarray
    ) -> Tuple[Circuit, np.ndarray]:
        """Single pass: partition ``circ``, OSR-compress each partition, stitch."""
        from squander.utils import circuit_to_CNOT_basis

        circ, parameters = circuit_to_CNOT_basis(circ, parameters)

        partitioned_circuit, parameters, _ = PartitionCircuit(
            circ,
            parameters,
            self.max_partition_size,
            strategy=self.config["partition_strategy"],
        )
        subcircuits = partitioned_circuit.get_Gates()

        in_parent = parent_process() is not None
        if not in_parent and self.config["verbosity"] >= 1:
            print(len(subcircuits), "partitions to compress")

        optimized_subcircuits: List[Optional[Circuit]] = [None] * len(subcircuits)
        optimized_parameter_list: List[Optional[np.ndarray]] = [None] * len(subcircuits)

        max_gates = sum(
            y for x, y in circ.get_Gate_Nums().items() if x not in CNOT_COUNT_DICT
        )

        slices = []
        for sub in subcircuits:
            start = sub.get_Parameter_Start_Index()
            slices.append(parameters[start : start + sub.get_Parameter_Num()])

        nproc = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else mp.cpu_count()
        )
        with (
            contextlib.nullcontext() if in_parent else Pool(processes=nproc)
        ) as pool:
            async_results = []
            for idx, sub in enumerate(subcircuits):
                args = (sub, slices[idx], self.config)
                if in_parent:
                    async_results.append(args)
                else:
                    async_results.append(pool.apply_async(self.CompressPartition, args))

            for idx, ar in enumerate(async_results):
                if in_parent:
                    new_sub, new_p = self.CompressPartition(*ar)
                else:
                    new_sub, new_p = ar.get(timeout=None)

                orig_score = CNOTGateCount(subcircuits[idx], max_gates)
                new_score = CNOTGateCount(new_sub, max_gates)
                if new_score < orig_score:
                    optimized_subcircuits[idx] = new_sub
                    optimized_parameter_list[idx] = new_p
                    if self.config["verbosity"] >= 2:
                        print(
                            f"partition {idx}: {subcircuits[idx].get_Gate_Nums()} -> {new_sub.get_Gate_Nums()}"
                        )
                else:
                    optimized_subcircuits[idx] = subcircuits[idx]
                    optimized_parameter_list[idx] = slices[idx]

                if self.config["verbosity"] >= 1 and (idx + 1) % 100 == 0:
                    print(idx + 1, "partitions compressed")

        wide_parameters = np.concatenate(
            cast(List[np.ndarray], optimized_parameter_list), axis=0
        )
        wide_circuit = Circuit(circ.get_Qbit_Num())
        for c in cast(List[Circuit], optimized_subcircuits):
            wide_circuit.add_Circuit(c)

        assert wide_circuit.get_Parameter_Num() == wide_parameters.size, (
            f"Mismatch in parameter counts: "
            f"{wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"
        )

        if not in_parent and self.config["verbosity"] >= 1:
            print("original circuit:   ", circ.get_Gate_Nums())
            print("compressed circuit: ", wide_circuit.get_Gate_Nums())

        qgd_Wide_Circuit_Optimization.check_valid_routing(
            wide_circuit, self.config["topology"]
        )
        if self.config["verbosity"] >= 2:
            print("InnerCompressWideCircuit: check_compare_circuits")
        if self.config["test_final_circuit"]:
            CompareCircuits(circ, parameters, wide_circuit, wide_parameters)

        return wide_circuit, wide_parameters

    def CompressWideCircuit(
        self, circ: Circuit, parameters: np.ndarray
    ) -> Tuple[Circuit, np.ndarray]:
        """Top-level: sweep partition sizes, repeat each pass until no improvement.

        Mirrors the outer loop of ``qgd_Wide_Circuit_Optimization.OptimizeWideCircuit``
        for the Squander branch. Records ``self.config['compression_time']``.
        Requires the input circuit to already respect ``config['topology']``;
        no routing is performed.
        """
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(
            circ, self.config["topology"]
        ):
            raise Exception(
                "Input circuit does not respect the configured topology; "
                "qgd_WideCircuitCompression does not perform routing."
            )

        start_time = time.time()
        part_size_start = self.max_partition_size
        part_size_end = self.config.get("part_size_end", self.max_partition_size)

        count = CNOTGateCount(circ, 0)
        for max_part_size in range(part_size_start, part_size_end + 1):
            inner = qgd_WideCircuitCompression(
                {**self.config, "max_partition_size": max_part_size}
            )
            while True:
                circ_flat, parameters = inner.InnerCompressWideCircuit(circ, parameters)
                circ = circ_flat.get_Flat_Circuit()
                newcount = CNOTGateCount(circ, 0)
                no_improve = newcount >= count
                count = newcount
                if no_improve:
                    break

        self.config["compression_time"] = time.time() - start_time
        return circ, parameters
