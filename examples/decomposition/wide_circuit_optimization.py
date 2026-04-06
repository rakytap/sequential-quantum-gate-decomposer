# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Peter Rakyta, Ph.D.
"""
## \file wide_circuit_optimization.py
## \brief Simple example python code demonstrating a wide circuit optimization

import squander.decomposition.qgd_Wide_Circuit_Optimization as Wide_Circuit_Optimization
from squander.decomposition.qgd_Wide_Circuit_Optimization import CNOTGateCount
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils
from squander import Qiskit_IO
import time, requests, os, zipfile, tempfile
from pathlib import Path


if __name__ == "__main__":

    config = {
        "strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "test_subcircuits": False,
        "test_final_circuit": False,
        "max_partition_size": 3,
        "beam": None,
        "use_osr": True,
        "use_graph_search": True,
        "pre-opt-strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "routing-strategy": "seqpam-ilp",  # possible values: "sabre", "light-sabre", "bqskit-sabre", "seqpam-quick", "seqpam-ilp"
        "tolerance": 1e-10,
        # **{'use_basin_hopping': True, 'bh_T': 1.1822334624366124, 'bh_stepsize': 0.9020671823381502, 'bh_interval': 165, 'bh_target_accept_rate': 0.7037812116166546, 'bh_stepwise_factor': 0.8254028860713254}
    }

    import os

    files = [os.path.join(Path(__file__).resolve().parent, "bv_n14.qasm")]

    results = {}
    for filename in files:
        print(f"executing optimization of circuit: {filename}")

        # load the circuit from a file
        circ, parameters, _ = utils.qasm_to_squander_circuit(filename)
        config["topology"] = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                circ.get_Qbit_Num()
            )
        )

        # run circuit optimization
        wide_circuit_optimizer = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization({**config})
        )
        start_time = time.time()
        optcirc, optparameters = wide_circuit_optimizer.OptimizeWideCircuit(
            circ, parameters
        )
        elapsed = time.time() - start_time
        init_cnot_count = CNOTGateCount(circ, 0)
        cnot_count, opt_time = CNOTGateCount(
            optcirc, 0
        ), wide_circuit_optimizer.config.get("optimization_time", None)
        a2a_cnot_count, routed_cnot_count = None, None
        a2a_time, routing_time = 0.0, 0.0

        if wide_circuit_optimizer.config.get("routed_circuit", None) is not None:
            init_map, final_map = (
                wide_circuit_optimizer.config["initial_mapping"],
                wide_circuit_optimizer.config["final_mapping"],
            )
            a2acirc, a2aparams = (
                wide_circuit_optimizer.config["all_to_all_circuit"],
                wide_circuit_optimizer.config["all_to_all_parameters"],
            )
            routedcirc, routedparams = (
                wide_circuit_optimizer.config["routed_circuit"],
                wide_circuit_optimizer.config["routed_parameters"],
            )
            a2a_cnot_count = CNOTGateCount(a2acirc, 0)
            routed_cnot_count = CNOTGateCount(routedcirc, 0)
            a2a_time = wide_circuit_optimizer.config.get(
                "all_to_all_optimization_time", None
            )
            routing_time = wide_circuit_optimizer.config.get("routing_time", None)
        results[os.path.basename(filename)] = (
            (init_cnot_count, a2a_cnot_count, routed_cnot_count, cnot_count),
            (a2a_time, routing_time, opt_time, elapsed),
        )
        wide_circuit_optimizer.check_compare_circuits(
            circ, optparameters, optcirc, optparameters, routing=True
        )
        with open("results.txt", "a") as f:
            f.write(
                f"{os.path.basename(filename)}: {config['pre-opt-strategy']}, {config['routing-strategy']}, {config['strategy']} CNOT count = {init_cnot_count, a2a_cnot_count, routed_cnot_count, cnot_count}, elapsed time = {a2a_time:.2f} + {routing_time:.2f} + {opt_time:.2f} = {elapsed:.2f} seconds\n"
            )

        print("--- %s seconds elapsed during optimization ---" % elapsed)
