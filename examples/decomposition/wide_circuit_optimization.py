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
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    CNOTGateCount, SingleQubitGateCount, TotalRawGateCount, CircuitGateStats,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils
from squander import Qiskit_IO
import time, requests, os, zipfile, tempfile, json, numpy as np
from pathlib import Path
from collections import Counter

# IBM Eagle native gate set (QMill benchmark basis)
IBM_EAGLE_BASIS = ["cx", "rz", "sx", "x"]


def transpile_to_ibm_eagle(qasm_path_or_circ, parameters=None):
    """Transpile a circuit to IBM Eagle's native gate set and return consistent
    gate stats via Squander's CircuitGateStats.

    Args:
        qasm_path_or_circ: Either a path to a .qasm file, or a Squander Circuit.
        parameters: Required if passing a Squander Circuit.

    Returns:
        dict from CircuitGateStats, using the same decomposition as the main
        benchmark pipeline.
    """
    from qiskit import QuantumCircuit, transpile

    if isinstance(qasm_path_or_circ, str):
        qc = QuantumCircuit.from_qasm_file(qasm_path_or_circ)
    else:
        qc = Qiskit_IO.get_Qiskit_Circuit(
            qasm_path_or_circ,
            np.asarray(parameters if parameters is not None else [], dtype=np.float64),
        )

    transpiled = transpile(qc, basis_gates=IBM_EAGLE_BASIS, optimization_level=0)
    eagle_circ, eagle_params = Qiskit_IO.convert_Qiskit_to_Squander(transpiled)
    return CircuitGateStats(eagle_circ)


if __name__ == "__main__":

    config = {
        "strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "test_subcircuits": False,
        "test_final_circuit": False,
        "max_partition_size": 4,
        "beam": None,
        "use_osr": True,
        "use_graph_search": True,
        "pre-opt-strategy": "TreeSearch",  # possible values: "TreeSearch", "qiskit", "bqskit", "TabuSearch"
        "routing-strategy": "seqpam-ilp",  # possible values: "sabre", "light-sabre", "bqskit-sabre", "seqpam-quick", "seqpam-ilp"
        "tolerance": 1e-5, #1e-5 for use_float and 1e-10 if not are sensible
        "use_float": True,  # whether to use single precision for the optimization (experimental, may cause instability in some cases, but can significantly reduce optimization time and memory usage for large circuits)
        # **{'use_basin_hopping': True, 'bh_T': 1.1822334624366124, 'bh_stepsize': 0.9020671823381502, 'bh_interval': 165, 'bh_target_accept_rate': 0.7037812116166546, 'bh_stepwise_factor': 0.8254028860713254}
    }

    #git clone https://github.com/onestruggler/qasm-quipper
    #sudo yum install gmp-devel
    #curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
    #~/.ghcup/bin/ghcup install ghc 8.8.4 #Quipper supports up to GHC 8.8.4
    #~/.ghcup/bin/ghcup set ghc 8.8.4
    #~/.ghcup/bin/ghcup rm ghc 9.6.7
    #PATH=~/.ghcup/bin:$PATH cabal update, cabal new-build
    #~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/quip_to_qasm/build/quip_to_qasm/quip_to_qasm -s inp -o inp.qasm
    #./scripts/quip_to_qasm2.sh -s inp -o inp.qasm
    """
    zip_url = "https://github.com/njross/optimizer/archive/refs/heads/master.zip"
    #zip_url = "https://zenodo.org/records/17293975/files/benchmark_circuit_QMill_IBM.zip?download=1"
    temp_dir = tempfile.mkdtemp(prefix="repos_")
    zip_path = os.path.join(temp_dir, "master.zip")
    qasm_files = []
    # Download zip
    r = requests.get(zip_url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {zip_url}: HTTP {r.status_code}")
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract zip
    extract_path = os.path.join(temp_dir, "master")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    # Find QASM files
    for path in Path(extract_path).rglob("*_before*"):
        inp = str(path.resolve())
        if "/PF/" in inp: continue
        outp = inp + ".qasm"
        #os.system(f"PATH=~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/elim_ctrls/build/elim_ctrls/:~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/elim_funs/build/elim_funs/:~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/elim_invs/build/elim_invs/:~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/elim_pows/build/elim_pows/:~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/format_qasm/build/format_qasm/:~/qasm-quipper/dist-newstyle/build/x86_64-linux/ghc-8.8.4/LinguaQuanta-0.1.0.0/x/quip_to_qasm/build/quip_to_qasm:$PATH ~/qasm-quipper/scripts/quip_to_qasm2.sh -s {inp} -o {outp}")
        os.system(f"~/qasm-quipper/quipper-qasm/dist-newstyle/build/x86_64-linux/ghc-8.8.4/quipper-qasm-0.1.0.0/x/quipper-qasm/build/quipper-qasm/quipper-qasm -inline {inp} > {outp}")
        cczdef = "gate ccz a,b,c { h c; ccx a,b,c; h c; }\n"
        qasm_files.append(outp)
        print(outp)
    with zipfile.ZipFile("IBMeagle.zip", "w") as zf:
        for path in Path(extract_path).rglob("*.qasm"):
            zf.write(str(path.resolve()))
    """
    #extract_dir = tempfile.mkdtemp(prefix="IBMeagle")
    #with zipfile.ZipFile("IBMeagle.zip", "r") as zf:
    #    zf.extractall(extract_dir)
    #qasm_files = [str(path.resolve()) for path in Path(extract_dir).rglob("*.qasm")]
    #filename = next(x for x in qasm_files if "mod5_4_before" in x)
    #filename = next(x for x in qasm_files if "gf2^E8_mult_before" in x)
    #filename = next(x for x in qasm_files if "csum_mux_9_before" in x)

    import glob, re, os

    def circuit_has_resets(filepath):
        """Return True if the QASM file contains non-unitary reset operations."""
        with open(filepath) as f:
            return 'reset' in f.read()

    def get_circuit_stats(filepath):
        """Return (cnot_count, qubit_count) for a QASM file via squander parsing."""
        circ, _, _ = utils.qasm_to_squander_circuit(filepath)
        return CNOTGateCount(circ, 0), circ.get_Qbit_Num()

    # Gather all original circuits (skip _u3cx, _after, and reset-containing circuits)
    all_qasm = glob.glob("/home/morse/processed_qasm_new/**/*.qasm", recursive=True)
    all_files = [f for f in all_qasm
                 if "_u3cx" not in f and "_after" not in os.path.basename(f)]
    skipped_reset = [f for f in all_files if circuit_has_resets(f)]
    if skipped_reset:
        print(f"Skipping {len(skipped_reset)} circuits with non-unitary reset gates:")
        for f in skipped_reset:
            print(f"  SKIP: {os.path.basename(f)}")
    all_files = [f for f in all_files if not circuit_has_resets(f)]
    files = sorted([(get_circuit_stats(f), f) for f in all_files], key=lambda x: x[0])
    print("=== Running circuits in order of (CNOT_count, qubit_count) ===")
    for (cnots, qubits), f in files:
        print(f"  {cnots:>8} CNOT, {qubits:>4} qubits  {os.path.basename(f)}")
    print("=" * 60)

    # Unpack just filenames in sorted order
    files = [f for _, f in files]


    #import os

    #files = [os.path.join(Path(__file__).resolve().parent, "bv_n14.qasm")]

    # JSON results file with resume support
    RESULTS_FILE = "results.json"
    import json

    # Load existing results to skip already-processed circuits
    existing_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results; will skip already-processed circuits.")

    results = existing_results  # merge new results into existing
    for filename in files:
        fname = os.path.basename(filename)
        if fname in results:
            print(f"Skipping already processed: {fname}")
            continue

        print(f"executing optimization of circuit: {filename}")
        #if not filename.endswith("_n140.qasm"): continue

        # load the circuit from a file
        circ, parameters, _ = utils.qasm_to_squander_circuit(filename)
        config["topology"] = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization.linear_topology(
                circ.get_Qbit_Num()
            )
        )

        # pre-optimization stats
        init_stats = CircuitGateStats(circ)
        init_ibm_eagle = transpile_to_ibm_eagle(filename)

        # run circuit optimization
        wide_circuit_optimizer = (
            Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization({**config})
        )
        start_time = time.time()
        optcirc, optparameters = wide_circuit_optimizer.OptimizeWideCircuit(
            circ, parameters
        )
        elapsed = time.time() - start_time

        # post-optimization stats
        opt_stats = CircuitGateStats(optcirc)
        final_ibm_eagle = transpile_to_ibm_eagle(optcirc, optparameters)
        opt_time = wide_circuit_optimizer.config.get("optimization_time", None)

        # routing stats (if routing was needed)
        a2a_stats = None
        routed_stats = None
        a2a_time = routing_time = None
        if wide_circuit_optimizer.config.get("routed_circuit", None) is not None:
            a2acirc = wide_circuit_optimizer.config["all_to_all_circuit"]
            routedcirc = wide_circuit_optimizer.config["routed_circuit"]
            a2a_stats = CircuitGateStats(a2acirc)
            routed_stats = CircuitGateStats(routedcirc)
            a2a_time = wide_circuit_optimizer.config.get("all_to_all_optimization_time", None)
            routing_time = wide_circuit_optimizer.config.get("routing_time", None)

        result_entry = {
            "file": fname,
            "strategy": config["strategy"],
            "pre_opt_strategy": config["pre-opt-strategy"],
            "routing_strategy": config["routing-strategy"],
            "init": init_stats,
            "init_ibm_eagle": init_ibm_eagle,
            "final": opt_stats,
            "final_ibm_eagle": final_ibm_eagle,
            "timing": {
                "a2a": round(a2a_time, 2) if a2a_time else None,
                "routing": round(routing_time, 2) if routing_time else None,
                "optimization": round(opt_time, 2) if opt_time else None,
                "total": round(elapsed, 2),
            },
        }
        if a2a_stats:
            result_entry["all_to_all"] = a2a_stats
        if routed_stats:
            result_entry["routed"] = routed_stats

        results[fname] = result_entry

        # save after each circuit so we don't lose progress
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        wide_circuit_optimizer.check_compare_circuits(
            circ,
            parameters,
            optcirc,
            optparameters,
            routing=wide_circuit_optimizer.config.get("routed_circuit", None) is not None,
            forced_test=True,
            label="example final original-to-output",
        )

        print(f"  init: {init_stats['cnot_equiv']} CNOT, {init_stats['single_qubit']} 1q, "
              f"{init_stats['total_raw']} total, {init_stats['qubits']}q")
        print(f"  final: {opt_stats['cnot_equiv']} CNOT, {opt_stats['single_qubit']} 1q, "
              f"{opt_stats['total_raw']} total")
        print(f"  time: {elapsed:.2f}s")
        print(f"--- {len(results)}/{len(files)} circuits processed ---")

        print("--- %s seconds elapsed during optimization ---" % elapsed)
