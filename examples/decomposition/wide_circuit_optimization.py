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
    files = list(sorted((filename for filename in glob.glob("/home/morse/processed_qasm/**/*.qasm", recursive=True) if "_u3cx" not in filename),
        key=lambda x: re.sub(r'(\d+)', lambda m: f"{int(m.group(1)):010}", x)))


    #import os

    #files = [os.path.join(Path(__file__).resolve().parent, "bv_n14.qasm")]

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
