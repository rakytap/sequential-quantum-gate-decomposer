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
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils
from squander import Qiskit_IO
import time, requests, os, zipfile, tempfile
from pathlib import Path
    

if __name__ == '__main__':


    config = {  
            'strategy': "TreeSearch", 
            'test_subcircuits': False,
            'test_final_circuit': True,
            'max_partition_size': 3,
            'beam': 16,
            "use_gl": True,
            'tolerance': 1e-10,
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

    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    start_time = time.time()

    # load the circuit from a file
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    #circ = Circuit(3); circ.add_CNOT(0, 1); circ.add_CNOT(2, 0); circ.add_CNOT(1, 2)
    #circ = Circuit(2); circ.add_CNOT(0, 1); circ.add_H(0)
    #circ = Circuit(2); circ.add_SWAP([0, 1])
    #circ = Circuit(2); circ.add_S(0); circ.add_S(1); circ.add_H(0); circ.add_CNOT(0, 1); circ.add_CNOT(1, 0); circ.add_H(1)
    #circ = Circuit(3); circ.add_U3(0); circ.add_U3(1); circ.add_CNOT(0, 1); circ.add_U3(0); circ.add_U3(1); circ.add_CNOT(0, 1); import numpy as np; parameters = np.array([0.7]*12)


    # run circuit optimization
    for max_part_size in range(3, 6):
        # instantiate the object for optimizing wide circuits
        wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( {**config, 'max_partition_size': max_part_size} )
        while True:
            count = Wide_Circuit_Optimization.CNOTGateCount(circ)
            circ_flat, parameters = wide_circuit_optimizer.OptimizeWideCircuit( circ, parameters )

            #config['topology'] = [
            #(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
            #(8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
            #(0, 8),
            #]
            wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )
            circo = Qiskit_IO.get_Qiskit_Circuit(circ_flat.get_Flat_Circuit(),parameters)
            # run circuit optimization
            circ, parameters = Qiskit_IO.convert_Qiskit_to_Squander(circo)
            if Wide_Circuit_Optimization.CNOTGateCount(circ) >= count: break

    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


