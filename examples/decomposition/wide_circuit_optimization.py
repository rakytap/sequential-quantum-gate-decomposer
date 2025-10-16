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
from squander import utils
from squander import Qiskit_IO
import time, requests, os, zipfile, tempfile
from pathlib import Path
    

if __name__ == '__main__':


    config = {  
            'strategy': "TreeSearch", 
            'test_subcircuits': True,
            'test_final_circuit': True,
            'max_partition_size': 3,
    }

    zip_url = "https://zenodo.org/records/17293975/files/benchmark_circuit_QMill_IBM.zip?download=1"
    temp_dir = tempfile.mkdtemp(prefix="repos_")
    zip_path = os.path.join(temp_dir, "benchmark_circuit_QMill_IBM.zip")
    qasm_files = []
    # Download zip
    r = requests.get(zip_url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {zip_url}: HTTP {r.status_code}")
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract zip
    extract_path = os.path.join(temp_dir, "benchmark_circuit_QMill_IBM")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    # Find QASM files
    for path in Path(extract_path).rglob("*.qasm"):
        qasm_files.append(str(path.resolve()))

    #filename = next(x for x in qasm_files if x.endswith("mod5_4_qmill_ibm.qasm"))
    #filename = next(x for x in qasm_files if x.endswith("gf2^E8_mult_qmill_ibm.qasm"))
    filename = next(x for x in qasm_files if x.endswith("csum_mux_9_qmill_ibm.qasm"))    

    #filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    
    start_time = time.time()

    # load the circuit from a file
    circ, parameters = utils.qasm_to_squander_circuit(filename)

    # instantiate the object for optimizing wide circuits
    wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )

    # run circuti optimization
    circ_flat, parameters = wide_circuit_optimizer.OptimizeWideCircuit( circ, parameters )

    config['topology'] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
    (0, 8),
    ]
    wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )
    circo = Qiskit_IO.get_Qiskit_Circuit(circ_flat.get_Flat_Circuit(),parameters)
    # run circuti optimization
    circ, parameters = Qiskit_IO.convert_Qiskit_to_Squander(circo)
    wide_circuit_optimizer.OptimizeWideCircuit( circ, parameters )

    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


