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
from squander import Partition_Aware_Mapping
from squander import utils
from squander import Qiskit_IO
import time
from squander import Circuit
import numpy as np
if __name__ == '__main__':


    config = {
            'strategy': "TreeSearch",
            'test_subcircuits': True,
            'test_final_circuit': True,
            'max_partition_size': 3,
    }

    filename = "benchmarks/qfast/4q/adder_q4.qasm"
    start_time = time.time()

    # load the circuit from a file
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    config['topology'] = [
    (0, 1), (0, 2), (0, 3), 
    ]
    wide_circuit_optimizer = Partition_Aware_Mapping( config )
    wide_circuit_optimizer.Partition_Aware_Mapping( circ, parameters )

    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


