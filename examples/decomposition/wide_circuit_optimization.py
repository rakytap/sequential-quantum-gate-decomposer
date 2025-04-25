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

from squander import Wide_Circuit_Optimization

import numpy as np
from qiskit import QuantumCircuit


from squander.partitioning.partition import (
    get_qubits,
    qasm_to_partitioned_circuit
)


filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"

max_partition_size = 3
partitined_circuit, parameters = qasm_to_partitioned_circuit( filename, max_partition_size )

print("iiiiiiiiiiiiiiiiiiiiiiiiI")
subcircuits = partitined_circuit.get_Gates()
#print(gates)

for subcircuit in subcircuits:
    print( subcircuit.get_Parameter_Start_Index(), subcircuit.get_Parameter_Num() )
 
