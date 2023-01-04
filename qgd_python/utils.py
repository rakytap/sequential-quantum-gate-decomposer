## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

## \file utils.py
##    \brief Utility function for SQUANDER python binding


from qiskit import execute

try:
    import qiskit_aer as Aer
    qiskit_version = 1
except ImportError as err:
    from qiskit import Aer
    qiskit_version = 0


##
# @brief Call to retrieve the unitary from QISKIT circuit
def get_unitary_from_qiskit_circuit( circuit ):


    if qiskit_version == 1:
        backend = Aer.AerSimulator(method='unitary')
    elif qiskit_version == 0:
        backend = Aer.get_backend('aer_simulator')


    circuit.save_unitary()
                
    # job execution and getting the result as an object
    job = execute(circuit, backend)
        
    # the result of the Qiskit job
    result=job.result()  


    return result.get_unitary(circuit)

