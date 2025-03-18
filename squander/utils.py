## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

## \file utils.py
##    \brief Utility function for SQUANDER python binding



import numpy as np

import qiskit
qiskit_version = qiskit.version.get_version_info()

if qiskit_version[0] == '1':
    import qiskit_aer as Aer
    from qiskit import transpile
else :
    from qiskit import Aer
    from qiskit import execute




##
# @brief Call to retrieve the unitary from QISKIT circuit
def get_unitary_from_qiskit_circuit( circuit ):

    
    
    if qiskit_version[0] == '1':

        circuit.save_unitary()
        backend = Aer.AerSimulator(method='unitary')
        
        compiled_circuit = transpile(circuit, backend)
        result = backend.run(compiled_circuit).result()
        
        return np.asarray( result.get_unitary(circuit) )    
        
    elif qiskit_version[0] == '0':
        backend = Aer.get_backend('aer_simulator')
        circuit.save_unitary()
        
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  


        return np.asarray( result.get_unitary(circuit) )        



                


