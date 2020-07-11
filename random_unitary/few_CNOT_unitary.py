#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:23:58 2020
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

import numpy as np
from operations.CNOT import CNOT
from operations.U3 import U3

##
# @brief Call to create a random unitary containing a given number of CNOT gates between randomly chosen qubits
# @param qbit_num The number of qubits spanning the unitary
# @param cnot_num The number of CNOT gates in the unitary
def few_CNOT_unitary( qbit_num, cnot_num):

    # the current number of CNOT gates
    cnot_num_curr = 0

    # The unitary discribing each qubits in their initial state
    mtx = np.identity( 2**qbit_num )

    # constructing the unitary
    while True :
        cnot_or_u3 = np.random.randint(0,6)

        if cnot_or_u3 <= 4:
            # creating random parameters for the U3 operation
            parameters = np.array([ np.random.uniform(0,4*np.pi), np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi)])

            # randomly choose the target qbit
            target_qbit = np.random.randint(0,qbit_num-1)

            # creating the U3 gate
            op = U3(qbit_num, target_qbit, ['Theta', 'Phi', 'Lambda'])

            # get the matrix of the operation
            gate_matrix = op.matrix(parameters)

        elif cnot_or_u3 == 5:
            # randomly choose the target qbit
            target_qbit = np.random.randint(0, qbit_num)

            # randomly choose the control qbit
            control_qbit = np.random.randint(0, qbit_num)

            if target_qbit == control_qbit:
                continue

            # creating the CNOT gate
            op = CNOT(qbit_num, control_qbit, target_qbit)

            # get the matrix of the operation
            gate_matrix = op.matrix

            cnot_num_curr = cnot_num_curr + 1

        else:
            continue


        # get the current unitary
        mtx = np.dot(gate_matrix, mtx)

        # exit the loop if the maximal number of CNOT gates reched
        if cnot_num_curr >= cnot_num:
            return mtx