# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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

# @brief Automatic test procedures to test the functionalities of the code

from test import operations
from test import decomposition

#%% Tests of operations
operations.test_general_operation()
operations.test_U3_operation()
operations.test_CNOT_operation()
operations.test_operations()

#%% Test of two-qubit decomposition
decomposition.two_qubit_decomposition()

#%% Test of general three-qubit decomposition
decomposition.three_qubit_decomposition()

#%% Test of three-qubit decomposition
decomposition.IBM_challenge_decomposition()

#%% Test of general four-qubit decomposition
decomposition.four_qubit_decomposition()

