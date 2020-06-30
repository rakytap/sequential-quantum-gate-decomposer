# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020

@author: rakytap
"""

from test import operations
from test import decomposition

#%% Tests of operations
operations.test_general_operation()
operations.test_U3_operation()
operations.test_CNOT_operation()
operations.test_operations()

#%% Test of two-qubit decomposition
#decomposition.two_qubit_decomposition()

#%% Test of general three-qubit decomposition
decomposition.three_qubit_decomposition()

#%% Test of general four-qubit decomposition
decomposition.four_qubit_decomposition()


#%% Test of three-qubit decomposition
decomposition.IBM_challenge_decomposition()