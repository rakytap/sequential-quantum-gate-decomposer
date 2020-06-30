#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:23:58 2020

@author: rakytap
"""

def two_qubit_decomposition():
 
    from decomposition.Two_Qubit_Decomposition import Two_Qubit_Decomposition
    import numpy as np
    
    print('****************************************')
    print('Test of two qubit decomposition')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group
    
    Umtx = unitary_group.rvs(4)
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')
    #print(np.dot(Umtx, Umtx.conj().T))

    cDecomposition = Two_Qubit_Decomposition( Umtx )
    
    #start the decomposition
    cDecomposition.start_decomposition()
    
    # finalize the decomposition
    cDecomposition.finalize_decomposition()
    
    # retrive the decomposed matrix
    U_decomposed = cDecomposition.get_transformed_matrix( cDecomposition.optimized_parameters, cDecomposition.operations, initial_matrix=cDecomposition.Umtx)
    print(U_decomposed.__str__())