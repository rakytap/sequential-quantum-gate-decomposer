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

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from qgd_python.nn.qgd_nn_Wrapper import qgd_nn_Wrapper



##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_nn(qgd_nn_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @return An instance of the class
    def __init__( self ):

        # call the constructor of the wrapper class
        super(qgd_nn, self).__init__()


##
# @brief Wrapper function to retrieve the data chanels for the neural network
    def get_nn_chanels(self, Umtx=None, qbit_num=-1, levels=-1 ):

	# call the C wrapper function
        if not(Umtx in None):
            super(qgd_nn, self).get_nn_chanels( Umtx=Umtx )
        elif qbit_num > 0 and levels >= 0:
            print( Umtx )
            super(qgd_nn, self).get_nn_chanels( qbit_num=qbit_num, levels=levels )            
        else:
            print( "invalid parameters were given")

