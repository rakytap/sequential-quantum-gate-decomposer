/*
Created on Fri Jun 26 14:13:26 2020
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
*/
/*! \file qgd/Sub_Matrix_Decomposition_Custom.h
    \brief Header file for a class responsible for the disentanglement of one qubit from the others.
    This class enables to define custom gate structure for the decomposition.
*/

#ifndef SUB_MATRIX_DECOMPOSITION_CUSTOM_H
#define SUB_MATRIX_DECOMPOSITION_CUSTOM_H

#include "Sub_Matrix_Decomposition.h"


/**
@brief A class responsible for the disentanglement of one qubit from the others.
This class enables to define custom gate structure for the decomposition.
The blocks of the defined custom gate structure are repeated until the maximal number of layer is reached.
*/
class Sub_Matrix_Decomposition_Custom : public Sub_Matrix_Decomposition {

public:

    // reusing the constructors of the superclass
    using Sub_Matrix_Decomposition::Sub_Matrix_Decomposition;



/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void add_operation_layers();



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Sub_Matrix_Decomposition_Custom* clone();


};


#endif //SUB_MATRIX_DECOMPOSITION_CUSTOM
