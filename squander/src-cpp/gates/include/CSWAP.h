/*
Created on Fri Jun 26 14:13:26 2020
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
*/
/*! \file CSWAP.h
    \brief Header file for a class representing a CSWAP (Controlled SWAP) operation.
*/

#ifndef CSWAP_H
#define CSWAP_H

#include "matrix.h"
#include "SWAP.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a CSWAP (Controlled SWAP) operation.
*/
class CSWAP: public SWAP {

protected:
    /// The index of the control qubit
    int control_qbit2;
public:

/**
@brief Nullary constructor of the class.
*/
CSWAP();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the first target qubit. (0 <= target_qbit <= qbit_num-1)
@param target_qbit2_in The identification number of the second target qubit. (0 <= target_qbit2 <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= control_qbit <= qbit_num-1)
*/
CSWAP(int qbit_num_in, int target_qbit_in, int target_qbit2_in, int control_qbit_in);

/**
@brief Destructor of the class
*/
virtual ~CSWAP();

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual CSWAP* clone();

/**
@brief Call to apply the gate operation on the input matrix
@param input The input matrix on which the transformation is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with Intel TBB
*/
virtual void apply_to(Matrix& input, int parallel = 0);

/**
@brief Get list of involved qubits
@param only_target If true, return only target qubits, otherwise include control qubits too
@return Vector of qubit indices
*/
std::vector<int> get_involved_qubits(bool only_target = false) override;

/**
@brief Get the control qubit
@return The index of the control qubit
*/
int get_control_qbit2() override;

/**
@brief Set the control qubit
@param control_qbit2_in The index of the control qubit
*/
void set_control_qbit2(int control_qbit2_in) override;

};

#endif //CSWAP