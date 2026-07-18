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
/*! \file SYC.h
    \brief Header file for a class representing a Sycamore gate.
*/

#ifndef SYC_H
#define SYC_H

#include "matrix.h"
#include "Gate.h"
#define _USE_MATH_DEFINES
#include <math.h>



/**
@brief A class representing a SYC operation.
*/
class SYC: public Gate {

protected:


public:

/**
@brief Nullary constructor of the class.
*/
SYC();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
SYC(int qbit_num_in, int target_qbit_in,  int control_qbit_in);

/**
@brief Destructor of the class
*/
~SYC();

/**
@brief Return the fixed 4x4 SYC two-qubit kernel.
*/
Matrix gate_kernel(const Matrix_real& precomputed_sincos) override;
Matrix_float gate_kernel(const Matrix_real_float& precomputed_sincos) override;
Matrix inverse_gate_kernel(const Matrix_real& precomputed_sincos) override;
Matrix_float inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) override;



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SYC* clone() override;


};

#endif //SYC
