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
/*! \file SYC.cpp
    \brief Class representing a SYC gate.
*/

#include "SYC.h"

#include <cmath>



using namespace std;


/**
@brief Nullary constructor of the class.
*/
SYC::SYC() {

    // A string labeling the gate operation
    name = "SYC";

    // number of qubits spanning the matrix of the gate
    qbit_num = -1;
    // the size of the matrix
    matrix_size = -1;
    // A string describing the type of the gate
    type = SYC_OPERATION;
    // The number of free parameters
    parameter_num = 0;

    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = -1;

    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
SYC::SYC(int qbit_num_in,  int target_qbit_in, int control_qbit_in) {

    // A string labeling the gate operation
    name = "SYC";

    // number of qubits spanning the matrix of the gate
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the gate
    type = SYC_OPERATION;
    // The number of free parameters
    parameter_num = 0;

    if (target_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);
	throw "The index of the target qubit is larger than the number of qubits";
    }
	
    // The index of the qubit on which the gate acts (target_qbit >= 0)
    target_qbit = target_qbit_in;


    if (control_qbit_in >= qbit_num) {
        std::stringstream sstream;
        sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
        print(sstream, 0);	
	throw "The index of the control qubit is larger than the number of qubits";
    }
	
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
    control_qbit = control_qbit_in;

}

/**
@brief Destructor of the class
*/
SYC::~SYC() {
}

namespace {

template<typename MT, typename CT, typename RT>
MT build_syc_kernel_impl(bool inverse) {
    MT m(4, 4);
    for (int idx = 0; idx < m.size(); ++idx) {
        m[idx] = CT();
    }

    m[0 * m.stride + 0].real = static_cast<RT>(1);

    const RT i_sign = inverse ? static_cast<RT>(1) : static_cast<RT>(-1);
    m[1 * m.stride + 2].imag = i_sign;
    m[2 * m.stride + 1].imag = i_sign;

    const RT phase_real = static_cast<RT>(std::sqrt(3.0) / 2.0);
    const RT phase_imag = inverse ? static_cast<RT>(0.5) : static_cast<RT>(-0.5);
    m[3 * m.stride + 3].real = phase_real;
    m[3 * m.stride + 3].imag = phase_imag;

    return m;
}

}  // namespace


Matrix
SYC::gate_kernel(const Matrix_real& /*precomputed_sincos*/) {
    return build_syc_kernel_impl<Matrix, QGD_Complex16, double>(false);
}


Matrix_float
SYC::gate_kernel(const Matrix_real_float& /*precomputed_sincos*/) {
    return build_syc_kernel_impl<Matrix_float, QGD_Complex8, float>(false);
}


Matrix
SYC::inverse_gate_kernel(const Matrix_real& /*precomputed_sincos*/) {
    return build_syc_kernel_impl<Matrix, QGD_Complex16, double>(true);
}


Matrix_float
SYC::inverse_gate_kernel(const Matrix_real_float& /*precomputed_sincos*/) {
    return build_syc_kernel_impl<Matrix_float, QGD_Complex8, float>(true);
}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
SYC* SYC::clone() {

    SYC* ret = new SYC( qbit_num, target_qbit, control_qbit );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;

}

