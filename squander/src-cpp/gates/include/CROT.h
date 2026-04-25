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
/*! \file CROT.h
    \brief Header file for a class representing a controlled rotation gate around the Y axis.
*/

#ifndef CROT_H
#define CROT_H

#include "RY.h"
#include "CNOT.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>



/**
@brief A class representing a CROT gate.
*/
class CROT: public Gate {

protected:
public:

/**
@brief Nullary constructor of the class.
*/
CROT();

CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in);

virtual ~CROT();

virtual Matrix gate_kernel(const Matrix_real& precomputed_sincos) override;
virtual Matrix_float gate_kernel(const Matrix_real_float& precomputed_sincos) override;
virtual Matrix inverse_gate_kernel(const Matrix_real& precomputed_sincos) override;
virtual Matrix_float inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) override;
virtual Matrix derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) override;
virtual Matrix_float derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) override;
virtual Matrix derivative_aux_kernel(const Matrix_real& precomputed_sincos, int param_idx) override;
virtual Matrix_float derivative_aux_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) override;

virtual CROT* clone() override;



std::vector<double> get_parameter_multipliers() const override;

};


#endif //CROT

