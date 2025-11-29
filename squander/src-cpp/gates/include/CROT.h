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
    
    


   Matrix_real parameters;
public:

/**
@brief Nullary constructor of the class.
*/
CROT();

CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in);

virtual ~CROT();

void apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input );

void apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) override;

virtual void apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) override;

virtual void apply_from_right( Matrix_real& parameters, Matrix& input );

virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) override;

virtual void set_qbit_num(int qbit_num_in) override;

virtual void reorder_qubits( std::vector<int> qbit_list) override;

virtual CROT* clone() override;

virtual Matrix_real extract_parameters( Matrix_real& parameters ) override;



};


#endif //CROT

