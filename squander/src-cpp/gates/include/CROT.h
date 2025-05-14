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


typedef enum crot_type {CONTROL_R, CONTROL_OPPOSITE, CONTROL_INDEPENDENT} crot_type;

/**
@brief A class representing a CROT gate.
*/
class CROT: public Gate {

protected:
    
    
   crot_type subtype;

   Matrix_real parameters;
public:

/**
@brief Nullary constructor of the class.
*/
CROT();

CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in, crot_type subtype_in);

virtual ~CROT();

void apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input );

void apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel );

virtual void apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel );

virtual void apply_from_right( Matrix_real& parameters, Matrix& input );

virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel );

virtual void set_qbit_num(int qbit_num_in);

virtual void reorder_qubits( std::vector<int> qbit_list);

void set_optimized_parameters(double Theta0Over2, double Phi0, double Theta1Over2, double Phi1 );

Matrix_real get_optimized_parameters();

crot_type get_subtype();

virtual CROT* clone();

Matrix calc_one_qubit_rotation(double ThetaOver2, double Phi);

Matrix calc_one_qubit_rotation_deriv_Phi(double ThetaOver2, double Phi);

virtual Matrix_real extract_parameters( Matrix_real& parameters );



};


#endif //CROT

