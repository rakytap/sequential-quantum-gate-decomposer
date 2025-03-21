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
/*! \file U3.h
    \brief Header file for a class representing a U3 gate.
*/

#ifndef U3_H
#define U3_H

#include "Gate.h"
#include "matrix.h"
#include "matrix_real.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a U3 gate.
*/
class U3: public Gate {

protected:

   /// logical value indicating whether the matrix creation takes an argument theta
   bool theta;
   /// logical value indicating whether the matrix creation takes an argument phi
   bool phi;
   /// logical value indicating whether the matrix creation takes an argument lambda
   bool lambda;
   /// value applied for theta if it is not varied during the gate decomposition
   double theta0;
   /// value applied for phi if it is not varied during the gate decomposition
   double phi0;
   /// value applied for lambda if it is not varied during the gate decomposition
   double lambda0;
   /// Parameters theta, phi, lambda of the U3 gate after the decomposition of the unitary is done
   Matrix_real parameters;



public:

/**
@brief Nullary constructor of the class.
*/
U3();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
U3(int qbit_num_in, int target_qbit_in, bool theta_in, bool phi_in, bool lambda_in);

/**
@brief Destructor of the class
*/
virtual ~U3();
/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@return Returns with a matrix of the gate
*/
Matrix get_matrix( Matrix_real& parameters  );


/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix get_matrix( Matrix_real& parameters, int parallel  );

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void apply_to_list( Matrix_real& parameters, std::vector<Matrix>& inputs, int parallel );


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual void apply_to( Matrix_real& parameters, Matrix& input, int parallel );


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
virtual std::vector<Matrix> apply_derivate_to( Matrix_real& parameters, Matrix& input, int parallel );



/**
@brief Call to apply the gate on the input array/matrix by input*U3
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
virtual void apply_from_right( Matrix_real& parameters, Matrix& input );



/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
virtual void set_qbit_num(int qbit_num_in);



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
virtual void reorder_qubits( std::vector<int> qbit_list);

/**
@brief Call to check whether theta is a free parameter of the gate
@return Returns with true if theta is a free parameter of the gate, or false otherwise.
*/
bool is_theta_parameter();

/**
@brief Call to check whether Phi is a free parameter of the gate
@return Returns with true if Phi is a free parameter of the gate, or false otherwise.
*/
bool is_phi_parameter();

/**
@brief Call to check whether Lambda is a free parameter of the gate
@return Returns with true if Lambda is a free parameter of the gate, or false otherwise.
*/
bool is_lambda_parameter();




/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual U3* clone();


/**
@brief Call to set the final optimized parameters of the gate.
@param Theta Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void set_optimized_parameters(double Theta, double Phi, double Lambda );

/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters Theta, Phi and Lambda of the U3 gate.
*/
Matrix_real get_optimized_parameters();


/**
@brief Call to set the parameter theta0
@param theta_in The value for the parameter theta0
*/
void set_theta(double theta_in );

/**
@brief Call to set the parameter phi0
@param theta_in The value for the parameter theta0
*/
void set_phi(double phi_in );


/**
@brief Call to set the parameter lambda0
@param theta_in The value for the parameter theta0
*/
void set_lambda(double lambda_in );


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
virtual Matrix_real extract_parameters( Matrix_real& parameters );



};


#endif //U3

