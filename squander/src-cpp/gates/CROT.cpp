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
/*! \file CROT.cpp
    \brief Class representing a controlled Y rotattion gate.
*/

#include "CROT.h"
#include "apply_large_kernel_to_input.h"
#include "apply_large_kernel_to_input_AVX.h"

static double M_PIOver2 = M_PI/2;

//static tbb::spin_mutex my_mutex;
/**
@brief Nullary constructor of the class.
*/
CROT::CROT(){

        // A string describing the type of the gate
        type = CROT_OPERATION;
        
        subtype = CONTROL_OPPOSITE;
        
                // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        
        target_qbit = -1;
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = -1;
        parameter_num = 0;

        name = "CROT";

}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param theta_in logical value indicating whether the matrix creation takes an argument theta.
@param phi_in logical value indicating whether the matrix creation takes an argument phi
@param lambda_in logical value indicating whether the matrix creation takes an argument lambda
*/
CROT::CROT(int qbit_num_in, int target_qbit_in, int control_qbit_in, crot_type subtype_in) {

        name = "CROT";

        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate

        // A string describing the type of the gate
        type = CROT_OPERATION;
        
        subtype = subtype_in;


        if (control_qbit_in >= qbit_num) {
	    std::stringstream sstream;
	    sstream << "The index of the control qubit is larger than the number of qubits in CROT gate." << std::endl;
	    print(sstream, 0);	  
            throw sstream.str();
        }

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gates
        control_qbit = control_qbit_in;


        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;
        
        if (subtype == CONTROL_R || subtype == CONTROL_OPPOSITE){
            parameter_num=2;
        }
        else if (subtype == CONTROL_INDEPENDENT){
            parameter_num = 4;
        }
        else{
	    std::stringstream sstream2;
	    sstream2 << "ERROR: CROT subtype not implemented" << std::endl;
	    print(sstream2, 0);	  
            throw sstream2.str();
        }
        parameters = Matrix_real(1, parameter_num);
}

/**
@brief Destructor of the class
*/
CROT::~CROT() {

}


/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CROT::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& input ) {


    for ( std::vector<Matrix>::iterator it=input.begin(); it != input.end(); it++ ) {
        apply_to( parameters_mtx, *it, 0);
    }

}

/**
@brief Call to apply the gate on the input array/matrix by U3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CROT::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( parameters_mtx, *input, parallel );

        }

    });

}

/**
@brief Call to apply the gate on the input array/matrix by CROT3*input
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CROT::apply_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {


    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CROT gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }


    double Theta0Over2, Theta1Over2, Phi0, Phi1;
    
    if (subtype == CONTROL_R){
       Theta0Over2 = parameters_mtx[0];
       Phi0 = parameters_mtx[1];
    }
    else if (subtype == CONTROL_OPPOSITE){
       Theta0Over2 = parameters_mtx[0];
       Theta1Over2 = -1.*parameters_mtx[0];
       Phi0 = parameters_mtx[1];
       Phi1 = parameters_mtx[1];
    }
    else if (subtype == CONTROL_INDEPENDENT){
       Theta0Over2 = parameters_mtx[0];
       Theta1Over2 = parameters_mtx[2];
       Phi0 = parameters_mtx[1];
       Phi1 = parameters_mtx[3];
    }
    else {
       Theta0Over2 = 0.0;
       Theta1Over2 = 0.0;
       Phi0 = 0.0;
       Phi1 = 0.0;
    }
    
    
    
    if (subtype == CONTROL_R){
        Matrix U3_matrix = calc_one_qubit_rotation(Theta0Over2,Phi0);
        apply_kernel_to( U3_matrix, input, false, parallel );
    }
    else{
        if (input.cols==1){

        Matrix U_2qbit(4,4);
        memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));
        U_2qbit[0].real = std::cos(Theta0Over2);
        U_2qbit[2].real = std::sin(Theta0Over2)*std::sin(Phi0);
        U_2qbit[2].imag = std::sin(Theta0Over2)*std::cos(Phi0);
        U_2qbit[1*4+3].real = -1.*std::sin(Theta1Over2)*std::sin(Phi1);
        U_2qbit[1*4+3].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1);
        U_2qbit[1*4+1].real = std::cos(Theta1Over2);
        U_2qbit[2*4+2].real = std::cos(Theta0Over2);
        U_2qbit[2*4].real = -1.*std::sin(Theta0Over2)*std::sin(Phi0);
        U_2qbit[2*4].imag = std::sin(Theta0Over2)*std::cos(Phi0);
        U_2qbit[3*4+3].real = std::cos(Theta1Over2);
        U_2qbit[3*4+1].real = std::sin(Theta1Over2)*std::sin(Phi1);
        U_2qbit[3*4+1].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1);
        //U_2qbit[0].real =1.;U_2qbit[7].real =1.;U_2qbit[10].real =1.;U_2qbit[13].real =1.; 
        // apply the computing kernel on the matrix
        std::vector<int> involved_qbits = {control_qbit,target_qbit};
        if (parallel){
          apply_large_kernel_to_input_AVX(U_2qbit,input,involved_qbits,input.size());
        }
        else{
            apply_large_kernel_to_input(U_2qbit,input,involved_qbits,input.size());
        }
        
        }
    
        else{
        
          Matrix U3_matrix = calc_one_qubit_rotation(Theta0Over2,Phi0);
          Matrix U3_matrix2 = calc_one_qubit_rotation(Theta1Over2,Phi1);
          if(parallel){
            apply_crot_kernel_to_matrix_input_AVX_parallel(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
          }
          else{
            apply_crot_kernel_to_matrix_input_AVX(U3_matrix2,U3_matrix, input, target_qbit, control_qbit, input.rows);
          }
      }

    }

}



/**
@brief Call to apply the gate on the input array/matrix by input*CROT
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param input The input array on which the gate is applied
*/
void 
CROT::apply_from_right( Matrix_real& parameters, Matrix& input ) {



}


/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
*/
std::vector<Matrix> 
CROT::apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CROT gate apply" << std::endl;
        print(sstream, 0);	   
        exit(-1);
    }

    std::vector<Matrix> ret;

    double Theta0Over2, Theta1Over2, Phi0, Phi1;

    if (subtype == CONTROL_R){
       Theta0Over2 = parameters_mtx[0];
       Phi0 = parameters_mtx[1];
    }
    else if (subtype == CONTROL_OPPOSITE){
       Theta0Over2 = parameters_mtx[0];
       Theta1Over2 = -1.*(parameters_mtx[0]);
       Phi0 = parameters_mtx[1];
       Phi1 = parameters_mtx[1];
    }
    else if (subtype == CONTROL_INDEPENDENT){
       Theta0Over2 = parameters_mtx[0];
       Theta1Over2 = parameters_mtx[2];
       Phi0 = parameters_mtx[1];
       Phi1 = parameters_mtx[3];
    }
    else {
       Theta0Over2 = 0.0;
       Theta1Over2 = 0.0;
       Phi0 = 0.0;
       Phi1 = 0.0;
    }



    // the resulting matrix
    if (subtype == CONTROL_R){



        Matrix res_mtx = input.copy();
        Matrix u3_1qbit = calc_one_qubit_rotation(Theta0Over2 +M_PIOver2 ,Phi0);
        apply_kernel_to( u3_1qbit, res_mtx, true, parallel );
        ret.push_back(res_mtx);
        Matrix res_mtx2 = input.copy();
        u3_1qbit = calc_one_qubit_rotation_deriv_Phi(Theta0Over2,Phi0+M_PIOver2 );
        apply_kernel_to( u3_1qbit, res_mtx2, true, parallel );
        ret.push_back(res_mtx2);
    }
    else{

    if (input.cols==1){
     if ( subtype == CONTROL_OPPOSITE){
        double Theta0Over2_shifted = Theta0Over2 + M_PIOver2;
        double Theta1Over2_shifted;

        Theta1Over2_shifted = -1.*Theta0Over2_shifted;
      
        double Phi0_shifted = Phi0 + M_PIOver2;
        double Phi1_shifted = Phi1 + M_PIOver2; 
    
    //Theta derivative
    Matrix res_mtx = input.copy();   
    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));
    
    U_2qbit[0].real = std::cos(Theta0Over2_shifted);
    U_2qbit[2].real = std::sin(Theta0Over2_shifted)*std::sin(Phi0);
    U_2qbit[2].imag = std::sin(Theta0Over2_shifted)*std::cos(Phi0);
    
    U_2qbit[2*4+2].real = std::cos(Theta0Over2_shifted);
    U_2qbit[2*4].real = -1.*std::sin(Theta0Over2_shifted)*std::sin(Phi0);
    U_2qbit[2*4].imag = std::sin(Theta0Over2_shifted)*std::cos(Phi0);
    
    U_2qbit[1*4+3].real = -1.*std::sin(Theta1Over2_shifted)*std::sin(Phi1);
    U_2qbit[1*4+3].imag = -1.*std::sin(Theta1Over2_shifted)*std::cos(Phi1);
    U_2qbit[1*4+1].real = std::cos(Theta1Over2_shifted);
    
    U_2qbit[3*4+3].real = std::cos(Theta1Over2_shifted);
    U_2qbit[3*4+1].real = std::sin(Theta1Over2_shifted)*std::sin(Phi1);
    U_2qbit[3*4+1].imag = -1.*std::sin(Theta1Over2_shifted)*std::cos(Phi1);


    std::vector<int> involved_qbits = {control_qbit,target_qbit};

    apply_large_kernel_to_input(U_2qbit,res_mtx,involved_qbits,res_mtx.size());
    ret.push_back(res_mtx);
    
    //Phi derivative
    Matrix res_mtx1 = input.copy();   
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));

    U_2qbit[2].real = std::sin(Theta0Over2)*std::sin(Phi0_shifted);
    U_2qbit[2].imag = std::sin(Theta0Over2)*std::cos(Phi0_shifted);

    U_2qbit[2*4].real = -1.*std::sin(Theta0Over2)*std::sin(Phi0_shifted);
    U_2qbit[2*4].imag = std::sin(Theta0Over2)*std::cos(Phi0_shifted);
    
    U_2qbit[1*4+3].real = -1.*std::sin(Theta1Over2)*std::sin(Phi1_shifted);
    U_2qbit[1*4+3].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1_shifted);

    U_2qbit[3*4+1].real = std::sin(Theta1Over2)*std::sin(Phi1_shifted);
    U_2qbit[3*4+1].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1_shifted);

    apply_large_kernel_to_input(U_2qbit,res_mtx1,involved_qbits,res_mtx1.size());
    ret.push_back(res_mtx1);

    }
    else{
    double Theta0Over2_shifted = Theta0Over2 + M_PIOver2;
    double Theta1Over2_shifted = Theta1Over2 + M_PIOver2;
    double Phi0_shifted = Phi0 + M_PIOver2;
    double Phi1_shifted = Phi1 + M_PIOver2; 
    
    //theta0 derivative
    Matrix res_mtx = input.copy();   
    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));
    U_2qbit[0].real = std::cos(Theta0Over2_shifted);
    U_2qbit[2].real = std::sin(Theta0Over2_shifted)*std::sin(Phi0);
    U_2qbit[2].imag = std::sin(Theta0Over2_shifted)*std::cos(Phi0);
    
    U_2qbit[2*4+2].real = std::cos(Theta0Over2_shifted);
    U_2qbit[2*4].real = -1.*std::sin(Theta0Over2_shifted)*std::sin(Phi0);
    U_2qbit[2*4].imag = std::sin(Theta0Over2_shifted)*std::cos(Phi0);

    std::vector<int> involved_qbits = {control_qbit,target_qbit};

    apply_large_kernel_to_input(U_2qbit,res_mtx,involved_qbits,res_mtx.size());
    ret.push_back(res_mtx);
    
    //Phi0 derivative
    Matrix res_mtx1 = input.copy();   
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));

    U_2qbit[2].real = std::sin(Theta0Over2)*std::sin(Phi0_shifted);
    U_2qbit[2].imag = std::sin(Theta0Over2)*std::cos(Phi0_shifted);

    U_2qbit[2*4].real = -1.*std::sin(Theta0Over2)*std::sin(Phi0_shifted);
    U_2qbit[2*4].imag = std::sin(Theta0Over2)*std::cos(Phi0_shifted);

    apply_large_kernel_to_input(U_2qbit,res_mtx1,involved_qbits,res_mtx1.size());
    ret.push_back(res_mtx1);
      
    //theta1 derivative
    Matrix res_mtx2 = input.copy();   
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));
    
    U_2qbit[1*4+3].real = -1.*std::sin(Theta1Over2_shifted)*std::sin(Phi1);
    U_2qbit[1*4+3].imag = -1.*std::sin(Theta1Over2_shifted)*std::cos(Phi1);
    U_2qbit[1*4+1].real = std::cos(Theta1Over2_shifted);
    
    U_2qbit[3*4+3].real = std::cos(Theta1Over2_shifted);
    U_2qbit[3*4+1].real = std::sin(Theta1Over2_shifted)*std::sin(Phi1);
    U_2qbit[3*4+1].imag = -1.*std::sin(Theta1Over2_shifted)*std::cos(Phi1);

    apply_large_kernel_to_input(U_2qbit,res_mtx2,involved_qbits,res_mtx2.size());
    ret.push_back(res_mtx2);
    
    //Phi1 derivative
    Matrix res_mtx3 = input.copy();   
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));

    U_2qbit[1*4+3].real = -1.*std::sin(Theta1Over2)*std::sin(Phi1_shifted);
    U_2qbit[1*4+3].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1_shifted);

    U_2qbit[3*4+1].real = std::sin(Theta1Over2)*std::sin(Phi1_shifted);
    U_2qbit[3*4+1].imag = -1.*std::sin(Theta1Over2)*std::cos(Phi1_shifted);

    apply_large_kernel_to_input(U_2qbit,res_mtx3,involved_qbits,res_mtx3.size());
    ret.push_back(res_mtx3);
    }
    }
    else{
     if (subtype == CONTROL_OPPOSITE){
        double Theta0Over2_shifted = Theta0Over2 + M_PIOver2;
        double Theta1Over2_shifted;

        Theta1Over2_shifted = -1.*Theta0Over2_shifted;
      
      //Theta derivative
      Matrix res_mtx = input.copy();   
      Matrix U3_matrix = calc_one_qubit_rotation(Theta0Over2_shifted,Phi0);
      Matrix U3_matrix2 = calc_one_qubit_rotation(Theta1Over2_shifted,Phi1);
      
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx, target_qbit, control_qbit, res_mtx.rows);
      ret.push_back(res_mtx);
      ///Phi derivative
      double Phi0_shifted = Phi0 + M_PIOver2;
      double Phi1_shifted = Phi1 + M_PIOver2; 
      Matrix res_mtx1 = input.copy();   
      U3_matrix = calc_one_qubit_rotation_deriv_Phi(Theta0Over2,Phi0_shifted);
      U3_matrix2 = calc_one_qubit_rotation_deriv_Phi(Theta1Over2,Phi1_shifted);
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx1, target_qbit, control_qbit, res_mtx1.rows);
      ret.push_back(res_mtx1); 
    }
    else if (subtype == CONTROL_INDEPENDENT){
    
      ///Theta0 derivative
      double Theta0Over2_shifted = Theta0Over2 + M_PIOver2;
      Matrix res_mtx = input.copy();   
      Matrix U3_matrix = calc_one_qubit_rotation(Theta0Over2_shifted,Phi0);
      Matrix U3_matrix2(2,2);
      memset(U3_matrix2.get_data(),0.0,U3_matrix2.size()*sizeof(QGD_Complex16));
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx, target_qbit, control_qbit, res_mtx.rows);
      ret.push_back(res_mtx);
      
      ///Phi0 derivative
      double Phi0_shifted = Phi0 + M_PIOver2;
      Matrix res_mtx1 = input.copy();   
      U3_matrix = calc_one_qubit_rotation_deriv_Phi(Theta0Over2,Phi0_shifted);
      memset(U3_matrix2.get_data(),0.0,U3_matrix2.size()*sizeof(QGD_Complex16));
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx1, target_qbit, control_qbit, res_mtx1.rows);
      ret.push_back(res_mtx1);
      
    ///Theta1 derivative
      double Theta1Over2_shifted = Theta1Over2 + M_PIOver2;
      Matrix res_mtx2 = input.copy();   
      U3_matrix2 = calc_one_qubit_rotation(Theta1Over2_shifted,Phi1);
      memset(U3_matrix.get_data(),0.0,U3_matrix.size()*sizeof(QGD_Complex16));
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx2, target_qbit, control_qbit, res_mtx2.rows);
      ret.push_back(res_mtx2);
      
      ///Phi1 derivative
      double Phi1_shifted = Phi1 + M_PIOver2;
      Matrix res_mtx3 = input.copy();   
      U3_matrix2 = calc_one_qubit_rotation_deriv_Phi(Theta1Over2,Phi1_shifted);
      memset(U3_matrix.get_data(),0.0,U3_matrix.size()*sizeof(QGD_Complex16));
      apply_crot_kernel_to_matrix_input_AVX(U3_matrix2, U3_matrix, res_mtx3, target_qbit, control_qbit, res_mtx3.rows);
      ret.push_back(res_mtx3);

    }
    else{
            std::stringstream sstream;
	sstream << "Subtype not implemented for gradient" << std::endl;
        print(sstream, 1);	      
        exit(-1);
    }
      
    }
    }

    return ret;


}
 crot_type CROT::get_subtype(){
    return subtype;
 }

/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num_in The number of qubits
*/
void CROT::set_qbit_num(int qbit_num_in) {

        // setting the number of qubits
        Gate::set_qbit_num(qbit_num_in);
}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CROT::reorder_qubits( std::vector<int> qbit_list) {

    Gate::reorder_qubits(qbit_list);

}

/**
@brief Call to set the final optimized parameters of the gate.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void CROT::set_optimized_parameters(double Theta0Over2, double Phi0, double Theta1Over2, double Phi1) {

    parameters = Matrix_real(1, 4);

    parameters[0] = Theta0Over2;
    parameters[1] = Phi0;
    parameters[2] = Theta1Over2;
    parameters[3] = Phi1;

}


/**
@brief Call to get the final optimized parameters of the gate.
@param parameters_in Preallocated pointer to store the parameters ThetaOver2, Phi and Lambda of the U3 gate.
*/
Matrix_real CROT::get_optimized_parameters() {

    return parameters.copy();

}



/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CROT* CROT::clone() {

    CROT* ret = new CROT(qbit_num, target_qbit, control_qbit, subtype);

    if ( parameters.size() >0 ) {
        ret->set_optimized_parameters(parameters[0],parameters[1],parameters[2],parameters[3]);
    }

    ret->set_parameter_start_idx( get_parameter_start_idx() );

    return ret;

}

Matrix CROT::calc_one_qubit_rotation(double ThetaOver2, double Phi) {
    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = std::cos(ThetaOver2); 
    u3_1qbit[0].imag = 0;
    
    u3_1qbit[1].real = -1.*std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[1].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[2].real = std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[2].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[3].real = std::cos(ThetaOver2); 
    u3_1qbit[3].imag = 0;
    
    return u3_1qbit;
}

Matrix CROT::calc_one_qubit_rotation_deriv_Phi(double ThetaOver2, double Phi) {
    Matrix u3_1qbit = Matrix(2,2);
    u3_1qbit[0].real = 0; 
    u3_1qbit[0].imag = 0;
    
    u3_1qbit[1].real = -1.*std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[1].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[2].real = std::sin(ThetaOver2)*std::sin(Phi); 
    u3_1qbit[2].imag = -1.*std::sin(ThetaOver2)*std::cos(Phi);
    
    u3_1qbit[3].real = 0; 
    u3_1qbit[3].imag = 0;
    
    return u3_1qbit;
}

/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
@param parameters The parameter array corresponding to the circuit in which the gate is embedded
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
CROT::extract_parameters( Matrix_real& parameters ) {

    if ( get_parameter_start_idx() + get_parameter_num() > parameters.size()  ) {
        std::string err("CROT::extract_parameters: Cant extract parameters, since the dinput arary has not enough elements.");
        throw err;     
    }
    if (subtype == CONTROL_R || subtype == CONTROL_OPPOSITE){
    Matrix_real extracted_parameters(1,2);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx() + 1 ], 2*M_PI);
    return extracted_parameters;
    }
    else if (subtype == CONTROL_INDEPENDENT){
    Matrix_real extracted_parameters(1,4);

    extracted_parameters[0] = std::fmod( 2*parameters[ get_parameter_start_idx() ], 4*M_PI);
    extracted_parameters[1] = std::fmod( parameters[ get_parameter_start_idx() + 1 ], 2*M_PI);
    extracted_parameters[2] = std::fmod( 2*parameters[ get_parameter_start_idx() +2 ], 4*M_PI);
    extracted_parameters[3] = std::fmod( parameters[ get_parameter_start_idx() + 3 ], 2*M_PI);
    return extracted_parameters;
    }
    Matrix_real extracted_parameters(1,1);
    return extracted_parameters;

}
