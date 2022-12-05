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
/*! \file NN.cpp
    \brief Class containing methods for SQUANDER neural network component
*/

#include "NN.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include <cstdlib>



/** Nullary constructor of the class
@return An instance of the class
*/
NN::NN() {



    srand(time(NULL));   // Initialization, should only be called once.
    
#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif



}


/** 
@brief Call to construct random parameter, with limited number of non-trivial adaptive layers
@param num_of_parameters The number of parameters
*/
void
NN::create_randomized_parameters( int num_of_parameters, int qbit_num, int levels, Matrix_real& parameters ) {



    parameters = Matrix_real( 1, num_of_parameters );


    // the number of adaptive layers in one level
    int num_of_adaptive_layers = qbit_num*(qbit_num-1)/2 * levels;

    
    //parameters[0:qbit_num*3] = np.random.rand(qbit_num*3)*2*np.pi
    //parameters[2*qbit_num:3*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    //parameters[qbit_num:2*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    //parameters[3*qbit_num-1] = 0
    //parameters[3*qbit_num-2] = 0
    
    matrix_base<int> nontrivial_adaptive_layers( num_of_adaptive_layers, 1);
    
    for(int idx = 0; idx < 3*qbit_num; idx++) {
        parameters[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;         
    }   
     

    for( int layer_idx=0; layer_idx<num_of_adaptive_layers; layer_idx++) {

        int nontrivial_adaptive_layer = rand() % 2;
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer;

        if (nontrivial_adaptive_layer) {
        
            // set the radom parameters of the chosen adaptive layer
            int start_idx = qbit_num*3 + layer_idx*7;

            int end_idx = start_idx + 7;
        
        
            for(int jdx = start_idx; jdx < end_idx; jdx++) {
                parameters[jdx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;         
            }       
            
            
        }
        
    }


    return;


}

	
/** 
@brief call retrieve the channels for the neural network associated with a single 2x2 kernel
@return return with an 1x4 array containing the chanels prepared for the neural network. (dimension 4 stands for theta_up, phi, theta_down , lambda)
*/
void 
NN::get_nn_chanels_from_kernel( Matrix& kernel_up, Matrix& kernel_down, Matrix_real& chanels) {

    //kernel.print_matrix(); 
    
    // calculate expectation values of the Pauli operators
    
    
    QGD_Complex16& element00 = kernel_up[0]; 
    QGD_Complex16& element01 = kernel_down[0];
    QGD_Complex16& element10 = kernel_up[kernel_up.stride];
    QGD_Complex16& element11 = kernel_down[kernel_down.stride];
    
    //conj(e00)*e00 - conj(e10)*e10 )  -- expectation value of Z operator
    double Z0 = element00.real*element00.real + element00.imag*element00.imag - element10.real*element10.real - element10.imag*element10.imag;
    double Z1 = element01.real*element01.real + element01.imag*element01.imag - element11.real*element11.real - element11.imag*element11.imag;

    //conj(e00)*e10 + conj(e10)*e00  -- expectation value of X operator    
    double X0 = element00.real*element10.real +  element00.imag*element10.imag + element10.real*element00.real + element10.imag*element00.imag;
    double X1 = element01.real*element11.real +  element01.imag*element11.imag + element11.real*element01.real + element11.imag*element01.imag;    
        
    //i*( conj(e00)*e10 - conj(e10)*e00 )  -- expectation value of Y operator
    double Y0 = (element00.real*element10.imag - element00.imag*element10.real - element10.real*element00.imag + element10.imag*element00.real);
    double Y1 = (element01.real*element11.imag - element01.imag*element11.real - element11.real*element01.imag + element11.imag*element01.real);   
    
    double phi1   = std::fmod( std::atan2( Y0, X0 ), 2*M_PI); 
    double phi2   = std::fmod( std::atan2( Y1, X1 ) + M_PI, 2*M_PI);

    
    double theta1;
    if ( std::abs(X0) > 1e-8 ) {
        theta1 = std::fmod( std::atan2( X0/cos(phi1), Z0 )/2, 2*M_PI) ;
    }
    else {
        theta1 = std::fmod( std::atan2( Y0/sin(phi1), Z0 )/2, 2*M_PI);    
    }
    

    double theta2;
    if ( std::abs(X1) > 1e-8 ) {
        theta2 = std::fmod( (M_PI + atan2( X1/cos(phi2), Z1 ))/2, 2*M_PI);
    }
    else {
        theta2 = std::fmod( (M_PI + atan2( Y1/sin(phi2), Z1 ))/2, 2*M_PI);    
    }    
    
    chanels[0] = theta1;
    chanels[1] = phi1;
    chanels[2] = theta2;
    chanels[3] = phi2;            
       
std::cout << theta1 << " " << phi1 << " " << theta2 << " " << phi2 << std::endl;

}



/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@param chanels output array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
void NN::get_nn_chanels( const Matrix& Umtx, Matrix_real& chanels) {

std::cout << "pppppppppppppppp " << std::endl;
Umtx.print_matrix();

    if ( Umtx.rows != Umtx.cols ) {
        std::string err("The unitary must be a square matrix.");
        throw err;
    }

    if ( Umtx.rows <= 0 ) {
        std::string err("The unitary must be larger than 0x0.");
        throw err;
    }

    

    int dim = Umtx.rows;
    int dim_over_2 = dim/2;

    chanels = Matrix_real( dim_over_2, dim_over_2*4 );
    
    int target_qbit = 1;
    int index_pair_distance = 1 << target_qbit;
    
    std::cout << "target_qbit: " << target_qbit << " index pair distance: " << index_pair_distance << std::endl;
    



    // calculate the individual chanels
    for (int idx = 0; idx<dim_over_2; idx++ ) {
    
        int row_idx = idx >> target_qbit; // higher bits of idx
        row_idx = row_idx << (target_qbit+1); 
        
        int tmp = (idx & ( (1 << (target_qbit)) - 1 ) ); // lower target_bit bits from idx

        
        row_idx = row_idx + tmp; // the index corresponding to state 0 of the target qbit
        
        int row_idx_pair = row_idx ^ index_pair_distance;
        //std::cout << idx << " " << row_idx << " " << row_idx_pair << " " << tmp << std::endl;
        
        int stride_kernel = index_pair_distance * Umtx.stride;
    
        for (int jdx = 0; jdx<dim_over_2; jdx++ ) {
        
            int col_idx = jdx >> target_qbit; // higher bits of idx
            col_idx = col_idx << (target_qbit+1); 
        
            int tmp = (jdx & ( (1 << (target_qbit)) - 1 ) ); // lower target_bit bits from idx

        
            col_idx = col_idx + tmp; // the index corresponding to state 0 of the target qbit
        
            int col_idx_pair = col_idx ^ index_pair_distance;


            Matrix kernel_up   = Matrix(Umtx.get_data() + row_idx*Umtx.stride + col_idx, 2, 1, stride_kernel );
            Matrix kernel_down = Matrix(Umtx.get_data() + row_idx*Umtx.stride + col_idx_pair, 2, 1, stride_kernel );            
            
            
            Matrix_real chanels_kernel( chanels.get_data() + idx*chanels.stride + 4*jdx, 1, 4, chanels.stride);
            get_nn_chanels_from_kernel( kernel_up, kernel_down, chanels_kernel);

            

        }
    }


    return;

}



/** 
@brief call retrieve the channels for the neural network associated with a single, randomly generated unitary
@param qbit_num The number of qubits
@param levels The number of adaptive levels to be randomly constructed
@param chanles output argument to return with an array containing the chanels prepared for the neural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
@param parameters output argument of the randomly created parameters
*/
void
NN::get_nn_chanels(int qbit_num, int levels, Matrix_real& chanels, Matrix_real& parameters) {





    //matrix size of the unitary
    int matrix_size = 1 << qbit_num;

    // creating a class to decompose the unitary
    N_Qubit_Decomposition_adaptive cDecompose( Matrix(0,0), qbit_num, 0, 0 );
        
    //adding decomposing layers to the gat structure
    for( int idx=0; idx<levels; idx++) {
        cDecompose.add_adaptive_layers();
    }        

    cDecompose.add_finalyzing_layer();


    //get the number of free parameters
    int num_of_parameters = cDecompose.get_parameter_num();
    
std::cout << "number of free parameters: " << num_of_parameters << std::endl;    


    // create randomized parameters having number of nontrivial adaptive blocks determined by the parameter nontrivial_ratio
    //parameters, nontrivial_adaptive_layers = 
    create_randomized_parameters( num_of_parameters, qbit_num, levels, parameters );
    
    parameters.print_matrix();

    // getting the unitary corresponding to quantum circuit
    Matrix&& Umtx = cDecompose.get_matrix( parameters );


    // generate chanels
    get_nn_chanels( Umtx, chanels ); 






}




