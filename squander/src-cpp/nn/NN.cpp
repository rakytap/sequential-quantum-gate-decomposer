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
/*! \file NN.cpp
    \brief Class containing methods for SQUANDER neural network component
*/

#include "NN.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include <cstdlib>
#include <time.h>

#include "tbb/tbb.h"


/** Nullary constructor of the class
@return An instance of the class
*/
NN::NN() {

    // seedign the random generator
    gen = std::mt19937(rd());

    
#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif



}



/** Constructor of the class
@param The list of conenctions between the qubits
@return An instance of the class
*/
NN::NN( std::vector<matrix_base<int>> topology_in ) {

    // seedign the random generator
    gen = std::mt19937(rd());


    // setting the topology
    topology = topology_in;

    
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
NN::create_randomized_parameters( int num_of_parameters, int qbit_num, int levels, Matrix_real& parameters, matrix_base<int8_t>& nontrivial_adaptive_layers ) {

    if ( parameters.size() != num_of_parameters ) {
        parameters = Matrix_real( 1, num_of_parameters );
        memset( parameters.get_data(), 0.0, parameters.size()*sizeof(double) );
    }


    // the number of adaptive layers in one level
    int num_of_adaptive_layers = qbit_num*(qbit_num-1)/2 * levels;

    if ( nontrivial_adaptive_layers.size() != num_of_adaptive_layers ) {
        nontrivial_adaptive_layers = matrix_base<int8_t>( num_of_adaptive_layers, 1);
    }

    
    //parameters[0:qbit_num*3] = np.random.rand(qbit_num*3)*2*np.pi
    //parameters[2*qbit_num:3*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    //parameters[qbit_num:2*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    //parameters[3*qbit_num-1] = 0
    //parameters[3*qbit_num-2] = 0

    std::uniform_int_distribution<int16_t> distrib(0, 1);    
    std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);    
    std::uniform_real_distribution<> distrib_real2(0.0, M_PI);    

    


    for(int idx = 0; idx < 3*qbit_num; idx++) {
        if ( idx % 3 == 0 ) {
            parameters[idx] = distrib_real2(gen);  // values for theta/2 of U3 can be reduced to [0,PI], since 2PI phase shift can be agregated into Phi and Lambda
        }
        else {
            parameters[idx] = distrib_real(gen);
        }        
    }   


    for( int layer_idx=0; layer_idx<num_of_adaptive_layers; layer_idx++) {

        int8_t nontrivial_adaptive_layer = distrib(gen);
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer;

        if (nontrivial_adaptive_layer) {
        
            // set the radom parameters of the chosen adaptive layer
            int start_idx = qbit_num*3 + layer_idx*7;

            int end_idx = start_idx + 7;
        
        
            for(int jdx = start_idx; jdx < end_idx; jdx++) {
                if ( (jdx-start_idx) % 3 == 0 ) {
                    parameters[jdx] = distrib_real2(gen);  // values for theta/2 of U3 can be reduced to [0,PI], since 2PI phase shift can be agregated into Phi and Lambda
                }
                else {
                    parameters[jdx] = distrib_real(gen);
                }
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
    
    double phi1   = std::atan2( Y0, X0 ); 
    phi1 = phi1 < 0 ? 2*M_PI + phi1 : phi1;
    double phi2   = std::atan2( Y1, X1 ) + M_PI;
    phi2 = phi2 < 0 ? 2*M_PI + phi2 : phi2;

    
    double theta1;
    if ( std::abs(X0) > 1e-8 ) {
        theta1 = std::atan2( X0/cos(phi1), Z0 )/2 ;
    }
    else {
        theta1 = std::atan2( Y0/sin(phi1), Z0 )/2;    
    }
    

    double theta2;
    if ( std::abs(X1) > 1e-8 ) {
        theta2 = (M_PI + atan2( X1/cos(phi2), Z1 ))/2;
    }
    else {
        theta2 = (M_PI + atan2( Y1/sin(phi2), Z1 ))/2;    
    }    
    
    chanels[0] = theta1;
    chanels[1] = phi1;
    chanels[2] = theta2;
    chanels[3] = phi2;            
       
//std::cout << theta1 << " " << phi1 << " " << theta2 << " " << phi2 << std::endl;

}

/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@param target_qbit The target qubit for which the chanels are calculated
@param chanels output array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
void NN::get_nn_chanels( const Matrix& Umtx, const int& target_qbit, Matrix_real& chanels) {


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

    if ( chanels.size() != dim_over_2*dim_over_2*4 ) {
        chanels = Matrix_real( dim_over_2, dim_over_2*4 );
    }
    
    Matrix_real chanels_reshaped( chanels.get_data(), dim_over_2, dim_over_2*4 );


    
    int index_pair_distance = 1 << target_qbit;
    
    //std::cout << "target_qbit: " << target_qbit << " index pair distance: " << index_pair_distance << std::endl;
    



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
            
            Matrix_real chanels_kernel( chanels_reshaped.get_data() + idx*chanels_reshaped.stride + 4*jdx, 1, 4, chanels_reshaped.stride);
            get_nn_chanels_from_kernel( kernel_up, kernel_down, chanels_kernel);

            

        }
    }


    return;

}

/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param qbit_num Th number of qubites
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@param chanels output array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
void NN::get_nn_chanels( int qbit_num, const Matrix& Umtx, Matrix_real& chanels) {


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

    if ( chanels.size() != dim_over_2*dim_over_2*4*qbit_num ) {
        chanels = Matrix_real( dim_over_2, dim_over_2*4*qbit_num );
    }
    
    Matrix_real chanels_reshaped( chanels.get_data(), dim_over_2, dim_over_2*4*qbit_num );

    

    
    //std::cout << "target_qbit: " << target_qbit << " index pair distance: " << index_pair_distance << std::endl;
    



    // calculate the individual chanels
    for (int idx = 0; idx<dim_over_2; idx++ ) {
    
        for (int jdx = 0; jdx<dim_over_2; jdx++ ) {

            for (int target_qbit=0; target_qbit<qbit_num; target_qbit++) {

                int index_pair_distance = 1 << target_qbit;

                // row index pairs
                int row_idx = idx >> target_qbit; // higher bits of idx
                row_idx = row_idx << (target_qbit+1); 
        
                int tmp_idx = (idx & ( (1 << (target_qbit)) - 1 ) ); // lower target_bit bits from idx

        
                row_idx = row_idx + tmp_idx; // the index corresponding to state 0 of the target qbit
        
                int row_idx_pair = row_idx ^ index_pair_distance;
            //std::cout << idx << " " << row_idx << " " << row_idx_pair << " " << tmp << std::endl;
        
                int stride_kernel = index_pair_distance * Umtx.stride;



                // column index pairs
                int col_idx = jdx >> target_qbit; // higher bits of idx
                col_idx = col_idx << (target_qbit+1); 
        
                int tmp_jdx = (jdx & ( (1 << (target_qbit)) - 1 ) ); // lower target_bit bits from idx

        
                col_idx = col_idx + tmp_jdx; // the index corresponding to state 0 of the target qbit
        
                int col_idx_pair = col_idx ^ index_pair_distance;


                Matrix kernel_up   = Matrix(Umtx.get_data() + row_idx*Umtx.stride + col_idx, 2, 1, stride_kernel );
                Matrix kernel_down = Matrix(Umtx.get_data() + row_idx*Umtx.stride + col_idx_pair, 2, 1, stride_kernel );            
            
                Matrix_real chanels_kernel( chanels_reshaped.get_data() + idx*chanels_reshaped.stride + 4*qbit_num*jdx + 4*target_qbit, 1, 4, chanels_reshaped.stride);
                get_nn_chanels_from_kernel( kernel_up, kernel_down, chanels_kernel);

            }

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
NN::get_nn_chanels(int qbit_num, int levels, Matrix_real& chanels, matrix_base<int8_t>& nontrivial_adaptive_layers) {





    //matrix size of the unitary
    int matrix_size = 1 << qbit_num;

    // empty config parameters
    std::map<std::string, Config_Element> config_int;


    // creating a class to decompose the unitary
    N_Qubit_Decomposition_adaptive cDecompose( Matrix(0,0), qbit_num, 0, 0, topology, config_int );
        
    //adding decomposing layers to the gat structure
    for( int idx=0; idx<levels; idx++) {
        cDecompose.add_adaptive_layers();
    }        

    cDecompose.add_finalyzing_layer();


    //get the number of free parameters
    int num_of_parameters = cDecompose.get_parameter_num();
    
//std::cout << "number of free parameters: " << num_of_parameters << std::endl;    


    // create randomized parameters having number of nontrivial adaptive blocks determined by the parameter nontrivial_ratio
    Matrix_real parameters;
    create_randomized_parameters( num_of_parameters, qbit_num, levels, parameters, nontrivial_adaptive_layers );
    
//parameters.print_matrix();

    // getting the unitary corresponding to quantum circuit
    Matrix&& Umtx = cDecompose.get_matrix( parameters );

    // generate chanels
    get_nn_chanels( qbit_num, Umtx, chanels );






}




/** 
@brief call retrieve the channels for the neural network associated with a single, randomly generated unitary
@param qbit_num The number of qubits
@param levels The number of adaptive levels to be randomly constructed
@param samples_num The number of samples
@param chanles output argument to return with an array containing the chanels prepared for the neural network. The array has dimensions [ samples_num, dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
@param parameters output argument of the randomly created parameters
*/
void
NN::get_nn_chanels(int qbit_num, int levels, int samples_num, Matrix_real& chanels, matrix_base<int8_t>& nontrivial_adaptive_layers) {


    // temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

//Matrix_real parameters;

    if ( samples_num == 1 ) {
        get_nn_chanels(qbit_num, levels, chanels, nontrivial_adaptive_layers);
        return;
    }

    if ( samples_num < 1 ) {
        std::string err("Number of samples must be greater than 0.");
        throw err;
    }


    // query the first sample to infer the memory needed to be allocated
    Matrix_real chanels_1;
    matrix_base<int8_t> nontrivial_adaptive_layers_1;

    get_nn_chanels(qbit_num, levels, chanels_1, nontrivial_adaptive_layers_1);

    // allocate memory for the outputs
    chanels    = Matrix_real(1, samples_num*chanels_1.size());
    //parameters = Matrix_real(samples_num, parameters_1.size());
    nontrivial_adaptive_layers = matrix_base<int8_t>( 1, samples_num*nontrivial_adaptive_layers_1.size() );
    memset( chanels.get_data(), 0.0, chanels.size()*sizeof(double) );
    memset( nontrivial_adaptive_layers.get_data(), 0, nontrivial_adaptive_layers.size()*sizeof(int8_t) );

    // copy the result of the first iteration into the output
    memcpy( chanels.get_data(), chanels_1.get_data(), chanels_1.size()*sizeof(double) );
    //memcpy( parameters.get_data(), parameters_1.get_data(), parameters_1.size()*sizeof(double) );
    memcpy( nontrivial_adaptive_layers.get_data(), nontrivial_adaptive_layers_1.get_data(), nontrivial_adaptive_layers_1.size()*sizeof(int8_t) );

    // do the remaining cycles

    tbb::parallel_for( tbb::blocked_range<int>(1,samples_num), [&](tbb::blocked_range<int> r) {

        for (int idx=r.begin(); idx<r.end(); idx++) {

//    for (int idx=1; idx<samples_num; idx++) {

            Matrix_real chanels_idx( chanels.get_data()+idx*chanels_1.size(), 1, chanels_1.size() );

            //Matrix_real parameters_idx;// parameters.get_data()+idx*parameters_1.size(), 1, parameters_1.size() );
            matrix_base<int8_t> nontrivial_adaptive_layers_idx( nontrivial_adaptive_layers.get_data()+idx*nontrivial_adaptive_layers_1.size(), 1, nontrivial_adaptive_layers_1.size() );

            get_nn_chanels(qbit_num, levels, chanels_idx, nontrivial_adaptive_layers_idx);
//    }


        }

    });





//Matrix_real chanels_reshaped(chanels.get_data(), chanels.size()/4, 4);
//chanels_reshaped.print_matrix();


//matrix_base<int8_t> nontrivial_adaptive_layers_reshaped(nontrivial_adaptive_layers.get_data(), nontrivial_adaptive_layers.size()/nontrivial_adaptive_layers_1.size(), nontrivial_adaptive_layers_1.size());
//nontrivial_adaptive_layers_reshaped.print_matrix();

//std::cout << chanels_1.size() << " " << parameters_1.size() << std::endl;


#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif


}


