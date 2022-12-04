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
#include <cstdlib>



/** Nullary constructor of the class
@return An instance of the class
*/
NN::NN() {


#if CBLAS==1
    num_threads = mkl_get_max_threads();
#elif CBLAS==2
    num_threads = openblas_get_num_threads();
#endif



}

	
/** 
@brief call retrieve the channels for the neural network associated with a single 2x2 kernel
@return return with an 1x4 array containing the chanels prepared for the neural network. (dimension 4 stands for theta_up, phi, theta_down , lambda)
*/
void 
NN::get_nn_chanels_from_kernel( Matrix& kernel, Matrix_real& chanels) {

    //kernel.print_matrix(); 
    
    // calculate expectation values of the Pauli operators
    
    
    QGD_Complex16& element00 = kernel[0]; 
    QGD_Complex16& element01 = kernel[1];
    QGD_Complex16& element10 = kernel[kernel.stride];
    QGD_Complex16& element11 = kernel[kernel.stride+1];
    
    //conj(e00)*e00 - conj(e10)*e10 )    
    double Z0 = element00.real*element00.real + element00.imag*element00.imag - element10.real*element10.real - element10.imag*element10.imag;
    double Z1 = element01.real*element01.real + element01.imag*element01.imag - element11.real*element11.real - element11.imag*element11.imag;

    //conj(e00)*e10 + conj(e10)*e00    
    double X0 = element00.real*element10.real +  element00.imag*element10.imag + element10.real*element00.real + element10.imag*element00.imag;
    double X1 = element01.real*element11.real +  element01.imag*element11.imag + element11.real*element01.real + element11.imag*element01.imag;    
        
    //i*( conj(e00)*e10 - conj(e10)*e00 )
    double Y0 = (element00.real*element10.imag - element00.imag*element10.real - element10.real*element00.imag + element10.imag*element00.real);
    double Y1 = (element01.real*element11.imag - element01.imag*element11.real - element11.real*element01.imag + element11.imag*element01.real);   
    
    double phi    = std::fmod( std::atan2( Y0, X0 ), 2*M_PI); 
    double lambda = std::fmod( std::atan2( Y1, X1 ) + M_PI, 2*M_PI);

    
    double theta1;
    if ( std::abs(X0) > 1e-8 ) {
        theta1 = std::fmod( std::atan2( X0/cos(phi), Z0 )/2, 2*M_PI) ;
    }
    else {
        theta1 = std::fmod( std::atan2( Y0/sin(phi), Z0 )/2, 2*M_PI);    
    }
    

    double theta2;
    if ( std::abs(X1) > 1e-8 ) {
        theta2 = std::fmod( (M_PI + atan2( X1/cos(phi), Z1 ))/2, 2*M_PI);
    }
    else {
        theta2 = std::fmod( (M_PI + atan2( Y1/sin(phi), Z1 ))/2, 2*M_PI);    
    }    
       
std::cout << theta1 << " " << phi << " " << theta2 << " " << lambda << std::endl;

}



/** 
@brief call retrieve the channels for the neural network associated with a single unitary
@param Umtx A unitary of dimension dim x dim, where dim is a power of 2.
@return return with an array containing the chanels prepared for th eneural network. The array has dimensions [ dim/2, dim/2, 4 ] (dimension "4" stands for theta_up, phi, theta_down , lambda)
*/
Matrix_real NN::get_nn_chanels( Matrix& Umtx) {

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

    Matrix_real chanels( dim_over_2, dim_over_2*4 );
    
    int target_qbit = 0;
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
    
        for (int col_idx = 0; col_idx<dim_over_2; col_idx++ ) {

            Matrix kernel = Matrix(Umtx.get_data() + row_idx*Umtx.stride + (col_idx<<1), 2, 2, stride_kernel );
            
            
            Matrix_real chanels_kernel( chanels.get_data() + idx*chanels.stride + (col_idx<<4), 1, 4, chanels.stride);
            get_nn_chanels_from_kernel( kernel, chanels_kernel);

            

        }
    }


    return chanels;

}




