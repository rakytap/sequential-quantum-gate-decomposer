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
/*! \file N_Qubit_Decomposition_Cost_Function.cpp
    \brief Methods to calculate the cost function of the final optimization problem (supporting parallel computations).
*/

#include "N_Qubit_Decomposition_Cost_Function.h"
//#include <tbb/parallel_for.h>



/**
@brief Call co calculate the cost function during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param trace_offset The offset in the first columns from which the "trace" is calculated. In this case Tr(A) = sum_(i-offset=j) A_{ij}
@return Returns with the calculated cost function.
*/
double get_cost_function(Matrix matrix, int trace_offset) {

    int matrix_size = matrix.cols ;
/*
    tbb::combinable<double> priv_partial_cost_functions{[](){return 0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, matrix_size, 1), functor_cost_fnc( matrix, &priv_partial_cost_functions ));
*/
/*
    //sequential version
    functor_cost_fnc tmp = functor_cost_fnc( matrix, matrix_size, partial_cost_functions, matrix_size );
    #pragma omp parallel for
    for (int idx=0; idx<matrix_size; idx++) {
        tmp(idx);
    }
*/
/*
    // calculate the final cost function
    double cost_function = 0;
    priv_partial_cost_functions.combine_each([&cost_function](double a) {
        cost_function = cost_function + a;
    });
*/

/*
#ifdef USE_AVX


    __m128d trace_128 = _mm_setr_pd(0.0, 0.0);
    double* matrix_data = (double*)matrix.get_data();
    int offset = 2*(matrix.stride+1);

    for (int idx=0; idx<matrix_size; idx++) {
        
        // get the diagonal element
        __m128d element_128 = _mm_load_pd(matrix_data);
        
        // add the diagonal elements to the trace
        trace_128 = _mm_add_pd(trace_128, element_128);

        matrix_data = matrix_data + offset;
    }


    trace_128 = _mm_mul_pd(trace_128, trace_128);    
    double cost_function = std::sqrt(1.0 - (trace_128[0] + trace_128[1])/(matrix_size*matrix_size));

#else

    QGD_Complex16 trace;
    memset( &trace, 0.0, 2*sizeof(double) );
    //trace.real = 0.0;
    //trace.imag = 0.0;

    for (int idx=0; idx<matrix_size; idx++) {
        
        trace.real += matrix[idx*matrix.stride + idx].real;
        trace.imag += matrix[idx*matrix.stride + idx].imag;
    }

    double cost_function = std::sqrt(1.0 - (trace.real*trace.real + trace.imag*trace.imag)/(matrix_size*matrix_size));
#endif
*/


    double trace_real = 0.0;

    if ( trace_offset == 0 ) {

        for (int idx=0; idx<matrix_size; idx++) {
         
            trace_real += matrix[idx*matrix.stride + idx].real;

        }
    }
    else {

        for (int idx=0; idx<matrix_size; idx++) {

            trace_real += matrix[(idx+trace_offset)*matrix.stride + idx].real;

        }

    }

    //double cost_function = std::sqrt(1.0 - trace_real/matrix_size);
    double cost_function = (1.0 - trace_real/matrix_size);

    return cost_function;

}



/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0) and the first correction (index 1).
*/
Matrix_real get_cost_function_with_correction(Matrix matrix, int qbit_num, int trace_offset) {

    Matrix_real ret(1,2);

    // calculate the cost function
    ret[0] = get_cost_function( matrix, trace_offset );



    // calculate teh first correction

    int matrix_size = matrix.cols;

    double trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = col_idx ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }

    }
    else {


        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = (col_idx + trace_offset) ^ qbit_error_mask;
// std::cout << matrix[row_idx*matrix.stride + col_idx].real << " " << row_idx << " " << col_idx << std::endl;
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }


    }

    //double cost_function = std::sqrt(1.0 - trace_real/matrix_size);
    double cost_function = trace_real/matrix_size;

//std::cout << cost_function << std::endl;
//exit(1);

    ret[1] = cost_function;

    return ret;

    

}




/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0), the first correction (index 1) and the second correction (index 2).
*/
Matrix_real get_cost_function_with_correction2(Matrix matrix, int qbit_num, int trace_offset) {


    Matrix_real ret(1,3);

    // calculate the cost function
    ret[0] = get_cost_function( matrix, trace_offset );



    // calculate the first correction

    int matrix_size = matrix.cols;

    double trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = col_idx ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }

    }
    else {


        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = (col_idx+trace_offset) ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }


    }

    double cost_function = trace_real/matrix_size;

    ret[1] = cost_function;




    // calculate the second correction

    trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num-1; qbit_idx++) {
            for (int qbit_idx2=qbit_idx+1; qbit_idx2<qbit_num; qbit_idx2++) {

                int qbit_error_mask = (1 << qbit_idx) + (1 << qbit_idx2);

                for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                    // determine the row index pair with one bit error at the given qbit_idx
                    int row_idx = col_idx ^ qbit_error_mask;
 
                    trace_real += matrix[row_idx*matrix.stride + col_idx].real;
                }

            }
        }
    }
    else {

        for (int qbit_idx=0; qbit_idx<qbit_num-1; qbit_idx++) {
            for (int qbit_idx2=qbit_idx+1; qbit_idx2<qbit_num; qbit_idx2++) {

                int qbit_error_mask = (1 << qbit_idx) + (1 << qbit_idx2);

                for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                    // determine the row index pair with one bit error at the given qbit_idx
                    int row_idx = (col_idx+trace_offset) ^ qbit_error_mask;
 
                    trace_real += matrix[row_idx*matrix.stride + col_idx].real;
                }

            }
        }


    }

    double cost_function2 = trace_real/matrix_size;

    ret[2] = cost_function2;

    return ret;





}


/**
@brief Constructor of the class.
@param matrix_in Arry containing the input matrix
@param matrix_size_in The number rows in the matrix.
@param partial_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param partial_cost_fnc_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_cost_fnc::functor_cost_fnc( Matrix matrix_in, tbb::combinable<double>* partial_cost_functions_in ) {

    matrix = matrix_in;
    data = matrix.get_data();
    partial_cost_functions = partial_cost_functions_in;
}

/**
@brief Operator to calculate the partial cost function derived from the row of the matrix labeled by row_idx
@param r A TBB range labeling the partial cost function to be calculated.
*/
void functor_cost_fnc::operator()( tbb::blocked_range<int> r ) const {

    int matrix_size = matrix.rows;
    double &cost_function_priv = partial_cost_functions->local();

    for ( int row_idx = r.begin(); row_idx != r.end(); row_idx++) {

        if ( row_idx > matrix_size ) {
            std::stringstream sstream;
      	    sstream << "Error: row idx should be less than the number of roes in the matrix" << std::endl;
            print(sstream, 0);   
            exit(-1);
        }

        // getting the corner element
        QGD_Complex16 corner_element = data[0];


        // Calculate the |x|^2 value of the elements of the matrix and summing them up to calculate the partial cost function
        double partial_cost_function = 0;
        int idx_offset = row_idx*matrix_size;
        int idx_max = idx_offset + row_idx;
        for ( int idx=idx_offset; idx<idx_max; idx++ ) {
            partial_cost_function = partial_cost_function + data[idx].real*data[idx].real + data[idx].imag*data[idx].imag;
        }

        int diag_element_idx = row_idx*matrix_size + row_idx;
        double diag_real = data[diag_element_idx].real - corner_element.real;
        double diag_imag = data[diag_element_idx].imag - corner_element.imag;
        partial_cost_function = partial_cost_function + diag_real*diag_real + diag_imag*diag_imag;


        idx_offset = idx_max + 1;
        idx_max = row_idx*matrix_size + matrix_size;
        for ( int idx=idx_offset; idx<idx_max; idx++ ) {
            partial_cost_function = partial_cost_function + data[idx].real*data[idx].real + data[idx].imag*data[idx].imag;
        }

        // storing the calculated partial cost function
        cost_function_priv = cost_function_priv + partial_cost_function;

    }
}








