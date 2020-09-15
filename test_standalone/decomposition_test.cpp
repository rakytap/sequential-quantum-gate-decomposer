/*
Created on Fri Jun 26 14:14:12 2020
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

#include <iostream>

#include <stdio.h>
#include <map>

#include "qgd/common.h"
#include "qgd/U3.h"
#include "qgd/CNOT.h"
#include "qgd/N_Qubit_Decomposition.h"
#include "qgd/Random_Unitary.h"

using namespace std;

////
// @brief Call to create a random unitary containing a given number of CNOT gates between randomly chosen qubits
// @param qbit_num The number of qubits spanning the unitary
// @param cnot_num The number of CNOT gates in the unitary
void few_CNOT_unitary( int qbit_num, int cnot_num, QGD_Complex16* mtx) {

    // the current number of CNOT gates
    int cnot_num_curr = 0;

    // the size of the matrix
    int matrix_size = Power_of_2(qbit_num);

    // The unitary discribing each qubits in their initial state
    memset( mtx, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );
    create_identity( mtx, matrix_size );

    QGD_Complex16* mtx_tmp = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);

    QGD_Complex16* gate_matrix = (QGD_Complex16*)qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
    memset( gate_matrix, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );

    // constructing the unitary
    while (true) {
        int cnot_or_u3 = rand() % 5 + 1;

        CNOT* cnot_op = NULL;
        U3* u3_op = NULL;

        if (cnot_or_u3 <= 4) {
            // creating random parameters for the U3 operation
            double parameters[3];

            parameters[0] = double(rand())/RAND_MAX*4*M_PI;
            parameters[1] = double(rand())/RAND_MAX*2*M_PI;
            parameters[2] = double(rand())/RAND_MAX*2*M_PI;
           

            // randomly choose the target qbit
            int target_qbit = rand() % qbit_num;

            // creating the U3 gate
            u3_op = new U3(qbit_num, target_qbit, true, true, true);

            // get the matrix of the operation
            u3_op->matrix(parameters, gate_matrix);
        }
        else if ( cnot_or_u3 == 5 ) {
            // randomly choose the target qbit
            int target_qbit = rand() % qbit_num;

            // randomly choose the control qbit
            int control_qbit = rand() % qbit_num;

            if (target_qbit == control_qbit) {
                create_identity( gate_matrix, matrix_size );
            }
            else {

                // creating the CNOT gate
                cnot_op = new CNOT(qbit_num, control_qbit, target_qbit);

                // get the matrix of the operation
                cnot_op->matrix(gate_matrix);

                cnot_num_curr = cnot_num_curr + 1;
            }
        }
        else {
            create_identity( gate_matrix, matrix_size );
        }


        // get the current unitary
        zgemm3m_wrapper(gate_matrix, mtx, mtx_tmp, matrix_size);
        memset( gate_matrix, 0, matrix_size*matrix_size*sizeof(QGD_Complex16) );
        memcpy( mtx, mtx_tmp, matrix_size*matrix_size*sizeof(QGD_Complex16) );


        delete u3_op;
        u3_op = NULL;

        delete cnot_op;
        cnot_op = NULL;




        // exit the loop if the maximal number of CNOT gates reched
        if (cnot_num_curr >= cnot_num) {
            
            if (mtx_tmp != NULL) {
                qgd_free( mtx_tmp );
            }

            if (gate_matrix != NULL) {
                qgd_free( gate_matrix );
            }

            return;
        }

    }

}

//
// @brief Decomposition of general two-qubit matrix into U3 and CNOT gates
int main() {
    
    printf("\n\n****************************************\n");
    printf("Test of N qubit decomposition\n");
    printf("****************************************\n\n\n");

#ifdef MIC
#pragma offload target(mic)
      {
#endif

    // creating random unitary
    int qbit_num = 4;

#ifdef MIC
      qbit_num = 7;
#endif

    // the size of the matrix
    int matrix_size = Power_of_2(qbit_num);
    //printf("The test matrix to be decomposed is:\n");
    //print_mtx( Umtx, matrix_size, matrix_size );

    //int cnot_num = 4;
    //QGD_Complex16* Umtx = qgd_calloc( matrix_size*matrix_size, sizeof(QGD_Complex16), 64);
    //QGD_Complex16* Umtx = few_CNOT_unitary( qbit_num, cnot_num, Umtx);





    Random_Unitary ru = Random_Unitary(matrix_size); 

    QGD_Complex16* Umtx = ru.Construct_Unitary_Matrix();
/*printf("resulting random matrix:\n");
print_mtx( Umtx, matrix_size,matrix_size);

// parameters alpha and beta for the cblas_zgemm3m function
    double alpha = 1;
    double beta = 0;

    // preallocate array for the result
    QGD_Complex16* C = (QGD_Complex16*)qgd_calloc(matrix_size*matrix_size*sizeof(QGD_Complex16), 64); 

    // calculate the product of A and B
    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasConjTrans, matrix_size, matrix_size, matrix_size, &alpha, Umtx, matrix_size, Umtx, matrix_size, &beta, C, matrix_size);    
print_mtx( C, matrix_size,matrix_size);
    qgd_free(C);*/

    
    // Creating the class to decompose the 2-qubit unitary

    std::map<int,int> num_of_layers;
    num_of_layers[2] = 3;
    num_of_layers[3] = 20;
    num_of_layers[4] = 60;

    std::map<int,int> identical_blocks;
    identical_blocks[2] = 1;
    identical_blocks[3] = 2;
    identical_blocks[4] = 1;
    identical_blocks[5] = 1;
    identical_blocks[6] = 1;
    identical_blocks[7] = 1;

    N_Qubit_Decomposition cDecomposition = N_Qubit_Decomposition( Umtx, qbit_num, num_of_layers, identical_blocks, false, ZEROS );

    cDecomposition.verbose = true;

//cDecomposition.set_iteration_loops( 4, 3 );
//cDecomposition.set_iteration_loops( 3, 3 );

    printf("Starting the decompsition\n");
    cDecomposition.start_decomposition(true, true);

    cDecomposition.list_operations(1);

    qgd_free( Umtx );

#ifdef MIC
      }
#endif


  return 0;  

};

