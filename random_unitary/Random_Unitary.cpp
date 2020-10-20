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
/*! \file Random_Unitary.cpp
    \brief A class and methods to cerate random unitary matrices
*/


#include "qgd/Random_Unitary.h"


/**
@brief Call to create a random unitary constructed by CNOT operation between randomly chosen qubits and by random U3 operations.
@param qbit_num The number of qubits spanning the unitary.
@param cnot_num The number of CNOT gates composing the random unitary.
@param mtx The preallocated array for the constructed unitary.
*/
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
                mtx_tmp = NULL;
            }

            if (gate_matrix != NULL) {
                qgd_free( gate_matrix );
                gate_matrix = NULL;
            }

            return;
        }

    }

}


/** 
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be ceated.
@return An instance of the class
*/
Random_Unitary::Random_Unitary( int dim_in ) {   
        
        if (dim_in < 2) {
            throw "wrong dimension";
        }
        
        // number of qubits
        dim = dim_in;  

}
        
 
/**
@brief Call to create a random unitary
@return Returns with a pointer to the created random unitary
*/
QGD_Complex16* Random_Unitary::Construct_Unitary_Matrix() {

    // create array of random parameters to construct random unitary
    double* vartheta = (double*) qgd_calloc( int(dim*(dim-1)/2),sizeof(double), 64);
    double* varphi = (double*) qgd_calloc( int(dim*(dim-1)/2),sizeof(double), 64);
    double* varkappa = (double*) qgd_calloc( (dim-1),sizeof(double), 64);

    // initialize random seed:
    srand (time(NULL));

    for (int idx=0; idx<dim*(dim-1)/2; idx++) {
        vartheta[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }


    for (int idx=0; idx<dim*(dim-1)/2; idx++) {
        varphi[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }


    for (int idx=0; idx<(dim-1); idx++) {
        varkappa[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }

    QGD_Complex16* Umtx = Construct_Unitary_Matrix( vartheta, varphi, varkappa );

    qgd_free(vartheta);
    qgd_free(varphi);
    qgd_free(varkappa);
    vartheta = NULL;
    varphi = NULL;
    varkappa = NULL;


    return Umtx;

}    
    
    
/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
QGD_Complex16* Random_Unitary::Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa ) {
        
        
        QGD_Complex16* ret = create_identity(dim);        
        for (int varalpha=1; varalpha<dim; varalpha++) { // = 2:obj.dim
           for (int varbeta = 0; varbeta<varalpha; varbeta++) {//   1:varalpha-1
           
               double theta_loc = vartheta[ convert_indexes(varalpha, varbeta) ];
               double phi_loc = varphi[ convert_indexes(varalpha, varbeta) ];

               
               // Eq (26)
               QGD_Complex16 a;
               a.real = cos( theta_loc )*cos(phi_loc);
               a.imag = cos( theta_loc )*sin(phi_loc);
              
               // Eq (28) and (26)
               double varepsilon = varkappa[varalpha-1]*kronecker( varalpha-1, varbeta);  
               QGD_Complex16 b;             
               b.real = sin( theta_loc )*cos(varepsilon);
               b.imag = sin( theta_loc )*sin(varepsilon);

               a.real = -a.real;
               b.imag = -b.imag;
               QGD_Complex16* Omega_loc = Omega( varalpha, varbeta, a, b );

 
               QGD_Complex16* ret_tmp = zgemm3m_wrapper( ret, Omega_loc, dim); //   ret * Omega_loc
               qgd_free(ret);
               qgd_free( Omega_loc);
               ret = NULL;
               Omega_loc = NULL;

               ret = ret_tmp;
           }            
        }
        

        QGD_Complex16 gamma_loc;
        gamma_loc.real = gamma();
        gamma_loc.imag = 0;
        
        for ( int idx=0; idx<dim*dim; idx++) {
            ret[idx] = mult(ret[idx], gamma_loc);
        }

        return ret;    


        
}



int  Random_Unitary::convert_indexes( int varalpha, int varbeta ) {
     int ret = varbeta + (varalpha-1)*(varalpha-2)/2;
     return ret; 
}
    
    
/**
@brief Generates a unitary matrix from parameters parameters according to arXiv:1303:5904v1
@param parameters array of (dim+1)*(dim-1) elements
@return The constructed unitary
*/
QGD_Complex16* Random_Unitary::Construct_Unitary_Matrix(double* parameters ) {
   QGD_Complex16* ret = Construct_Unitary_Matrix( parameters, parameters+int(dim*(dim-1)/2), parameters+int(dim*(dim-1)));
        
    return ret;
}
    
    
/**
@brief Eq (6) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param x A complex number
@param y A complex number
@return Return with a pointer to the calculated Omega matrix of Eq. (6) of arXiv:1303:5904v1
*/
QGD_Complex16* Random_Unitary::Omega(int varalpha, int varbeta, QGD_Complex16 x, QGD_Complex16 y )   {
        
        QGD_Complex16* ret = I_alpha_beta( varalpha, varbeta );
        
        
        QGD_Complex16* Mloc;
        
        if (varalpha + varbeta != (3 + kronecker( dim, 2 )) ) {
            Mloc = M( varalpha, varbeta, x, y );
            
        }
        else {
            y.imag = -y.imag;
            Mloc = M( varalpha, varbeta, x, mult(gamma(), y) );
        }


        //#pragma omp parallel for
        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx].real = ret[idx].real + Mloc[idx].real;
            ret[idx].imag = ret[idx].imag + Mloc[idx].imag;
        }

        qgd_free( Mloc );
        Mloc = NULL;

        return ret;
 
        
}   
    
    
/**
@brief Implements Eq (8) of arXiv:1303:5904v1
@param varalpha An integer
@param varbeta An integer
@param s A complex number
@param t A complex number
@return Return with a pointer to the calculated M matrix of Eq. (8) of arXiv:1303:5904v1
*/
QGD_Complex16* Random_Unitary::M( int varalpha, int varbeta, QGD_Complex16 s, QGD_Complex16 t )   {

        QGD_Complex16* Qloc = Q( s, t);

        QGD_Complex16* ret1 = E_alpha_beta( varbeta, varbeta );
        QGD_Complex16* ret2 = E_alpha_beta( varbeta, varalpha );
        QGD_Complex16* ret3 = E_alpha_beta( varalpha, varbeta );
        QGD_Complex16* ret4 = E_alpha_beta( varalpha, varalpha );


        mult(Qloc[0], ret1, dim);
        mult(Qloc[1], ret2, dim);
        mult(Qloc[2], ret3, dim);
        mult(Qloc[3], ret4, dim);
        qgd_free(Qloc);
        Qloc = NULL;


        QGD_Complex16* ret = (QGD_Complex16*)qgd_calloc(dim*dim,sizeof(QGD_Complex16), 64);
        //#pragma omp parallel for
        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx].real = ret1[idx].real + ret2[idx].real + ret3[idx].real + ret4[idx].real;
            ret[idx].imag = ret1[idx].imag + ret2[idx].imag + ret3[idx].imag + ret4[idx].imag;
        }

        qgd_free(ret1);
        qgd_free(ret2);
        qgd_free(ret3);
        qgd_free(ret4);
        ret1 = NULL;
        ret2 = NULL;
        ret3 = NULL;
        ret4 = NULL;
      

        return ret;
        
}
    
    
/**
@brief Implements Eq (9) of arXiv:1303:5904v1 
@param u1 A complex number
@param u2 A complex number
@return Return with a pointer to the calculated Q matrix of Eq. (9) of arXiv:1303:5904v1
*/
QGD_Complex16* Random_Unitary::Q(  QGD_Complex16 u1, QGD_Complex16 u2 )   {

    QGD_Complex16* ret = (QGD_Complex16*)qgd_calloc(2*2, sizeof(QGD_Complex16), 64);    
    ret[0] = u2;       
    ret[1] = u1;
    ret[2].real = -u1.real;
    ret[2].imag = u1.imag;
    ret[3].real = u2.real;
    ret[3].imag = -u2.imag;

    return ret;
        
}
    
    
/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1 
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated E matrix of Eq. (7) of arXiv:1303:5904v1
*/
QGD_Complex16* Random_Unitary::E_alpha_beta( int varalpha, int varbeta )   {
        
    QGD_Complex16* ret = (QGD_Complex16*)qgd_calloc(dim*dim, sizeof(QGD_Complex16), 64);
    ret[varalpha*dim+varbeta].real = 1;

    return ret;
        
}
    
/**
@brief Implements matrix I below Eq (7) of arXiv:1303:5904v1 
@param varalpha An integer
@param varbeta An integer
@return Return with a pointer to the calculated I matrix of Eq. (7) of arXiv:1303:5904v1
*/
QGD_Complex16* Random_Unitary::I_alpha_beta( int varalpha, int varbeta ) {

 
   QGD_Complex16* ret = create_identity(dim);

   ret[varalpha*dim+varalpha].real = 0;
   ret[varbeta*dim+varbeta].real = 0;

   return ret;
        
}       
    
    
/**
@brief Implements Eq (11) of arXiv:1303:5904v1 
@return Returns eith the value of gamma
*/
double Random_Unitary::gamma() {
        
    double ret = pow(-1, 0.25*(2*dim-1+pow(-1,dim)));//(-1)^(0.25*(2*dim-1+(-1)^dim));
    return ret;
        
}
    
/**
@brief Kronecker delta
@param a An integer
@param b An integer
@return Returns with the Kronecker delta value of a and b.
*/
double Random_Unitary::kronecker( int a, int b ) {
        
        if (a == b) {
            return 1;
        }
        else {
            return 0;
        }

}


    



