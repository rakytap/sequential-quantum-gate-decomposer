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

//
// @brief A base class responsible for constructing matrices of C-NOT gates
// gates acting on the N-qubit space

#include "qgd/Random_Unitary.h"


//// Contructor of the class
//> @brief Constructor of the class.
//> @param ??????????????????
//> @return An instance of the class
Random_Unitary::Random_Unitary( int dim_in ) {   
        
        if (dim_in < 2) {
            throw "wrong dimension";
        }
        
        // number of qubits
        dim = dim_in;  

}
        
 
//// 
//> @brief Create random unitary
//> @param parameters array of (dim+1)*(dim-1) elements
//> @return The constructed matrix
MKL_Complex16* Random_Unitary::Construct_Unitary_Matrix() {

    // create array of random parameters to construct random unitary
    double* vartheta = (double*) mkl_malloc( int(dim*(dim-1)/2)*sizeof(double), 64);
    double* varphi = (double*) mkl_malloc( int(dim*(dim-1)/2)*sizeof(double), 64);
    double* varkappa = (double*) mkl_malloc( (dim-1)*sizeof(double), 64);

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

    MKL_Complex16* Umtx = Construct_Unitary_Matrix( vartheta, varphi, varkappa );

    mkl_free(vartheta);
    mkl_free(varphi);
    mkl_free(varkappa);


    return Umtx;

}    
    
    
//// 
//> @brief Create random unitary
//> @param vartheta array of dim*(dim-1)/2 elements
//> @param varphi array of dim*(dim-1)/2 elements
//> @param varkappa array of dim-1 elements
//> @return The constructed matrix
MKL_Complex16* Random_Unitary::Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa ) {
        
        
        MKL_Complex16* ret = create_identity(dim);        
        for (int varalpha=1; varalpha<dim; varalpha++) { // = 2:obj.dim
           for (int varbeta = 0; varbeta<varalpha; varbeta++) {//   1:varalpha-1
           
               double theta_loc = vartheta[ convert_indexes(varalpha, varbeta) ];
               double phi_loc = varphi[ convert_indexes(varalpha, varbeta) ];

               
               // Eq (26)
               MKL_Complex16 a;
               a.real = cos( theta_loc )*cos(phi_loc);
               a.imag = cos( theta_loc )*sin(phi_loc);
              
               // Eq (28) and (26)
               double varepsilon = varkappa[varalpha-1]*kronecker( varalpha-1, varbeta);  
               MKL_Complex16 b;             
               b.real = sin( theta_loc )*cos(varepsilon);
               b.imag = sin( theta_loc )*sin(varepsilon);

               a.real = -a.real;
               b.imag = -b.imag;
               MKL_Complex16* Omega_loc = Omega( varalpha, varbeta, a, b );

 
               MKL_Complex16* ret_tmp = zgemm3m_wrapper( ret, Omega_loc, dim); //   ret * Omega_loc
               mkl_free(ret);
               mkl_free( Omega_loc);

               ret = ret_tmp;
           }            
        }
        

        MKL_Complex16 gamma_loc;
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
    
    
////
//> @brief Create random unitary
//> @param parameters array of (dim+1)*(dim-1) elements
//> @return The constructed matrix
MKL_Complex16* Random_Unitary::Construct_Unitary_Matrix(double* parameters ) {
   MKL_Complex16* ret = Construct_Unitary_Matrix( parameters, parameters+int(dim*(dim-1)/2), parameters+int(dim*(dim-1)));
        
    return ret;
}
    
    
//// Omega
//> @brief Eq (6)
MKL_Complex16* Random_Unitary::Omega(int varalpha, int varbeta, MKL_Complex16 x, MKL_Complex16 y )   {
        
        MKL_Complex16* ret = I_alpha_beta( varalpha, varbeta );
        

        MKL_Complex16 alpha;
        MKL_Complex16 beta;
        alpha.real = 1;
        alpha.imag = 0;
        beta.real = 1;
        beta.imag = 0;
        
        MKL_Complex16* Mloc;
        
        if (varalpha + varbeta != (3 + kronecker( dim, 2 )) ) {
            Mloc = M( varalpha, varbeta, x, y );
            
        }
        else {
            y.imag = -y.imag;
            Mloc = M( varalpha, varbeta, x, mult(gamma(), y) );
        }


        #pragma omp parallel for
        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx].real = ret[idx].real + Mloc[idx].real;
            ret[idx].imag = ret[idx].imag + Mloc[idx].imag;
        }

        mkl_free( Mloc );

        return ret;
 
        
}   
    
    
//// M
//> @brief Eq (8)
MKL_Complex16* Random_Unitary::M( int varalpha, int varbeta, MKL_Complex16 s, MKL_Complex16 t )   {

        MKL_Complex16* Qloc = Q( s, t);

        MKL_Complex16* ret1 = E_alpha_beta( varbeta, varbeta );
        MKL_Complex16* ret2 = E_alpha_beta( varbeta, varalpha );
        MKL_Complex16* ret3 = E_alpha_beta( varalpha, varbeta );
        MKL_Complex16* ret4 = E_alpha_beta( varalpha, varalpha );


        MKL_Complex16 alpha;
        MKL_Complex16 beta;
        alpha.real = 1;
        alpha.imag = 0;
        beta.real = 1;
        beta.imag = 0;

        mult(Qloc[0], ret1, dim);
        mult(Qloc[1], ret2, dim);
        mult(Qloc[2], ret3, dim);
        mult(Qloc[3], ret4, dim);
        mkl_free(Qloc);


        MKL_Complex16* ret = (MKL_Complex16*)mkl_malloc(dim*dim*sizeof(MKL_Complex16), 64);
        #pragma omp parallel for
        for ( int idx=0; idx<dim*dim; idx++ ) {
            ret[idx].real = ret1[idx].real + ret2[idx].real + ret3[idx].real + ret4[idx].real;
            ret[idx].imag = ret1[idx].imag + ret2[idx].imag + ret3[idx].imag + ret4[idx].imag;
        }

        mkl_free(ret1);
        mkl_free(ret2);
        mkl_free(ret3);
        mkl_free(ret4);
      

        return ret;
        
}
    
    
//// Q
//> @brief Eq (9)
MKL_Complex16* Random_Unitary::Q(  MKL_Complex16 u1, MKL_Complex16 u2 )   {

    MKL_Complex16* ret = (MKL_Complex16*)mkl_calloc(2*2, sizeof(MKL_Complex16), 64);    
    ret[0] = u2;       
    ret[1] = u1;
    ret[2].real = -u1.real;
    ret[2].imag = u1.imag;
    ret[3].real = u2.real;
    ret[3].imag = -u2.imag;

    return ret;
        
}
    
    
//// E_n_m
//> @brief below Eq (7)
MKL_Complex16* Random_Unitary::E_alpha_beta( int varalpha, int varbeta )   {
        
    MKL_Complex16* ret = (MKL_Complex16*)mkl_calloc(dim*dim, sizeof(MKL_Complex16), 64);
    ret[varalpha*dim+varbeta].real = 1;

    return ret;
        
}
    
//// I_n
//> @brief below Eq (7)
MKL_Complex16* Random_Unitary::I_alpha_beta( int varalpha, int varbeta ) {

 
   MKL_Complex16* ret = create_identity(dim);

   ret[varalpha*dim+varalpha].real = 0;
   ret[varbeta*dim+varbeta].real = 0;

   return ret;
        
}       
    
    
//// 
//> @brief Eq (11)
double Random_Unitary::gamma() {
        
    double ret = pow(-1, 0.25*(2*dim-1+pow(-1,dim)));//(-1)^(0.25*(2*dim-1+(-1)^dim));
    return ret;
        
}
    
////
//> @brief Eq (11)
double Random_Unitary::kronecker( int a, int b ) {
        
        if (a == b) {
            return 1;
        }
        else {
            return 0;
        }

}


    



