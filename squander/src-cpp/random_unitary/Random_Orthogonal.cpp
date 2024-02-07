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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Random_Orthogonal.cpp
    \brief A class and methods to cerate random unitary matrices
*/


#include "Random_Orthogonal.h"
#include "logging.h"



/**
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be ceated.
@return An instance of the class
*/
Random_Orthogonal::Random_Orthogonal( int dim_in ) {

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
Matrix
Random_Orthogonal::Construct_Orthogonal_Matrix() {

    // create array of random parameters to construct random unitary
    Matrix_real vargamma(1, dim*(dim-1)/2 );
    

    // initialize random seed:
    srand (time(NULL));

    for (int idx=0; idx<dim*(dim-1)/2; idx++) {
        vargamma[idx] = (2*double(rand())/double(RAND_MAX)-1)*2*M_PI;
    }
    //vargamma[dim*(dim-1)/2-1] = 3.14159265359/2;

    Matrix Umtx = Construct_Orthogonal_Matrix( vargamma );


    return Umtx;

}


/**
@brief Generates a unitary matrix from parameters vartheta, varphi, varkappa according to arXiv:1303:5904v1
@param vartheta array of dim*(dim-1)/2 elements
@param varphi array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return Returns with a pointer to the generated unitary
*/
Matrix
Random_Orthogonal::Construct_Orthogonal_Matrix( Matrix_real &vargamma ) {

//The stringstream input to store the output messages.
std::stringstream sstream;

//Integer value to set the verbosity level of the output messages.
int verbose_level;

    if (vargamma.size() != dim*(dim-1)/2) {
	sstream << "Wring number of parameters in Random_Orthogonal::Construct_Orthogonal_Matrix" << std::endl;
	verbose_level=1;
        print(sstream, verbose_level);	
        exit(-1);
    }

    // construct vargamma matrix elements
    Matrix_real vargamma_mtx(dim, dim);
    memset( vargamma_mtx.get_data(), 0.0, vargamma_mtx.size()*sizeof(double) );
    int gamma_index = 0;
    for (int idx=0; idx<dim; idx++) {
        for (int jdx=idx+1; jdx<dim; jdx++) {
            vargamma_mtx[idx*vargamma_mtx.stride + jdx] = vargamma[gamma_index];
            gamma_index++;
        }
        //vargamma_mtx[ idx*dim + idx ] = 3.14159265358979323846/2;
    }
/*
//vargamma_mtx[0*vargamma_mtx.stride + dim-7] = 0.0;
//vargamma_mtx[0*vargamma_mtx.stride + dim-6] = 0.0;
//vargamma_mtx[0*vargamma_mtx.stride + dim-5] = 0.0;
vargamma_mtx[0*vargamma_mtx.stride + dim-4] = 0.0;
vargamma_mtx[0*vargamma_mtx.stride + dim-3] = 0.0;
vargamma_mtx[0*vargamma_mtx.stride + dim-2] = 0.0;
vargamma_mtx[0*vargamma_mtx.stride + dim-1] = 0.0;

//vargamma_mtx[1*vargamma_mtx.stride + dim-6] = 0.0;
//vargamma_mtx[1*vargamma_mtx.stride + dim-5] = 0.0;
vargamma_mtx[1*vargamma_mtx.stride + dim-4] = 0.0;
vargamma_mtx[1*vargamma_mtx.stride + dim-3] = 0.0;
vargamma_mtx[1*vargamma_mtx.stride + dim-2] = 0.0;
vargamma_mtx[1*vargamma_mtx.stride + dim-1] = 0.0;


//vargamma_mtx[2*vargamma_mtx.stride + dim-5] = 0.0;
vargamma_mtx[2*vargamma_mtx.stride + dim-4] = 0.0;
vargamma_mtx[2*vargamma_mtx.stride + dim-3] = 0.0;
vargamma_mtx[2*vargamma_mtx.stride + dim-2] = 0.0;
vargamma_mtx[2*vargamma_mtx.stride + dim-1] = 0.0;

vargamma_mtx[3*vargamma_mtx.stride + dim-4] = 0.0;
vargamma_mtx[3*vargamma_mtx.stride + dim-3] = 0.0;
vargamma_mtx[3*vargamma_mtx.stride + dim-2] = 0.0;
vargamma_mtx[3*vargamma_mtx.stride + dim-1] = 0.0;

//vargamma_mtx[4*vargamma_mtx.stride + dim-3] = 0.0;
//vargamma_mtx[4*vargamma_mtx.stride + dim-2] = 0.0;
//vargamma_mtx[4*vargamma_mtx.stride + dim-1] = 0.0;

//vargamma_mtx[5*vargamma_mtx.stride + dim-2] = 0.0;
//vargamma_mtx[5*vargamma_mtx.stride + dim-1] = 0.0;

//vargamma_mtx[6*vargamma_mtx.stride + dim-1] = 0.0;
*/

    Matrix_real T2(1,1);
    T2[0] = 1.0;
    
    // spawn iterations to construct dim x dim orthogonal matrix
    Matrix_real Tn = T2;
    for (int ndx=2; ndx<=dim; ndx++) {

        // preallocate matrix for the new T
        Matrix_real Tn_new(ndx, ndx);

        // construct matrix tn from Eq (6) of  https://doi.org/10.1002/qua.560040725
        Matrix_real tn(ndx, ndx);
        memset( tn.get_data(), 0.0, tn.size()*sizeof(double) );
        for ( int row_idx=0; row_idx<ndx-1; row_idx++) {
            memcpy( tn.get_data()+row_idx*tn.stride, Tn.get_data() + row_idx*Tn.stride, (ndx-1)*sizeof(double) );
        }
        tn[ndx*tn.stride -1] = 1.0;

        // construct matrix Tn from Eq (14) in  https://doi.org/10.1002/qua.560040725
        for ( int col_idx=0; col_idx<ndx; col_idx++) {

            // allocate a column in matrix s defined by Eq (15)
            Matrix_real sl(ndx, 1);

            // k = 0 case of Eq (16)
            sl[0] = -tn[col_idx*tn.stride + ndx-1];  // Eq (16)

            // k = 0 case in Eq (14)
            Tn_new[col_idx] = tn[col_idx]*cos(vargamma_mtx[ndx-1]) - sl[0]*sin(vargamma_mtx[ndx-1]);

            // k > 0 case in Eq (14), (15)
            for ( int row_idx=1; row_idx<ndx; row_idx++) {
 
                int kdx = row_idx-1;
                sl[row_idx] = tn[kdx*tn.stride+col_idx] * sin(vargamma_mtx[kdx*dim+ndx-1]) + sl[kdx] * cos(vargamma_mtx[kdx*dim+ndx-1]);

                if ( row_idx == ndx-1 ) {
                    Tn_new[row_idx*Tn_new.stride + col_idx] = - sl[row_idx];
                }
                else {
                    Tn_new[row_idx*Tn_new.stride + col_idx] = tn[row_idx*tn.stride + col_idx] * cos(vargamma_mtx[row_idx*dim+ndx-1]) - sl[row_idx] * sin(vargamma_mtx[row_idx*dim+ndx-1]);
                }
            
            }
/*
std::cout << "sl" << std::endl;
sl.print_matrix();
*/
        }

        Tn = Tn_new;

    }

//Tn.print_matrix();
    Matrix ret(Tn.rows, Tn.cols);
    for ( int idx=0; idx<ret.size(); idx++) {
        ret[idx].real = Tn[idx];
        ret[idx].imag = 0.0;
    }

    return ret;



}

