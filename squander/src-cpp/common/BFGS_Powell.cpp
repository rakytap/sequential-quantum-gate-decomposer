/*
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

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <BFGS_Powell.h>



/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
BFGS_Powell::BFGS_Powell(void (* f_pointer) (Matrix_real , void *, double *, Matrix_real&), void* meta_data_in) : Grad_Descend(f_pointer, meta_data_in) {

}



/**
@brief Initialize the matrix Z to identity. See the definition of matrix Z at Eq (1.8) in M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
*/
void  BFGS_Powell::Initialize_Zmtx()
{
   

    // initialize array Z
    memset( Zmtx.get_data(), 0.0, variable_num*variable_num*sizeof(double) );

    
    // initialite Z to identiy matrix
    long row_offset = 0;     
    for (long idx = 0; idx < variable_num; idx++) {
    	    
        Zmtx[row_offset + idx] = 1.0;
        row_offset             = row_offset + variable_num;        
        
    }
    
    status = VARIABLES_INITIALIZED;

    return;
}



/**
@brief Call this method to start the optimization process
@param x The guess for the starting point. The coordinated of the optimized cost function are returned via x.
@param f The value of the minimized cost function
*/
void BFGS_Powell::Optimize(Matrix_real& x, double& f)
{


    // the initial gradient during the line search
    Matrix_real g0_search( variable_num, 1 ); 

    // the initial point during the line search
    Matrix_real x0_search( variable_num, 1 ); 

    // The calculated graient of the cost function at position x
    Matrix_real g( variable_num, 1 );       

    // the current search direction
    Matrix_real search_direction( variable_num, 1 );  


    // Z @ Z^T = B^-1 approximated Hesse matrix (second derivate of the cost function) (B is the inverse of the Hesse matrix) Eq (1.8) in 
    // M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
    // B is not stored, Z is updated during the BFGS formula
    Zmtx = Matrix_real( variable_num * (variable_num), 1 );   


    // Initialize Z
    Initialize_Zmtx();


    // Z.T @ g as an intermediate result for search direction update and the BFGS iteration 
    Z_T__dot__g = Matrix_real( variable_num, 1 ); 

    status = VARIABLES_INITIALIZED; 


    // The norm of the matrix Z (initialized to an imposiible value)
    double norm_Z = -1.0;
     

    // inner product of the search direction and the gradient
    double d__dot__g;

    double maximal_step; // The upper bound of the allowed step in the search direction
    
    if (function_call_count == 0 || status == VARIABLES_INITIALIZED) {
        get_f_ang_gradient(x, f, g);
    }
    
    double fprev = fabs(f + f + 1.0);

    // Calculate the next search direction. 


    while ( true ) {

        maximal_step = 0.0;
    
        double search_direction__grad_overlap = 0.0;
        get_search_direction(g, search_direction, search_direction__grad_overlap);


        // Calculate the bound on the step-length

        if (search_direction__grad_overlap < 0.0) {
            get_Maximal_Line_Search_Step(search_direction, maximal_step, search_direction__grad_overlap);
        }  
        d__dot__g = search_direction__grad_overlap;




        double gradient_norm = 0.0;
        for (long idx = 0; idx < variable_num; idx++) {
            gradient_norm += g[idx] * g[idx];
        }

        double acc=1e-19;
        // Test for convergence by measuring the magnitude of the gradient.
        if (gradient_norm <= acc * acc) {
            status = MINIMUM_REACHED;
            return;
        }
    
        // in case the cost fuction does not show decrement in the direction of the line search
        if (d__dot__g >= 0.0) {
            status = NO_DECREASING_SEARCH_DIRECTION;
            return;
        }


        // terminate cycles if the cost function is not decreased any more
        if (f >= fprev) {
            status = MINIMUM_REACHED;
            return;
        }

        fprev = f;


        // terminate if maximal number of iteration reached
        if (function_call_count >= maximal_iterations) {
            status = MAXIMAL_ITERATIONS_REACHED;
            return;
        }

        
        // perform line search in the direction search_direction
        line_search(x, g, search_direction, x0_search, g0_search, maximal_step, d__dot__g, f);
    
        if (status == ZERO_STEP_SIZE_OCCURED) {
            return;
        }


        // update the second derivative approximation. 
        BFGS_iteration(x, g, x0_search, g0_search, norm_Z);



    }

    


}

/**
@brief Method implementing the BFGS update of the matrix Z. See M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46 for details.
@param x The current coordinates.
@param g The current coordinates gradient.
@param x0_search The starting point used at the previous line search
@param g0_search The gradient at x0_search
@param d__dot__g The overlap of the search direction and the gradient calculated in get_search_direction
*/
void BFGS_Powell::BFGS_iteration(Matrix_real& x, Matrix_real& g, Matrix_real& x0, Matrix_real& g0, double& norm_Z)
{

    // Test if there is sufficient convexity for the update.

    double norm_delta = 0.0;
    double delta_T__dot__gamma = 0.0; //in Eq (2.6) in M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46



    double tmp = 0.0;

    Matrix_real delta( x0.size(), 1 );
    Matrix_real gamma( g0.size(), 1 );

    for (long idx = 0; idx < variable_num; idx++) {
	delta[idx] = x[idx] - x0[idx];

	norm_delta += delta[idx] * delta[idx];
	tmp        += g0[idx] * delta[idx];

	gamma[idx] = g[idx] - g0[idx];

	delta_T__dot__gamma     += gamma[idx] * delta[idx];
    }


    if (delta_T__dot__gamma < fabs(tmp) * 0.1) {
	return;
    }

    // delta_T__dot__gamma should not be zero or negative. Might happen on the first iteration when delta is zero
    delta_T__dot__gamma = std::max( num_precision, delta_T__dot__gamma ); 



    /*
    Transform the Z matrix with Givens rotations. Eq (1.7)  M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46
    this transformation provides a convenient way to satisfy active constraints.
    also, the freedom to postmultiply Z by an orthogonal matrix is sometimes important to maintain good accuracy in the presence of computer rounding errors.
    For example if B (Hessian of the cos function) has (n-1) eigenvalues of magnitude one and one tiny eigenvalue whose
    eigenvector is e = ( 1 1 ... 1), then inv(B) is dominated by a large multiple of the
    n x n matrix whose elements are all one. Therefore, if just the elements of inv(B) are
    stored, then rounding errors prevent the accurate determination of B from inv(B).
    However, using this transformation all the large elements of Z can be confined to a single column, which can make the determination
    of B from Z well-conditioned.
    This algorithm allows large differences in the magnitudes of the column norms { ||z_i|| [i= 1, 2 . . . . , n}.
    */ 
    for (long col_idx = variable_num-1; col_idx > 0; col_idx--) {

    	if (Z_T__dot__g[col_idx] == 0.0) {
	    continue;
	}

        long col_idx_tmp = col_idx - 1;

        double abs = sqrt( Z_T__dot__g[col_idx_tmp]*Z_T__dot__g[col_idx_tmp]   +    Z_T__dot__g[col_idx]*Z_T__dot__g[col_idx] );

	double cos_val = Z_T__dot__g[col_idx_tmp] / abs;
	double sin_val = Z_T__dot__g[col_idx] / abs;

	Z_T__dot__g[col_idx_tmp] = abs;

	long row_offset = col_idx_tmp;

	for (long row_idx = 0; row_idx < variable_num; row_idx++) {

	    tmp                 = cos_val * Zmtx[row_offset + 1] - sin_val * Zmtx[row_offset];
	    Zmtx[row_offset]     = cos_val * Zmtx[row_offset] + sin_val * Zmtx[row_offset + 1];
	    Zmtx[row_offset + 1] = tmp;

	    row_offset += variable_num;
	}

    }



    // Update the value of the norm of the Z matrix encoded in norm_Z. 

    if (norm_Z < 0.0) {
        // the first column of Z is delta/sqrt(delta.T @ gamma)
	norm_Z = norm_delta / delta_T__dot__gamma;
    } else {
	tmp = sqrt(norm_Z * norm_delta / delta_T__dot__gamma);

	norm_Z = std::min(norm_Z, tmp);
	norm_Z = std::max(norm_Z, tmp*0.1);
    }

    // Complete the updating of Z. 

    tmp = sqrt(delta_T__dot__gamma);
    
    long row_offset = 0;

    // Eq (2.2) in M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46
    for (long idx = 0; idx < variable_num; idx++) {
        if (fabs(delta[idx]) > num_precision ) {
            Zmtx[row_offset] = delta[idx] / tmp;
        }
        else {
            Zmtx[row_offset] = 0;
        }


        row_offset += variable_num;

    }


    // calculate the dot product of gamma.T @ Z
    Matrix_real gamma_T__dot__Z(1, variable_num);
    memset( gamma_T__dot__Z.get_data(), 0.0, gamma_T__dot__Z.size()*sizeof(double) );

    row_offset = 0;
    for (long row_idx = 0; row_idx < variable_num; row_idx++) {

        double& gamma_element = gamma[row_idx];

        for (long col_idx = 1; col_idx < variable_num; col_idx++) {
            gamma_T__dot__Z[col_idx] += gamma_element *  Zmtx[row_offset + col_idx];
        }

            row_offset = row_offset + variable_num;


    }



    for (long col_idx = 1; col_idx < variable_num; col_idx++) {
            gamma_T__dot__Z[col_idx]   = gamma_T__dot__Z[col_idx]/delta_T__dot__gamma;
    }

    // Eq (2.2) in M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46 + column-wise Frobenius norm of Z
    Matrix_real norm_Z_tmp_mtx( 1, variable_num);
    memset( norm_Z_tmp_mtx.get_data(), 0.0, norm_Z_tmp_mtx.size()*sizeof(double) );

    row_offset = 0;
    for (long row_idx = 0; row_idx < variable_num; row_idx++) {

        double& delta_element = delta[row_idx];

        for (long col_idx = 1; col_idx < variable_num; col_idx++) {
            double& Z_element = Zmtx[row_offset + col_idx];

            Z_element = Z_element - gamma_T__dot__Z[col_idx] *  delta_element;

            norm_Z_tmp_mtx[col_idx] += Z_element * Z_element;

        }

        row_offset = row_offset + variable_num;


    }


    // renormalize the columns of the Z matrix     
    // When the column norm of the new Z is small, need to scale it up to preserve numerical accuracy
    // this scaling will broke the Z^T @ Z = inv(B) relation, but it will preserve the quadratic termination properties of the BFGS 
    // (see M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46)
    for (long col_idx = 1; col_idx < variable_num; col_idx++) {

        if (norm_Z_tmp_mtx[col_idx] < norm_Z && norm_Z_tmp_mtx[col_idx] > num_precision) {

            tmp = sqrt(norm_Z / norm_Z_tmp_mtx[col_idx]);

            row_offset = col_idx;
            for (long row_idx = 0; row_idx < variable_num; row_idx++) {

                Zmtx[row_offset] = tmp * Zmtx[row_offset];
                row_offset      += variable_num;
            }
        }

    }



    

    return;
}



/**
@brief Method to get the search direction in the next line search
@param g The gradient at the current coordinates.
@param search_direction The search direction is returned via this argument (calculated with g and with Zmtx)
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
void BFGS_Powell::get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap )
{


    // calculate Z.T @ g
    memset(Z_T__dot__g.get_data(), 0.0, Z_T__dot__g.size()*sizeof(double) );
    long row_offset = 0;
    for (long row_idx = 0; row_idx < variable_num; row_idx++) {

        double& g_element = g[row_idx];
 
        for (long col_idx = 0; col_idx < variable_num; col_idx++) {
            Z_T__dot__g[col_idx] += Zmtx[row_offset + col_idx] * g_element;

        }
        row_offset = row_offset + variable_num;
    }



    // calculate the search direction d by -Z @ Z_T__dot__g
    memset(search_direction.get_data(), 0.0, search_direction.size()*sizeof(double) );
    row_offset = 0;
    for (long row_idx = 0; row_idx < variable_num; row_idx++) {

	for (long col_idx = 0; col_idx < variable_num; col_idx++) {
	    search_direction[row_idx] = search_direction[row_idx] - Zmtx[row_offset + col_idx] * Z_T__dot__g[col_idx];
	}

        row_offset = row_offset + variable_num;

    }


    // test variable to see downhill or uphill
    search_direction__grad_overlap = 0.0;    

    for (long idx = 0; idx < variable_num; idx++) {
            search_direction__grad_overlap += search_direction[idx] * g[idx];
    }

    return;

} 

/**
@brief Destructor of the class
*/
BFGS_Powell::~BFGS_Powell()  {

}
