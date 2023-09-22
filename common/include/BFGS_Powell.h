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

# ifndef __BFGS_POWELL__H
# define __BFGS_POWELL__H

#include "grad_descend.h"

/**
@brief A class implementing the BFGS optimizer based on conjugate gradient direction method of M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
*/
class BFGS_Powell : public Grad_Descend 
{
protected:


    // Z @ Z^T = B^-1 approximated Hesse matrix (second derivative of the cost function) (B is the inverse of the Hesse matrix) Eq (1.8) in 
    // M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
    // B is not stored, Z is updated during the BFGS formula
    Matrix_real Zmtx;   

    // Z.T @ g to determine the search direction 
    Matrix_real Z_T__dot__g;  


protected:

/**
@brief Initialize the matrix Z to identity. See the definition of matrix Z at Eq (1.8) in M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
*/
    void Initialize_Zmtx( );


/**
@brief Call this method to start the optimization process
@param x The guess for the starting point. The coordinated of the optimized cost function are returned via x.
@param f The value of the minimized cost function
*/
    void Optimize(Matrix_real& x, double& f);


/**
@brief Method implementing the BFGS update of the matrix Z. See M.J.D. POWELL: UPDATING CONJUGATE DIRECTIONS BY THE BFGS FORMULA, Mathematical Programming 38 (1987) 29-46 for details.
@param x The current coordinates.
@param g The gradient at the current coordinates.
@param x0_search The starting point used at the previous line search
@param g0_search The gradient at x0_search
@param norm_Z The column-wise norm of the matrix Z is updated (and returned) via this argument.
*/
    void BFGS_iteration(Matrix_real& x, Matrix_real& g, Matrix_real& x0_search, Matrix_real& g0_search, double& norm_Z);
    
/**
@brief Method to get the search direction in the next line search
@param g The gradient at the current coordinates.
@param search_direction The search direction is returned via this argument (calculated with g and with Zmtx)
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
    void get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap );




public:

/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
    BFGS_Powell(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in);


/**
@brief Destructor of the class
*/
    ~BFGS_Powell();
};


# endif
