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

# ifndef __GRAD_DESCEND__H
# define __GRAD_DESCEND__H

#include "matrix_real.h"
#include <vector>

/// status indicators of the solver
enum solver_status{INITIAL_STATE=0, VARIABLES_INITIALIZED=1, ZERO_STEP_SIZE_OCCURED=2, MAXIMAL_ITERATIONS_REACHED=4, NO_DECREASING_SEARCH_DIRECTION=5, MINIMUM_REACHED=6, MAX_ITERATIONS_REACHED_DURING_LINE_SEARCH=7};


/**
 * @brief A class implementing the BFGS iterations on the 
 */
class Grad_Descend {


protected:


    /// number of independent variables in the problem
    int variable_num;

    /// maximal count of iterations during the optimization
    long maximal_iterations;

    /// number of function calls during the optimization process
    long function_call_count;

    /// numerical precision used in the calculations
    double num_precision;

    /// function pointer to evaluate the cost function and its gradient vector
    void (*costfnc__and__gradient) (Matrix_real x, void * params, double * f, Matrix_real& g);
    
    /// function pointer to evaluate the cost function and its gradient vector
    void (*export_fnc) (double , Matrix_real&, void* );    
     
    /// additional data needed to evaluate the cost function
    void* meta_data;

    /// status of the solver
    enum solver_status status;

protected:

/**
@brief Call to perform inexact line search terminated with Wolfe 1st and 2nd conditions
@param x The guess for the starting point. The coordinated of the optimized cost function are returned via x.
@param g The gradient at x. The updated gradient is returned via this argument.
@param search_direction The search direction.
@param x0_search Stores the starting point. (automatically updated with the starting x during the execution)
@param g0_search Stores the starting gradient. (automatically updated with the starting x during the execution)
@param maximal_step The maximal allowed step in the search direction
@param d__dot__g The overlap of the gradient with the search direction to test downhill.
@param f The value of the minimized cost function is returned via this argument
*/  
void line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double& maximal_step, double& d__dot__g, double& f);

/**
@brief Call this method to start the optimization process
@param x The guess for the starting point. The coordinated of the optimized cost function are returned via x.
@param f The value of the minimized cost function
*/
virtual void Optimize(Matrix_real& x, double& f);


/**
@brief Call this method to obtain the maximal step size during the line search. providing at least 2*PI periodicity, unless the search direction component in a specific direction is to small.
@param search_direction The search direction.
@param maximal_step The maximal allowed step in the search direction returned via this argument.
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
void get_Maximal_Line_Search_Step(Matrix_real& search_direction, double& maximal_step, double& search_direction__grad_overlap);


/**
@brief Method to get the search direction in the next line search
@param g The gradient at the current coordinates.
@param search_direction The search direction is returned via this argument (it is -g)
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
virtual void get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap);


/**
@brief Call this method to obtain the cost function and its gradient at a gives position given by x.
@param x The array of the current coordinates
@param f The value of the cost function is returned via this argument.
@param g The gradient of  the cost function at x.
*/
void get_f_ang_gradient(Matrix_real& x, double& f, Matrix_real& g);    


public:

/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in);


/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void (* export_pointer)(double , Matrix_real&, void* ), void* meta_data_in);


/**
@brief Call this method to start the optimization.
@param x The initial solution guess.
@param maximal_iterations_in The maximal number of function+gradient evaluations. Reaching this threshold the solver returns with the current solution.
*/
double Start_Optimization(Matrix_real &x, long maximal_iterations_in = 5001);

/**
@brief Destructor of the class
*/
~Grad_Descend();



};


# endif
