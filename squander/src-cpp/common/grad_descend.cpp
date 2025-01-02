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
#define _USE_MATH_DEFINES
#include <math.h>
#include <cfloat>
#include <grad_descend.h>


/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Grad_Descend::Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in) {

    maximal_iterations = 5001;
    status             = INITIAL_STATE;
    
    // numerical precision used in the calculations
    num_precision = 1.42e-14;    


    
    costfnc__and__gradient = f_pointer;
    export_fnc = NULL;
    meta_data = meta_data_in;

    // number of function calls during the optimization process
    function_call_count = 0;

}


/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Grad_Descend::Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void (* export_pointer)(double , Matrix_real&, void* ), void* meta_data_in) {

    maximal_iterations = 5001;
    status             = INITIAL_STATE;
    
    // numerical precision used in the calculations
    num_precision = 1.42e-14;    


    
    costfnc__and__gradient = f_pointer;
    export_fnc             = export_pointer;
    meta_data              = meta_data_in;

    // number of function calls during the optimization process
    function_call_count = 0;

}



/**
@brief Call this method to start the optimization.
@param x The initial solution guess.
@param maximal_iterations_in The maximal number of function+gradient evaluations. Reaching this threshold the solver returns with the current solution.
*/
double Grad_Descend::Start_Optimization(Matrix_real &x, long maximal_iterations_in)
{



    variable_num       = x.size();

    
    // set the maximal number of iterations
    maximal_iterations = maximal_iterations_in;
    
    status = INITIAL_STATE;
    
    

      
    // test for dimension
    if ( variable_num <= 0 ) {
        return DBL_MAX;   
    }


    //     Minimize the objective function
    double f;  
    Optimize(x, f);
    
    return f;

}




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
void  Grad_Descend::line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double& maximal_step, double& d__dot__g0, double& f)
{




    memcpy( x0_search.get_data(), x.get_data(), x.size()*sizeof(double) );
    memcpy( g0_search.get_data(), g.get_data(), g.size()*sizeof(double) );


    long max_loops = 50;//1 << 30;



    Matrix_real g_best = g.copy(); 


    double step = std::min(1.0, maximal_step);
    
    double f_lowest  = f;
    double f_highest = f;

    double step_highest = maximal_step;
    double step_lowest = 0.0;

    double step_best = 0.0;
    double f_best         = f;
    double d__dot__g_abs_best = fabs(d__dot__g0);

    double d__dot__g_lowest = d__dot__g0;
    double d__dot__g_highest = 0.0;

    
    for( long iter_idx=0; iter_idx<max_loops; iter_idx++) { 
    
        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }


        get_f_ang_gradient(x, f, g);

        // overlap between the search direction and the gradient 
        double d__dot__g_current = 0.0;
        for (long idx = 0; idx < variable_num; idx++) {
            d__dot__g_current += search_direction[idx] * g[idx];
        }

        // update best solution
        if (f < f_best || fabs(d__dot__g_current) < d__dot__g_abs_best) {

            step_best      = step;
            f_best         = f;
            d__dot__g_abs_best = fabs(d__dot__g_current);

            memcpy( g_best.get_data(), g.get_data(), g.size()*sizeof(double) );
	
        }

        // exit the loop if maximal function calls reached
        if (function_call_count == maximal_iterations) {
            break;
        }


        // modify the upper and lower bound of the step inetrval 
        if (f < f_lowest + step * 0.1 * d__dot__g0) {  // Armijo test (Wolfe 1st condition)

            if (d__dot__g_current >= d__dot__g0 * 0.7) { // Wolfe 2nd (curvature) condition to exit the iterations
                break;
            }

            step_lowest      = step;
            f_lowest         = f;
            d__dot__g_lowest = d__dot__g_current;
            
        
        }
        else {

            step_highest      = step;
            f_highest         = f;
            d__dot__g_highest = d__dot__g_current;


        }



    

        // Calculate the next step length

        if (iter_idx == 0 || step_lowest > 0.0) {

            // It is critical to use the inexact line search (ILS) to ensure the global convergence of the nonlinear conjugate gradient (CG) method

            // fit a parabola a*step^2 + b*step + c to the points (step_lowest, f_lowest), (step_highest, f_highest) with a derivate d__dot__g_highest at step=step_highest
            // then d__dot__g_lowest_fit is the derivate of the parabola at step=step_lowest
            // f_highest = a*step_highest^2 + b*step_highest + c
            // f_lowest = a*step_lowest^2 + b*step_lowest + c
            double d__dot__g_lowest_fit = (f_highest - f_lowest) / (step_highest - step_lowest)*2.0 - d__dot__g_highest;

            // move on the parabola
            double scale = 0.0;

            // the tip of the parabola is at step_lowest + scale * (step_highest - step_lowest)
            scale = -d__dot__g_lowest_fit*0.5 / (d__dot__g_highest - d__dot__g_lowest_fit);
            scale = std::max( 0.1, scale);

            step = step_lowest + scale * (step_highest - step_lowest);

        } 
        else {

            // In later BFGS iterations the minimum might be closer to the initial x0_search. 
            // Thus scaling down the step when step_lowest is still zero and the landscape near x0_serach need to be explored.
            step = step * 0.1;

            if (step < num_precision) {
                break;
            }

        }
    
    } // for loop



    // copy the best solution in place of the current solution
    if (step != step_best) {

        step = step_best;
        f = f_best;

        g = g_best;

        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }
    }

    if (step == 0.0 ) {
        status = ZERO_STEP_SIZE_OCCURED;
    }

    return;
} 




/**
@brief Call this method to start the optimization process
@param x The guess for the starting point. The coordinated of the optimized cost function are returned via x.
@param f The value of the minimized cost function
*/
void Grad_Descend::Optimize(Matrix_real& x, double& f)
{

    // the initial gradient during the line search
    Matrix_real g0_search( variable_num, 1 ); 

    // the initial point during the line search
    Matrix_real x0_search( variable_num, 1 ); 

    // The calculated graient of the cost function at position x
    Matrix_real g( variable_num, 1 );       

    // the current search direction
    Matrix_real search_direction( variable_num, 1 );  

    status = VARIABLES_INITIALIZED;

   

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
        
        if ( export_fnc ) {
            export_fnc( f, x, meta_data );
        }


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



    }

    


}






/**
@brief Call this method to obtain the maximal step size during the line search. Providing at least 2*PI periodicity, unless the search direction component in a specific direction is to small.
@param search_direction The search direction.
@param maximal_step The maximal allowed step in the search direction returned via this argument.
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
void Grad_Descend::get_Maximal_Line_Search_Step(Matrix_real& search_direction, double& maximal_step, double& search_direction__grad_overlap)
{

    // Set steps to constraint boundaries and find the least positive one.


    maximal_step = 0.0;
    
    // the optimization landscape is periodic in 2PI
    // the maximal step will be the 2PI step in the direction of the smallest component of the search direction
    for( long kdx = 0; kdx < variable_num; kdx++ ) {

        // skip the current direction if it is too small    
        if ( fabs(search_direction[kdx]) < 1e-5 ) {
            continue;
        } 

        double step_bound_tmp = std::abs(2*M_PI/search_direction[kdx]);
  
  

        if (maximal_step == 0.0 || step_bound_tmp < maximal_step) {

            maximal_step = step_bound_tmp;

        }
        
    }

    return;

}




/**
@brief Call this method to obtain the cost function and its gradient at a gives position given by x.
@param x The array of the current coordinates
@param f The value of the cost function is returned via this argument.
@param g The gradient of  the cost function at x.
*/
void Grad_Descend::get_f_ang_gradient(Matrix_real& x, double& f, Matrix_real& g)
{

    function_call_count++;
    costfnc__and__gradient(x, meta_data, &f, g);
    
    return;
}




/**
@brief Method to get the search direction in the next line search
@param g The gradient at the current coordinates.
@param search_direction The search direction is returned via this argument (it is -g)
@param search_direction__grad_overlap The overlap of the gradient with the search direction to test downhill.
*/
void Grad_Descend::get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap )
{


    // calculate the search direction d by d = -g
    memset(search_direction.get_data(), 0.0, search_direction.size()*sizeof(double) );

    for (long row_idx = 0; row_idx < variable_num; row_idx++) {
	search_direction[row_idx] = -g[row_idx];       
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
Grad_Descend::~Grad_Descend()  {

}
