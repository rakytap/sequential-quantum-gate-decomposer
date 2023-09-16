# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include <grad_descend.h>




Grad_Descend::Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in) {

    maximal_iterations = 5001;
    status             = INITIAL_STATE;
    
    // numerical precision used in the calculations
    num_precision = 1.42e-14;    


    
    costfnc__and__gradient = f_pointer;
meta_data = meta_data_in;

    // number of function calls during the optimization process
    function_call_count = 0;

}




double Grad_Descend::Start_Optimization(Matrix_real &x, long maximal_iterations_in)
{



    variable_num       = x.size();

    
    // set the maximal number of iterations
    maximal_iterations = maximal_iterations_in;
    
    status = INITIAL_STATE;
    
    
    Matrix_real g( variable_num, 1 );      

    /// the current search direction
    Matrix_real search_direction( variable_num, 1 ); 

    // the initial gradient during the line search
    Matrix_real g0_search( variable_num, 1 ); 

    // the initial point during the line search
    Matrix_real x0_search( variable_num, 1 ); 
    
    status = WORKSPACE_RESERVED;

      
    // test for dimension
    if ( variable_num <= 1 ) {
        return 0;   
    }


    //     Minimize the objective function
    double f;  
    Optimize(x, g, search_direction, x0_search, g0_search, &f);
    
    return f;

}





int  Grad_Descend::line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double *stepcb, double& d__dot__g0, double *f)
{


    long max_loops = 50;//1 << 30;

    memcpy( x0_search.get_data(), x.get_data(), x.size()*sizeof(double) );
    memcpy( g0_search.get_data(), g.get_data(), g.size()*sizeof(double) );

    Matrix_real g_best = g.copy(); //gopt


    double step = std::min(1.0,*stepcb);
    
    double f_lowest  = *f;
    double f_highest = *f;

    double step_highest = *stepcb;//0.0; (a)
    double step_lowest = 0.0;

    double step_best = 0.0;
    double f_best         = *f;
    double d__dot__g_abs_best = fabs(d__dot__g0);

    double d__dot__g_lowest = d__dot__g0;
    double d__dot__g_highest = 0.0;

    
    for( long iter_idx=0; iter_idx<max_loops; iter_idx++) { //while( true ) {
    
        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }


        get_f_ang_fradient(x, f, g);

        // overlap between the search direction and the gradient 
        double d__dot__g_current = 0.0;
        for (long idx = 0; idx < variable_num; idx++) {
            d__dot__g_current += search_direction[idx] * g[idx];
        }

        // update best solution
        if (*f < f_best || fabs(d__dot__g_current) < d__dot__g_abs_best) {

            step_best      = step;
            f_best         = *f;
            d__dot__g_abs_best = fabs(d__dot__g_current);

            memcpy( g_best.get_data(), g.get_data(), g.size()*sizeof(double) );
	
        }

        // exit the loop if maximal function calls reached
        if (function_call_count == maximal_iterations) {
            break;
        }

        // modify the upper and lower bound of the step inetrval 
        if (*f >= f_lowest + step * 0.1 * d__dot__g0) {  // Armijo test (Wolfe 1st condition)

            step_highest      = step;
            f_highest         = *f;
            d__dot__g_highest = d__dot__g_current;
        
        }
        else {

            if (d__dot__g_current >= d__dot__g0 * 0.7) { // Wolfe 2nd (curvature) condition to exit the iterations
                break;
            }

            step_lowest      = step;
            f_lowest         = *f;
            d__dot__g_lowest = d__dot__g_current;
        }
    
    

/*     Calculate the next step length or end the iterations. */

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
        *f = f_best;

        g = g_best;

        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }
    }

    if (step == 0.0 ) {
        status = ZERO_STEP_SIZE_OCCURED;
    }

    return 0;
} 





int Grad_Descend::Optimize(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double *f)
{


    long iterc = 0;
    // The norm of the matrix Z (initialized to an imposiible value)
    double norm_Z = -1.0;
     

    // inner product of the search direction and the gradient
    double d__dot__g;

    double stepcb; // The upper bound of the allowed step in the search direction
    
    if (function_call_count == 0 || status == VARIABLES_INITIALIZED) {
        get_f_ang_fradient(x, f, g);
    }
    
    double fprev = fabs(*f + *f + 1.0);
    long iterp = -1;

    // Calculate the next search direction. 


    while ( true ) {

        stepcb = 0.0;
    
        double search_direction__grad_overlap = 0.0;
        get_search_direction(g, search_direction, search_direction__grad_overlap);


       /*     Calculate the (bound on the) step-length due to the constraints. */

        if (search_direction__grad_overlap < 0.0) {
            get_Maximal_Line_Search_Step(search_direction, stepcb, search_direction__grad_overlap);
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
            return 0;
        }
    
        // in case the cost fuction does not show decrement in the direction of the line search
        if (d__dot__g >= 0.0) {
            status = NO_DECREASING_SEARCH_DIRECTION;
            return 0;
        }


        // terminate cycles if the cost function is not decreased any more
        if (*f >= fprev) {
            status = MINIMUM_REACHED;
            return 0;
        }

        fprev = *f;


        // terminate if maximal number of iteration reached
        if (function_call_count >= maximal_iterations) {
            status = MAXIMAL_ITERATIONS_REACHED;
            return 0;
        }


        if (iterp == iterc) {
            iterp = iterc;

            if (iterp < 0) {
                return 0;
            }


        }

        ++(iterc);
        
        // perform line search in the direction search_direction
        //line_search_Thuente(x, g, search_direction, x0_search, g0_search, &stepcb, d__dot__g, f);
        line_search(x, g, search_direction, x0_search, g0_search, &stepcb, d__dot__g, f);
    
        if (status == ZERO_STEP_SIZE_OCCURED) {
            return 0;
        }



    } //while

    


}







int Grad_Descend::get_Maximal_Line_Search_Step(Matrix_real& search_direction, double& stepcb, double& search_direction__grad_overlap)
{

    // Set steps to constraint boundaries and find the least positive one.


    stepcb = 0.0;
    
    // the optimization landscape is periodic in 2PI
    // the maximal step will be the 2PI step in the direction of the smallest component of the search direction
    for( long kdx = 0; kdx < variable_num; kdx++ ) {
    
        double step_bound_tmp = std::abs(2*M_PI/search_direction[kdx]);

        if (stepcb == 0.0 || step_bound_tmp < stepcb) {

            stepcb = step_bound_tmp;

        }
        
    }

    return 0;

}





int Grad_Descend::get_f_ang_fradient(Matrix_real& x, double *f, Matrix_real& g)
{

    function_call_count++;
    costfnc__and__gradient(x, meta_data, f, g);
    
    return 0;
}





int Grad_Descend::get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap )
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

    return 0;
} 


Grad_Descend::~Grad_Descend()  {

}
