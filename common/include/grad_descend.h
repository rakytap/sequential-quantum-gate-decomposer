/*
This code is credited to the OPTIMUS project of Ioannis G. Tsoulos
*/

# ifndef __GRAD_DESCEND__H
# define __GRAD_DESCEND__H

#include "matrix_real.h"
#include <vector>

enum solver_status{INITIAL_STATE=0, VARIABLES_INITIALIZED=1, ZERO_STEP_SIZE_OCCURED=2, WORKSPACE_RESERVED=3, MAXIMAL_ITERATIONS_REACHED=4, NO_DECREASING_SEARCH_DIRECTION=5, MINIMUM_REACHED=6, MAX_ITERATIONS_REACHED_DURING_LINE_SEARCH=7};


/**
 * @brief A class implementing the BFGS iterations on the 
 */
class Grad_Descend
{
private:

  
    int line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double *stepcb, double& ddotg, double *f);



    int Optimize(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double *f);


    int get_Maximal_Line_Search_Step(Matrix_real& search_direction, double& stepcb, double& search_direction__grad_overlap);



    int get_search_direction(Matrix_real& g, Matrix_real& search_direction, double& search_direction__grad_overlap);

    int get_f_ang_fradient(Matrix_real& x, double *f, Matrix_real& g);
    

    /// number of independent variables in the problem
    int variable_num;

    /// maximal count of iterations during the optimiation
    long maximal_iterations;

    /// number of function calls during the optimization process
    long function_call_count;

    /// numerical precision used in the calculations
    double num_precision;

    /// function pointer to evaluate the cost function and its gardient vector
    void (*costfnc__and__gradient) (Matrix_real x, void * params, double * f, Matrix_real& g);
     
void* meta_data;

enum solver_status status;



public:
    Grad_Descend(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in);

    double Start_Optimization(Matrix_real &x, long maximal_iterations_in = 5001);

    ~Grad_Descend();
};


# endif
