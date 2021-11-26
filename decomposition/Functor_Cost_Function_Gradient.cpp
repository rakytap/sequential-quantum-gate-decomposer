#include "Functor_Cost_Function_Gradient.h"
#include "Sub_Matrix_Decomposition.h"
#include "N_Qubit_Decomposition_Base.h"
#include "N_Qubit_Decomposition.h"
#include "N_Qubit_Decomposition_adaptive.h"
#include "N_Qubit_Decomposition_custom.h"
#include <tbb/parallel_for.h>


/*! \file Functor_Cost_Function_Gradient.cpp
    \brief Methods for the parallel calculation of the gradient components of the cost functions (supporting TBB and OpenMP).
*/


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/

void N_Qubit_Decomposition_Base::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad ) {

    // The function value at x0
    double f0;

    // calculate the approximate gradient
    optimization_problem_combined( parameters, void_instance, &f0, grad);

}


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/

void N_Qubit_Decomposition_Base::optimization_problem_combined( const gsl_vector* parameters, void* void_instance, double* f0, gsl_vector* grad ) {

    N_Qubit_Decomposition_Base* instance = reinterpret_cast<N_Qubit_Decomposition_Base*>(void_instance);

    int parameter_num_loc = instance->get_parameter_num();

    // storage for the function values calculated at the displaced points x
    gsl_vector* f = gsl_vector_alloc(grad->size);

    // the difference in one direction in the parameter for the gradient calculation
    double dparam = 1e-8;

    // calculate the function values at displaced x and the central x0 points through TBB parallel for
    tbb::parallel_for(0, parameter_num_loc+1, 1, functor_grad<N_Qubit_Decomposition_Base>( parameters, instance, f, f0, dparam ));


    for (int idx=0; idx<parameter_num_loc; idx++) {
        // calculate and set the gradient
        gsl_vector_set(grad, idx, (f->data[idx]-(*f0))/dparam);
    }


    gsl_vector_free(f);
}




/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
void Sub_Matrix_Decomposition::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad ) {

    // The function value at x0
    double f0;

    // calculate the approximate gradient
    optimization_problem_combined( parameters, void_instance, &f0, grad);

}



/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
*/
void Sub_Matrix_Decomposition::optimization_problem_combined( const gsl_vector* parameters, void* void_instance, double* f0, gsl_vector* grad ) {

    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(void_instance);

    int parameter_num_loc = instance->get_parameter_num();

    // storage for the function values calculated at the displaced points x
    gsl_vector* f = gsl_vector_alloc(grad->size);

    // the difference in one direction in the parameter for the gradient calculation
    double dparam = 1e-8;

    // calculate the function values at displaced x and the central x0 points through TBB parallel for
    tbb::parallel_for(0, parameter_num_loc+1, 1, functor_grad<Sub_Matrix_Decomposition>( parameters, instance, f, f0, dparam ));

/*
    // sequential version
    functor_sub_optimization_grad<Sub_Matrix_Decomposition> tmp = functor_grad<Sub_Matrix_Decomposition>( parameters, instance, f, f0, dparam );
    #pragma omp parallel for
    for (int idx=0; idx<parameter_num_loc+1; idx++) {
        tmp(idx);
    }
*/


    for (int idx=0; idx<parameter_num_loc; idx++) {
        // set the gradient
#ifdef DEBUG
        if (isnan(f->data[idx])) {
            std::cout << "Sub_Matrix_Decomposition::optimization_problem_combined: f->data[i] is NaN " << std::endl;
            exit(-1);
        }
#endif // DEBUG
        gsl_vector_set(grad, idx, (f->data[idx]-(*f0))/dparam);
    }


    gsl_vector_free(f);

}




/**
@brief Constructor of the class.
@param parameters_in A GNU Scientific Library vector containing the free parameters to be optimized.
@param instance_in A pointer pointing to the instance of a class Sub_Matrix_Decomposition.
@param f_in A GNU Scientific Library vector to store the calculated function values at the displaced points x.
@param f0_in The value of the cost function at parameters_in.
@return Returns with the instance of the class.
*/
template<typename decomp_class>
functor_grad<decomp_class>::functor_grad( const gsl_vector* parameters_in, decomp_class* instance_in, gsl_vector* f_in, double* f0_in, double dparam_in ) {

    parameters = parameters_in;
    instance = instance_in;
    f = f_in;
    f0 = f0_in;

    // the difference in one direction in the parameter for the gradient calculation
    dparam = dparam_in;

}


/**
@brief Operator to calculate a gradient component of a cost function labeled by index i.
@param i The index labeling the component of the gradient to be calculated.
*/
template<typename decomp_class>
void functor_grad<decomp_class>::operator()( int i ) const {


    if (i == (int)parameters->size) {
        // calculate function value at x0
        *f0 = instance->optimization_problem(parameters, reinterpret_cast<void*>(instance));
    }
    else {

        gsl_vector* parameters_d = gsl_vector_calloc(parameters->size);
        memcpy( parameters_d->data, parameters->data, parameters->size*sizeof(double) );
        parameters_d->data[i] = parameters_d->data[i] + dparam;

        // calculate the cost function at the displaced point
        f->data[i] = instance->optimization_problem(parameters_d, reinterpret_cast<void*>(instance));

        // release vectors
        gsl_vector_free(parameters_d);
        parameters_d = NULL;

    }


}





