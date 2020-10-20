#include "qgd/Functor_Cost_Function_Gradient.h"
#include "qgd/Sub_Matrix_Decomposition.h"
#include "qgd/N_Qubit_Decomposition.h"


/*! \file Functor_Cost_Function_Gradient.cpp
    \brief Methods for the paralleized calculation of the gradient components of the cost functions (supporting TBB and OpenMP).
*/


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
@param f0 The value of the cost function at x0.
*/
void N_Qubit_Decomposition::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad, double f0 ) {

    N_Qubit_Decomposition* instance = reinterpret_cast<N_Qubit_Decomposition*>(void_instance);

    int parameter_num_loc = instance->get_parameter_num();

#ifdef TBB
    // calculate the gradient components through TBB parallel for
    tbb::parallel_for(0, parameter_num_loc, 1, functor_sub_optimization_grad<N_Qubit_Decomposition>( parameters, instance, grad, f0 ));
#else
    functor_sub_optimization_grad<N_Qubit_Decomposition> tmp = functor_sub_optimization_grad<N_Qubit_Decomposition>( parameters, instance, grad, f0 );
    #pragma omp parallel for
    for (int idx=0; idx<parameter_num_loc; idx++) {
        tmp(idx);
    }
#endif // TBB

}


/**
@brief Calculate the approximate derivative (f-f0)/(x-x0) of the cost function with respect to the free parameters.
@param parameters A GNU Scientific Library vector containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad A GNU Scientific Library vector containing the calculated gradient components.
@param f0 The value of the cost function at x0.
*/
void Sub_Matrix_Decomposition::optimization_problem_grad( const gsl_vector* parameters, void* void_instance, gsl_vector* grad, double f0 ) {

    Sub_Matrix_Decomposition* instance = reinterpret_cast<Sub_Matrix_Decomposition*>(void_instance);

    int parameter_num_loc = instance->get_parameter_num();

#ifdef TBB
    // calculate the gradient components through TBB parallel for
    tbb::parallel_for(0, parameter_num_loc, 1, functor_sub_optimization_grad<Sub_Matrix_Decomposition>( parameters, instance, grad, f0 ));
#else
    functor_sub_optimization_grad<Sub_Matrix_Decomposition> tmp = functor_sub_optimization_grad<Sub_Matrix_Decomposition>( parameters, instance, grad, f0 );
    #pragma omp parallel for
    for (int idx=0; idx<parameter_num_loc; idx++) {
        tmp(idx);
    }
#endif

}


/**
@brief Constructor of the class.
@param parameters_in A GNU Scientific Library vector containing the free parameters to be optimized.
@param instance_in A pointer pointing to the instance of a class Sub_Matrix_Decomposition.
@param grad_in A GNU Scientific Library vector containing the calculated gradient components.
@param f0_in The value of the cost function at parameters_in.
@return Returns with the instance of the class.
*/
template<typename decomp_class>
functor_sub_optimization_grad<decomp_class>::functor_sub_optimization_grad( const gsl_vector* parameters_in, decomp_class* instance_in, gsl_vector* grad_in, double f0_in ) {

    parameters = parameters_in;
    instance = instance_in;
    grad = grad_in;
    f0 = f0_in;

    // the difference in one direction in the parameter for the gradient calculaiton
    dparam = 1e-8;

}


/**
@brief Operator to calculate a gradient component of a cost function labeled by index i.
@param i The index labeling the component of the gradient to be calculated.
*/
template<typename decomp_class>
void functor_sub_optimization_grad<decomp_class>::operator()( int i ) const {

    decomp_class* instance_loc = NULL;
    if (i == 0) {
        instance_loc = instance;
    }
    else {
        instance_loc = instance->clone();
    }



    gsl_vector* parameters_d = gsl_vector_calloc(parameters->size);
    memcpy( parameters_d->data, parameters->data, parameters->size*sizeof(double) );
    parameters_d->data[i] = parameters_d->data[i] + dparam;

    // calculate the cost function at the displaced point
    double f = instance_loc->optimization_problem(parameters_d, reinterpret_cast<void*>(instance_loc));

    // calculate and set the gradient
    gsl_vector_set(grad, i, (f-f0)/dparam);

    // release vectors
    gsl_vector_free(parameters_d);
    parameters_d = NULL;



    if ( i >0 ) {
        delete instance_loc;
    }

}





