//#ifndef FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED
//#define FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED

#pragma once
#include <gsl/gsl_vector.h>
#include "qgd/common.h"

/**
@brief Function operator class to calculate the gradient components of the cost function in parallel
*/
template<typename decomp_class>
class functor_sub_optimization_grad {

protected:

    /// A GNU Scientific Library vector containing the free parameters to be optimized.
    const gsl_vector* parameters;
    /// A pointer pointing to the instance of a class Sub_Matrix_Decomposition.
    decomp_class* instance;
    /// A GNU Scientific Library vector containing the calculated gradient components.
    gsl_vector* grad;
    /// The value of the cost function at parameters_in.
    double f0;
    /// the difference in one direction in the parameter for the gradient calculaiton
    double dparam;

public:

/**
@brief Constructor of the class.
@param parameters_in A GNU Scientific Library vector containing the free parameters to be optimized.
@param instance_in A pointer pointing to the instance of a class Sub_Matrix_Decomposition.
@param grad_in A GNU Scientific Library vector containing the calculated gradient components.
@param f0_in The value of the cost function at parameters_in.
@return Returns with the instance of the class.
*/
functor_sub_optimization_grad( const gsl_vector* parameters_in, decomp_class* instance_in, gsl_vector* grad_in, double f0_in );

/**
@brief Operator to calculate a gradient component of a cost function labeled by index i.
@param i The index labeling the component of the gradien to be calculated.
*/
void operator()( int i ) const;

};




//#endif // FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED
