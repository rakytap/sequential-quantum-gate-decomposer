/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file qgd/Functor_Cost_Function_Gradient.h
    \brief Header file for the paralleized calculation of the gradient components of the cost functions (supporting TBB and OpenMP).
*/

#ifndef FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED
#define FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED

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




#endif // FUNCTOR_COST_FUNCTION_GRADIENT_H_INCLUDED
