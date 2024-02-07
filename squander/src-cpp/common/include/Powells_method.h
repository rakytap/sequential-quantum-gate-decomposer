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

# ifndef __POWELLS_METHOD__H
# define __POWELLS_METHOD__H

#include "matrix_real.h"
#include <vector>
#include <random>
/**
 * @brief A class implementing the Powells-algorithm as seen in: https://academic.oup.com/comjnl/article-abstract/7/2/155/335330?redirectedFrom=fulltext&login=false
 */
class Powells_method {
    protected:
    /// number of independent variables in the problem
    int variable_num;
    
    Matrix_real u;    

    Matrix_real v;

    /// function pointer to evaluate the cost function and its gradient vector
    double (*costfnc) (Matrix_real x, void * params);
     
    /// additional data needed to evaluate the cost function
    void* meta_data;
    
    protected:
    double direction(double s,Matrix_real x);
    
    void bracket(double x1, double h, double& a, double& b, Matrix_real x);
    
    void search(double a, double b, double& s, double& f_val, double tol, Matrix_real x);
    public:
    double Start_Optimization(Matrix_real& x, long max_iter);
    
    Powells_method(double (* f_pointer) (Matrix_real, void *), void* meta_data_in);
   
   ~Powells_method();
};


# endif
