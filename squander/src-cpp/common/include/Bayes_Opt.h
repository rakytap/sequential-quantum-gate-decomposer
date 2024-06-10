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

# ifndef __BAYES_OPT__H
# define __BAYES_OPT__H

#include "matrix_real.h"
#include <vector>
#include <random>
#ifdef __cplusplus
extern "C" 
{
#endif
int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs, double* A, int LDA, double* B, int LDB); 

#ifdef __cplusplus
}
#endif

/**
@brief ???????????????
@return ???????????
*/
double HS_partial_optimization_problem( Matrix_real parameters, void* void_params);


/**
@brief ???????????????
@return ???????????
*/
void HS_partial_optimization_problem_grad( Matrix_real parameters, void* void_params, Matrix_real& grad);


/**
@brief ???????????????
@return ???????????
*/
void HS_partial_optimization_problem_combined( Matrix_real parameters, void* void_params, double* f0, Matrix_real& grad);



/**
 * @brief A class implementing the BayesOpt algorithm as seen in: https://browse.arxiv.org/pdf/1807.02811.pdf
 */
class Bayes_Opt {
    public:
        ///constant for the mean function
        double mu_0;
        
        ///covariance matrix 
     Matrix_real covariance;
    protected:
    /// number of independent variables in the problem
    int variable_num;

    /// maximal count of iterations during the optimization
    long maximal_iterations;

    /// numerical precision used in the calculations
    double num_precision;
    
    /// amplitude of the kernel
    double alpha0;

    /// function pointer to evaluate the cost function and its gradient vector
    double (*costfnc) (Matrix_real x, void * params);
     
    /// additional data needed to evaluate the cost function
    void* meta_data;
    
    ///previous parameters
    std::vector<Matrix_real> x_prev;
    
    
    //previous cost functions
    std::vector<double> f_prev;
    
    
    //current minimum
    double current_maximum;
    
    //also known as n0
    int initial_samples;
    
    std::mt19937 gen; 
    protected:
    

    
    static void optimization_problem_combined(Matrix_real x, void* void_instance, double* f0, Matrix_real& grad );
    
    static double optimization_problem(Matrix_real x_Powell, void* void_instance);
    
    double expected_improvement(double mu_n, double sigma_n);
    
    void expected_improvement_combined(double mu_n, double sigma_n, Matrix_real& grad_mu, Matrix_real& grad_sigma, double* f, Matrix_real& grad);
    
    void calculate_conditional_distribution(Matrix_real x, Matrix_real cov_x, double& mu_n, double& sigma2_n);
    
    void calculate_conditional_distribution_combined(Matrix_real x, Matrix_real cov_x, Matrix_real cov_x_grad, Matrix_real cov_self_grad, double& mu_n, double& sigma2_n, Matrix_real& grad_mu, Matrix_real& grad_sigma);
    
    double kernel(Matrix_real x0, Matrix_real x1);
    
    void kernel_combined(Matrix_real x0, Matrix_real x, double& f, Matrix_real& grad, int grad_var, bool self);
    
    double pdf(double mu, double sigma);
    
    double cdf(double mu, double sigma);
    
    double grad_pdf(double mu, double sigma, double grad_mu, double grad_sigma);
    
    void update_covariance(Matrix_real cov_new);
    
    public:
    double Start_Optimization(Matrix_real& x, int max_iterations_in);
    
    Bayes_Opt(double (* f_pointer) (Matrix_real, void *), void* meta_data_in);
   
   ~Bayes_Opt();
};

class Bayes_Opt_Beam{
    protected:
    /// number of independent variables in the problem
    int variable_num;

    /// maximal count of iterations during the optimization
    long maximal_iterations;

    /// numerical precision used in the calculations
    double num_precision;

    /// function pointer to evaluate the cost function and its gradient vector
    double (*costfnc) (Matrix_real x, void * params);
     
    /// additional data needed to evaluate the cost function
    void* meta_data;
    
    //current minimum
    double current_maximum;
    
    Matrix_real parameters;
    
    int start;
    
    std::mt19937 gen; 
    protected:
    static double optimization_problem(Matrix_real x_Beam, void* void_instance);
    public:
    Bayes_Opt_Beam(double (* f_pointer) (Matrix_real, void *), void* meta_data_in, int start_in, Matrix_real parameters_original_in);
    
    double Start_Optimization(Matrix_real& x, int max_iterations_in);
    
    ~Bayes_Opt_Beam();
    
};

# endif
