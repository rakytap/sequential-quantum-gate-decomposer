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
#include <Bayes_Opt.h>
#include <Powells_method.h>
#include <common.h>
#include "tbb/tbb.h"

extern "C" int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs, double* A, int LDA, double* B, int LDB); 	


/**
@brief ???????????????
@return ???????????
*/
double HS_partial_optimization_problem( Matrix_real parameters, void* void_params) {

    double* params = (double*)void_params;

    return params[0]*sin(2*parameters[0] + params[1]) + params[2]*sin(parameters[0] + params[3] ) + params[4];
}


/**
@brief ???????????????
@return ???????????
*/
void HS_partial_optimization_problem_grad( Matrix_real parameters, void* void_params, Matrix_real& grad) {


    double* params = (double*)void_params;
    grad[0] = 2*params[0]*cos(2*parameters[0] + params[1]) + params[2]*cos(parameters[0] + params[3] );

}


/**
@brief ???????????????
@return ???????????
*/
void HS_partial_optimization_problem_combined( Matrix_real parameters, void* void_params, double* f0, Matrix_real& grad) {

    *f0 = HS_partial_optimization_problem( parameters, void_params );
    HS_partial_optimization_problem_grad( parameters, void_params, grad);


}




/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Bayes_Opt::Bayes_Opt(double (* f_pointer) (Matrix_real, void *), void* meta_data_in) {

    maximal_iterations = 101;
    
    // numerical precision used in the calculations
    num_precision = 1.42e-14;    

    alpha0 = 1;
    
    costfnc = f_pointer;
    
    meta_data = meta_data_in;

    
    initial_samples = 12;
    
    mu_0 = M_PI;
    
    current_maximum = -10000.;
    
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  

    // seedign the random generator
    gen = std::mt19937(rd());
    
}

double Bayes_Opt::Start_Optimization(Matrix_real& x, int max_iterations_in){

    variable_num = x.size();

    
    maximal_iterations = max_iterations_in;
    
    //get samples of f0 at n0 points
    for (int sample_idx=0; sample_idx<initial_samples; sample_idx++){
        Matrix_real covariance_new(sample_idx,sample_idx);
        Matrix_real parameters_new(1,variable_num);
        std::normal_distribution<> distrib_real(0, M_PI/4);
        double f_random;
        for(int idx = 0; idx < variable_num; idx++) {
            double random = distrib_real(gen);
            parameters_new[idx] = x[idx] + random;
        }
        f_random = -1.*costfnc(parameters_new,meta_data);

        x_prev.push_back(parameters_new);
        f_prev.push_back(f_random);
        if (f_random>current_maximum) {
            current_maximum = f_random;
            memcpy(x.get_data(),parameters_new.get_data(),sizeof(double)*variable_num);
        }
    }
    //construct covariance matrix
    covariance = Matrix_real(initial_samples,initial_samples);
    for (int idx=0; idx<initial_samples; idx++){
        for (int jdx=0; jdx<initial_samples; jdx++){
            covariance[idx*initial_samples + jdx] = kernel(x_prev[idx],x_prev[jdx]);
            if (idx==jdx){
            covariance[idx*initial_samples + jdx] = covariance[idx*initial_samples + jdx] +1e-4;
            }
        }
    }

    //Start optimization
    int iterations = initial_samples;
    for (int iter = iterations; iter<maximal_iterations;iter++){


        Matrix_real solution_guess = x_prev[iterations-1];
        
        Powells_method cPowells_method(optimization_problem,this);
        double f_Powell = cPowells_method.Start_Optimization(solution_guess, 100);
        double f = -1.*costfnc(solution_guess,meta_data);
        
        Matrix_real cov_x(1,(int)x_prev.size());
        
        for (int idx=0; idx<(int)x_prev.size();idx++){
            cov_x[idx] = kernel(x_prev[idx],solution_guess);
        }
        
        update_covariance(cov_x);
        x_prev.push_back(solution_guess);
        f_prev.push_back(f);
        if (f>current_maximum){
            current_maximum = f;
            memcpy(x.get_data(),solution_guess.get_data(),sizeof(double)*variable_num);
        }
        /*Grad_Descend cBFGS_Powell(optimization_problem_combined, this);

        double f_bfgs = cBFGS_Powell.Start_Optimization(solution_guess, 100);
        //solution_guess.print_matrix();
        double f = -1.*costfnc(solution_guess,meta_data);


        Matrix_real cov_x(1,(int)x_prev.size());
        double self_cov = 1e-4;

        for (int idx=0; idx<(int)x_prev.size();idx++){
            cov_x[idx] = kernel(x_prev[idx],solution_guess);
        }

        update_covariance(cov_x);

        x_prev.push_back(solution_guess);
        f_prev.push_back(f);
        if (f>current_maximum){
            current_maximum = f;
            memcpy(solution_guess.get_data(),x.get_data(),sizeof(double)*variable_num);
        }

        //std::cout<<iter<<std::endl;*/

    }
    return -1.*current_maximum;

}

double Bayes_Opt::optimization_problem(Matrix_real x_Powell, void* void_instance){
    Bayes_Opt* instance = reinterpret_cast<Bayes_Opt*>(void_instance);
    int samples_n = (int)instance->x_prev.size();
    int parameter_num = instance->variable_num;
    Matrix_real cov_x(1,samples_n);
   for (int idx=0; idx<samples_n; idx++){
        cov_x[idx] = instance->kernel(x_Powell,(Matrix_real)instance->x_prev[idx]);
   }
   double mu_n = 0.0; 
   double sigma2_n = 0.0;
   instance->calculate_conditional_distribution(x_Powell, cov_x, mu_n, sigma2_n);
   double sigma_n = std::sqrt(std::fabs(sigma2_n));
   double EI = -1.0*instance->expected_improvement(mu_n, sigma_n);
   return EI;
}

double Bayes_Opt::expected_improvement(double mu_n, double sigma_n){
    double deltax = mu_n - current_maximum;
    double deltax_max = (deltax>0.) ? deltax:0.0;

    double EI = deltax_max + sigma_n*pdf(deltax,sigma_n) - fabs(deltax)*cdf(deltax,sigma_n);
    return EI;
}


void Bayes_Opt::calculate_conditional_distribution(Matrix_real x, Matrix_real cov_x, double& mu_n, double& sigma2_n){
    double tol = 1e-6;
    int samples = (int)f_prev.size();
    Matrix_real mu_rhs(1,samples); 
    Matrix_real x0(1,samples);
    for (int idx=0;idx<samples;idx++){
        mu_rhs[idx] = f_prev[idx] - mu_0;
        x0[idx] = M_PI;
    }
    
    conjugate_gradient(covariance,mu_rhs,x0,tol);
    
    for (int idx=0;idx<samples;idx++){
        mu_n = mu_n + x0[idx]*cov_x[idx];
    }
    
    Matrix_real sigma2_rhs = cov_x.copy();
    memset(x0.get_data(), M_PI, samples*sizeof(double) );
    conjugate_gradient(covariance,mu_rhs,x0,tol);
    
    for (int idx=0;idx<mu_rhs.cols;idx++){
        sigma2_n = sigma2_n + x0[idx] * cov_x[idx];
    }
    
    sigma2_n = -1.0*(sigma2_n - kernel(x,x));

    
    return;

}



double Bayes_Opt::kernel(Matrix_real x0, Matrix_real x1){

    double dist=0.;
    double sigma01 = 0.;
    for (int idx=0; idx<variable_num; idx++){
        dist = dist + (x0[idx]-x1[idx])*(x0[idx]-x1[idx]);
    }
    dist = std::sqrt(dist)/variable_num/variable_num;
    sigma01 = alpha0*std::exp(-1.*std::sin(2*dist)*std::sin(2*dist)) + 1e-6;
    
    return sigma01;

}

double Bayes_Opt::pdf(double mu, double sigma){
    return 1/sigma/std::sqrt(2*M_PI)*std::exp(0.5*(mu/sigma)*(mu/sigma));
}

double Bayes_Opt::cdf(double mu, double sigma){
    return 0.5+0.5*std::erf(mu/sigma/std::sqrt(2));
}

double Bayes_Opt::grad_pdf(double mu, double sigma, double grad_mu, double grad_sigma){
    return -1.0/sigma*grad_sigma*pdf(mu,sigma) - pdf(mu,sigma)*(grad_mu*sigma - mu*grad_sigma)/sigma/sigma;
}

void Bayes_Opt::update_covariance(Matrix_real cov_new){

    int samples = covariance.cols+1;

    Matrix_real covariance_new(samples,samples);
    double* data_new = covariance_new.get_data();
    double* data_old = covariance.get_data();
    ///covariance matrix 

    //copy old covariance matrix

    for (int idx=0;idx<covariance.cols; idx++){
        memcpy(data_new + samples*idx,data_old + idx*covariance.cols,sizeof(double)*covariance.cols);
        covariance_new[samples-1+idx*samples] = cov_new[idx]; 
    }

    memcpy(data_new + (samples-1)*samples,cov_new.get_data(),sizeof(double)*covariance.cols);
    covariance_new[samples*samples-1] = alpha0;
    
    covariance = covariance_new;
    return;
}

void Bayes_Opt::kernel_combined(Matrix_real x0, Matrix_real x, double& f, Matrix_real& grad, int grad_var, bool self){

    double dist=0.;
    for (int idx=0; idx<variable_num; idx++){
        dist = dist + (x0[idx]-x[idx])*(x0[idx]-x[idx]);
    }
    dist = std::sqrt(dist);
    double cost_func_base = alpha0*std::exp(-1.*dist) + 1e-6;
    f = cost_func_base;
    for (int grad_idx=0; grad_idx<variable_num; grad_idx++){
        double x1 = x[grad_idx];
        grad[grad_var*variable_num + grad_idx] = 0.;
        for (int idx=0; idx<variable_num; idx++){
            if (grad_idx == idx && self == true){
                    grad[grad_var*variable_num + grad_idx] = 0.;
                    continue; 
            }
            grad[grad_var*variable_num + grad_idx] = grad[grad_var*variable_num + grad_idx]  - cost_func_base*(2.*x1-2.*x0[idx]);
        }
    }
    return;

}

void Bayes_Opt::calculate_conditional_distribution_combined(Matrix_real x, Matrix_real cov_x, Matrix_real cov_x_grad, Matrix_real cov_self_grad, double& mu_n, double& sigma2_n, Matrix_real& grad_mu, Matrix_real& grad_sigma){

    int samples = (int)f_prev.size();
    double tol = 1e-4;
    Matrix_real b(1,samples);
    Matrix_real mu_rhs(1,samples);
    Matrix_real sigma2_rhs(1,samples);


    Matrix_real sigma2_grad_rhs (samples,variable_num);
    for (int idx=0;idx<samples;idx++){
        b[idx] = f_prev[idx] - mu_0;
        mu_rhs[idx] = 1.0;
        sigma2_rhs[idx] = M_PI;
    }
    conjugate_gradient(covariance,b,mu_rhs,tol);
    mu_n = 0.;
    for (int idx=0; idx<samples; idx++){
        mu_n = mu_n + mu_rhs[idx]*cov_x[idx];
    }
    mu_n = std::fabs(mu_n);
    for (int grad_idx=0; grad_idx<variable_num; grad_idx++){
            for (int idx=0;idx<samples;idx++){
                grad_mu[grad_idx] = grad_mu[grad_idx] + cov_x_grad[idx*variable_num + grad_idx]*mu_rhs[idx];
                //sigma2_grad_rhs[idx*variable_num + grad_idx] = M_PI;
            }
    }

   b = cov_x.copy();
    
    conjugate_gradient(covariance,b,sigma2_rhs,tol);
    sigma2_n = 0.;
    for (int idx=0;idx<samples;idx++){
        sigma2_n = sigma2_n + sigma2_rhs[idx] * cov_x[idx];
    }
    sigma2_n = kernel(x,x)-sigma2_n;
    //std::cout<<mu_n<<" "<<sigma2_n<<std::endl;
    //b = cov_x_grad.copy();
   // conjugate_gradient_parallel(covariance,b,sigma2_grad_rhs,tol);
   for (int grad_idx=0; grad_idx<variable_num; grad_idx++){
        b = Matrix_real(1,samples);
        for (int idx=0;idx<samples;idx++){
            b[idx] = cov_x_grad[idx*variable_num+grad_idx];
        }
        conjugate_gradient(covariance,b,sigma2_rhs,tol);
        grad_sigma[grad_idx] = cov_self_grad[grad_idx];
        for (int idx=0;idx<samples;idx++){
            grad_sigma[grad_idx] = grad_sigma[grad_idx] - 2.*cov_x_grad[idx*variable_num + grad_idx];
     }
      grad_sigma[grad_idx] = grad_sigma[grad_idx]/std::sqrt(sigma2_n)/2;
    }
    //grad_sigma.print_matrix();
    return;

}

void Bayes_Opt::expected_improvement_combined(double mu_n, double sigma_n, Matrix_real& grad_mu, Matrix_real& grad_sigma, double* f, Matrix_real& grad){
    double deltax = mu_n - current_maximum;
    double deltax_max = (deltax>0.) ? deltax:0.0;

    double pdf_mu = pdf(mu_n,sigma_n);
    double cdf_mu = cdf(mu_n,sigma_n);
    *f = -1.*(deltax_max + sigma_n*pdf_mu - fabs(deltax)*cdf_mu);
    
    for (int idx=0; idx<variable_num; idx++){
        double deltax_grad =  grad_mu[idx];
        double deltax_max_grad = (deltax_grad>0.) ? grad_mu[idx] : 0.0;
        double grad_rhs = -1.*std::fabs(deltax_grad)*cdf_mu - std::fabs(deltax)*pdf_mu*(grad_mu[idx]*sigma_n-mu_n*grad_sigma[idx])/sigma_n/sigma_n;
        double grad_lhs = grad_sigma[idx]*pdf_mu + sigma_n*grad_pdf(mu_n,sigma_n,grad_mu[idx],grad_sigma[idx]);
        grad[idx] = -1.;//*(deltax_max_grad + grad_lhs + grad_rhs);
    }
    return;
}
void Bayes_Opt::optimization_problem_combined(Matrix_real x_bfgs, void* void_instance, double* f0, Matrix_real& grad ){

    Bayes_Opt* instance = reinterpret_cast<Bayes_Opt*>(void_instance);
    int samples_n = (int)instance->x_prev.size();
    int parameter_num = instance->variable_num;
    Matrix_real cov_x(1,samples_n);
    Matrix_real cov_x_grad(samples_n,parameter_num);
   //x_bfgs.print_matrix();
   for (int grad_idx=0; grad_idx<samples_n; grad_idx++){
        double cov_new;
        instance->kernel_combined(x_bfgs, (Matrix_real)instance->x_prev[grad_idx], cov_new, cov_x_grad, grad_idx, false);            
        cov_x[grad_idx] = cov_new;
    }
    //cov_x_grad.print_matrix();
    double placeholder;
    Matrix_real cov_self_grad(1,parameter_num);

    instance->kernel_combined(x_bfgs, x_bfgs, placeholder,cov_self_grad,0,true);

    double mu_n = M_PI; 
    double sigma2_n;
    Matrix_real grad_mu(1,parameter_num);
    Matrix_real grad_sigma(1,parameter_num);

    instance -> calculate_conditional_distribution_combined(x_bfgs, cov_x, cov_x_grad, cov_self_grad, mu_n, sigma2_n, grad_mu, grad_sigma);

    double sigma_n = std::sqrt(sigma2_n);
    instance -> expected_improvement_combined(mu_n,sigma_n,grad_mu,grad_sigma,f0,grad);
    //grad.print_matrix();
    return;
}
/**
@brief Destructor of the class
*/
Bayes_Opt::~Bayes_Opt()  {

}



Bayes_Opt_Beam::Bayes_Opt_Beam(double (* f_pointer) (Matrix_real, void *), void* meta_data_in, int start_in, Matrix_real parameters_original_in) {

    maximal_iterations = 101;
    
    costfnc = f_pointer;
    
    meta_data = meta_data_in;
    
    current_maximum = -10000.;
    
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  

    // seedign the random generator
    gen = std::mt19937(rd());
    
    parameters = parameters_original_in;
    
    start = start_in;
    
    
}



double Bayes_Opt_Beam::optimization_problem(Matrix_real x_Beam, void* void_instance){
    Bayes_Opt_Beam* instance = reinterpret_cast<Bayes_Opt_Beam*>(void_instance);
    Matrix_real x = instance->parameters.copy();
    memcpy(x.get_data() + instance->start,x_Beam.get_data(),sizeof(double)*x_Beam.size());
    return instance->costfnc(x,instance->meta_data);
}



double Bayes_Opt_Beam::Start_Optimization(Matrix_real& x, int max_iterations_in){

    variable_num = x.size();

    maximal_iterations = max_iterations_in;
    Bayes_Opt cBayes_opt(optimization_problem,this);
    double f = cBayes_opt.Start_Optimization(x,maximal_iterations);
    
    return f;
}
Bayes_Opt_Beam::~Bayes_Opt_Beam()  {

}
