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
#include <math.h>
#include <cfloat>
#include <Powells_method.h>
#include "tbb/tbb.h"

extern "C" int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs, double* A, int LDA, double* B, int LDB); 	
/**
@brief Constructor of the class.
@param f_pointer A function pointer (x, meta_data, f, grad) to evaluate the cost function and its gradients. The cost function and the gradient vector are returned via reference by the two last arguments.
@param meta_data void pointer to additional meta data needed to evaluate the cost function.
@return An instance of the class
*/
Powells_method::Powells_method(double (* f_pointer) (Matrix_real, void *), void* meta_data_in) {
    
    costfnc = f_pointer;
    
    meta_data = meta_data_in;
    
}

double Powells_method::direction(double s,Matrix_real x){
    Matrix_real x_new(1,variable_num);
    for (int idx=0;idx<variable_num;idx++){
        x_new[idx] = x[idx] + s*v[idx];
    }
    return costfnc(x_new,meta_data);
}

void Powells_method::bracket(double x1, double h, double& a, double& b, Matrix_real x){
    double c=1.618033989;
    double f1 = direction(x1,x);
    double x2 = x1 + h;
    double f2 = direction(x2,x);
    if (f2>f1){
        h = -1.*h;
        x2 = x1 +h;
        f2 = direction(x2,x);
        if (f2>f1){
            a = x2;
            b = x1 - h;
            return;
        }
    }
    for (int idx=0;idx<100;idx++){
        h = c*h;
        double x3 = x2 + h;
        double f3 = direction(x3,x);
        if (f3>f2){
            a = x1;
            b = x3;
            return;
        }
        x1 = x2;
        x2 = x3;
        f1 = f2;
        f2 = f3;
    }
}
void Powells_method::search(double a, double b, double& s, double& f_val, double tol, Matrix_real x){

    int nIter = (int)std::ceil(-2.078087*std::log(tol/std::fabs(b-a)));
    double R = 0.618033989;
    double C = 1.0 - R;
    double x1 = R*a + C*b; 
    double x2 = C*a + R*b;
    double f1 = direction(x1,x); 
    double f2 = direction(x2,x);
    for (int iter=0; iter<nIter;iter++){
        if (f1>f2){
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = C*a + R*b;
            f2 = direction(x2,x);
        }
        else {
            b=x2;
            x2 = x1;
            f2 = f1;
            x1 = R*a + C*b;
            f1 = direction(x1,x);
        }
    }
  if (f1<f2){
    s = x1;
    f_val = f1;
  }
  else{
    s = x2;
    f_val = f2;
  }
  return;
}
double Powells_method::Start_Optimization(Matrix_real& x, long max_iter){
    double h = 0.1;
    double tol = 1e-6;
    variable_num = x.size();
    u = Matrix_real(variable_num,variable_num);
    memset(u.get_data(), 0.0, variable_num*variable_num*sizeof(double) );
    for (int var_idx=0;var_idx<variable_num;var_idx++){
        u[var_idx*variable_num + var_idx] = 1.0;
    }
    Matrix_real df(1,variable_num);
    memset(df.get_data(), 0.0, variable_num*sizeof(double) );
    double a;
    double b;
    double s;
    double fMin;
    double fLast;
    v = Matrix_real(1,variable_num);
    for (int iter=0;iter<max_iter;iter++){
        Matrix_real xOld = x.copy();
        double fOld = costfnc(xOld,meta_data);
        for (int var_idx=0;var_idx<variable_num;var_idx++){
            memcpy(v.get_data(),u.get_data()+var_idx*variable_num,sizeof(double)*variable_num);
            bracket(0.0,h,a,b,x);
            search(a,b,s,fMin,tol,x);
            df[var_idx] = fOld - fMin;
            fOld = fMin;
            for (int idx=0;idx<variable_num;idx++){
                x[idx] = x[idx] + s*v[idx];
            }
        }
        for (int idx=0;idx<variable_num;idx++){
            v[idx] = x[idx] - xOld[idx];
        }
        bracket(0.0,h,a,b,x);
        search(a,b,s,fLast,tol,x);
        double dot=0.;
        for (int idx=0;idx<variable_num;idx++){
            x[idx] = x[idx] + s*v[idx];
            dot = dot + (x[idx] - xOld[idx])*(x[idx] - xOld[idx]);
        }
        if(std::sqrt(dot)/variable_num<tol){return costfnc(x,meta_data);}
        int Imax=0.;
        for (int idx=1;idx<variable_num;idx++){
            Imax = (df[idx]>df[Imax]) ? idx:Imax;
        }
        for (int idx = Imax;idx<variable_num-1;idx++){
            memcpy(u.get_data()+idx*variable_num,u.get_data()+(idx+1)*variable_num,sizeof(double)*variable_num);
        }
        memcpy(u.get_data()+(variable_num-1)*variable_num,v.get_data(),sizeof(double)*variable_num);
   }
    return costfnc(x,meta_data);
}
/**
@brief Destructor of the class
*/
Powells_method::~Powells_method()  {

}
