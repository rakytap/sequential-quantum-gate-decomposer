/*
Created on Fri Jun 26 14:13:26 2020
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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Random_Unitary.cpp
    \brief A class and methods to cerate random unitary matrices
*/


#include "Random_Unitary.h"
#include <vector>


/**
@brief Call to create a random unitary constructed by CNOT operation between randomly chosen qubits and by random U3 operations.
@param qbit_num The number of qubits spanning the unitary.
@param cnot_num The number of CNOT gates composing the random unitary.
@return Returns with the the constructed random unitary.
*/
Matrix
few_CNOT_unitary( int qbit_num, int cnot_num) {

    // the current number of CNOT gates
    int cnot_num_curr = 0;

    // the size of the matrix
    int matrix_size = Power_of_2(qbit_num);

    // The unitary describing each qubits in their initial state
    Matrix mtx = create_identity( matrix_size );

    // constructing the unitary
    while (true) {
        int cnot_or_u3 = rand() % 5 + 1;

        CNOT* cnot_op = NULL;
        U3* u3_op = NULL;

        Matrix gate_matrix;

        if (cnot_or_u3 <= 4) {
            // creating random parameters for the U3 operation
            Matrix_real parameters(1, 3);
            parameters[0] = double(rand())/RAND_MAX*4*M_PI;
            parameters[1] = double(rand())/RAND_MAX*2*M_PI;
            parameters[2] = double(rand())/RAND_MAX*2*M_PI;


            // randomly choose the target qbit
            int target_qbit = rand() % qbit_num;

            // creating the U3 gate
            u3_op = new U3(qbit_num, target_qbit);

            // get the matrix of the operation
            
            gate_matrix = u3_op->get_matrix(parameters);
        }
        else if ( cnot_or_u3 == 5 ) {
            // randomly choose the target qbit
            int target_qbit = rand() % qbit_num;

            // randomly choose the control qbit
            int control_qbit = rand() % qbit_num;

            if (target_qbit == control_qbit) {
                gate_matrix = create_identity( matrix_size );
            }
            else {

                // creating the CNOT gate
                cnot_op = new CNOT(qbit_num, control_qbit, target_qbit);

                // get the matrix of the operation
                gate_matrix = cnot_op->get_matrix();

                cnot_num_curr = cnot_num_curr + 1;
            }
        }
        else {
            gate_matrix = create_identity( matrix_size );
        }


        // get the current unitary
        Matrix mtx_tmp = dot(gate_matrix, mtx);
        mtx = mtx_tmp;

        delete u3_op;
        u3_op = NULL;

        delete cnot_op;
        cnot_op = NULL;

        // exit the loop if the maximal number of CNOT gates reached
        if (cnot_num_curr >= cnot_num) {
            return mtx;
        }

    }

}


// -----------------------------------------------------------------------
// Implementation helpers – anonymous namespace
//
// Templatized on scalar (double or float) so both precisions share one
// code path with no conversion.  The public class methods are thin
// dispatchers that call construct_unitary_tmpl<scalar>.
// -----------------------------------------------------------------------

namespace {

// Type traits: map scalar type → complex element type + matrix type + identity factory
template<typename scalar> struct RunTraits;

template<> struct RunTraits<double> {
    using complex = QGD_Complex16;
    using matrix  = Matrix;
    static matrix identity(int dim) { return create_identity(dim); }
};

template<> struct RunTraits<float> {
    using complex = QGD_Complex8;
    using matrix  = Matrix_float;
    static matrix identity(int dim) { return create_identity_float(dim); }
};

// γ – Eq (11): always ±1; computed in double, cast to scalar
template<typename scalar>
inline scalar gamma_impl(int dim) {
    return static_cast<scalar>(
        std::pow(-1.0, 0.25*(2*dim - 1 + std::pow(-1.0, dim)))
    );
}

// Index mapping: β + (α-1)(α-2)/2
inline int convert_index_impl(int varalpha, int varbeta) {
    return varbeta + (varalpha - 1)*(varalpha - 2)/2;
}

// Q matrix – Eq (9)
template<typename scalar>
typename RunTraits<scalar>::matrix
Q_tmpl( typename RunTraits<scalar>::complex u1,
        typename RunTraits<scalar>::complex u2 )
{
    using MatrixT = typename RunTraits<scalar>::matrix;
    MatrixT ret(2, 2);
    ret[0] = u2;
    ret[1] = u1;
    ret[2].real = -u1.real;
    ret[2].imag =  u1.imag;
    ret[3].real =  u2.real;
    ret[3].imag = -u2.imag;
    return ret;
}

// E_{α,β}: dim×dim unit-basis matrix with 1 at (varalpha, varbeta)
template<typename scalar>
typename RunTraits<scalar>::matrix
E_alpha_beta_tmpl(int varalpha, int varbeta, int dim)
{
    using ComplexT = typename RunTraits<scalar>::complex;
    using MatrixT  = typename RunTraits<scalar>::matrix;
    MatrixT ret(dim, dim);
    memset(ret.get_data(), 0, dim*dim*sizeof(ComplexT));
    ret[varalpha*dim + varbeta].real = scalar(1);
    return ret;
}

// I_{α,β}: identity with zeros at diagonal positions (α,α) and (β,β)
template<typename scalar>
typename RunTraits<scalar>::matrix
I_alpha_beta_tmpl(int varalpha, int varbeta, int dim)
{
    typename RunTraits<scalar>::matrix ret = RunTraits<scalar>::identity(dim);
    ret[varalpha*dim + varalpha].real = scalar(0);
    ret[varbeta*dim  + varbeta ].real = scalar(0);
    return ret;
}

// M matrix – Eq (8)
template<typename scalar>
typename RunTraits<scalar>::matrix
M_tmpl(int varalpha, int varbeta,
       typename RunTraits<scalar>::complex s,
       typename RunTraits<scalar>::complex t,
       int dim)
{
    using MatrixT = typename RunTraits<scalar>::matrix;
    MatrixT Qloc = Q_tmpl<scalar>(s, t);
    MatrixT ret1 = E_alpha_beta_tmpl<scalar>(varbeta,  varbeta,  dim);
    MatrixT ret2 = E_alpha_beta_tmpl<scalar>(varbeta,  varalpha, dim);
    MatrixT ret3 = E_alpha_beta_tmpl<scalar>(varalpha, varbeta,  dim);
    MatrixT ret4 = E_alpha_beta_tmpl<scalar>(varalpha, varalpha, dim);
    mult(Qloc[0], ret1);
    mult(Qloc[1], ret2);
    mult(Qloc[2], ret3);
    mult(Qloc[3], ret4);
    MatrixT ret(dim, dim);
    for (int idx = 0; idx < dim*dim; ++idx) {
        ret[idx].real = ret1[idx].real + ret2[idx].real + ret3[idx].real + ret4[idx].real;
        ret[idx].imag = ret1[idx].imag + ret2[idx].imag + ret3[idx].imag + ret4[idx].imag;
    }
    return ret;
}

// Ω matrix – Eq (6)
template<typename scalar>
typename RunTraits<scalar>::matrix
Omega_tmpl(int varalpha, int varbeta,
           typename RunTraits<scalar>::complex x,
           typename RunTraits<scalar>::complex y,
           int dim)
{
    using MatrixT = typename RunTraits<scalar>::matrix;
    MatrixT ret = I_alpha_beta_tmpl<scalar>(varalpha, varbeta, dim);
    // threshold = 3 + δ(dim,2): equals 4 when dim==2, 3 otherwise
    const int threshold = 3 + (dim == 2 ? 1 : 0);
    MatrixT Mloc;
    if (varalpha + varbeta != threshold) {
        Mloc = M_tmpl<scalar>(varalpha, varbeta, x, y, dim);
    } else {
        y.imag = -y.imag;
        Mloc = M_tmpl<scalar>(varalpha, varbeta, x,
                              mult(gamma_impl<scalar>(dim), y), dim);
    }
    for (int idx = 0; idx < dim*dim; ++idx) {
        ret[idx].real += Mloc[idx].real;
        ret[idx].imag += Mloc[idx].imag;
    }
    return ret;
}

// Core algorithm – arXiv:1303.5904v1, native in scalar precision
template<typename scalar>
typename RunTraits<scalar>::matrix
construct_unitary_tmpl(scalar* vartheta, scalar* varphi, scalar* varkappa, int dim)
{
    using ComplexT = typename RunTraits<scalar>::complex;
    using MatrixT  = typename RunTraits<scalar>::matrix;

    MatrixT ret = RunTraits<scalar>::identity(dim);

    for (int varalpha = 1; varalpha < dim; ++varalpha) {
        for (int varbeta = 0; varbeta < varalpha; ++varbeta) {
            const int    param_idx  = convert_index_impl(varalpha, varbeta);
            const scalar theta_loc  = vartheta[param_idx];
            const scalar phi_loc    = varphi[param_idx];

            // Eq (26): a = cos(θ)·exp(iφ), sign-negated per paper
            ComplexT a;
            a.real = scalar(std::cos(theta_loc) * std::cos(phi_loc));
            a.imag = scalar(std::cos(theta_loc) * std::sin(phi_loc));

            // Eq (28): ε = δ(α-1,β)·κ_{α-1}; b = sin(θ)·exp̄(iε)
            const scalar varepsilon = (varalpha - 1 == varbeta)
                                      ? varkappa[varalpha - 1] : scalar(0);
            ComplexT b;
            b.real = scalar(std::sin(theta_loc) * std::cos(varepsilon));
            b.imag = scalar(std::sin(theta_loc) * std::sin(varepsilon));

            a.real = -a.real;
            b.imag = -b.imag;

            MatrixT Omega_loc = Omega_tmpl<scalar>(varalpha, varbeta, a, b, dim);
            ret = dot(ret, Omega_loc);
        }
    }

    // Apply overall phase γ – Eq (11)
    ComplexT gamma_loc;
    gamma_loc.real = gamma_impl<scalar>(dim);
    gamma_loc.imag = scalar(0);
    for (int idx = 0; idx < dim*dim; ++idx) {
        ret[idx] = mult(ret[idx], gamma_loc);
    }

    return ret;
}

} // anonymous namespace


/**
@brief Constructor of the class.
@param dim_in The number of rows in the random unitary to be created.
@return An instance of the class
*/
Random_Unitary::Random_Unitary( int dim_in ) {

        if (dim_in < 2) {
            throw "wrong dimension";
        }

        // number of qubits
        dim = dim_in;

}


/**
@brief Construct a random unitary with internally generated float64 parameters.
@return The constructed random unitary.
*/
Matrix
Random_Unitary::Construct_Unitary_Matrix() {

    const int n = int(dim*(dim-1)/2);

    double* vartheta = (double*) qgd_calloc( n,       sizeof(double), 64 );
    double* varphi   = (double*) qgd_calloc( n,       sizeof(double), 64 );
    double* varkappa = (double*) qgd_calloc( dim - 1, sizeof(double), 64 );

    srand( static_cast<unsigned int>(time(NULL)) );

    for (int idx = 0; idx < n;       ++idx) vartheta[idx] = (2*double(rand())/RAND_MAX - 1)*2*M_PI;
    for (int idx = 0; idx < n;       ++idx) varphi[idx]   = (2*double(rand())/RAND_MAX - 1)*2*M_PI;
    for (int idx = 0; idx < dim - 1; ++idx) varkappa[idx] = (2*double(rand())/RAND_MAX - 1)*2*M_PI;

    Matrix Umtx = construct_unitary_tmpl<double>(vartheta, varphi, varkappa, dim);

    qgd_free(vartheta);
    qgd_free(varphi);
    qgd_free(varkappa);

    return Umtx;
}


/**
@brief Construct a float64 unitary from explicit parameter arrays.
@param vartheta array of dim*(dim-1)/2 elements
@param varphi   array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return The constructed float64 unitary.
*/
Matrix
Random_Unitary::Construct_Unitary_Matrix( double* vartheta, double* varphi, double* varkappa ) {
    return construct_unitary_tmpl<double>(vartheta, varphi, varkappa, dim);
}


/**
@brief Construct a float32 unitary from explicit parameter arrays (native float32,
       no precision conversion).
@param vartheta array of dim*(dim-1)/2 elements
@param varphi   array of dim*(dim-1)/2 elements
@param varkappa array of dim-1 elements
@return The constructed float32 unitary.
*/
Matrix_float
Random_Unitary::Construct_Unitary_Matrix( float* vartheta, float* varphi, float* varkappa ) {
    return construct_unitary_tmpl<float>(vartheta, varphi, varkappa, dim);
}


/**
@brief Construct a float64 unitary from a packed parameter array.
@param parameters array of (dim+1)*(dim-1) elements (vartheta | varphi | varkappa contiguous)
@return The constructed float64 unitary.
*/
Matrix
Random_Unitary::Construct_Unitary_Matrix( double* parameters ) {
    return construct_unitary_tmpl<double>(
        parameters,
        parameters + int(dim*(dim-1)/2),
        parameters + int(dim*(dim-1)),
        dim );
}





