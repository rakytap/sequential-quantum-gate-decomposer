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

@author: Peter Rakyta, Ph.D.
*/
/*! \file N_Qubit_Decomposition_Cost_Function.cpp
    \brief Methods to calculate the cost function of the final optimization problem (supporting parallel computations).
*/

#include "N_Qubit_Decomposition_Cost_Function.h"
//#include <tbb/parallel_for.h>



/**
@brief Call co calculate the cost function during the final optimization process.
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param trace_offset The offset in the first columns from which the "trace" is calculated. In this case Tr(A) = sum_(i-offset=j) A_{ij}
@return Returns with the calculated cost function.
*/
double get_cost_function(Matrix matrix, int trace_offset) {

    int matrix_size = matrix.cols ;
/*
    tbb::combinable<double> priv_partial_cost_functions{[](){return 0;}};
    tbb::parallel_for( tbb::blocked_range<int>(0, matrix_size, 1), functor_cost_fnc( matrix, &priv_partial_cost_functions ));
*/
/*
    //sequential version
    functor_cost_fnc tmp = functor_cost_fnc( matrix, matrix_size, partial_cost_functions, matrix_size );
    #pragma omp parallel for
    for (int idx=0; idx<matrix_size; idx++) {
        tmp(idx);
    }
*/
/*
    // calculate the final cost function
    double cost_function = 0;
    priv_partial_cost_functions.combine_each([&cost_function](double a) {
        cost_function = cost_function + a;
    });
*/

/*
#ifdef USE_AVX


    __m128d trace_128 = _mm_setr_pd(0.0, 0.0);
    double* matrix_data = (double*)matrix.get_data();
    int offset = 2*(matrix.stride+1);

    for (int idx=0; idx<matrix_size; idx++) {
        
        // get the diagonal element
        __m128d element_128 = _mm_load_pd(matrix_data);
        
        // add the diagonal elements to the trace
        trace_128 = _mm_add_pd(trace_128, element_128);

        matrix_data = matrix_data + offset;
    }


    trace_128 = _mm_mul_pd(trace_128, trace_128);    
    double cost_function = std::sqrt(1.0 - (trace_128[0] + trace_128[1])/(matrix_size*matrix_size));

#else

    QGD_Complex16 trace;
    memset( &trace, 0.0, 2*sizeof(double) );
    //trace.real = 0.0;
    //trace.imag = 0.0;

    for (int idx=0; idx<matrix_size; idx++) {
        
        trace.real += matrix[idx*matrix.stride + idx].real;
        trace.imag += matrix[idx*matrix.stride + idx].imag;
    }

    double cost_function = std::sqrt(1.0 - (trace.real*trace.real + trace.imag*trace.imag)/(matrix_size*matrix_size));
#endif
*/


    double trace_real = 0.0;

    if ( trace_offset == 0 ) {

        for (int idx=0; idx<matrix_size; idx++) {
         
            trace_real += matrix[idx*matrix.stride + idx].real;

        }
    }
    else {

        for (int idx=0; idx<matrix_size; idx++) {

            trace_real += matrix[(idx+trace_offset)*matrix.stride + idx].real;

        }

    }

    //double cost_function = std::sqrt(1.0 - trace_real/matrix_size);
    double cost_function = (1.0 - trace_real/matrix_size);

    return cost_function;

}



/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0) and the first correction (index 1).
*/
Matrix_real get_cost_function_with_correction(Matrix matrix, int qbit_num, int trace_offset) {

    Matrix_real ret(1,2);

    // calculate the cost function
    ret[0] = get_cost_function( matrix, trace_offset );



    // calculate teh first correction

    int matrix_size = matrix.cols;

    double trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = col_idx ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }

    }
    else {


        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = (col_idx + trace_offset) ^ qbit_error_mask;
// std::cout << matrix[row_idx*matrix.stride + col_idx].real << " " << row_idx << " " << col_idx << std::endl;
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }


    }

    //double cost_function = std::sqrt(1.0 - trace_real/matrix_size);
    double cost_function = trace_real/matrix_size;

//std::cout << cost_function << std::endl;
//exit(1);

    ret[1] = cost_function;

    return ret;

    

}




/**
@brief Call co calculate the cost function of the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0), the first correction (index 1) and the second correction (index 2).
*/
Matrix_real get_cost_function_with_correction2(Matrix matrix, int qbit_num, int trace_offset) {


    Matrix_real ret(1,3);

    // calculate the cost function
    ret[0] = get_cost_function( matrix, trace_offset );



    // calculate the first correction

    int matrix_size = matrix.cols;

    double trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = col_idx ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }

    }
    else {


        for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

            int qbit_error_mask = 1 << qbit_idx;

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = (col_idx+trace_offset) ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            }
        }


    }

    double cost_function = trace_real/matrix_size;

    ret[1] = cost_function;




    // calculate the second correction

    trace_real = 0.0;

    if ( trace_offset == 0 ) {
        for (int qbit_idx=0; qbit_idx<qbit_num-1; qbit_idx++) {
            for (int qbit_idx2=qbit_idx+1; qbit_idx2<qbit_num; qbit_idx2++) {

                int qbit_error_mask = (1 << qbit_idx) + (1 << qbit_idx2);

                for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                    // determine the row index pair with one bit error at the given qbit_idx
                    int row_idx = col_idx ^ qbit_error_mask;
 
                    trace_real += matrix[row_idx*matrix.stride + col_idx].real;
                }

            }
        }
    }
    else {

        for (int qbit_idx=0; qbit_idx<qbit_num-1; qbit_idx++) {
            for (int qbit_idx2=qbit_idx+1; qbit_idx2<qbit_num; qbit_idx2++) {

                int qbit_error_mask = (1 << qbit_idx) + (1 << qbit_idx2);

                for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                    // determine the row index pair with one bit error at the given qbit_idx
                    int row_idx = (col_idx+trace_offset) ^ qbit_error_mask;
 
                    trace_real += matrix[row_idx*matrix.stride + col_idx].real;
                }

            }
        }


    }

    double cost_function2 = trace_real/matrix_size;

    ret[2] = cost_function2;

    return ret;





}

double get_cost_function_sum_of_squares(Matrix& matrix)
{
    double ret = 0.0;
    for (int rowidx = 0; rowidx < matrix.rows; rowidx++) {
        int baseidx = rowidx*matrix.stride;
        for (int colidx = 0; colidx < matrix.cols; colidx++) {
            if (rowidx == colidx) {
                ret += (matrix[baseidx+colidx].real - 1.0) * (matrix[baseidx+colidx].real - 1.0) + matrix[baseidx+colidx].imag * matrix[baseidx+colidx].imag;
            } else {
                ret += matrix[baseidx+colidx].real * matrix[baseidx+colidx].real + matrix[baseidx+colidx].imag * matrix[baseidx+colidx].imag;
            }
        }
    }
    return ret;
}

Matrix get_deriv_sum_of_squares(Matrix& matrix)
{
    Matrix ret(matrix.rows, matrix.cols);
    for (int rowidx = 0; rowidx < matrix.rows; rowidx++) {
        int baseidx = rowidx*matrix.stride;
        for (int colidx = 0; colidx < matrix.cols; colidx++) {
            if (rowidx == colidx) {
                ret[baseidx+colidx].real = 2 * (matrix[baseidx+colidx].real - 1.0);
                ret[baseidx+colidx].imag = 2 * matrix[baseidx+colidx].imag;
            } else {
                ret[baseidx+colidx].real = 2 * matrix[baseidx+colidx].real;
                ret[baseidx+colidx].imag = 2 * matrix[baseidx+colidx].imag;
            }
        }
    }
    return ret;
}

/**
@brief Call to calculate the real and imaginary parts of the trace
@param matrix The square shaped complex matrix from which the trace is calculated.
@return Returns with the calculated trace.
*/
QGD_Complex16 get_trace(Matrix& matrix){

    int matrix_size = matrix.cols;
    double trace_real=0.0;
    double trace_imag=0.0;
    QGD_Complex16 ret;
    
    for (int idx=0; idx<matrix_size; idx++) {
        
        trace_real += matrix[idx*matrix.stride + idx].real;
        trace_imag += matrix[idx*matrix.stride + idx].imag;

    }
    ret.real = trace_real;
    ret.imag = trace_imag;
    
    return ret;
}

/**
@brief Call co calculate the cost function of the optimization process according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns the cost function
*/
double get_hilbert_schmidt_test(Matrix& matrix){
    
    double d = 1.0/matrix.cols;
    double cost_function = 0.0;
    QGD_Complex16 ret = get_trace(matrix);
    cost_function = 1.0-d*d*(ret.real*ret.real+ret.imag*ret.imag);

    return cost_function;
}

double get_infidelity(Matrix& matrix){
    
    double d = matrix.cols;
    double cost_function = 0.0;
    QGD_Complex16 ret = get_trace(matrix);
    cost_function = 1.0-((ret.real*ret.real+ret.imag*ret.imag)/d+1)/(d+1);

    return cost_function;
}

/**
@brief Call co calculate the Hilbert Schmidt testof the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0-1) and the first correction (index 2-3).
*/
Matrix get_trace_with_correction(Matrix& matrix, int qbit_num) {
    
    Matrix ret(1,2);
    
    QGD_Complex16 trace_tmp = get_trace(matrix);
    
    int matrix_size = matrix.cols;

    double trace_real = 0.0;
    double trace_imag = 0.0;

    for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

        int qbit_error_mask = 1 << qbit_idx;

        for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

            // determine the row index pair with one bit error at the given qbit_idx
            int row_idx = col_idx ^ qbit_error_mask;
 
            trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            trace_imag += matrix[row_idx*matrix.stride + col_idx].imag;
        }
    }
    
    ret[0].real = trace_tmp.real;
    ret[0].imag = trace_tmp.imag;
    ret[1].real = trace_real;
    ret[1].imag = trace_imag;
    
    return ret;
}


/**
@brief Call co calculate the Hilbert Schmidt testof the optimization process, and the first correction to the cost finction according to https://arxiv.org/pdf/2210.09191.pdf
@param matrix The square shaped complex matrix from which the cost function is calculated.
@param qbit_num The number of qubits
@return Returns with the matrix containing the cost function (index 0-1), the first correction (index 2-3) and the second correction (index 4-5).
*/
Matrix get_trace_with_correction2(Matrix& matrix, int qbit_num) {

    Matrix ret(1,3);
    
    QGD_Complex16 trace_tmp = get_trace(matrix);
    
    int matrix_size = matrix.cols;

    double trace_real = 0.0;
    double trace_imag = 0.0;

    for (int qbit_idx=0; qbit_idx<qbit_num; qbit_idx++) {

        int qbit_error_mask = 1 << qbit_idx;

        for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

            // determine the row index pair with one bit error at the given qbit_idx
            int row_idx = col_idx ^ qbit_error_mask;
 
            trace_real += matrix[row_idx*matrix.stride + col_idx].real;
            trace_imag += matrix[row_idx*matrix.stride + col_idx].imag;
        }
    }


    ret[1].real = trace_real;
    ret[1].imag = trace_imag;



    // calculate the second correction

    trace_real = 0.0;
    trace_imag = 0.0;

    for (int qbit_idx=0; qbit_idx<qbit_num-1; qbit_idx++) {
        for (int qbit_idx2=qbit_idx+1; qbit_idx2<qbit_num; qbit_idx2++) {

            int qbit_error_mask = (1 << qbit_idx) + (1 << qbit_idx2);

            for (int col_idx=0; col_idx<matrix_size; col_idx++) {        

                // determine the row index pair with one bit error at the given qbit_idx
                int row_idx = col_idx ^ qbit_error_mask;
 
                trace_real += matrix[row_idx*matrix.stride + col_idx].real;
                trace_imag += matrix[row_idx*matrix.stride + col_idx].imag;
            }

        }
    }


    ret[2].real = trace_real;
    ret[2].imag = trace_imag;
    ret[0].real = trace_tmp.real;
    ret[0].imag = trace_tmp.imag;
    
    return ret;
}

/**
@brief Constructor of the class.
@param matrix_in Arry containing the input matrix
@param matrix_size_in The number rows in the matrix.
@param partial_cost_functions_in Preallocated array storing the calculated partial cost functions.
@param partial_cost_fnc_num_in The number of partial cost function values (equal to the number of distinct submatrix products.)
@return Returns with the instance of the class.
*/
functor_cost_fnc::functor_cost_fnc( Matrix matrix_in, tbb::combinable<double>* partial_cost_functions_in ) {

    matrix = matrix_in;
    data = matrix.get_data();
    partial_cost_functions = partial_cost_functions_in;
}

/**
@brief Operator to calculate the partial cost function derived from the row of the matrix labeled by row_idx
@param r A TBB range labeling the partial cost function to be calculated.
*/
void functor_cost_fnc::operator()( tbb::blocked_range<int> r ) const {

    int matrix_size = matrix.rows;
    double &cost_function_priv = partial_cost_functions->local();

    for ( int row_idx = r.begin(); row_idx != r.end(); row_idx++) {

        if ( row_idx > matrix_size ) {
            std::string err("Error: row idx should be less than the number of roes in the matrix.");
            throw err;
        }

        // getting the corner element
        QGD_Complex16 corner_element = data[0];


        // Calculate the |x|^2 value of the elements of the matrix and summing them up to calculate the partial cost function
        double partial_cost_function = 0;
        int idx_offset = row_idx*matrix_size;
        int idx_max = idx_offset + row_idx;
        for ( int idx=idx_offset; idx<idx_max; idx++ ) {
            partial_cost_function = partial_cost_function + data[idx].real*data[idx].real + data[idx].imag*data[idx].imag;
        }

        int diag_element_idx = row_idx*matrix_size + row_idx;
        double diag_real = data[diag_element_idx].real - corner_element.real;
        double diag_imag = data[diag_element_idx].imag - corner_element.imag;
        partial_cost_function = partial_cost_function + diag_real*diag_real + diag_imag*diag_imag;


        idx_offset = idx_max + 1;
        idx_max = row_idx*matrix_size + matrix_size;
        for ( int idx=idx_offset; idx<idx_max; idx++ ) {
            partial_cost_function = partial_cost_function + data[idx].real*data[idx].real + data[idx].imag*data[idx].imag;
        }

        // storing the calculated partial cost function
        cost_function_priv = cost_function_priv + partial_cost_function;

    }
}




#include <complex.h>
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
#define lapack_int     int
#define lapack_complex_double   double _Complex
extern "C" lapack_complex_double lapack_make_complex_double(double re, double im);
extern "C" lapack_int LAPACKE_zgesvd( int matrix_order, char jobu, char jobvt,
                            lapack_int m, lapack_int n, lapack_complex_double* a,
                            lapack_int lda, double* s, lapack_complex_double* u,
                            lapack_int ldu, lapack_complex_double* vt,
                            lapack_int ldvt, double* superb );
extern "C" lapack_int LAPACKE_zgesdd(int matrix_order, char jobz, lapack_int m, lapack_int n, lapack_complex_double* a,
                          lapack_int lda, double* s, lapack_complex_double* u, lapack_int ldu,
                          lapack_complex_double* vt, lapack_int ldvt);

// Helper: extract bits at positions 'pos' from integer x into a packed integer (LSB order)
static inline int extract_bits(int x, const std::vector<int>& pos) {
    int y = 0, k = 0;
    for (int p : pos) { y |= ((x >> p) & 1) << k; ++k; }
    return y;
}

// Index: row-major 2^n x 2^n
static inline size_t rm_idx(int row, int col, int N) {
    return (size_t)row * (size_t)N + (size_t)col;
}

//https://www.sciencedirect.com/science/article/pii/S0024379518303446
//https://arxiv.org/abs/2007.02490
//https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.062430
//https://arxiv.org/pdf/2111.03132
// Build the (dA*dA) x (dB*dB) OSR matrix M for cut A|B from U (2^n x 2^n), row-major.
// M_{ (a' * dA + a), (b' * dB + b) } = U_{ (a',b'), (a,b) }.
static void build_osr_matrix(const Matrix& U, int n,
                             const std::vector<int>& A, // qubits on A
                             std::vector<lapack_complex_double>& M, int& m_rows, int& m_cols)
{
    std::vector<int> A_sorted = A;
    std::sort(A_sorted.begin(), A_sorted.end());
    std::vector<int> B;
    B.reserve(n - (int)A_sorted.size());
    for (int q = 0; q < n; ++q)
        if (!std::binary_search(A_sorted.begin(), A_sorted.end(), q)) B.push_back(q);

    const int dA = 1 << (int)A_sorted.size();
    const int dB = 1 << (n - (int)A_sorted.size());
    const int N  = 1 << n;

    m_rows = dA * dA;
    m_cols = dB * dB;
    M.resize((size_t)m_rows * (size_t)m_cols);

    // Row-major indexing: U[in + out*N] is element (in, out)
    for (int in = 0; in < N; ++in) {
        const int a  = extract_bits(in, A_sorted) * dA;
        const int b  = extract_bits(in, B) * dB;
        for (int out = 0; out < N; ++out) {
            const int ap = extract_bits(out, A_sorted);
            const int bp = extract_bits(out, B);
            const int r = a + ap;   // row in M
            const int c = b + bp;   // col in M
            const auto& val = U[(size_t)in + (size_t)out * (size_t)N];
            M[rm_idx(r, c, m_cols)] = lapack_make_complex_double(val.real, val.imag);
        }
    }
}

static Matrix reverse_osr_matrix(const Matrix& M, int n,
                             const std::vector<int>& A) // qubits on A
{
    std::vector<int> A_sorted = A;
    std::sort(A_sorted.begin(), A_sorted.end());
    std::vector<int> B;
    B.reserve(n - (int)A_sorted.size());
    for (int q = 0; q < n; ++q)
        if (!std::binary_search(A_sorted.begin(), A_sorted.end(), q)) B.push_back(q);

    const int dA = 1 << (int)A_sorted.size();
    const int dB = 1 << (n - (int)A_sorted.size());
    const int N  = 1 << n;

    int m_rows = dA * dA;
    int m_cols = dB * dB;

    Matrix U(N, N);

    // Row-major indexing: U[in + out*N] is element (in, out)
    for (int in = 0; in < N; ++in) {
        const int a  = extract_bits(in, A_sorted) * dA;
        const int b  = extract_bits(in, B) * dB;
        for (int out = 0; out < N; ++out) {
            const int ap = extract_bits(out, A_sorted);
            const int bp = extract_bits(out, B);
            const int r = a + ap;   // row in M
            const int c = b + bp;   // col in M
            U[(size_t)in + (size_t)out * (size_t)N] = M[rm_idx(r, c, m_cols)];
        }
    }
    return U;
}

static int osr(std::vector<lapack_complex_double>& A, int m_rows, int m_cols, double Fnorm, std::vector<double>& S)
{
    S.resize(std::min(m_rows, m_cols));
    std::vector<double> superb(std::max(1, std::min(m_rows, m_cols) - 1));  // REQUIRED for complex *gesvd
    // We don’t need U/V; job='N' for economy; gesvd is fine too.
    int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR,
                              'N','N',
                              m_rows, m_cols,
                              A.data(), m_cols,
                              S.data(),
                              nullptr, 1,
                              nullptr, 1,
                              superb.data());
    if (info != 0) {
        throw std::runtime_error("zgesdd failed, info=" + std::to_string(info));
    }
    for (double& s : S) s /= Fnorm; // guard against tiny negative singular values
    return info;
}

// base-2 logarithm, rounding down
static inline uint32_t lg_down(uint32_t v) {
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
                                            r |= (v >> 1);
    return r;
}

// base-2 logarithm, rounding up
static inline uint32_t lg_up(uint32_t x) {
    return x <= 1 ? 0 : lg_down(x - 1) + 1;
}

// Numerical rank via LAPACKE_zgesdd (SVD)
static std::tuple<int, double, double> numerical_rank_osr(std::vector<lapack_complex_double>& M, int m_rows, int m_cols, double Fnorm, double tol)
{
    std::vector<double> S;
    int info = osr(M, m_rows, m_cols, Fnorm, S);
    if (info != 0) return std::make_tuple(0, 0.0, 1.0); // fall back safely

    //std::copy(S.begin(), S.end(), std::ostream_iterator<double>(std::cout, " ")); std::cout << std::endl;
    int rnk = 0;
    for (double s : S) if (s > S[0]*tol) ++rnk;
    double s1norm = S[1]/S[0];
    return std::tuple<int, double, double>(lg_up(rnk), std::isnan(s1norm) ? INFINITY : s1norm, S[0]);
}

// Public: operator-Schmidt rank across cut A|B
std::tuple<int, double, double> operator_schmidt_rank(const Matrix& U, int n,
                          const std::vector<int>& A_qubits,
                          double Fnorm, double tol = 1e-10)
{
    std::vector<lapack_complex_double> M;
    int mr=0, mc=0;
    build_osr_matrix(U, n, A_qubits, M, mr, mc);
    return numerical_rank_osr(M, mr, mc, Fnorm, tol);
}

#include <vector>
#include <algorithm>

// Generate all k-combinations of {0,1,...,n-1} of size r
static void combinations_recursive(int n, int r, int start,
                                   std::vector<int>& current,
                                   std::vector<std::vector<int>>& out)
{
    if ((int)current.size() == r) {
        out.push_back(current);
        return;
    }
    for (int i = start; i < n; ++i) {
        current.push_back(i);
        combinations_recursive(n, r, i + 1, current, out);
        current.pop_back();
    }
}

// Return all nontrivial unordered bipartitions (no complements)
std::vector<std::vector<int>> unique_cuts(int n)
{
    std::vector<std::vector<int>> cuts;
    if (n <= 1) return cuts;

    for (int r = 1; r <= n / 2; ++r) {
        std::vector<std::vector<int>> combs;
        std::vector<int> current;
        combinations_recursive(n, r, 0, current, combs);

        for (auto& S : combs) {
            if (r < n - r) {
                cuts.push_back(S);
            } else { // r == n - r (only for even n)
                std::vector<int> comp;
                for (int q = 0; q < n; ++q)
                    if (std::find(S.begin(), S.end(), q) == S.end())
                        comp.push_back(q);
                // lexicographic tie-break: keep only smaller one
                if (S < comp)
                    cuts.push_back(S);
            }
        }
    }
    return cuts;
}

double get_osr_entanglement_test(Matrix& matrix) {
    //double hscost = get_hilbert_schmidt_test(matrix);
    int qbit_num = lg_down(matrix.rows);
    const auto& cuts = unique_cuts(qbit_num);
    double cost = 0.0;
    double Fnorm = std::sqrt(matrix.rows);
    for (const auto& cut : cuts) {
        auto [rnk, s1norm, s0] = operator_schmidt_rank(matrix, qbit_num, cut, Fnorm);
        cost += 1.0 - s0 * s0;
        //printf("%f ", s0);
    }
    //printf(" %f\n", cost / cuts.size());
    return cost / cuts.size();
}

using OSRTriplet = std::tuple<double, std::vector<std::complex<double>>, std::vector<std::complex<double>>>;

// Build M with build_osr_matrix, then SVD (econ) and grab top triplet.
static OSRTriplet top_singular_triplet_for_cut(
    const Matrix& U, // (N x N), row-major, N = 1<<q
    int q,                  // number of qubits
    const std::vector<int>& A,  // qubits on side A
    double Fnorm            // e.g., sqrt(N)
){
    // 1) Build M for this cut
    std::vector<lapack_complex_double> M;
    int m_rows = 0, m_cols = 0;
    build_osr_matrix(U, q, A, M, m_rows, m_cols); // your function (no normalization)

    const int k = std::min(m_rows, m_cols);

    // 2) Allocate outputs for SVD (econ)
    std::vector<double> S(k);
    std::vector<lapack_complex_double> Umat((size_t)m_rows * (size_t)k); // m x k
    std::vector<lapack_complex_double> VTmat((size_t)k * (size_t)m_cols); // k x n
    //std::vector<double> superb(std::max(1, k - 1));

    // 3) SVD: M = U * diag(S) * VT  (VT = V^H)
    // Row-major API handles leading dims as col counts.
    int info = LAPACKE_zgesdd(
        LAPACK_ROW_MAJOR,
        'S',           // econ U, VT
        m_rows, m_cols,
        M.data(), m_cols,   // a, lda (row-major -> lda = n)
        S.data(),
        Umat.data(), k,    // U (m x k), ldu = k (row-major)
        VTmat.data(), m_cols     // VT (k x n), ldvt = n
    );
    if (info != 0) {
        throw std::runtime_error("zgesdd failed, info=" + std::to_string(info));
    }

    // 4) Extract top singular vectors: u0 = first column of Umat, v0 = first right vec
    // VT stores V^H, so row 0 of VT is v0^H -> take conjugate to get v0.

    std::vector<std::complex<double>> u0;
    std::vector<std::complex<double>> v0;
    u0.resize(m_rows);
    for (int i = 0; i < m_rows; ++i) {
        // first column: U(i,0). In row-major, columns are strided by 'k'
        lapack_complex_double z = Umat[(size_t)i * (size_t)k + 0];
        u0[i] = { creal(z), cimag(z) };
    }

    v0.resize(m_cols);
    for (int j = 0; j < m_cols; ++j) {
        // VT(0,j) = conj(v0(j))  -> v0(j) = conj(VT(0,j))
        lapack_complex_double z = VTmat[0 * (size_t)m_cols + (size_t)j];
        v0[j] = {  creal(z), -cimag(z) }; // conjugate
    }
    // normalized singular value
    return OSRTriplet(S[0] / Fnorm, u0, v0);
}

Matrix get_deriv_osr_entanglement(Matrix &matrix) {
    int qbit_num = lg_down(matrix.rows);
    const auto& cuts = unique_cuts(qbit_num);
    double Fnorm = std::sqrt(matrix.rows);
    Matrix deriv(matrix.rows, matrix.cols);
    std::fill(deriv.data, deriv.data+deriv.size(), QGD_Complex16{0.0, 0.0});
    // Compute the derivative of the OSR entanglement cost function
    for (const auto& cut : cuts) {
        // 1) top singular triplet on the normalized reshape M_c
        auto [s0, u0, v0] = top_singular_triplet_for_cut(matrix, qbit_num, cut, Fnorm);

        // 2) form dL/dM_c = -2 s0 u0 v0^†
        const int m = static_cast<int>(u0.size());
        const int n = static_cast<int>(v0.size());
        Matrix dLdM(m, n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                auto val = -2.0 * s0 / Fnorm * u0[i] * std::conj(v0[j]);
                dLdM[i*dLdM.stride+j].real = val.real();
                dLdM[i*dLdM.stride+j].imag = val.imag();
            }
        }

        // 3) map back: unreshape_c(dL/dM_c) to dLdU
        Matrix contrib = reverse_osr_matrix(dLdM, qbit_num, cut);
        for (int r = 0; r < matrix.rows; ++r) {
            int off = r * deriv.stride;
            for (int c = 0; c < matrix.cols; ++c) {
                deriv[off + c].real += contrib[off + c].real;
                deriv[off + c].imag += contrib[off + c].imag;
            }
        }
    }
    // average gradient too
    for (int r = 0; r < matrix.rows; ++r) {
        int off = r * deriv.stride;
        for (int c = 0; c < matrix.cols; ++c) {
            deriv[off + c].real /= static_cast<double>(cuts.size());
            deriv[off + c].imag /= static_cast<double>(cuts.size());
        }
    }
    return deriv;
}

// Compute grad component = Re Tr( A^† B ) for A = dL/dU, B = dU/dθ
// A and B are (rows x cols) with row-major leading dimension.
double real_trace_conj_dot(Matrix& A, Matrix& B)
{
    double acc = 0.0;
    for (int r = 0; r < A.rows; ++r) {
        int offs = r * A.stride;
        for (int c = 0; c < A.cols; ++c) {
            acc += A[offs + c].real * B[offs + c].real + A[offs + c].imag * B[offs + c].imag;
        }
    }
    return acc; // Re Tr(A^† B)
}
