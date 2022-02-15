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
/*! \file N_Qubit_Decomposition.cpp
    \brief Base class to determine the decomposition of a unitary into a sequence of two-qubit and one-qubit gate gates.
    This class contains the non-template implementation of the decomposition class
*/

#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"


//setting local_verbose_parameter 
std::stringstream ss;
int verbose_level;
char bufferprint [100];
int bufferprintf;


/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
N_Qubit_Decomposition_custom::N_Qubit_Decomposition_custom() : N_Qubit_Decomposition_Base() {

    // initialize custom gate structure
    gate_structure = NULL;

    iter_max = 10000;
    gradient_threshold = 1e-8;
}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@return An instance of the class
*/
N_Qubit_Decomposition_custom::N_Qubit_Decomposition_custom( Matrix Umtx_in, int qbit_num_in, bool optimize_layer_num_in, guess_type initial_guess_in= CLOSE_TO_ZERO ) : N_Qubit_Decomposition_Base(Umtx_in, qbit_num_in, optimize_layer_num_in, initial_guess_in) {

    // initialize custom gate structure
    gate_structure = NULL;

    iter_max = 10000;
    gradient_threshold = 1e-8;

}



/**
@brief Destructor of the class
*/
N_Qubit_Decomposition_custom::~N_Qubit_Decomposition_custom() {

    if ( gate_structure != NULL ) {
        // release custom gate structure
        delete gate_structure;
    }

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
N_Qubit_Decomposition_custom::start_decomposition(bool prepare_export) {



    if (verbose) {

	verbose_level=1;
	bufferprintf=sprintf (bufferprint,"***************************************************************\n");
	ss << bufferprint << std::endl;
	bufferprintf=sprintf (bufferprint,"Starting to disentangle %d-qubit matrix via custom gate structure\n", qbit_num);
	ss << bufferprint << std::endl;
	bufferprintf=sprintf (bufferprint,"***************************************************************\n\n\n");
	ss << bufferprint << std::endl;
	logging::printnewsq(ss,verbose_level);	    	
	ss.str("");

    }

    // temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    //measure the time for the decompositin
    tbb::tick_count start_time = tbb::tick_count::now();

    // setting the gate structure for optimization
    add_gate_layers();
//std::cout << optimization_problem( optimized_parameters_mtx.get_data() ) << std::endl;
//tbb::task_arena ta(16);
//ta.execute([&]() {
    // final tuning of the decomposition parameters
    final_optimization();
//});   



    // prepare gates to export
    if (prepare_export) {
        prepare_gates_to_export();
    }

    // calculating the final error of the decomposition
    Matrix matrix_decomposed = get_transformed_matrix(optimized_parameters_mtx, gates.begin(), gates.size(), Umtx );
    calc_decomposition_error( matrix_decomposed );


    // get the number of gates used in the decomposition
    gates_num gates_num = get_gate_nums();

    if (verbose) {

	verbose_level=1;
    	ss << "In the decomposition with error = " << decomposition_error << " were used " << layer_num << " gates with:" << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");

	verbose_level=1;
      
        if ( gates_num.u3>0 ) {ss << gates_num.u3 << " U3 opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.rx>0 ) {ss << gates_num.rx << " RX opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.ry>0 ) {ss << gates_num.ry << " RY opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.rz>0 ) {ss << gates_num.rz << " RZ opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.cnot>0 ){ss << gates_num.cnot << " CNOT opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.cz>0 ) {ss << gates_num.cz << " CZ opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.ch>0 ) {ss << gates_num.ch << " CH opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");}

        if ( gates_num.x>0 ) {ss << gates_num.x << " X opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");} 

        if ( gates_num.sx>0 ) {ss << gates_num.sx << " SX opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");} 
 
        if ( gates_num.syc>0 ){ss << gates_num.syc << " Sycamore opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");} 

        if ( gates_num.un>0 ){ss << gates_num.un << " UN opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");} 

        if ( gates_num.adap>0 ){ss << gates_num.adap << " Adaptive opeartions," << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");} 

        std::cout << std::endl;
        tbb::tick_count current_time = tbb::tick_count::now();

	ss << "--- In total " << (current_time - start_time).seconds() << " seconds elapsed during the decomposition ---" << std::endl;
    	logging::printnewsq(ss,verbose_level);	    	
    	ss.str("");


       }

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif

}


/**
@brief Call to add further layer to the gate structure used in the subdecomposition.
*/
void 
N_Qubit_Decomposition_custom::add_gate_layers() {

    release_gates();

    //////////////////////////////////////
    // add custom gate structure

    // the  number of succeeding identical layers in the subdecomposition
    int identical_blocks_loc;
    try {
        identical_blocks_loc = identical_blocks[qbit_num];
        if (identical_blocks_loc==0) {
            identical_blocks_loc = 1;
        }
    }
    catch (...) {
        identical_blocks_loc=1;
    }

    // get the list of sub-blocks in the custom gate structure
    std::vector<Gate*> gates = gate_structure->get_gates();

    for (std::vector<Gate*>::iterator gates_it = gates.begin(); gates_it != gates.end(); gates_it++ ) {

        // The current gate
        Gate* gate = *gates_it;

        for (int idx=0;  idx<identical_blocks_loc; idx++) {
            if (gate->get_type() == CNOT_OPERATION ) {
                CNOT* cnot_gate = static_cast<CNOT*>( gate );
                add_gate_to_end( (Gate*)cnot_gate->clone() );
            }
            else if (gate->get_type() == CZ_OPERATION ) {
                CZ* cz_gate = static_cast<CZ*>( gate );
                add_gate_to_end( (Gate*)cz_gate->clone() );
            }
            else if (gate->get_type() == CH_OPERATION ) {
                CH* ch_gate = static_cast<CH*>( gate );
                add_gate_to_end( (Gate*)ch_gate->clone() );
            }
            else if (gate->get_type() == SYC_OPERATION ) {
                SYC* syc_gate = static_cast<SYC*>( gate );
                add_gate_to_end( (Gate*)syc_gate->clone() );
            }
            else if (gate->get_type() == GENERAL_OPERATION ) {
                add_gate_to_end( gate->clone() );
            }
            else if (gate->get_type() == U3_OPERATION ) {
                U3* u3_gate = static_cast<U3*>( gate );
                add_gate_to_end( (Gate*)u3_gate->clone() );
            }
            else if (gate->get_type() == RX_OPERATION ) {
                RX* rx_gate = static_cast<RX*>( gate );
                add_gate_to_end( (Gate*)rx_gate->clone() );
            }
            else if (gate->get_type() == RY_OPERATION ) {
                RY* ry_gate = static_cast<RY*>( gate );
                add_gate_to_end( (Gate*)ry_gate->clone() );
            }
            else if (gate->get_type() == CRY_OPERATION ) {
                CRY* cry_gate = static_cast<CRY*>( gate );
                add_gate_to_end( (Gate*)cry_gate->clone() );
            }
            else if (gate->get_type() == RZ_OPERATION ) {
                RZ* rz_gate = static_cast<RZ*>( gate );
                add_gate_to_end( (Gate*)rz_gate->clone() );
            }
            else if (gate->get_type() == X_OPERATION ) {
                X* x_gate = static_cast<X*>( gate );
                add_gate_to_end( (Gate*)x_gate->clone() );
            }
            else if (gate->get_type() == SX_OPERATION ) {
                SX* sx_gate = static_cast<SX*>( gate );
                add_gate_to_end( (Gate*)sx_gate->clone() );
            }
            else if (gate->get_type() == UN_OPERATION ) {
                UN* un_gate = static_cast<UN*>( gate );
                add_gate_to_end( (Gate*)un_gate->clone() );
            }
            else if (gate->get_type() == ON_OPERATION ) {
                ON* on_gate = static_cast<ON*>( gate );
                add_gate_to_end( (Gate*)on_gate->clone() );
            }
            else if (gate->get_type() == COMPOSITE_OPERATION ) {
                Composite* com_gate = static_cast<Composite*>( gate );
                add_gate_to_end( (Gate*)com_gate->clone() );
            }
            else if (gate->get_type() == ADAPTIVE_OPERATION ) {
                Adaptive* ad_gate = static_cast<Adaptive*>( gate );
                add_gate_to_end( (Gate*)ad_gate->clone() );
            }
            else if (gate->get_type() == BLOCK_OPERATION ) {
                Gates_block* block_gate = static_cast<Gates_block*>( gate );
                add_gate_to_end( (Gate*)block_gate->clone() );
            }

        }
    }


}





/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param gate_structure An <int, Gates_block*> map containing the gate structure used in the individual subdecomposition (default is used, if a gate structure for specific subdecomposition is missing).
*/
void N_Qubit_Decomposition_custom::set_custom_gate_structure( Gates_block* gate_structure_in ) {

    gate_structure = gate_structure_in->clone();

}



