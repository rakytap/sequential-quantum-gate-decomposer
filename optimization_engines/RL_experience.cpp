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
/*! \file RL_experience.cpp
    \brief A class for RL_experience 
*/

#include "RL_experience.h"
#include "tbb/tbb.h"


#include <cfloat>	

/** Nullary constructor of the class
@return An instance of the class
*/
RL_experience::RL_experience() {

    gates = NULL;
    iteration_num = 0;

    exploration_rate = 0.25;

    reset();
	

}


/** Contructor of the class
@brief Constructor of the class.
@param ???????????????????????????????
@param ???????????????????????????????
@param ???????????????????????????????
@return An instance of the class
*/
RL_experience::RL_experience( Gates_block* gates_in, unsigned long long iteration_num_in ) {


    gates = gates_in;

    iteration_num = iteration_num_in;


    exploration_rate = 0.25;


    reset();


}


/**
@brief Copy constructor.
@param experience An instance of class
@return Returns with the instance of the class.
*/
RL_experience::RL_experience(const RL_experience& experience ) {

    gates = experience.gates;
    iteration_num = experience.iteration_num;

    parameter_num = gates->get_parameter_num();

    parameter_probs    = experience.parameter_probs.copy();
    parameter_counts   = experience.parameter_counts.copy();
    total_counts       = experience.total_counts.copy();
    total_counts_probs = experience.total_counts_probs.copy();

    exploration_rate   = experience.exploration_rate;

    history            = experience.history;

}


/**
@brief Assignment operator.
@param experience An instance of class
@return Returns with the instance of the class.
*/
RL_experience& RL_experience::operator= (const RL_experience& experience ) {

    gates = experience.gates;
    iteration_num = experience.iteration_num;

    parameter_num = gates->get_parameter_num();


    parameter_probs  = experience.parameter_probs.copy();
    parameter_counts = experience.parameter_counts.copy();
    total_counts     = experience.total_counts.copy();
    total_counts_probs = experience.total_counts_probs.copy();

    exploration_rate   = experience.exploration_rate;

    history            = experience.history;


    return *this;
}




/**
@brief Call to make a copy of the current instance.
@return Returns with the instance of the class.
*/
RL_experience RL_experience::copy() {

    RL_experience experience( gates, iteration_num );



    experience.parameter_probs  = parameter_probs.copy();
    experience.parameter_counts = parameter_counts.copy();

    experience.exploration_rate   = exploration_rate;

    experience.history            = history; 

    return experience;
}



/**
@brief Destructor of the class
*/

RL_experience::~RL_experience() {

}



/**
@brief ?????????????
*/
void RL_experience::reset() {

    if (gates == NULL ) {
        parameter_num = 0;

        parameter_probs  = Matrix_real( 0, 0 );
        parameter_counts = matrix_base<int>( 0, 0 );
        total_counts     = matrix_base<int>( 0 ,0 );
        total_counts_probs = matrix_base<unsigned long long>( 0, 0);
        return;
    }

    parameter_num = gates->get_parameter_num();

    parameter_probs  = Matrix_real( parameter_num, parameter_num );
    parameter_counts = matrix_base<int>( parameter_num, parameter_num );
    total_counts     = matrix_base<int>( parameter_num, 1 );
    total_counts_probs = matrix_base<unsigned long long>( parameter_num, 1);


    // initial set of the counts and probabilities --- equally weighted probabilities and counts


    double init_prob      = 1.0/parameter_num;
    for ( int row_idx=0; row_idx<parameter_num; row_idx++ ) {

        total_counts_probs[ row_idx ] = parameter_num*20;

        for ( int col_idx=0; col_idx<parameter_num; col_idx++ ) {
            parameter_probs[row_idx*parameter_probs.stride + col_idx] = init_prob;
        }
    }

    memset( parameter_counts.get_data(), 0, parameter_counts.size()*sizeof( int ) );
    memset( total_counts.get_data(), 0, total_counts.size()*sizeof( int ) );

///////////
    history.clear();
}



/** 
@brief Call to draw the next index
@param curent_index The current index after which the next one is selected
@return Returns with the selected index
*/
int RL_experience::draw( const int& curent_index, std::mt19937& gen ) {


    // decide whether explore the action space or draw from trained probabilities
    std::uniform_real_distribution<> distrib_to_choose(0.0, 1.0); 
    double random_num = distrib_to_choose( gen );

    int selected_idx = 0;

    if ( random_num > exploration_rate ) {

        // draw from trained probabilities

        // probabilities in the current row of the table
        Matrix_real current_probs = Matrix_real( parameter_probs.get_data() + parameter_probs.stride*curent_index, 1, parameter_num );


        random_num = distrib_to_choose( gen );

        double prob_tmp = 0.0;
        for ( int idx=0; idx<parameter_num; idx++ ) {

            if ( prob_tmp >= random_num ) {
                selected_idx = idx;
                break;
            }

            prob_tmp += current_probs[idx];

        }

    }
    else {
        // random selection

        std::uniform_int_distribution<> distrib_int(0, parameter_num-1); 
        selected_idx = distrib_int( gen );
    }

    // update the selection counts table

    parameter_counts[ curent_index*parameter_counts.stride + selected_idx]++;
    total_counts[selected_idx]++;

    //history.push_back( selected_idx );
    
//std::cout << "uuuuuuuuu " << parameter_counts[ curent_index*parameter_counts.stride + selected_idx] << std::endl;
    return selected_idx;



}



/** 
@brief Call to update the trained probabilities and reset the counts
*/
void RL_experience::update_probs( ) {


    for ( int row_idx=0; row_idx<parameter_num; row_idx++ ) {


        double prob_sum = 0.0;


        for ( int col_idx=0; col_idx<parameter_num; col_idx++ ) {

            unsigned long long counts_loc_old = (unsigned long long) (parameter_probs[row_idx*parameter_counts.stride + col_idx] * total_counts_probs[ row_idx ]);
            unsigned long long counts_loc_new = (unsigned long long) parameter_counts[row_idx*parameter_counts.stride + col_idx];


            counts_loc_old += counts_loc_new;

            total_counts_probs[row_idx] += (unsigned long long) total_counts[row_idx];

            double prob_new = (double)counts_loc_old/total_counts_probs[row_idx];


//if ( parameter_counts[row_idx*parameter_counts.stride + col_idx] > 0 ) {
//std::cout <<    prob_new << " " << counts_loc_old << " "  << counts_loc_new << " " << total_counts_probs[row_idx] << " " << total_counts[row_idx] << std::endl;         

//}

            parameter_probs[row_idx*parameter_counts.stride + col_idx] = prob_new;
            prob_sum += prob_new;
    

            parameter_counts[row_idx*parameter_counts.stride + col_idx] = 0;

        }  

        total_counts[row_idx] = 0;

        // renormalize the probabilities
        for ( int col_idx=0; col_idx<parameter_num; col_idx++ ) {
            parameter_probs[row_idx*parameter_counts.stride + col_idx] = parameter_probs[row_idx*parameter_counts.stride + col_idx]/prob_sum;
        }  


    }


//parameter_probs.print_matrix();

}





/**
@brief ???????????
@param ???????????
*/
void 
RL_experience::export_probabilities(){

    FILE* pFile;
std::string filename = "probabilities.bin";

    const char* c_filename = filename.c_str();
    pFile = fopen(c_filename, "wb");
    if (pFile==NULL) {
        fputs ("File error",stderr); 
        std::string error("Cannot open file.");
        throw error;
    }


    fwrite(&iteration_num, sizeof(unsigned long long), 1, pFile);
    fwrite(&parameter_num, sizeof(int), 1, pFile);

    int element_num = parameter_probs.size();
 
    fwrite( &element_num, sizeof(int), 1, pFile);
    fwrite(parameter_probs.get_data(), sizeof(double), element_num, pFile);

    element_num = total_counts_probs.size();
    fwrite( &element_num, sizeof(int), 1, pFile);
    fwrite(total_counts_probs.get_data(), sizeof(unsigned long long), element_num, pFile);



    fclose(pFile);
    return;
}



/**
@brief ???????????
@param ???????????
*/
void 
RL_experience::import_probabilities(){

    FILE* pFile;
std::string filename = "probabilities.bin";

    const char* c_filename = filename.c_str();
    pFile = fopen(c_filename, "rb");
    if (pFile==NULL) {
        fputs ("File error",stderr); 
        std::string error("Cannot open file.");
        throw error;
    }

    fread(&iteration_num, sizeof(unsigned long long), 1, pFile);
    fread(&parameter_num, sizeof(int), 1, pFile);

    int element_num;
 
    fread( &element_num, sizeof(int), 1, pFile);
    parameter_probs = Matrix_real( element_num, 1 );
    fread(parameter_probs.get_data(), sizeof(double), element_num, pFile);

    fread( &element_num, sizeof(int), 1, pFile);
    total_counts_probs = matrix_base<unsigned long long>( element_num, 1 );
    fread(total_counts_probs.get_data(), sizeof(unsigned long long), element_num, pFile);


    fclose(pFile);
    return;
}


