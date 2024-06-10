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
/*! \file logging.cpp
    \brief Class containing basic methods for setting up the verbosity level.
*/

#include "logging.h"
 


/** Nullary constructor of the class
*/




logging::logging() {

	// Number of the verbosity level in order to control the amount of the output messages. If this value is higher or equal than the local verbosity levels belongs to the messages, than the messages 	will be seen on the standart output screen.  
	verbose=3;



	// Logical variable. Set true to write output messages to a user defined file. 
	
	debug=false;
	

}



/**
@brief Call to print output messages in the function of the verbosity level.
@param sstream The stringstream input to store the output messages.
@param verbose_level Integer input. High level means write more to the standart output, 0 means write nothing. The default value is set to 1. 
*/
void logging::print(const std::stringstream& sstream, int verbose_level) const {

 
	if (debug) { 
		
		std::ofstream debug_file;         
		debug_file.open(debugfile_name, std::ios_base::app);
 		debug_file << sstream.str(); 
		debug_file.close();		
        }
	

	if (verbose_level<=verbose) { 
        	std::cout << sstream.str();
                fflush(stdout);   
        }

	


	
}




/**
@brief Call to set the verbose attribute.
@param verbose_in Integer variable. Set the number to specify the verbosity level for output messages.
*/
void logging::set_verbose( int verbose_in ) {

    verbose = verbose_in;

}

/**
@brief Call to set the debugfile name.
@param debugfile String variable. Set the debugfile name. 
*/
void logging::set_debugfile(std::string debugfile ) {
    
    debugfile_name = debugfile;

    if (debugfile_name!="<NULL>")
		debug=true;

    if (debugfile_name.c_str()) std::remove(debugfile_name.c_str());

}

