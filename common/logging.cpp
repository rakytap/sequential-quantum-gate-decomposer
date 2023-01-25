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

