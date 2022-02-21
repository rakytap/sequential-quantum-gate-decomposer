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
/*! \file Decomposition_Base.cpp
    \brief Class containing basic methods for setting up the verbosity level.
*/

#include "logging.h"
 

/**
@brief Call to print output messages in the function of the verbosity level.
@param ssq The stringstream input to store the output messages.
@param verbose_level Integer input. High level means write more to the standart output, 0 means write nothing. The default value is set to 1. 
*/
void logging::print(std::stringstream& ssq, int verbose_level) 

{


// Number of the verbosity level in order to control the amount of the output messages. If this value is higher or equal than the local verbosity levels belongs to the messages, than the messages will be seen on the standart output screen.  
	verbose=3;


// Logical variable. Set true to write output messages to the 'debug.txt' file.
	debug=true;


	if (debug==true) { 

		std::ofstream debug_file; 
		debug_file.open("debug.txt", std::ios_base::app);
 		debug_file << ssq.str() << '\n'; 
		debug_file.close();
		
        }
	

	if (verbose_level>=verbose) { 
        	std::cout << ssq.str() <<'\n';
		
        }

	


	
}

