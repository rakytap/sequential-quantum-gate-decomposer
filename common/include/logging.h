/*brief Header file for a print class in order to control the verbosity levels of output messages. 
*/
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
/*! \file logging.h
    \brief Header file for a class containing basic methods for setting up the verbosity level.
*/

#ifndef PRINTCLASSSQ_H
#define PRINTCLASSSQ_H

#include <string>
#include <iostream>
#include <sstream> 
#include <fstream>
#include <stdio.h> 
#include <ostream>




/**
@brief A class containing basic methods for setting up the verbosity level. 
*/
class logging 
{ 


public:


/// Set the verbosity level of the output messages. 
int verbose; 

/// Set the verbosity level of the individual messages.
int verbose_level;

/// Stringstream input to store the output messages.
std::stringstream ss;

/// Logical variable. Set true to write output messages to  the 'debug.txt' file.
bool debug; 	

 
/**
@brief Call to print output messages in the function of the verbosity level.
@param ssq The stringstream input to store the output messages.
@param verbose_level Integer input. High level means write more to the standart output, 0 means write nothing. The default value is set to 1. 
*/
void print(std::stringstream& ssq, int verbose_level=1); 

};

#endif
