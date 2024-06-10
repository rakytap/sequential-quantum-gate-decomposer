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
/*! \file logging.h
    \brief Header file for a class containing basic methods for setting up the verbosity level.
*/

#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <iostream>
#include <sstream> 
#include <fstream>


#ifdef __MPI__
#include "mpi_base.h"
#endif


/**
@brief A class containing basic methods for setting up the verbosity level. 
*/
#ifdef __MPI__
class logging : public mpi_base {
#else
class logging {
#endif

public:


    /// Set the verbosity level of the output messages. 
    int verbose; 

    /// Logical variable. Set true to write output messages to  the 'debug.txt' file.
    bool debug; 	

    /// String variable. Set the debug file name. 
    std::string debugfile_name;


    /** Nullary constructor of the class
    @return An instance of the class
    */
    logging();
 
    /**
    @brief Call to print output messages in the function of the verbosity level.
    @param sstream The stringstream input to store the output messages.
    @param verbose_level Integer input. High level means write more to the standart output, 0 means write nothing. The default value is set to 1. 
    */
    void print(const std::stringstream& sstream, int verbose_level=1) const; 




    /**
    @brief Call to set the verbose attribute.
    @param verbose_in Integer variable. Set the number to specify the verbosity level for output messages.
    */
    void set_verbose( int verbose_in );


    /**
    @brief Call to set the debugfile name.
    @param debugfile String variable. Set the debugfile name. 
    */
    void set_debugfile( std::string debugfile );



};





#endif
