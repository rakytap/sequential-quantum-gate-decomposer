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
/*! \file mpi_base.h
    \brief Header file for a class containing data on the current MPI process
*/

#ifndef MPIBASE_H
#define MPIBASE_H

#include <iostream>


#ifdef __MPI__
#include <mpi.h>
#endif // MPI	

/**
@brief A class containing data on the current MPI process
*/
class mpi_base {

protected:


#ifdef __MPI__
    /// The number of processes
    int world_size;
    /// The rank of the MPI process
    int current_rank;
#endif


public:

    /** Nullary constructor of the class
    @return An instance of the class
    */
    mpi_base();
 

};





#endif
