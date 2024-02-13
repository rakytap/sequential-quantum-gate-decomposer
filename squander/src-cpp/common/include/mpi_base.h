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
