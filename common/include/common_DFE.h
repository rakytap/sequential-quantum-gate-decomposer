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
/*! \file common_DFE.h
    \brief Header file for DFE support
*/

#ifndef common_DFE_H
#define common_DFE_H


#include <omp.h>
#include "QGDTypes.h"
#include "dot.h"


#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <sstream>

extern "C"
{


/**
 * \brief ???????????
 * 
 */
typedef struct {
  float real;
  float imag;
} Complex8;



/**
@brief ????????????
@return ??????????
*/
int calcqgdKernelDFE(size_t dim, DFEgate_kernel_type* gates, int gatesNum);

/**
@brief ????????????
@return ??????????
*/
int load2LMEM( Complex8* data, size_t dim );

/**
@brief ????????????
@return ??????????
*/
void releive_DFE();

/**
@brief ????????????
@return ??????????
*/
int initialize_DFE();

/**
 * \brief ???????????
 * 
 */
int downloadFromLMEM( Complex8** data, size_t dim );


/**
@brief ????????????
@return ??????????
*/
void uploadMatrix2DFE( Matrix& input );


/**
@brief ????????????
@return ??????????
*/
void DownloadMatrixFromDFE( std::vector<Matrix>& output_vec );

}


#endif
