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
/*! \file CRX.h
    \brief Header file for a class representing a controlled X rotation gate.
*/

#ifndef CRX_H
#define CRX_H

#include "matrix.h"
#include "RX.h"
#define _USE_MATH_DEFINES
#include <math.h>


/**
@brief A class representing a controlled RX gate.
*/
class CRX: public RX {

protected:


public:

/**
@brief Nullary constructor of the class.
*/
CRX();


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the gate.
@param target_qbit_in The 0<=ID<qbit_num of the target qubit.
@param control_qbit_in The identification number of the control qubit. (0 <= control_qbit <= qbit_num-1)
*/
CRX(int qbit_num_in, int target_qbit_in, int control_qbit_in);

/**
@brief Destructor of the class
*/
virtual ~CRX();


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
virtual CRX* clone() override;

};

#endif //CRX_H