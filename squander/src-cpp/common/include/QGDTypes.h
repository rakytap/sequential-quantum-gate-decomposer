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
/*! \file QGDTypes.h
    \brief Custom types for the SQUANDER package
*/

#ifndef QGDTypes_H
#define QGDTypes_H

#include <assert.h>
#include <cstddef>

// platform independent types
#include <stdint.h>

#ifndef CACHELINE
#define CACHELINE 64
#endif


/// @brief Structure type representing complex numbers in the SQUANDER package
struct QGD_Complex16 {
  /// the real part of a complex number
  double real;
  /// the imaginary part of a complex number
  double imag;
};

/// @brief Structure type conatining numbers of gates.
struct gates_num {
  /// The number of U3 gates
  int u3;
  /// The number of RX gates
  int rx;
  /// The number of RY gates
  int ry;
  /// The number of CRY gates
  int cry;
  /// The number of RZ gates
  int rz;
  /// The number of CNOT gates
  int cnot;
  /// The number of CZ gates
  int cz;
  /// The number of CH (i.e. controlled Hadamard) gates
  int ch;
  /// The number of Hadamard gates
  int h;
  /// The number of X gates
  int x;
  /// The number of Y gates
  int y;
  /// The number of Z gates
  int z;
  /// The number of SX gates
  int sx;
  /// The number of Sycamore gates
  int syc;
  /// The number of CZ_NU gates
  int cz_nu;  
  /// The number of general gates
  int general;
  /// The number of general UN gates
  int un;
  /// The number of general ON gates
  int on;
  /// The number of composite gates
  int com;
  /// The number of adaptive gates
  int adap;
  /// The total number of gates
  int total;
};







#endif
