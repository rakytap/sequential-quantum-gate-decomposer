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
