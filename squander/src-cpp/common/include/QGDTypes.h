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

/// @brief Structure type representing single-precision complex numbers
struct alignas(8) QGD_Complex8 {
  float real; ///< real part
  float imag; ///< imaginary part
};

// Exact bitwise equality (no epsilon tolerance). use with caution for floats
inline bool operator==(const QGD_Complex16& a, const QGD_Complex16& b) noexcept {
  return (a.real == b.real) && (a.imag == b.imag);
}
inline bool operator!=(const QGD_Complex16& a, const QGD_Complex16& b) noexcept {
  return (a.real != b.real) || (a.imag != b.imag);
}
inline bool operator==(const QGD_Complex8& a, const QGD_Complex8& b) noexcept {
  return (a.real == b.real) && (a.imag == b.imag);
}
inline bool operator!=(const QGD_Complex8& a, const QGD_Complex8& b) noexcept {
  return (a.real != b.real) || (a.imag != b.imag);
}

// Lexicographic ordering for container compatibility (NOT mathematically meaningful)
// Other comparisons omitted to avoid confusion
inline bool operator<(const QGD_Complex16& a, const QGD_Complex16& b) noexcept {
  if (a.real != b.real) return a.real < b.real;
  return a.imag < b.imag;
}
inline bool operator<(const QGD_Complex8& a, const QGD_Complex8& b) noexcept {
  if (a.real != b.real) return a.real < b.real;
  return a.imag < b.imag;
}

// Stream output operators for printing
#ifdef __cplusplus
#include <ostream>
inline std::ostream& operator<<(std::ostream& os, const QGD_Complex16& c) {
  os << "(" << c.real << ", " << c.imag << "*i)";
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const QGD_Complex8& c) {
  os << "(" << c.real << ", " << c.imag << "*i)";
  return os;
}
#endif

#endif
