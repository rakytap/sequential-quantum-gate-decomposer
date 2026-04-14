/*
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef QGD_MATH_H
#define QGD_MATH_H

#include <cmath>
#if !defined(_WIN32)
#include <math.h>
#endif

template <typename T>
inline T qgd_sin(T x);

template <typename T>
inline T qgd_cos(T x);

template <typename T>
inline void qgd_sincos(T x, T* s, T* c);

template <>
inline float qgd_sin<float>(float x) {
    return std::sin(x);
}

template <>
inline double qgd_sin<double>(double x) {
    return std::sin(x);
}

template <>
inline float qgd_cos<float>(float x) {
    return std::cos(x);
}

template <>
inline double qgd_cos<double>(double x) {
    return std::cos(x);
}

template <>
inline void qgd_sincos<float>(float x, float* s, float* c) {
#if defined(_WIN32)
    *s = std::sin(x);
    *c = std::cos(x);
#else
    ::sincosf(x, s, c);
#endif
}

template <>
inline void qgd_sincos<double>(double x, double* s, double* c) {
#if defined(_WIN32)
    *s = std::sin(x);
    *c = std::cos(x);
#elif defined(__APPLE__)
    ::__sincos(x, s, c);
#else
    ::sincos(x, s, c);
#endif
}

#endif
