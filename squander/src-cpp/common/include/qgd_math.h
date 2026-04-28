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

// Provide standard C math constants when the platform headers do not.
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif

#ifndef M_LOG10E
#define M_LOG10E 0.434294481903251827651
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#ifndef M_PI_4
#define M_PI_4 0.785398163397448309616
#endif

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

#ifndef M_2_PI
#define M_2_PI 0.636619772367581343076
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
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
#elif defined(__APPLE__)
    ::__sincosf(x, s, c);
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
