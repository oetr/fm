// Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
// License: GNU AGPLv3

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

#define PARALLEL_THRESHOLD_BYTES (4 * 1024 * 1024)

#define GENERATE_FILL_FUNC(SUFFIX, TYPE)                                       \
  API_EXPORT void _fill_##SUFFIX(TYPE *mem, const size_t len,                  \
                                 const TYPE value) {                           \
    const size_t data_size = len * sizeof(TYPE);                               \
    if (value == 0) {                                                          \
      memset(mem, 0, data_size);                                               \
      return;                                                                  \
    }                                                                          \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < len; \
                                                        i++) {                 \
        mem[i] = value;                                                        \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < len; i++) {                                       \
        mem[i] = value;                                                        \
      }                                                                        \
    }                                                                          \
  }

// caller has to ensure that len is even
#define GENERATE_CFILL_FUNC(SUFFIX, TYPE)                                      \
  API_EXPORT void _fill_##SUFFIX(TYPE *mem, const size_t len, const TYPE real, \
                                 const TYPE imag) {                            \
    const size_t data_size = len * sizeof(TYPE);                               \
    if (real == 0 && imag == 0) {                                              \
      memset(mem, 0, data_size);                                               \
      return;                                                                  \
    }                                                                          \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0; i < len; \
                                                        i += 2) {              \
        mem[i] = real;                                                         \
        mem[i + 1] = imag;                                                     \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < len; i += 2) {                                    \
        mem[i] = real;                                                         \
        mem[i + 1] = imag;                                                     \
      }                                                                        \
    }                                                                          \
  }

GENERATE_FILL_FUNC(double, double)
GENERATE_FILL_FUNC(float, float)
GENERATE_FILL_FUNC(uint8, uint8_t)
GENERATE_FILL_FUNC(int8, int8_t)
GENERATE_FILL_FUNC(uint16, uint16_t)
GENERATE_FILL_FUNC(int16, int16_t)
GENERATE_FILL_FUNC(uint32, uint32_t)
GENERATE_FILL_FUNC(int32, int32_t)
GENERATE_FILL_FUNC(uint64, uint64_t)
GENERATE_FILL_FUNC(int64, int64_t)
GENERATE_CFILL_FUNC(cdouble, double)
GENERATE_CFILL_FUNC(cfloat, float)

#define GENERATE_MUL_FUNC(SUFFIX, TYPE)                                        \
  API_EXPORT void _mul_##SUFFIX(                                               \
      TYPE *__restrict out, const TYPE *__restrict in1,                        \
      const TYPE *__restrict in2, const size_t len) {                          \
    const size_t data_size = len * sizeof(TYPE);                               \
                                                                               \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for simd schedule(static)") for (size_t i = 0;     \
                                                             i < len; i++) {   \
        out[i] = in1[i] * in2[i];                                              \
      }                                                                        \
    } else {                                                                   \
      _Pragma("omp simd") for (size_t i = 0; i < len; i++) {                   \
        out[i] = in1[i] * in2[i];                                              \
      }                                                                        \
    }                                                                          \
  }

// TODO: cdouble and cfloat

GENERATE_MUL_FUNC(double, double)
GENERATE_MUL_FUNC(float, float)
GENERATE_MUL_FUNC(uint8, uint8_t)
GENERATE_MUL_FUNC(int8, int8_t)
GENERATE_MUL_FUNC(uint16, uint16_t)
GENERATE_MUL_FUNC(int16, int16_t)
GENERATE_MUL_FUNC(uint32, uint32_t)
GENERATE_MUL_FUNC(int32, int32_t)
GENERATE_MUL_FUNC(uint64, uint64_t)
GENERATE_MUL_FUNC(int64, int64_t)

#define GENERATE_MUL_ADD_FUNC(SUFFIX, TYPE)                                    \
  API_EXPORT void _mul_then_add_##SUFFIX(                                      \
      TYPE *__restrict out, const TYPE *__restrict in1,                        \
      const TYPE *__restrict in2, const size_t len) {                          \
    const size_t data_size = len * sizeof(TYPE);                               \
                                                                               \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      /* 'simd' ensures the compiler uses AVX/FMA instructions inside the      \
       * threads */                                                            \
      _Pragma("omp parallel for simd schedule(static)") for (size_t i = 0;     \
                                                             i < len; i++) {   \
        out[i] += in1[i] * in2[i];                                             \
      }                                                                        \
    } else {                                                                   \
      /* Force vectorization even on the serial path */                        \
      _Pragma("omp simd") for (size_t i = 0; i < len; i++) {                   \
        out[i] += in1[i] * in2[i];                                             \
      }                                                                        \
    }                                                                          \
  }

GENERATE_MUL_ADD_FUNC(double, double)
GENERATE_MUL_ADD_FUNC(float, float)
GENERATE_MUL_ADD_FUNC(uint8, uint8_t)
GENERATE_MUL_ADD_FUNC(int8, int8_t)
GENERATE_MUL_ADD_FUNC(uint16, uint16_t)
GENERATE_MUL_ADD_FUNC(int16, int16_t)
GENERATE_MUL_ADD_FUNC(uint32, uint32_t)
GENERATE_MUL_ADD_FUNC(int32, int32_t)
GENERATE_MUL_ADD_FUNC(uint64, uint64_t)
GENERATE_MUL_ADD_FUNC(int64, int64_t)

/* returns a 1D-array containing the 'len' numbers that start from 'from' and
 * stepsize 'step'
 * This assumes that sanity checks have been done in racket and that 'mem' has
 * 'len'*sizeof(int64) bytes */
API_EXPORT void _arange(int64_t *mem, const uint64_t len, const int64_t from,
                        const int64_t step) {
  if (len > 10000000) {
#pragma omp parallel for
    for (uint64_t i = 0; i < len; i++) {
      mem[i] = from + i * step;
    }
  } else {
    for (uint64_t i = 0; i < len; i++) {
      mem[i] = from + i * step;
    }
  }
}

/* returns a 1D-array containing the 'len' numbers that start from 'from' and
 * stepsize 'step' */
/* This assumes that sanity checks have been done in racket and that 'mem' has
 * 'len'*sizeof(double) bytes */
API_EXPORT void _arange_double(double *mem, const uint64_t len,
                               const double from, const double step) {
#pragma omp parallel for
  for (uint64_t i = 0; i < len; i++) {
    mem[i] = from + i * step;
  }
}
