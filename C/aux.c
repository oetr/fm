// Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
// License: GNU AGPLv3

#include <cblas.h>
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

// Fill
GENERATE_FILL_FUNC(double, double)
GENERATE_FILL_FUNC(float, float)
GENERATE_FILL_FUNC(uint8_t, uint8_t)
GENERATE_FILL_FUNC(int8_t, int8_t)
GENERATE_FILL_FUNC(uint16_t, uint16_t)
GENERATE_FILL_FUNC(int16_t, int16_t)
GENERATE_FILL_FUNC(uint32_t, uint32_t)
GENERATE_FILL_FUNC(int32_t, int32_t)
GENERATE_FILL_FUNC(uint64_t, uint64_t)
GENERATE_FILL_FUNC(int64_t, int64_t)
GENERATE_CFILL_FUNC(cdouble, double)
GENERATE_CFILL_FUNC(cfloat, float)

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
