// Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
// License: GNU AGPLv3

#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

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
