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
    const size_t data_size = len * sizeof(TYPE) * 2;                           \
    if (real == 0 && imag == 0) {                                              \
      memset(mem, 0, data_size);                                               \
      return;                                                                  \
    }                                                                          \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0;          \
                                                        i < len * 2; i += 2) { \
        mem[i] = real;                                                         \
        mem[i + 1] = imag;                                                     \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < len * 2; i += 2) {                                \
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

// -----------------------------------------------------------------------------
// Activation Functions
// -----------------------------------------------------------------------------

#define GENERATE_UNARY_FUNC(NAME, SUFFIX, TYPE, FUNC_BODY)                     \
  API_EXPORT void _##NAME##_##SUFFIX(TYPE *out, const TYPE *in,                \
                                     const size_t len) {                       \
    const size_t data_size = len * sizeof(TYPE);                               \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for simd schedule(static)") for (size_t i = 0;     \
                                                             i < len; i++) {   \
        out[i] = FUNC_BODY(in[i]);                                             \
      }                                                                        \
    } else {                                                                   \
      _Pragma("omp simd") for (size_t i = 0; i < len; i++) {                   \
        out[i] = FUNC_BODY(in[i]);                                             \
      }                                                                        \
    }                                                                          \
  }

// --- Tanh ---
// tanh(x) is standard C math
GENERATE_UNARY_FUNC(tanh, double, double, tanh)
GENERATE_UNARY_FUNC(tanh, float, float, tanhf)

// --- Sigmoid ---
// sigmoid(x) = 1 / (1 + exp(-x))
static inline double sigmoid_double(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline float sigmoid_float(float x) { return 1.0f / (1.0f + expf(-x)); }

GENERATE_UNARY_FUNC(sigmoid, double, double, sigmoid_double)
GENERATE_UNARY_FUNC(sigmoid, float, float, sigmoid_float)

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

/* Helper: Reconstructs the memory offset for A and B from a linear
 * 'outer_index'. We map the linear index back to N-dimensional coordinates,
 * then mapped to memory offsets. This runs once per row (not per pixel), so the
 * division overhead is negligible.
 */
static inline void get_broadcast_offsets(size_t outer_index, int ndim,
                                         const int64_t *restrict shape,
                                         const int64_t *restrict stridesA,
                                         const int64_t *restrict stridesB,
                                         int64_t *offsetA, int64_t *offsetB) {
  *offsetA = 0;
  *offsetB = 0;

  // We iterate from the second-to-last dimension up to 0
  // (The last dimension is handled by the inner SIMD loop)
  for (int d = ndim - 2; d >= 0; d--) {
    size_t dim_size = shape[d];
    size_t coord = outer_index % dim_size;
    outer_index /= dim_size;

    *offsetA += coord * stridesA[d];
    *offsetB += coord * stridesB[d];
  }
}

/* * Macro to generate generic Broadcast Add Kernels
 * Supports:
 * - Arbitrary Ranks
 * - Arbitrary Strides
 * - OpenMP Parallelism (Row-based)
 * - SIMD Vectorization (Inner-dimension)
 */
#define GENERATE_BROADCAST_ADD_FUNC(SUFFIX, TYPE)                              \
  API_EXPORT void _add_broadcast_##SUFFIX(                                     \
      uint64_t ndim, const int64_t *__restrict shape,                          \
      const TYPE *__restrict A_data, const int64_t *__restrict A_strides,      \
      const TYPE *__restrict B_data, const int64_t *__restrict B_strides,      \
      TYPE *__restrict out_data) {                                             \
                                                                               \
    /* 1. Base Case: Handle scalar/empty or 1D quickly */                      \
    if (ndim == 0)                                                             \
      return;                                                                  \
    size_t inner_len = shape[ndim - 1];                                        \
    size_t outer_len = 1;                                                      \
    for (int i = 0; i < ndim - 1; i++)                                         \
      outer_len *= shape[i];                                                   \
                                                                               \
    /* Strides for the innermost loop (critical for SIMD) */                   \
    int64_t stride_A_inner = A_strides[ndim - 1];                              \
    int64_t stride_B_inner = B_strides[ndim - 1];                              \
                                                                               \
    size_t total_bytes = outer_len * inner_len * sizeof(TYPE);                 \
                                                                               \
    /* 2. Parallel Loop over the "Outer" dimensions (Rows) */                  \
    if (total_bytes >= PARALLEL_THRESHOLD_BYTES) {                             \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0;          \
                                                        i < outer_len; i++) {  \
        int64_t offA, offB;                                                    \
        get_broadcast_offsets(i, ndim, shape, A_strides, B_strides, &offA,     \
                              &offB);                                          \
                                                                               \
        /* 3. SIMD Loop over the "Inner" dimension (Columns) */                \
        _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {           \
          out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] +    \
                                        B_data[offB + j * stride_B_inner];     \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      /* Serial Fallback for small tensors */                                  \
      for (size_t i = 0; i < outer_len; i++) {                                 \
        int64_t offA, offB;                                                    \
        get_broadcast_offsets(i, ndim, shape, A_strides, B_strides, &offA,     \
                              &offB);                                          \
                                                                               \
        _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {           \
          out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] +    \
                                        B_data[offB + j * stride_B_inner];     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

// Instantiate for all required types
GENERATE_BROADCAST_ADD_FUNC(double, double)
GENERATE_BROADCAST_ADD_FUNC(float, float)
GENERATE_BROADCAST_ADD_FUNC(uint8, uint8_t)
GENERATE_BROADCAST_ADD_FUNC(int8, int8_t)
GENERATE_BROADCAST_ADD_FUNC(uint16, uint16_t)
GENERATE_BROADCAST_ADD_FUNC(int16, int16_t)
GENERATE_BROADCAST_ADD_FUNC(uint32, uint32_t)
GENERATE_BROADCAST_ADD_FUNC(int32, int32_t)
GENERATE_BROADCAST_ADD_FUNC(uint64, uint64_t)
GENERATE_BROADCAST_ADD_FUNC(int64, int64_t)

/* Backward Tanh Kernel
 * Computes: grad_in += grad_out * (1 - out^2)
 * TODO: We iterate linearly. Add support for non-contiguous views/slices later.
 * Will need strided iterators here like in the broadcast kernel.
 */
#define GENERATE_BACKWARD_TANH_FUNC(SUFFIX, TYPE)                              \
  API_EXPORT void _backward_tanh_##SUFFIX(                                     \
      const TYPE *__restrict out_data, const TYPE *__restrict grad_out,        \
      TYPE *__restrict grad_in, const size_t len) {                            \
                                                                               \
    const size_t data_size = len * sizeof(TYPE);                               \
                                                                               \
    if (data_size >= PARALLEL_THRESHOLD_BYTES) {                               \
      _Pragma("omp parallel for simd schedule(static)") for (size_t i = 0;     \
                                                             i < len; i++) {   \
        TYPE y = out_data[i];                                                  \
        TYPE dy = grad_out[i];                                                 \
        grad_in[i] += dy * (1.0 - y * y);                                      \
      }                                                                        \
    } else {                                                                   \
      _Pragma("omp simd") for (size_t i = 0; i < len; i++) {                   \
        TYPE y = out_data[i];                                                  \
        TYPE dy = grad_out[i];                                                 \
        grad_in[i] += dy * (1.0 - y * y);                                      \
      }                                                                        \
    }                                                                          \
  }

GENERATE_BACKWARD_TANH_FUNC(double, double)
GENERATE_BACKWARD_TANH_FUNC(float, float)

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
