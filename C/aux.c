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

// --- Exp ---
GENERATE_UNARY_FUNC(exp, double, double, exp)
GENERATE_UNARY_FUNC(exp, float, float, expf)

// --- Log ---
GENERATE_UNARY_FUNC(log, double, double, log)
GENERATE_UNARY_FUNC(log, float, float, logf)

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

static inline void get_broadcast_coords(size_t index, int ndim,
                                        const int64_t *restrict shape,
                                        int64_t *restrict coords) {
  for (int d = ndim - 2; d >= 0; d--) {
    coords[d] = index % shape[d];
    index /= shape[d];
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

/* * Macro to generate generic Broadcast Sub Kernels
 * Identical to Add, but with subtraction.
 */
#define GENERATE_BROADCAST_SUB_FUNC(SUFFIX, TYPE)                              \
  API_EXPORT void _sub_broadcast_##SUFFIX(                                     \
      uint64_t ndim, const int64_t *__restrict shape,                          \
      const TYPE *__restrict A_data, const int64_t *__restrict A_strides,      \
      const TYPE *__restrict B_data, const int64_t *__restrict B_strides,      \
      TYPE *__restrict out_data) {                                             \
                                                                               \
    if (ndim == 0)                                                             \
      return;                                                                  \
    size_t inner_len = shape[ndim - 1];                                        \
    size_t outer_len = 1;                                                      \
    for (int i = 0; i < ndim - 1; i++)                                         \
      outer_len *= shape[i];                                                   \
                                                                               \
    int64_t stride_A_inner = A_strides[ndim - 1];                              \
    int64_t stride_B_inner = B_strides[ndim - 1];                              \
                                                                               \
    size_t total_bytes = outer_len * inner_len * sizeof(TYPE);                 \
                                                                               \
    if (total_bytes >= PARALLEL_THRESHOLD_BYTES) {                             \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0;          \
                                                        i < outer_len; i++) {  \
        int64_t offA, offB;                                                    \
        get_broadcast_offsets(i, ndim, shape, A_strides, B_strides, &offA,     \
                              &offB);                                          \
                                                                               \
        _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {           \
          out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] -    \
                                        B_data[offB + j * stride_B_inner];     \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < outer_len; i++) {                                 \
        int64_t offA, offB;                                                    \
        get_broadcast_offsets(i, ndim, shape, A_strides, B_strides, &offA,     \
                              &offB);                                          \
                                                                               \
        _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {           \
          out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] -    \
                                        B_data[offB + j * stride_B_inner];     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

GENERATE_BROADCAST_SUB_FUNC(double, double)
GENERATE_BROADCAST_SUB_FUNC(float, float)
GENERATE_BROADCAST_SUB_FUNC(uint8, uint8_t)
GENERATE_BROADCAST_SUB_FUNC(int8, int8_t)
GENERATE_BROADCAST_SUB_FUNC(uint16, uint16_t)
GENERATE_BROADCAST_SUB_FUNC(int16, int16_t)
GENERATE_BROADCAST_SUB_FUNC(uint32, uint32_t)
GENERATE_BROADCAST_SUB_FUNC(int32, int32_t)
GENERATE_BROADCAST_SUB_FUNC(uint64, uint64_t)
GENERATE_BROADCAST_SUB_FUNC(int64, int64_t)

/* * Macro to generate generic Broadcast Div Kernels
 * Identical to Mul, but with division.
 */
#define GENERATE_BROADCAST_DIV_FUNC(SUFFIX, TYPE)                              \
  API_EXPORT void _div_broadcast_##SUFFIX(                                     \
      uint64_t ndim, const int64_t *__restrict shape,                          \
      const TYPE *__restrict A_data, const int64_t *__restrict A_strides,      \
      const TYPE *__restrict B_data, const int64_t *__restrict B_strides,      \
      TYPE *__restrict out_data) {                                             \
    if (ndim == 0)                                                             \
      return;                                                                  \
                                                                               \
    size_t inner_len = shape[ndim - 1];                                        \
    size_t outer_len = 1;                                                      \
    for (int i = 0; i < ndim - 1; i++)                                         \
      outer_len *= shape[i];                                                   \
                                                                               \
    int64_t stride_A_inner = A_strides[ndim - 1];                              \
    int64_t stride_B_inner = B_strides[ndim - 1];                              \
                                                                               \
    /* OMP Parallel Region */                                                  \
    _Pragma("omp parallel") {                                                  \
      int thread_id = omp_get_thread_num();                                    \
      int num_threads = omp_get_num_threads();                                 \
      size_t chunk_size = (outer_len + num_threads - 1) / num_threads;         \
      size_t start = thread_id * chunk_size;                                   \
      size_t end = start + chunk_size;                                         \
      if (end > outer_len)                                                     \
        end = outer_len;                                                       \
                                                                               \
      if (start < end) {                                                       \
        int64_t coords[32];                                                    \
        int64_t offA = 0, offB = 0;                                            \
                                                                               \
        get_broadcast_coords(start, ndim, shape, coords);                      \
                                                                               \
        for (int d = 0; d < ndim - 1; d++) {                                   \
          offA += coords[d] * A_strides[d];                                    \
          offB += coords[d] * B_strides[d];                                    \
        }                                                                      \
                                                                               \
        for (size_t i = start; i < end; i++) {                                 \
          _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {         \
            out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] /  \
                                          B_data[offB + j * stride_B_inner];   \
          }                                                                    \
                                                                               \
          for (int d = ndim - 2; d >= 0; d--) {                                \
            coords[d]++;                                                       \
            if (coords[d] < shape[d]) {                                        \
              offA += A_strides[d];                                            \
              offB += B_strides[d];                                            \
              break;                                                           \
            } else {                                                           \
              coords[d] = 0;                                                   \
              offA -= A_strides[d] * (shape[d] - 1);                           \
              offB -= B_strides[d] * (shape[d] - 1);                           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
GENERATE_BROADCAST_DIV_FUNC(double, double)
GENERATE_BROADCAST_DIV_FUNC(float, float)
GENERATE_BROADCAST_DIV_FUNC(uint8, uint8_t)
GENERATE_BROADCAST_DIV_FUNC(int8, int8_t)
GENERATE_BROADCAST_DIV_FUNC(uint16, uint16_t)
GENERATE_BROADCAST_DIV_FUNC(int16, int16_t)
GENERATE_BROADCAST_DIV_FUNC(uint32, uint32_t)
GENERATE_BROADCAST_DIV_FUNC(int32, int32_t)
GENERATE_BROADCAST_DIV_FUNC(uint64, uint64_t)
GENERATE_BROADCAST_DIV_FUNC(int64, int64_t)

/* * Macro to generate generic Broadcast Pow Kernels
 */
#define GENERATE_BROADCAST_POW_FUNC(SUFFIX, TYPE, POW_FUNC)                    \
  API_EXPORT void _pow_broadcast_##SUFFIX(                                     \
      uint64_t ndim, const int64_t *__restrict shape,                          \
      const TYPE *__restrict A_data, const int64_t *__restrict A_strides,      \
      const TYPE *__restrict B_data, const int64_t *__restrict B_strides,      \
      TYPE *__restrict out_data) {                                             \
    if (ndim == 0)                                                             \
      return;                                                                  \
                                                                               \
    size_t inner_len = shape[ndim - 1];                                        \
    size_t outer_len = 1;                                                      \
    for (int i = 0; i < ndim - 1; i++)                                         \
      outer_len *= shape[i];                                                   \
                                                                               \
    int64_t stride_A_inner = A_strides[ndim - 1];                              \
    int64_t stride_B_inner = B_strides[ndim - 1];                              \
                                                                               \
    _Pragma("omp parallel") {                                                  \
      int thread_id = omp_get_thread_num();                                    \
      int num_threads = omp_get_num_threads();                                 \
      size_t chunk_size = (outer_len + num_threads - 1) / num_threads;         \
      size_t start = thread_id * chunk_size;                                   \
      size_t end = start + chunk_size;                                         \
      if (end > outer_len)                                                     \
        end = outer_len;                                                       \
                                                                               \
      if (start < end) {                                                       \
        int64_t coords[32];                                                    \
        int64_t offA = 0, offB = 0;                                            \
                                                                               \
        get_broadcast_coords(start, ndim, shape, coords);                      \
                                                                               \
        for (int d = 0; d < ndim - 1; d++) {                                   \
          offA += coords[d] * A_strides[d];                                    \
          offB += coords[d] * B_strides[d];                                    \
        }                                                                      \
                                                                               \
        for (size_t i = start; i < end; i++) {                                 \
          /* pow might not be vectorizable easily, removing simd for safety */ \
          for (size_t j = 0; j < inner_len; j++) {                             \
            out_data[i * inner_len + j] =                                      \
                POW_FUNC(A_data[offA + j * stride_A_inner],                    \
                         B_data[offB + j * stride_B_inner]);                   \
          }                                                                    \
                                                                               \
          for (int d = ndim - 2; d >= 0; d--) {                                \
            coords[d]++;                                                       \
            if (coords[d] < shape[d]) {                                        \
              offA += A_strides[d];                                            \
              offB += B_strides[d];                                            \
              break;                                                           \
            } else {                                                           \
              coords[d] = 0;                                                   \
              offA -= A_strides[d] * (shape[d] - 1);                           \
              offB -= B_strides[d] * (shape[d] - 1);                           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

GENERATE_BROADCAST_POW_FUNC(double, double, pow)
GENERATE_BROADCAST_POW_FUNC(float, float, powf)
// For integers, we probably shouldn't support pow natively or cast to double?
// Let's stick to float/double for pow for now as it's the most common use case
// for tensors.

/* * Macro to generate generic Broadcast Mul Kernels
 * Identical to Add, but with multiplication.
 */
#define GENERATE_BROADCAST_MUL_FUNC(SUFFIX, TYPE)                              \
  API_EXPORT void _mul_broadcast_##SUFFIX(                                     \
      uint64_t ndim, const int64_t *__restrict shape,                          \
      const TYPE *__restrict A_data, const int64_t *__restrict A_strides,      \
      const TYPE *__restrict B_data, const int64_t *__restrict B_strides,      \
      TYPE *__restrict out_data) {                                             \
    if (ndim == 0)                                                             \
      return;                                                                  \
                                                                               \
    /* Dimensions setup */                                                     \
    size_t inner_len = shape[ndim - 1];                                        \
    size_t outer_len = 1;                                                      \
    for (int i = 0; i < ndim - 1; i++)                                         \
      outer_len *= shape[i];                                                   \
                                                                               \
    int64_t stride_A_inner = A_strides[ndim - 1];                              \
    int64_t stride_B_inner = B_strides[ndim - 1];                              \
                                                                               \
    /* OMP Parallel Region: We manage the loop bounds manually */              \
    _Pragma("omp parallel") {                                                  \
      /* A. Calculate Chunk per Thread */                                      \
      int thread_id = omp_get_thread_num();                                    \
      int num_threads = omp_get_num_threads();                                 \
      size_t chunk_size = (outer_len + num_threads - 1) / num_threads;         \
      size_t start = thread_id * chunk_size;                                   \
      size_t end = start + chunk_size;                                         \
      if (end > outer_len)                                                     \
        end = outer_len;                                                       \
                                                                               \
      if (start < end) {                                                       \
        /* B. Stack-allocated coords to track state */                         \
        /* Max dims 32 is standard safe limit, or malloc if dynamic */         \
        int64_t coords[32];                                                    \
        int64_t offA = 0, offB = 0;                                            \
                                                                               \
        /* C. EXPENSIVE INIT: Run only ONCE per thread */                      \
        get_broadcast_coords(start, ndim, shape, coords);                      \
                                                                               \
        /* Calculate initial offsets based on coords */                        \
        for (int d = 0; d < ndim - 1; d++) {                                   \
          offA += coords[d] * A_strides[d];                                    \
          offB += coords[d] * B_strides[d];                                    \
        }                                                                      \
                                                                               \
        /* D. MAIN LOOP: "The Odometer" - No division, only addition */        \
        for (size_t i = start; i < end; i++) {                                 \
                                                                               \
          /* 1. Compute Inner Loop (SIMD) */                                   \
          _Pragma("omp simd") for (size_t j = 0; j < inner_len; j++) {         \
            out_data[i * inner_len + j] = A_data[offA + j * stride_A_inner] *  \
                                          B_data[offB + j * stride_B_inner];   \
          }                                                                    \
                                                                               \
          /* 2. Increment Coords & Offsets (Cheap!) */                         \
          /* We carry the one up the dimensions */                             \
          for (int d = ndim - 2; d >= 0; d--) {                                \
            coords[d]++;                                                       \
            /* Did we overflow this dimension? */                              \
            if (coords[d] < shape[d]) {                                        \
              /* No overflow: Add stride and continue working */               \
              offA += A_strides[d];                                            \
              offB += B_strides[d];                                            \
              break;                                                           \
            } else {                                                           \
              /* Overflow: Reset to 0, subtract full stride, carry over */     \
              coords[d] = 0;                                                   \
              /* We subtract (dim-1)*stride to "rewind" the pointer */         \
              offA -= A_strides[d] * (shape[d] - 1);                           \
              offB -= B_strides[d] * (shape[d] - 1);                           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
GENERATE_BROADCAST_MUL_FUNC(double, double)
GENERATE_BROADCAST_MUL_FUNC(float, float)
GENERATE_BROADCAST_MUL_FUNC(uint8, uint8_t)
GENERATE_BROADCAST_MUL_FUNC(int8, int8_t)
GENERATE_BROADCAST_MUL_FUNC(uint16, uint16_t)
GENERATE_BROADCAST_MUL_FUNC(int16, int16_t)
GENERATE_BROADCAST_MUL_FUNC(uint32, uint32_t)
GENERATE_BROADCAST_MUL_FUNC(int32, int32_t)
GENERATE_BROADCAST_MUL_FUNC(uint64, uint64_t)
GENERATE_BROADCAST_MUL_FUNC(int64, int64_t)

// -----------------------------------------------------------------------------
// Reduction Sum Kernel (For Backward Broadcast)
// -----------------------------------------------------------------------------
// This kernel reduces a 3D tensor-view of shape (outer, reduce, inner)
// along the 'reduce' dimension.
// The output shape is (outer, inner).
// This is sufficient to handle any single-dimension reduction.
// For multi-dimension reduction, one can call this iteratively (though less
// efficient). But for standard backward broadcasting, we usually reduce along
// one or more axes.
//
// Arguments:
// - outer_dim: Product of dimensions "before" the reduction axis.
// - reduce_dim: Size of the dimension to reduce.
// - inner_dim: Product of dimensions "after" the reduction axis.
// - in_data: Input pointer.
// - out_data: Output pointer (Accumulates!).

#define GENERATE_REDUCE_SUM_FUNC(SUFFIX, TYPE)                                 \
  API_EXPORT void _reduce_sum_##SUFFIX(                                        \
      const size_t outer_dim, const size_t reduce_dim, const size_t inner_dim, \
      const TYPE *__restrict in_data, TYPE *__restrict out_data) {             \
                                                                               \
    const size_t total_out_elements = outer_dim * inner_dim;                   \
                                                                               \
    if (total_out_elements >= PARALLEL_THRESHOLD_BYTES / sizeof(TYPE)) {       \
      _Pragma("omp parallel for schedule(static)") for (size_t i = 0;          \
                                                        i < outer_dim; i++) {  \
        for (size_t k = 0; k < inner_dim; k++) {                               \
          TYPE sum = 0;                                                        \
          size_t out_idx = i * inner_dim + k;                                  \
          size_t in_base = i * reduce_dim * inner_dim + k;                     \
          for (size_t j = 0; j < reduce_dim; j++) {                            \
            sum += in_data[in_base + j * inner_dim];                           \
          }                                                                    \
          out_data[out_idx] += sum;                                            \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < outer_dim; i++) {                                 \
        for (size_t k = 0; k < inner_dim; k++) {                               \
          TYPE sum = 0;                                                        \
          size_t out_idx = i * inner_dim + k;                                  \
          size_t in_base = i * reduce_dim * inner_dim + k;                     \
          for (size_t j = 0; j < reduce_dim; j++) {                            \
            sum += in_data[in_base + j * inner_dim];                           \
          }                                                                    \
          out_data[out_idx] += sum;                                            \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

GENERATE_REDUCE_SUM_FUNC(double, double)
GENERATE_REDUCE_SUM_FUNC(float, float)
GENERATE_REDUCE_SUM_FUNC(int64, int64_t)
GENERATE_REDUCE_SUM_FUNC(uint64, uint64_t)
GENERATE_REDUCE_SUM_FUNC(int32, int32_t)
GENERATE_REDUCE_SUM_FUNC(uint32, uint32_t)
GENERATE_REDUCE_SUM_FUNC(int16, int16_t)
GENERATE_REDUCE_SUM_FUNC(uint16, uint16_t)
GENERATE_REDUCE_SUM_FUNC(int8, int8_t)
GENERATE_REDUCE_SUM_FUNC(uint8, uint8_t)

/* * Helper: Unravels a linear index into offsets based on shape/strides.
 * We use this to map linear loop indices to memory offsets.
 */
static inline size_t get_offset(size_t linear_index, size_t ndim,
                                const size_t *shape, const size_t *strides) {
  size_t offset = 0;
  for (int i = ndim - 1; i >= 0; i--) {
    size_t dim_sz = shape[i];
    size_t coord = linear_index % dim_sz;
    linear_index /= dim_sz;
    offset += coord * strides[i];
  }
  return offset;
}

#define GENERATE_GENERIC_REDUCE(SUFFIX, TYPE)                                  \
  API_EXPORT void _reduce_sum_generic_##SUFFIX(                                \
      const size_t n_keep,                                                     \
      const size_t *keep_shape,       /* Shape of parallel dimensions */       \
      const size_t *keep_src_strides, /* Stride in the BIG tensor */           \
      const size_t *keep_dst_strides, /* Stride in the SMALL tensor */         \
      const size_t n_reduce,                                                   \
      const size_t *reduce_shape,       /* Shape of collapsed dimensions */    \
      const size_t *reduce_src_strides, /* Stride in the BIG tensor */         \
      const TYPE *__restrict src_data, TYPE *__restrict dst_data) {            \
                                                                               \
    /* 1. Calculate total iterations for outer and inner loops */              \
    size_t total_keep = 1;                                                     \
    for (size_t i = 0; i < n_keep; i++)                                        \
      total_keep *= keep_shape[i];                                             \
                                                                               \
    size_t total_reduce = 1;                                                   \
    for (size_t i = 0; i < n_reduce; i++)                                      \
      total_reduce *= reduce_shape[i];                                         \
                                                                               \
    /* 2. Parallel Loop over the Output (Keep Dims) */                         \
    _Pragma("omp parallel for schedule(static)") for (size_t i = 0;            \
                                                      i < total_keep; i++) {   \
      /* Map linear 'i' to offsets in Source and Dest */                       \
      size_t dst_offset = get_offset(i, n_keep, keep_shape, keep_dst_strides); \
      size_t src_base = get_offset(i, n_keep, keep_shape, keep_src_strides);   \
                                                                               \
      TYPE sum = 0;                                                            \
                                                                               \
      /* 3. Inner Loop over the Reduction Volume */                            \
      for (size_t j = 0; j < total_reduce; j++) {                              \
        size_t reduce_offset =                                                 \
            get_offset(j, n_reduce, reduce_shape, reduce_src_strides);         \
        sum += src_data[src_base + reduce_offset];                             \
      }                                                                        \
                                                                               \
      /* Accumulate into existing gradient (AXPY style) or Overwrite? */       \
      /* Standard is += for gradients. Ensure dst is zeroed before if          \
       * needed.*/                                                             \
      dst_data[dst_offset] += sum;                                             \
    }                                                                          \
  }

GENERATE_GENERIC_REDUCE(double, double)
GENERATE_GENERIC_REDUCE(float, float)
GENERATE_GENERIC_REDUCE(int64, int64_t)
GENERATE_GENERIC_REDUCE(uint64, uint64_t)
GENERATE_GENERIC_REDUCE(int32, int32_t)
GENERATE_GENERIC_REDUCE(uint32, uint32_t)
GENERATE_GENERIC_REDUCE(int16, int16_t)
GENERATE_GENERIC_REDUCE(uint16, uint16_t)
GENERATE_GENERIC_REDUCE(int8, int8_t)
GENERATE_GENERIC_REDUCE(uint8, uint8_t)

/* * Macro to generate generic Broadcast Mul Reduce Kernels
 * Used for fusing multiplication and reduction in backward passes.
 */
#define GENERATE_MUL_BROADCAST_REDUCE(SUFFIX, TYPE)                            \
  API_EXPORT void _mul_broadcast_reduce_sum_##SUFFIX(                          \
      const size_t n_keep,                                                     \
      const size_t *keep_shape,       /* Shape of parallel dimensions */       \
      const size_t *keep_A_strides,   /* Stride in grad_out */                 \
      const size_t *keep_B_strides,   /* Stride in other_tensor */             \
      const size_t *keep_dst_strides, /* Stride in accumulator */              \
      const size_t n_reduce,                                                   \
      const size_t *reduce_shape,     /* Shape of collapsed dimensions */      \
      const size_t *reduce_A_strides, /* Stride in grad_out */                 \
      const size_t *reduce_B_strides, /* Stride in other_tensor */             \
      const TYPE *__restrict A_data,  /* grad_out */                           \
      const TYPE *__restrict B_data,  /* other_tensor */                       \
      TYPE *__restrict dst_data) {    /* accumulator */                        \
                                                                               \
    /* 1. Calculate total iterations */                                        \
    size_t total_keep = 1;                                                     \
    for (size_t i = 0; i < n_keep; i++)                                        \
      total_keep *= keep_shape[i];                                             \
                                                                               \
    size_t total_reduce = 1;                                                   \
    for (size_t i = 0; i < n_reduce; i++)                                      \
      total_reduce *= reduce_shape[i];                                         \
                                                                               \
    /* 2. Parallel Loop over the Output (Keep Dims) */                         \
    _Pragma("omp parallel for schedule(static)") for (size_t i = 0;            \
                                                      i < total_keep; i++) {   \
      /* Map linear i to offsets */                                            \
      size_t dst_offset = get_offset(i, n_keep, keep_shape, keep_dst_strides); \
      size_t A_base = get_offset(i, n_keep, keep_shape, keep_A_strides);       \
      size_t B_base = get_offset(i, n_keep, keep_shape, keep_B_strides);       \
                                                                               \
      TYPE sum = 0;                                                            \
                                                                               \
      /* 3. Inner Loop over the Reduction Volume */                            \
      for (size_t j = 0; j < total_reduce; j++) {                              \
        size_t A_off =                                                         \
            get_offset(j, n_reduce, reduce_shape, reduce_A_strides);           \
        size_t B_off =                                                         \
            get_offset(j, n_reduce, reduce_shape, reduce_B_strides);           \
                                                                               \
        sum += A_data[A_base + A_off] * B_data[B_base + B_off];                \
      }                                                                        \
                                                                               \
      dst_data[dst_offset] += sum;                                             \
    }                                                                          \
  }

GENERATE_MUL_BROADCAST_REDUCE(double, double)
GENERATE_MUL_BROADCAST_REDUCE(float, float)
GENERATE_MUL_BROADCAST_REDUCE(int64, int64_t)
GENERATE_MUL_BROADCAST_REDUCE(uint64, uint64_t)
GENERATE_MUL_BROADCAST_REDUCE(int32, int32_t)
GENERATE_MUL_BROADCAST_REDUCE(uint32, uint32_t)
GENERATE_MUL_BROADCAST_REDUCE(int16, int16_t)
GENERATE_MUL_BROADCAST_REDUCE(uint16, uint16_t)
GENERATE_MUL_BROADCAST_REDUCE(int8, int8_t)
GENERATE_MUL_BROADCAST_REDUCE(uint8, uint8_t)

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
