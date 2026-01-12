#lang racket

(require "main.rkt"
         ;;"../unsafe/cuda-rt.rkt"
         "../unsafe/cuda.rkt"
         "../unsafe/cublas.rkt"
         ffi/unsafe/cvector
         ffi/unsafe)

;; --- Benchmarking ---

;; Helper to convert Integer Address (Driver) -> Pointer Object (for cuBLAS)
(define (addr->ptr addr)
  (cast (ptr-ref addr _pointer) _pointer _uint64))

(define (->ptr addr)
  (cast addr _uint64 _pointer))

(define (make-float-array size)
  (make-cvector _float size))

(define (make-scalar-ptr val)
  (let ([p (malloc _float 'interior)])
    (ptr-set! p _float val)
    p))

(define (benchmark-gemm N)
  (printf "Benchmarking N=~a (Driver API)... " N)
  (flush-output)

  ;; 1. Initialize Driver & Create Context
  (cuInit 0)
  (define device (cuDeviceGet 0))
  (define context (cuDevicePrimaryCtxRetain device))
  (cuCtxPushCurrent_v2 context)

  ;; 2. Initialize cuBLAS (It attaches to the pushed Driver context automatically)
  (define handle (cublasCreate_v2))


  ;; 3. Setup Data
  (define size (* N N))
  (define bytes (* size (ctype-sizeof _float)))

  (define h_A (make-float-array size))
  (define h_B (make-float-array size))
  ;; (Fill h_A/h_B with random data here if needed)

  ;; 4. Device Allocation (Returns Integer Addresses)
  (define d_A_addr (cuMemAlloc_v2 bytes))
  (define d_B_addr (cuMemAlloc_v2 bytes))
  (define d_C_addr (cuMemAlloc_v2 bytes))

  ;;(define alpha_addr (cuMemAlloc_v2 4))
  ;;(define beta_addr (cuMemAlloc_v2 4))

  ;; 5. Copy HtoD
  (cuMemcpyHtoD_v2 d_A_addr (cvector-ptr h_A) bytes)
  (cuMemcpyHtoD_v2 d_B_addr (cvector-ptr h_B) bytes)

  ;; 6. Convert Addresses to Pointers for cuBLAS
  (define d_A_ptr (->ptr d_A_addr))
  (define d_B_ptr (->ptr d_B_addr))
  (define d_C_ptr (->ptr d_C_addr))

  (define alpha (make-scalar-ptr 1.0))
  (define beta  (make-scalar-ptr 0.0))

  ;; 7. Warmup
  (cublasSgemm_v2 handle 'CUBLAS_OP_N 'CUBLAS_OP_N N N N alpha d_A_ptr N d_B_ptr N beta d_C_ptr N)
  (cuCtxSynchronize)

  ;; 8. Benchmark
  (define-values (results cpu-time real-time gc-time)
    (time-apply
     (lambda ()
       (for ([i 10])
         (cublasSgemm_v2 handle 'CUBLAS_OP_N 'CUBLAS_OP_N N N N alpha d_A_ptr N d_B_ptr N beta d_C_ptr N))
       ;; Wait for GPU to finish
       (cuCtxSynchronize))
     '()))

  (printf "Time: ~a ms\n" real-time)

  ;; 9. Cleanup
  (cuMemFree_v2 d_A_addr)
  (cuMemFree_v2 d_B_addr)
  (cuMemFree_v2 d_C_addr)
  (cuCtxSynchronize)
  (cublasDestroy_v2 handle)
  (cuCtxPopCurrent_v2 context)
  (cuDevicePrimaryCtxRelease_v2 device))

(module+ main
  (benchmark-gemm 1024)
  (benchmark-gemm 2048)
  (benchmark-gemm (expt 2 12))
  (benchmark-gemm (expt 2 13))
  (benchmark-gemm (expt 2 14))
  (benchmark-gemm (expt 2 15))
  ;; reaching 27.16 TFLOPS on my RTX 3090
  )


