;; Copyright © 2026 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         "cuda-error-codes.rkt")

(define cuda-lib (ffi-lib "libcuda" #:fail (λ () #f)))
(define-ffi-definer define-cuda cuda-lib)

(define-cuda cuInit
  (_fun (flags : _uint)
        -> (status : cu-result)))

(define-cuda cuDeviceGet
  (_fun (device : (_ptr o _int))
        (ordinal : _int)
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               device
               (error 'cuDeviceGet "error with result: " result))))

(define-cuda cuCtxCreate
  (_fun (context : _pointer = (let ([ptr (malloc _uint 'atomic)])
                                ptr))
        (flags : _uint)
        (device : _int)
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               context
               (error 'cuDeviceGet "error with result: " result))))

(define-cuda cuMemAlloc
  (_fun (ptr : (_ptr o _uint64 raw))
        _size
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               ptr
               (error "CUDA Malloc failed: " result))))

(define-cuda cuMemFree
  (_fun _uint64 -> _int))

(define-cuda cuMemcpyHtoD
  (_fun _uint64 _pointer _size -> _int))

(cuInit 0)
(define device (cuDeviceGet 0))
(define context (cuCtxCreate 0 device))
(define ptr (cuMemAlloc 1024))
