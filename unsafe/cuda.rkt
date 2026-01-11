;; Copyright © 2026 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         "cuda-types.rkt")

(define cuda-lib (ffi-lib "libcuda" #:fail (λ () #f)))
(define-ffi-definer define-cuda cuda-lib)

(define-syntax define+provide-cuda
  (syntax-rules ()
    [(_ name body)
     (begin
       (provide name)
       (define-cuda name body))]))

(define+provide-cuda cuInit
  (_fun (flags : _uint)
        -> (status : cu-result)))

(define+provide-cuda cuDeviceGet
  (_fun (device : (_ptr o _int))
        (ordinal : _int)
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               device
               (error 'cuDeviceGet "error with result: ~a" result))))

(define+provide-cuda cuCtxCreate
  (_fun (context : _pointer = (let ([ptr (malloc _uint 'atomic)])
                                ptr))
        (flags : _uint)
        (device : _int)
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               context
               (error 'cuCtxCreate "error with result: ~a" result))))

(define+provide-cuda cuMemAlloc
  (_fun (ptr : _pointer = (malloc _pointer 'atomic))
        _size
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               ptr
               (error "CUDA Malloc failed: " result))))

(define+provide-cuda cuMemFree
  (_fun _uint64 -> _int))

(define+provide-cuda cuModuleLoadData
  (_fun (mod : (_ptr o _pointer))
        _bytes
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               mod
               (error "CUDA cuModuleLoadData failed: " result))))

(define+provide-cuda cuModuleGetFunction
  (_fun (func : (_ptr o _pointer))
        (mod  : _pointer)
        (name : _string)
        -> (result : cu-result)
        -> (if (symbol=? 'CUDA_SUCCESS result)
               func
               (error "CUDA cuModuleGetFunction failed: " result))))

(define+provide-cuda cuLaunchKernel
  (_fun (f              : _pointer)
        (gridDimX       : _uint)
        (gridDimY       : _uint)
        (gridDimZ       : _uint)
        (blockDimX      : _uint)
        (blockDimY      : _uint)
        (blockDimZ      : _uint)
        (sharedMemBytes : _uint)
        (hStream        : _pointer)
        (kernelParams   : _pointer)
        (extra          : _pointer)
        -> (result : cu-result)
        -> (unless (symbol=? 'CUDA_SUCCESS result)
             (error "cuLaunchKernel failed: " result))))

(define+provide-cuda cuMemcpyDtoH
  (_fun (dstHost   : _pointer)
        (srcDevice : _uint64)
        (byteCount : _size)
        -> (result : cu-result)
        -> (unless (symbol=? 'CUDA_SUCCESS result)
             (error 'cuMemcpyDtoH "failed: ~a" result))))

(define+provide-cuda cuMemcpyHtoD
  (_fun (dstDevice : _uint64)
        (srcHost   : _pointer)
        (byteCount : _size)
        -> (result : cu-result)
        -> (unless (symbol=? 'CUDA_SUCCESS result)
             (error 'cuMemcpyHtoD "failed: ~a" result))))
