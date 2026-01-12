;; Copyright © 2026 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         (for-syntax racket/base
                     racket/syntax
                     syntax/parse))


(define cblas-lib (ffi-lib "libcublas" #:get-lib-dirs (lambda () (list "/usr/local/cuda-12.6/lib64")) #:fail (λ () #f)))

;;(define cblas-lib (ffi-lib "libcublas" #:fail (λ () #f)))
(define-ffi-definer define-cublas-internal cblas-lib)

(define-syntax define+provide-cublas
  (syntax-rules ()
    [(_ name body)
     (begin
       (provide name)
       (define-cublas-internal name body))]))

(define cublasStatus_t
  (_enum '(
           CUBLAS_STATUS_SUCCESS          =  0
           CUBLAS_STATUS_NOT_INITIALIZED  =  1
           CUBLAS_STATUS_ALLOC_FAILED     =  3
           CUBLAS_STATUS_INVALID_VALUE    =  7
           CUBLAS_STATUS_ARCH_MISMATCH    =  8
           CUBLAS_STATUS_MAPPING_ERROR    = 11
           CUBLAS_STATUS_EXECUTION_FAILED = 13
           CUBLAS_STATUS_INTERNAL_ERROR   = 14
           CUBLAS_STATUS_NOT_SUPPORTED    = 15
           CUBLAS_STATUS_LICENSE_ERROR    = 16)))

(define cublasPointerMode_t
  (_enum '(
           CUBLAS_POINTER_MODE_HOST   = 0
           CUBLAS_POINTER_MODE_DEVICE = 1)))

(define int-placeholder (malloc _int 'interior))
(define cublasPointerMode-placeholder (malloc cublasPointerMode_t 'interior))

(define (handle-status fn-name status)
  (unless (symbol=? status 'CUBLAS_STATUS_SUCCESS)
    (error fn-name "cublas error with status: ~a~n" status)))

(define-syntax with-handle-error
  (syntax-rules ()
    [(_ fn-name status body ...)
     (begin (handle-status fn-name status)
            body ...)]))

(define+provide-cublas cublasCreate_v2
  (_fun (handle : (_ptr o _pointer interior))
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasCreate_v2 status
             handle)))

(define+provide-cublas cublasDestroy_v2
  (_fun (handle : _pointer)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasCreate_v2 status)))

(define+provide-cublas cublasGetVersion
  (_fun (value : _pointer = int-placeholder)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasGetVersion status
             (ptr-ref value _int))))

(define+provide-cublas cublasGetVersion_v2
  (_fun (handle : _pointer)
        (value : _pointer = int-placeholder)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasGetVersion status
             (ptr-ref value _int))))

(define+provide-cublas cublasSetPointerMode_v2
  (_fun (handle : _pointer)
        (value : cublasPointerMode_t)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasGetVersion status)))

(define+provide-cublas cublasGetPointerMode_v2
  (_fun (handle : _pointer)
        (mode : _pointer = cublasPointerMode-placeholder)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasGetVersion status
             (ptr-ref mode cublasPointerMode_t))))

(define cublasOperation_t
  (_enum '(CUBLAS_OP_N = 0  ;; Non-transpose
           CUBLAS_OP_T = 1  ;; Transpose
           CUBLAS_OP_C = 2 ;; Conjugate Transpose
           )))

;; We use Sgemm (Single precision/Float32). Change to Dgemm and _double* for Float64.
(define+provide-cublas cublasSgemm_v2
  (_fun (handle : _pointer)
        (transa : cublasOperation_t)
        (transb : cublasOperation_t)
        (m : _int)
        (n : _int)
        (k : _int)
        (alpha : _pointer) ;; Pointer to scalar
        (A : _pointer) (lda : _int)
        (B : _pointer) (ldb : _int)
        (beta : _pointer) ;; Pointer to scalar
        (C : _pointer) (ldc : _int)
        -> (status : cublasStatus_t)
        -> (with-handle-error 'cublasSgemm_v2 status)))

;;(define handle (cublasCreate_v2))
;;(cublasGetVersion)
;;(cublasGetVersion_v2 handle)
;;(cublasGetVersion_v2 handle)
;;(cublasGetPointerMode_v2 handle)

;;(cublasDestroy_v2 handle)
