;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         "./unsafe/tensor.rkt"
         "./internal/utilities.rkt")

(define+provide (tensor-length A) (apply * (tensor-shape A)))

(define+provide (tensor-rank A) (length (tensor-shape A)))

(define+provide
 (make-tensor shape (fill #f) #:type (type 'double) #:children (children empty))
 (when (for/or ([dim-length shape])
         (<= dim-length 0))
   (error 'make-tensor "dimensions should be >= 1"))
 (define _type #f)
 (define fill-real #f)
 (define fill-imag #f)
 (define n-elements (apply * shape))
 (set! _type (symbol->type type))
 (match type
   [(== 'bool)
    (if fill
        (set! fill-real 1)
        (set! fill-real 0))]
   [(or 'int64 'int32 'int16 'int8 'uint64 'uint32 'uint16 'uint8) (set! fill-real fill)]
   [(== 'double)
    (when fill
      (set! fill-real (real->double-flonum fill)))]
   [(== 'float)
    (when fill
      (set! fill-real (real->double-flonum fill)))]
   [(== 'cfloat)
    (when fill
      (set! fill-real (real->double-flonum (real-part fill)))
      (set! fill-imag (real->double-flonum (imag-part fill))))
    (set! n-elements (* n-elements 2))]
   [(== 'cdouble)
    (when fill
      (set! fill-real (real->double-flonum (real-part fill)))
      (set! fill-imag (real->double-flonum (imag-part fill))))
    (set! n-elements (* n-elements 2))]
   [_ (error 'make-tensor "unsupported type: ~a~n" type)])
 (define data (alloc-tensor n-elements #:type _type))
 (cond
   [(symbol=? type 'bool) (memset data fill-real n-elements _type)]
   [(not fill)] ;; just pass it along
   [(= fill 0) (memset data 0 n-elements _type)]
   [(not (= fill 0))
    (when (or (< fill 0) (> fill 255))
      (error 'make-tensor "fill value should be a byte, but is ~a~n" fill))
    (printf "fill real: ~a~n" fill-real)
    (memset data fill-real n-elements _type)])
 ;; TODO: fill
 (tensor shape type data empty void children))

(define+provide
 (tensor-requires-grad! T)
 (when (empty? (tensor-grad T))
   (set-tensor-grad! T (alloc-tensor (tensor-length T) #:type (symbol->type (tensor-type T))))
   (memset (tensor-grad T) 0 (tensor-length T) (symbol->type (tensor-type T)))))

(define+provide (tensor-zeros shape #:type (type 'double)) (make-tensor shape 0 #:type type))

(module+ test
  (make-tensor (list 100 1) 15 #:type 'uint8)
  (make-tensor (list 100 100) 15 #:type 'uint8)
  (make-tensor (list 100 100) #b0 #:type 'bool)
  (make-tensor (list 100 100) 0 #:type 'double)
  (make-tensor (list 100 100) 0.0 #:type 'cfloat))
