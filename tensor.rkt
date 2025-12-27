;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         "./unsafe/tensor.rkt"
         "./private/utilities.rkt"
         "./C/aux.rkt")

(define+provide (tensor-length A) (apply * (tensor-shape A)))

(define+provide (tensor-rank A) (length (tensor-shape A)))

(define+provide (make-tensor shape (fill #f) #:type (type 'double) #:children (children empty))
  (when (for/or ([dim-length shape])
          (<= dim-length 0))
    (error 'make-tensor "dimensions should be >= 1"))
  (define _type (symbol->type type))

  ;; alloc
  (define n-elements (apply * shape))
  (when (or (symbol=? type 'cfloat)
            (symbol=? type 'cdouble))
    (set! n-elements (* n-elements 2)))

  (define data (alloc-tensor n-elements #:type _type))

  ;; fill
  (match type
    [(== 'bool)
     (unless (or (number? fill) (boolean? fill))
       (error 'make-tensor "Expected a boolean or a number for for the fill value, but got ~a~n" fill))
     (memset data (if (or (and (number? fill) (not (zero? fill)))
                          (and (boolean? fill) fill))
                      1 0)
             n-elements _type)]
    [(== 'int8) (_fill_int8_t data n-elements (validate-and-get-fill fill type -128 127))]
    [(== 'uint8) (_fill_uint8_t data n-elements (validate-and-get-fill fill type 0 255))]
    [(== 'int16) (_fill_int16_t data n-elements (validate-and-get-fill fill type -32768 32767))]
    [(== 'uint16) (_fill_uint16_t data n-elements (validate-and-get-fill fill type 0 65535))]
    [(== 'int32) (_fill_int32_t data n-elements (validate-and-get-fill fill type -2147483648 2147483647))]
    [(== 'uint32) (_fill_uint32_t data n-elements (validate-and-get-fill fill type 0 4294967295))]
    [(== 'int64) (_fill_int64_t data n-elements
                                (validate-and-get-fill fill type -9223372036854775808 9223372036854775807))]
    [(== 'uint64) (_fill_uint64_t data n-elements (validate-and-get-fill fill type 0 18446744073709551615))]
    [(== 'double) (_fill_double data n-elements (real->double-flonum (if fill fill 0.0)))]
    [(== 'cdouble) (_fill_cdouble data n-elements
                                  (real->double-flonum (if fill (real-part fill) 0.0))
                                  (real->double-flonum (if fill (imag-part fill) 0.0)))]
    [(== 'float) (_fill_float data n-elements (real->single-flonum (if fill fill 0.0)))]
    [(== 'cfloat) (_fill_cfloat data n-elements
                                (real->double-flonum (if fill (real-part fill) 0.0))
                                (real->double-flonum (if fill (imag-part fill) 0.0)))]
    [_ (error 'make-tensor "unsupported type: ~a~n" type)])

  (define strides (shape->strides shape))
  (tensor shape strides type data empty void children))

(define+provide (tensor-requires-grad! T)
  (when (empty? (tensor-grad T))
    (set-tensor-grad! T (alloc-tensor (tensor-length T) #:type (symbol->type (tensor-type T))))
    (memset (tensor-grad T) 0 (tensor-length T) (symbol->type (tensor-type T)))))

(define+provide (tensor-zeros shape #:type (type 'double))
  (make-tensor shape 0 #:type type))

(define (validate-and-get-fill fill type valid-from valid-to)
  (if (and fill (or (< fill valid-from) (> fill valid-to)))
      (error 'make-tensor "Fill value ~a exceeds the type ~a~n" fill type)
      (if fill fill 0)))

(module+ test
  (make-tensor (list 100 1) 15 #:type 'uint8)
  (make-tensor (list 100 100) -128 #:type 'int8)
  (make-tensor (list 100 100) 0 #:type 'int64)
  (make-tensor (list 3 3) (- (- (expt 2 63)) 0) #:type 'int64)
  (make-tensor (list 3 3) 18446744073709551615 #:type 'uint64)
  (make-tensor (list 100 100) #t #:type 'bool)
  (make-tensor (list 100 100) 0.15 #:type 'double)
  (make-tensor (list 100 100) 0.0+1i #:type 'cfloat)
  (make-tensor (list 100 100) #:type 'cdouble)
  (make-tensor (list 100 100)))
