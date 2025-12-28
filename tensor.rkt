;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         "unsafe/cblas.rkt"
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
    [(== 'float) (_fill_float data n-elements (real->double-flonum (if fill fill 0.0)))]
    [(== 'cfloat) (_fill_cfloat data n-elements
                                (real->double-flonum (if fill (real-part fill) 0.0))
                                (real->double-flonum (if fill (imag-part fill) 0.0)))]
    [_ (error 'make-tensor "unsupported type: ~a~n" type)])

  (define strides (shape->strides shape))
  (tensor shape strides type data empty void children))

(define+provide (tensor-zeros shape #:type (type 'double))
  (make-tensor shape 0 #:type type))

(define (validate-and-get-fill fill type valid-from valid-to)
  (if (and fill (or (< fill valid-from) (> fill valid-to)))
      (error 'make-tensor "Fill value ~a exceeds the type ~a~n" fill type)
      (if fill fill 0)))

(define+provide (tensor-requires-grad! . tensors)
  (for ([T (in-list tensors)])
    (when (empty? (tensor-grad T))
      (set-tensor-grad! T (alloc-tensor (tensor-length T) #:type (symbol->type (tensor-type T))))
      (memset (tensor-grad T) 0 (tensor-length T) (symbol->type (tensor-type T))))))

(define+provide (tensor-grad-fill T val)
  (match (tensor-type T)
    [(== 'float) (_fill_float (tensor-grad T) (tensor-length T) (real->double-flonum val))]
    [(== 'double) (_fill_double (tensor-grad T) (tensor-length T) (real->double-flonum val))]
    [_ (error 'tensor-grad-fill "only floats and doubles are supported ATM~n")]))

(define+provide (backward! T)
  (define visited (mutable-set))
  (define topo '())
  (define (build-topo! T)
    (unless (set-member? visited T)
      (set-add! visited T)
      (for ([child (tensor-children T)])
        (build-topo! child))
      (set! topo (cons T topo))))
  
  (tensor-requires-grad! T)
  (build-topo! T)
  (tensor-grad-fill T 1.0)
  (for ([tensor topo])
    ((tensor-backward tensor))))

(define+provide (tensor-add A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))
  
  ;; check supported types
  (unless (symbol=? type-A type-B)
    (error 'tensor-add "both tensors should be of the same type, but were: ~a and ~a~n" type-A type-B))
  (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
    (raise-argument-error 'tensor-add "tensor:double or tensor:float" 0 A B))
  (unless (or (symbol=? type-B 'double) (symbol=? type-B 'float))
    (raise-argument-error 'tensor-add "tensor:double or tensor:float" 1 A B))
  
  ;; check equal dimensions
  (unless (equal? (tensor-shape A)
                  (tensor-shape B))
    (error  'tensor-add "tensor shapes should equal, but are: ~a vs ~a" (tensor-shape A) (tensor-shape B)))
  
  (define out (make-tensor (tensor-shape A) #:type type-A #:children (list A B)))
  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (define len (apply * (tensor-shape A)))

  (cond [(symbol=? type-A 'double)
         (cblas_dcopy len (tensor-data B) 1 (tensor-data out) 1)
         (cblas_daxpy len 1.0 (tensor-data A) 1 (tensor-data out) 1)]
        [else
         (cblas_scopy len (tensor-data B) 1 (tensor-data out) 1)
         (cblas_saxpy len 1.0 (tensor-data A) 1 (tensor-data out) 1)])

  (define backward    
    (cond [(symbol=? type-A 'double)
           (lambda ()
             (cblas_daxpy len 1.0 (tensor-grad out) 1 (tensor-grad A) 1)
             (cblas_daxpy len 1.0 (tensor-grad out) 1 (tensor-grad B) 1))]
          [else
           (lambda ()
             (cblas_saxpy len 1.0 (tensor-grad out) 1 (tensor-grad A) 1)
             (cblas_saxpy len 1.0 (tensor-grad out) 1 (tensor-grad B) 1))]))

  (when grad?
    (set-tensor-backward! out backward))
  out)

(module+ test-tensor-add
  (define A (make-tensor (list 3 3) 1.0 #:type 'double))
  (define B (make-tensor (list 3 3) 2.0 #:type 'double))
  (tensor-requires-grad! A B)
  (define C (tensor-add A B))
  C
  (backward! C)
  )

(module+ another-test
  (make-tensor (list 100 1) 15 #:type 'uint8)
  (make-tensor (list 3 3) 1.0 #:type 'double)
  (make-tensor (list 100 100) -128 #:type 'int8)
  (make-tensor (list 100 100) 0 #:type 'int64)
  (make-tensor (list 3 3) (- (- (expt 2 63)) 0) #:type 'int64)
  (make-tensor (list 3 3) 18446744073709551615 #:type 'uint64)
  (make-tensor (list 100 100) #t #:type 'bool)
  (make-tensor (list 100 100) 0.15 #:type 'double)
  (make-tensor (list 100 100) 0.0+1i #:type 'cfloat)
  (make-tensor (list 100 100) #:type 'cdouble)
  (make-tensor (list 100 100)))
