;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         racket/runtime-path
         (for-syntax racket/base
                     racket/syntax
                     syntax/parse))

(define-runtime-path aux-library-path (build-path "build" "libaux"))

(define helper-lib (ffi-lib aux-library-path))

(define-ffi-definer define-ffi-lib-internal helper-lib)

(define-syntax define-ffi+provide
  (syntax-rules ()
    [(_ name body)
     (begin
       (provide name)
       (define-ffi-lib-internal name body))]))

(define-syntax-rule (define-type-group name id ...)
  (define-syntax name (list #'id ...)))

(define _cdouble _double)
(define _cfloat _float)

(define-type-group *all-types*
  _cdouble _cfloat
  _double _float
  _uint8 _int8 _uint16 _int16 _uint32 _int32 _uint64 _int64)

(define-type-group *all-types-without-complex*
  _double _float
  _uint8 _int8 _uint16 _int16 _uint32 _int32 _uint64 _int64)

(define-syntax (define+provide-aux* stx)
  (syntax-parse stx
    [(_ base:id (t:id ...+) sig)
     (with-syntax ([(full ...)
                    (for/list ([type-id (attribute t)])
                      (format-id #'base #:source #'base
                                 "~a~a" #'base type-id))]
                   [(exp ...)
                    (for/list ([type (attribute t)])
                      (define T (syntax->datum type))
                      (let subst ([datum (syntax->datum #'sig)])
                        (cond [(eq? datum 'T) T]
                              [(list? datum) (datum->syntax type (map subst datum))]
                              [else datum])))])
       #'(begin (provide full) ...
                (define-ffi-lib-internal full exp) ...))]
    [(_ base:id var:id sig)
     (let ([ids (syntax-local-value #'var (lambda () #f))])
       (unless (list? ids)
         (raise-syntax-error #f "Not a defined group" #'var))
       (with-syntax ([(t ...) ids])
         #'(define+provide-aux* base (t ...) sig)))]))


(define+provide-aux* _fill *all-types-without-complex*
  (_fun (mem   :  _gcpointer)
        (len   :  _ulong)
        (value : T)
        -> _void))

(define+provide-aux* _fill (_cdouble _cfloat)
  (_fun (mem   :  _gcpointer)
        (len   :  _ulong)
        (real  : T)
        (imag  : T)
        -> _void))

(define+provide-aux* _mul *all-types-without-complex*
  (_fun (out   :  _gcpointer)
        (in1   :  _gcpointer)
        (in2   :  _gcpointer)
        (len   :  _ulong)
        -> _void))

(define+provide-aux* _mul_then_add *all-types-without-complex*
  (_fun (out   :  _gcpointer)
        (in1   :  _gcpointer)
        (in2   :  _gcpointer)
        (len   :  _ulong)
        -> _void))

(define+provide-aux* _tanh (_double _float)
  (_fun (out : _gcpointer)
        (in  : _gcpointer)
        (len : _ulong)
        -> _void))

(define+provide-aux* _sigmoid (_double _float)
  (_fun (out : _gcpointer)
        (in  : _gcpointer)
        (len : _ulong)
        -> _void))

(define-ffi+provide _arange
  (_fun (mem  :  _gcpointer)
        (len  :  _ulong)
        (from : _long)
        (step : _long)
        -> _void))

(define-ffi+provide _arange_double
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (from : _double)
        (step : _double)
        -> _void))

(define+provide-aux* _add_broadcast *all-types-without-complex*
  (_fun (ndim      : _ulong)
        (shape     : (_list i _uint64))
        (A-data    : _gcpointer)
        (A-strides : (_list i _uint64))
        (B-data    : _gcpointer)
        (B-strides : (_list i _uint64))
        (out-data  : _gcpointer)
        -> _void))

(define+provide-aux* _reduce_sum (_double _float)
  (_fun (outer-dim  : _size)
        (reduce-dim : _size)
        (inner-dim  : _size)
        (in-data    : _gcpointer)
        (out-data   : _gcpointer)
        -> _void))

(define+provide-aux* _reduce_sum_generic (_double _float)
  (_fun
   (n-keep             : _size)
   (keep-shape         : (_list i _size))
   (keep-src-strides   : (_list i _size))
   (keep-dst-strides   : (_list i _size))
   (n-reduce           : _size)
   (reduce-shape       : (_list i _size))
   (reduce-src-strides : (_list i _size))
   (src-data           : _gcpointer)
   (dst-data           : _gcpointer)
   -> _void))

(define+provide-aux* _backward_tanh (_double _float)
  (_fun (out-data : _gcpointer)
        (grad-out : _gcpointer)
        (grad-in  : _gcpointer)
        (len      : _size)
        -> _void))
