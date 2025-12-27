;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         ffi/unsafe/define
         racket/runtime-path)

(define-runtime-path aux-library-path (build-path "build" "libaux"))

(define helper-lib (ffi-lib aux-library-path))

(define-ffi-definer define-ffi-lib-internal helper-lib)

(define-syntax define-ffi+provide
  (syntax-rules ()
    [(_ name body)
     (begin
       (provide name)
       (define-ffi-lib-internal name body))]))

(define-ffi+provide _fill_double
  (_fun (mem   :  _gcpointer)
        (len   :  _ulong)
        (value : _double)
        -> _void))

(define-ffi+provide _fill_cdouble
  (_fun (mem   :  _gcpointer)
        (len   :  _ulong)
        (real  : _double)
        (imag  : _double)
        -> _void))

(define-ffi+provide _fill_float
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _float)
        -> _void))

(define-ffi+provide _fill_cfloat
  (_fun (mem  :  _gcpointer)
        (len  :  _ulong)
        (real : _float)
        (imag : _float)
        -> _void))

(define-ffi+provide _fill_uint8_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _uint8)
        -> _void))

(define-ffi+provide _fill_int8_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _int8)
        -> _void))

(define-ffi+provide _fill_uint16_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _uint16)
        -> _void))

(define-ffi+provide _fill_int16_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _int16)
        -> _void))

(define-ffi+provide _fill_uint32_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _uint32)
        -> _void))

(define-ffi+provide _fill_int32_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _int32)
        -> _void))

(define-ffi+provide _fill_uint64_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _uint64)
        -> _void))

(define-ffi+provide _fill_int64_t
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (value : _int64)
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

