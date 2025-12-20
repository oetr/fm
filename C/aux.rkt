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

(define-ffi+provide _arange
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (from : _long)
        (step : _long)
        -> _void))

(define-ffi+provide _arange_double
  (_fun (mem :  _gcpointer)
        (len :  _ulong)
        (from : _double)
        (step : _double)
        -> _void))

