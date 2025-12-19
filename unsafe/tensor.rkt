;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         racket/struct
         "../private/utilities.rkt")

(define (tensor->print-shorted-lists A)
  (define shape (tensor-shape A))
  (define type  (tensor-type A))
  (define data  (tensor-data A))
  (define grad  (tensor-grad A))
  (define _type (symbol->type type))
  
  (define (dim-width->lof-indices data dim-width offset (lookup? #f))
    (define start-addresses (if (< dim-width 10)
                                (range dim-width)
                                (append (range 3)
                                        '(...)
                                        (range (- dim-width 3) dim-width))))
    (for/list ([start-address start-addresses])
      (cond [(or (symbol? start-address) (symbol? offset)) '...]
            [else
             (define address (+ start-address offset))
             (if (not lookup?)
                 address
                 (if (or (symbol=? type 'cdouble) (symbol=? type 'cfloat))
                     (let ([real (ptr-ref data _type (* address 2))]
                           [imag (ptr-ref data _type (+ (* address 2) 1))])
                       (+ real (* imag 0+1i)))
                     (ptr-ref data _type address)))])))
  

  (define (assemble-indices data shape offset)
    (cond [(empty? shape) empty]
          [(= 1 (length shape))
           (dim-width->lof-indices data (car shape) offset #t)]
          [else
           (define offsets(dim-width->lof-indices data (car shape) 0))
           (for/list [(o offsets)]
             (if (symbol? o)
                 '...
                 (assemble-indices data (cdr shape)
                                   (+ offset (* o (apply * (cdr shape)))))))]))

  (if (empty? grad)
      (list (assemble-indices data shape 0))
      (list (assemble-indices data shape 0)
            (assemble-indices grad shape 0))))


(define+provide (type->symbol a-type)
  (match a-type
    [(== _int64)      'int64]
    [(== _uint64)     'uint64]
    [(== _int32)      'int32]
    [(== _uint32)     'uint32]
    [(== _int16)      'int16]
    [(== _uint16)     'uint16]
    [(== _int8)       'int8]
    [(== _uint8)      'uint8]
    [(== _double)     'double]
    [(== _longdouble) 'longdouble]
    [(== _float)      'float]
    [(== _stdbool)     'bool]
    [_ (error 'type->symbol "unsupported type: ~a~n" a-type)]))


(define+provide (symbol->type a-symbol)
  (match a-symbol
    [(== 'int64)      _int64]
    [(== 'uint64)     _uint64]
    [(== 'int32)      _int32]
    [(== 'uint32)     _uint32]
    [(== 'int16)      _int16]
    [(== 'uint16)     _uint16]
    [(== 'int8)       _int8]
    [(== 'uint8)      _uint8]
    [(== 'longdouble) _longdouble]
    [(== 'double)     _double]
    [(== 'float)      _float]
    [(== 'cfloat)     _float]
    [(== 'cdouble)    _double]
    [(== 'bool)       _stdbool]
    [_ (error 'symbol->type "unsupported type: ~a~n" a-symbol)]))

(struct tensor (shape strides type data grad backward children)
  #:mutable
  #:methods gen:custom-write
  [(define write-proc
     (make-constructor-style-printer
      (lambda (A)
        (define shape (tensor-shape A))
        (define type (tensor-type A))
        (define strides (tensor-strides A))
        (define data-grad (tensor->print-shorted-lists A))
        ;; TODO: print correctly now using strides
        (if (= 1 (length data-grad))
            (format "tensor:~a ~a~n  ~a~n" type shape (car data-grad))
            (format "tensor:~a ~a~n  d: ~a~n  ∇: ~a~n" type shape (car data-grad) (cadr data-grad))))
      (lambda (_) empty)))])

(provide (struct-out tensor))

(define+provide (alloc-tensor n-elements #:type (type _double))
  ;; align the data to 8 bytes for booleans
  (when (eq? type _stdbool)
    (define missing-bytes (modulo n-elements 8))
    (unless (zero? missing-bytes)
      (set! n-elements (+ n-elements missing-bytes))))
  (cast (malloc n-elements type 'atomic-interior)
        _pointer
        _gcpointer))

(define+provide (shape->strides shape)
  (for/fold ([strides '()]
             [running-product 1]
             #:result strides)
            ([dim (in-list (reverse shape))])
    (values (cons running-product strides)
            (* running-product dim))))

