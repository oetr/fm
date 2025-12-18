#lang racket/base

(require ffi/unsafe
         ffi/unsafe/define
         "../private/utilities.rkt"
         (for-syntax racket/base   
                     racket/syntax 
                     syntax/parse))

(define cblas-lib (ffi-lib "libblas" #:fail (λ () #f)))
(define-ffi-definer define-cblas-lib-internal cblas-lib)

;; This must be begin-for-syntax so the macro can call it at compile-time
(begin-for-syntax
  (define (prefix->type p)
    (case p
      [(s)  #'_float]
      [(d)  #'_double]
      [(c)  #'_float]
      [(z)  #'_double]
      [else (error "Unknown BLAS prefix: " p)]))
  
  (define (prefix->types p)
    (define (syntax->symbols syntx)
      (define str (symbol->string (syntax->datum syntx)))
      (for/list ([i (in-range (string-length str))])
        (string->symbol (substring str i (+ i 1)))))
    (define prefix (syntax->symbols p))
    (when (> (length prefix) 2)
      (error "BLAS prefix cannot have more than 2 types: " (syntax->datum p)))
    (for/fold ([T #f]
               [R #f])
              ([p prefix])
      (define type (prefix->type p))
      (if T
          (values T type)
          (values type #f)))))

(define-syntax (define+provide-cblas* stx)
  (syntax-parse stx
    [(_ base:id (p:id ...) sig)
     (with-syntax ([(full ...)
                    (for/list ([cblas-type (attribute p)])
                      (format-id cblas-type "cblas_~a~a" cblas-type #'base))]
                   [(exp ...)
                    (for/list ([cblas-type (attribute p)])
                      (define-values (T R) (prefix->types cblas-type))
                      (let subst ([datum (syntax->datum #'sig)])
                        (cond [(eq? datum 'T) T]
                              [(eq? datum 'R) R]
                              [(list? datum) (datum->syntax cblas-type (map subst datum))]
                              [else datum])))])
       #'(begin (provide full) ... (define-cblas-lib-internal full exp) ...))]))

(define+provide-cblas* asum (s d sc dz)
  (_fun (n    : _int)
        (X    : _pointer)
        (incX : _int)
        -> T))

(define+provide-cblas* axpy (s d)
  (_fun (n    : _int)
        (a    : T)
        (X    : _pointer)
        (incX : _int)
        (Y    : _pointer)
        (incY : _int)
        -> _void))

(define+provide-cblas* copy (s d c z)
  (_fun (n    : _int)
        (X    : _pointer)
        (incX : _int)
        (Y    : _pointer)
        (incY : _int)
        -> _void))

(define+provide-cblas* dot (s d)
  (_fun (n    : _int)
        (X    : _pointer)
        (incX : _int)
        (Y    : _pointer)
        (incY : _int)
        -> T))

(define+provide-cblas* sdot (sd)
  (_fun (n    : _int)
        (sb   : _float)   ; Single precision scalar to be added to inner product
        (sX   : _pointer)
        (incX : _int)
        (sY   : _pointer)
        (incY : _int)
        -> T))

(define+provide-cblas* sdot (d)
  (_fun (n    : _int)
        (sX   : _pointer)
        (incX : _int)
        (sY   : _pointer)
        (incY : _int)
        -> T))

(define+provide-cblas* scal (s d cs zd)
  (_fun (n    : _int)
        (a    : T)
        (X    : _pointer)
        (incX : _int)
        -> _void))

(define+provide-cblas* scal (c z)
  (_fun (n    : _int)
        (a    : _pointer)
        (X    : _pointer)
        (incX : _int)
        -> _void))
