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
    (case (syntax->datum p)
      [(s)  #'_float]
      [(d)  #'_double]
      [(sc) #'_float]
      [(sd) #'_float]
      [(c)  #'_float]
      [(dz) #'_double]
      [(z)  #'_double]
      [else (error "Unknown BLAS prefix: " (syntax->datum p))])))

(define-syntax (define+provide-cblas* stx)
  (syntax-parse stx
    [(_ base (p ...) sig)
     (with-syntax ([(full ...)
                    (for/list ([cblas-type (attribute p)])
                      (format-id cblas-type "cblas_~a~a" cblas-type #'base))]
                   [(exp ...)
                    (for/list ([cblas-type (attribute p)])
                      (define T (prefix->type cblas-type))
                      (let subst ([datum (syntax->datum #'sig)])
                        (cond [(eq? datum 'T) T]
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
  
  
