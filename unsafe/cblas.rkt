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
      (for/list ([c (symbol->string (syntax->datum syntx))])
        (string->symbol (string c))))
    
    (define prefix (syntax->symbols p))
    (when (> (length prefix) 2)
      (error "BLAS prefix cannot have more than 2 types: " (syntax->datum p)))
    (for/fold ([T #f] [R #f])
              ([p prefix])
      (define type (prefix->type p))
      (if T
          (values T type)
          (values type #f)))))

(define-syntax (define+provide-cblas* stx)
  (syntax-parse stx
    [(_ base:id (p:id ...) sig)
     (with-syntax ([(full ...)
                    (let ()
                      (define cblas-types (attribute p))
                      (when (zero? (length cblas-types))
                        (error 'define "Expected at least one cblas type"))
                      (for/list ([cblas-type cblas-types])
                        (format-id cblas-type #:source cblas-type
                                   "cblas_~a~a" cblas-type #'base)))]
                   [(exp ...)
                    (for/list ([cblas-type (attribute p)])
                      (define-values (T R) (prefix->types cblas-type))
                      (let subst ([datum (syntax->datum #'sig)])
                        (cond [(eq? datum 'T) T]
                              [(eq? datum 'R)
                               (unless R
                                 (error 'define+provide-cblas*
                                        "Cannot use 'R without providing a cblas return type (e.g. sc)"))
                               R]
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

(module+ test
  (require rackunit
           ffi/unsafe
           ffi/cvector)

  ;; --- Helpers for Memory Management ---
  
  ;; Allocate a C float array from a Racket list
  (define (f32-vector . args)
    (list->cvector args _float))
  
  ;; Allocate a C double array from a Racket list
  (define (f64-vector . args)
    (list->cvector args _double))

  ;; Extract Racket list from C pointer using standard pointer arithmetic
  (define (ptr->f32-list ptr len)
    (for/list ([i (in-range len)])
      (ptr-ref ptr _float i)))

  (define (ptr->f64-list ptr len)
    (for/list ([i (in-range len)])
      (ptr-ref ptr _double i)))

  ;; --- Tests ---
  (test-case "ASUM: Sum of absolute values"
    ;; sasum: float input
    (define s-vec (f32-vector 1.0 -2.0 3.0))
    (check-= (cblas_sasum 3 (cvector-ptr s-vec) 1) 6.0 0.001)

    ;; dasum: double input
    (define d-vec (f64-vector 1.0 -2.0 3.0))
    (check-= (cblas_dasum 3 (cvector-ptr d-vec) 1) 6.0 0.001)

    ;; scasum: Complex float input (Interleaved real/imag)
    ;; NOTE: BLAS computes sum of components (|Re| + |Im|), NOT Euclidean norm.
    ;; Vector: [3+4i, 0.0+0.0i]. Result: |3| + |4| = 7.0
    (define sc-vec (f32-vector 3.0 4.0 0.0 0.0))
    (check-= (cblas_scasum 2 (cvector-ptr sc-vec) 1) 7.0 0.001)
    
    ;; dzasum: Complex double input
    (define dz-vec (f64-vector 3.0 4.0)) 
    (check-= (cblas_dzasum 1 (cvector-ptr dz-vec) 1) 7.0 0.001))

  (test-case "AXPY: Y = alpha*X + Y"
    ;; saxpy: Y = 2.0 * X + Y
    (define sa-X (f32-vector 1.0 2.0))
    (define sa-Y (f32-vector 10.0 20.0))
    (cblas_saxpy 2 2.0 (cvector-ptr sa-X) 1 (cvector-ptr sa-Y) 1)
    (check-equal? (ptr->f32-list (cvector-ptr sa-Y) 2) '(12.0 24.0))

    ;; daxpy: Double precision
    (define da-X (f64-vector 1.0 2.0))
    (define da-Y (f64-vector 10.0 20.0))
    (cblas_daxpy 2 3.0 (cvector-ptr da-X) 1 (cvector-ptr da-Y) 1)
    (check-equal? (ptr->f64-list (cvector-ptr da-Y) 2) '(13.0 26.0)))

  (test-case "COPY: Y = X"
    ;; scopy
    (define sc-X (f32-vector 5.0 6.0))
    (define sc-Y (f32-vector 0.0 0.0))
    (cblas_scopy 2 (cvector-ptr sc-X) 1 (cvector-ptr sc-Y) 1)
    (check-equal? (ptr->f32-list (cvector-ptr sc-Y) 2) '(5.0 6.0))

    ;; zcopy (Complex double copy)
    ;; Copy 1 complex number (2 doubles)
    (define zc-X (f64-vector 1.5 -1.5))
    (define zc-Y (f64-vector 0.0 0.0))
    (cblas_zcopy 1 (cvector-ptr zc-X) 1 (cvector-ptr zc-Y) 1)
    (check-equal? (ptr->f64-list (cvector-ptr zc-Y) 2) '(1.5 -1.5)))

  (test-case "DOT: Dot Product"
    ;; sdot: Single precision
    ;; [1, 2] . [3, 4] = 3 + 8 = 11
    (define sd-X (f32-vector 1.0 2.0))
    (define sd-Y (f32-vector 3.0 4.0))
    (check-= (cblas_sdot 2 (cvector-ptr sd-X) 1 (cvector-ptr sd-Y) 1) 11.0 0.001)

    ;; ddot: Double precision
    (define dd-X (f64-vector 1.0 2.0))
    (define dd-Y (f64-vector 3.0 4.0))
    (check-= (cblas_ddot 2 (cvector-ptr dd-X) 1 (cvector-ptr dd-Y) 1) 11.0 0.001))

  (test-case "Extended DOT variants"
    ;; cblas_dsdot: Computes dot product of float arrays with double precision accumulation
    ;; Inputs must be float (f32), result is double (f64)
    (define ds-X (f32-vector 1.1 2.2))
    (define ds-Y (f32-vector 3.3 4.4))
    ;; 1.1*3.3 + 2.2*4.4 = 3.63 + 9.68 = 13.31
    (check-= (cblas_dsdot 2 (cvector-ptr ds-X) 1 (cvector-ptr ds-Y) 1) 13.31 0.00001)

    ;; cblas_sdsdot: float dot product plus scalar `b`
    ;; Result = b + X.Y
    (define sds-X (f32-vector 1.0 2.0))
    (define sds-Y (f32-vector 3.0 4.0))
    (define sb 10.0)
    ;; 10 + (3+8) = 21
    (check-= (cblas_sdsdot 2 sb (cvector-ptr sds-X) 1 (cvector-ptr sds-Y) 1) 21.0 0.001))

  (test-case "SCAL: X = alpha * X"
    ;; sscal: Scale float array by float
    (define ss-X (f32-vector 1.0 2.0))
    (cblas_sscal 2 2.0 (cvector-ptr ss-X) 1)
    (check-equal? (ptr->f32-list (cvector-ptr ss-X) 2) '(2.0 4.0))

    ;; csscal: Scale complex float array by real float
    ;; [1+2i] * 2 -> [2+4i]
    (define cs-X (f32-vector 1.0 2.0))
    (cblas_csscal 1 2.0 (cvector-ptr cs-X) 1)
    (check-equal? (ptr->f32-list (cvector-ptr cs-X) 2) '(2.0 4.0))

    ;; cscal: Scale complex float array by complex float
    ;; IMPORTANT: Alpha is passed by pointer here
    ;; [1+1i] * [0+1i] = -1 + 1i
    (define c-X (f32-vector 1.0 1.0))
    (define c-alpha (f32-vector 0.0 1.0)) ;; i
    (cblas_cscal 1 (cvector-ptr c-alpha) (cvector-ptr c-X) 1)
    (check-equal? (ptr->f32-list (cvector-ptr c-X) 2) '(-1.0 1.0))

    ;; zscal: Scale complex double array by complex double
    ;; [2+0i] * [3+0i] = [6+0i]
    (define z-X (f64-vector 2.0 0.0))
    (define z-alpha (f64-vector 3.0 0.0))
    (cblas_zscal 1 (cvector-ptr z-alpha) (cvector-ptr z-X) 1)
    (check-equal? (ptr->f64-list (cvector-ptr z-X) 2) '(6.0 0.0)))
  )
