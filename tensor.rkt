;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         "unsafe/cblas.rkt"
         "./unsafe/tensor.rkt"
         "./private/utilities.rkt"
         "./C/aux.rkt")

(define+provide (tensor-length A) (for/product ([dim-len (tensor-shape A)]) dim-len))

(define+provide (tensor-rank A) (length (tensor-shape A)))

(define+provide (make-tensor shape (fill #f) #:type (type 'double) #:children (children empty))
  (when (for/or ([dim-length shape])
          (<= dim-length 0))
    (error 'make-tensor "dimensions should be >= 1"))
  (define _type (symbol->type type))

  ;; alloc
  (define n-elements (for/product ([s shape]) s))
  (when (or (symbol=? type 'cfloat)
            (symbol=? type 'cdouble))
    (set! n-elements (* n-elements 2)))

  (define data (alloc-tensor n-elements #:type _type))
  (tensor-fill-data! data type n-elements fill)

  (define strides (shape->strides shape))
  (tensor shape strides type data empty void children))


(define (tensor-fill-data! data type n-elements fill)
  (match type
    [(== 'bool)
     (unless (or (number? fill) (boolean? fill))
       (error 'make-tensor "Expected a boolean or a number for the fill value, but got ~a~n" fill))
     ;; Logic: If number != 0 -> true; If boolean true -> true; else 0
     (define byte-val (if (or (and (number? fill) (not (zero? fill)))
                              (and (boolean? fill) fill))
                          1 0))
     (memset data byte-val n-elements (symbol->type type))]
    [(== 'int8)   (_fill_int8 data n-elements (validate-and-get-fill fill type -128 127))]
    [(== 'uint8)  (_fill_uint8 data n-elements (validate-and-get-fill fill type 0 255))]
    [(== 'int16)  (_fill_int16 data n-elements (validate-and-get-fill fill type -32768 32767))]
    [(== 'uint16) (_fill_uint16 data n-elements (validate-and-get-fill fill type 0 65535))]
    [(== 'int32)  (_fill_int32 data n-elements (validate-and-get-fill fill type -2147483648 2147483647))]
    [(== 'uint32) (_fill_uint32 data n-elements (validate-and-get-fill fill type 0 4294967295))]
    [(== 'int64)  (_fill_int64 data n-elements
                               (validate-and-get-fill fill type -9223372036854775808 9223372036854775807))]
    [(== 'uint64) (_fill_uint64 data n-elements (validate-and-get-fill fill type 0 18446744073709551615))]
    [(== 'double) (_fill_double data n-elements (real->double-flonum (if fill fill 0.0)))]
    [(== 'float)  (_fill_float data n-elements (real->double-flonum (if fill fill 0.0)))]
    [(== 'cdouble)
     (_fill_cdouble data n-elements
                    (real->double-flonum (if fill (real-part fill) 0.0))
                    (real->double-flonum (if fill (imag-part fill) 0.0)))]
    [(== 'cfloat)
     (_fill_cfloat data n-elements
                   (real->double-flonum (if fill (real-part fill) 0.0))
                   (real->double-flonum (if fill (imag-part fill) 0.0)))]
    [_ (error 'make-tensor "unsupported type: ~a~n" type)]))

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

;; -----------------------------------------------------------------------------
;; Broadcasting Helpers
;; -----------------------------------------------------------------------------

(define (compatible-shapes? shape-A shape-B)
  (equal? shape-A shape-B)) 

(define (broadcast-compatible? shape-A shape-B)
  ;; Simplified broadcasting:
  ;; 1. If shapes are equal -> compatible
  ;; 2. If rank(A) == 2 and rank(B) == 1 and B dim matches last dim of A -> compatible (add vector to rows)
  ;; This handles the bias addition case (Matrix + Vector)
  (or (equal? shape-A shape-B)
      (and (= (length shape-A) 2)
           (= (length shape-B) 1)
           (= (second shape-A) (first shape-B)))))

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
  
  ;; check dimensions
  (unless (broadcast-compatible? (tensor-shape A) (tensor-shape B))
    (error  'tensor-add "tensor shapes incompatible: ~a vs ~a" (tensor-shape A) (tensor-shape B)))
  
  (define broadcast? (not (equal? (tensor-shape A) (tensor-shape B))))
  
  (define out (make-tensor (tensor-shape A) #:type type-A #:children (list A B)))
  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (define len (tensor-length A))

  ;; Forward Pass
  (cond 
    [(not broadcast?)
     ;; Standard element-wise add
     (let-typed type-A
       ([copy cblas_dcopy cblas_scopy]
        [axpy cblas_daxpy cblas_saxpy])
       (copy len (tensor-data B) 1 (tensor-data out) 1)
       (axpy len 1.0 (tensor-data A) 1 (tensor-data out) 1))]
    [else
     ;; Broadcasting: A is (M, N), B is (N)
     ;; We add B to every row of A
     (match-define (list M N) (tensor-shape A))
     ;; First copy A to Out
     (let-typed type-A
       ([copy cblas_dcopy cblas_scopy])
       (copy len (tensor-data A) 1 (tensor-data out) 1))
     
     ;; Now add B to every row of Out
     ;; We can do this by looping M times and axpy-ing B into the row pointer
     (define type-size (if (symbol=? type-A 'double) 8 4))
     (let-typed type-A
       ([axpy cblas_daxpy cblas_saxpy])
       (for ([i (in-range M)])
         (define offset (* i N type-size))
         (axpy N 1.0 (tensor-data B) 1 (ptr-add (tensor-data out) offset) 1)))])

  (define backward    
    (cond [(not broadcast?)
           (let-typed type-A
             ([axpy cblas_daxpy cblas_saxpy])
             (lambda ()
               (axpy len 1.0 (tensor-grad out) 1 (tensor-grad A) 1)
               (axpy len 1.0 (tensor-grad out) 1 (tensor-grad B) 1)))]
          [else
           ;; Backward for Broadcast
           ;; dOut -> dA is Identity (since Out = A + broadcast(B))
           ;; dOut -> dB is sum over rows (since B was added to every row)
           (match-define (list M N) (tensor-shape A))
           (lambda ()
             ;; dA += dOut
             (let-typed type-A
               ([axpy cblas_daxpy cblas_saxpy])
               (axpy len 1.0 (tensor-grad out) 1 (tensor-grad A) 1)
               
               ;; dB += sum_rows(dOut)
               ;; We can loop and accumulate dOut rows into dB
               (define type-size (if (symbol=? type-A 'double) 8 4))
               (for ([i (in-range M)])
                 (define offset (* i N type-size))
                 (axpy N 1.0 (ptr-add (tensor-grad out) offset) 1 (tensor-grad B) 1))))]))

  (when grad?
    (set-tensor-backward! out backward))
  out)


(define+provide (tensor-mul A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))
  ;; check supported types
  (unless (symbol=? type-A type-B)
    (error 'tensor-mul "both tensors should be of the same type, but were: ~a and ~a~n" type-A type-B))
  (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
    (raise-argument-error 'tensor-mul "tensor:double or tensor:float" 0 A B))
  (unless (or (symbol=? type-B 'double) (symbol=? type-B 'float))
    (raise-argument-error 'tensor-mul "tensor:double or tensor:float" 1 A B))
  (define out (make-tensor (tensor-shape A) #:type type-A #:children (list A B)))
  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (let-typed type-A
    ([mul! _mul_double _mul_float])
    (mul! (tensor-data out) (tensor-data A) (tensor-data B) (tensor-length A)))
  
  (define (backward)
    (let-typed type-A
      ([mul-then-add! _mul_then_add_double _mul_then_add_float])
      (mul-then-add! (tensor-grad A)
                     (tensor-data B) (tensor-grad out) (tensor-length A))
      (mul-then-add! (tensor-grad B)
                     (tensor-data A) (tensor-grad out) (tensor-length A))))

  (when grad?
    (set-tensor-backward! out backward))
  out)

(define+provide (tensor-matmul A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))
  
  (unless (symbol=? type-A type-B)
    (error 'tensor-matmul "tensors must have same type, got: ~a and ~a" type-A type-B))
  (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
    (raise-argument-error 'tensor-matmul "tensor:double or tensor:float" 0 A B))

  ;; For now, assume 2D matrices
  (when (or (not (= (tensor-rank A) 2)) (not (= (tensor-rank B) 2)))
    (error 'tensor-matmul "only 2D matrices supported currently"))

  (match-define (list M K) (tensor-shape A))
  (match-define (list K2 N) (tensor-shape B))
  
  (unless (= K K2)
    (error 'tensor-matmul "shapes incompatible for matmul: ~a vs ~a" (tensor-shape A) (tensor-shape B)))

  (define out (make-tensor (list M N) #:type type-A #:children (list A B)))

  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (let-typed type-A
    ([gemm cblas_dgemm cblas_sgemm])
    ;; C = alpha*A*B + beta*C
    ;; We want C = 1*A*B + 0*C
    ;; RowMajor = 101, NoTrans = 111
    (gemm 101 111 111 M N K 1.0 (tensor-data A) K (tensor-data B) N 0.0 (tensor-data out) N))

  (define (backward)
    (let-typed type-A
      ([gemm cblas_dgemm cblas_sgemm])
      ;; dA += dC * B^T
      (gemm 101 111 112 M K N 1.0 (tensor-grad out) N (tensor-data B) N 1.0 (tensor-grad A) K)
      ;; dB += A^T * dC
      (gemm 101 112 111 K N M 1.0 (tensor-data A) K (tensor-grad out) N 1.0 (tensor-grad B) N)))

  (when grad?
    (set-tensor-backward! out backward))
  out)

(define+provide (tensor-tanh T)
  (define type (tensor-type T))
  (unless (or (symbol=? type 'double) (symbol=? type 'float))
    (raise-argument-error 'tensor-tanh "tensor:double or tensor:float" 0 T))

  (define out (make-tensor (tensor-shape T) #:type type #:children (list T)))

  (when (not (empty? (tensor-grad T)))
    (tensor-requires-grad! out))

  (let-typed type
    ([tanh! _tanh_double _tanh_float])
    (tanh! (tensor-data out) (tensor-data T) (tensor-length T)))

  ;; d/dx tanh(x) = 1 - tanh(x)^2 = 1 - out^2
  (define (backward)
    ;; We need to compute: grad_T += grad_out * (1 - out^2)
    ;; Ideally, we would have a vectorized "mul_one_minus_square" kernel.
    ;; For now, we'll do this in Racket loop or using basic ops if possible.
    ;; But wait! We don't have a "1 - x^2" op exposed yet.
    ;; Let's implement a simple loop here for now, or add a C helper.
    ;; Given the performance goals, let's assume we should add a C helper or use existing primitives.
    ;; But I'll use a Racket loop for immediate correctness, then we can optimize.
    ;; Actually, better: let's use the fact that we have data access.
    ;; But accessing pointers in Racket loop is slow.
    ;; I'll stick to a simple loop for now to verify logic.
    
    (define len (tensor-length T))
    (define g-out (tensor-grad out))
    (define g-in (tensor-grad T))
    (define val-out (tensor-data out))
    
    (define (bwd-cpu-double)
      (for ([i (in-range len)])
        (define y (ptr-ref val-out _double i))
        (define dy (ptr-ref g-out _double i))
        (define dx (* dy (- 1.0 (* y y))))
        (define cur-dx (ptr-ref g-in _double i))
        (ptr-set! g-in _double i (+ cur-dx dx))))

    (define (bwd-cpu-float)
      (for ([i (in-range len)])
        (define y (ptr-ref val-out _float i))
        (define dy (ptr-ref g-out _float i))
        (define dx (* dy (- 1.0 (* y y))))
        (define cur-dx (ptr-ref g-in _float i))
        (ptr-set! g-in _float i (+ cur-dx dx))))

    (if (symbol=? type 'double) (bwd-cpu-double) (bwd-cpu-float)))

  (when (not (empty? (tensor-grad T)))
    (set-tensor-backward! out backward))
  out)

(module+ test
  (require rackunit)

  (test-case "tensor-tanh"
    (define A (make-tensor (list 2) 0.0 #:type 'double))
    (tensor-requires-grad! A)
    (define B (tensor-tanh A))
    ;; tanh(0) = 0
    (check-= (ptr-ref (tensor-data B) _double 0) 0.0 1e-5)
    
    (backward! B)
    ;; d/dx tanh(x) at 0 = 1 - 0^2 = 1
    (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5))

  (test-case "tensor-add-broadcast"
    (define A (make-tensor (list 2 2) 1.0 #:type 'double))
    (define B (make-tensor (list 2) 2.0 #:type 'double)) ;; Vector of 2s
    (tensor-requires-grad! A B)
    
    ;; C = A + B (broadcast)
    ;; [[1,1],[1,1]] + [2,2] -> [[3,3],[3,3]]
    (define C (tensor-add A B))
    (define data (tensor-data C))
    (check-= (ptr-ref data _double 0) 3.0 1e-5)
    (check-= (ptr-ref data _double 1) 3.0 1e-5)
    (check-= (ptr-ref data _double 2) 3.0 1e-5)
    (check-= (ptr-ref data _double 3) 3.0 1e-5)
    
    (backward! C)
    ;; dC/dA = 1. dC/dB = sum_rows(1) = 2 (since A has 2 rows)
    ;; Grad C is filled with 1.0 by default in backward!
    
    ;; Grad A should be 1.0
    (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5)
    
    ;; Grad B should be 2.0 (sum of gradients from both rows)
    (check-= (ptr-ref (tensor-grad B) _double 0) 2.0 1e-5)
    (check-= (ptr-ref (tensor-grad B) _double 1) 2.0 1e-5))

  (test-case "tensor-add"
    (define A (make-tensor (list 3 3) 1.0 #:type 'double))
    (define B (make-tensor (list 3 3) 2.0 #:type 'double))
    (tensor-requires-grad! A B)
    (define C (tensor-add A B))
    (define D (tensor-mul A C))
    (backward! D)
    (check-true (tensor? D)))

  (test-case "tensor-matmul"
    (define A (make-tensor '(2 3) 1.0 #:type 'double)) ;; 2x3 matrix of 1s
    (define B (make-tensor '(3 2) 2.0 #:type 'double)) ;; 3x2 matrix of 2s
    
    ;; C = A * B -> (2x3) * (3x2) -> (2x2)
    ;; each element C_ij = sum(A_ik * B_kj) = sum(1.0 * 2.0) for k=0..2 = 2.0 * 3 = 6.0
    (define C (tensor-matmul A B))
    
    (check-equal? (tensor-shape C) '(2 2))
    
    ;; Check values
    (define data (tensor-data C))
    (check-= (ptr-ref data _double 0) 6.0 1e-5)
    (check-= (ptr-ref data _double 1) 6.0 1e-5)
    (check-= (ptr-ref data _double 2) 6.0 1e-5)
    (check-= (ptr-ref data _double 3) 6.0 1e-5)
    ))


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
