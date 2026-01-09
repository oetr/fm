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

  (define alloc-elements
    (if (or (symbol=? type 'cfloat)
            (symbol=? type 'cdouble))
        (* n-elements 2)
        n-elements))

  (define data (alloc-tensor alloc-elements #:type _type))
  (tensor-fill-data! data type n-elements fill)

  (define strides (shape->strides shape))
  
  (tensor void children data empty n-elements shape strides type))


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
(define (broadcast-structs shape-A strides-A shape-B strides-B)
  ;; 1. Pad shapes with 1s on the left to match ranks
  (define rank-A (length shape-A))
  (define rank-B (length shape-B))
  (define max-rank (max rank-A rank-B))
  
  (define pad-A (make-list (- max-rank rank-A) 1))
  (define pad-B (make-list (- max-rank rank-B) 1))
  
  ;; padded shapes
  (define sA (append pad-A shape-A))
  (define sB (append pad-B shape-B))
  
  ;; padded strides (stride for a padded '1' dimension is mathematically irrelevant, use 0)
  (define stA (append (make-list (- max-rank rank-A) 0) strides-A))
  (define stB (append (make-list (- max-rank rank-B) 0) strides-B))
  
  ;; 2. Unify dimensions
  (define-values (out-shape new-stA new-stB)
    (for/lists (dims stAs stBs)
               ([dimA (in-list sA)]
                [dimB (in-list sB)]
                [strideA (in-list stA)]
                [strideB (in-list stB)])
      (cond
        [(= dimA dimB)
         (values dimA strideA strideB)]
        [(= dimA 1)
         ;; Broadcast A: Effective stride becomes 0
         (values dimB 0 strideB)]
        [(= dimB 1)
         ;; Broadcast B: Effective stride becomes 0
         (values dimA strideA 0)]
        [else
         (error 'broadcast "Incompatible shapes: ~a and ~a" shape-A shape-B)])))
  
  (values out-shape new-stA new-stB))

(define (broadcast-compatible? shape-A shape-B)
  ;; Simplified broadcasting:
  ;; 1. If shapes are equal -> compatible
  ;; 2. If rank(A) == 2 and rank(B) == 1 and B dim matches last dim of A -> compatible (add vector to rows)
  ;; This handles the bias addition case (Matrix + Vector)
  (or (equal? shape-A shape-B)
      (with-handlers ([exn:fail? (lambda (_) #f)])
        ;; Try to calculate broadcast structs; if it fails, they are incompatible
        (broadcast-structs shape-A (make-list (length shape-A) 1)
                           shape-B (make-list (length shape-B) 1))
        #t)))

(define+provide (tensor-add A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))
  
  ;; check supported types
  (unless (symbol=? type-A type-B)
    (error 'tensor-add "both tensors should be of the same type, but were: ~a and ~a~n" type-A type-B))
  
  
  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  ;; Calculate broadcast shapes and strides
  (define-values (out-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))
  
  (define out (make-tensor out-shape #:type type-A #:children (list A B)))
  
  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
      (raise-argument-error 'tensor-add "tensor:double or tensor:float" 0 A B))
    (unless (or (symbol=? type-B 'double) (symbol=? type-B 'float))
      (raise-argument-error 'tensor-add "tensor:double or tensor:float" 1 A B))
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  ;; Prepare arguments for C function
  (define ndim (length out-shape))

  ;; Forward Pass using generic C broadcast kernel
  (let-typed-c type-A
    ([add! _add_broadcast])
    (add! ndim
          out-shape
          (tensor-data A)
          broadcast-strides-A
          (tensor-data B)
          broadcast-strides-B
          (tensor-data out)))

  (define (backward)
    ;; Helper to accumulate gradients properly when broadcasting occurs.
    ;; If shape(out) == shape(input), direct axpy.
    ;; If broadcast occurred, we need to reduce sum over the broadcasted dimensions.
    (define (accumulate-grad! input-tensor grad-out)
    (define in-shape (tensor-shape input-tensor))
    (define out-shape (tensor-shape out))
    
    (cond
      ;; Optimization: Simple 1:1 match
      [(equal? in-shape out-shape)
       (let-typed type-A
         ([axpy cblas_daxpy cblas_saxpy])
         (axpy (tensor-length out) 1.0 grad-out 1 (tensor-grad input-tensor) 1))]

      [else
       ;; 1. Pad input shape to match output rank
       (define rank-out (length out-shape))
       (define pad-len  (- rank-out (length in-shape)))
       (define padded-in-shape (append (make-list pad-len 1) in-shape))
       
       ;; 2. Categorize Dimensions into KEEP vs REDUCE
       (define-values (keep-shape keep-src-st keep-dst-st
                       reduce-shape reduce-src-st)
         (for/fold ([ks '()] [kss '()] [kds '()]
                    [rs '()] [rss '()])
                   ([d-in  (in-list padded-in-shape)]
                    [d-out (in-list out-shape)]
                    [st-out (in-list (tensor-strides out))]
                    [st-in  (in-list (append (make-list pad-len 0) (tensor-strides input-tensor)))])
           (cond
             ;; If input dim is 1 and output > 1, this is a REDUCTION axis
             [(and (= d-in 1) (> d-out 1))
              (values ks kss kds
                      (cons d-out rs) (cons st-out rss))]
             ;; Otherwise, it is a PARALLEL (Keep) axis
             [else
              (values (cons d-out ks) (cons st-out kss) (cons st-in kds)
                      rs rss)])))
       
       ;; 3. Call Generic C Kernel
       (let-typed-c type-A
         ([reduce! _reduce_sum_generic])
         
         ;; Note: We collected in reverse order, so we reverse back to C order
         (reduce! (length keep-shape)
                  (reverse keep-shape)
                  (reverse keep-src-st)
                  (reverse keep-dst-st)
                  (length reduce-shape)
                  (reverse reduce-shape)
                  (reverse reduce-src-st)
                  (tensor-grad out)          ;; Source (Big)
                  (tensor-grad input-tensor) ;; Dest (Small)
                  ))]))

  (accumulate-grad! A (tensor-grad out))
  (accumulate-grad! B (tensor-grad out)))

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

  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  ;; Calculate broadcast shapes and strides
  (define-values (out-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (define out (make-tensor out-shape #:type type-A #:children (list A B)))

  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (define ndim (length out-shape))

  (let-typed-c type-A
    ([mul! _mul_broadcast])
    (mul! ndim
          out-shape
          (tensor-data A)
          broadcast-strides-A
          (tensor-data B)
          broadcast-strides-B
          (tensor-data out)))
  
  (define (backward)
    ;; Helper to accumulate gradients properly when broadcasting occurs.
    ;; For mult: dA += (dC * B) reduced to A shape
    ;;           dB += (dC * A) reduced to B shape
    
    (define (accumulate-grad-mul! input-tensor other-tensor grad-out broadcast-strides-input broadcast-strides-other)
      (define in-shape (tensor-shape input-tensor))
      (define out-shape (tensor-shape out))
    
      (cond
        ;; Optimization: Simple 1:1 match
        [(equal? in-shape out-shape)
         (let-typed-c type-A
           ([mul-then-add! _mul_then_add])
           (mul-then-add! (tensor-grad input-tensor)
                          (tensor-data other-tensor)
                          grad-out
                          (tensor-length out)))]

        [else
         ;; 1. Compute full unreduced gradient term: dC * Other (broadcasted)
         ;; We need a temporary buffer for this because _reduce_sum_generic expects a contiguous source?
         ;; Actually, _reduce_sum_generic iterates over the source with strides.
         ;; But here the source is a virtual tensor (dC * Other).
         ;; We must instantiate it.
         ;; TODO: We could write a fused kernel "_mul_broadcast_reduce" to avoid this alloc.
         ;; For now, allocate temp.
         
         (define temp-grad (alloc-tensor (tensor-length out) #:type (symbol->type type-A)))
         
         ;; Compute temp = grad_out * other_data (broadcasted)
         ;; grad_out is contiguous (standard strides for out-shape).
         ;; other_data might be broadcasted (use broadcast-strides-other).
         (let-typed-c type-A
            ([mul! _mul_broadcast])
            (mul! ndim
                  out-shape
                  grad-out
                  (shape->strides out-shape) ;; grad_out is contiguous
                  (tensor-data other-tensor)
                  broadcast-strides-other
                  temp-grad))
         
         ;; 2. Reduce temp-grad into input-tensor.grad
         ;; This is identical logic to tensor-add's reduction
         (define rank-out (length out-shape))
         (define pad-len  (- rank-out (length in-shape)))
         (define padded-in-shape (append (make-list pad-len 1) in-shape))
         
         (define-values (keep-shape keep-src-st keep-dst-st
                         reduce-shape reduce-src-st)
           (for/fold ([ks '()] [kss '()] [kds '()]
                      [rs '()] [rss '()])
                     ([d-in  (in-list padded-in-shape)]
                      [d-out (in-list out-shape)]
                      [st-out (in-list (shape->strides out-shape))] ;; Strides of temp-grad (standard)
                      [st-in  (in-list (append (make-list pad-len 0) (tensor-strides input-tensor)))])
             (cond
               [(and (= d-in 1) (> d-out 1))
                (values ks kss kds (cons d-out rs) (cons st-out rss))]
               [else
                (values (cons d-out ks) (cons st-out kss) (cons st-in kds) rs rss)])))
         
         (let-typed-c type-A
           ([reduce! _reduce_sum_generic])
           (reduce! (length keep-shape)
                    (reverse keep-shape)
                    (reverse keep-src-st)
                    (reverse keep-dst-st)
                    (length reduce-shape)
                    (reverse reduce-shape)
                    (reverse reduce-src-st)
                    temp-grad
                    (tensor-grad input-tensor)))
         
         ;; Free temp? Racket GC handles it, but explicit might be nicer if not GC pointer.
         ;; It is alloc-tensor (GC pointer), so we are good.
         ]))

    (accumulate-grad-mul! A B (tensor-grad out) broadcast-strides-A broadcast-strides-B)
    (accumulate-grad-mul! B A (tensor-grad out) broadcast-strides-B broadcast-strides-A))

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

  (define (backward)
    (let-typed type
      ([kernel _backward_tanh_double _backward_tanh_float])
      (kernel (tensor-data out) (tensor-grad out) (tensor-grad T) (tensor-length T))))

  (when (not (empty? (tensor-grad T)))
    (set-tensor-backward! out backward))
  out)


(module+ test
  (require rackunit)
  ;; Helper to make tests concise
  (define (check-broadcast shape-A strides-A shape-B strides-B expected-shape expected-strides-A expected-strides-B)
    (define-values (out sA sB) (broadcast-structs shape-A strides-A shape-B strides-B))
    (check-equal? out expected-shape "Output shape mismatch")
    (check-equal? sA expected-strides-A "Strides A mismatch")
    (check-equal? sB expected-strides-B "Strides B mismatch"))

  ;; 1. Identity Broadcast (Same Shape)
  ;; A: (2, 3), B: (2, 3) -> Out: (2, 3)
  ;; Strides should remain unchanged.
  (test-case "Identity: (2,3) + (2,3)"
    (check-broadcast '(2 3) '(3 1)
                     '(2 3) '(3 1)
                     '(2 3) '(3 1) '(3 1)))

  ;; 2. Standard Matrix + Vector (Bias Add)
  ;; A: (2, 3), B: (3) -> Out: (2, 3)
  ;; B is implicitly padded to (1, 3) with stride (0, 1).
  ;; Then dim 0 of B (1) is broadcast to (2), so stride becomes 0.
  (test-case "Bias Add: (2,3) + (3)"
    (check-broadcast '(2 3) '(3 1)
                     '(3)   '(1)
                     '(2 3) '(3 1) '(0 1)))

  ;; 3. Column Vector Broadcast
  ;; A: (2, 3), B: (2, 1) -> Out: (2, 3)
  ;; B matches rows (2), but has 1 column.
  ;; B's column stride (stride-B[1]) becomes 0.
  (test-case "Column Broadcast: (2,3) + (2,1)"
    (check-broadcast '(2 3) '(3 1)
                     '(2 1) '(1 1)
                     '(2 3) '(3 1) '(1 0)))

  ;; 4. Scalar Broadcast
  ;; A: (2, 2), B: () -> Out: (2, 2)
  ;; B is padded to (1, 1). Both dims are broadcast, so strides become (0, 0).
  (test-case "Scalar: (2,2) + ()"
    (check-broadcast '(2 2) '(2 1)
                     '()    '()
                     '(2 2) '(2 1) '(0 0)))

  ;; 5. Outer Product (Interleaved Dimensions)
  ;; A: (3, 1), B: (1, 4) -> Out: (3, 4)
  ;; A broadcasts col (stride 0), B broadcasts row (stride 0).
  (test-case "Outer Product: (3,1) + (1,4)"
    (check-broadcast '(3 1) '(1 1)
                     '(1 4) '(4 1)
                     '(3 4) '(1 0) '(0 1)))

  ;; 6. Leading Dimensions (3D + 2D)
  ;; A: (2, 3, 4), B: (3, 4) -> Out: (2, 3, 4)
  ;; B padded to (1, 3, 4). B dim 0 broadcasts to 2 (stride 0).
  (test-case "Rank mismatch: (2,3,4) + (3,4)"
    (check-broadcast '(2 3 4) '(12 4 1)
                     '(3 4)   '(4 1)
                     '(2 3 4) '(12 4 1) '(0 4 1)))

  ;; 7. Complex Interleaved 3D
  ;; A: (1, 2, 1), B: (5, 1, 3) -> Out: (5, 2, 3)
  ;; A broadcasts dims 0 and 2 (strides -> 0).
  ;; B broadcasts dim 1 (stride -> 0).
  (test-case "Interleaved 3D: (1,2,1) + (5,1,3)"
    (check-broadcast '(1 2 1) '(2 1 1)
                     '(5 1 3) '(3 3 1)
                     '(5 2 3) '(0 1 0) '(3 0 1)))

  ;; 8. Single Dimension Broadcast
  ;; A: (1), B: (5) -> Out: (5)
  ;; A broadcasts to match B.
  (test-case "1D Broadcast: (1) + (5)"
    (check-broadcast '(1) '(1)
                     '(5) '(1)
                     '(5) '(0) '(1)))

  ;; 9. Error: Dimension Mismatch (Non-Singleton)
  ;; (2, 3) vs (2, 4). 3 != 4 and neither is 1.
  (test-case "Error: Mismatch (2,3) vs (2,4)"
    (check-exn exn:fail?
               (lambda ()
                 (broadcast-structs '(2 3) '(3 1)
                                    '(2 4) '(4 1)))))

  ;; 10. Error: Incompatible Leading Dimensions
  ;; (2) vs (3). Padded, this is (2) vs (3).
  ;; Unlike (2,3) vs (3) where alignment happens on the right,
  ;; direct 1D mismatch 2 vs 3 is fatal.
  (test-case "Error: Mismatch (2) vs (3)"
    (check-exn exn:fail?
               (lambda ()
                 (broadcast-structs '(2) '(1)
                                    '(3) '(1)))))
)


