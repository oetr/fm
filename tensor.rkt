;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(require ffi/unsafe
         "unsafe/cblas.rkt"
         "./unsafe/tensor.rkt"
         "./private/utilities.rkt"
         "./C/aux.rkt")

(module+ test (require rackunit))

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
  (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
    (raise-argument-error 'tensor-add "tensor:double or tensor:float" 0 A B))
  (unless (or (symbol=? type-B 'double) (symbol=? type-B 'float))
    (raise-argument-error 'tensor-add "tensor:double or tensor:float" 1 A B))
  
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

  ;; Prepare arguments for C function
  (define ndim (length out-shape))

  ;; Forward Pass using generic C broadcast kernel
  (let-typed type-A
    ([add! _add_broadcast_double _add_broadcast_float])
    (add! ndim
          out-shape
          (tensor-data A)
          broadcast-strides-A
          (tensor-data B)
          broadcast-strides-B
          (tensor-data out)))

  (define (backward)
    ;; For backward pass, we need to accumulate gradients.
    ;; Since A and B might be broadcasted, we essentially need to "un-broadcast" (sum) the gradients 
    ;; if the dimension was broadcasted.
    ;; However, our current C kernel infrastructure is for forward broadcasting.
    ;; To do this efficiently, we can use the same broadcasting kernel structure but we need
    ;; to map dOut -> dA and dOut -> dB.
    ;;
    ;; dOut has shape 'out-shape' and strides 'out-strides' (contiguous).
    ;; dA should accumulate dOut. If A was broadcasted, multiple dOut elements map to one A element.
    ;;
    ;; Current limitation: The generic _add_broadcast kernel writes to a contiguous output.
    ;; It does NOT support accumulating into a strided output (which would be needed for dA += ...).
    ;;
    ;; TEMPORARY SOLUTION:
    ;; We fall back to the simple logic for the common non-broadcast case,
    ;; and the specialized row-broadcast case implemented previously,
    ;; but ideally we need a `_reduce_sum` kernel to support full backward broadcasting.
    ;;
    ;; Since the user asked to UNIFY the logic, we should probably stick to the generic kernel
    ;; for the forward pass (done above).
    ;; For the backward pass, fully generic un-broadcasting requires a reduction kernel which we don't have yet.
    ;;
    ;; Strategy:
    ;; 1. If shapes match (no broadcast), use simple axpy.
    ;; 2. If row-broadcast (M,N) + (N), use the manual loop (existing logic).
    ;; 3. Otherwise, error for now on backward (until reduction kernel is added).
    
    (cond
      [(equal? shape-A shape-B)
       (let-typed type-A
         ([axpy cblas_daxpy cblas_saxpy])
         (axpy (tensor-length out) 1.0 (tensor-grad out) 1 (tensor-grad A) 1)
         (axpy (tensor-length out) 1.0 (tensor-grad out) 1 (tensor-grad B) 1))]
      
      ;; Special case: Bias add (M, N) + (N)
      ;; We use the specialized reduction kernel _reduce_sum_axis0
      ;; M = outer-dim, N = inner-dim
      [(and (= (length shape-A) 2)
            (= (length shape-B) 1)
            (= (second shape-A) (first shape-B)))
       (let-typed type-A
         ([axpy cblas_daxpy cblas_saxpy])
         ;; dA += dOut (Direct copy/add since shape matches)
         (axpy (tensor-length out) 1.0 (tensor-grad out) 1 (tensor-grad A) 1))
         
         ;; dB += sum_rows(dOut)
         ;; (M, N) -> (N) reduction along axis 0
         (define M (first shape-A))
         (define N (second shape-A))
         (let-typed type-A
           ([reduce! _reduce_sum_axis0_double _reduce_sum_axis0_float])
           (reduce! M N (tensor-grad out) (tensor-grad B)))]
      
      [else
       (error 'tensor-add:backward "Generic backward broadcasting not yet implemented (requires generic reduction kernel)")]))

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

  (define (backward)
    (let-typed type
      ([kernel _backward_tanh_double _backward_tanh_float])
      (kernel (tensor-data out) (tensor-grad out) (tensor-grad T) (tensor-length T))))

  (when (not (empty? (tensor-grad T)))
    (set-tensor-backward! out backward))
  out)


(module+ test
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
(module+ test
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
