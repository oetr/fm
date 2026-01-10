#lang racket

(require rackunit
         ffi/unsafe
         "../tensor.rkt"
         (only-in "../unsafe/tensor.rkt" tensor-data tensor-grad tensor? tensor-shape tensor-type))

;; -----------------------------------------------------------------------------
;; Basic Operations
;; -----------------------------------------------------------------------------
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

(test-case "tensor-mul-broadcast"
  ;; A: (2, 2)
  ;; B: (2, 1) -> broadcasts to (2, 2)
  ;; C = A * B
  (define A (make-tensor '(2 2) 2.0 #:type 'double))
  (define B (make-tensor '(2 1) 3.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-mul A B))
  ;; C should be all 6.0
  (check-= (ptr-ref (tensor-data C) _double 0) 6.0 1e-5)
  (check-= (ptr-ref (tensor-data C) _double 3) 6.0 1e-5)

  (backward! C)

  ;; dC/dA = B (broadcasted) -> (2, 2) full of 3.0s
  (check-= (ptr-ref (tensor-grad A) _double 0) 3.0 1e-5)

  ;; dC/dB = sum(dC/dC * A) along axis 1
  ;;       = sum(1 * A)
  ;;       = sum([2, 2]) = 4.0 for each row
  (check-= (ptr-ref (tensor-grad B) _double 0) 4.0 1e-5)
  (check-= (ptr-ref (tensor-grad B) _double 1) 4.0 1e-5))

(test-case "tensor-add"
  (define A (make-tensor (list 3 3) 1.0 #:type 'double))
  (define B (make-tensor (list 3 3) 2.0 #:type 'double))
  (tensor-requires-grad! A B)
  (define C (tensor-add A B))
  (define D (tensor-mul A C))
  (backward! D)
  (check-true (tensor? D)))

(test-case "tensor-sub-broadcast"
  (define A (make-tensor '(2 2) 3.0 #:type 'double))
  (define B (make-tensor '(2 1) 1.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-sub A B))
  ;; [[3,3],[3,3]] - [[1],[1]] = [[2,2],[2,2]]
  (check-= (ptr-ref (tensor-data C) _double 0) 2.0 1e-5)
  (check-= (ptr-ref (tensor-data C) _double 1) 2.0 1e-5)

  (backward! C)

  ;; dC/dA = 1.0
  (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5)

  ;; dC/dB = -1.0 (broadcast sum)
  ;; Each element of B is subtracted from 2 elements of A.
  ;; Sum = -1 + -1 = -2.
  (check-= (ptr-ref (tensor-grad B) _double 0) -2.0 1e-5)
  (check-= (ptr-ref (tensor-grad B) _double 1) -2.0 1e-5))

(test-case "tensor-div-broadcast"
  (define A (make-tensor '(2 2) 6.0 #:type 'double))
  (define B (make-tensor '(2 1) 2.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-div A B))
  ;; 6 / 2 = 3
  (check-= (ptr-ref (tensor-data C) _double 0) 3.0 1e-5)

  (backward! C)

  ;; dA = 1/B = 0.5
  (check-= (ptr-ref (tensor-grad A) _double 0) 0.5 1e-5)

  ;; dB = -A/B^2 = -6/4 = -1.5
  ;; Broadcast sum: -1.5 * 2 = -3.0
  (check-= (ptr-ref (tensor-grad B) _double 0) -3.0 1e-5))

(test-case "tensor-pow-broadcast"
  (define A (make-tensor '(2 2) 2.0 #:type 'double))
  (define B (make-tensor '(1) 3.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-pow A B))
  ;; 2^3 = 8
  (check-= (ptr-ref (tensor-data C) _double 0) 8.0 1e-5)

  (backward! C)

  ;; dA = B * A^(B-1) = 3 * 2^2 = 12
  (check-= (ptr-ref (tensor-grad A) _double 0) 12.0 1e-5)

  ;; dB = A^B * ln(A) = 8 * ln(2) ~= 8 * 0.693147 = 5.545
  ;; Sum over 4 elements: 5.545 * 4 = 22.18
  (check-= (ptr-ref (tensor-grad B) _double 0) (* 4.0 8.0 (log 2.0)) 1e-4))

(test-case "tensor-exp"
  (define A (make-tensor '(2) 0.0 #:type 'double))
  (tensor-requires-grad! A)
  (define B (tensor-exp A))
  (check-= (ptr-ref (tensor-data B) _double 0) 1.0 1e-5) ;; exp(0)=1

  (backward! B)
  (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5))

(test-case "tensor-log"
  (define A (make-tensor '(2) (exp 1.0) #:type 'double))
  (tensor-requires-grad! A)
  (define B (tensor-log A))
  (check-= (ptr-ref (tensor-data B) _double 0) 1.0 1e-5)

  (backward! B)
  ;; d/dx log(x) = 1/x = 1/e
  (check-= (ptr-ref (tensor-grad A) _double 0) (/ 1.0 (exp 1.0)) 1e-5))

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
  )

(test-case "tensor-add-out!"
  (define A (make-tensor '(2) 1.0 #:type 'double))
  (define B (make-tensor '(2) 2.0 #:type 'double))
  (define out (make-tensor '(2) 0.0 #:type 'double))
  (tensor-requires-grad! A B out)

  (tensor-add-out! out A B)
  (check-= (ptr-ref (tensor-data out) _double 0) 3.0 1e-5)

  (backward! out)
  (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5)
  (check-= (ptr-ref (tensor-grad B) _double 0) 1.0 1e-5))

(test-case "tensor-mul-out!"
  (define A (make-tensor '(2) 2.0 #:type 'double))
  (define B (make-tensor '(2) 3.0 #:type 'double))
  (define out (make-tensor '(2) 0.0 #:type 'double))
  (tensor-requires-grad! A B out)

  (tensor-mul-out! out A B)
  (check-= (ptr-ref (tensor-data out) _double 0) 6.0 1e-5)

  (backward! out)
  (check-= (ptr-ref (tensor-grad A) _double 0) 3.0 1e-5)
  (check-= (ptr-ref (tensor-grad B) _double 0) 2.0 1e-5))

;; -----------------------------------------------------------------------------
;; Gradient Accumulation & Broadcasting
;; -----------------------------------------------------------------------------
(test-case "gradient-accumulation-mul-self"
  ;; y = x * x
  ;; dy/dx = 2 * x
  ;; Let x = 3.0. dy/dx should be 6.0.

  (define x (make-tensor '(1) 3.0 #:type 'double))
  (tensor-requires-grad! x)

  (define y (tensor-mul x x))

  (backward! y)

  (check-= (ptr-ref (tensor-grad x) _double 0) 6.0 1e-5))

(test-case "gradient-accumulation-add-self"
  ;; y = x + x
  ;; dy/dx = 2

  (define x (make-tensor '(1) 10.0 #:type 'double))
  (tensor-requires-grad! x)

  (define y (tensor-add x x))

  (backward! y)

  (check-= (ptr-ref (tensor-grad x) _double 0) 2.0 1e-5))

(test-case "gradient-accumulation-split-path"
  ;; x = 2.0
  ;; a = x * x  (4.0)
  ;; b = tanh(x) (tanh(2.0) ~= 0.96402758)
  ;; y = a + b
  ;; dy/dx = d(x^2)/dx + d(tanh(x))/dx
  ;;       = 2x + (1 - tanh^2(x))
  ;;       = 4.0 + (1 - 0.96402758^2)
  ;;       = 4.0 + (1 - 0.92934993)
  ;;       = 4.0 + 0.07065007
  ;;       = 4.07065007

  (define x (make-tensor '(1) 2.0 #:type 'double))
  (tensor-requires-grad! x)

  (define a (tensor-mul x x))
  (define b (tensor-tanh x))
  (define y (tensor-add a b))

  (backward! y)

  (define expected (+ 4.0 (- 1.0 (expt (tanh 2.0) 2))))
  (check-= (ptr-ref (tensor-grad x) _double 0) expected 1e-5))

(test-case "gradient-accumulation-diamond"
  ;;   x
  ;;  / \
  ;; a   b
  ;;  \ /
  ;;   y
  ;; a = 2 * x
  ;; b = 3 * x
  ;; y = a + b

  (define x (make-tensor '(1) 2.0 #:type 'double))
  (tensor-requires-grad! x)

  (define two (make-tensor '(1) 2.0 #:type 'double))
  (define three (make-tensor '(1) 3.0 #:type 'double))

  (define a (tensor-mul x two))
  (define b (tensor-mul x three))
  (define y (tensor-add a b))

  (backward! y)

  ;; dy/dx = 2 + 3 = 5
  (check-= (ptr-ref (tensor-grad x) _double 0) 5.0 1e-5))

(test-case "gradient-accumulation-broadcast-multi-use"
  ;; x: (2, 2) all 0
  ;; b: (2) all 0
  ;; y = (x + b) + b
  ;; dy/db ?

  (define x (make-tensor '(2 2) 0.0 #:type 'double))
  (define b (make-tensor '(2) 0.0 #:type 'double))
  (tensor-requires-grad! x b)

  (define z (tensor-add x b))
  (define y (tensor-add z b))

  (backward! y)

  ;; b has shape (2).
  ;; The gradient should be accumulated from two paths.
  ;; path 1: y -> z -> b. dz/db involves reduction of 1s along axis 0.
  ;;         z (2,2) -> b (2). Reduction over dim 0 (size 2).
  ;;         Each element of b affects 2 elements of z.
  ;;         Sum is 2.
  ;; path 2: y -> b. dy/db (direct) involves reduction of 1s along axis 0.
  ;;         Sum is 2.
  ;; Total grad for each element of b should be 4.

  (check-= (ptr-ref (tensor-grad b) _double 0) 4.0 1e-5)
  (check-= (ptr-ref (tensor-grad b) _double 1) 4.0 1e-5))

(test-case "gradient-accumulation-3d-broadcast"
  ;; A: (2, 3, 4)
  ;; B: (3, 4)
  ;; C = A + B (B broadcasted along axis 0)
  ;; D = C + B (Using B again)
  ;; D = A + 2B

  (define A (make-tensor '(2 3 4) 0.0 #:type 'double))
  (define B (make-tensor '(3 4) 0.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-add A B))
  (define D (tensor-add C B))

  (backward! D)

  ;; D has shape (2, 3, 4). Gradient of D is all 1s.
  ;; D = A + 2*B.
  ;; B is broadcast along axis 0 (size 2).
  ;; dD/dB = 2 * (sum over axis 0 of dD/dD)
  ;;       = 2 * 2 * 1.0 = 4.0.

  ;; Let's check a few indices.
  (define grad-B (tensor-grad B))
  (for* ([i 3] [j 4])
    (define idx (+ (* i 4) j))
    (check-= (ptr-ref grad-B _double idx) 4.0 1e-5)))

(test-case "gradient-accumulation-4d-interleaved-broadcast"
  ;; A: (2, 2, 3, 2)
  ;; B: (1, 2, 3, 1)
  ;; D = A + B + B
  ;; B is broadcast along axis 0 (size 2) and axis 3 (size 2).

  (define A (make-tensor '(2 2 3 2) 0.0 #:type 'double))
  (define B (make-tensor '(1 2 3 1) 0.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-add A B))
  (define D (tensor-add C B))

  (backward! D)

  ;; D has shape (2, 2, 3, 2). Gradient is all 1s.
  ;; B is used twice.
  ;; Each usage involves broadcasting 1->2 on axis 0 and 1->2 on axis 3.
  ;; Multiplier per usage = 2 * 2 = 4.
  ;; Total gradient = 2 * 4 = 8.

  (define grad-B (tensor-grad B))
  ;; B has 6 elements (1*2*3*1)
  (for ([i (in-range 6)])
    (check-= (ptr-ref grad-B _double i) 8.0 1e-5)))

;; -----------------------------------------------------------------------------
;; Safety & Error Handling
;; -----------------------------------------------------------------------------
(test-case "safety-tensor-mul-mismatch"
  (define A (make-tensor '(2 2) 1.0))
  (define B (make-tensor '(3) 1.0))
  (check-exn exn:fail? (lambda () (tensor-mul A B)) "Should reject shape mismatch"))

(test-case "safety-tensor-matmul-rank"
  (define A (make-tensor '(2 2 2) 1.0)) ;; 3D
  (define B (make-tensor '(2 2) 1.0))
  (check-exn exn:fail? (lambda () (tensor-matmul A B)) "Should reject 3D tensor"))

(test-case "tensor-mul-broadcast-4d"
  ;; A: (2, 1, 3, 1)
  ;; B: (1, 2, 1, 3)
  ;; Out: (2, 2, 3, 3)
  (define A (make-tensor '(2 1 3 1) 2.0 #:type 'double))
  (define B (make-tensor '(1 2 1 3) 3.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-mul A B))

  (check-equal? (tensor-shape C) '(2 2 3 3))
  (check-= (ptr-ref (tensor-data C) _double 0) 6.0 1e-5)
  (check-= (ptr-ref (tensor-data C) _double 35) 6.0 1e-5)

  (backward! C)

  ;; Grad A: Sum of (1.0 * 3.0) over broadcast axes of A (axis 1 size 2, axis 3 size 3)
  ;; Factor = 2 * 3 = 6.
  ;; Val = 3.0 * 6 = 18.0
  (check-= (ptr-ref (tensor-grad A) _double 0) 18.0 1e-5)

  ;; Grad B: Sum of (1.0 * 2.0) over broadcast axes of B (axis 0 size 2, axis 2 size 3)
  ;; Factor = 2 * 3 = 6.
  ;; Val = 2.0 * 6 = 12.0
  (check-= (ptr-ref (tensor-grad B) _double 0) 12.0 1e-5))

(test-case "tensor-mul-broadcast-5d"
  ;; A: (2, 1, 2, 1, 2)
  ;; B: (1, 2, 1, 2, 1)
  ;; Out: (2, 2, 2, 2, 2) -> 32 elements
  (define A (make-tensor '(2 1 2 1 2) 2.0 #:type 'double))
  (define B (make-tensor '(1 2 1 2 1) 4.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-mul A B))
  (check-equal? (tensor-shape C) '(2 2 2 2 2))
  (check-= (ptr-ref (tensor-data C) _double 0) 8.0 1e-5)
  (check-= (ptr-ref (tensor-data C) _double 31) 8.0 1e-5)

  (backward! C)

  ;; Grad A: broadcasts on axes 1 (size 2) and 3 (size 2). Total factor 4.
  ;; Gradient from C is 1.0 (implicit).
  ;; dC/dA contribution = B * 1.0.
  ;; Sum(B) over axes 1,3? No, A is smaller.
  ;; dA += sum(dC * B) over broadcast axes.
  ;; B is 4.0. dC is 1.0. Product is 4.0.
  ;; Summing 4.0 over 4 broadcasted slots = 16.0.
  (check-= (ptr-ref (tensor-grad A) _double 0) 16.0 1e-5)

  ;; Grad B: broadcasts on axes 0 (size 2), 2 (size 2), 4 (size 2). Total factor 8.
  ;; A is 2.0. dC is 1.0. Product is 2.0.
  ;; Summing 2.0 over 8 broadcasted slots = 16.0.
  (check-= (ptr-ref (tensor-grad B) _double 0) 16.0 1e-5))

(test-case "tensor-mul-broadcast-mixed-rank-5d"
  ;; A: (2, 2, 2, 2, 2)
  ;; B: (2) -> broadcasts to (2, 2, 2, 2, 2)
  (define A (make-tensor '(2 2 2 2 2) 0.5 #:type 'double))
  (define B (make-tensor '(2) 10.0 #:type 'double))
  (tensor-requires-grad! A B)

  (define C (tensor-mul A B))
  (check-equal? (tensor-shape C) '(2 2 2 2 2))
  (check-= (ptr-ref (tensor-data C) _double 0) 5.0 1e-5)

  (backward! C)

  ;; Grad A: B is effectively scalar 10.0 per element.
  (check-= (ptr-ref (tensor-grad A) _double 0) 10.0 1e-5)

  ;; Grad B: A is 0.5. Sum over 2*2*2*2 = 16 elements.
  ;; 16 * 0.5 = 8.0.
  (check-= (ptr-ref (tensor-grad B) _double 0) 8.0 1e-5))

(test-case "safety-type-mixing"
  (define A (make-tensor '(2 2) 1.0 #:type 'double))
  (define B (make-tensor '(2 2) 1.0 #:type 'float))
  (check-exn exn:fail? (lambda () (tensor-add A B)) "Should reject mixed types"))

;; -----------------------------------------------------------------------------
;; Float Support
;; -----------------------------------------------------------------------------
(test-case "float-basic-ops"
  (define A (make-tensor '(2) 2.0 #:type 'float))
  (define B (make-tensor '(2) 3.0 #:type 'float))
  (tensor-requires-grad! A B)

  ;; Mul
  (define C (tensor-mul A B)) ;; [6.0, 6.0]
  (check-equal? (tensor-type C) 'float)
  (check-= (ptr-ref (tensor-data C) _float 0) 6.0 1e-4)

  (backward! C)
  ;; dC/dA = B = 3.0
  (check-= (ptr-ref (tensor-grad A) _float 0) 3.0 1e-4))

;; -----------------------------------------------------------------------------
;; Geometry & Edge Cases
;; -----------------------------------------------------------------------------
(test-case "broadcast-scalar-to-matrix"
  (define A (make-tensor '(2 2) 1.0 #:type 'double))
  (define s (make-tensor '(1) 5.0 #:type 'double))
  (tensor-requires-grad! s)

  (define C (tensor-add A s)) ;; All 6.0
  (backward! C)

  ;; dC/ds = sum(dC/dC) = sum(ones(2,2)) = 4.0
  (check-= (ptr-ref (tensor-grad s) _double 0) 4.0 1e-5))

(test-case "broadcast-sandwich-dim"
  ;; A: (2, 1, 2)
  ;; B: (2, 2, 2)
  (define A (make-tensor '(2 1 2) 1.0 #:type 'double))
  (define B (make-tensor '(2 2 2) 2.0 #:type 'double))
  (tensor-requires-grad! A)

  (define C (tensor-add A B))
  (backward! C)

  ;; Gradient for A should sum over the middle dimension (size 2)
  ;; A has size 4. C has size 8.
  ;; Each element of A broadcasts to 2 elements in C.
  (check-= (ptr-ref (tensor-grad A) _double 0) 2.0 1e-5))

(test-case "matmul-outer-product"
  ;; (3, 1) @ (1, 3) -> (3, 3)
  (define A (make-tensor '(3 1) 1.0 #:type 'double))
  (define B (make-tensor '(1 3) 2.0 #:type 'double))

  (define C (tensor-matmul A B))
  (check-equal? (tensor-shape C) '(3 3))
  ;; Result should be all 2.0
  (check-= (ptr-ref (tensor-data C) _double 0) 2.0 1e-5))

(test-case "matmul-inner-product"
  ;; (1, 3) @ (3, 1) -> (1, 1)
  (define A (make-tensor '(1 3) 1.0 #:type 'double))
  (define B (make-tensor '(3 1) 2.0 #:type 'double))

  (define C (tensor-matmul A B))
  (check-equal? (tensor-shape C) '(1 1))
  ;; Dot product: 1*2 + 1*2 + 1*2 = 6.0
  (check-= (ptr-ref (tensor-data C) _double 0) 6.0 1e-5))

;; -----------------------------------------------------------------------------
;; Integration (MLP Layer)
;; -----------------------------------------------------------------------------
(test-case "integration-mlp-layer"
  ;; y = tanh(Wx + b)
  ;; W: (2, 2) = 0.5
  ;; x: (2, 1) = 1.0
  ;; b: (2, 1) = 0.5

  (define W (make-tensor '(2 2) 0.5 #:type 'double))
  (define x (make-tensor '(2 1) 1.0 #:type 'double))
  (define b (make-tensor '(2 1) 0.5 #:type 'double))
  (tensor-requires-grad! W b)

  ;; z = Wx + b
  ;; Wx = [[0.5, 0.5], [0.5, 0.5]] @ [[1], [1]] = [[1.0], [1.0]]
  ;; z = [[1.0], [1.0]] + [[0.5], [0.5]] = [[1.5], [1.5]]
  (define z (tensor-add (tensor-matmul W x) b))

  ;; a = tanh(z)
  ;; tanh(1.5) ~= 0.905148
  (define a (tensor-tanh z))

  (check-= (ptr-ref (tensor-data a) _double 0) 0.905148 1e-5)
  (check-= (ptr-ref (tensor-data a) _double 1) 0.905148 1e-5)

  (backward! a)

  ;; Gradient check
  ;; da/dz = 1 - tanh^2(z) = 1 - 0.905148^2 ~= 1 - 0.81929 = 0.18071
  ;; da/db = da/dz * dz/db = da/dz * 1
  (check-= (ptr-ref (tensor-grad b) _double 0) 0.18071 1e-4)

  ;; da/dW = da/dz * dz/dW = da/dz * x^T
  ;; dz/dW is outer product of da/dz vector and x vector
  ;; 0.18071 * 1.0 = 0.18071
  (check-= (ptr-ref (tensor-grad W) _double 0) 0.18071 1e-4))

;; -----------------------------------------------------------------------------
;; Extended Unary & Binary Ops Tests
;; -----------------------------------------------------------------------------

(test-case "tensor-sub-simple"
  ;; A - B where A=5, B=3 -> 2
  (define A (make-tensor '(1) 5.0 #:type 'double))
  (define B (make-tensor '(1) 3.0 #:type 'double))
  (tensor-requires-grad! A B)
  (define C (tensor-sub A B))
  (check-= (ptr-ref (tensor-data C) _double 0) 2.0 1e-5)
  (backward! C)
  (check-= (ptr-ref (tensor-grad A) _double 0) 1.0 1e-5)
  (check-= (ptr-ref (tensor-grad B) _double 0) -1.0 1e-5))

(test-case "tensor-div-simple"
  ;; A / B where A=6, B=3 -> 2
  (define A (make-tensor '(1) 6.0 #:type 'double))
  (define B (make-tensor '(1) 3.0 #:type 'double))
  (tensor-requires-grad! A B)
  (define C (tensor-div A B))
  (check-= (ptr-ref (tensor-data C) _double 0) 2.0 1e-5)
  (backward! C)
  ;; dA = 1/B = 1/3
  (check-= (ptr-ref (tensor-grad A) _double 0) (/ 1.0 3.0) 1e-5)
  ;; dB = -A/B^2 = -6/9 = -2/3
  (check-= (ptr-ref (tensor-grad B) _double 0) (/ -2.0 3.0) 1e-5))

(test-case "tensor-pow-simple"
  ;; A ^ B where A=2, B=3 -> 8
  (define A (make-tensor '(1) 2.0 #:type 'double))
  (define B (make-tensor '(1) 3.0 #:type 'double))
  (tensor-requires-grad! A B)
  (define C (tensor-pow A B))
  (check-= (ptr-ref (tensor-data C) _double 0) 8.0 1e-5)
  (backward! C)
  ;; dA = B * A^(B-1) = 3 * 2^2 = 12
  (check-= (ptr-ref (tensor-grad A) _double 0) 12.0 1e-5)
  ;; dB = A^B * ln(A) = 8 * ln(2)
  (check-= (ptr-ref (tensor-grad B) _double 0) (* 8.0 (log 2.0)) 1e-5))

(test-case "tensor-exp-simple"
  ;; exp(A) where A=1 -> e
  (define A (make-tensor '(1) 1.0 #:type 'double))
  (tensor-requires-grad! A)
  (define B (tensor-exp A))
  (check-= (ptr-ref (tensor-data B) _double 0) (exp 1.0) 1e-5)
  (backward! B)
  ;; dA = exp(A) = e
  (check-= (ptr-ref (tensor-grad A) _double 0) (exp 1.0) 1e-5))

(test-case "tensor-log-simple"
  ;; log(A) where A=e -> 1
  (define A (make-tensor '(1) (exp 1.0) #:type 'double))
  (tensor-requires-grad! A)
  (define B (tensor-log A))
  (check-= (ptr-ref (tensor-data B) _double 0) 1.0 1e-5)
  (backward! B)
  ;; dA = 1/A = 1/e
  (check-= (ptr-ref (tensor-grad A) _double 0) (/ 1.0 (exp 1.0)) 1e-5))

