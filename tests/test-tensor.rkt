#lang racket

(require rackunit
         ffi/unsafe
         "../tensor.rkt"
         (only-in "../unsafe/tensor.rkt" tensor-data tensor-grad tensor? tensor-shape))

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
  )

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
