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

