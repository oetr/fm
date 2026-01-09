#lang racket

(require "../tensor.rkt")

(define N-ITERATIONS 20)

(define (time-op name thunk [iterations 10])
  (printf "Benchmarking ~a (~a iterations)...\n" name iterations)
  (collect-garbage)
  (define start (current-inexact-milliseconds))
  (for ([_ (in-range iterations)])
    (thunk))
  (define end (current-inexact-milliseconds))
  (define total-time (- end start))
  (printf "  Total: ~a ms\n" (~r total-time #:precision 2))
  (printf "  Avg:   ~a ms/op\n\n" (~r (/ total-time iterations) #:precision 4)))

(define (bench-add-same-shape size)
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size size) 2.0 #:type 'double))
  
  (time-op (format "tensor-add (~ax~a)" size size)
           (lambda () (tensor-add A B))
           N-ITERATIONS))

(define (bench-mul-same-shape size)
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size size) 2.0 #:type 'double))
  
  (time-op (format "tensor-mul (~ax~a)" size size)
           (lambda () (tensor-mul A B))
           N-ITERATIONS))

(define (bench-add-broadcast size)
  ;; Broadcast vector to matrix
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size) 2.0 #:type 'double))
  
  (time-op (format "tensor-add broadcast (~ax~a + ~a)" size size size)
           (lambda () (tensor-add A B))
           N-ITERATIONS))

(define (bench-mul-broadcast size)
  ;; Broadcast vector to matrix
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size) 2.0 #:type 'double))
  
  (time-op (format "tensor-mul broadcast (~ax~a * ~a)" size size size)
           (lambda () (tensor-mul A B))
           N-ITERATIONS))

;; Run benchmarks
(displayln "=== Tensor Operation Benchmarks ===\n")

;; Small-ish size (1000x1000 = 1M elements)
(define SIZE 1000)

(bench-add-same-shape SIZE)
(bench-mul-same-shape SIZE)
(bench-add-broadcast SIZE)
(bench-mul-broadcast SIZE)

;; Larger size (2000x2000 = 4M elements)
(define SIZE-LARGE 2000)
(bench-add-same-shape SIZE-LARGE)
(bench-mul-same-shape SIZE-LARGE)
(bench-add-broadcast SIZE-LARGE)
(bench-mul-broadcast SIZE-LARGE)
