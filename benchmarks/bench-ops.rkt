#lang racket

(require "../tensor.rkt"
         racket/date)

(define N-ITERATIONS 20)

;; Create data directory if it doesn't exist
(unless (directory-exists? "benchmarks/data")
  (make-directory "benchmarks/data"))

;; Generate filename based on current timestamp
(define timestamp (current-date))
(define filename
  (format "benchmarks/data/benchmark_~a~a~a_~a~a~a.csv"
          (date-year timestamp)
          (~r (date-month timestamp) #:min-width 2 #:pad-string "0")
          (~r (date-day timestamp) #:min-width 2 #:pad-string "0")
          (~r (date-hour timestamp) #:min-width 2 #:pad-string "0")
          (~r (date-minute timestamp) #:min-width 2 #:pad-string "0")
          (~r (date-second timestamp) #:min-width 2 #:pad-string "0")))

(define out-port (open-output-file filename #:exists 'replace))
(fprintf out-port "Operation,Size,Iterations,TotalTime_ms,AvgTime_ms\n")

(define (time-op name params size iterations thunk)
  (printf "Benchmarking ~a, params: ~a,  (~a iterations)...\n" name params iterations)
  (collect-garbage)
  (define start (current-inexact-milliseconds))
  (for ([_ (in-range iterations)])
    (thunk))
  (define end (current-inexact-milliseconds))
  (define total-time (- end start))
  (define avg-time (/ total-time iterations))

  (printf "  Total: ~a ms\n" (~r total-time #:precision 2))
  (printf "  Avg:   ~a ms/op\n\n" (~r avg-time #:precision 4))

  ;; Write to CSV
  (fprintf out-port "~a,~a,~a,~a,~a\n"
           name size iterations
           (~r total-time #:precision 2)
           (~r avg-time #:precision 4)))

(define (bench-add-same-shape size)
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size size) 2.0 #:type 'double))

  (time-op "tensor-add"
           (list size size size size)
           (format "~ax~a" size size)
           N-ITERATIONS
           (lambda () (tensor-add A B))))

(define (bench-mul-same-shape size)
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size size) 2.0 #:type 'double))

  (time-op "tensor-mul"
           (list size size size size)
           (format "~ax~a" size size)
           N-ITERATIONS
           (lambda () (tensor-mul A B))))

(define (bench-add-broadcast size)
  ;; Broadcast vector to matrix
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size) 2.0 #:type 'double))

  (time-op "tensor-add-broadcast"
           (list size size size)
           (format "~ax~a + ~a" size size size)
           N-ITERATIONS
           (lambda () (tensor-add A B))))

(define (bench-mul-broadcast size)
  ;; Broadcast vector to matrix
  (define A (make-tensor (list size size) 1.0 #:type 'double))
  (define B (make-tensor (list size) 2.0 #:type 'double))

  (time-op "tensor-mul-broadcast"
           (list size size size)
           (format "~ax~a * ~a" size size size)
           N-ITERATIONS
           (lambda () (tensor-mul A B))))


(define (bench-skinny-no-alloc)
  (define rows 1000000)
  (define cols 4)
  (define A (make-tensor (list rows cols) 1.0 #:type 'double))
  (define B (make-tensor (list rows 1)    2.0 #:type 'double))
  (define out (tensor-add A B))

  (time-op "Skinny Kernel Only (No Alloc)"
           (list rows cols rows 1)
           (format "~ax~a" rows cols)
           N-ITERATIONS
           ;; Only measure the C-function call
           (lambda () (tensor-add-out! out A B))))


(define (bench-skinny-broadcast)
  ;; 1 Million rows, 4 columns
  ;; This forces the "Expensive Math" to run 1,000,000 times!
  (define rows 1000000)
  (define cols 4)

  (define A (make-tensor (list rows cols) 1.0 #:type 'double))
  (define B (make-tensor (list rows 1)    2.0 #:type 'double))

  (time-op "Skinny Broadcast (1M x 4)"
           (list rows cols rows 1)
           (format "~ax~a * ~a" rows cols rows)
           N-ITERATIONS
           (lambda () (tensor-add A B))))

;; Run benchmarks
(displayln "=== Tensor Operation Benchmarks ===\n")
(printf "Saving results to: ~a\n\n" filename)

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

;; tall and skinny
(bench-skinny-no-alloc)
(bench-skinny-broadcast)

(close-output-port out-port)
