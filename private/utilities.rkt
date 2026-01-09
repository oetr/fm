;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket

(provide define+provide)

(define-syntax define+provide
  (syntax-rules ()
    [(_ (name args ... . rest) body ...)
     (begin
       (provide name)
       (define (name args ... . rest)
         body ...))]
    [(_ (name args ...) body ...)
     (begin
       (provide name)
       (define (name args ...)
         body ...))]
    [(_ name body)
     (begin
       (provide name)
       (define name body))]))

(provide let-typed)
(define-syntax (let-typed stx)
  (syntax-case stx ()
    [(_ type ([id double-val float-val] ...) body ...)
     #'(if (symbol=? type 'double)
           (let ([id double-val] ...)
             body ...)
           (let ([id float-val] ...)
             body ...))]))
