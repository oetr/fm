;; Copyright © 2025 Peter Samarin <peter.samarin@gmail.com>
;; License: GNU AGPLv3
#lang racket
(require (for-syntax racket/syntax))


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

(provide let-typed-c)
(define-syntax (let-typed-c stx)
  (syntax-case stx ()
    [(_ type-expr ([local-id base-id] ...) body ...)
     (with-syntax ([(double-id ...) (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_double" b))]
                   [(float-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_float" b))]
                   [(uint64-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_uint64" b))]
                   [(int64-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_int64" b))]
                   [(int32-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_int32" b))]
                   [(uint32-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                       (format-id stx "~a_uint32" b))]
                   [(int16-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_int16" b))]
                   [(uint16-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                       (format-id stx "~a_uint16" b))]
                   [(int8-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_int8" b))]
                   [(uint8-id ...)  (for/list ([b (syntax->list #'(base-id ...))])
                                      (format-id stx "~a_uint8" b))]
                   )
       #'(case type-expr
           [(double)
            (let ([local-id double-id] ...)
              body ...)]
           [(float)
            (let ([local-id float-id] ...)
              body ...)]
           [(int64)
            (let ([local-id int64-id] ...)
              body ...)]
           [(unt64)
            (let ([local-id uint64-id] ...)
              body ...)]
           [(unt32)
            (let ([local-id uint32-id] ...)
              body ...)]
           [(int32)
            (let ([local-id int32-id] ...)
              body ...)]
           [(unt16)
            (let ([local-id uint16-id] ...)
              body ...)]
           [(int16)
            (let ([local-id int16-id] ...)
              body ...)]
           [(unt8)
            (let ([local-id uint8-id] ...)
              body ...)]
           [(int8)
            (let ([local-id int8-id] ...)
              body ...)]
           [else (error "Unsupported type")]))]))

(module+ test
  (require rackunit)

  (define (_bla_double) 'double)
  (define (_bla_float) 'float)
  (define (_bla_uint64) 'uint64)
  (define (_bla_int64) 'int64)
  (define (_bla_uint32) 'uint32)
  (define (_bla_int32) 'int32)
  (define (_bla_uint16) 'uint16)
  (define (_bla_int16) 'int16)
  (define (_bla_uint8) 'uint8)
  (define (_bla_int8) 'int8)

  (for ([type '(int32)])
    (test-eq? (symbol->string type)
              (let-typed-c type ([bla _bla]) (bla))
              type))
  )

