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


(define (accumulate-grad! input-tensor out)
  (define type-A (tensor-type input-tensor))
  (define grad-out (tensor-grad out))
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
       (for/fold ([ks '()]
                  [kss '()] [kds '()]
                  [rs '()]  [rss '()])
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

(define+provide (tensor-sub A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))

  (unless (symbol=? type-A type-B)
    (error 'tensor-sub "both tensors should be of the same type, but were: ~a and ~a~n" type-A type-B))

  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  (define-values (out-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (define out (make-tensor out-shape #:type type-A #:children (list A B)))

  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
      (raise-argument-error 'tensor-sub "tensor:double or tensor:float" 0 A B))
    (unless (or (symbol=? type-B 'double) (symbol=? type-B 'float))
      (raise-argument-error 'tensor-sub "tensor:double or tensor:float" 1 A B))
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (let-typed-c type-A
    ([sub! _sub_broadcast])
    (sub! (length out-shape)
          out-shape
          (tensor-data A)
          broadcast-strides-A
          (tensor-data B)
          broadcast-strides-B
          (tensor-data out)))

  (define (backward)
    (accumulate-grad! A out)
    ;; For B, we accumulate -1 * grad_out
    ;; We can create a temporary view or negative gradient?
    ;; Or use a weighted accumulation.
    ;; Current accumulate-grad! is hardcoded to +1.0 axpy/reduce.
    ;; We need to subtract.
    ;; Let's multiply grad_out by -1 temporarily? No, that modifies shared state.
    ;;
    ;; Better: tensor-grad-fill with -1? No.
    ;;
    ;; Let's implement accumulation with a scalar multiplier in accumulate-grad!.
    ;; But accumulate-grad! uses _reduce_sum_generic which is raw sum.
    ;;
    ;; Quick fix: Scale tensor-grad(out) by -1, accumulate to B, then scale back.
    ;; This is safe because backward passes are sequential in our engine (for now).
    ;; Wait, if `out` is used multiple times, we can't mutate its grad like that.
    ;;
    ;; Correct approach:
    ;; dA += dC
    ;; dB += -dC
    ;;
    ;; We can make a negative view of dC?
    ;; Or just write a explicit "accumulate-sub-grad!"
    (accumulate-grad-negative! B out))

  (when grad?
    (set-tensor-backward! out backward))
  out)

(define (accumulate-grad-negative! input-tensor out)
  ;; This is accumulate-grad! but with a negative sign during accumulation.
  ;; We can reuse the logic but need a negative-adding kernel.
  ;;
  ;; Strategy:
  ;; 1. Alloc temp = -1 * grad(out)
  ;; 2. Accumulate temp into grad(input)
  ;; This works for both 1:1 and broadcast cases.
  (define type (tensor-type out))
  (define grad-out (tensor-grad out))
  (define temp (alloc-tensor (tensor-length out) #:type (symbol->type type)))

  (let-typed type
    ([copy! cblas_dcopy cblas_scopy]
     [scal! cblas_dscal cblas_sscal])
    (copy! (tensor-length out) grad-out 1 temp 1)
    (scal! (tensor-length out) -1.0 temp 1))

  ;; Now treat 'temp' as the source gradient
  ;; We need a version of accumulate-grad! that takes an explicit source buffer
  ;; instead of (tensor-grad out).
  (accumulate-grad-from-buffer! input-tensor out temp)
  ;; Free temp? (GC handles it)
  )

;; Refactored helper to allow custom source gradient (e.g. for sub or mul)
(define (accumulate-grad-from-buffer! input-tensor out-tensor source-grad-ptr)
  (define type-A (tensor-type input-tensor))
  (define in-shape (tensor-shape input-tensor))
  (define out-shape (tensor-shape out-tensor))

  (cond
    [(equal? in-shape out-shape)
     (let-typed type-A
       ([axpy cblas_daxpy cblas_saxpy])
       (axpy (tensor-length out-tensor) 1.0 source-grad-ptr 1 (tensor-grad input-tensor) 1))]

    [else
     (define rank-out (length out-shape))
     (define pad-len  (- rank-out (length in-shape)))
     (define padded-in-shape (append (make-list pad-len 1) in-shape))

     (define-values (keep-shape keep-src-st keep-dst-st
                                reduce-shape reduce-src-st)
       (for/fold ([ks '()] [kss '()] [kds '()] [rs '()] [rss '()])
                 ([d-in  (in-list padded-in-shape)]
                  [d-out (in-list out-shape)]
                  [st-out (in-list (tensor-strides out-tensor))]
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
                source-grad-ptr
                (tensor-grad input-tensor)))]))

(define+provide (tensor-div A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))

  (unless (symbol=? type-A type-B)
    (error 'tensor-div "both tensors should be of the same type, but were: ~a and ~a~n" type-A type-B))

  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  (define-values (out-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (define out (make-tensor out-shape #:type type-A #:children (list A B)))

  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (when grad?
    (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
      (raise-argument-error 'tensor-div "tensor:double or tensor:float" 0 A B))
    (tensor-requires-grad! out)
    (tensor-requires-grad! A)
    (tensor-requires-grad! B))

  (let-typed-c type-A
    ([div! _div_broadcast])
    (div! (length out-shape)
          out-shape
          (tensor-data A)
          broadcast-strides-A
          (tensor-data B)
          broadcast-strides-B
          (tensor-data out)))

  (define (backward)
    ;; C = A / B
    ;; dA = dC / B = dC * (1/B)
    ;; dB = -dC * A / B^2 = -dC * C / B

    ;; For dA: Accumulate (dC / B)
    (define inv-B (tensor-pow B (make-tensor '() -1.0 #:type type-B))) ;; 1/B
    ;; Warning: Creating intermediate tensors here might be slow.
    ;; Ideally we want a kernel "accumulate_div_grad(dC, B, dA)" and "accumulate_div_grad_B(dC, A, B, dB)"

    ;; Using existing primitives for now to save dev time, optimize later if needed.
    ;; But wait, tensor-pow isn't defined yet in Racket land (only C).
    ;; Let's define tensor-pow first.
    ;;
    ;; Actually, let's just use the logic from tensor-mul backward but adapted.
    ;; dA term: dC * (1/B). We need a temporary tensor for (1/B) broadcasted?
    ;; No, dC is shape(out), B is shape(B).
    ;; We can compute temp = dC / B (broadcasted).
    ;; Then accumulate temp into A.

    (define ndim (length out-shape))
    (define temp-grad-A (alloc-tensor (tensor-length out) #:type (symbol->type type-A)))

    ;; temp_A = grad_out / B
    (let-typed-c type-A
      ([div! _div_broadcast])
      (div! ndim
            out-shape
            (tensor-grad out) (shape->strides out-shape)
            (tensor-data B)   broadcast-strides-B
            temp-grad-A))

    (accumulate-grad-from-buffer! A out temp-grad-A)

    ;; dB term: -dC * A / (B*B)
    ;;        = - (dC * A / B^2)
    ;;        = - (temp_A * A / B)
    ;;        = - temp_A * (A/B)
    ;;        = - temp_A * out  (since out = A/B)
    ;; So dB term is -1 * temp_A * out.

    (define temp-grad-B (alloc-tensor (tensor-length out) #:type (symbol->type type-A)))

    ;; temp_B = temp_A * out
    (let-typed-c type-A
      ([mul! _mul_broadcast])
      (mul! ndim
            out-shape
            temp-grad-A (shape->strides out-shape)
            (tensor-data out) (shape->strides out-shape)
            temp-grad-B))

    ;; Negate temp_B
    (let-typed type-A
      ([scal! cblas_dscal cblas_sscal])
      (scal! (tensor-length out) -1.0 temp-grad-B 1))

    (accumulate-grad-from-buffer! B out temp-grad-B)
    )

  (when grad?
    (set-tensor-backward! out backward))
  out)


(define+provide (tensor-pow A B)
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))

  (unless (symbol=? type-A type-B)
    (error 'tensor-pow "both tensors should be of the same type"))

  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  (define-values (out-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (define out (make-tensor out-shape #:type type-A #:children (list A B)))

  (define grad? (or (not (empty? (tensor-grad A))) (not (empty? (tensor-grad B)))))

  (when grad?
    (unless (or (symbol=? type-A 'double) (symbol=? type-A 'float))
      (raise-argument-error 'tensor-pow "tensor:double or tensor:float" 0 A B))
    (tensor-requires-grad! out A B))

  (let-typed type-A
    ([pow! _pow_broadcast_double _pow_broadcast_float])
    (pow! (length out-shape)
          out-shape
          (tensor-data A) broadcast-strides-A
          (tensor-data B) broadcast-strides-B
          (tensor-data out)))

  (define (backward)
    ;; C = A ^ B
    ;; dA = dC * B * A^(B-1) = dC * B * (C/A) = dC * C * B / A
    ;; dB = dC * A^B * ln(A) = dC * C * ln(A)

    (define ndim (length out-shape))
    (define len (tensor-length out))
    (define type (symbol->type type-A))

    ;; Common term: dC * C
    (define dC_C (alloc-tensor len #:type type))
    (let-typed-c type-A
      ([mul! _mul_broadcast])
      (mul! ndim out-shape
            (tensor-grad out) (shape->strides out-shape)
            (tensor-data out) (shape->strides out-shape)
            dC_C))

    ;; dA term: (dC * C) * B / A
    ;; temp = (dC_C * B) / A
    (define temp-dA (alloc-tensor len #:type type))

    ;; 1. temp = dC_C * B
    (let-typed-c type-A
      ([mul! _mul_broadcast])
      (mul! ndim out-shape
            dC_C (shape->strides out-shape)
            (tensor-data B) broadcast-strides-B
            temp-dA))

    ;; 2. temp = temp / A
    (let-typed-c type-A
      ([div! _div_broadcast])
      (div! ndim out-shape
            temp-dA (shape->strides out-shape)
            (tensor-data A) broadcast-strides-A
            temp-dA))

    (accumulate-grad-from-buffer! A out temp-dA)

    ;; dB term: (dC * C) * ln(A)
    ;; temp = dC_C * ln(A)
    (define temp-dB (alloc-tensor len #:type type))
    (define ln-A (alloc-tensor len #:type type))

    ;; We need to compute ln(A) but broadcasted to out-shape!
    ;; Or we can compute ln(A) in place? No A is const.
    ;;
    ;; Let's make a broadcasted copy of A, then log it.
    ;; "copy broadcast" is just mul by 1s? Or add 0s.
    (define ones (make-tensor '() 0.0 #:type type-A)) ;; 0s
    (let-typed-c type-A
      ([add! _add_broadcast])
      (add! ndim out-shape
            (tensor-data A) broadcast-strides-A
            (tensor-data ones) (make-list ndim 0) ;; Broadcast 0, strides should be valid list of ints
            ln-A))

    ;; Now log(ln-A)
    (let-typed type-A
      ([log! _log_double _log_float])
      (log! ln-A ln-A len))

    ;; temp-dB = dC_C * ln-A
    (let-typed-c type-A
      ([mul! _mul_broadcast])
      (mul! ndim out-shape
            dC_C (shape->strides out-shape)
            ln-A (shape->strides out-shape)
            temp-dB))

    (accumulate-grad-from-buffer! B out temp-dB)
    )

  (when grad?
    (set-tensor-backward! out backward))
  out)

(define+provide (tensor-exp T)
  (define type (tensor-type T))
  (unless (or (symbol=? type 'double) (symbol=? type 'float))
    (raise-argument-error 'tensor-exp "tensor:double or tensor:float" 0 T))

  (define out (make-tensor (tensor-shape T) #:type type #:children (list T)))

  (when (not (empty? (tensor-grad T)))
    (tensor-requires-grad! out))

  (let-typed type
    ([exp! _exp_double _exp_float])
    (exp! (tensor-data out) (tensor-data T) (tensor-length T)))

  (define (backward)
    ;; y = exp(x)
    ;; dy/dx = y
    ;; grad_in += grad_out * y
    (define len (tensor-length T))
    (let-typed-c type
      ([mul-add! _mul_then_add])
      (mul-add! (tensor-grad T)
                (tensor-grad out)
                (tensor-data out)
                len)))

  (when (not (empty? (tensor-grad T)))
    (set-tensor-backward! out backward))
  out)

(define+provide (tensor-log T)
  (define type (tensor-type T))
  (unless (or (symbol=? type 'double) (symbol=? type 'float))
    (raise-argument-error 'tensor-log "tensor:double or tensor:float" 0 T))

  (define out (make-tensor (tensor-shape T) #:type type #:children (list T)))

  (when (not (empty? (tensor-grad T)))
    (tensor-requires-grad! out))

  (let-typed type
    ([log! _log_double _log_float])
    (log! (tensor-data out) (tensor-data T) (tensor-length T)))

  (define (backward)
    ;; y = ln(x)
    ;; dy/dx = 1/x
    ;; grad_in += grad_out / x
    ;; Since we don't have _div_then_add, we compute temp = grad_out/x then add?
    ;; Or use: grad_out * (1/x).
    ;; Let's alloc temp.
    (define len (tensor-length T))
    (define temp (alloc-tensor len #:type (symbol->type type)))

    ;; temp = grad_out / x
    (let-typed-c type
      ([div! _div_broadcast])
      ;; Using broadcast div with same shapes (effectively elementwise)
      (div! (tensor-rank T) (tensor-shape T)
            (tensor-grad out) (tensor-strides out)
            (tensor-data T) (tensor-strides T)
            temp))

    ;; accumulate
    (let-typed type
      ([axpy cblas_daxpy cblas_saxpy])
      (axpy len 1.0 temp 1 (tensor-grad T) 1)))

  (when (not (empty? (tensor-grad T)))
    (set-tensor-backward! out backward))
  out)

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
    (accumulate-grad! A out)
    (accumulate-grad! B out))

  (when grad?
    (set-tensor-backward! out backward))
  out)

(define+provide (tensor-add-out! out A B)
  (define type-out (tensor-type out))
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))

  ;; 1. Type Validation
  (unless (and (symbol=? type-out type-A) (symbol=? type-out type-B))
    (error 'tensor-add-out!
           "Type mismatch: out=~a, A=~a, B=~a. All must match."
           type-out type-A type-B))

  ;; 2. Shape Validation
  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  ;; Calculate what the result shape should be
  (define-values (expected-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (unless (equal? (tensor-shape out) expected-shape)
    (error 'tensor-add-out!
           "Shape mismatch: 'out' has shape ~a but result of A+B is ~a"
           (tensor-shape out) expected-shape))

  ;; Aliasing Safety Check (The 'Restrict' Guard)
  (define ptr-out (tensor-data out))
  (define ptr-A (tensor-data A))
  (define ptr-B (tensor-data B))

  (when (or (equal? ptr-out ptr-A)
            (equal? ptr-out ptr-B))
    (error 'tensor-add-out!
           "Aliasing detected: 'out' shares memory with input A or B.
           The current C kernel uses 'restrict' and cannot handle in-place operations safe.
           Please use a distinct tensor for output."))

  ;; Autograd Graph Wiring
  (define A-grad? (not (empty? (tensor-grad A))))
  (define B-grad? (not (empty? (tensor-grad B))))
  (define grad? (or A-grad? B-grad?))

  (cond
    [grad?
     ;; Ensure 'out' can propagate gradients back
     (tensor-requires-grad! out)
     (tensor-requires-grad! A)
     (tensor-requires-grad! B)

     ;; Wire the graph: out is now the parent of A and B
     (set-tensor-children! out (list A B))

     ;; Define the backward pass
     ;; Note: Addition is stateless, so we don't need to capture 'A' or 'B' values,
     ;; just their shapes/nodes. This is safe even if A or B change later.
     (define (backward)
       (accumulate-grad! A out)
       (accumulate-grad! B out))

     (set-tensor-backward! out backward)]

    [else
     ;; Else: Detach 'out' from any previous history to prevent zombie graphs
     (begin
       (set-tensor-children! out empty)
       (set-tensor-backward! out void))])

  ;; 5. Execution (The Fast Path)
  (let-typed-c type-out
    ([add! _add_broadcast])
    (add! (tensor-rank out)
          (tensor-shape out)
          (tensor-data A)   broadcast-strides-A
          (tensor-data B)   broadcast-strides-B
          (tensor-data out)))

  ;; Return 'out' to allow chaining (e.g., (tensor-print (tensor-add-out! ...)))
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

(define+provide (tensor-mul-out! out A B)
  (define type-out (tensor-type out))
  (define type-A (tensor-type A))
  (define type-B (tensor-type B))

  (unless (and (symbol=? type-out type-A) (symbol=? type-out type-B))
    (error 'tensor-mul-out! "Type mismatch"))

  (define shape-A (tensor-shape A))
  (define shape-B (tensor-shape B))
  (define strides-A (tensor-strides A))
  (define strides-B (tensor-strides B))

  (define-values (expected-shape broadcast-strides-A broadcast-strides-B)
    (broadcast-structs shape-A strides-A shape-B strides-B))

  (unless (equal? (tensor-shape out) expected-shape)
    (error 'tensor-mul-out! "Shape mismatch"))

  (define ptr-out (tensor-data out))
  (define ptr-A (tensor-data A))
  (define ptr-B (tensor-data B))

  (when (or (equal? ptr-out ptr-A) (equal? ptr-out ptr-B))
    (error 'tensor-mul-out! "Aliasing detected"))

  (define grad? (or (not (empty? (tensor-grad A))) (not (empty? (tensor-grad B)))))

  (cond
    [grad?
     (tensor-requires-grad! out A B)
     (set-tensor-children! out (list A B))
     (define (backward)
       ;; Copied from tensor-mul backward logic
       (define (accumulate-grad-mul! input-tensor other-tensor grad-out broadcast-strides-input broadcast-strides-other)
         ;; ... (same logic as before, but accessing out/A/B from closure)
         ;; We need to factor out the accumulation logic or duplicate it.
         ;; Given the complexity, let's call the helper if we can.
         ;; But the helper was internal to tensor-mul.
         ;; Let's inline a simplified version using the new accumulate-grad-from-buffer!
         (define ndim (length expected-shape))
         (define temp-grad (alloc-tensor (tensor-length out) #:type (symbol->type type-A)))

         (let-typed-c type-A
           ([mul! _mul_broadcast])
           (mul! ndim expected-shape
                 grad-out (shape->strides expected-shape)
                 (tensor-data other-tensor) broadcast-strides-other
                 temp-grad))

         (accumulate-grad-from-buffer! input-tensor out temp-grad))

       (accumulate-grad-mul! A B (tensor-grad out) broadcast-strides-A broadcast-strides-B)
       (accumulate-grad-mul! B A (tensor-grad out) broadcast-strides-B broadcast-strides-A))

     (set-tensor-backward! out backward)]
    [else
     (set-tensor-children! out empty)
     (set-tensor-backward! out void)])

  (let-typed-c type-out
    ([mul! _mul_broadcast])
    (mul! (length expected-shape)
          expected-shape
          (tensor-data A) broadcast-strides-A
          (tensor-data B) broadcast-strides-B
          (tensor-data out)))
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


