#lang racket/base

(require (for-syntax racket/base
                     racket/syntax
                     racket/set
                     racket/string
                     racket/match
                     syntax/parse
                     syntax/stx)
         racket/string
         racket/match
         racket/list
         racket/format)

(provide (except-out (all-from-out racket/base) #%module-begin)
         (rename-out [ptx-module-begin #%module-begin])
         kernel)

;; ---------------------------------------------------------
;; 1. COMPILER LOGIC (Runtime Generation)
;; ---------------------------------------------------------
;; (Same as before, generates the strings)

(define label-counter 0)
(define (reset-labels!) (set! label-counter 0))

(define (fresh-label prefix)
  (set! label-counter (add1 label-counter))
  (format "~a_~a" prefix label-counter))

(define (format-op op)
  (match op
    [(list reg) (format "[~a]" reg)]
    [_ (symbol->string op)]))

(define (compile-instrs instrs)
  (flatten
   (for/list ([instr instrs])
     (match instr
       ;; Control Flow
       [(list 'ptx-loop idx start limit body ...)
        (define top (fresh-label "L_top"))
        (define end (fresh-label "L_end"))
        (list
         (format "\tmov.u32 \t~a, ~a;" idx start)
         (format "~a:" top)
         (format "\tsetp.ge.u32 \t%p1, ~a, ~a;" idx limit)
         (format "\t@%p1 bra \t~a;" end)
         (compile-instrs body)
         (format "\tadd.u32 \t~a, ~a, 1;" idx idx)
         (format "\tbra \t~a;" top)
         (format "~a:" end))]

       ;; Standard Instructions
       [(? string? s) s]
       [(list opcode type dest src ...)
        (format "\t~a.~a \t~a;" opcode type (string-join (map format-op (cons dest src)) ", "))]
       [(list opcode operand)
        (format "\t~a \t~a;" opcode operand)]
       [(list 'ret) "\tret;"]
       [_ (error "Runtime Error: Unknown instruction:" instr)]))))

;; ---------------------------------------------------------
;; 2. STATIC TYPE CHECKER (Compile-Time Validation)
;; ---------------------------------------------------------

(begin-for-syntax

  ;; Helper: Check if a syntax object is a register (starts with %)
  (define (is-register? stx)
    (and (identifier? stx)
         (string-prefix? (symbol->string (syntax-e stx)) "%")))

  ;; Helper: Check if an operand is compatible with a required type
  (define (check-operand-type stx expected-type type-map)
    (cond
      ;; Case 1: It is a Register
      [(is-register? stx)
       (define reg-name (syntax-e stx))
       (define actual-type (hash-ref type-map reg-name #f))

       (unless actual-type
         (raise-syntax-error 'ptx-error "Undeclared register" stx))

       (unless (eq? actual-type expected-type)
         (raise-syntax-error 'type-error
                             (format "Type mismatch. Instruction expects '~a', but register '~a' is '~a'."
                                     expected-type reg-name actual-type)
                             stx))]

      ;; Case 2: It is a Memory Access e.g., (%ptr)
      [(stx-pair? stx)
       (define inner (stx-car stx)) ;; Extract the register inside
       (define inner-type (hash-ref type-map (syntax-e inner) #f))
       ;; PTX Addresses must be u64 (on 64-bit systems)
       (unless (eq? inner-type 'u64)
         (raise-syntax-error 'type-error
                             (format "Memory address register '~a' must be u64, found '~a'."
                                     (syntax-e inner) inner-type)
                             stx))]

      ;; Case 3: It is an Immediate Value (Number)
      [(number? (syntax-e stx))
       ;; Loose check: Allow numbers for standard types
       (define n (syntax-e stx))
       (cond
         [(and (integer? n) (memq expected-type '(u32 u64 s32 s64))) (void)]
         [(and (real? n)    (memq expected-type '(f32 f64))) (void)]
         [else (raise-syntax-error 'type-error
                                   (format "Immediate value '~a' is not compatible with type '~a'." n expected-type)
                                   stx)])]

      [else (void)])) ;; Ignore other cases for simplicity

  ;; Main Type Checker Function
  (define (validate-types body-stx type-map)
    (define instrs (syntax->list body-stx))
    (for ([instr instrs])
      (syntax-parse instr
        ;; Rule 1: Typed Instructions (add u32 %a %b %c)
        [(opcode type:id dest src ...)
         (define t (syntax-e #'type))
         ;; Check all operands against the instruction type 't'
         (check-operand-type #'dest t type-map)
         (for ([s (syntax->list #'(src ...))])
           (check-operand-type s t type-map))]

        ;; Rule 2: Loops (ptx-loop index start limit body ...)
        [(loop-op:id idx start limit inner-body ...)
         #:when (eq? 'ptx-loop (syntax-e #'loop-op))
         ;; Loop indices MUST be u32 for our simple implementation
         (check-operand-type #'idx 'u32 type-map)
         (check-operand-type #'limit 'u32 type-map)
         ;; Recurse into body
         (validate-types #'(inner-body ...) type-map)]

        ;; Rule 3: Implicit/Untyped instructions (ret, bra) - ignored
        [_ (void)]))))

;; ---------------------------------------------------------
;; 3. THE MACRO
;; ---------------------------------------------------------

(define-syntax (kernel stx)
  (syntax-parse stx
    [(_ name
        #:params ((pt pn) ...)
        #:regs   ((rt rn) ...)
        instr ...)

     ;; --- A. Build Symbol Table ---
     (define param-types (syntax->datum #'(pt ...)))
     (define param-names (map (lambda (p) (string->symbol (format "%~a" (syntax-e p))))
                              (syntax->list #'(pn ...))))

     (define reg-types (syntax->datum #'(rt ...)))
     (define reg-names (syntax->datum #'(rn ...)))

     ;; Create Hash Map: { '%reg -> 'type }
     (define type-map (make-hash))

     ;; 1. Add Params
     (for ([n param-names] [t param-types])
       (hash-set! type-map n t))

     ;; 2. Add Registers
     (for ([n reg-names] [t reg-types])
       (hash-set! type-map n t))

     ;; 3. Add Implicit Predicate
     (hash-set! type-map '%p1 'pred)

     ;; --- B. Validate Body ---
     (validate-types #'(instr ...) type-map)

     ;; --- C. Expansion (Same as before) ---
     #'(begin
         (reset-labels!)
         (displayln (format "// Kernel: ~a" 'name))
         (displayln ".version 8.5\n.target sm_86\n.address_size 64\n")

         (printf ".visible .entry ~a(\n" 'name)
         (displayln (string-join
                     (map (lambda (t n) (format "\t.param .~a ~a" t n))
                          '(pt ...) '(pn ...)) ",\n"))
         (displayln ") {")

         (displayln "\t.reg .pred %p1;")
         (for ([t '(rt ...)] [n '(rn ...)])
           (printf "\t.reg .~a \t~a;\n" t n))
         ;; Emit registers for parameters
         (for ([t '(pt ...)] [n '(pn ...)])
           (printf "\t.reg .~a \t%~a;\n" t n))
         (newline)

         (displayln "\t// Preamble: Load params")
         (for ([t '(pt ...)] [n '(pn ...)])
           (printf "\tld.param.~a \t%~a, [~a];\n" t n n))
         (newline)

         (displayln "\t// Kernel Body")
         (for ([line (compile-instrs (list 'instr ...))])
           (displayln line))

         (displayln "}"))]))

(define-syntax-rule (ptx-module-begin form ...)
  (#%module-begin form ...))
