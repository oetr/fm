#lang racket

(require "main.rkt"
         "compiler.rkt"
         ffi/unsafe)

;; ---------------------------------------------------------
;; 1. Define the Kernel using the DSL
;; ---------------------------------------------------------
;; Note: We wrap this in a function or use syntax->datum if we want
;; to capture the string output programmatically.
;; For this example, we'll assume the DSL exports a way to get the string,
;; or we capture stdout.

(define (get-my-kernel-ptx)
  (with-output-to-string
    (lambda ()
      (kernel myTensorAdd
        #:params ((u64 ptr_A) (u64 ptr_B) (u64 ptr_C))
        #:regs   ((u32 %i) (u32 %N) (u64 %offset) (u64 %addrA) (u64 %addrB) (u64 %addrC) (f32 %val))
          ;; Simple logic: C[0] = A[0] + B[0] (Just for a pointer demo)
          (ld.global f32 %val (%ptr_A)) ;; Load from A
          (st.global f32 (%ptr_C) %val) ;; Store to C
          (ret)
        ))))

;; ---------------------------------------------------------
;; 2. Create Data (Tensors)
;; ---------------------------------------------------------
(define tensor-A (malloc 1024 _float))
(define tensor-B (malloc 1024 _float))
(define tensor-C (malloc 1024 _float))

(define (tensor-ptr-as-int mem)
  (cast mem _pointer _uint64))

(printf "Tensor A as u64:  0x~x\n" (tensor-ptr-as-int tensor-A))

;; ---------------------------------------------------------
;; 3. Compile and Link
;; ---------------------------------------------------------

(define ptx-source (get-my-kernel-ptx))
(define sass-binary (compile-ptx-to-sass ptx-source "sm_75" #:path "/usr/local/cuda-12.5/bin/ptxas"))

;; ---------------------------------------------------------
;; 4. The Launch Configuration (Simulated)
;; ---------------------------------------------------------

;; When using the CUDA Driver API (cuLaunchKernel), you pass an array of void*
;; pointing to the arguments.
;;
;; Argument 0: ptr_A (u64)
;; Argument 1: ptr_B (u64)
;; Argument 2: ptr_C (u64)

(define (prepare-launch-args . tensors)
  (map tensor-ptr-as-int tensors))

(define kernel-args (prepare-launch-args tensor-A tensor-B tensor-C))

(printf "\nReady to Launch!\n")
(printf "Kernel Name: myTensorAdd\n")
(printf "Binary Size: ~a bytes\n" (bytes-length sass-binary))
(printf "Arguments (Raw GPU Addresses):\n")
(for ([arg kernel-args] [i (in-naturals)])
  (printf "  Arg ~a: 0x~x\n" i arg))

;; HERE you would call the FFI function:
;; (cuLaunchKernel function-handle
;;                 gridX gridY gridZ
;;                 blockX blockY blockZ
;;                 sharedMem stream
;;                 (list->cblock kernel-args)
;;                 #f)
