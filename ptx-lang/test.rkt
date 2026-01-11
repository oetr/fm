#lang racket

(require "main.rkt"
         "compiler.rkt"
         "../unsafe/cuda.rkt"
         ffi/unsafe)

;; ---------------------------------------------------------
;; 1. Define the Kernel using the DSL
;; ---------------------------------------------------------
;; Note: We wrap this in a function or use syntax->datum if we want
;; to capture the string output programmatically.
;; For this example, we'll assume the DSL exports a way to get the string,
;; or we capture stdout.

(define (get-my-kernel-ptx)
  (with-output-to-bytes
    (lambda ()
      (kernel myTensorAdd
        #:params ((u64 ptr_A) (u64 ptr_B) (u64 ptr_C))
        ;; Added %valB to hold the second number
        #:regs   ((f32 %valA) (f32 %valB) (f32 %sum))
        
        (ld.global f32 %valA (%ptr_A))
        (ld.global f32 %valB (%ptr_B)) 
        (add f32 %sum %valA %valB)
        (st.global f32 (%ptr_C) %sum)
        (ret)))))


;; ---------------------------------------------------------
;; 3. Compile and Link
;; ---------------------------------------------------------
(define ptx-source (get-my-kernel-ptx))

;; RTX 3090
;; TODO: get the GPU arch from CUDA API
;;(define sass-binary (compile-ptx-to-sass ptx-source "sm_86" #:path "/usr/local/cuda-12.5/bin/ptxas"))

;; wrap GPU pointers in C pointers to be passed to cuLaunchKernel 
(define (pack-kernel-mem-args . args)
  (list->cblock
   (for/list ([arg args])
     (define mem (malloc _pointer))
     (ptr-set! mem _pointer arg)
     mem)
   _pointer))

(printf "\nReady to Launch!\n")
(printf "Kernel Name: myTensorAdd\n")
(printf "Binary Size: ~a bytes\n" (bytes-length ptx-source))
(printf "Arguments (Raw GPU Addresses):\n")

;; ---------------------------------------------------------
;; 2. Create Data (Tensors)
;; ---------------------------------------------------------
(cuInit 0)
(define device (cuDeviceGet 0))
;;(define device 0)
(define context (cuCtxCreate 0 device))

(define mod (cuModuleLoadData ptx-source))
(define func (cuModuleGetFunction mod "myTensorAdd"))

(define cpu-A (malloc 1 _float))
(define cpu-B (malloc 1 _float))
(define cpu-C (malloc 1 _float))
(define gpu-A (cuMemAlloc 4))
(define gpu-B (cuMemAlloc 4))
(define gpu-C (cuMemAlloc 4))

(ptr-set! cpu-A _float 10.0)
(ptr-set! cpu-B _float 32.0)

(cuMemcpyHtoD (cast (ptr-ref gpu-A _pointer) _pointer _uint64) cpu-A 4)
(cuMemcpyHtoD (cast (ptr-ref gpu-B _pointer)  _pointer _uint64) cpu-B 4)
(cuMemcpyDtoH cpu-C (cast (ptr-ref gpu-C _pointer) _pointer _uint64) 4)

(cuLaunchKernel func
                1 1 1 ;; grid dim
                1 1 1 ;; block dim
                0 ;; shared mem
                #f ;; stream
                (list->cblock (list gpu-A gpu-B gpu-C) _pointer)
                #f)

(cuMemcpyDtoH cpu-C (cast (ptr-ref gpu-C _pointer) _pointer _uint64) 4)

(ptr-ref cpu-C _float)




