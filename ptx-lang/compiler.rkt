#lang racket/base
(require racket/system racket/file racket/port)

(provide compile-ptx-to-sass)

(define (compile-ptx-to-sass ptx-code [arch "sm_50"]
                             #:path [path "ptxas"])
  (define ptx-filename "temp_kernel.ptx")
  (define sass-filename "temp_kernel.cubin")

  ;; 1. Write the DSL output to disk
  (display-to-file ptx-code ptx-filename #:exists 'replace)

  ;; 2. Invoke NVIDIA's ptxas compiler
  ;; -v : verbose (shows register pressure)
  ;; -o : output filename
  (printf "Compiling to SASS for ~a...\n" arch)
  (define cmd (format "~a -v --gpu-name=~a ~a -o ~a"
                      path arch ptx-filename sass-filename))

  (unless (system cmd)
    (error "Compilation failed! Check your PTX syntax."))

  ;; 3. Read the binary (SASS/Cubin) back into Racket
  (define binary-data (file->bytes sass-filename))

  ;; Cleanup
  (delete-file ptx-filename)
  (delete-file sass-filename)

  (printf "Success! Loaded ~a bytes of SASS.\n" (bytes-length binary-data))
  binary-data)
