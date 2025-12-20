#find . -name "*.rkt" | xargs raco fmt --config - -i
find . -name "*.c" -o -name "*.h" | xargs clang-format -i
