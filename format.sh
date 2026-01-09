#find . -name "*.rkt" | xargs raco fmt --config - -i  # not my favorite formatting
find . -path "./C/build" -prune -o \( -name "*.c" -o -name "*.h" \) -print | xargs clang-format -i
find . -name "*.rkt" | xargs perl -pi -e 's/ +$//'

