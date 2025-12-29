(cd C
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-save-temps"
  cmake --build build --config Release
)
