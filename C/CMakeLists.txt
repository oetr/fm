cmake_minimum_required(VERSION 3.15)
project(RacketTensorEngine C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_VISIBILITY_PRESET hidden) # Hide internal symbols by default

# Find Dependencies
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# Define Library
add_library(aux SHARED aux.c)

target_link_libraries(aux PRIVATE 
    OpenMP::OpenMP_C 
    ${BLAS_LIBRARIES}
)

# Handle cross-platform PIC
set_target_properties(aux PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    # Windows
    target_compile_options(aux PRIVATE
        /O2          # Max Speed
        /arch:AVX2   # Hardware: Enable AVX2 registers
        /fp:fast     # Math: Allow aggressive reordering and FMA usage
        /Oi          # Enable intrinsic functions
    )
else()
    # Linux / macOS (GCC, Clang)
    target_compile_options(aux PRIVATE
        -O3                  # Max Optimization
        -mavx2               # Hardware: Enable AVX2
        -mfma                # Hardware: Enable FMA instructions
        -ffp-contract=fast   # Math: Force fusion of multiply-add
        -fno-math-errno      # Math: Don't set 'errno' (allows vectorizing sqrt/pow)
    )
endif()
