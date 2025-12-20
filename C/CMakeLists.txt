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

# Enable SIMD instructions
if(MSVC)
    target_compile_options(aux PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(aux PRIVATE -O3 -mavx2 -mfma)
endif()
