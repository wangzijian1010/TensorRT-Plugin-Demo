cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(layernorm_plugin LANGUAGES CXX CUDA)

# Require C++11 and CUDA (use newer standard as needed)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# (Optional) Specify CUDA architectures
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86) # adjust to your GPU

# Find CUDA toolkit (CMake 3.18+ provides CUDA::cudart)
find_package(CUDAToolkit REQUIRED)

# List plugin source files (CUDA kernel and C++ plugin)
set(SRC_FILES
    layernorm_kernel.cu
    layernorm_plugin.cpp
    # (add other sources if needed)
)

# Define shared library target
add_library(layernormplugin SHARED ${SRC_FILES})

# Enable separable compilation so that CUDA kernels are linked in
set_target_properties(layernormplugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include TensorRT and CUDA include dirs (adjust TENSORRT_ROOT if needed)
# e.g. target_include_directories(layernormplugin PRIVATE ${TENSORRT_ROOT}/include)
# CUDA include is handled by find_package(CUDAToolkit)

# Link against TensorRT and CUDA runtime libraries
target_link_libraries(layernormplugin PRIVATE
    nvinfer            # TensorRT core library
    nvinfer_plugin     # TensorRT plugin registry
    CUDA::cudart       # CUDA runtime
)

# Set output directory for the plugin library
set_target_properties(layernormplugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "/home/TensorRT_Plugin_Demo/layernorm"
)
