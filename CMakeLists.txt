cmake_minimum_required(VERSION 3.25)

# 显式设置 CUDA 工具链
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(TensorRTPluginDemo LANGUAGES CXX CUDA)

# 设置 CUDA 架构 - RTX 4090 使用 Ada Lovelace 架构 (SM 8.9 = 89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(
        /usr/local/cuda/include
        /usr/local/tensorrt/include
        ${CMAKE_SOURCE_DIR}
)

# 链接目录
link_directories(
        /usr/local/cuda/lib64
        /usr/local/tensorrt/lib
)

# 定义插件库
add_library(myselu_plugin SHARED
        myselu_plugin.cpp  # 确保文件名正确
        myselu_kernel.cu
)

# 插件库链接
target_link_libraries(myselu_plugin
        nvinfer
        nvinfer_plugin
        cudart
)

# 设置插件库输出位置
set_target_properties(myselu_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# 主程序构建
add_executable(main main.cpp)

# 主程序链接
target_link_libraries(main
        nvinfer
        cudart
        nvonnxparser
        myselu_plugin
        nvinfer_plugin
)

add_subdirectory(layernorm)
