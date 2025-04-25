# 生成简单的CMAKE
cmake_minimum_required(VERSION 3.25)
project(TensorRTPluginDemo)

include_directories(/usr/local/cuda/include)
include_directories(/usr/local/tensorrt/include)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/tensorrt/lib)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)

# 在 CMake 中定义插件库
add_library(layernormplugin SHARED
        src/layernormKernel.cu
        src/layernormPlugin.cpp
)

# 指定需要链接的库
target_link_libraries(layernormplugin
        nvinfer
        nvinfer_plugin
        cudart
)

# 指定插件库的位置，使用绝对路径
set_target_properties(layernormplugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "/home/TensorRT_Plugin_Demo/cmake-build-debug"
)



add_executable(test_layernorm test/testLayerNorm.cpp)
target_link_libraries(test_layernorm layernorm_plugin)


