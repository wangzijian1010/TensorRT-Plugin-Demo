#pragma once
#include <cuda_runtime.h>

namespace layernorm {
void layernormKernelLauncher(
    const float* input, 
    float* output, 
    int batch_size, 
    int hidden_size, 
    float epsilon, 
    cudaStream_t stream);
}
