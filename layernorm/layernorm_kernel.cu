#include "layernorm_kernel.h"
#include <cuda_fp16.h>

namespace layernorm {

__global__ void layernormKernel(
    const float* input, 
    float* output, 
    int hidden_size, 
    float epsilon) 
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    const float* input_batch = input + batch_idx * hidden_size;
    float* output_batch = output + batch_idx * hidden_size;

    // 计算均值
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += input_batch[i];
    }
    shared[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        float mean = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            mean += shared[i];
        }
        mean /= hidden_size;
        shared[0] = mean;
    }
    __syncthreads();

    // 计算方差
    float mean = shared[0];
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input_batch[i] - mean;
        var_sum += diff * diff;
    }
    shared[tid] = var_sum;
    __syncthreads();

    if (tid == 0) {
        float var = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            var += shared[i];
        }
        var /= hidden_size;
        shared[0] = var;
    }
    __syncthreads();

    // 最后归一化
    float var = shared[0];
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        output_batch[i] = (input_batch[i] - mean) / sqrtf(var + epsilon);
    }
}

void layernormKernelLauncher(
    const float* input, 
    float* output, 
    int batch_size, 
    int hidden_size, 
    float epsilon, 
    cudaStream_t stream) 
{
    int threads = min(hidden_size, 256);
    layernormKernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        input, output, hidden_size, epsilon);
}

} // namespace layernorm
