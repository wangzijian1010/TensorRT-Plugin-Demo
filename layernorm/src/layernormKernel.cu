#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// CUDA LayerNorm 核函数
__global__ void layerNormKernel(
        float* output, const float* input, const float* gamma, const float* beta,
        int batchSize, int seqLength, int hiddenSize, float epsilon)
{
    // 每个线程处理一个序列中的一个位置（一行）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 遍历所有批次和序列位置
    for (int i = idx; i < batchSize * seqLength; i += stride) {
        // 计算当前处理的行在展平输入中的起始索引
        int row_start = i * hiddenSize;

        // 步骤1: 计算均值
        float mean = 0.0f;
        for (int j = 0; j < hiddenSize; j++) {
            mean += input[row_start + j];
        }
        mean /= hiddenSize;

        // 步骤2: 计算方差
        float variance = 0.0f;
        for (int j = 0; j < hiddenSize; j++) {
            float diff = input[row_start + j] - mean;
            variance += diff * diff;
        }
        variance /= hiddenSize;

        // 步骤3: 规范化、缩放和平移
        float inv_std = rsqrtf(variance + epsilon);
        for (int j = 0; j < hiddenSize; j++) {
            float normalized = (input[row_start + j] - mean) * inv_std;
            output[row_start + j] = gamma[j] * normalized + beta[j];
        }
    }
}

// C++ 函数接口，在 layernormPlugin.cpp 中调用
void launchLayerNormKernel(
        float* output, const float* input, const float* gamma, const float* beta,
        int batchSize, int seqLength, int hiddenSize, float epsilon, cudaStream_t stream)
{
    // 配置CUDA核函数调用参数
    const int blockSize = 256;
    const int gridSize = (batchSize * seqLength + blockSize - 1) / blockSize;

    // 启动CUDA核函数
    layerNormKernel<<<gridSize, blockSize, 0, stream>>>(
            output, input, gamma, beta, batchSize, seqLength, hiddenSize, epsilon);
}
