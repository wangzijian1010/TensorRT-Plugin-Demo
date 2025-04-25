#include "cuda_runtime.h"

extern "C" void myselu_inference(const float* x, float* output, int n, cudaStream_t stream);