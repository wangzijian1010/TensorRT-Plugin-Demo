#include "NvInfer.h"
#include "../include/layernormPlugin.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

// 自定义TensorRT日志记录器
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// 安全删除函数
template <typename T>
void safeCudaFree(T& ptr)
{
    if (ptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

// 主测试函数
int main()
{
    // 创建一个简单的网络来测试LayerNorm插件
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // 创建输入层
    nvinfer1::Dims4 inputDims(1, 1, 1, 64); // 批次大小1，序列长度1，隐藏大小64
    auto data = network->addInput("input", nvinfer1::DataType::kFLOAT, inputDims);

    // 准备LayerNorm参数
    float epsilon = 1e-5f;
    std::vector<float> gamma(64, 1.0f); // 所有元素为1的gamma
    std::vector<float> beta(64, 0.0f);  // 所有元素为0的beta

    // 创建插件工厂
    auto creator = nvinfer1::plugin::getPluginRegistry()->getPluginCreator("CustomLayerNorm", "1");
    assert(creator != nullptr);

    // 设置插件参数
    nvinfer1::PluginField gamma_field("gamma", gamma.data(), nvinfer1::PluginFieldType::kFLOAT32, 64);
    nvinfer1::PluginField beta_field("beta", beta.data(), nvinfer1::PluginFieldType::kFLOAT32, 64);
    nvinfer1::PluginField eps_field("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);

    std::vector<nvinfer1::PluginField> fields = {gamma_field, beta_field, eps_field};
    nvinfer1::PluginFieldCollection pluginFC;
    pluginFC.nbFields = fields.size();
    pluginFC.fields = fields.data();

    // 创建插件
    auto plugin = creator->createPlugin("layernorm", &pluginFC);

    // 添加插件到网络
    // 添加插件到网络
    auto pluginLayer = network->addPluginV2(&data, 1, *plugin);
    assert(pluginLayer != nullptr);

    // 设置输出
    pluginLayer->getOutput(0)->setName("output");
    network->markOutput(*pluginLayer->getOutput(0));

    // 创建优化配置
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1ULL << 20); // 1 MB

    // 构建引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // 准备输入数据
    float* h_input = new float[64];
    for (int i = 0; i < 64; i++) {
        h_input[i] = i * 0.1f; // 简单的测试数据
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, 64 * sizeof(float));
    cudaMalloc(&d_output, 64 * sizeof(float));
    cudaMemcpy(d_input, h_input, 64 * sizeof(float), cudaMemcpyHostToDevice);

    // 设置输入输出
    void* bindings[2] = {d_input, d_output};

    // 执行推理
    bool status = context->executeV2(bindings);
    assert(status);

    // 获取输出
    float* h_output = new float[64];
    cudaMemcpy(h_output, d_output, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    std::cout << "LayerNorm Results:" << std::endl;

    // 手动计算参考LayerNorm结果进行验证
    float sum = 0.0f;
    for (int i = 0; i < 64; i++) {
        sum += h_input[i];
    }
    float mean = sum / 64;

    float variance = 0.0f;
    for (int i = 0; i < 64; i++) {
        float diff = h_input[i] - mean;
        variance += diff * diff;
    }
    variance /= 64;

    float inv_std = 1.0f / sqrt(variance + epsilon);

    bool correct = true;
    for (int i = 0; i < 10; i++) { // 只打印前10个值
        float expected = (h_input[i] - mean) * inv_std * gamma[i] + beta[i];
        std::cout << "Output[" << i << "]: " << h_output[i] << ", Expected: " << expected << std::endl;

        if (fabs(h_output[i] - expected) > 1e-5) {
            correct = false;
        }
    }

    if (correct) {
        std::cout << "LayerNorm test PASSED!" << std::endl;
    } else {
        std::cout << "LayerNorm test FAILED!" << std::endl;
    }

    // 清理资源
    delete[] h_input;
    delete[] h_output;
    safeCudaFree(d_input);
    safeCudaFree(d_output);

    delete context;
    delete engine;
    delete network;
    delete config;
    delete builder;
    delete plugin;

    return 0;
}

