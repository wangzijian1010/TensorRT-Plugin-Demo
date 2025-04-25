#include "../include/layernormPlugin.h"
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>

// 声明在CUDA文件中定义的函数
void launchLayerNormKernel(
        float* output, const float* input, const float* gamma, const float* beta,
        int batchSize, int seqLength, int hiddenSize, float epsilon, cudaStream_t stream);

namespace nvinfer1 {
    namespace plugin {

// 插件的类型名和版本号常量
        static const char* LAYERNORM_PLUGIN_VERSION{"1"};
        static const char* LAYERNORM_PLUGIN_NAME{"CustomLayerNorm"};

// 初始化静态成员
        PluginFieldCollection LayerNormPluginCreator::mFC{};
        std::vector<PluginField> LayerNormPluginCreator::mPluginAttributes;

// -------------------------- LayerNorm 插件实现 --------------------------

        LayerNormPlugin::LayerNormPlugin(float epsilon, const std::vector<float>& gamma, const std::vector<float>& beta)
                : mEpsilon(epsilon), mGamma(gamma), mBeta(beta), mGammaDev(nullptr), mBetaDev(nullptr)
        {
            mNormalizedShape = gamma.size();
        }

        LayerNormPlugin::LayerNormPlugin(const void* data, size_t length)
        {
            // 从序列化数据中恢复状态
            const char* d = static_cast<const char*>(data);
            const char* a = d;

            // 读取epsilon
            mEpsilon = *reinterpret_cast<const float*>(d);
            d += sizeof(float);

            // 读取归一化形状
            mNormalizedShape = *reinterpret_cast<const int*>(d);
            d += sizeof(int);

            // 读取gamma数组
            int gamma_size = *reinterpret_cast<const int*>(d);
            d += sizeof(int);
            mGamma.resize(gamma_size);
            memcpy(mGamma.data(), d, gamma_size * sizeof(float));
            d += gamma_size * sizeof(float);

            // 读取beta数组
            int beta_size = *reinterpret_cast<const int*>(d);
            d += sizeof(int);
            mBeta.resize(beta_size);
            memcpy(mBeta.data(), d, beta_size * sizeof(float));
            d += beta_size * sizeof(float);

            // 验证是否读取了所有数据
            assert(d == a + length);

            // 初始化GPU缓冲区为空
            mGammaDev = nullptr;
            mBetaDev = nullptr;
        }

        LayerNormPlugin::~LayerNormPlugin()
        {
            terminate();
        }

// 返回插件类型标识符
        const char* LayerNormPlugin::getPluginType() const noexcept
        {
            return LAYERNORM_PLUGIN_NAME;
        }

// 返回插件版本号
        const char* LayerNormPlugin::getPluginVersion() const noexcept
        {
            return LAYERNORM_PLUGIN_VERSION;
        }

// 返回输出张量数量
        int LayerNormPlugin::getNbOutputs() const noexcept
        {
            return 1; // LayerNorm只有一个输出
        }

// 初始化插件资源
        int LayerNormPlugin::initialize() noexcept
        {
            // 在GPU上分配并复制gamma和beta参数
            cudaMalloc(&mGammaDev, mGamma.size() * sizeof(float));
            cudaMemcpy(mGammaDev, mGamma.data(), mGamma.size() * sizeof(float), cudaMemcpyHostToDevice);

            cudaMalloc(&mBetaDev, mBeta.size() * sizeof(float));
            cudaMemcpy(mBetaDev, mBeta.data(), mBeta.size() * sizeof(float), cudaMemcpyHostToDevice);

            return 0;
        }

// 释放插件资源
        void LayerNormPlugin::terminate() noexcept
        {
            // 释放GPU内存
            if (mGammaDev)
            {
                cudaFree(mGammaDev);
                mGammaDev = nullptr;
            }

            if (mBetaDev)
            {
                cudaFree(mBetaDev);
                mBetaDev = nullptr;
            }
        }

// 返回序列化后的数据大小
        size_t LayerNormPlugin::getSerializationSize() const noexcept
        {
            return sizeof(float) + sizeof(int) +
                   sizeof(int) + mGamma.size() * sizeof(float) +
                   sizeof(int) + mBeta.size() * sizeof(float);
        }

// 序列化插件状态
        void LayerNormPlugin::serialize(void* buffer) const noexcept
        {
            char* d = static_cast<char*>(buffer);
            const char* a = d;

            // 写入epsilon
            *reinterpret_cast<float*>(d) = mEpsilon;
            d += sizeof(float);

            // 写入归一化形状
            *reinterpret_cast<int*>(d) = mNormalizedShape;
            d += sizeof(int);

            // 写入gamma数组长度和数据
            *reinterpret_cast<int*>(d) = static_cast<int>(mGamma.size());
            d += sizeof(int);
            memcpy(d, mGamma.data(), mGamma.size() * sizeof(float));
            d += mGamma.size() * sizeof(float);

            // 写入beta数组长度和数据
            *reinterpret_cast<int*>(d) = static_cast<int>(mBeta.size());
            d += sizeof(int);
            memcpy(d, mBeta.data(), mBeta.size() * sizeof(float));
            d += mBeta.size() * sizeof(float);

            assert(d == a + getSerializationSize());
        }

// 销毁插件对象
        void LayerNormPlugin::destroy() noexcept
        {
            // 释放资源并删除该对象
            terminate();
            delete this;
        }

// 设置插件命名空间
        void LayerNormPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
        {
            mNamespace = pluginNamespace;
        }

// 获取插件命名空间
        const char* LayerNormPlugin::getPluginNamespace() const noexcept
        {
            return mNamespace.c_str();
        }

// 确定输出数据类型
        DataType LayerNormPlugin::getOutputDataType(
                int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
        {
            // 输出数据类型与输入相同
            assert(index == 0);
            return inputTypes[0];
        }

// 创建插件副本
        // 创建插件副本
        IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept
        {
            LayerNormPlugin* plugin = new LayerNormPlugin(mEpsilon, mGamma, mBeta);
            plugin->setPluginNamespace(mNamespace.c_str());
            return plugin;
        }

// 确定输出张量的维度
        DimsExprs LayerNormPlugin::getOutputDimensions(
                int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
        {
            // LayerNorm 的输出维度与输入相同
            assert(outputIndex == 0);
            return inputs[0];
        }

// 检查格式组合是否受支持
        bool LayerNormPlugin::supportsFormatCombination(
                int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
        {
            // 支持 1 或 3 个输入（灵活处理）
            if (nbInputs != 1 && nbInputs != 3)
            {
                std::cerr << "LayerNormPlugin expects 1 or 3 inputs and 1 output, but got "
                          << nbInputs << " inputs and " << nbOutputs << " outputs" << std::endl;
                return false;
            }

            if (nbOutputs != 1)
            {
                std::cerr << "LayerNormPlugin expects 1 output, but got " << nbOutputs << " outputs" << std::endl;
                return false;
            }

            if (pos >= nbInputs + nbOutputs)
            {
                std::cerr << "LayerNormPlugin invalid position: " << pos
                          << ", with " << nbInputs << " inputs and " << nbOutputs << " outputs" << std::endl;
                return false;
            }

            // 所有的张量都应该是 kLINEAR 格式的 kFLOAT 类型
            bool isCorrectFormat = (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                                    inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);

            // 数据输入和输出可以是任何维度，但 gamma 和 beta 应该是 1D 的
            if (nbInputs == 3 && (pos == 1 || pos == 2)) {
                // 检查 gamma 和 beta 是否是 1D 的
                isCorrectFormat = isCorrectFormat && (inOut[pos].dims.nbDims == 1);
            }

            return isCorrectFormat;
        }



        // 配置插件
        void LayerNormPlugin::configurePlugin(
                const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
        {
            // 支持 1 或 3 个输入
            if (nbInputs != 1 && nbInputs != 3)
            {
                std::cerr << "LayerNormPlugin expects 1 or 3 inputs and 1 output" << std::endl;
                return;
            }

            if (nbOutputs != 1)
            {
                std::cerr << "LayerNormPlugin expects 1 output" << std::endl;
                return;
            }

            // 保存输入形状信息
            mInputDims = in[0].desc.dims;

            // 如果有 3 个输入，则确定是否有 gamma 和 beta
            mHasGammaBeta = (nbInputs == 3);

            if (mHasGammaBeta)
            {
                // 最后一个维度是归一化的维度
                if (mInputDims.nbDims > 0)
                {
                    mHiddenSize = mInputDims.d[mInputDims.nbDims - 1];

                    // 检查 gamma 和 beta 维度是否匹配
                    if (in[1].desc.dims.nbDims == 1 && in[2].desc.dims.nbDims == 1)
                    {
                        if (in[1].desc.dims.d[0] != mHiddenSize || in[2].desc.dims.d[0] != mHiddenSize)
                        {
                            std::cerr << "LayerNormPlugin: gamma and beta dimensions do not match hidden size" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "LayerNormPlugin: gamma and beta must be 1D tensors" << std::endl;
                    }
                }
            }
        }


// -------------------------- LayerNorm 插件创建器实现 --------------------------

        LayerNormPluginCreator::LayerNormPluginCreator()
        {
            // 注册插件参数
            mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
            mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32, 1));
            mPluginAttributes.emplace_back(PluginField("beta", nullptr, PluginFieldType::kFLOAT32, 1));

            // 设置插件字段集合
            mFC.nbFields = mPluginAttributes.size();
            mFC.fields = mPluginAttributes.data();
        }

        const char* LayerNormPluginCreator::getPluginName() const noexcept
        {
            return LAYERNORM_PLUGIN_NAME;
        }

        const char* LayerNormPluginCreator::getPluginVersion() const noexcept
        {
            return LAYERNORM_PLUGIN_VERSION;
        }

        const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept
        {
            return &mFC;
        }

        IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
        {
            float epsilon = 1e-5f;
            std::vector<float> gamma;
            std::vector<float> beta;

            // 解析传入的插件参数
            for (int i = 0; i < fc->nbFields; i++)
            {
                const char* attrName = fc->fields[i].name;
                if (!strcmp(attrName, "epsilon"))
                {
                    assert(fc->fields[i].type == PluginFieldType::kFLOAT32);
                    epsilon = *(static_cast<const float*>(fc->fields[i].data));
                }
                else if (!strcmp(attrName, "gamma"))
                {
                    assert(fc->fields[i].type == PluginFieldType::kFLOAT32);
                    int size = fc->fields[i].length;
                    gamma.resize(size);
                    const auto* gamma_data = static_cast<const float*>(fc->fields[i].data);
                    memcpy(gamma.data(), gamma_data, size * sizeof(float));
                }
                else if (!strcmp(attrName, "beta"))
                {
                    assert(fc->fields[i].type == PluginFieldType::kFLOAT32);
                    int size = fc->fields[i].length;
                    beta.resize(size);
                    const auto* beta_data = static_cast<const float*>(fc->fields[i].data);
                    memcpy(beta.data(), beta_data, size * sizeof(float));
                }
            }

            // 创建插件实例
            LayerNormPlugin* obj = new LayerNormPlugin(epsilon, gamma, beta);
            obj->setPluginNamespace(mNamespace.c_str());
            return obj;
        }

        IPluginV2* LayerNormPluginCreator::deserializePlugin(
                const char* name, const void* serialData, size_t serialLength) noexcept
        {
            // 从序列化数据创建插件
            LayerNormPlugin* obj = new LayerNormPlugin(serialData, serialLength);
            obj->setPluginNamespace(mNamespace.c_str());
            return obj;
        }

        void LayerNormPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
        {
            mNamespace = pluginNamespace;
        }

        const char* LayerNormPluginCreator::getPluginNamespace() const noexcept
        {
            return mNamespace.c_str();
        }

// 注册插件创建器
        REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

    } // namespace plugin
} // namespace nvinfer1
