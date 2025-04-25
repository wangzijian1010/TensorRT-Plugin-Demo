#ifndef TRT_LAYERNORM_PLUGIN_H
#define TRT_LAYERNORM_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <memory>

namespace nvinfer1
{
    namespace plugin
    {

// LayerNorm 插件类实现 IPluginV2DynamicExt 接口
        class LayerNormPlugin : public nvinfer1::IPluginV2DynamicExt
        {
        public:

            // 添加成员变量来存储权重
            float* mGamma{nullptr};
            float* mBeta{nullptr};
            int mHiddenSize{0};
            bool mHasGammaBeta{false};
            size_t mGammaSize{0};
            size_t mBetaSize{0};

            // 构造函数，用于创建新插件实例
            LayerNormPlugin(float epsilon, const std::vector<float>& gamma, const std::vector<float>& beta);

            // 反序列化构造函数，从已保存的状态创建插件
            LayerNormPlugin(const void* data, size_t length);

            // 析构函数
            ~LayerNormPlugin() override;

            // 复制构造函数删除
            LayerNormPlugin(const LayerNormPlugin&) = delete;

            // 赋值运算符删除
            LayerNormPlugin& operator=(const LayerNormPlugin&) = delete;

            // ----------- IPluginV2 接口方法 -----------

            // 返回插件类型标识符
            const char* getPluginType() const noexcept override;

            // 返回插件版本号
            const char* getPluginVersion() const noexcept override;

            // 返回插件输出张量数量
            int getNbOutputs() const noexcept override;

            // 初始化插件资源（如分配GPU内存）
            int initialize() noexcept override;

            // 释放插件资源
            void terminate() noexcept override;

            // 返回序列化后的数据大小
            size_t getSerializationSize() const noexcept override;

            // 将插件状态序列化到缓冲区
            void serialize(void* buffer) const noexcept override;

            // 释放插件对象
            void destroy() noexcept override;

            // 设置插件命名空间
            void setPluginNamespace(const char* pluginNamespace) noexcept override;

            // 获取插件命名空间
            const char* getPluginNamespace() const noexcept override;

            // ----------- IPluginV2Ext 接口方法 -----------

            // 返回输出张量的数据类型
            nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                 int nbInputs) const noexcept override;

            // ----------- IPluginV2DynamicExt 接口方法 -----------

            // 创建插件的一个副本
            IPluginV2DynamicExt* clone() const noexcept override;

            // 确定输出张量的维度
            nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

            // 检查输入/输出张量格式组合是否受支持
            bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                           int nbInputs, int nbOutputs) noexcept override;

            // 配置插件
            void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

            // 返回执行所需的工作空间大小
            size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

            // 执行插件操作
            int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                        const void* const* inputs, void* const* outputs,
                        void* workspace, cudaStream_t stream) noexcept override;

        private:
            // 插件命名空间
            std::string mNamespace;

            // LayerNorm的epsilon参数
            float mEpsilon;

            // Layer Norm的gamma参数(权重)
            std::vector<float> mGamma;

            // Layer Norm的beta参数(偏置)
            std::vector<float> mBeta;

            // 标准化的维度大小
            int mNormalizedShape;

            // GPU上的gamma缓冲区
            void* mGammaDev;

            // GPU上的beta缓冲区
            void* mBetaDev;
        };

// 插件创建器类，实现IPluginCreator接口
        class LayerNormPluginCreator : public nvinfer1::IPluginCreator
        {
        public:
            // 构造函数
            LayerNormPluginCreator();

            // 获取插件名称
            const char* getPluginName() const noexcept override;

            // 获取插件版本
            const char* getPluginVersion() const noexcept override;

            // 获取插件字段名称集合
            const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

            // 创建新的插件实例
            IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

            // 从序列化数据反序列化创建插件
            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

            // 设置插件命名空间
            void setPluginNamespace(const char* pluginNamespace) noexcept override;

            // 获取插件命名空间
            const char* getPluginNamespace() const noexcept override;

        private:
            // 插件字段集合
            static nvinfer1::PluginFieldCollection mFC;

            // 插件属性
            static std::vector<nvinfer1::PluginField> mPluginAttributes;

            // 插件命名空间
            std::string mNamespace;
        };

    } // namespace plugin
} // namespace nvinfer1

#endif // TRT_LAYERNORM_PLUGIN_H
