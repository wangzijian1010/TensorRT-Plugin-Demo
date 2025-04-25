//
// Created by root on 12/19/24.
//

#ifndef TENSORRTPLUGINDEMO_MYSELU_PLUGIN_H
#define TENSORRTPLUGINDEMO_MYSELU_PLUGIN_H
#include "NvInferPlugin.h"
#include "string"
#include "vector"
#include <NvInferPlugin.h> // 确保包含此头文件

// 继承 plugin 和 createor两个类
class MySELUPlugin : public  nvinfer1::IPluginV2DynamicExt{
public:
    MySELUPlugin(const std::string name, const std::string attr1, float attr3);  // 接受算子名称属性，build engine时构造函数
    MySELUPlugin(const std::string name, const void* data, size_t length);  // 接受算子名称和反序列化的engine data，推理时构造函数
    MySELUPlugin() = delete; // 析构函数

    int getNbOutputs() const noexcept override;


    virtual nvinfer1::DataType getOutputDataType(int32_t index,
                                                 nvinfer1::DataType const* inputTypes,
                                                 int32_t nbInputs) const noexcept override {
        // 返回第一个输入的类型作为输出类型
        return inputTypes[0];}

    virtual nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
                                                    const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;
    void terminate() noexcept override;

    virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                                    int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,
                                    int32_t nbOutputs) const noexcept override {
        return 0;
    };

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer)  const noexcept override;

    virtual void configurePlugin(const  nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
                                 const  nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

    virtual bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs,
                                           int32_t nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override;
    const char* getPluginNamespace()const noexcept override;


private:
    const std::string mLayerName;
    std::string mattr1;
    float mattr3;
    size_t mInputVolume;
    std::string mNamespace;
};

class MySELUPluginCreator : public nvinfer1::IPluginCreator {
public:
    MySELUPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(nvinfer1::AsciiChar const* name,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(nvinfer1::AsciiChar const* name,
                                           void const* serialData, size_t serialLength)noexcept override;
    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
private:
    static nvinfer1::PluginFieldCollection mfc;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};


#endif //TENSORRTPLUGINDEMO_MYSELU_PLUGIN_H
