#include "layernorm_plugin.h"
#include "layernorm_kernel.h"
#include <cstring>

using namespace nvinfer1;
namespace custom_layernorm {

// Plugin =========================================

LayerNormPlugin::LayerNormPlugin(float epsilon) : epsilon_(epsilon) {}
LayerNormPlugin::LayerNormPlugin(const void* data, size_t length) {
    const float* d = reinterpret_cast<const float*>(data);
    epsilon_ = *d;
}

const char* LayerNormPlugin::getPluginType() const noexcept { return "CustomLayerNorm"; }
const char* LayerNormPlugin::getPluginVersion() const noexcept { return "1"; }
int LayerNormPlugin::getNbOutputs() const noexcept { return 1; }
int LayerNormPlugin::initialize() noexcept { return 0; }
void LayerNormPlugin::terminate() noexcept {}
size_t LayerNormPlugin::getSerializationSize() const noexcept { return sizeof(float); }
void LayerNormPlugin::serialize(void* buffer) const noexcept {
    *reinterpret_cast<float*>(buffer) = epsilon_;
}
void LayerNormPlugin::destroy() noexcept { delete this; }
IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept {
    auto* plugin = new LayerNormPlugin(epsilon_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
void LayerNormPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* LayerNormPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType LayerNormPlugin::getOutputDataType(int, const DataType* inputTypes, int) const noexcept {
    return inputTypes[0];
}

DimsExprs LayerNormPlugin::getOutputDimensions(int, const DimsExprs* inputs, int, IExprBuilder&) noexcept {
    return inputs[0];
}

bool LayerNormPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR);
}

void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc*, int, const PluginTensorDesc*, int) const noexcept {
    return 0;
}

int LayerNormPlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    int batch_size = inputDesc[0].dims.d[0];
    int hidden_size = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    layernorm::layernormKernelLauncher(input, output, batch_size, hidden_size, epsilon_, stream);
    return 0;
}

// Creator =========================================

LayerNormPluginCreator::LayerNormPluginCreator() {
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayerNormPluginCreator::getPluginName() const noexcept { return "CustomLayerNorm"; }
const char* LayerNormPluginCreator::getPluginVersion() const noexcept { return "1"; }
const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    float epsilon = 1e-5f;
    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fc->fields[i].name, "epsilon") == 0) {
            epsilon = *(static_cast<const float*>(fc->fields[i].data));
        }
    }
    auto* plugin = new LayerNormPlugin(epsilon);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* LayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    auto* plugin = new LayerNormPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void LayerNormPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* LayerNormPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace custom_layernorm

// 注册插件
using namespace custom_layernorm;
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
