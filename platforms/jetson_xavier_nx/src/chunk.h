#ifndef CHUNK_H_
#define CHUNK_H_

#include <string>
#include <NvInfer.h>

namespace nvinfer1 {

class ChunkPlugin : public IPluginV2IOExt
{
public:
    ChunkPlugin() {};
    ChunkPlugin(int32_t chunk_sizee) : chunk_size(chunk_sizee) {};
    ChunkPlugin(const void* data, size_t size);
    ~ChunkPlugin() override = default;
    
    // inherit form IPluginV2
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int32_t getNbOutputs() const override;
    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) override;
    // bool supportsFormat(DataType type, PluginFormat format) const override;
    // void configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
    //     DataType type, PluginFormat format, int32_t maxBatchSize) override;
    int32_t initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const override;
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    // IPluginV2* clone() const override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

    // inherit from IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted,
        int32_t nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const override;
    void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) override;
    void detachFromContext() override;
    IPluginV2Ext* clone() const override;
    
    void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
        int32_t nbOutput) override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
        int32_t nbOutputs) const override;
private:
    int32_t chunk_size;
    std::string plugin_namespace;
};

class ChunkPluginCreator : public IPluginCreator
{
public:
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
private:
    std::string plugin_name;
    std::string plugin_namespace;
    PluginFieldCollection plugin_field_collection;
};

REGISTER_TENSORRT_PLUGIN(ChunkPluginCreator);

}   // namespace nvinfer1

#endif