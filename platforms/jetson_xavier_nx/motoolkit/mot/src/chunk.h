#ifndef CHUNK_H_
#define CHUNK_H_

#include <vector>
#include <string>
#include <fstream>

#include <NvInfer.h>

namespace nvinfer1 {

class ChunkPlugin : public IPluginV2IOExt
{
public:
    ChunkPlugin() {};
    ChunkPlugin(int32_t chunkSize) : mChunkSize(chunkSize) {};
    ChunkPlugin(const void* data, size_t size);
    ~ChunkPlugin() override = default;
    
    // Inherit form IPluginV2
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int32_t getNbOutputs() const override;
    Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) override;
    int32_t initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int32_t maxBatchSize) const override;
    int32_t enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

    // Inherit from IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted,
        int32_t nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int32_t inputIndex) const override;
    void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) override;
    void detachFromContext() override;
    IPluginV2Ext* clone() const override;
    
    // Inherit from IPluginV2IOExt
    void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
        int32_t nbOutput) override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
        int32_t nbOutputs) const override;
private:
    int32_t mChunkSize;
    std::string mPluginNamespace;
};

class ChunkPluginCreator : public IPluginCreator
{
public:
    ChunkPluginCreator();
    ~ChunkPluginCreator() {};
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;
    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;
private:
    int32_t mChunkSize;
    std::string mPluginNamespace;
    static std::vector<PluginField> mPluginAttributes;
    static PluginFieldCollection mPluginFieldCollection;
};

REGISTER_TENSORRT_PLUGIN(ChunkPluginCreator);

}   // namespace nvinfer1

#endif