#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>

#include "chunk.h"

namespace nvinfer1 {

// 运行时从字节流创建插件
ChunkPlugin::ChunkPlugin(const void* data, size_t size)
{
    assert(size == sizeof(chunk_size));
    chunk_size = *reinterpret_cast<const int32_t*>(data);
}

const char* ChunkPlugin::getPluginType() const
{
    return "Chunk";
}

const char* ChunkPlugin::getPluginVersion() const
{
    return "100";
}

int32_t ChunkPlugin::getNbOutputs() const
{
    return 2;
}

Dims ChunkPlugin::getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims)
{
    assert(0 == index || 1 == index);
    assert(1 == nbInputDims);
    return Dims3(inputs[0].d[0] >> 1, inputs[0].d[1], inputs[0].d[2]);
}

// bool ChunkPlugin::supportsFormat(DataType type, PluginFormat format) const
// {
//     int device;
//     auto ret = cudaGetDevice(&device);
//     if (0 != ret) {
//         std::cout << "cudaGetDevice fail" << std::endl;
//         abort();
//     }
//     
//     cudaDeviceProp props;
//     cudaGetDeviceProperties(&props, device);
//     int sm_version = props.major << 8 | props.minor;
//     
//     return (DataType::kFLOAT == type || (DataType::kHALF == type && sm_version >= 0x600))
//         && PluginFormat::kNCHW == format;
// }

// void ChunkPlugin::configureWithFormat(const Dims* inputDims, int32_t nbInputs, const Dims* outputDims, int32_t nbOutputs,
//     DataType type, PluginFormat format, int32_t maxBatchSize)
// {
//     chunk_size = (inputDims[0].d[0] >> 1) * inputDims[0].d[1] * inputDims[0].d[2] * sizeof(float);
//     if (DataType::kHALF == type) {
//         chunk_size = (chunk_size >> 1);
//     }
// }

int32_t ChunkPlugin::initialize()
{
    return 0;
}

void ChunkPlugin::terminate()
{
}

size_t ChunkPlugin::getWorkspaceSize(int32_t maxBatchSize) const
{
    return 0;
}

int32_t ChunkPlugin::enqueue(int32_t batchSize, const void* const* inputs, void** outputs, void* workspace,
    cudaStream_t stream)
{
    if (cudaSuccess != cudaMemcpy(outputs[0], inputs[0], chunk_size, cudaMemcpyDeviceToDevice)) {
        std::cout << "cudaMemcpy fail" << std::endl;
    }
    if (cudaSuccess != cudaMemcpy(outputs[1], (void*)((char*)inputs[0] + chunk_size),
        chunk_size, cudaMemcpyDeviceToDevice)) {
        std::cout << "cudaMemcpy fail" << std::endl;
    }
    return 0;
}

size_t ChunkPlugin::getSerializationSize() const
{
    return sizeof(chunk_size);
}

void ChunkPlugin::serialize(void* buffer) const
{
    *reinterpret_cast<int32_t*>(buffer) = chunk_size;
}

void ChunkPlugin::destroy()
{
    delete this;
}

// IPluginV2* ChunkPlugin::clone() const
// {
//     auto* plugin = new ChunkPlugin(*this);
//     return plugin;
// }

void ChunkPlugin::setPluginNamespace(const char* pluginNamespace)
{
    plugin_namespace = pluginNamespace;
}

const char* ChunkPlugin::getPluginNamespace() const
{
    return plugin_namespace.data();
}

DataType ChunkPlugin::getOutputDataType(int32_t index, const DataType* inputTypes,
    int32_t nbInputs) const
{
    assert(0 == index || 1 == index);
    assert(1 == nbInputs);
    return inputTypes[index];
}

bool ChunkPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted,
    int32_t nbInputs) const
{
    return false;
}

bool ChunkPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
{
    return false;
}
void ChunkPlugin::attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*)
{
}

void ChunkPlugin::detachFromContext()
{
}

IPluginV2Ext* ChunkPlugin::clone() const
{
    auto* plugin = new ChunkPlugin(*this);
    return plugin;
}

void ChunkPlugin::configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
    int32_t nbOutput)
{
    chunk_size = (in[0].dims.d[0] >> 1) * in[0].dims.d[1] * in[0].dims.d[2] * sizeof(float);
    if (DataType::kHALF == in[0].type) {
        chunk_size = (chunk_size >> 1);
    }    
}

bool ChunkPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
    int32_t nbOutputs) const
{
    int device;
    auto ret = cudaGetDevice(&device);
    if (0 != ret) {
        std::cout << "cudaGetDevice fail" << std::endl;
        abort();
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_version = props.major << 8 | props.minor;

    return (DataType::kFLOAT == inOut[pos].type || (DataType::kHALF == inOut[pos].type && sm_version >= 0x600))
        && PluginFormat::kNCHW == inOut[pos].format;
}

const char* ChunkPluginCreator::getPluginName() const
{
    return "Chunk";
}

const char* ChunkPluginCreator::getPluginVersion() const
{
    return "100";
}

const PluginFieldCollection* ChunkPluginCreator::getFieldNames()
{
    return &plugin_field_collection;
}

IPluginV2* ChunkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    auto plugin = new ChunkPlugin();
    plugin_field_collection = *fc;
    plugin_name = name;
    return plugin;
}

IPluginV2* ChunkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    auto plugin = new ChunkPlugin(serialData, serialLength);
    plugin_name = name;
    return plugin;
}

void ChunkPluginCreator::setPluginNamespace(const char* libNamespace)
{
    plugin_namespace = libNamespace;
}

const char* ChunkPluginCreator::getPluginNamespace() const
{
    return plugin_namespace.c_str();
}












}   // namespace nvinfer1