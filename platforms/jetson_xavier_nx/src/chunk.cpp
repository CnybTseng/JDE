#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "chunk.h"

namespace nvinfer1 {

template <typename T>
static void write(char*& buffer, const T& data)
{
    *reinterpret_cast<T*>(buffer) = data;
    buffer += sizeof(data);
}

template <typename T>
static T read(const char*& buffer)
{
    T data = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return data;
}

// 运行时从字节流创建插件
ChunkPlugin::ChunkPlugin(const void* data, size_t size)
{
    const char* p = reinterpret_cast<const char*>(data);
    const char* b = p;
    mChunkSize = read<int32_t>(p);
    assert(p == b + size);
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
    if (cudaSuccess != cudaMemcpy(outputs[0], inputs[0], mChunkSize, cudaMemcpyDeviceToDevice)) {
        std::cout << "cudaMemcpy fail" << std::endl;
    }
    if (cudaSuccess != cudaMemcpy(outputs[1], (void*)((char*)inputs[0] + mChunkSize),
        mChunkSize, cudaMemcpyDeviceToDevice)) {
        std::cout << "cudaMemcpy fail" << std::endl;
    }
    
    return 0;
}

size_t ChunkPlugin::getSerializationSize() const
{
    return sizeof(mChunkSize);
}

void ChunkPlugin::serialize(void* buffer) const
{
    char* p = reinterpret_cast<char*>(buffer);
    char* b = p;
    write(p, mChunkSize);
    assert(p == b + getSerializationSize());
}

void ChunkPlugin::destroy()
{
    delete this;
}

void ChunkPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* ChunkPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

DataType ChunkPlugin::getOutputDataType(int32_t index, const DataType* inputTypes,
    int32_t nbInputs) const
{
    assert(0 == index || 1 == index);
    assert(1 == nbInputs);
    return inputTypes[0];
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
    return new ChunkPlugin(*this);
}

void ChunkPlugin::configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out,
    int32_t nbOutput)
{    
    mChunkSize = sizeof(float);
    for (int i = 0; i < in[0].dims.nbDims; ++i) {
        mChunkSize *= in[0].dims.d[i];
    }
    
    mChunkSize = mChunkSize >> 1;    
    if (DataType::kHALF == in[0].type) {
        mChunkSize = (mChunkSize >> 1);
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

std::vector<PluginField> ChunkPluginCreator::mPluginAttributes;
PluginFieldCollection ChunkPluginCreator::mPluginFieldCollection{};

ChunkPluginCreator::ChunkPluginCreator()
{
    mPluginAttributes.emplace_back(
        PluginField("chunkSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginFieldCollection.nbFields = mPluginAttributes.size();
    mPluginFieldCollection.fields = mPluginAttributes.data();
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
    return &mPluginFieldCollection;
}

IPluginV2IOExt* ChunkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* name = fields[i].name;
        if (!strcmp(name, "chunkSize"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mChunkSize = *static_cast<const int32_t*>(fields[i].data);
        }
    }
    
    return new ChunkPlugin(mChunkSize);
}

IPluginV2IOExt* ChunkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new ChunkPlugin(serialData, serialLength);
}

void ChunkPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mPluginNamespace = libNamespace;
}

const char* ChunkPluginCreator::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

}   // namespace nvinfer1