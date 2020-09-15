#include <memory>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

#include <sys/stat.h>
#include <NvOnnxParser.h>
#include <NvOnnxConfig.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntimeCommon.h>

#include "jde.h"
#include "logger.h"

namespace mot {
  
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) {
        if (obj) {
            obj->destroy();
        }
    }
};

// 兆字节转字节
constexpr long long int operator""_MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
  
JDE* JDE::me = nullptr;

JDE* JDE::instance(void)
{
    if (!me) {
        me = new JDE();
    }
    return me;
}

bool JDE::init(void)
{
    // 创建推理运行时
    runtime = nvinfer1::createInferRuntime(logger);
    
    // 创建推理引擎
    std::ifstream fin("./jde.trt", std::ios::in | std::ios::binary);
    if (fin.good()) {       // 读取序列化的引擎
        fin.seekg(0, std::ios::end);
        size_t size = fin.tellg();
        fin.seekg(0, std::ios::beg);
        std::shared_ptr<char> blob = std::shared_ptr<char>(new char[size]);
        fin.read(blob.get(), size);
        fin.close();
        engine = runtime->deserializeCudaEngine(blob.get(), size);
    } else {                // 从ONNX构建引擎
        if (!build_onnx_model()) {
            return false;
        }
    }

    // 创建执行上下文
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    
    // 分配设备端的输入和输出缓存
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
            std::cout << dims.d[j] << "X";
        }
        cudaMalloc(&buffers[i], size * sizeof(float));
        std::cout << ", ";
    }
    
    std::cout << std::endl;    
    std::cout << engine->getBindingIndex("data") << std::endl;
    std::cout << engine->getBindingIndex("out1") << std::endl;
    std::cout << engine->getBindingIndex("out2") << std::endl;
    std::cout << engine->getBindingIndex("out3") << std::endl;
    
    return true;
}

bool JDE::infer(float *in, float *out)
{
    cudaMemcpyAsync(buffers[0], in, 1*3*320*576*sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(out, buffers[1], 1*536*10*18*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return true;
}

bool JDE::destroy(void)
{
    for (int i = 0; i < 4; ++i) {
        cudaFree(buffers[i]);
    }
    
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return true;
}

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

bool JDE::build_onnx_model(void)
{    
    // 创建推理编译器
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    nvinfer1::NetworkDefinitionCreationFlags flags = explicit_batch;
    
    // 创建推理网络定义
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    if (!network) {
        return false;
    }
    
    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // 创建ONNX的解析器
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    
    // 读取ONNX模型文件    
    std::string trt_path = "./jde.trt";
    
    auto parsed = parser->parseFromFile("jde.onnx", static_cast<int>(Logger::Severity::kWARNING));
    if (!parsed) {
        return false;
    }
    
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1_MiB);
    
    // 创建推理引擎
    engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cout << "buildEngineWithConfig fail" << std::endl;
        return false;
    }
    
    if (network->getNbInputs() != 1) {
        std::cout << "getNbInputs(" << network->getNbInputs() << ") != 1" << std::endl;
        return false;
    }
    
    if (network->getNbOutputs() != 3) {
        std::cout << "getNbOutputs(" << network->getNbOutputs() << ") != 3" << std::endl;
        return false;
    }
    
    nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
    if (input_dims.nbDims != 4) {
        std::cout << "nbDims(" << input_dims.nbDims << ") != 4" << std::endl;
        return false;
    }
    
    std::cout << "Input size: ";
    for (int i = 0; i < input_dims.nbDims; ++i) {
        std::cout << input_dims.d[i] << "x";
    }
    std::cout << std::endl << "Output size: ";
    
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::Dims output_dims = network->getOutput(i)->getDimensions();
        for (int j = 0; j < output_dims.nbDims; ++j) {
            std::cout << output_dims.d[j] << "x";
        }
        std::cout << ", ";
    }
    std::cout << std::endl;
    
    // 序列化模型
    nvinfer1::IHostMemory* nstream = engine->serialize();
    std::ofstream fout(trt_path, std::ios::out | std::ios::binary);
    fout.write((const char*)nstream->data(), nstream->size());
    fout.close();
    
    return true;
}

}   // namespace mot 