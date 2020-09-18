#include <map>
#include <vector>
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
#include "utils.h"
#include "logger.h"

#define PROFILE

namespace mot {
  
JDE* JDE::me = nullptr;
#ifdef PROFILE
static SimpleProfiler profiler("jde");
#endif

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
#ifdef PROFILE
    context->setProfiler(&profiler);
#endif
    cudaStreamCreate(&stream);
    
    // 分配设备端的输入和输出缓存
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        binding_sizes[i] = size;
        cudaMalloc(&bindings[i], size * sizeof(float));
        std::cout << "Binding " << i << " dimensions: " << dims << std::endl;
    }
    
    return true;
}

bool JDE::infer(float *in, float *out)
{
#ifdef PROFILE
    for (int i = 0; i < 1000; ++i) {
#endif
        cudaMemcpyAsync(bindings[0], in, binding_sizes[0] * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueueV2(bindings, stream, nullptr);
        cudaMemcpyAsync(out, bindings[1], binding_sizes[1] * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
#ifdef PROFILE
    }
    std::cout << profiler << std::endl;
#endif
    return true;
}

bool JDE::destroy(void)
{
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        cudaFree(bindings[i]);
    }
    
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return true;
}

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

    // 创建ONNX的解析器
    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    
    // 读取ONNX模型文件    
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
    
    // 网络输入个数判断
    if (network->getNbInputs() != 1) {
        std::cout << "getNbInputs(" << network->getNbInputs() << ") != 1" << std::endl;
        return false;
    }
    
    // 网络输出个数判断
    if (network->getNbOutputs() != engine->getNbBindings() - 1) {
        std::cout << "getNbOutputs(" << network->getNbOutputs() << ") != 3" << std::endl;
        return false;
    }
    
    // 网络输入维度判断
    nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
    if (input_dims.nbDims != 4) {
        std::cout << "nbDims(" << input_dims.nbDims << ") != 4" << std::endl;
        return false;
    }
    
    // 序列化模型并保存为TRT文件
    std::string trt_path = "./jde.trt";
    nvinfer1::IHostMemory* nstream = engine->serialize();
    std::ofstream fout(trt_path, std::ios::out | std::ios::binary);
    fout.write((const char*)nstream->data(), nstream->size());
    fout.close();
    
    return true;
}

}   // namespace mot 