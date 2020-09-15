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
#include "buffer.h"

namespace mot {

// 兆字节转字节
constexpr long long int operator""_MiB(long long unsigned int val)
{
    return val * (1 << 20);
}

// 兆字节转字节
constexpr long double operator""_MiB(long double val)
{
    return val * (1 << 20);
}

// 使能DLA硬件加速
inline void enable_dla(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
    int use_dla_core, bool allow_gpu_fallback=true)
{
    if (use_dla_core >= 0) {
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Trying to use DLA core " <<
                use_dla_core << " on a platform that doesn't have any DLA cores" << std::endl;
            return;
        }
        if (allow_gpu_fallback) {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        // 没有使用INT8模式, 将运行设置成FP16模式, 禁用FP32模式
        if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
            builder->setFp16Mode(true);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(use_dla_core);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
} 

bool JDE::build()
{
    // 1. 创建运行时
    using namespace nvinfer1;
    runtime = UniquePtr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
    
    std::string trt_path = "./mot.trt";
    struct stat st;

    // 2. 创建推理引擎
    int ret = stat(trt_path.c_str(), &st);
    if (0 == ret) {     // 反序列化TRT文件
        std::ifstream fin;
        fin.open(trt_path, std::ios::in | std::ios::binary);
        fin.seekg(0, std::ios::end);
        int length = fin.tellg();
        fin.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> blob(new char[length]);
        fin.read(blob.get(), length);
        fin.close();
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(blob.get(), length), InferDeleter());
    } else {            // 从ONNX文件解析    
        // initLibNvInferPlugins(&gLogger, "");
        
        // 创建推理编译器
        auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
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
        auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        
        // 读取ONNX模型文件
        // 创建推理引擎
        auto constructed = construct_network(builder, network, config, parser);
        if (!constructed) {
            return false;
        }    
    }
    
    // 3. 创建推理执行的上下文
    context = UniquePtr<nvinfer1::IExecutionContext>(engine.get()->createExecutionContext());
    if (!context) {
        return false;
    }
    
    return true;
}

bool JDE::infer()
{
    // 创建RAII内存管理对象
    BufferManager buffers(engine, params.batch_size);
    
    // 输入图像预处理
    
    // 将数据从主机端拷贝到设备端
    buffers.copy_input_to_device();
    
    bool status = context.get()->execute(params.batch_size, buffers.get_device_bindings().data());
    if (!status) {
        return false;
    }
    
    // 将推理输出从设备端拷贝到主机端
    buffers.copy_output_to_host();
    
    // 推理结果的后处理
    
    return true;
}

// 使用ONNX模型不需要shutdownProtobufLibrary()?
bool JDE::teardown()
{
    return true;
}

// 构建卷积神经网络
bool JDE::construct_network(UniquePtr<nvinfer1::IBuilder>& builder,
    UniquePtr<nvinfer1::INetworkDefinition>& network,
    UniquePtr<nvinfer1::IBuilderConfig>& config,
    UniquePtr<nvonnxparser::IParser>& parser)
{
    std::string trt_path = "./mot.trt";
    
    auto parsed = parser->parseFromFile("jde.onnx", static_cast<int>(Logger::Severity::kWARNING));
    if (!parsed) {
        return false;
    }
    
    builder->setMaxBatchSize(params.batch_size);
    config->setMaxWorkspaceSize(1_MiB);
    
    // if (params.fp16) {
    //     config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // }
    
    // if (params.int8) {
    //     config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // }
    
    // enable_dla(builder.get(), config.get(), params.dla_core);
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());
    if (!engine) {
        std::cout << "buildEngineWithConfig fail" << std::endl;
        return false;
    }
    
    if (network->getNbInputs() != 1) {
        std::cout << "getNbInputs(" << network->getNbInputs() << ") != 1" << std::endl;
        return false;
    }
    
    input_dims = network->getInput(0)->getDimensions();
    if (input_dims.nbDims != 4) {
        std::cout << "nbDims(" << input_dims.nbDims << ") != 4" << std::endl;
        return false;
    }
    
    // 序列化模型
    nvinfer1::IHostMemory* nstream = engine->serialize();
    std::ofstream fout(trt_path, std::ios::out | std::ios::binary);
    fout.write((const char*)nstream->data(), nstream->size());
    fout.close();
    
    return true;
}

}   // namespace mot