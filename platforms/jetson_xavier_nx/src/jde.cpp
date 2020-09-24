#include <map>
#include <cmath>
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
#include "chunk.h"
#include "utils.h"
#include "logger.h"

#define PROFILE
#define INPUT_BLOB_NAME     "data"
#define INPUT_WIDTH         576
#define INPUT_HEIGHT        320

namespace mot {
  
JDE* JDE::me = nullptr;
#ifdef PROFILE
static SimpleProfiler profiler("jde");
#endif

typedef std::map<std::string, nvinfer1::Weights> wts_weights_t;

static bool load_weights(const std::string path, wts_weights_t& weights);
static nvinfer1::IScaleLayer* addBatchnorm2d(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor& input,
    nvinfer1::Weights weight,
    nvinfer1::Weights bias,
    nvinfer1::Weights running_mean,
    nvinfer1::Weights running_var,
    float eps=1e-5);
static nvinfer1::ILayer* addShuffleNetV2Block(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor& input,
    int32_t in_channels,
    int32_t out_channels,
    int32_t stride,
    wts_weights_t& weights,
    std::string layer_name);

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
        if (!create_network_from_parser()) {
            return false;
        }
    }
    create_network_from_scratch();
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
    for (int i = 0; i < 10; ++i) {
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

bool JDE::create_network_from_scratch(void)
{
    nvinfer1::DataType dt = nvinfer1::DataType::kFLOAT;
    
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
    
    // 读取模型权重
    wts_weights_t weights;
    if (!load_weights("./jde.wts", weights)) {
        return false;
    }
    
    // 添加输入层
    auto data = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims{4,{1,3,INPUT_HEIGHT,INPUT_WIDTH},{}});
    std::cout << "data: " << data->getDimensions() << std::endl;
    
    // 零偏置
    nvinfer1::Weights zero_b{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    // conv1
    auto conv1 = network->addConvolutionNd(*data, 24, nvinfer1::Dims{2,{3,3},{}}, weights["conv1.0.weight"], zero_b);
    conv1->setStrideNd(nvinfer1::Dims2{2,2});
    conv1->setPaddingNd(nvinfer1::Dims2{1,1});
    
    auto bnom1 = addBatchnorm2d(network.get(), *conv1->getOutput(0), weights["conv1.1.weight"],
        weights["conv1.1.bias"], weights["conv1.1.running_mean"], weights["conv1.1.running_var"]);

    auto relu1 = network->addActivation(*bnom1->getOutput(0), nvinfer1::ActivationType::kRELU);
    std::cout << "conv1: " << relu1->getOutput(0)->getDimensions() << std::endl;
    
    // maxpool
    auto mpool = network->addPoolingNd(*relu1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::Dims2{3,3}); 
    mpool->setStrideNd(nvinfer1::Dims2{2,2});
    mpool->setPaddingNd(nvinfer1::Dims2{1,1});
    std::cout << "maxpool: " << mpool->getOutput(0)->getDimensions() << std::endl;
    
    // stage2
    nvinfer1::ILayer* layer = nullptr;
    layer = addShuffleNetV2Block(network.get(), *mpool->getOutput(0), 24, 48, 2, weights, "stage2.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.1");
    /*layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.2");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.3");
    std::cout << "stage2: " << layer->getOutput(0)->getDimensions() << std::endl;
    
    // stage3
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 2, weights, "stage3.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.2");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.3");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.4");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.5");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.6");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.7");
    
    // stage4
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 96,  192, 2, weights, "stage4.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.2");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.3");

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1_MiB);*/
    
    // engine = builder->buildEngineWithConfig(*network, *config);

    // 释放权重空间
    for (auto& weight : weights) {
        free(const_cast<void*>(weight.second.values));
    }
    
    return true;
}

bool JDE::create_network_from_parser(void)
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

bool load_weights(const std::string path, wts_weights_t& weights)
{    
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cout << "open " << path << " fail" << std::endl;
        return false;
    }
    
    // number of weight arrays
    int32_t count = 0;
    ifs >> count;
    
    while (count--) {
        // read layer name and weight size
        std::string name;
        uint32_t size;
        ifs >> name >> std::dec >> size;
        std::cout << name << " " << size << std::endl;
        // allocate weight buffer
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, size};
        uint32_t* values = reinterpret_cast<uint32_t*>(calloc(size, sizeof(int32_t)));
        if (!values) {
            std::cout << "calloc fail" << std::endl;
            return false;
        }
        
        // read weight
        for (uint32_t i = 0; i < size; ++i) {
            ifs >> std::hex >> values[i];
        }
        
        weight.values = values;
        weights[name] = weight;
    }
    
    return true;
}

nvinfer1::IScaleLayer* addBatchnorm2d(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor& input,
    nvinfer1::Weights weight,
    nvinfer1::Weights bias,
    nvinfer1::Weights running_mean,
    nvinfer1::Weights running_var,
    float eps)
{
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::IScaleLayer* layer;
    const int32_t channel_axis = 1;
    
    float* scl = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));
    float* sht = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));
    float* pwr = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));    
    if (!scl || !sht || !pwr) {
        std::cout << "calloc fail" << std::endl;
        goto error;
    }
    
    for (int64_t i = 0; i < weight.count; ++i) {
        scl[i] = ((float*)weight.values)[i] / sqrt(
            ((float*)running_var.values)[i]);
        sht[i] = ((float*)bias.values)[i] -
            ((float*)weight.values)[i] *
            ((float*)running_mean.values)[i] / sqrt(
            ((float*)running_var.values)[i]);
        pwr[i] = 1.f;
    }
    
    scale.values = scl;    
    shift.values = sht;
    power.values = pwr;

    layer = network->addScaleNd(input, nvinfer1::ScaleMode::kCHANNEL,
        shift, scale, power, channel_axis);
    return layer;
    
    error:
    SAFETY_FREE(scl);
    SAFETY_FREE(sht);
    SAFETY_FREE(pwr);
    return nullptr;
}

nvinfer1::ILayer* addShuffleNetV2Block(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor& input,
    int32_t in_channels,
    int32_t out_channels,
    int32_t stride,
    wts_weights_t& weights,
    std::string layer_name)
{
    nvinfer1::ITensor* x1 = nullptr;    // 副分支输出
    nvinfer1::ITensor* x2 = nullptr;    // 主分支输入
    int32_t mid_channels = out_channels >> 1;
    nvinfer1::Weights zero_b{nvinfer1::DataType::kFLOAT, nullptr, 0};
    
    // 副分支的层
    if (stride > 1) {
        auto conv1 = network->addConvolutionNd(input, in_channels, nvinfer1::Dims{2,{3,3},{}},
            weights[layer_name + ".minor_branch.0.weight"], zero_b);
        conv1->setStrideNd(nvinfer1::Dims2{stride,stride});
        conv1->setPaddingNd(nvinfer1::Dims2{1,1});
        conv1->setNbGroups(in_channels);
        auto bnom1 = addBatchnorm2d(network, *conv1->getOutput(0),
            weights[layer_name + ".minor_branch.1.weight"],
            weights[layer_name + ".minor_branch.1.bias"],
            weights[layer_name + ".minor_branch.1.running_mean"],
            weights[layer_name + ".minor_branch.1.running_var"]);
        auto conv2 = network->addConvolutionNd(*bnom1->getOutput(0), mid_channels, nvinfer1::Dims{2,{1,1},{}},
            weights[layer_name + ".minor_branch.2.weight"], zero_b);
        auto bnom2 = addBatchnorm2d(network, *conv2->getOutput(0),
            weights[layer_name + ".minor_branch.3.weight"],
            weights[layer_name + ".minor_branch.3.bias"],
            weights[layer_name + ".minor_branch.3.running_mean"],
            weights[layer_name + ".minor_branch.3.running_var"]);
        auto relu1 = network->addActivation(*bnom2->getOutput(0), nvinfer1::ActivationType::kRELU);
        x1 = relu1->getOutput(0);
        x2 = &input;
    } else {
        nvinfer1::ITensor* inputs[] = {&input};
        const int32_t num_inputs = 1;
        nvinfer1::ChunkPlugin *plugin = new nvinfer1::ChunkPlugin();
        auto chunk = network->addPluginV2(inputs, num_inputs, *plugin);
        chunk->setName((layer_name + "chunk").c_str());
        x1 = chunk->getOutput(0);
        x2 = chunk->getOutput(1);
    }
    std::cout << "x1: " << x1->getDimensions() << std::endl;
    std::cout << "x2: " << x2->getDimensions() << std::endl;
    // 主分支的层
    auto conv3 = network->addConvolutionNd(*x2, mid_channels, nvinfer1::Dims{2,{1,1},{}},
        weights[layer_name + ".major_branch.0.weight"], zero_b);
    auto bnom3 = addBatchnorm2d(network, *conv3->getOutput(0),
        weights[layer_name + ".major_branch.1.weight"],
        weights[layer_name + ".major_branch.1.bias"],
        weights[layer_name + ".major_branch.1.running_mean"],
        weights[layer_name + ".major_branch.1.running_var"]);
    auto relu2 = network->addActivation(*bnom3->getOutput(0), nvinfer1::ActivationType::kRELU);
    auto conv4 = network->addConvolutionNd(*relu2->getOutput(0), mid_channels, nvinfer1::Dims{2,{3,3},{}},
        weights[layer_name + ".major_branch.3.weight"], zero_b);
    conv4->setStrideNd(nvinfer1::Dims2{stride,stride});
    conv4->setPaddingNd(nvinfer1::Dims2{1,1});
    conv4->setNbGroups(mid_channels);
    auto bnom4 = addBatchnorm2d(network, *conv4->getOutput(0),
        weights[layer_name + ".major_branch.4.weight"],
        weights[layer_name + ".major_branch.4.bias"],
        weights[layer_name + ".major_branch.4.running_mean"],
        weights[layer_name + ".major_branch.4.running_var"]);
    auto conv5 = network->addConvolutionNd(*bnom4->getOutput(0), mid_channels, nvinfer1::Dims{2,{1,1},{}},
        weights[layer_name + ".major_branch.5.weight"], zero_b);
    auto bnom5 = addBatchnorm2d(network, *conv5->getOutput(0),
        weights[layer_name + ".major_branch.6.weight"],
        weights[layer_name + ".major_branch.6.bias"],
        weights[layer_name + ".major_branch.6.running_mean"],
        weights[layer_name + ".major_branch.6.running_var"]);
    auto relu3 = network->addActivation(*bnom5->getOutput(0), nvinfer1::ActivationType::kRELU);
    
    // 合并主副分支
    nvinfer1::ITensor* inputs[] = {x1, relu3->getOutput(0)};
    const int32_t num_inputs = 2;
    auto concat = network->addConcatenation(inputs, num_inputs);
    std::cout << "concat: " << concat->getOutput(0)->getDimensions() << std::endl;
    // 通道洗牌
    nvinfer1::Dims dims = concat->getOutput(0)->getDimensions();
    auto shuf1 = network->addShuffle(*concat->getOutput(0));
    // ABCDEF -> ABC,DEF
    shuf1->setReshapeDimensions(nvinfer1::Dims{5,{dims.d[0],2,dims.d[1] >> 1,dims.d[2],dims.d[3]},{}});
    // ABC,DEF -> AD,BE,CF
    shuf1->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3, 4});
    
    std::cout << "shuf1: " << shuf1->getOutput(0)->getDimensions() << std::endl;
    auto shuf2 = network->addShuffle(*shuf1->getOutput(0));
    // AD,BE,CF -> ADBECF
    shuf2->setReshapeDimensions(dims);
    std::cout << "shuf2: " << shuf2->getOutput(0)->getDimensions() << std::endl;
    return shuf2;
}

}   // namespace mot 