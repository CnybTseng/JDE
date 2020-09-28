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

// #define PROFILE
#define USE_ONNX_PARSER     0
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
    } else {                // 从ONNX或者API构建引擎
#if USE_ONNX_PARSER
        if (!create_network_from_parser()) {
#else
        if (!create_network_from_scratch()) {
#endif
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
        binding_dims[i] = engine->getBindingDimensions(i);
        cudaMalloc(&bindings[i], binding_dims[i].numel() * sizeof(float));
        std::cout << "Binding " << i << " dimensions: " << binding_dims[i] <<
            ", " << binding_dims[i].numel() << std::endl;
    }
    
    return true;
}

bool JDE::infer(std::shared_ptr<float> in, std::vector<std::shared_ptr<float>>& out)
{
    cudaMemcpyAsync(bindings[0], in.get(), binding_dims[0].numel() * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bindings, stream, nullptr);
    for (int i = 0; i < out.size(); ++i) {
        cudaMemcpyAsync(out[i].get(), bindings[i + 1], binding_dims[i + 1].numel() * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
#ifdef PROFILE
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

DimsX JDE::get_binding_dims(int index)
{
    if (index < 0 || index >= NUM_BINDINGS) {
        std::cerr << "binding index is out of range!" << std::endl;
        return DimsX();
    }
    return binding_dims[index];
}

DimsX JDE::get_binding_dims(int index) const
{
    if (index < 0 || index >= NUM_BINDINGS) {
        std::cerr << "binding index is out of range!" << std::endl;
        return DimsX();
    }
    return binding_dims[index];
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
    nvinfer1::ILayer* layer = nullptr, *stag2 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *mpool->getOutput(0), 24, 48, 2, weights, "stage2.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.2");
    stag2 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 48, 1, weights, "stage2.3");
    std::cout << "stage2: " << stag2->getOutput(0)->getDimensions() << std::endl;
    
    // stage3
    nvinfer1::ILayer* stag3 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *stag2->getOutput(0), 48, 96, 2, weights, "stage3.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.2");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.3");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.4");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.5");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.6");
    stag3 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 48, 96, 1, weights, "stage3.7");
    std::cout << "stage3: " << stag3->getOutput(0)->getDimensions() << std::endl;
    
    // stage4
    nvinfer1::ILayer* stag4 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *stag3->getOutput(0), 96,  192, 2, weights, "stage4.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.2");
    stag4 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), 192, 192, 1, weights, "stage4.3");
    std::cout << "stage4: " << stag4->getOutput(0)->getDimensions() << std::endl;

    // stage5
    auto stag5 = addShuffleNetV2Block(network.get(), *stag4->getOutput(0), 192, 192, 1, weights, "stage5.0");
    
    // conv6
    auto conv6 = network->addConvolutionNd(*stag5->getOutput(0), 512, nvinfer1::Dims{2,{3,3},{}},
        weights["conv6.0.weight"], zero_b);
    conv6->setPaddingNd(nvinfer1::Dims2{1,1});
    auto bnom6 = addBatchnorm2d(network.get(), *conv6->getOutput(0), weights["conv6.1.weight"],
        weights["conv6.1.bias"], weights["conv6.1.running_mean"], weights["conv6.1.running_var"]);
    auto relu6 = network->addActivation(*bnom6->getOutput(0), nvinfer1::ActivationType::kRELU);
    std::cout << "conv6: " << relu6->getOutput(0)->getDimensions() << std::endl;
    
    // output1
    auto conv11 = network->addConvolutionNd(*relu6->getOutput(0),  24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv11.weight"], weights["conv11.bias"]);
    conv11->setPaddingNd(nvinfer1::Dims2{1,1});
    auto conv12 = network->addConvolutionNd(*relu6->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv12.weight"], weights["conv12.bias"]);
    conv12->setPaddingNd(nvinfer1::Dims2{1,1});
    nvinfer1::ITensor* output1_[] = {conv11->getOutput(0), conv12->getOutput(0)};
    auto output1 = network->addConcatenation(output1_, 2);
    network->markOutput(*output1->getOutput(0));
    
    // stage7
    const float scales[] = {1.f, 1.f, 2.f, 2.f};
    auto upsp1 = network->addResize(*relu6->getOutput(0));
    upsp1->setScales(scales, 4);
    nvinfer1::ITensor* inputs[] = {stag3->getOutput(0), upsp1->getOutput(0)};
    const int32_t num_inputs = 2;
    auto concat = network->addConcatenation(inputs, num_inputs);
    auto stag7 = addShuffleNetV2Block(network.get(), *concat->getOutput(0), 608, 608, 1, weights, "stage7.0");
    
    // conv8
    auto conv8 = network->addConvolutionNd(*stag7->getOutput(0), 256, nvinfer1::Dims{2,{3,3},{}},
        weights["conv8.0.weight"], zero_b);
    conv8->setPaddingNd(nvinfer1::Dims2{1,1});
    auto bnom8 = addBatchnorm2d(network.get(), *conv8->getOutput(0), weights["conv8.1.weight"],
        weights["conv8.1.bias"], weights["conv8.1.running_mean"], weights["conv8.1.running_var"]);
    auto relu8 = network->addActivation(*bnom8->getOutput(0), nvinfer1::ActivationType::kRELU);
    std::cout << "conv8: " << relu8->getOutput(0)->getDimensions() << std::endl;
    
    // output2
    auto conv13 = network->addConvolutionNd(*relu8->getOutput(0),  24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv13.weight"], weights["conv13.bias"]);
    conv13->setPaddingNd(nvinfer1::Dims2{1,1});
    auto conv14 = network->addConvolutionNd(*relu8->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv14.weight"], weights["conv14.bias"]);
    conv14->setPaddingNd(nvinfer1::Dims2{1,1});
    nvinfer1::ITensor* output2_[] = {conv13->getOutput(0), conv14->getOutput(0)};
    auto output2 = network->addConcatenation(output2_, 2);
    network->markOutput(*output2->getOutput(0));
        
    // stage9
    auto upsp2 = network->addResize(*relu8->getOutput(0));
    upsp2->setScales(scales, 4);
    inputs[0] = stag2->getOutput(0);
    inputs[1] = upsp2->getOutput(0);
    concat = network->addConcatenation(inputs, num_inputs);
    auto stag9 = addShuffleNetV2Block(network.get(), *concat->getOutput(0), 304, 304, 1, weights, "stage9.0");
    
    // conv10
    auto conv10 = network->addConvolutionNd(*stag9->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv10.0.weight"], zero_b);
    conv10->setPaddingNd(nvinfer1::Dims2{1,1});
    auto bnom10 = addBatchnorm2d(network.get(), *conv10->getOutput(0), weights["conv10.1.weight"],
        weights["conv10.1.bias"], weights["conv10.1.running_mean"], weights["conv10.1.running_var"]);
    auto relu10 = network->addActivation(*bnom10->getOutput(0), nvinfer1::ActivationType::kRELU);
    std::cout << "conv10: " << relu10->getOutput(0)->getDimensions() << std::endl;
    
    // output3
    auto conv15 = network->addConvolutionNd(*relu10->getOutput(0),  24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv15.weight"], weights["conv15.bias"]);
    conv15->setPaddingNd(nvinfer1::Dims2{1,1});
    auto conv16 = network->addConvolutionNd(*relu10->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv16.weight"], weights["conv16.bias"]);
    conv16->setPaddingNd(nvinfer1::Dims2{1,1});
    nvinfer1::ITensor* output3_[] = {conv15->getOutput(0), conv16->getOutput(0)};
    auto output3 = network->addConcatenation(output3_, 2);
    network->markOutput(*output3->getOutput(0));

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(10_MiB);
    
    engine = builder->buildEngineWithConfig(*network, *config);
    
    // 序列化模型并保存为TRT文件
    std::string trt_path = "./jde.trt";
    nvinfer1::IHostMemory* nstream = engine->serialize();
    std::ofstream fout(trt_path, std::ios::out | std::ios::binary);
    fout.write((const char*)nstream->data(), nstream->size());
    fout.close();

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
    config->setMaxWorkspaceSize(10_MiB);
    
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
    
    //! scale = gamma / sqrt(running_var)
    //! shift = beta - gamma * running_mean / sqrt(running_var)
    //!     y = (scale * x + shift)^power
    
    for (int64_t i = 0; i < weight.count; ++i) {
        scl[i] = ((float*)weight.values)[i] / sqrt(
            ((float*)running_var.values)[i] + eps);
        sht[i] = ((float*)bias.values)[i] -
            ((float*)weight.values)[i] *
            ((float*)running_mean.values)[i] / sqrt(
            ((float*)running_var.values)[i] + eps);
        pwr[i] = 1.f;
    }
    
    scale.values = scl;
    shift.values = sht;
    power.values = pwr;
    layer = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    
    //! don't free(scl)!!!
    //! don't free(sht)!!!
    //! don't free(pwr)!!!
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
        // 通道洗牌
        nvinfer1::Dims dims = input.getDimensions();
        auto shuf1 = network->addShuffle(input);
        // ABCDEF -> AB,CD,EF
        shuf1->setReshapeDimensions(nvinfer1::Dims{5,{dims.d[0],dims.d[1] >> 1,2,dims.d[2],dims.d[3]},{}});
        // AB,CD,EF -> ACE,BDF
        shuf1->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3, 4});
        auto shuf2 = network->addShuffle(*shuf1->getOutput(0));
        // ACE,BDF -> ACEBDF
        shuf2->setReshapeDimensions(dims);
        
        // 创建Chunk插件
        const int32_t num_inputs = 1;
        nvinfer1::ITensor* inputs[] = {shuf2->getOutput(0)};        
        dims = inputs[0]->getDimensions();
        int32_t chunkSize = sizeof(float) * (dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3]) >> 1;
        auto creator = getPluginRegistry()->getPluginCreator("Chunk", "100");
        nvinfer1::PluginField pluginField[1];
        pluginField[0] = nvinfer1::PluginField("chunkSize", &chunkSize, nvinfer1::PluginFieldType::kINT32);
        nvinfer1::PluginFieldCollection fc;
        fc.fields = pluginField;
        fc.nbFields = 1;
        nvinfer1::IPluginV2* plugin = creator->createPlugin("Chunk", &fc);       
        
        // 通道分组
        auto chunk = network->addPluginV2(inputs, num_inputs, *plugin);
        chunk->setName((layer_name + "chunk").c_str());
        x1 = chunk->getOutput(0);
        x2 = chunk->getOutput(1);
    }

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
    return concat;
}

}   // namespace mot 