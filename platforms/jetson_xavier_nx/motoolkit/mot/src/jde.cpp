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
#include "config.h"
#include "logger.h"
#include "jdecoderv2.h"

#define RUN_BACKBONE_ONLY 0
#define USE_ONNX_PARSER 0
#define USE_STANDARD_CONVOLUTION_IN_NECK 0
#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "decoder"
#define INPUT_WIDTH 576
#define INPUT_HEIGHT 320

namespace mot {
  
JDE* JDE::me = nullptr;

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
#if PROFILE_JDE
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

const void* const JDE::get_binding(int index)
{
    if (index < 0 || index >= NUM_BINDINGS) {
        std::cerr << "binding index is out of range!" << std::endl;
        return nullptr;
    }
    return bindings[index];
}

const void* const JDE::get_binding(int index) const
{
    if (index < 0 || index >= NUM_BINDINGS) {
        std::cerr << "binding index is out of range!" << std::endl;
        return nullptr;
    }
    return bindings[index];
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
    int in_channels = arch["stage2"].first;
    int out_channels = arch["stage2"].second;
    nvinfer1::ILayer* layer = nullptr, *stag2 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *mpool->getOutput(0), 24, out_channels, 2, weights, "stage2.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage2.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage2.2");
    stag2 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage2.3");
    std::cout << "stage2: " << stag2->getOutput(0)->getDimensions() << std::endl;
    
    // stage3
    in_channels = arch["stage3"].first;
    out_channels = arch["stage3"].second;
    nvinfer1::ILayer* stag3 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *stag2->getOutput(0), in_channels, out_channels, 2, weights, "stage3.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.2");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.3");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.4");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.5");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.6");
    stag3 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage3.7");
    std::cout << "stage3: " << stag3->getOutput(0)->getDimensions() << std::endl;
    
    // stage4
    in_channels = arch["stage4"].first;
    out_channels = arch["stage4"].second;
    nvinfer1::ILayer* stag4 = nullptr;
    layer = addShuffleNetV2Block(network.get(), *stag3->getOutput(0), in_channels, out_channels, 2, weights, "stage4.0");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage4.1");
    layer = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage4.2");
    stag4 = addShuffleNetV2Block(network.get(), *layer->getOutput(0), in_channels, out_channels, 1, weights, "stage4.3");
    std::cout << "stage4: " << stag4->getOutput(0)->getDimensions() << std::endl;
#if (!RUN_BACKBONE_ONLY)
    // Task head1: 192->128
    auto conv5 = network->addConvolutionNd(*stag4->getOutput(0), 128, nvinfer1::Dims{2,{1,1},{}},
        weights["conv5.0.weight"], zero_b);
    conv5->setName("conv5");
    auto bnom5 = addBatchnorm2d(network.get(), *conv5->getOutput(0), weights["conv5.1.weight"],
        weights["conv5.1.bias"], weights["conv5.1.running_mean"], weights["conv5.1.running_var"]);
    auto relu5 = network->addActivation(*bnom5->getOutput(0), nvinfer1::ActivationType::kRELU);
    std::cout << "conv5: " << relu5->getOutput(0)->getDimensions() << std::endl;
    
    nvinfer1::ILayer* shbk6 = nullptr;
    shbk6 = addShuffleNetV2Block(network.get(), *relu5->getOutput(0), 64, 128, 1, weights, "shbk6.0");
    shbk6 = addShuffleNetV2Block(network.get(), *shbk6->getOutput(0), 64, 128, 1, weights, "shbk6.1");
    shbk6 = addShuffleNetV2Block(network.get(), *shbk6->getOutput(0), 64, 128, 1, weights, "shbk6.2");
    auto conv7 = network->addConvolutionNd(*shbk6->getOutput(0), 24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv7.weight"], weights["conv7.bias"]);
    conv7->setPaddingNd(nvinfer1::Dims2{1,1});
    conv7->setName("conv7");
    
    nvinfer1::ILayer* shbk8 = nullptr;
    shbk8 = addShuffleNetV2Block(network.get(), *relu5->getOutput(0), 64, 128, 1, weights, "shbk8.0");
    shbk8 = addShuffleNetV2Block(network.get(), *shbk8->getOutput(0), 64, 128, 1, weights, "shbk8.1");
    shbk8 = addShuffleNetV2Block(network.get(), *shbk8->getOutput(0), 64, 128, 1, weights, "shbk8.2");
    auto conv9 = network->addConvolutionNd(*shbk8->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv9.weight"], weights["conv9.bias"]);
    conv9->setPaddingNd(nvinfer1::Dims2{1,1});
    conv9->setName("conv9");
    
    nvinfer1::ITensor* output1_inputs[] = {conv7->getOutput(0), conv9->getOutput(0)};
    auto output1_input = network->addConcatenation(output1_inputs, 2);
    auto output1 = network->addShuffle(*output1_input->getOutput(0));
    output1->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
#if (!INTEGRATES_DECODER)
    network->markOutput(*output1->getOutput(0));
#endif  // INTEGRATES_DECODER
    // Task head2: 128+96=224->128
    const float scales[] = {1.f, 1.f, 2.f, 2.f};
    auto upsp10 = network->addResize(*relu5->getOutput(0));
    upsp10->setScales(scales, 4);
    upsp10->setName("upsp10");
    nvinfer1::ITensor* conc10_inputs[] = {stag3->getOutput(0), upsp10->getOutput(0)};
    auto conc10 = network->addConcatenation(conc10_inputs, 2);
#if USE_STANDARD_CONVOLUTION_IN_NECK    
    auto conv10 = network->addConvolutionNd(*conc10->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv10.0.weight"], zero_b);
    conv10->setPaddingNd(nvinfer1::Dims2{1,1});
    conv10->setName("conv10");
    auto bnom10 = addBatchnorm2d(network.get(), *conv10->getOutput(0), weights["conv10.1.weight"],
        weights["conv10.1.bias"], weights["conv10.1.running_mean"], weights["conv10.1.running_var"]);
    auto relu10 = network->addActivation(*bnom10->getOutput(0), nvinfer1::ActivationType::kRELU);
#else   // USE_STANDARD_CONVOLUTION_IN_NECK
    // Depthwise
    out_channels = conc10->getOutput(0)->getDimensions().d[1];
    auto conv10_dw = network->addConvolutionNd(*conc10->getOutput(0), out_channels, nvinfer1::Dims{2,{3,3},{}},
        weights["conv10.0.weight"], zero_b);
    conv10_dw->setPaddingNd(nvinfer1::Dims2{1,1});
    conv10_dw->setNbGroups(out_channels);
    conv10_dw->setName("conv10_dw");
    auto bnom10_dw = addBatchnorm2d(network.get(), *conv10_dw->getOutput(0), weights["conv10.1.weight"],
        weights["conv10.1.bias"], weights["conv10.1.running_mean"], weights["conv10.1.running_var"]);
    // Pointwise
    auto conv10 = network->addConvolutionNd(*bnom10_dw->getOutput(0), 128, nvinfer1::Dims{2,{1,1},{}},
        weights["conv10.2.weight"], zero_b);
    conv10->setName("conv10_pw");
    auto bnom10 = addBatchnorm2d(network.get(), *conv10->getOutput(0), weights["conv10.3.weight"],
        weights["conv10.3.bias"], weights["conv10.3.running_mean"], weights["conv10.3.running_var"]);
    auto relu10 = network->addActivation(*bnom10->getOutput(0), nvinfer1::ActivationType::kRELU);
#endif  // USE_STANDARD_CONVOLUTION_IN_NECK
    std::cout << "conv10: " << relu10->getOutput(0)->getDimensions() << std::endl;
    
    nvinfer1::ILayer* shbk11 = nullptr;
    shbk11 = addShuffleNetV2Block(network.get(), *relu10->getOutput(0), 64, 128, 1, weights, "shbk11.0");
    shbk11 = addShuffleNetV2Block(network.get(), *shbk11->getOutput(0), 64, 128, 1, weights, "shbk11.1");
    shbk11 = addShuffleNetV2Block(network.get(), *shbk11->getOutput(0), 64, 128, 1, weights, "shbk11.2");
    auto conv12 = network->addConvolutionNd(*shbk11->getOutput(0), 24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv12.weight"], weights["conv12.bias"]);
    conv12->setPaddingNd(nvinfer1::Dims2{1,1});
    conv12->setName("conv12");

    nvinfer1::ILayer* shbk13 = nullptr;
    shbk13 = addShuffleNetV2Block(network.get(), *relu10->getOutput(0), 64, 128, 1, weights, "shbk13.0");
    shbk13 = addShuffleNetV2Block(network.get(), *shbk13->getOutput(0), 64, 128, 1, weights, "shbk13.1");
    shbk13 = addShuffleNetV2Block(network.get(), *shbk13->getOutput(0), 64, 128, 1, weights, "shbk13.2");
    auto conv14 = network->addConvolutionNd(*shbk13->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv14.weight"], weights["conv14.bias"]);
    conv14->setPaddingNd(nvinfer1::Dims2{1,1});
    conv14->setName("conv14");
    
    nvinfer1::ITensor* output2_inputs[] = {conv12->getOutput(0), conv14->getOutput(0)};
    auto output2_input = network->addConcatenation(output2_inputs, 2);
    auto output2 = network->addShuffle(*output2_input->getOutput(0));
    output2->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
#if (!INTEGRATES_DECODER)
    network->markOutput(*output2->getOutput(0));
#endif  // INTEGRATES_DECODER
    // Task head3: 128+48=176->128
    auto upsp15 = network->addResize(*relu10->getOutput(0));
    upsp15->setScales(scales, 4);
    upsp15->setName("upsp15");
    nvinfer1::ITensor* conc15_inputs[] = {stag2->getOutput(0), upsp15->getOutput(0)};
    auto conc15 = network->addConcatenation(conc15_inputs, 2);
#if USE_STANDARD_CONVOLUTION_IN_NECK    
    auto conv15 = network->addConvolutionNd(*conc15->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv15.0.weight"], zero_b);
    conv15->setPaddingNd(nvinfer1::Dims2{1,1});
    conv15->setName("conv15");
    auto bnom15 = addBatchnorm2d(network.get(), *conv15->getOutput(0), weights["conv15.1.weight"],
        weights["conv15.1.bias"], weights["conv15.1.running_mean"], weights["conv15.1.running_var"]);
    auto relu15 = network->addActivation(*bnom15->getOutput(0), nvinfer1::ActivationType::kRELU);
#else   // USE_STANDARD_CONVOLUTION_IN_NECK
    // Depthwise
    out_channels = conc15->getOutput(0)->getDimensions().d[1];
    auto conv15_dw = network->addConvolutionNd(*conc15->getOutput(0), out_channels, nvinfer1::Dims{2,{3,3},{}},
        weights["conv15.0.weight"], zero_b);
    conv15_dw->setPaddingNd(nvinfer1::Dims2{1,1});
    conv15_dw->setNbGroups(out_channels);
    conv15_dw->setName("conv15_dw");
    auto bnom15_dw = addBatchnorm2d(network.get(), *conv15_dw->getOutput(0), weights["conv15.1.weight"],
        weights["conv15.1.bias"], weights["conv15.1.running_mean"], weights["conv15.1.running_var"]);
    // Pointwise
    auto conv15 = network->addConvolutionNd(*bnom15_dw->getOutput(0), 128, nvinfer1::Dims{2,{1,1},{}},
        weights["conv15.2.weight"], zero_b);
    conv15->setName("conv15_pw");
    auto bnom15 = addBatchnorm2d(network.get(), *conv15->getOutput(0), weights["conv15.3.weight"],
        weights["conv15.3.bias"], weights["conv15.3.running_mean"], weights["conv15.3.running_var"]);
    auto relu15 = network->addActivation(*bnom15->getOutput(0), nvinfer1::ActivationType::kRELU);
#endif  // USE_STANDARD_CONVOLUTION_IN_NECK
    std::cout << "conv15: " << relu15->getOutput(0)->getDimensions() << std::endl;
    
    nvinfer1::ILayer* shbk16 = nullptr;
    shbk16 = addShuffleNetV2Block(network.get(), *relu15->getOutput(0), 64, 128, 1, weights, "shbk16.0");
    shbk16 = addShuffleNetV2Block(network.get(), *shbk16->getOutput(0), 64, 128, 1, weights, "shbk16.1");
    shbk16 = addShuffleNetV2Block(network.get(), *shbk16->getOutput(0), 64, 128, 1, weights, "shbk16.2");
    auto conv17 = network->addConvolutionNd(*shbk16->getOutput(0), 24, nvinfer1::Dims{2,{3,3},{}},
        weights["conv17.weight"], weights["conv17.bias"]);
    conv17->setPaddingNd(nvinfer1::Dims2{1,1});
    conv17->setName("conv17");
    
    nvinfer1::ILayer* shbk18 = nullptr;
    shbk18 = addShuffleNetV2Block(network.get(), *relu15->getOutput(0), 64, 128, 1, weights, "shbk18.0");
    shbk18 = addShuffleNetV2Block(network.get(), *shbk18->getOutput(0), 64, 128, 1, weights, "shbk18.1");
    shbk18 = addShuffleNetV2Block(network.get(), *shbk18->getOutput(0), 64, 128, 1, weights, "shbk18.2");
    auto conv19 = network->addConvolutionNd(*shbk18->getOutput(0), 128, nvinfer1::Dims{2,{3,3},{}},
        weights["conv19.weight"], weights["conv19.bias"]);
    conv19->setPaddingNd(nvinfer1::Dims2{1,1});
    conv19->setName("conv19");
    
    nvinfer1::ITensor* output3_inputs[] = {conv17->getOutput(0), conv19->getOutput(0)};
    auto output3_input = network->addConcatenation(output3_inputs, 2);
    auto output3 = network->addShuffle(*output3_input->getOutput(0));
    output3->setFirstTranspose(nvinfer1::Permutation{0, 2, 3, 1});
#if (!INTEGRATES_DECODER)
    network->markOutput(*output3->getOutput(0));
#endif  // INTEGRATES_DECODER

#if INTEGRATES_DECODER
    // Decoder
    const int32_t num_inputs = 3;
    nvinfer1::ITensor* inputs[] = {output1->getOutput(0), output2->getOutput(0), output3->getOutput(0)};
    auto creator = getPluginRegistry()->getPluginCreator("JDecoderPlugin", "100");
    const nvinfer1::PluginFieldCollection* fc = creator->getFieldNames();
    nvinfer1::IPluginV2* plugin = creator->createPlugin("JDecoderPlugin", fc);
    auto decoder = network->addPluginV2(inputs, num_inputs, *plugin);
    decoder->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decoder->getOutput(0));
#endif  // INTEGRATES_DECODER

#else   // RUN_BACKBONE_ONLY
    network->markOutput(*stag4->getOutput(0));
#endif  // RUN_BACKBONE_ONLY
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
        auto conv2 = network->addConvolutionNd(*bnom1->getOutput(0), in_channels, nvinfer1::Dims{2,{1,1},{}},
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
        chunk->setName((layer_name + ".chunk").c_str());
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
    int major_out_channels = out_channels - in_channels;
    auto conv5 = network->addConvolutionNd(*bnom4->getOutput(0), major_out_channels, nvinfer1::Dims{2,{1,1},{}},
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