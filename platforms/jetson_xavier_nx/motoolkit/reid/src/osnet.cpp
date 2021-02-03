#include <map>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>

#include <sys/stat.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntimeCommon.h>

#include "osnet.h"
#include "logger.h"

#define MUL_BASE 100.f  //! OSNet multiplier base.

namespace reid {

using namespace mot;
typedef std::map<std::string, nvinfer1::Weights> WeightMap;

//*******************************************************************
// Destroy weight memory.
//*******************************************************************
static void free_weight(WeightMap &weights)
{
    for (auto &weight : weights) {
        free(const_cast<void*>(weight.second.values));
    }
}

//*******************************************************************
// Load weight from local file.
//*******************************************************************
static bool load_weight(const std::string path, WeightMap &weights)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "open " << path << " fail" << std::endl;
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
        // read weight
        nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, nullptr, size};
        uint32_t *values = reinterpret_cast<uint32_t *>(calloc(size, sizeof(int32_t)));
        if (!values) {
            std::cerr << "calloc fail" << std::endl;
            free_weight(weights);
            return false;
        }
        
        for (uint32_t i = 0; i < size; ++i) {
            ifs >> std::hex >> values[i];
        }
        
        weight.values = values;
        weights[name] = weight;
    }
    
    return true;
}

//*******************************************************************
// Add BatchNorm2d (in PyTorch) layer.
//*******************************************************************
static nvinfer1::ILayer *addBatchnorm2d(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, float eps=1e-5)
{
    nvinfer1::Weights &weight = weights[name + ".weight"];
    nvinfer1::Weights &bias = weights[name + ".bias"];
    nvinfer1::Weights &running_mean = weights[name + ".running_mean"];
    nvinfer1::Weights &running_var = weights[name + ".running_var"];

    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, weight.count};
    nvinfer1::IScaleLayer* layer = nullptr;
    
    float* scl = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));
    float* sht = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));
    float* pwr = reinterpret_cast<float*>(calloc(weight.count, sizeof(float)));    
    if (!scl || !sht || !pwr) {
        std::cerr << "calloc fail" << std::endl;
        SAFETY_FREE(scl);
        SAFETY_FREE(sht);
        SAFETY_FREE(pwr);
        return layer;
    }
    
    //! scale = gamma / sqrt(running_var)
    //! shift = beta - gamma * running_mean / sqrt(running_var)
    //! power = 1
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
}

//*******************************************************************
// Add Conv2d (in PyTorch) layer.
//*******************************************************************
static nvinfer1::ILayer *addConvolution(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t nbOutputMaps,
    nvinfer1::DimsHW kernelSize, nvinfer1::DimsHW stride={1, 1},
    nvinfer1::DimsHW padding={0, 0}, bool with_bias=false,
    int32_t nbGroups=1)
{
    nvinfer1::Weights &weight = weights[name + ".weight"];
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    if (with_bias) {
        bias = weights[name + ".bias"];
    }
    auto conv = network->addConvolutionNd(input, nbOutputMaps, kernelSize, weight, bias);
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    conv->setNbGroups(nbGroups);
    return conv;
}

//*******************************************************************
// Add Conv2d and BatchNorm2d (in PyTorch) layer.
//*******************************************************************
static nvinfer1::ILayer *addConvBN(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t nbOutputMaps,
    nvinfer1::DimsHW kernelSize, nvinfer1::DimsHW stride={1, 1},
    nvinfer1::DimsHW padding={0, 0})
{
    auto conv = addConvolution(network, input, weights, name + ".conv",
        nbOutputMaps, kernelSize, stride, padding);
    auto bn = addBatchnorm2d(network, *conv->getOutput(0), weights, name + ".bn");
    return bn;
}

//*******************************************************************
// Add Conv2d, BatchNorm2d, and ReLU (in PyTorch) layer.
//*******************************************************************
static nvinfer1::ILayer *addConvBNReLU(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t nbOutputMaps,
    nvinfer1::DimsHW kernelSize, nvinfer1::DimsHW stride={1, 1},
    nvinfer1::DimsHW padding={0, 0})
{
    auto cb = addConvBN(network, input, weights, name, nbOutputMaps,
        kernelSize, stride, padding);
    auto relu = network->addActivation(*cb->getOutput(0), nvinfer1::ActivationType::kRELU);
    return relu;
}

//*******************************************************************
// Add any type of Pooling layer.
//*******************************************************************
static nvinfer1::ILayer *addPooling(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    nvinfer1::DimsHW windowSize, nvinfer1::PoolingType type=nvinfer1::PoolingType::kMAX,
    nvinfer1::DimsHW stride={1, 1}, nvinfer1::DimsHW padding={0, 0})
{
    auto pool = network->addPoolingNd(input, type, windowSize); 
    pool->setStrideNd(stride);
    pool->setPaddingNd(padding);
    return pool;
}

//*******************************************************************
// Add Lite3x3 (in OSNet) layer.
//*******************************************************************
static nvinfer1::ILayer *addLiteConv(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t nbOutputMaps)
{
    // conv1x1
    auto conv1 = addConvolution(network, input, weights, name + ".conv1",
        nbOutputMaps, nvinfer1::DimsHW{1, 1});
    // dw-conv3x3
    auto conv2 = addConvolution(network, *conv1->getOutput(0), weights,
        name + ".conv2", nbOutputMaps, nvinfer1::DimsHW{3, 3},
        nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{1, 1}, false,
        nbOutputMaps);
    auto bn = addBatchnorm2d(network, *conv2->getOutput(0), weights, name + ".bn");
    auto relu = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kRELU);
    return relu;
}

//*******************************************************************
// Add unified aggregation gate (in OSNet) layer.
//*******************************************************************
static nvinfer1::ILayer *addChannelGate(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t reduction=16)
{
    // adaptive average pooling
    nvinfer1::Dims dims = input.getDimensions();    //! CHW instead of NCHW
    auto aap = addPooling(network, input, nvinfer1::DimsHW{dims.d[1], dims.d[2]},
        nvinfer1::PoolingType::kAVERAGE);
    auto fc1 = addConvolution(network, *aap->getOutput(0), weights, name + ".fc1",
        dims.d[0] / reduction, nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{1, 1},
        nvinfer1::DimsHW{0, 0}, true);
    auto relu = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU);
    auto fc2 = addConvolution(network, *relu->getOutput(0), weights, name + ".fc2",
        dims.d[0], nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{1, 1},
        nvinfer1::DimsHW{0, 0}, true);
    auto sigmoid = network->addActivation(*fc2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    auto prod = network->addElementWise(input, *sigmoid->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD);
    return prod;
}

//*******************************************************************
// Add bottleneck (in OSNet) layer.
//*******************************************************************
static nvinfer1::ILayer *addOSBlock(
    nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input,
    WeightMap &weights, std::string name, int32_t nbOutputMaps,
    int32_t reduction=4)
{
    // conv1
    const int32_t nbInputMaps = input.getDimensions().d[0];
    const int32_t mid_channels = nbOutputMaps / reduction;
    auto conv1 = addConvBNReLU(network, input, weights, name + ".conv1",
        mid_channels, nvinfer1::DimsHW{1, 1});
    // conv2a
    auto conv2a0 = addLiteConv(network, *conv1->getOutput(0), weights,
        name + ".conv2a", mid_channels);
    // conv2b
    auto conv2b0 = addLiteConv(network, *conv1->getOutput(0), weights,
        name + ".conv2b.0", mid_channels);
    auto conv2b1 = addLiteConv(network, *conv2b0->getOutput(0), weights,
        name + ".conv2b.1", mid_channels);     
    // conv2c
    auto conv2c0 = addLiteConv(network, *conv1->getOutput(0), weights,
        name + ".conv2c.0", mid_channels);
    auto conv2c1 = addLiteConv(network, *conv2c0->getOutput(0), weights,
        name + ".conv2c.1", mid_channels);
    auto conv2c2 = addLiteConv(network, *conv2c1->getOutput(0), weights,
        name + ".conv2c.2", mid_channels);   
    // conv2d
    auto conv2d0 = addLiteConv(network, *conv1->getOutput(0), weights,
        name + ".conv2d.0", mid_channels);
    auto conv2d1 = addLiteConv(network, *conv2d0->getOutput(0), weights,
        name + ".conv2d.1", mid_channels);
    auto conv2d2 = addLiteConv(network, *conv2d1->getOutput(0), weights,
        name + ".conv2d.2", mid_channels);
    auto conv2d3 = addLiteConv(network, *conv2d2->getOutput(0), weights,
        name + ".conv2d.3", mid_channels);
    // gate
    auto gate2a = addChannelGate(network, *conv2a0->getOutput(0), weights, name + ".gate"); 
    auto gate2b = addChannelGate(network, *conv2b1->getOutput(0), weights, name + ".gate");
    auto gate2c = addChannelGate(network, *conv2c2->getOutput(0), weights, name + ".gate");
    auto gate2d = addChannelGate(network, *conv2d3->getOutput(0), weights, name + ".gate");
    auto gateab = network->addElementWise(*gate2a->getOutput(0), *gate2b->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    auto gatecd = network->addElementWise(*gate2c->getOutput(0), *gate2d->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    auto gate   = network->addElementWise(*gateab->getOutput(0), *gatecd->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    // conv3
    auto conv3 = addConvBN(network, *gate->getOutput(0), weights, name + ".conv3",
        nbOutputMaps, nvinfer1::DimsHW{1, 1});
    // residual + sum
    nvinfer1::ITensor *residual;
    if (nbInputMaps != nbOutputMaps) {
        auto ds = addConvBN(network, input, weights, name + ".downsample",
            nbOutputMaps, nvinfer1::DimsHW{1, 1});
        residual = ds->getOutput(0);
    } else {
        residual = &input;
    }
    auto sum = network->addElementWise(*residual, *conv3->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM);
    // relu
    auto relu = network->addActivation(*sum->getOutput(0), nvinfer1::ActivationType::kRELU);
    return relu;
}

/**
 * @brief Actual implementor for OSNet.
 */
class OSNet::Impl
{
public:
    Impl();
public:
    bool init(std::string engine_path, std::string weight_path="",
        int beta=100, int gamma=100);
    bool deinit(void);
    bool forward(std::shared_ptr<float> in, std::shared_ptr<float> &out,
        int batch_size);
    mot::DimsX get_input_dim() const;
    mot::DimsX get_output_dim() const;
    int get_max_batch_size() const;  
    bool build_engine(WeightMap weights);
    nvinfer1::INetworkDefinition* build_network(nvinfer1::IBuilder* builder,
        WeightMap weights);
    const std::string in_blob_name; //! The input blob name.
    const std::string out_blob_name;    //! The output blob name.
    int in_width;   //! OSNet input width.
    int in_height;  //! OSNet input height.
    const int max_batch_size;   //! The maximum batch size supported.
    std::vector<int> chan;  //! OSNet width configuration.
    const std::vector<int> multipliers; //! OSNet width multiplier options.
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    std::map<std::string, mot::DimsX> blob_dims;    //! Input and output blob dimensions.
    std::map<std::string, void *> bindings; //! Input and output bindings.
};

//*******************************************************************
// OSNet::Impl constructor.
//*******************************************************************
OSNet::Impl::Impl() :
    in_blob_name("input"),
    out_blob_name("output"),
    in_width(128),
    in_height(256),
    max_batch_size(16),
    chan{64, 256, 384, 512},
    multipliers{25, 50, 75, 100}
{
}

//*******************************************************************
// Actual executor for OSNet::init().
//*******************************************************************
bool OSNet::Impl::init(std::string engine_path, std::string weight_path,
    int beta, int gamma)
{
    //! beta and gamma multiplier
    if (multipliers.end() == find(multipliers.begin(),
        multipliers.end(), beta) ||
        multipliers.end() == find(multipliers.begin(),
        multipliers.end(), gamma)) {
        std::cerr << "unsupported nerual network multiplier" << std::endl;
        return false;
    }
    
    in_width = static_cast<int>(gamma * in_width / MUL_BASE);
    in_height = static_cast<int>(gamma * in_height / MUL_BASE);
    for (size_t i = 0; i < chan.size(); ++i) {
        chan[i] = static_cast<int>(beta * chan[i] / MUL_BASE);
    }
    
    runtime = nvinfer1::createInferRuntime(mot::logger);
    std::ifstream ifs(engine_path, std::ios::in | std::ios::binary);
    if (ifs.good()) {
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        std::shared_ptr<char> blob(new char[size]);
        ifs.read(blob.get(), size);
        ifs.close();
        engine = runtime->deserializeCudaEngine(blob.get(), size);
    } else {
        WeightMap weights;
        if (!load_weight(weight_path, weights)) {
            return false;
        }
        if (!build_engine(weights)) {
            return false;
        }
        nvinfer1::IHostMemory* hmem = engine->serialize();
        std::ofstream ofs(engine_path, std::ios::out | std::ios::binary);
        ofs.write((const char*)hmem->data(), hmem->size());
        ofs.close();
        free_weight(weights);   //! free weight buffer after engine->serialize!!!
    }

    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    
    const int32_t in_idx = engine->getBindingIndex(in_blob_name.c_str());
    const int32_t out_idx = engine->getBindingIndex(out_blob_name.c_str());
    blob_dims[in_blob_name] = engine->getBindingDimensions(in_idx);
    blob_dims[out_blob_name] = engine->getBindingDimensions(out_idx);
    
    cudaError_t err;
    bindings[in_blob_name] = nullptr;
    bindings[out_blob_name] = nullptr;
    if (cudaSuccess != cudaMalloc(&bindings[in_blob_name],
        max_batch_size * blob_dims[in_blob_name].numel() * sizeof(float)) ||
        cudaSuccess != cudaMalloc(&bindings[out_blob_name],
        max_batch_size * blob_dims[out_blob_name].numel() * sizeof(float))) {
        std::cerr << "cudaMalloc fail" << std::endl;
        deinit();
        return false;
    }
    
    return true;
}

//*******************************************************************
// Actual executor for OSNet::deinit().
//*******************************************************************
bool OSNet::Impl::deinit(void)
{
    for (auto &binding : bindings) {
        if (binding.second) {
            cudaFree(binding.second);
        }
    }
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return true;
}

//*******************************************************************
// Actual executor for OSNet::forward().
//*******************************************************************
bool OSNet::Impl::forward(std::shared_ptr<float> in, std::shared_ptr<float> &out,
    int batch_size)
{
    if (batch_size > max_batch_size) {
        std::cerr << "batch_size must be small than " << max_batch_size << std::endl;
        return false;
    }
    
    if (cudaSuccess != cudaMemcpyAsync(bindings[in_blob_name], in.get(),
        batch_size * blob_dims[in_blob_name].numel() * sizeof(float),
        cudaMemcpyHostToDevice, stream)) {
        std::cerr << "cudaMemcpyAsync fail" << std::endl;
        return false;
    }

    void *binding_arr[] = {bindings[in_blob_name], bindings[out_blob_name]};
    if (!context->enqueue(batch_size, binding_arr, stream, nullptr)) {
        std::cerr << "enqueue fail" << std::endl;
        return false;
    }
    
    if (cudaSuccess != cudaMemcpyAsync(out.get(), bindings[out_blob_name],
        batch_size * blob_dims[out_blob_name].numel() * sizeof(float),
        cudaMemcpyDeviceToHost, stream)) {
        std::cerr << "cudaMemcpyAsync fail" << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);
    return true;
}

//*******************************************************************
// Actual executor for OSNet::get_input_dim().
//*******************************************************************
mot::DimsX OSNet::Impl::get_input_dim() const
{
    //! using [] operator will throw error
    return blob_dims.at(in_blob_name);
}

//*******************************************************************
// Actual executor for OSNet::get_output_dim().
//*******************************************************************
mot::DimsX OSNet::Impl::get_output_dim() const
{
    return blob_dims.at(out_blob_name);
}

//*******************************************************************
// Actual executor for OSNet::get_max_batch_size().
//*******************************************************************
int OSNet::Impl::get_max_batch_size() const
{
    return max_batch_size;
}

//*******************************************************************
// Build inference engine from weights.
//*******************************************************************
bool OSNet::Impl::build_engine(WeightMap weights)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(mot::logger);    
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = build_network(builder, weights);
    builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(100_MiB);
#if defined(USE_FP16) || defined(USE_INT8)   // Use FP16 or FP32 (rollback).
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        std::cerr << "The platform do not support FP16, "
            "rollback to FP32" << std::endl;
    }
#ifdef USE_INT8 // Use INT8 or FP16 (rollback) or FP32 (rollback), almost optimal.
    if (builder->platformHasFastInt8()) {
        std::cerr << "INT8 calibrator has not been implemented yet, "
            "rollback to FP32 or FP16" << std::endl;
    } else {
        std::cerr << "The platform do not support INT8, "
            "rollback to FP32 or FP16" << std::endl;
    }
#endif
#endif
    engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
    config->destroy();
    builder->destroy();   
    return true;
}

//*******************************************************************
// Build network from weights.
//*******************************************************************
nvinfer1::INetworkDefinition* OSNet::Impl::build_network(nvinfer1::IBuilder* builder,
    WeightMap weights)
{
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
    // input
    auto data = network->addInput(in_blob_name.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims3{3, in_height, in_width});    
    // conv1
    auto conv1 = addConvBNReLU(network, *data, weights, "conv1", chan[0],
        nvinfer1::DimsHW{7, 7}, nvinfer1::DimsHW{2, 2}, nvinfer1::DimsHW{3, 3});
    auto maxpool = addPooling(network, *conv1->getOutput(0), nvinfer1::DimsHW{3, 3},
        nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2}, nvinfer1::DimsHW{1, 1});
    // conv2 + transition
    auto conv20 = addOSBlock(network, *maxpool->getOutput(0), weights, "conv2.0", chan[1]);
    auto conv21 = addOSBlock(network, *conv20->getOutput(0), weights, "conv2.1", chan[1]);
    auto tran20 = addConvBNReLU(network, *conv21->getOutput(0), weights, "conv2.2.0",
        chan[1], nvinfer1::DimsHW{1, 1});
    auto tran21 = addPooling(network, *tran20->getOutput(0), nvinfer1::DimsHW{2, 2},
        nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{2, 2});
    // conv3 + transition
    auto conv30 = addOSBlock(network, *tran21->getOutput(0), weights, "conv3.0", chan[2]);
    auto conv31 = addOSBlock(network, *conv30->getOutput(0), weights, "conv3.1", chan[2]);
    auto tran30 = addConvBNReLU(network, *conv31->getOutput(0), weights, "conv3.2.0",
        chan[2], nvinfer1::DimsHW{1, 1});
    auto tran31 = addPooling(network, *tran30->getOutput(0), nvinfer1::DimsHW{2, 2},
        nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{2, 2});
    // conv4
    auto conv40 = addOSBlock(network, *tran31->getOutput(0), weights, "conv4.0", chan[3]);
    auto conv41 = addOSBlock(network, *conv40->getOutput(0), weights, "conv4.1", chan[3]);
    // conv5
    auto conv5 = addConvBNReLU(network, *conv41->getOutput(0), weights, "conv5", chan[3],
        nvinfer1::DimsHW{1, 1});
    // gap
    nvinfer1::Dims dims = conv5->getOutput(0)->getDimensions(); //! CHW instead of NCHW
    auto gap = addPooling(network, *conv5->getOutput(0), nvinfer1::DimsHW{dims.d[1],
        dims.d[2]}, nvinfer1::PoolingType::kAVERAGE);
    // fc
    auto fc = network->addFullyConnected(*gap->getOutput(0), chan[3],
        weights["fc.0.weight"], weights["fc.0.bias"]);
    auto fb = addBatchnorm2d(network, *fc->getOutput(0), weights, "fc.1");
    auto fba = network->addActivation(*fb->getOutput(0), nvinfer1::ActivationType::kRELU);
    fba->getOutput(0)->setName(out_blob_name.c_str());
    network->markOutput(*fba->getOutput(0));
    return network;
}

//*******************************************************************
// OSNet constructor.
//*******************************************************************
OSNet::OSNet() : impl(new Impl)
{
}

//*******************************************************************
// OSNet destructor. 
// Must be defined out of line in the implementation file.
//*******************************************************************
OSNet::~OSNet()
{
}

//*******************************************************************
// OSNet initialization.
//*******************************************************************
bool OSNet::init(std::string engine_path, std::string weight_path,
    int beta, int gamma)
{
    return impl->init(engine_path, weight_path, beta, gamma);
}

//*******************************************************************
// OSNet deinitialization.
//*******************************************************************
bool OSNet::deinit(void)
{
    return impl->deinit();
}

//*******************************************************************
// OSNet forward.
//*******************************************************************
bool OSNet::forward(std::shared_ptr<float> in, std::shared_ptr<float> &out,
    int batch_size)
{
    return impl->forward(in, out, batch_size);
}

//*******************************************************************
// Get input dimension.
//*******************************************************************
mot::DimsX OSNet::get_input_dim() const
{
    return impl->get_input_dim();
}

//*******************************************************************
// Get output dimension.
//*******************************************************************
mot::DimsX OSNet::get_output_dim() const
{
    return impl->get_output_dim();
}

//*******************************************************************
// Get maximum batch size supported.
//*******************************************************************
int OSNet::get_max_batch_size() const
{
    return impl->get_max_batch_size();
}

}   // namespace reid