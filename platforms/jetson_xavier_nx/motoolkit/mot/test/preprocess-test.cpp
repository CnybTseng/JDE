#include <chrono>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "jde.h"
#include "utils.h"

#define INPUT_BLOB_NAME         "data"
#define INPUT_WIDTH             1920
#define INPUT_HEIGHT            1080
#define NETWORK_INPUT_WIDTH     576
#define NETWORK_INPUT_HEIGHT    320

using namespace mot;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} logger;

int main(int argc, char* argv[])
{
    cv::Mat im = cv::imread(argv[1]);
    if (im.empty()) {
        fprintf(stderr, "imread %s fail.\n", argv[1]);
        return -1;
    }
    
    int method = 1;
    if (argc > 2) {
        method = atoi(argv[2]);
    }
    
    // OpenCV Implementation.
    if (0 == method)
    {
        std::shared_ptr<unsigned char> resized_im_data(
            new unsigned char[NETWORK_INPUT_WIDTH * NETWORK_INPUT_HEIGHT * 3]);
        cv::Mat resized_im(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_8UC3, resized_im_data.get());
        
        std::shared_ptr<unsigned char> resized_im_chw_data(
            new unsigned char[NETWORK_INPUT_WIDTH * NETWORK_INPUT_HEIGHT * 3]);
        const int stride = NETWORK_INPUT_WIDTH * NETWORK_INPUT_HEIGHT;
        std::vector<cv::Mat> resized_im_chw_vec = {
            cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_8UC1, resized_im_chw_data.get()),
            cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_8UC1, resized_im_chw_data.get() + stride),
            cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_8UC1, resized_im_chw_data.get() + (stride << 1))
        };
        cv::Mat resized_im_chw(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_HEIGHT, CV_8UC3, resized_im_chw_data.get());
        
        std::shared_ptr<float> result_im_data(
            new float[NETWORK_INPUT_WIDTH * NETWORK_INPUT_HEIGHT * 3]);
        cv::Mat result_im(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_32FC3, result_im_data.get());
        
        const int loops = 1000;
        mot::SimpleProfiler profiler("preprocess-opencv");
        for (int i = 0; i < loops; ++i) {
            auto start_resize = std::chrono::high_resolution_clock::now();
            cv::resize(im, resized_im, resized_im.size());
            profiler.reportLayerTime("resize", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_resize).count());
            
            auto start_shuffle = std::chrono::high_resolution_clock::now();
            cv::split(resized_im, resized_im_chw_vec);
            profiler.reportLayerTime("shuffle", std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - start_shuffle).count());
            
            auto start_scale = std::chrono::high_resolution_clock::now();
            resized_im_chw.convertTo(result_im, CV_32FC3, 1.f / 255);
            profiler.reportLayerTime("scale", std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - start_scale).count());
        }
        std::cout << profiler << std::endl;
        
        return 0;
    }
    
    auto runtime = nvinfer1::createInferRuntime(logger);
    auto builder = nvinfer1::createInferBuilder(logger);
    
    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);    
    nvinfer1::NetworkDefinitionCreationFlags flags = explicit_batch;
    auto network = builder->createNetworkV2(flags);
    
    // The memory layout of input data is NHWC.
    auto data = network->addInput(INPUT_BLOB_NAME, nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims{4, {1, im.rows, im.cols, 3}, {}});
    
    // Resize.
    auto resize_layer = network->addResize(*data);
    resize_layer->setOutputDimensions(nvinfer1::Dims{4, {1, NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, 3}, {}});
    resize_layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
    
    // Normalize.
    float _scale = 1. / 255;
    float _shift = 0.f;
    float _power = 1.f;
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, &_scale, 1};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, &_shift, 1};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, &_power, 1};
    auto scale_layer = network->addScale(*resize_layer->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, shift, scale, power);
    
    // NHWC to NCHW.
    auto shuffle_layer = network->addShuffle(*scale_layer->getOutput(0));
    shuffle_layer->setSecondTranspose(nvinfer1::Permutation{0, 3, 1, 2});
    network->markOutput(*shuffle_layer->getOutput(0));
    
    auto config = builder->createBuilderConfig();
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1_MiB);
    
    auto engine = builder->buildEngineWithConfig(*network, *config);
    auto context = engine->createExecutionContext();
    
    mot::SimpleProfiler net_profiler("preprocess in network");
    context->setProfiler(&net_profiler);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    void* bindings[2];
    DimsX binding_dims[2];
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        binding_dims[i] = engine->getBindingDimensions(i);
        cudaMalloc(&bindings[i], binding_dims[i].numel() * sizeof(float));
        std::cout << "Binding " << i << " dimensions: " << binding_dims[i] <<
            ", " << binding_dims[i].numel() << std::endl;
    }
    
    void* im_f32_data;
    cudaHostAlloc(&im_f32_data, binding_dims[0].numel() * sizeof(float), cudaHostAllocWriteCombined);
    
    cv::Mat im_f32(im.rows, im.cols, CV_32FC3, im_f32_data);
    im.convertTo(im_f32, CV_32FC3);
    
    std::shared_ptr<float> buffer(new float[binding_dims[1].numel()]);
    const int stride = NETWORK_INPUT_HEIGHT * NETWORK_INPUT_WIDTH * sizeof(float);
    std::vector<cv::Mat> resized_im_f32_vec = {
        cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_32FC1, (char*)buffer.get()),
        cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_32FC1, (char*)buffer.get() + stride),
        cv::Mat(NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH, CV_32FC1, (char*)buffer.get() + (stride << 1))
    };
    
    const int loops = 1000;
    for (int i = 0; i < loops; ++i) {
        auto start_type = std::chrono::high_resolution_clock::now();
        im.convertTo(im_f32, CV_32FC3);
        net_profiler.reportLayerTime("convertTo", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_type).count());

        auto start_copy = std::chrono::high_resolution_clock::now();
        cudaMemcpy(bindings[0], im_f32_data, binding_dims[0].numel() * sizeof(float), cudaMemcpyHostToDevice);
        net_profiler.reportLayerTime("cudaMemcpy", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_copy).count());
        
        context->enqueueV2(bindings, stream, nullptr);
        cudaStreamSynchronize(stream);
    }
    cudaMemcpy(buffer.get(), bindings[1], binding_dims[1].numel() * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << net_profiler << std::endl;
    
    cv::Mat resized_im_f32;
    cv::merge(resized_im_f32_vec, resized_im_f32);
    
    cv::Mat resized_im;
    resized_im_f32.convertTo(resized_im, CV_8UC3, 255.f);
    cv::imwrite("resized.jpg", resized_im);
    
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        cudaFree(bindings[i]);
    }
    
    cudaFreeHost(im_f32_data);
    cudaStreamDestroy(stream);
    
    context->destroy();
    engine->destroy();
    runtime->destroy();
    
    return 0;
}