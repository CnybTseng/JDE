#include <bitset>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "jdecoderv2.h"

struct Detection {
    float category;
    float score;
    struct {
        float top;
        float left;
        float bottom;
        float right;
    } bbox;
    float embedding[nvinfer1::jdec::embeDim];
} __attribute__((packed));

static inline float CalcInterArea(const Detection& a, const Detection& b)
{
    if (a.bbox.top > b.bbox.bottom || a.bbox.bottom < b.bbox.top ||
        a.bbox.left > b.bbox.right || a.bbox.right < b.bbox.left)
        return 0.f;
    
    float w = std::min(a.bbox.right, b.bbox.right) - std::max(a.bbox.left, b.bbox.left);
    float h = std::min(a.bbox.bottom, b.bbox.bottom) - std::max(a.bbox.top, b.bbox.top);
    return w * h;
}

static void QsortDescentInplace(std::vector<Detection>& data, int left, int right)
{
    int i = left;
    int j = right;
    float pivot = data[(left + right) >> 1].score;
    while (i <= j)
    {
        while (data[i].score > pivot)
            ++i;
        
        while (data[j].score < pivot)
            --j;
        
        if (i <= j)
        {
            std::swap(data[i], data[j]);
            ++i;
            --j;
        }
    }
    
    if (left <= j)
        QsortDescentInplace(data, left, j);
    
    if (right >= i)
        QsortDescentInplace(data, i, right);
}

static void QsortDescentInplace(std::vector<Detection>& data)
{
    if (data.empty())
        return;
    
    QsortDescentInplace(data, 0, static_cast<int>(data.size() - 1));
}

static void NonmaximumSuppression(const std::vector<Detection>& dets, std::vector<size_t>& keeps, float iou_thresh)
{
    keeps.clear();
    const size_t n = dets.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
    {
        float w = dets[i].bbox.right - dets[i].bbox.left;
        float h = dets[i].bbox.bottom - dets[i].bbox.top;
        areas[i] = w * h;
    }
    
    for (size_t i = 0; i < n; ++i)
    {
        const Detection& deti = dets[i];
        int flag = 1;
        for (size_t j = 0; j < keeps.size(); ++j)
        {
            const Detection& detj = dets[keeps[j]];
            float inters = CalcInterArea(deti, detj);
            float unionn = areas[i] + areas[keeps[j]] - inters;
            if (inters / unionn > iou_thresh)
            {
                flag = 0;
                break;
            }
        }
        
        if (flag)
            keeps.push_back(i);
    }
}

int main(int argc, char *argv[])
{
    int loops = 1;
    if (argc > 1) {
        loops = atoi(argv[1]);
    }
    
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "num_devices: " << num_devices << std::endl;
    std::cout << "compute_capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    
    float* dets_gpu;
    // Ensure that the data is 8 bytes alignment for float2 SIMD computation.
    // The first 4 bytes store the number of valid output boxes, and the second
    // 4 bytes are not used, just for data alignment.
    size_t numel = nvinfer1::jdec::maxNumOutputBox * nvinfer1::jdec::decOutputDim + 2;
    cudaMalloc((void**)&dets_gpu, numel * sizeof(float));        
    std::shared_ptr<float> dets_cpu(new float[numel]);
    std::vector<Detection> dets(nvinfer1::jdec::maxNumOutputBox);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // out0.bin, out1.bin, and out2.bin are coming from mot-test app.
    void* in_gpu[3];
    constexpr int num_outs = 3;
    std::vector<size_t> size(num_outs);
    std::vector<std::shared_ptr<char>> in_cpu(num_outs);
    for (int i = 0; i < num_outs; ++i) {
        std::ifstream ifs("out" + std::to_string(i) + ".bin", std::ios::binary);
        ifs.seekg(0, std::ios::end);
        size[i] = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        in_cpu[i] = std::shared_ptr<char>(new char[size[i]]);
        ifs.read(in_cpu[i].get(), size[i]);
        ifs.close();
        cudaMalloc((void**)&in_gpu[i], size[i]); 
    }
    
    float iou_thresh = 0.4f;
    std::vector<Detection> nms_dets;
    nvinfer1::JDecoderV2 jdecoderv2;
    for (int i = 0; i < loops; ++i) {
        for (int j = 0; j < num_outs; ++j) {
            cudaMemcpy(in_gpu[j], in_cpu[j].get(), size[j], cudaMemcpyHostToDevice);
        }
        
        mot::SimpleProfiler profiler("jdecoderv2");
        auto start_test = std::chrono::high_resolution_clock::now();
        jdecoderv2.forward_test((const float* const*)in_gpu, dets_gpu, stream);
        cudaMemcpyAsync(dets_cpu.get(), dets_gpu, numel * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaError_t code = cudaStreamSynchronize(stream);
        profiler.reportLayerTime("forward_test", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_test).count());
        
        auto start_qsort = std::chrono::high_resolution_clock::now();
        memcpy(dets.data(), &dets_cpu.get()[2], dets_cpu.get()[0] * nvinfer1::jdec::decOutputDim * sizeof(float));
        QsortDescentInplace(dets);
        profiler.reportLayerTime("QsortDescentInplace", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_qsort).count());
        
        auto start_nms = std::chrono::high_resolution_clock::now();
        std::vector<size_t> keeps;
        NonmaximumSuppression(dets, keeps, iou_thresh);
        int num_dets = static_cast<int>(keeps.size());
        for (int i = 0; i < num_dets; ++i)
            nms_dets.push_back(dets[keeps[i]]);
        profiler.reportLayerTime("NonmaximumSuppression", std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - start_nms).count());
        std::cout << profiler << std::endl;
        
        if (cudaSuccess != code) {
            printf("cudaStreamSynchronize fail: %s\n", cudaGetErrorString(code));
        } else {
            printf("cudaStreamSynchronize success\n");
        }
    }
    cudaFree(dets_gpu);
    cudaStreamDestroy(stream);    
    std::cout << "Test JDecoderV2: " << dets_cpu.get()[0] << std::endl;

    constexpr int thickness = 1;
    cv::Mat input = cv::imread("input.jpg");
    cv::Mat output = input.clone();
    float* pout = &dets_cpu.get()[2];
    for (int i = 0; i < dets_cpu.get()[0]; ++i) {
        // for (int j = 0; j < nvinfer1::jdec::boxDim + nvinfer1::jdec::classDim; ++j) {
        //     std::cout << pout[j] << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << dets[i].category << ", " << dets[i].score << ", " << dets[i].bbox.top;
        // std::cout << ", " << dets[i].bbox.left << ", " << dets[i].bbox.bottom << ", " << dets[i].bbox.right << std::endl;
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::rectangle(output, cv::Point(pout[3], pout[2]), cv::Point(pout[5], pout[4]), color, thickness);
        pout += nvinfer1::jdec::decOutputDim;
    }
    cv::imwrite("output.jpg", output);
    
    output = input.clone();
    for (size_t i = 0; i < nms_dets.size(); ++i) {
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::rectangle(output, cv::Point(nms_dets[i].bbox.left, nms_dets[i].bbox.top),
            cv::Point(nms_dets[i].bbox.right, nms_dets[i].bbox.bottom), color, thickness);
    }
    cv::imwrite("output_nms.jpg", output);

    return 0;
}