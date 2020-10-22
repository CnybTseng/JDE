#include <map>
#include <mutex>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "jde.h"
#include "mot.h"
#include "utils.h"
#include "jdecoder.h"
#include "jdecoderv2.h"
#include "jdetracker.h"

namespace mot {

static void correct_bbox(float *ltrb, int imw, int imh, int niw, int nih)
{
    int dx = 0;
    int dy = 0;
    float sx = 1.0f;
    float sy = 1.0f;
    float _niw = imw;
    float _nih = imh;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    
    if (niw / nih < (float)imw / imh)
    {
        _nih = round(_niw * nih / niw);
    }
    else
    {
        _niw = round(_nih * niw / nih);
    }
    
    dx = (static_cast<int>(_niw) - imw) >> 1;
    dy = (static_cast<int>(_nih) - imh) >> 1;
    
    sx = _niw / niw;
    sy = _nih / nih;
    
    x1 = static_cast<int>(sx * ltrb[0] - dx + .5f);
    y1 = static_cast<int>(sy * ltrb[1] - dy + .5f);
    x2 = static_cast<int>(sx * ltrb[2] - dx + .5f);
    y2 = static_cast<int>(sy * ltrb[3] - dy + .5f);

    ltrb[0] = std::max<int>(x1, 0);
    ltrb[1] = std::max<int>(y1, 0);
    ltrb[2] = std::min<int>(x2, imw - 1);
    ltrb[3] = std::min<int>(y2, imh - 1);
}

class MOT
{
public:
    MOT() : traj_cache_len(1), categories{"person"}
    {
        out.resize(NUM_BINDINGS - 1);   // Exclude the input binding
    }
    ~MOT() {}
public:
    DimsX indims;
    std::shared_ptr<unsigned char> rszim_hwc;
    std::shared_ptr<unsigned char> rszim_chw;
    std::shared_ptr<float> in;
    std::vector<std::shared_ptr<float>> out;
#if (USE_DECODERV2 && (!INTEGRATES_DECODER))
    float* dets_gpu;
    std::shared_ptr<float> dets_cpu;
#endif
    std::vector<Detection> rawdet;
    std::vector<Detection> nmsdet;
    const int traj_cache_len;
    const std::vector<std::string> categories;
    std::shared_ptr<JDE> jde;
    std::shared_ptr<JDecoder> decoder;
#if (USE_DECODERV2 && (!INTEGRATES_DECODER))
    std::shared_ptr<nvinfer1::JDecoderPlugin> decoderv2;
#endif
    std::shared_ptr<JDETracker> tracker;
};

static std::map<std::thread::id, std::shared_ptr<MOT>> models;
static std::mutex mtx;

int load_mot_model(const char *cfg_path)
{
    mtx.lock();
    std::thread::id tid = std::this_thread::get_id();
    models[tid] = std::make_shared<MOT>();
    
    MOT& model = *models[tid].get();
    model.jde = std::make_shared<JDE>();
    model.decoder = std::make_shared<JDecoder>();
#if (USE_DECODERV2 && (!INTEGRATES_DECODER))
    model.decoderv2 = std::make_shared<nvinfer1::JDecoderPlugin>();
#endif
    model.tracker = std::make_shared<JDETracker>();
    
    // Initialize JDE instance.
    bool status = model.jde->init();
    if (!status) {
        std::cerr << "build JDE fail" << std::endl;
        mtx.unlock();
        return -1;
    }
    
    // Initialize JDE Decoder.
    status = model.decoder->init();
    if (!status) {
        std::cerr << "build JDecoder fail" << std::endl;
        mtx.unlock();
        return -1;
    }
    
    // Initialize object tracker.
    if (!model.tracker->init())
    {
        std::cerr << "build JDETracker fail" << std::endl;
        mtx.unlock();
        return -1;
    }
    
    // Allocate algorithm usage buffers.
    model.indims = model.jde->get_binding_dims(0);
    model.rszim_hwc = std::shared_ptr<unsigned char>(new unsigned char[model.indims.numel()]);
    model.rszim_chw = std::shared_ptr<unsigned char>(new unsigned char[model.indims.numel()]);
    model.in = std::shared_ptr<float>(new float[model.indims.numel()]);
    
    for (int i = 0; i < model.out.size(); ++i) {
        DimsX dims = model.jde->get_binding_dims(i + 1);
        model.out[i] = std::shared_ptr<float>(new float[dims.numel()]);
    }
#if (USE_DECODERV2 && (!INTEGRATES_DECODER))    
    size_t numel = numel_after_align(nvinfer1::jdec::maxNumOutputBox *
        nvinfer1::jdec::decOutputDim + 1, sizeof(float), 8);
    cudaMalloc((void**)&model.dets_gpu, numel * sizeof(float));    
    model.dets_cpu = std::shared_ptr<float>(new float[numel]);
#endif    
    model.rawdet.reserve(nvinfer1::jdec::maxNumOutputBox);
    model.nmsdet.reserve(nvinfer1::jdec::maxNumOutputBox);
    
    mtx.unlock();
    
    return 0;
}

int unload_mot_model()
{
    std::thread::id tid = std::this_thread::get_id();
    MOT& model = *models[tid].get();
#if (USE_DECODERV2 && (!INTEGRATES_DECODER))    
    cudaFree(model.dets_gpu);
#endif    
    // Destroy JDE instance.
    bool status = model.jde->destroy();
    if (!status) {
        std::cout << "destroy JDE fail" << std::endl;
        return -1;
    }
    
    // Destroy JDecoder instance.
    status = model.decoder->destroy();
    if (!status) {
        std::cout << "destroy JDecoder fail" << std::endl;
        return -1;
    }
    
    // Destroy object tracker.
    model.tracker->free();    
    
    return 0;
}

int forward_mot_model(const unsigned char *data, int width, int height, int stride, MOT_Result &result)
{
    std::thread::id tid = std::this_thread::get_id();
    MOT& model = *models[tid].get();
#if PROFILE
    auto start_prep = std::chrono::high_resolution_clock::now();
#endif
    // Resize the image to the neural network input size.
    cv::Mat src(height, width, CV_8UC3, const_cast<unsigned char*>(data));
    cv::Mat rszim_hwc(model.indims.d[2], model.indims.d[3], CV_8UC3, model.rszim_hwc.get());
    cv::resize(src, rszim_hwc, cv::Size(model.indims.d[3], model.indims.d[2]));

    // Convert HWC to CHW and swap the red and blue channel if necessary.
    unsigned char* pchannel = model.rszim_chw.get();
    size_t channel_size = model.indims.d[2] * model.indims.d[3];
    std::vector<cv::Mat> rszim_chw_vec = {
        cv::Mat(model.indims.d[2], model.indims.d[3], CV_8UC1, pchannel),
        cv::Mat(model.indims.d[2], model.indims.d[3], CV_8UC1, pchannel + channel_size),
        cv::Mat(model.indims.d[2], model.indims.d[3], CV_8UC1, pchannel + (channel_size << 1))
    };
    cv::split(rszim_hwc, rszim_chw_vec);
    
    // Normalizate the fake three channel image.
    cv::Mat in(model.indims.d[2], model.indims.d[3], CV_32FC3, model.in.get());
    cv::Mat rszim_chw(model.indims.d[2], model.indims.d[3], CV_8UC3, model.rszim_chw.get());
    rszim_chw.convertTo(in, CV_32FC3);
#if PROFILE   
    profiler.reportLayerTime("image preprocessing", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_prep).count());
    auto start_jde = std::chrono::high_resolution_clock::now();
#endif    
    // Neural network inference.
    bool status = model.jde->infer(model.in, model.out);
    if (!status) {
        std::cout << "infer JDE fail" << std::endl;
        return -1;
    }
#if PROFILE
    profiler.reportLayerTime("JDE inference", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_jde).count());
    auto start_dec = std::chrono::high_resolution_clock::now();
#endif    
    // Test jdecoderv2
    // {
    //     for (int i = 0; i < model.out.size(); ++i) {
    //         std::ofstream ofs("out" + std::to_string(i) + ".bin", std::ios::binary);
    //         DimsX dims = model.jde->get_binding_dims(i + 1);
    //         ofs.write(reinterpret_cast<char*>(model.out[i].get()), dims.numel() * sizeof(float));
    //         ofs.close();
    //     }
    //     cv::imwrite("input.jpg", rszim_hwc);
    // }
    // Decoding neural network output.
#if USE_DECODERV2
#if INTEGRATES_DECODER
    size_t num_det = static_cast<size_t>(model.out[0].get()[0]);
    model.rawdet.resize(num_det);
    memcpy((char*)(model.rawdet.data()), (char*)(&model.out[0].get()[2]),
        num_det * nvinfer1::jdec::decOutputDim * sizeof(float));
#else   // INTEGRATES_DECODER
    const void *ins[] = {model.jde.get()->get_binding(1), model.jde.get()->get_binding(2),
        model.jde.get()->get_binding(3)};
    model.decoderv2.get()->forward_test((const float* const*)ins, model.dets_gpu, 0);
    size_t numel = numel_after_align(nvinfer1::jdec::maxNumOutputBox *
        nvinfer1::jdec::decOutputDim + 1, sizeof(float), 8);
    cudaMemcpy(model.dets_cpu.get(), model.dets_gpu, numel * sizeof(float), cudaMemcpyDeviceToHost);
    size_t num_det = static_cast<size_t>(model.dets_cpu.get()[0]);
    model.rawdet.resize(num_det);
    memcpy((char*)(model.rawdet.data()), (char*)(&model.dets_cpu.get()[2]),
        num_det * nvinfer1::jdec::decOutputDim * sizeof(float));
#endif  // INTEGRATES_DECODER
    QsortDescentInplace(model.rawdet);
    std::vector<size_t> keeps;
    const float iou_thresh = 0.4f;
    NonmaximumSuppression(model.rawdet, keeps, iou_thresh);
    model.nmsdet.resize(keeps.size());
    for (size_t i = 0; i < keeps.size(); ++i) {
        model.nmsdet[i] = model.rawdet[keeps[i]];
    }
    // std::cout << tid << "=> " << model.rawdet.size() << "," << model.nmsdet.size() << std::endl;

    size_t i = 0;
    std::vector<Detection>::iterator iter;
    cv::Mat dets(model.nmsdet.size(), 6 + EMBD_DIM, CV_32FC1);
    for (iter = model.nmsdet.begin(); iter != model.nmsdet.end(); ++iter, ++i) {
        *dets.ptr<float>(i, 0) = iter->category;
        *dets.ptr<float>(i, 1) = iter->score;
        *dets.ptr<float>(i, 2) = iter->bbox.left;
        *dets.ptr<float>(i, 3) = iter->bbox.top;
        *dets.ptr<float>(i, 4) = iter->bbox.right;
        *dets.ptr<float>(i, 5) = iter->bbox.bottom;
        correct_bbox(dets.ptr<float>(i) + 2, width, height, model.indims.d[3], model.indims.d[2]);
        memcpy(dets.ptr<float>(i) + 6, iter->embedding, sizeof(iter->embedding));
    }
#else   // USE_DECODERV2
    std::vector<Detection> dets_;
    status = model.decoder->infer(model.out, dets_);
    if (!status) {
        std::cout << "infer JDecoder fail" << std::endl;
        return -1;
    }
    
    size_t i = 0;
    std::vector<Detection>::iterator iter;
    cv::Mat dets(dets_.size(), 6 + EMBD_DIM, CV_32FC1);
    for (iter = dets_.begin(); iter != dets_.end(); ++iter, ++i) {
        *dets.ptr<float>(i, 0) = iter->category;
        *dets.ptr<float>(i, 1) = iter->score;
        *dets.ptr<float>(i, 2) = iter->bbox.left;
        *dets.ptr<float>(i, 3) = iter->bbox.top;
        *dets.ptr<float>(i, 4) = iter->bbox.right;
        *dets.ptr<float>(i, 5) = iter->bbox.bottom;
        correct_bbox(dets.ptr<float>(i) + 2, width, height, model.indims.d[3], model.indims.d[2]);
        memcpy(dets.ptr<float>(i) + 6, iter->embedding, sizeof(iter->embedding));
    }
#endif  // USE_DECODERV2
#if PROFILE
    profiler.reportLayerTime("JDE decode", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_dec).count());
    auto start_asso = std::chrono::high_resolution_clock::now();
#endif
    // Update object trackers' state.
    std::vector<mot::Track> tracks;
    model.tracker->update(dets, tracks);
    
    // Associate object trajectories.
    std::vector<MOT_Track>::iterator riter;
    for (riter = result.begin(); riter != result.end();)
    {
        bool match = false;
        std::vector<mot::Track>::iterator titer;
        for (titer = tracks.begin(); titer != tracks.end(); )
        {
            if (riter->identifier == titer->id)
            {
                MOT_Rect rect = {
                    .top = titer->ltrb[1],
                    .left = titer->ltrb[0],
                    .bottom = titer->ltrb[3],
                    .right = titer->ltrb[2]};
                riter->rects.push_front(rect);
                riter->rects.pop_back();
                titer = tracks.erase(titer);
                match = true;
            }
            else
                titer++;
        }
        if (match)
            riter++;
        else
            riter = result.erase(riter);
    }
    
    // Initialize new tracks.
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        MOT_Rect rect = {
            .top = tracks[i].ltrb[1],
            .left = tracks[i].ltrb[0],
            .bottom = tracks[i].ltrb[3],
            .right = tracks[i].ltrb[2]};
        MOT_Track track = {
            .identifier = tracks[i].id,
            .category = std::string(model.categories[0])};
        track.rects.resize(model.traj_cache_len);
        track.rects.push_front(rect);
        track.rects.pop_back();
        result.push_back(track);
    }
#if PROFILE    
    profiler.reportLayerTime("online association", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_asso).count());
#endif    
    return 0;
}

}   // namespace mot