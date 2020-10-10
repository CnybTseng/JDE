#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "jde.h"
#include "utils.h"
#include "jdecoder.h"
#include "mot.h"
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

static class MOT
{
public:
    MOT() : traj_cache_len(30), categories{"person"}
    {
        out.resize(3);
    }
    ~MOT() {}
public:
    DimsX indims;
    std::shared_ptr<unsigned char> rszim_hwc;
    std::shared_ptr<unsigned char> rszim_chw;
    std::shared_ptr<float> in;
    std::vector<std::shared_ptr<float>> out;
    const int traj_cache_len;
    const std::vector<std::string> categories;
} __model;

int load_mot_model(const char *cfg_path)
{
    // Initialize JDE instance.
    bool status = mot::JDE::instance()->init();
    if (!status) {
        std::cout << "build JDE fail" << std::endl;
        return -1;
    }
    
    // Initialize JDE Decoder.
    status = mot::JDecoder::instance()->init();
    if (!status) {
        std::cout << "build JDecoder fail" << std::endl;
        return -1;
    }
    
    // Initialize object tracker.
    if (!JDETracker::instance()->init())
    {
        std::cerr << "JDETracker::instance()->init() fail" << std::endl;
        return -1;
    }
    
    // Allocate algorithm needed buffers.
    __model.indims = mot::JDE::instance()->get_binding_dims(0);
    __model.rszim_hwc = std::shared_ptr<unsigned char>(new unsigned char[__model.indims.numel()]);
    __model.rszim_chw = std::shared_ptr<unsigned char>(new unsigned char[__model.indims.numel()]);
    __model.in = std::shared_ptr<float>(new float[__model.indims.numel()]);
    
    for (int i = 0; i < __model.out.size(); ++i) {
        mot::DimsX dims = mot::JDE::instance()->get_binding_dims(i + 1);
        __model.out[i] = std::shared_ptr<float>(new float[dims.numel()]);
    }
    
    return 0;
}

int unload_mot_model()
{
    // Destroy JDE instance.
    bool status = mot::JDE::instance()->destroy();
    if (!status) {
        std::cout << "destroy JDE fail" << std::endl;
        return -1;
    }
    
    // Destroy JDecoder instance.
    status = mot::JDecoder::instance()->destroy();
    if (!status) {
        std::cout << "destroy JDecoder fail" << std::endl;
        return -1;
    }
    
    // Destroy object tracker.
    JDETracker::instance()->free();
    
    return 0;
}

int forward_mot_model(const unsigned char *rgb, int width, int height, int stride, MOT_Result &result)
{
    SimpleProfiler profiler("mot");
    auto start_prep = std::chrono::high_resolution_clock::now();
    
    // Resize the image to the neural network input size.
    cv::Mat src(height, width, CV_8UC3, const_cast<unsigned char*>(rgb));
    cv::Mat rszim_hwc(__model.indims.d[2], __model.indims.d[3], CV_8UC3, __model.rszim_hwc.get());
    cv::resize(src, rszim_hwc, cv::Size(__model.indims.d[3], __model.indims.d[2]));
    
    // Convert HWC to CHW and swap the red and blue channel if needed.
    unsigned char* pchannel = __model.rszim_chw.get();
    size_t channel_size = __model.indims.d[2] * __model.indims.d[3];
    std::vector<cv::Mat> rszim_chw_vec = {
        cv::Mat(__model.indims.d[2], __model.indims.d[3], CV_8UC1, pchannel),
        cv::Mat(__model.indims.d[2], __model.indims.d[3], CV_8UC1, pchannel + channel_size),
        cv::Mat(__model.indims.d[2], __model.indims.d[3], CV_8UC1, pchannel + (channel_size << 1))
    };
    cv::split(rszim_hwc, rszim_chw_vec);
    
    // Normalizate the fake three channel image.
    cv::Mat in(__model.indims.d[2], __model.indims.d[3], CV_32FC3, __model.in.get());
    cv::Mat rszim_chw(__model.indims.d[2], __model.indims.d[3], CV_8UC3, __model.rszim_chw.get());
    rszim_chw.convertTo(in, CV_32FC3, 1.f / 255);
   
    profiler.reportLayerTime("image preprocessing", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_prep).count());
    auto start_jde = std::chrono::high_resolution_clock::now();
    
    // Neural network inference.
    bool status = mot::JDE::instance()->infer(__model.in, __model.out);
    if (!status) {
        std::cout << "infer JDE fail" << std::endl;
        return -1;
    }

    profiler.reportLayerTime("JDE inference", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_jde).count());
    
    // Test jdecoderv2
    {
        for (int i = 0; i < __model.out.size(); ++i) {
            std::ofstream ofs("out" + std::to_string(i) + ".bin", std::ios::binary);
            mot::DimsX dims = mot::JDE::instance()->get_binding_dims(i + 1);
            ofs.write(reinterpret_cast<char*>(__model.out[i].get()), dims.numel() * sizeof(float));
            ofs.close();
        }
        cv::imwrite("input.jpg", rszim_hwc);
    }    
    
    auto start_jdec = std::chrono::high_resolution_clock::now();
    
    // Decode neural network outputs.
    std::vector<Detection> dets_;
    status = mot::JDecoder::instance()->infer(__model.out, dets_);
    if (!status) {
        std::cout << "infer JDecoder fail" << std::endl;
        return -1;
    }
    
    profiler.reportLayerTime("JDE decode", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_jdec).count());
    auto start_asso = std::chrono::high_resolution_clock::now();
    
    // Update object trackers' states.
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
        correct_bbox(dets.ptr<float>(i) + 2, width, height, __model.indims.d[3], __model.indims.d[2]);
        memcpy(dets.ptr<float>(i) + 6, iter->embedding, sizeof(iter->embedding));
    }
    
    std::vector<mot::Track> tracks;
    JDETracker::instance()->update(dets, tracks);
    
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
            .category = std::string(__model.categories[0])};
        track.rects.resize(__model.traj_cache_len);
        track.rects.push_front(rect);
        result.push_back(track);
    }
    
    profiler.reportLayerTime("online association", std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - start_asso).count());
    std::cout << profiler << std::endl;
    
    return 0;
}

}   // namespace mot