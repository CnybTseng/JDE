#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#   include "gpu.h"
#endif  // NCNN_VULKAN

struct Detection
{
    cv::Rect_<float> rect;
    int category;
    float score;
};

int main(int argc, char *argv[])
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage:\n%s param_path model_path image_path neth netw\n", argv[0]);
        return -1;
    }
    
    cv::Mat bgr = cv::imread(argv[3]);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread(%s) fail!\n", argv[3]);
        return -1;
    }
    
#if NCNN_VULKAN
    //ncnn::create_gpu_instance();
#endif  // NCNN_VULKAN
    
    ncnn::Net jde;

#if NCNN_VULKAN
    //jde.opt.use_vulkan_compute = true;
#endif  // NCNN_VULKAN
    
    jde.load_param(argv[1]);
    jde.load_model(argv[2]);
    
    int netw = atoi(argv[5]);
    int neth = atoi(argv[4]);
    
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, netw, neth);
    
    const float means[] = {0.f, 0.f, 0.f};
    const float norms[] = {0.0039215686f, 0.0039215686f, 0.0039215686f};
    in.substract_mean_normalize(means, norms);
    
    ncnn::Extractor ext = jde.create_extractor();
    ext.set_num_threads(6);
    
    ext.input("data", in);
    
    ncnn::Mat out;
    ext.extract("detout", out);
    
    std::vector<Detection> dets;
    for (int i = 0; i < out.h; ++i)
    {
        const float* val = out.row(i);
        Detection det;
        det.category = static_cast<int>(val[0]);
        det.score = val[1];
        // det.rect.x = val[2];
        fprintf(stderr, "%d %.0f %f %f %f %f %f\n", i, val[0], val[1], val[2], val[3], val[4], val[5]);
    }
    
#if NCNN_VULKAN
    //ncnn::destroy_gpu_instance();
#endif  // NCNN_VULKAN

    return 0;
}