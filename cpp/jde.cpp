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

static void calcbbox(float *x1y1x2y2, int imw, int imh, int niw, int nih)
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
    
    x1 = static_cast<int>(sx * x1y1x2y2[0] - dx + .5f);
    y1 = static_cast<int>(sy * x1y1x2y2[1] - dy + .5f);
    x2 = static_cast<int>(sx * x1y1x2y2[2] - dx + .5f);
    y2 = static_cast<int>(sy * x1y1x2y2[3] - dy + .5f);

    x1y1x2y2[0] = std::max<int>(x1, 0);
    x1y1x2y2[1] = std::max<int>(y1, 0);
    x1y1x2y2[2] = std::min<int>(x2, imw - 1);
    x1y1x2y2[3] = std::min<int>(y2, imh - 1);
}

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

    // We have to ensure that ncnn::destroy_gpu_instance() is called
    // after ncnn::Net::~Net(). Please reference
    // https://github.com/Tencent/ncnn/issues/1495
    
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif  // NCNN_VULKAN
    {
        ncnn::Net jde;

#if NCNN_VULKAN
        jde.opt.use_vulkan_compute = true;
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
            float x1y1x2y2[4] = {val[3], val[2], val[5], val[4]};
            calcbbox(x1y1x2y2, bgr.cols, bgr.rows, netw, neth);
            int l = static_cast<int>(x1y1x2y2[0]);
            int t = static_cast<int>(x1y1x2y2[1]);
            int r = static_cast<int>(x1y1x2y2[2]);
            int b = static_cast<int>(x1y1x2y2[3]);
            cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), cv::Scalar(0, 255, 255), 2);
        }
    }    
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif  // NCNN_VULKAN

    cv::imwrite("result.png", bgr);

    return 0;
}