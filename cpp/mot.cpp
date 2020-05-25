#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#   include "gpu.h"
#endif  // NCNN_VULKAN

#include "jdetracker.h"

namespace mot {

static struct
{
    int netw = 576;
    int neth = 320;
    ncnn::Net *jde;
    float means[3] = {0.f, 0.f, 0.f};
    float norms[3] = {0.0039215686f, 0.0039215686f, 0.0039215686f};
} __model;

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

int load_mot_model(const char *cfg_path)
{    
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif  // NCNN_VULKAN
    
    __model.jde = new ncnn::Net;
    
#if NCNN_VULKAN
    __model.jde->opt.use_vulkan_compute = true;
#endif  // NCNN_VULKAN
    
    __model.jde->load_param("../baseline.param");
    __model.jde->load_model("../baseline.bin");
    
    JDETracker::instance()->init();
    
    return 0;
}

int unload_mot_model()
{
    JDETracker::instance()->free();
    
    delete __model.jde;
    
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif  // NCNN_VULKAN
    
    return 0;
}

int forward_mot_model(const unsigned char *rgb, int width, int height, int stride, MOT_Result &result)
{
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB, width, height, __model.netw, __model.neth);
    in.substract_mean_normalize(__model.means, __model.norms);
        
    ncnn::Extractor ext = __model.jde->create_extractor();
    ext.set_num_threads(6);        
    ext.input("data", in);
    
    ncnn::Mat out;
    ext.extract("detout", out);
    
    for (int i = 0; i < out.h; ++i)
    {
        float* val = out.row(i);
        correct_bbox(val + 2, width, height, __model.netw, __model.neth);
    }
    
    std::vector<mot::Track> tracks;
    cv::Mat dets(out.h, out.w, CV_32FC1, out.data);
    JDETracker::instance()->update(dets, tracks);
    
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        bool match = false;
        for (size_t j = 0; j < result.size(); ++j)
        {
            if (result[j].identifier == tracks[i].id)
            {
                MOT_Rect rect = {
                    .top = tracks[i].ltrb[1],
                    .left = tracks[i].ltrb[0],
                    .bottom = tracks[i].ltrb[3],
                    .right = tracks[i].ltrb[2]};
                result[j].rects.push_front(rect);
                match = true;
                break;
            }
        }
        
        if (match)
            continue;
        
        MOT_Track track = {
            .identifier = tracks[i].id,
            .posture = STANDING,
            .category = std::string("")};
        track.rects.resize(30);
        result.push_back(track);
    }
    
    return 0;
}

}

static int test(int argc, char *argv[])
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage:\n%s param_path model_path images_path neth netw\n", argv[0]);
        return -1;
    }
    
    struct stat statbuf;
    if (0 != stat(argv[3], &statbuf))
    {
        fprintf(stderr, "stat error: %d\n", errno);
        return -1;
    }
    
    if (!S_ISDIR(statbuf.st_mode))
    {
        fprintf(stderr, "%s is not a directory!\n", argv[3]);
        return -1;
    }
    
    if (0 != access("./result", F_OK))
    {
        printf("create directory: result\n");
        system("mkdir ./result");
    }
    
    mot::JDETracker::instance()->init();
    
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif  // NCNN_VULKAN    
    ncnn::Net *jde = new ncnn::Net;
#if NCNN_VULKAN
    jde->opt.use_vulkan_compute = true;
#endif  // NCNN_VULKAN
    jde->load_param(argv[1]);
    jde->load_model(argv[2]);
    
    int netw = atoi(argv[5]);
    int neth = atoi(argv[4]);

    dirent **dir = NULL;
    int num = scandir(argv[3], &dir, 0, alphasort);
    for (int i = 0; i < num; ++i)
    {
        if (DT_REG != dir[i]->d_type || !strstr(dir[i]->d_name, "jpg"))
            continue;

        char filein[128] = {0};
        strcat(filein, argv[3]);
        strcat(filein, dir[i]->d_name);
        fprintf(stdout, "%s\n", filein);
        
        cv::Mat bgr = cv::imread(filein);
        if (bgr.empty())
        {
            fprintf(stdout, "cv::imread(%s) fail!\n", filein);
            continue;
        }
        
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, netw, neth);
        
        const float means[] = {0.f, 0.f, 0.f};
        const float norms[] = {0.0039215686f, 0.0039215686f, 0.0039215686f};
        in.substract_mean_normalize(means, norms);
        
        ncnn::Extractor ext = jde->create_extractor();
        ext.set_num_threads(6);        
        ext.input("data", in);
        
        ncnn::Mat out;
        ext.extract("detout", out);
        
        for (int i = 0; i < out.h; ++i)
        {
            float* val = out.row(i);
            mot::correct_bbox(val + 2, bgr.cols, bgr.rows, netw, neth);
            int l = static_cast<int>(val[2]);
            int t = static_cast<int>(val[3]);
            int r = static_cast<int>(val[4]);
            int b = static_cast<int>(val[5]);
            cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), cv::Scalar(0, 255, 255), 1);
        }

        std::vector<mot::Track> tracks;
        cv::Mat dets(out.h, out.w, CV_32FC1, out.data);
        mot::JDETracker::instance()->update(dets, tracks);
        fprintf(stdout, "tracks %lu\n", tracks.size());
        
        int fontface = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double fontscale = 1;
        int thickness = 1;
        for (size_t i = 0; i < tracks.size(); ++i)
        {
            int l = static_cast<int>(tracks[i].ltrb[0]);
            int t = static_cast<int>(tracks[i].ltrb[1]);
            int r = static_cast<int>(tracks[i].ltrb[2]);
            int b = static_cast<int>(tracks[i].ltrb[3]);
            
            srand(tracks[i].id);
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
            cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), color, 2);
            
            std::ostringstream oss;
            oss << tracks[i].id;
            cv::String text = oss.str();
            
            int baseline;
            cv::Size tsize = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
            
            int x = std::min(std::max(l, 0), bgr.cols - tsize.width - 1);
            int y = std::min(std::max(b - baseline, tsize.height), bgr.rows - baseline - 1);
            cv::putText(bgr, text, cv::Point(x, y), fontface, fontscale, cv::Scalar(0,255,255), thickness);
        }
        
        char fileout[128] = {0};
        strcat(fileout, "./result/");
        strcat(fileout, dir[i]->d_name);
        cv::imwrite(fileout, bgr);
        free(dir[i]);
    }
    
    free(dir);
    delete jde;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif  // NCNN_VULKAN
    
    mot::JDETracker::instance()->free();    
    system("./ffmpeg -i result/%06d.jpg result.mp4 -y");
    
    return 0;
}

int main(int argc, char *argv[])
{
#if 0
    return test(argc, argv);
#else
    if (argc < 3)
    {
        fprintf(stderr, "Usage:\n%s cfg_path images_path\n", argv[0]);
        return -1;
    }
    
    struct stat statbuf;
    if (0 != stat(argv[2], &statbuf))
    {
        fprintf(stderr, "stat error: %d\n", errno);
        return -1;
    }
    
    if (!S_ISDIR(statbuf.st_mode))
    {
        fprintf(stderr, "%s is not a directory!\n", argv[2]);
        return -1;
    }
    
    if (0 != access("./result", F_OK))
    {
        printf("create directory: result\n");
        system("mkdir ./result");
    }

    int ret = mot::load_mot_model(argv[1]);
    if (ret)
        return -1;
    
    mot::MOT_Result result;
    int fontface = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontscale = 1;
    int thickness = 1;
        
    dirent **dir = NULL;
    int num = scandir(argv[2], &dir, 0, alphasort);
    for (int i = 0; i < num; ++i)
    {
        if (DT_REG != dir[i]->d_type || !strstr(dir[i]->d_name, "jpg"))
            continue;

        char filein[128] = {0};
        strcat(filein, argv[2]);
        strcat(filein, dir[i]->d_name);
        fprintf(stdout, "%s\n", filein);
        
        cv::Mat bgr = cv::imread(filein);
        if (bgr.empty())
        {
            fprintf(stdout, "cv::imread(%s) fail!\n", filein);
            continue;
        }
        
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        ret = mot::forward_mot_model(rgb.data, rgb.cols, rgb.rows, rgb.cols * 3, result);
        
        std::vector<mot::MOT_Track>::iterator riter;
        for (riter = result.begin(); riter != result.end();)
        {
            bool flag = false;
            cv::Point pt1;
            std::deque<mot::MOT_Rect>::iterator iter;
            for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++)
            {
                int l = static_cast<int>(iter->left);
                int t = static_cast<int>(iter->top);
                int r = static_cast<int>(iter->right);
                int b = static_cast<int>(iter->bottom);
                
                if (l == 0 && t == 0 && r == 0 && b == 0)
                    continue;
                
                srand(riter->identifier);
                cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                if (iter != riter->rects.begin())
                {
                    cv::Point pt2 = cv::Point((l + r) >> 1, (t + b) >> 1);
                    cv::line(bgr, pt1, pt2, color);
                    pt1 = pt2;
                    continue;
                }
                                
                cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), color, 2);
                
                std::ostringstream oss;
                oss << riter->identifier;
                cv::String text = oss.str();
                
                int baseline;
                cv::Size tsize = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
                
                int x = std::min(std::max(l, 0), bgr.cols - tsize.width - 1);
                int y = std::min(std::max(b - baseline, tsize.height), bgr.rows - baseline - 1);
                cv::putText(bgr, text, cv::Point(x, y), fontface, fontscale, cv::Scalar(0,255,255), thickness);
                
                pt1 = cv::Point((l + r) >> 1, (t + b) >> 1);
                flag = true;
            }
            
            if (!flag)
                riter = result.erase(riter);
            else
                riter++;
        }
        
        char fileout[128] = {0};
        strcat(fileout, "./result/");
        strcat(fileout, dir[i]->d_name);
        cv::imwrite(fileout, bgr);
        free(dir[i]);
    }
    
    free(dir);
    mot::unload_mot_model();
#endif
}