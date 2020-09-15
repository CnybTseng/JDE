#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#include "yaml-cpp/yaml.h"
#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#   include "gpu.h"
#endif  // NCNN_VULKAN

#include "jdetracker.h"

#include "SH_ImageAlgLogSystem.h"

/* 判断yaml指定节点定义且定义为指定类型 */
#define __yaml_node_is_defined_as_type(node, type)  \
do {                                                \
    if (!node.IsDefined())                          \
    {                                               \
        LogError(#node " isn't defined");           \
        return -1;                               \
    }                                               \
    if (!node.Is##type())                           \
    {                                               \
        LogError(#node " isn't " #type);            \
        return -1;                               \
    }                                               \
} while (0)

/*****************************************************************************
 函 数 名  : yaml_node_is_defined_as_type
 功能描述  : 判断yaml指定节点定义且定义为指定类型
 输入参数  : node  
             type  
 输出参数  : 无
 返 回 值  : #define
 调用函数  : 
 被调函数  : 
 
 修改历史      :
  1.日    期   : 2020年3月12日
    作    者   : Zeng Zhiwei
    修改内容   : 新生成函数

*****************************************************************************/
#define yaml_node_is_defined_as_type(node, type) __yaml_node_is_defined_as_type(node, type)

namespace mot {

static struct
{
    int netw = 576;
    int neth = 320;
    int traj_cache_len = 30;
    std::string param_path;
    std::string bin_path;
    ncnn::Net *jde;
    std::vector<std::string> categories;
    float means[3] = {0.f, 0.f, 0.f};
    float norms[3] = {0.0039215686f, 0.0039215686f, 0.0039215686f};
} __model;

static int load_config(const char *cfg_path)
{
    try
    {
        YAML::Node root = YAML::LoadFile(cfg_path);
        if (!root["MOT"].IsDefined())
        {
            LogError("MOT isn't defined");
            return -1;
        }
        
        YAML::Node mot = root["MOT"];       
        yaml_node_is_defined_as_type(mot["netw"], Scalar);
        __model.netw = mot["netw"].as<int>();
        
        yaml_node_is_defined_as_type(mot["neth"], Scalar);
        __model.neth = mot["neth"].as<int>();
        
        yaml_node_is_defined_as_type(mot["traj_cache_len"], Scalar);
        __model.traj_cache_len = mot["traj_cache_len"].as<int>();
        
        yaml_node_is_defined_as_type(mot["param_path"], Scalar);
        __model.param_path = mot["param_path"].as<std::string>();
        
        yaml_node_is_defined_as_type(mot["bin_path"], Scalar);
        __model.bin_path = mot["bin_path"].as<std::string>();
        
        yaml_node_is_defined_as_type(mot["categories"], Sequence);
        __model.categories = mot["categories"].as<std::vector<std::string>>();
        
        yaml_node_is_defined_as_type(mot["mean_vals"], Sequence);
        std::vector<float> mean_vals = mot["mean_vals"].as<std::vector<float>>();
        std::copy(mean_vals.begin(), mean_vals.end(), __model.means);
        
        yaml_node_is_defined_as_type(mot["norm_vals"], Sequence);
        std::vector<float> norm_vals = mot["norm_vals"].as<std::vector<float>>();
        std::copy(norm_vals.begin(), norm_vals.end(), __model.norms);
    }
    catch (...)
    {
        LogError("invalid yaml configuration file");
        return -1;
    }
   
    return 0;
}

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
    
    if (load_config(cfg_path))
    {
        LogError("load_config() fail");
        return -1;
    }
    
    __model.jde->load_param(__model.param_path.c_str());
    __model.jde->load_model(__model.bin_path.c_str());
    
    if (!JDETracker::instance()->init())
    {
        LogError("JDETracker::instance()->init() fail");
        return -1;
    }
    
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
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    
    ncnn::Extractor ext = __model.jde->create_extractor();
    ext.set_num_threads(6);        
    ext.input("data", in);
    
    ncnn::Mat out;
    ext.extract("detout", out);
    
    gettimeofday(&t2, NULL);
    fprintf(stdout, "inference %fms\n", (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) * 0.001f);
    // gettimeofday(&t1, NULL);
    
    for (int i = 0; i < out.h; ++i)
    {
        float* val = out.row(i);
        correct_bbox(val + 2, width, height, __model.netw, __model.neth);
    }
    
    std::vector<mot::Track> tracks;
    cv::Mat dets(out.h, out.w, CV_32FC1, out.data);
    JDETracker::instance()->update(dets, tracks);
    
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
    
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        MOT_Rect rect = {
            .top = tracks[i].ltrb[1],
            .left = tracks[i].ltrb[0],
            .bottom = tracks[i].ltrb[3],
            .right = tracks[i].ltrb[2]};
        MOT_Track track = {
            .identifier = tracks[i].id,
            .posture = STANDING,
            .category = std::string(__model.categories[0])};
        track.rects.resize(__model.traj_cache_len);
        track.rects.push_front(rect);
        result.push_back(track);
    }
    // gettimeofday(&t2, NULL);
    // fprintf(stdout, "association %fms\n", (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) * 0.001f);
    return 0;
}

}