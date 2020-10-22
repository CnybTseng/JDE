#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <map>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "mot.h"
#include "utils.h"
#include "SH_ImageAlgLogSystem.h"

static void log_fprintf(E_CommonLogLevel level, const char * p_log_info, const char * p_file, int line)
{
    fprintf(stderr, "level:[%d], file:[%s], line:[%d], info:[%s]\n", level, p_file, line, p_log_info);
}

static void do_work(int argc, char* argv[], const char* path)
{
    std::thread::id tid = std::this_thread::get_id();

    // 判断用户提供的图像文件目录是否合法
    struct stat statbuf;
    if (0 != stat(path, &statbuf))
    {
        fprintf(stderr, "stat error: %d\n", errno);
        return;
    }
    
    if (!S_ISDIR(statbuf.st_mode))
    {
        fprintf(stderr, "%s is not a directory!\n", path);
        return;
    }
    
    int save = 0;
    if (argc > 4)
    {
        save = atoi(argv[4]);
    }
    
    // 创建存放结果的目录
    std::stringstream ss;
    ss << tid;
    std::string resdir = "./result" + ss.str() + "/";
    if (save && 0 != access(resdir.c_str(), F_OK))
    {
        printf("create directory: %s\n", resdir.c_str());
        std::string cmd = "mkdir " + resdir;
        if (-1 == system(cmd.c_str()))
        {
            fprintf(stderr, "mkdir fail\n");
            return;
        }
    }
    
    // 2. 加载MOT模型
    int ret = mot::load_mot_model(argv[1]);
    if (ret)
        return;
    
    // OpenCV绘图相关参数
    mot::MOT_Result result;
    int fontface = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontscale = 1;
    int thickness = 1;
    
    // 读取目录中的文件
    dirent **dir = NULL;
    int num = scandir(path, &dir, 0, alphasort);
    float latency = 0;
    for (int i = 0; i < num; ++i)
    {
        // 只处理jpg后缀文件
        if (DT_REG != dir[i]->d_type || !strstr(dir[i]->d_name, "jpg"))
            continue;

        char filein[128] = {0};
        strcat(filein, path);
        strcat(filein, dir[i]->d_name);
        
        // 读取图像文件和解码成BGR888
        cv::Mat bgr = cv::imread(filein);
        if (bgr.empty())
        {
            fprintf(stdout, "cv::imread(%s) fail!\n", filein);
            continue;
        }
#if (!PROFILE)
        auto start = std::chrono::high_resolution_clock::now();
#endif
        // 3. 执行推理, 检测和跟踪目标
        ret = mot::forward_mot_model(bgr.data, bgr.cols, bgr.rows, bgr.step, result);
#if (!PROFILE)
        auto end = std::chrono::high_resolution_clock::now();
        latency = std::chrono::duration<float, std::milli>(end - start).count();
        fprintf(stdout, "\r%s: %s %fms", ss.str().c_str(), filein, latency);
#else
        fprintf(stdout, "\r%s: %s", ss.str().c_str(), filein);
#endif
        fflush(stdout);
        
        if (save)
        {
            // 叠加检测和跟踪结果到图像上
            std::vector<mot::MOT_Track>::iterator riter;
            for (riter = result.begin(); riter != result.end(); riter++)
            {
                cv::Point pt1;
                std::deque<mot::MOT_Rect>::iterator iter;
                for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++)
                {
                    int l = static_cast<int>(iter->left);
                    int t = static_cast<int>(iter->top);
                    int r = static_cast<int>(iter->right);
                    int b = static_cast<int>(iter->bottom);
                    
                    // 过滤无效的检测框
                    if (l == 0 && t == 0 && r == 0 && b == 0)
                        break;
                    
                    // 画轨迹
                    srand(riter->identifier);
                    cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                    if (iter != riter->rects.begin())
                    {
                        cv::Point pt2 = cv::Point((l + r) >> 1, b);
                        cv::line(bgr, pt1, pt2, color, 2);
                        pt1 = pt2;
                        continue;
                    }
                    
                    // 画边框                
                    cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), color, 2);
                    
                    // 叠加轨迹ID号
                    std::ostringstream oss;
                    oss << riter->identifier;
                    cv::String text = oss.str();
                    
                    int baseline;
                    cv::Size tsize = cv::getTextSize(text, fontface, fontscale, thickness, &baseline);
                    
                    int x = std::min(std::max(l, 0), bgr.cols - tsize.width - 1);
                    int y = std::min(std::max(b - baseline, tsize.height), bgr.rows - baseline - 1);
                    cv::putText(bgr, text, cv::Point(x, y), fontface, fontscale, cv::Scalar(0,255,255), thickness);
                    
                    pt1 = cv::Point((l + r) >> 1, b);
                }
            }
            
            // 保存结果图像
            char fileout[128] = {0};
            strcat(fileout, resdir.c_str());
            strcat(fileout, dir[i]->d_name);
            cv::imwrite(fileout, bgr);
        }
        free(dir[i]);
    }
    
    free(dir);
    // 4. 卸载MOT模型
    mot::unload_mot_model();
    fprintf(stdout, "\n");
    
    // std::string vdo(resdir);
    // vdo.pop_back();
    // std::string cmd = "ffmpeg -i " + resdir + "%06d.jpg " + vdo + ".mp4 -y";
    // if (-1 == system(cmd.c_str()))
    // {
    //     fprintf(stderr, "%s: failed\n", cmd.c_str());
    // }
}

int main(int argc, char *argv[])
{
    // 1. 注册日志回调函数, 详情请参考苏永生的日志模块接口文件SH_ImageAlgLogSystem.h
    ImgAlgRegisterLogSystemCallBack((PFUN_LogSystemCallBack)log_fprintf);
    if (argc < 3)
    {
        fprintf(stderr, "Usage:\n%s cfg_path images_path [num_thread,[save<0,1>]]\n", argv[0]);
        return -1;
    }
    
    int num_thread = 1;
    if (argc > 3) {
        num_thread = atoi(argv[3]);
    }
    
    std::vector<std::string> paths;
    if (num_thread > 1) {
        char* path = strtok(argv[2], ",");
        do {
            paths.emplace_back(path);
        } while (path = strtok(NULL, ","));
        if (paths.size() < num_thread) {
            std::string path = paths.back();
            for (size_t i = paths.size(); i < num_thread; ++i) {
                paths.emplace_back(path);
            }
        }
    } else {
        paths.emplace_back(argv[2]);
    }
    
    for (size_t i = 0; i < paths.size(); ++i) {
        std::cout << paths[i] << std::endl;
    }

    // Multiple threads simulate multiple channel video streams.
    std::vector<std::thread> work_groups(num_thread);
    for (size_t i = 0; i < work_groups.size(); ++i) {
        work_groups[i] = std::thread(do_work, argc, argv, paths[i].c_str());
    }
    
    for (size_t i = 0; i < work_groups.size(); ++i) {
        work_groups[i].join();
    }
#if PROFILE    
    std::cout << std::endl << mot::profiler << std::endl;
#endif
}