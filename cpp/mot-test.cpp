#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mot.h"
#include "SH_ImageAlgLogSystem.h"

static void log_fprintf(E_CommonLogLevel level, const char * p_log_info, const char * p_file, int line)
{
    fprintf(stderr, "level:[%d], file:[%s], line:[%d], info:[%s]\n", level, p_file, line, p_log_info);
}

int main(int argc, char *argv[])
{
    ImgAlgRegisterLogSystemCallBack((PFUN_LogSystemCallBack)log_fprintf);
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
        ret = mot::forward_mot_model(rgb.data, rgb.cols, rgb.rows, rgb.step, result);
        
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
                
                if (l == 0 && t == 0 && r == 0 && b == 0)
                    break;
                
                srand(riter->identifier);
                cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                if (iter != riter->rects.begin())
                {
                    cv::Point pt2 = cv::Point((l + r) >> 1, b);
                    cv::line(bgr, pt1, pt2, color, 2);
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
                
                pt1 = cv::Point((l + r) >> 1, b);
            }
        }
        
        char fileout[128] = {0};
        strcat(fileout, "./result/");
        strcat(fileout, dir[i]->d_name);
        cv::imwrite(fileout, bgr);
        free(dir[i]);
    }
    
    free(dir);
    mot::unload_mot_model();
}