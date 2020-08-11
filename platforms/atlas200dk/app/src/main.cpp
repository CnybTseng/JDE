#include <cstdio>
#include <dirent.h>
#include <unistd.h>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "mot.h"
#include "datastore.h"

int main(int argc, char *argv[])
{
    if (mot::load_mot_model("")) {
        fprintf(stderr, "load_mot_model fail\n");
        return -1;
    }
   
    int width = 1920;
    int height = 1080;
    int stride = width * 3 / 2;
        
    std::vector<std::string> fpath;
    read_path_files("/home/HwHiAiUser/imgs", fpath);
    std::sort(fpath.begin(), fpath.end());
    
    bool flag = false;
    mot::MOT_Result result;int fontface = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontscale = 1;
    int thickness = 1;    

    for (std::string path : fpath) {
        struct JpegdOut jpegd_out;
        jpeg_decode(path, jpegd_out);
        mot::forward_mot_model((const unsigned char *)jpegd_out.yuvData,
            jpegd_out.imgWidthAligned, jpegd_out.imgHeightAligned, stride, result);

        cv::Mat bgr = cv::imread(path); 
        std::vector<mot::MOT_Track>::iterator riter;
        for (riter = result.begin(); riter != result.end(); riter++) {
            cv::Point pt1;
            std::deque<mot::MOT_Rect>::iterator iter;
            for (iter = riter->rects.begin(); iter != riter->rects.end(); iter++) {
                int l = static_cast<int>(iter->left);
                int t = static_cast<int>(iter->top);
                int r = static_cast<int>(iter->right);
                int b = static_cast<int>(iter->bottom);
                
                if (l == 0 && t == 0 && r == 0 && b == 0) {
                    break;
                }
                
                if (-1 != riter->identifier) {
                    srand(riter->identifier);
                } else {
                    srand(std::distance(result.begin(), riter));
                }
                
                cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
                if (iter != riter->rects.begin()) {
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
        
        std::string rpath = path;
        string_replace(rpath, std::string("imgs"), std::string("results"));        
        fprintf(stdout, "%s => %s: %lu\n", path.c_str(), rpath.c_str(), result.size());
        cv::imwrite(rpath, bgr);
    }

    mot::unload_mot_model();
    
    return 0;
}