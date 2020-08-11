#ifndef DATASTORE_H_
#define DATASTORE_H_

#include <Dvpp.h>
#include <opencv2/opencv.hpp>

bool is_directory(const std::string path);

void read_path_files(const std::string path, std::vector<std::string> &fvec);

void bgr2_yuv420sp_nv12(const cv::Mat &bgr, cv::Mat &nv12);

char *read_binary_file(std::string path, size_t &size);

char *read_binary_file_for_dvpp(std::string path, size_t &size);

void string_replace(std::string &str, const std::string &old,
    const std::string &neww);

int jpeg_decode(const std::string &path, struct JpegdOut &jpegd_out);

#endif