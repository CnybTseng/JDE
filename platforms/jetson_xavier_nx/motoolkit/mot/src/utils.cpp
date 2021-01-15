#include <iostream>
#include <cstring>
#include <dirent.h>
#include "utils.h"

namespace mot {

SimpleProfiler profiler("mot");

// 重载cout打印nvinfer1::Dims型变量
std::ostream& operator<<(std::ostream& os, nvinfer1::Dims dims)
{
    for (int i = 0; i < dims.nbDims - 1; ++i) {
        os << dims.d[i] << "x";
    }
    os << dims.d[dims.nbDims - 1];
    return os;
}

bool read_file_list(const char *dirname, std::vector<std::string> &filenames)
{
    DIR *dirp = opendir(dirname);
    if (nullptr == dirp) {
        std::cerr << "opendir(" << dirname << ") fail" << std::endl;
        return false;
    }
    
    struct dirent *dent = nullptr;
    while (nullptr != (dent = readdir(dirp))) {
        if (0 == strcmp(dent->d_name, ".") ||
            0 == strcmp(dent->d_name, "..")) {
            continue;
        }
        
        filenames.emplace_back(dent->d_name);
    }
    
    closedir(dirp);
    return true;
}

}