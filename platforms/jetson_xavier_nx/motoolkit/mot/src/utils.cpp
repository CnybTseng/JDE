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

}