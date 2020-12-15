#ifndef LOGGER_H_
#define LOGGER_H_

#include <cuda_runtime_api.h>

namespace mot {

// 创建全局日志对象
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} logger;

}   // namespace mot

#endif