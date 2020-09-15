#ifndef JDE_H_
#define JDE_H_

#include <memory>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

namespace mot {

// JDE的配置参数
struct JDEParams
{
    int batch_size{1};
    int dla_core{-1};
    bool int8{false};
    bool fp16{false};
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) {
        if (obj) {
            obj->destroy();
        }
    }
};

class JDE
{
    template <typename T>
    using UniquePtr = std::unique_ptr<T, InferDeleter>;
public:
    JDE(){};
    bool build();
    bool infer();
    bool teardown();
private:
    JDEParams params;
    nvinfer1::Dims input_dims;
    UniquePtr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    UniquePtr<nvinfer1::IExecutionContext> context;

    bool construct_network(UniquePtr<nvinfer1::IBuilder>& builder,
        UniquePtr<nvinfer1::INetworkDefinition>& network,
        UniquePtr<nvinfer1::IBuilderConfig>& config,
        UniquePtr<nvonnxparser::IParser>& parser);
};

}   // namespace mot

#endif