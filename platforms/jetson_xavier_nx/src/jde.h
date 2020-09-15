#ifndef JDE_H_
#define JDE_H_

#include <NvInfer.h>

namespace mot {

class JDE
{
public:
    static JDE* instance(void);
    bool init(void);
    bool infer(float *in, float *out);
    bool destroy(void);
private:
    static JDE* me;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void *buffers[4];
    bool build_onnx_model(void);
    JDE(void) {};
    ~JDE() {};
};

}   // namespace mot

#endif