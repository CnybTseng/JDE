#ifndef JDE_H_
#define JDE_H_

#include <NvInfer.h>

#define NUM_BINDINGS    4

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
    void *bindings[NUM_BINDINGS];
    size_t binding_sizes[NUM_BINDINGS];
    bool build_onnx_model(void);
    JDE(void) {};
    ~JDE() {};
};

}   // namespace mot

#endif