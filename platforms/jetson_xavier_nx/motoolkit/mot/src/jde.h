#ifndef JDE_H_
#define JDE_H_

#include <vector>
#include <memory>
#include <NvInfer.h>
#include "utils.h"

#define NUM_BINDINGS 4

namespace mot {

class JDE
{
public:
    static JDE* instance(void);
    JDE(void) {};
    ~JDE() {};
    bool init(void);
    bool infer(std::shared_ptr<float> in, std::vector<std::shared_ptr<float>>& out);
    bool destroy(void);
    DimsX get_binding_dims(int index);
    DimsX get_binding_dims(int index) const;
    const void* const get_binding(int index);
    const void* const get_binding(int index) const;
private:
    static JDE* me;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void *bindings[NUM_BINDINGS];
    DimsX binding_dims[NUM_BINDINGS];
    bool create_network_from_scratch(void);
    bool create_network_from_parser(void);
};

}   // namespace mot

#endif