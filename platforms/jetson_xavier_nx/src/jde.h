#ifndef JDE_H_
#define JDE_H_

#include <vector>
#include <memory>
#include <NvInfer.h>

#define NUM_BINDINGS 2

namespace mot {

class DimsX : public nvinfer1::Dims
{
public:
    DimsX()
    {
        _numel = 0;
    };
    DimsX(nvinfer1::Dims dims) : nvinfer1::Dims(dims)
    {
        if (0 == nbDims) {
            _numel = 0;
        } else {
            _numel = 1;
            for (int i = 0; i < nbDims; ++i) {
                _numel *= d[i];
            }
        }
    };
    int32_t numel()
    {
        return _numel;
    }
    int32_t numel() const
    {
        return _numel;
    }
private:
    int32_t _numel;
};

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
    bool create_network_from_scratch_v2(void);
    bool create_network_from_parser(void);
};

}   // namespace mot

#endif