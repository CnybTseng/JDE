#ifndef INPUT_ENGINE_H_
#define INPUT_ENGINE_H_

#include <hiaiengine/api.h>
#include <hiaiengine/multitype_queue.h>

#include "dtype.h"

#define INPUT_ENGINE_INPUT_SIZE 1
#define INPUT_ENGINE_OUTPUT_SIZE 1

class InputEngine : public hiai::Engine
{
public:
    InputEngine();
    virtual ~InputEngine();
    HIAI_StatusT Init(const hiai::AIConfig &config_,
        const std::vector<hiai::AIModelDescription>& model_desc_);
    HIAI_DEFINE_PROCESS(INPUT_ENGINE_INPUT_SIZE, INPUT_ENGINE_OUTPUT_SIZE)
private:
    HIAI_StatusT ResizeKeepAspectRatio(std::shared_ptr<Image> im);
private:
    std::shared_ptr<Image> output;
    IDVPPAPI *idvppapi;
};

#endif