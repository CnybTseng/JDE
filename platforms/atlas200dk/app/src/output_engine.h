#ifndef OUTPUT_ENGINE_H_
#define OUTPUT_ENGINE_H_

#include <hiaiengine/api.h>
#include <hiaiengine/multitype_queue.h>

#define OUTPUT_ENGINE_INPUT_SIZE 1
#define OUTPUT_ENGINE_OUTPUT_SIZE 1

class OutputEngine : public hiai::Engine
{
public:
    OutputEngine();
    HIAI_DEFINE_PROCESS(OUTPUT_ENGINE_INPUT_SIZE, OUTPUT_ENGINE_OUTPUT_SIZE)
private:
    int32_t DecodeOutput(const std::shared_ptr<hiai::AISimpleTensor> &tensor,
        int32_t i, std::vector<Detection> &dets);
private:
    hiai::MultiTypeQueue input_queue;
    enum
    {
        NUM_OUTPUTS = 3,
        NUM_ANCHORS = 12
    };
    const int32_t num_classes;
    const int32_t num_boxes;
    float conf_thresh;
    float iou_thresh;
    const float biases[NUM_ANCHORS * 2];
    const int32_t masks[NUM_ANCHORS];
    const int32_t inwidth;
    const int32_t inheight;
    const int32_t strides[NUM_OUTPUTS];
};

#endif