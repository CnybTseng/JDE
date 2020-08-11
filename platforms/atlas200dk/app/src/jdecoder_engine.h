#ifndef JDECODER_ENGINE_H_
#define JDECODER_ENGINE_H_

#include <hiaiengine/api.h>
#include <hiaiengine/multitype_queue.h>

#define JDECODER_ENGINE_INPUT_SIZE 1
#define JDECODER_ENGINE_OUTPUT_SIZE 1

class JDEcoderEngine : public hiai::Engine
{
public:
    JDEcoderEngine();
    HIAI_DEFINE_PROCESS(JDECODER_ENGINE_INPUT_SIZE, JDECODER_ENGINE_OUTPUT_SIZE)
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