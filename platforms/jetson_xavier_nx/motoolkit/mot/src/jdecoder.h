#ifndef JDECODER_H_
#define JDECODER_H_

#include <vector>
#include <memory>
#include <cstdlib>

#define EMBD_DIM    128

namespace mot {

struct Detection {
    float category;
    float score;
    struct {
        float top;
        float left;
        float bottom;
        float right;
    } bbox;
    float embedding[EMBD_DIM];
}__attribute__((packed));

void QsortDescentInplace(std::vector<Detection>& data);
void NonmaximumSuppression(const std::vector<Detection>& dets, std::vector<size_t>& keeps, float iou_thresh);

class JDecoder
{
public:
    static JDecoder* instance(void);
    JDecoder();
    ~JDecoder() {};
    bool init(void);
    bool infer(std::vector<std::shared_ptr<float>>& in, std::vector<Detection>& dets);
    bool destroy(void);
private:
    static JDecoder* me;
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

}   // namespace mot

#endif