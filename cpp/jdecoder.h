#ifndef LAYER_JDECODER_H
#define LAYER_JDECODER_H

#include "layer.h"

namespace ncnn {

class JDEcoder : public Layer
{
public:
    JDEcoder();
    ~JDEcoder();
    virtual int load_param(const ParamDict& pd);
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
public:
    int num_classes;
    int num_boxes;
    float conf_thresh;
    float iou_thresh;
    Mat biases;
    Mat masks;
    Mat strides;
    int num_mask_groups;
    ncnn::Layer* softmax;
    ncnn::Layer* permute;
};

}   // namespace ncnn

#endif  // LAYER_JDECODER_H