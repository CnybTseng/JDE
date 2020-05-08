#include <algorithm>
#include <limits>
#include <math.h>

#include "layer_type.h"

#include "jdecoder.h"

#ifndef EMBD_DIM
#   define EMBD_DIM 512
#endif

namespace ncnn {

struct Detection {
    int category;
    struct {
        float top;
        float left;
        float bottom;
        float right;
    } bbox;
    float embedding[EMBD_DIM];
};

DEFINE_LAYER_CREATOR(JDEcoder)

JDEcoder::JDEcoder()
{
    one_blob_only = false;
    support_inplace = false;
    
    ncnn::ParamDict pd;
    pd.set(0, 3);
    permute->load_param(pd);
}

JDEcoder::~JDEcoder()
{
    
}

int JDEcoder::load_param(const ParamDict& pd)
{
    num_classes = pd.get(0, 1);
    num_boxes = pd.get(1, 4);
    conf_thresh = pd.get(2, 0.5f);
    iou_thresh = pd.get(3, 0.45f);
    biases = pd.get(4, Mat());
    masks = pd.get(5, Mat());
    strides = pd.get(6, Mat());
    return 0;
}

int JDEcoder::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    std::vector<std::vector<Detection>> dets;
    for (size_t i = 0; i < bottom_blobs.size(); ++i)
    {
        std::vector<Detection> deti;
        deti.resize(num_boxes);
        
        const Mat& bottom_blobi = bottom_blobs[i];        
        int w = bottom_blobi.w;
        int h = bottom_blobi.h;
        int c = bottom_blobi.c;
        
        int embd_offset = num_boxes * (4 + 1 + num_classes);
        if (c != embd_offset + EMBD_DIM)
            return -1;
        
        // chw to hwc
        Mat embedding;
        permute->forward(bottom_blobi.channel_range(embd_offset, EMBD_DIM), embedding);
        
        const int chan_per_box = embd_offset / num_boxes;
        size_t mask_offset = i * num_boxes;
        int netw = (int)(w * strides[i]);
        int neth = (int)(h * strides[i]);
        
        for (int j = 0; j < num_boxes; ++j)
        {
            int p = j * chan_per_box;
            const float* px = bottom_blobi.channel(p);
            const float* py = bottom_blobi.channel(p + 1);
            const float* pw = bottom_blobi.channel(p + 2);
            const float* ph = bottom_blobi.channel(p + 3);
            
            Mat scores = bottom_blobi.channel_range(p + 4, 1 + num_classes);
            softmax->forward_inplace(scores, opt);
            
            int bias_index = static_cast<int>(masks[mask_offset + j]);
            const float biasw = biases[ bias_index << 1];
            const float biash = biases[(bias_index << 1) + 1];
            
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    int category = 0;
                    float score = -std::numeric_limits<float>::max();
                    for (int z = 0; z < 1 + num_classes; ++z)
                    {
                        float scorez = scores.channel(z).row(y)[x];
                        if (scorez > score)
                        {
                            score = scorez;
                            category = z;
                        }
                    }
                    
                    if (score > conf_thresh)
                    {
                        float bx = (x + px[0] * biasw) * strides[i];
                        float by = (y + py[0] * biash) * strides[i];
                        float bw = static_cast<float>(strides[i] * exp(pw[0]));
                        float bh = static_cast<float>(strides[i] * exp(ph[0]));

                        Detection det = {
                            .category = category,
                            .bbox = {
                                .top = by - bh * 0.5f,
                                .left = bx - bw * 0.5f,
                                .bottom = by + bh * 0.5f,
                                .right = bx + bw * 0.5f
                            }
                        };
                        
                        memcpy(det.embedding, embedding.row(y)[x], EMBD_DIM * embedding.elemsize);
                        deti[i].push_back(det);
                    }
                    
                    ++px;
                    ++py;
                    ++pw;
                    ++ph;
                }
            }
        }
    }
    
    return 0;
}

}   // namespace ncnn