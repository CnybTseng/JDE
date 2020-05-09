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
    float score;
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
    
    ncnn::ParamDict pd1;
    pd1.set(0, 0);
    softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
    softmax->load_param(pd1);
    
    ncnn::ParamDict pd2;
    pd2.set(0, 3);
    permute = ncnn::create_layer(ncnn::LayerType::Permute);
    permute->load_param(pd2);
    
    ncnn::ParamDict pd3;
    pd3.set(0, 1);
    pd3.set(4, 1);
    pd3.set(1, 0);
    pd3.set(2, 0.0001f);
    pd3.set(9, 1);
    pd3.set(3, 1);
    normalize = ncnn::create_layer(ncnn::LayerType::Normalize);
    normalize->load_param(pd3);
    Mat scale[1];
    scale[0] = Mat(1);
    scale[0][0] = 1.f;
    normalize->load_model(ModelBinFromMatArray(scale));
}

JDEcoder::~JDEcoder()
{
    delete softmax;
    delete permute;
    delete normalize;
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

static inline float calc_inter_area(const Detection& a, const Detection& b)
{
    if (a.bbox.top > b.bbox.bottom || a.bbox.bottom < b.bbox.top ||
        a.bbox.left > b.bbox.right || a.bbox.right < b.bbox.left)
        return 0.f;
    
    float w = std::min(a.bbox.right, b.bbox.right) - std::max(a.bbox.left, b.bbox.left);
    float h = std::min(a.bbox.bottom, b.bbox.bottom) - std::max(a.bbox.top, b.bbox.top);
    return w * h;
}

static void qsort_descent_inplace(std::vector<Detection>& data, int left, int right)
{
    int i = left;
    int j = right;
    float pivot = data[(left + right) >> 1].score;
    while (i <= j)
    {
        while (data[i].score > pivot)
            ++i;
        
        while (data[j].score < pivot)
            --j;
        
        if (i <= j)
        {
            std::swap(data[i], data[j]);
            ++i;
            --j;
        }
    }
    
    if (left < j)
        qsort_descent_inplace(data, left, j);
    
    if (right > i)
        qsort_descent_inplace(data, i, right);
}

static void qsort_descent_inplace(std::vector<Detection>& data)
{
    if (data.empty())
        return;
    
    qsort_descent_inplace(data, 0, static_cast<int>(data.size() - 1));
}

static void nonmaximum_suppression(const std::vector<Detection>& dets, std::vector<size_t>& keeps, float iou_thresh)
{
    keeps.clear();
    const size_t n = dets.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
    {
        float w = dets[i].bbox.right - dets[i].bbox.left;
        float h = dets[i].bbox.bottom - dets[i].bbox.top;
        areas[i] = w * h;
    }
    
    for (size_t i = 0; i < n; ++i)
    {
        const Detection& deti = dets[i];
        int flag = 1;
        for (size_t j = 0; j < keeps.size(); ++j)
        {
            const Detection& detj = dets[keeps[j]];
            float inters = calc_inter_area(deti, detj);
            float unionn = areas[i] + areas[j] - inters;
            if (inters / unionn > iou_thresh)
            {
                flag = 0;
                break;
            }
        }
        
        if (flag)
            keeps.push_back(i);
    }
}

int JDEcoder::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    std::vector<Detection> dets;
    for (size_t i = 0; i < bottom_blobs.size(); ++i)
    {        
        const Mat& bottom_blobi = bottom_blobs[i];        
        int w = bottom_blobi.w;
        int h = bottom_blobi.h;
        int c = bottom_blobi.c;
        
        int embd_offset = num_boxes * (4 + 1 + num_classes);
        if (c != embd_offset + EMBD_DIM)
            return -1;

        Mat embeddings;
        permute->forward(bottom_blobi.channel_range(embd_offset, EMBD_DIM), embeddings, opt);
        
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
                    
                    if (category != 0 && score > conf_thresh)
                    {
                        float bx = (x + px[0] * biasw) * strides[i];
                        float by = (y + py[0] * biash) * strides[i];
                        float bw = static_cast<float>(strides[i] * exp(pw[0]));
                        float bh = static_cast<float>(strides[i] * exp(ph[0]));

                        Detection det = {
                            .category = category,
                            .score = score,
                            .bbox = {
                                .top = by - bh * 0.5f,
                                .left = bx - bw * 0.5f,
                                .bottom = by + bh * 0.5f,
                                .right = bx + bw * 0.5f
                            },
                            .embedding = {0}
                        };
                        
                        Mat embedding(EMBD_DIM, embeddings.channel(y).row(x));
                        normalize->forward_inplace(embedding, opt);
                        memcpy(det.embedding, embedding.data, EMBD_DIM * embedding.elemsize);
                        dets.push_back(det);
                    }
                    
                    ++px;
                    ++py;
                    ++pw;
                    ++ph;
                }
            }
        }
    }
    
    qsort_descent_inplace(dets);
    
    std::vector<size_t> keeps;
    nonmaximum_suppression(dets, keeps, iou_thresh);
    int num_dets = static_cast<int>(keeps.size());
    if (num_dets == 0)
        return 0;
    
    Mat& top_blob = top_blobs[0];
    top_blob.create(6 + EMBD_DIM, num_dets, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    
    for (int i = 0; i < num_dets; ++i)
    {
        Detection& det = dets[keeps[i]];
        float* ptr = top_blob.row(i);
        ptr[0] = static_cast<float>(det.category + 1);
        ptr[1] = det.score;
        ptr[2] = det.bbox.top;
        ptr[3] = det.bbox.left;
        ptr[4] = det.bbox.bottom;
        ptr[5] = det.bbox.right;
        memcpy(ptr + 6, det.embedding, sizeof(det.embedding));
    }
    
    return 0;
}

}   // namespace ncnn