#include <cstdio>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <arm_neon.h>
#include <hiaiengine/data_type.h>

#include "dtype.h"
#include "errcode.h"
#include "jdecoder_engine.h"

HIAI_REGISTER_DATA_TYPE("IAITensorVector", IAITensorVector);
HIAI_REGISTER_DATA_TYPE("DetectionVector", DetectionVector);

static void DumpTensor(const std::shared_ptr<hiai::AISimpleTensor> &tensor)
{
    string path = "./test/data/" + std::to_string(tensor->GetSize()) + ".bin";
    std::ofstream ofs(path, std::ios::binary);
    for (uint32_t i = 0; i < tensor->GetSize(); ++i)
        ofs << ((char *)tensor->GetBuffer())[i];
    ofs.close();
}

static inline void SoftmaxInplace(float *X, size_t N)
{
    float max = -FLT_MAX;
    for (size_t i = 0; i < N; ++i) {
        if (X[i] > max)
            max = X[i];
    }
    
    float sum = 0.f;
    for (size_t i = 0; i < N; ++i) {
        X[i] = exp(X[i] - max);
        sum += X[i];
    }
    
    for (size_t i = 0; i < N; ++i)
        X[i] /= sum;
}

static inline void Normalize(float *X, size_t N, float *Y, float eps=1e-12)
{
    float ssum = 0.f;
    size_t i = 0;

#if __ARM_NEON && (!DISABLE_NEON)
    float *ptr1 = X;
    float32x4_t ssum0 = {0, 0, 0, 0};
    for (; i < N; i += 4, ptr1 += 4) {
        float32x4_t x = vld1q_f32(ptr1);
        vmlaq_f32(ssum0, x, x);
    }

    float32x2_t ssum1 = vadd_f32(vget_high_f32(ssum0), vget_low_f32(ssum0));
    ssum += vget_lane_f32(vpadd_f32(ssum1, ssum1), 0);
#endif    

    for (; i < N; ++i)
        ssum += X[i] * X[i];
    
    ssum = sqrt(ssum);
    float s = ssum > eps ? 1.f / ssum : 1.f / eps;
    i = 0;

#if __ARM_NEON && (!DISABLE_NEON)
    ptr1 = X;
    float *ptr2 = Y;
    for (; i < N; i += 4, ptr1 += 4, ptr2 += 4) {
        float32x4_t x = vld1q_f32(ptr1);
        float32x4_t y = vmulq_n_f32(x, s);
        vst1q_f32(ptr2, y);
    }
#endif
    
    for (; i < N; ++i)
        Y[i] = s * X[i];
}

static inline float CalcInterArea(const Detection& a, const Detection& b)
{
    if (a.bbox.top > b.bbox.bottom || a.bbox.bottom < b.bbox.top ||
        a.bbox.left > b.bbox.right || a.bbox.right < b.bbox.left)
        return 0.f;
    
    float w = std::min(a.bbox.right, b.bbox.right) - std::max(a.bbox.left, b.bbox.left);
    float h = std::min(a.bbox.bottom, b.bbox.bottom) - std::max(a.bbox.top, b.bbox.top);
    return w * h;
}

static void QsortDescentInplace(std::vector<Detection>& data, int left, int right)
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
    
    if (left <= j)
        QsortDescentInplace(data, left, j);
    
    if (right >= i)
        QsortDescentInplace(data, i, right);
}

static void QsortDescentInplace(std::vector<Detection>& data)
{
    if (data.empty())
        return;
    
    QsortDescentInplace(data, 0, static_cast<int>(data.size() - 1));
}

static void NonmaximumSuppression(const std::vector<Detection>& dets, std::vector<size_t>& keeps, float iou_thresh)
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
            float inters = CalcInterArea(deti, detj);
            float unionn = areas[i] + areas[keeps[j]] - inters;
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

JDEcoderEngine::JDEcoderEngine() : input_queue(JDECODER_ENGINE_INPUT_SIZE),
    num_classes(2), num_boxes(4), conf_thresh(0.5f), iou_thresh(0.4f),
    biases{6, 16, 8, 23, 11, 32, 16, 45, 21, 64, 30, 90, 43, 128,
           60, 180, 85, 255, 120, 360, 170, 420, 340, 320},
    masks{8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3},
    inwidth(576), inheight(320), strides{32, 16, 8}
{
    ;
}

int32_t JDEcoderEngine::DecodeOutput(const std::shared_ptr<hiai::AISimpleTensor> &tensor,
    int32_t i, std::vector<Detection> &dets)
{
    float *data = (float *)tensor->GetBuffer();    // NHWC
    const int32_t chan_per_box = 4 + num_classes;
    const int32_t embd_offset = num_boxes * chan_per_box;
    const size_t mask_offset = i * num_boxes;
    register int32_t stridei = strides[i];
    const int32_t w = inwidth / stridei;
    const int32_t h = inheight / stridei;
    const int32_t c = embd_offset + EMBD_DIM;

    for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
#pragma omp parallel for num_threads(4)
            for (int32_t j = 0; j < num_boxes; ++j) {
                float *px = data + j * chan_per_box;    // box center x
                float *py = px + 1;                     // box center y
                float *pw = py + 1;                     // box width
                float *ph = pw + 1;                     // box height
                float *pc = ph + 1;                     // class logits
                float *pe = pc + num_classes;           // embedding vector
                
                // a prior anchor parameters
                int32_t bias_index = static_cast<int>(masks[mask_offset + j]);
                const float biasw = biases[ bias_index << 1];
                const float biash = biases[(bias_index << 1) + 1];
                
                // class probabilities
                SoftmaxInplace(pc, num_classes);
                
                // argmax(class probabilities)
                int32_t category = 0;
                float score = -std::numeric_limits<float>::max();
                for (int z = 0; z < num_classes; ++z) {
                    if (pc[z] > score) {
                        score = pc[z];
                        category = z;
                    }
                }
                
                // decode box and embedding
                if (category != 0 && score > conf_thresh) {
                    float bx = px[0] * biasw + x * stridei;
                    float by = py[0] * biash + y * stridei;
                    float bw = static_cast<float>(biasw * expf(pw[0]));
                    float bh = static_cast<float>(biash * expf(ph[0]));
                    
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
                    
                    Normalize(pe, EMBD_DIM, det.embedding);
                    dets.push_back(det);
                }
            }
            data += c;
        }
    }   

    return 0;
}

HIAI_IMPL_ENGINE_PROCESS("JDEcoderEngine", JDEcoderEngine, JDECODER_ENGINE_INPUT_SIZE)
{
    input_queue.PushData(0, arg0);
    std::shared_ptr<IAITensorVector> outputs = \
        std::static_pointer_cast<IAITensorVector>(arg0);
    
    if (!input_queue.PopAllData(outputs)) {
        HIAI_ENGINE_LOG("fail to get all messages");
        return HIAI_INVALID_INPUT_MSG;
    }
    
    std::shared_ptr<DetectionVector> dets = std::make_shared<DetectionVector>();
    for (size_t i = 0; i < outputs.get()->size(); ++i) {
        std::shared_ptr<hiai::AISimpleTensor> output = \
            std::static_pointer_cast<hiai::AISimpleTensor>((*outputs.get())[i]);
        // DumpTensor(output);
        DecodeOutput(output, i, dets.get()->data);
    }
    
    QsortDescentInplace(dets.get()->data);    
    std::vector<size_t> keeps;
    NonmaximumSuppression(dets.get()->data, keeps, iou_thresh);
    
    int num_dets = static_cast<int>(keeps.size());
    std::shared_ptr<DetectionVector> nms_dets = std::make_shared<DetectionVector>();
    for (int i = 0; i < num_dets; ++i)
        nms_dets.get()->data.push_back(dets.get()->data[keeps[i]]);
    
    hiai::Engine::SendData(0, "DetectionVector", std::static_pointer_cast<void>(nms_dets));

    return HIAI_OK;
}