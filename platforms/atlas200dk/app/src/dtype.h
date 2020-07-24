#ifndef DTYPE_H_
#define DTYPE_H_

#include <hiaiengine/data_type_reg.h>

#define EMBD_DIM 512

typedef std::vector<std::shared_ptr<hiai::IAITensor>> IAITensorVector;

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

struct DetectionVector {
    std::vector<Detection> data;
};

template<class Archive>
void serialize(Archive &ar, IAITensorVector &data)
{
    ar(data);
}

template<class Archive>
void serialize(Archive &ar, Detection &data)
{
    ar(data.category,
        data.score,
        data.bbox.top,
        data.bbox.left,
        data.bbox.bottom,
        data.bbox.right,
        data.embedding);
}

template<class Archive>
void serialize(Archive &ar, DetectionVector &data)
{
    ar(data.data);
}

#endif