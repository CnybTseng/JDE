#ifndef DTYPE_H_
#define DTYPE_H_

#include <Dvpp.h>
#include <c_graph.h>
#include <hiaiengine/data_type.h>
#include <hiaiengine/data_type_reg.h>

#define EMBD_DIM 512

typedef std::vector<std::shared_ptr<hiai::IAITensor>> IAITensorVector;

struct Image {
    VpcInputFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channel = 0;
    uint32_t depth = 0;
    uint32_t ystride = 0;
    uint32_t xstride = 0;
    uint32_t size = 0;
    std::shared_ptr<uint8_t> data;
};

struct Detection {
    int32_t category;
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

struct HTrack {
    int32_t id;
    struct {
        float top;
        float left;
        float bottom;
        float right;
    } bbox;
};

struct TrackVector {
    std::vector<HTrack> data;
};

template<class Archive>
void serialize(Archive &ar, IAITensorVector &data)
{
    ar(data);
}

template<class Archive>
void serialize(Archive &ar, Image &data)
{
    ar(data.format, data.width, data.height, data.channel,
        data.depth, data.ystride, data.xstride, data.size);
    if (data.size > 0 && nullptr == data.data.get())
        data.data.reset(static_cast<uint8_t *>(HIAI_DVPP_DMalloc(data.size)));
    
    ar(cereal::binary_data(data.data.get(), data.size));
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

template<class Archive>
void serialize(Archive &ar, HTrack &data)
{
    ar(data.id,
        data.bbox.top,
        data.bbox.left,
        data.bbox.bottom,
        data.bbox.right);
}

template<class Archive>
void serialize(Archive &ar, TrackVector &data)
{
    ar(data.data);
}

#endif