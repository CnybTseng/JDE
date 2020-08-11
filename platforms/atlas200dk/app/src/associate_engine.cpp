#include <cstdio>
#include <unistd.h>
#include <queue>
#include <fstream>
#include <hiaiengine/api.h>
#include <opencv2/opencv.hpp>

#include "dtype.h"
#include "errcode.h"
#include "jdetracker.h"
#include "associate_engine.h"

HIAI_REGISTER_DATA_TYPE("TrackVector", TrackVector);

static void CorrectBBox(float *ltrb, int imw, int imh, int niw, int nih)
{
    int dx = 0;
    int dy = 0;
    float sx = 1.0f;
    float sy = 1.0f;
    float _niw = imw;
    float _nih = imh;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    
    if (niw / nih < (float)imw / imh)
    {
        _nih = round(_niw * nih / niw);
    }
    else
    {
        _niw = round(_nih * niw / nih);
    }
    
    dx = (static_cast<int>(_niw) - imw) >> 1;
    dy = (static_cast<int>(_nih) - imh) >> 1;
    
    sx = _niw / niw;
    sy = _nih / nih;
    
    x1 = static_cast<int>(sx * ltrb[0] - dx + .5f);
    y1 = static_cast<int>(sy * ltrb[1] - dy + .5f);
    x2 = static_cast<int>(sx * ltrb[2] - dx + .5f);
    y2 = static_cast<int>(sy * ltrb[3] - dy + .5f);

    ltrb[0] = std::max<int>(x1, 0);
    ltrb[1] = std::max<int>(y1, 0);
    ltrb[2] = std::min<int>(x2, imw - 1);
    ltrb[3] = std::min<int>(y2, imh - 1);
}

HIAI_IMPL_ENGINE_PROCESS("AssociateEngine", AssociateEngine, ASSOCIATE_ENGINE_OUTPUT_SIZE)
{
    std::shared_ptr<DetectionVector> dets_ = \
        std::static_pointer_cast<DetectionVector>(arg0);
    if (nullptr == dets_) {
        HIAI_ENGINE_LOG("fail to receive messages");
        return HIAI_INVALID_INPUT_MSG;
    }

    size_t i = 0;
    std::vector<Detection>::iterator iter;
    cv::Mat dets(dets_.get()->data.size(), 6 + EMBD_DIM, CV_32FC1);
    for (iter = dets_.get()->data.begin(); iter != dets_.get()->data.end(); ++iter, ++i) {
        *dets.ptr<float>(i, 0) = iter->category;
        *dets.ptr<float>(i, 1) = iter->score;
        *dets.ptr<float>(i, 2) = iter->bbox.left;
        *dets.ptr<float>(i, 3) = iter->bbox.top;
        *dets.ptr<float>(i, 4) = iter->bbox.right;
        *dets.ptr<float>(i, 5) = iter->bbox.bottom;
        CorrectBBox(dets.ptr<float>(i) + 2, 1920, 1088, 576, 320);
        memcpy(dets.ptr<float>(i) + 6, iter->embedding, sizeof(iter->embedding));
    }

    std::shared_ptr<TrackVector> tracks = std::make_shared<TrackVector>();

#if ONLY_DETECTION
    for (int i = 0; i < dets.rows; ++i) {
        HTrack track = {
            .id = -1,
            .bbox = {
                .top = *dets.ptr<float>(i, 3),
                .left = *dets.ptr<float>(i, 2),
                .bottom = *dets.ptr<float>(i, 5),
                .right = *dets.ptr<float>(i, 4)
            }
        };
        tracks.get()->data.push_back(track);
    }
#else   // ONLY_DETECTION
    
    std::vector<mot::Track> tracks_;
    mot::JDETracker::instance()->update(dets, tracks_);
        
    for (size_t i = 0; i < tracks_.size(); ++i) {
        HTrack track = {
            .id = tracks_[i].id,
            .bbox = {
                .top = tracks_[i].ltrb[1],
                .left = tracks_[i].ltrb[0],
                .bottom = tracks_[i].ltrb[3],
                .right = tracks_[i].ltrb[2]
            }
        };
        tracks.get()->data.push_back(track);
    }
#endif  // ONLY_DETECTION
    
    hiai::Engine::SendData(0, "TrackVector", std::static_pointer_cast<void>(tracks));

    return HIAI_OK;
}