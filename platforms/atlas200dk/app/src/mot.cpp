#include <cstdio>
#include <unistd.h>
#include <queue>
#include <fstream>
#include <Dvpp.h>
#include <hiaiengine/api.h>
#include <hiaiengine/data_type.h>
#include <opencv2/opencv.hpp>

#include "dtype.h"
#include "errcode.h"
#include "jdetracker.h"
#include "mot.h"

#define GRAPH_CONFIG_PATH       "./test/config/graph.prototxt"

#define DEVICE_ID               0
#define GRAPH_ID                100
#define INPUT_ENGINE_ID         1000
#define INPUT_PORT_ID           0
#define ASSOCIATE_ENGINE_ID     1003
#define ASSOCIATE_PORT_ID       0

HIAI_REGISTER_DATA_TYPE("Image", Image);

namespace mot {

class MOTDataRecvInterface : public hiai::DataRecvInterface
{
public:
    MOTDataRecvInterface() = default;
    ~MOTDataRecvInterface() = default;
    HIAI_StatusT RecvData(const std::shared_ptr<void> &message);
    static std::queue<std::vector<HTrack>> tracks_queue;
};

std::queue<std::vector<HTrack>> MOTDataRecvInterface::tracks_queue = std::queue<std::vector<HTrack>>();

HIAI_StatusT MOTDataRecvInterface::RecvData(const std::shared_ptr<void> &message)
{   
    std::shared_ptr<TrackVector> tracks = \
        std::static_pointer_cast<TrackVector>(message);
    tracks_queue.push(tracks.get()->data);
   
    return HIAI_OK;
}

HIAI_StatusT InitJDEGraph(std::shared_ptr<hiai::Graph> &graph)
{
    HIAI_StatusT status = HIAI_Init(DEVICE_ID);
    if (HIAI_OK != status) {
        fprintf(stderr, "HIAI_Init fail\n");
        return status;
    }
    
    status = hiai::Graph::CreateGraph(GRAPH_CONFIG_PATH);
    if (HIAI_OK != status) {
        fprintf(stderr, "CreateGraph fail\n");
        return status;
    }
    
    graph = hiai::Graph::GetInstance(GRAPH_ID);
    if (nullptr == graph) {
        fprintf(stderr, "GetInstance fail\n");
        return HIAI_ERROR;
    }
    
    hiai::EnginePortID tpc;
    tpc.graph_id = GRAPH_ID;
    tpc.engine_id = ASSOCIATE_ENGINE_ID;
    tpc.port_id = ASSOCIATE_PORT_ID;
    graph->SetDataRecvFunctor(tpc, std::shared_ptr<MOTDataRecvInterface>(
        new MOTDataRecvInterface()));
    
    return HIAI_OK;
}

static struct
{
    std::shared_ptr<hiai::Graph> graph;
    int traj_cache_len = 30;
    std::vector<std::string> categories;
} __model;

int load_mot_model(const char *cfg_path)
{
    HIAI_StatusT ret = InitJDEGraph(__model.graph);
    if (HIAI_OK != ret) {
        fprintf(stderr, "create JDE graph fail\n");
        return -1;
    }
    
    if (!JDETracker::instance()->init()) {
        fprintf(stderr, "JDETracker::instance init fail\n");
        return -2;
    }
    
    __model.categories.push_back("person");
    
    return 0;
}

int unload_mot_model()
{
    hiai::Graph::DestroyGraph(GRAPH_ID);
    JDETracker::instance()->free();
    
    return 0;
}

int forward_mot_model(const unsigned char *im, int width, int height, int stride, MOT_Result &result)
{
    hiai::EnginePortID tpc;
    tpc.graph_id = GRAPH_ID;
    tpc.engine_id = INPUT_ENGINE_ID;
    tpc.port_id = INPUT_PORT_ID;

    // package image and send it to the pre-preocessing engine
    std::shared_ptr<Image> image = std::make_shared<Image>();
    image.get()->format = INPUT_YUV420_SEMI_PLANNER_UV;
    image.get()->width = width;
    image.get()->height = height;
    image.get()->channel = 3;
    image.get()->depth = 8;
    image.get()->ystride = height;
    image.get()->xstride = width;
    image.get()->size = width * height * 3 / 2;
    image.get()->data.reset(const_cast<uint8_t *>(im), HIAI_DVPP_DFree);
    
    __model.graph->SendData(tpc, "Image", std::static_pointer_cast<void>(image));
    
    int tolerate = 10001;
    MOTDataRecvInterface recv;
    while (--tolerate) {
        if (recv.tracks_queue.size() <= 0) {
            usleep(1000);
            continue;
        }

        fprintf(stderr, "get tracks costs %d ms\n", (10001 - tolerate));
        std::vector<HTrack> &tracks = recv.tracks_queue.front();

#if ONLY_DETECTION
        result.clear();
#endif   // ONLY_DETECTION
        
        // update existing tracks with new positions
        std::vector<MOT_Track>::iterator riter;
        for (riter = result.begin(); riter != result.end();) {
            bool match = false;
            std::vector<HTrack>::iterator titer;
            for (titer = tracks.begin(); titer != tracks.end(); ) {
                if (riter->identifier == titer->id) {
                    MOT_Rect rect = {
                        .top = titer->bbox.top,
                        .left = titer->bbox.left,
                        .bottom = titer->bbox.bottom,
                        .right = titer->bbox.right};
                    riter->rects.push_front(rect);
                    riter->rects.pop_back();
                    titer = tracks.erase(titer);
                    match = true;
                }
                else
                    titer++;
            }
            if (match)
                riter++;
            else
                riter = result.erase(riter);
        }

        // insert new tracks into tracking result
        for (size_t i = 0; i < tracks.size(); ++i) {
            MOT_Rect rect = {
                .top = tracks[i].bbox.top,
                .left = tracks[i].bbox.left,
                .bottom = tracks[i].bbox.bottom,
                .right = tracks[i].bbox.right};
            MOT_Track track = {
                .identifier = tracks[i].id,
                .posture = STANDING,
                .category = __model.categories[0]};
            track.rects.resize(__model.traj_cache_len - 1);
            track.rects.push_front(rect);
            result.push_back(track);
        }

        recv.tracks_queue.pop();
        break;
    }
    
    return 0;
}

}   // namespace mot