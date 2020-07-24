#include <cstdio>
#include <unistd.h>
#include <fstream>
#include <hiaiengine/api.h>
#include <opencv2/opencv.hpp>

#include "dtype.h"
#include "errcode.h"

#define GRAPH_CONFIG_PATH "./test/config/graph.prototxt"
#define TEST_IMG_PATH "./test/data/000002.bin"

#define DEVICE_ID 0
#define GRAPH_ID 100
#define INPUT_ENGINE_ID 1000
#define INPUT_PORT_ID 0
#define OUTPUT_ENGINE_ID 1002
#define OUTPUT_PORT_ID 0

class JDEDataRecvInterface : public hiai::DataRecvInterface
{
public:
    JDEDataRecvInterface() = default;
    ~JDEDataRecvInterface() = default;
    HIAI_StatusT RecvData(const std::shared_ptr<void> &message)
    {
        std::shared_ptr<DetectionVector> dets = \
            std::static_pointer_cast<DetectionVector>(message);
        if (nullptr == dets) {
            HIAI_ENGINE_LOG("fail to receive messages");
            return HIAI_INVALID_INPUT_MSG;
        }
        
        std::vector<cv::Mat> mats = {
            cv::Mat(320, 576, CV_32FC1),
            cv::Mat(320, 576, CV_32FC1),
            cv::Mat(320, 576, CV_32FC1)
        };
        
        std::ifstream istrm(TEST_IMG_PATH, std::ios::binary);
        if (!istrm.is_open()) {
            fprintf(stderr, "fail to open %s\n", TEST_IMG_PATH);
        } else {
            for (int32_t i = 0; i < mats.size(); ++i)
                istrm.read(reinterpret_cast<char *>(mats[i].data), mats[i].total() * sizeof(float));
            istrm.close();
        }
        
        cv::Mat rgb_norm;
        cv::merge(mats, rgb_norm);
        
        cv::Mat rgb;
        rgb_norm.convertTo(rgb, CV_8UC3, 255);
        
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        
        std::vector<Detection>::iterator iter;
        for (iter = dets.get()->data.begin(); iter != dets.get()->data.end(); ++iter) {            
            int32_t l = static_cast<int>(iter->bbox.left + 0.5f);
            int32_t t = static_cast<int>(iter->bbox.top + 0.5f);
            int32_t r = static_cast<int>(iter->bbox.right + 0.5f);
            int32_t b = static_cast<int>(iter->bbox.bottom + 0.5f);
            cv::rectangle(bgr, cv::Point(l, t), cv::Point(r, b), cv::Scalar(0, 255, 255), 1);
        }
        
        cv::imwrite("result.png", bgr);
        
        return HIAI_OK;
    }    
};

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
    
    hiai::EnginePortID outcfg;
    outcfg.graph_id = GRAPH_ID;
    outcfg.engine_id = OUTPUT_ENGINE_ID;
    outcfg.port_id = OUTPUT_PORT_ID;
    graph->SetDataRecvFunctor(outcfg, std::shared_ptr<JDEDataRecvInterface>(
        new JDEDataRecvInterface()));
    
    return HIAI_OK;
}

int main(int argc, char *argv[])
{
    std::shared_ptr<hiai::Graph> graph;
    HIAI_StatusT ret = InitJDEGraph(graph);
    if (HIAI_OK != ret) {
        fprintf(stderr, "create JDE graph fail\n");
        return -1;
    }
    
    hiai::EnginePortID incfg;
    incfg.graph_id = GRAPH_ID;
    incfg.engine_id = INPUT_ENGINE_ID;
    incfg.port_id = INPUT_PORT_ID;
    
    std::shared_ptr<std::string> impath = \
        std::shared_ptr<std::string>(new std::string(TEST_IMG_PATH));
    graph->SendData(incfg, "string", std::static_pointer_cast<void>(impath));
    
    sleep(1);
    
    hiai::Graph::DestroyGraph(GRAPH_ID);
    
    return 0;
}