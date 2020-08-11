#include <cmath>
#include <cstdio>
#include <string>
#include <fstream>
#include <hiaiengine/data_type.h>
#include <Dvpp.h>
#include <c_graph.h>
#include <opencv2/opencv.hpp>

#include "dtype.h"
#include "errcode.h"
#include "input_engine.h"

InputEngine::InputEngine() : idvppapi(nullptr)
{
}

InputEngine::~InputEngine()
{
    HIAI_DVPP_DFree(output.get()->data.get());
    DestroyDvppApi(idvppapi);
}

HIAI_StatusT InputEngine::Init(const hiai::AIConfig &config_,
    const std::vector<hiai::AIModelDescription>& model_desc_)
{
    output = std::make_shared<Image>();
    output.get()->format = INPUT_YUV420_SEMI_PLANNER_UV;
    output.get()->width = 576;
    output.get()->height = 320;
    output.get()->channel = 3;
    output.get()->depth = 8;
    output.get()->ystride = 320;
    output.get()->xstride = 576;
    output.get()->size = output.get()->ystride * output.get()->xstride * 3 / 2;

    uint8_t *data = static_cast<uint8_t *>(HIAI_DVPP_DMalloc(output.get()->size));
    if (nullptr == data) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_DMALLOC_FAIL, "fail to malloc dvpp memory");
        return HIAI_DVPP_DMALLOC_FAIL;
    }
    
    size_t ysize = output.get()->ystride * output.get()->xstride;
    memset(data, 0x7E, ysize);
    memset(data + ysize, 0x80, ysize >> 1);
    
    output.get()->data.reset(data, HIAI_DVPP_DFree);

    int32_t ret = CreateDvppApi(idvppapi);
    if (0 != ret) {
        HIAI_ENGINE_LOG(this, HIAI_CREATE_DVPP_API_FAIL, "fail to create dvpp api");
        return HIAI_CREATE_DVPP_API_FAIL;
    }

    return HIAI_OK;
}

HIAI_StatusT InputEngine::ResizeKeepAspectRatio(std::shared_ptr<Image> im)
{    
    std::shared_ptr<VpcUserRoiConfigure> urc(new VpcUserRoiConfigure);
    urc->next = nullptr;
    
    VpcUserRoiInputConfigure *uric = &urc->inputConfigure;
    uric->cropArea.leftOffset = 0;
    uric->cropArea.rightOffset = im.get()->width - 1;
    uric->cropArea.upOffset = 0;
    uric->cropArea.downOffset = im.get()->height - 1;
    
    float sx = output.get()->width / static_cast<float>(im.get()->width);
    float sy = output.get()->height / static_cast<float>(im.get()->height);
    float s = std::min(sx, sy);
    uint32_t swidth = static_cast<uint32_t>(std::round(s * im.get()->width));
    uint32_t sheight = static_cast<uint32_t>(std::round(s * im.get()->height));
    uint32_t dx = (output.get()->width - swidth) >> 1;
    uint32_t dy = (output.get()->height - sheight) >> 1;

    VpcUserRoiOutputConfigure *uroc = &urc->outputConfigure;
    uroc->addr = output.get()->data.get();
    uroc->bufferSize = output.get()->size;
    uroc->widthStride = output.get()->xstride;
    uroc->heightStride = output.get()->ystride;
    uroc->outputArea.leftOffset = 0;
    uroc->outputArea.rightOffset = output.get()->width - 1;
    uroc->outputArea.upOffset = 0;
    uroc->outputArea.downOffset = output.get()->height - 1;
    
    std::shared_ptr<VpcUserImageConfigure> uic(new VpcUserImageConfigure);
    uic->bareDataAddr = im.get()->data.get();
    uic->bareDataBufferSize = im.get()->size;
    uic->widthStride = im.get()->xstride;
    uic->heightStride = im.get()->ystride;
    uic->inputFormat = im.get()->format;
    uic->outputFormat = OUTPUT_YUV420SP_UV;
    uic->yuvSumEnable = false;
    uic->cmdListBufferAddr = nullptr;
    uic->cmdListBufferSize = 0;    
    uic->roiConfigure = urc.get();
    
    dvppapi_ctl_msg dcmsg;
    dcmsg.in = static_cast<void *>(uic.get());
    dcmsg.in_size = sizeof(VpcUserImageConfigure);
    int32_t ret = DvppCtl(idvppapi, DVPP_CTL_VPC_PROC, &dcmsg);
    if (0 != ret) {
        HIAI_ENGINE_LOG(this, HIAI_DVPP_CTL_FAIL, "fail to execute dvpp ctl");
        return HIAI_DVPP_CTL_FAIL;
    }
    
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("InputEngine", InputEngine, INPUT_ENGINE_INPUT_SIZE)
{
    std::shared_ptr<Image> image = std::static_pointer_cast<Image>(arg0);
    if (nullptr == image) {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }
    
    ResizeKeepAspectRatio(image);   
    // {
    //     cv::Mat yuv420sp(output.get()->height * 3 / 2, output.get()->width, CV_8UC1,
    //         output.get()->data.get());
    //     cv::Mat bgr;
    //     cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);
    //     cv::imwrite("resize.png", bgr);
    // }
    
    std::shared_ptr<hiai::RawDataBuffer> rdbuf = std::make_shared<hiai::RawDataBuffer>();
    rdbuf->len_of_byte = output.get()->size;
    rdbuf->data = output.get()->data;
    
    hiai::Engine::SendData(0, "RawDataBuffer", std::static_pointer_cast<void>(rdbuf));

    return HIAI_OK;
}