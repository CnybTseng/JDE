#include <cstdio>
#include <hiaiengine/data_type.h>

#include "dtype.h"
#include "errcode.h"
#include "jde_engine.h"

HIAI_REGISTER_DATA_TYPE("IAITensorVector", IAITensorVector);

HIAI_StatusT JDEngine::Init(const hiai::AIConfig &config_,
    const std::vector<hiai::AIModelDescription>& model_desc_)
{
    fprintf(stderr, "JDEngine init\n");
    hiai::AIStatus ret = hiai::SUCCESS;
    
    config.clear();
    for (auto item : config_.items())
        config[item.name()] = item.value();
    
    if (nullptr == ai_model_manager)
        ai_model_manager = std::make_shared<hiai::AIModelManager>();
    
    const char *model_path = config["model_path"].c_str();
    hiai::AIModelDescription model_desc;
    model_desc.set_path(model_path);
    model_desc.set_key("");
    
    std::vector<hiai::AIModelDescription> model_descs;
    model_descs.push_back(model_desc);
    
    ret = ai_model_manager->Init(config_, model_descs);
    if (hiai::SUCCESS != ret) {
        HIAI_ENGINE_LOG(this, HIAI_JDE_MANAGER_INIT_FAIL,
            "hiai ai model manager init fail");
        return HIAI_JDE_MANAGER_INIT_FAIL;
    }
    
    fprintf(stderr, "JDEngine init success\n");
    return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("JDEngine", JDEngine, JDE_ENGINE_INPUT_SIZE)
{
    fprintf(stderr, "JDEngine process start\n");
    HIAI_StatusT ret = HIAI_OK;
    
    std::shared_ptr<hiai::RawDataBuffer> rdbuf = \
        std::static_pointer_cast<hiai::RawDataBuffer>(arg0);
    if (nullptr == rdbuf) {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }
    
    std::shared_ptr<hiai::AINeuralNetworkBuffer> nnbuf = \
        std::shared_ptr<hiai::AINeuralNetworkBuffer>(new hiai::AINeuralNetworkBuffer());
    nnbuf->SetBuffer((void *)(rdbuf->data.get()), (uint32_t)rdbuf->len_of_byte, false);
    
    std::shared_ptr<hiai::IAITensor> input = \
        std::static_pointer_cast<hiai::IAITensor>(nnbuf);
    
    IAITensorVector inputs;
    inputs.push_back(input);
    
    std::shared_ptr<IAITensorVector> outputs = std::make_shared<IAITensorVector>();
    ret = ai_model_manager->CreateOutputTensor(inputs, *outputs.get());
    if (hiai::SUCCESS != ret) {
        HIAI_ENGINE_LOG(this, HIAI_JDE_CREATE_OUTPUT_FAIL, "fail to create output tensor");
        return HIAI_JDE_CREATE_OUTPUT_FAIL;
    }
    
    hiai::AIContext ai_ctx;
    ret = ai_model_manager->Process(ai_ctx, inputs, *outputs.get(), 0);
    if (hiai::SUCCESS != ret) {
        HIAI_ENGINE_LOG(this, HIAI_JDE_MANAGER_PROCESS_FAIL, "fail to process ai model manager");
        return HIAI_JDE_MANAGER_PROCESS_FAIL;
    }
    
    hiai::Engine::SendData(0, "IAITensorVector", static_pointer_cast<void>(outputs));
    
    fprintf(stderr, "JDEngine process success\n");
    return HIAI_OK;
}