#ifndef JDE_ENGINE_H_
#define JDE_ENGINE_H_

#include <hiaiengine/api.h>
#include <hiaiengine/ai_model_manager.h>

#define JDE_ENGINE_INPUT_SIZE 1
#define JDE_ENGINE_OUTPUT_SIZE 1

class JDEngine : public hiai::Engine
{
public:
    HIAI_StatusT Init(const hiai::AIConfig &config_,
        const std::vector<hiai::AIModelDescription>& model_desc_);
    HIAI_DEFINE_PROCESS(JDE_ENGINE_INPUT_SIZE, JDE_ENGINE_OUTPUT_SIZE)
private:
    std::map<std::string, std::string> config;
    std::shared_ptr<hiai::AIModelManager> ai_model_manager;
};

#endif