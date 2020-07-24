#ifndef INPUT_ENGINE_H_
#define INPUT_ENGINE_H_

#include <hiaiengine/api.h>
#include <hiaiengine/multitype_queue.h>

#define INPUT_ENGINE_INPUT_SIZE 1
#define INPUT_ENGINE_OUTPUT_SIZE 1

class InputEngine : public hiai::Engine
{
public:
    HIAI_DEFINE_PROCESS(INPUT_ENGINE_INPUT_SIZE, INPUT_ENGINE_OUTPUT_SIZE)
private:
    static char *ReadBinFile(std::shared_ptr<std::string> file_name_ptr, uint32_t *file_size);
};

#endif