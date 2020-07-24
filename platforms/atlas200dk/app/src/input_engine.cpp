#include <cstdio>
#include <string>
#include <fstream>
#include <hiaiengine/data_type.h>

#include "errcode.h"
#include "input_engine.h"

static void DeleteMemoryNew(void *ptr)
{
    if (nullptr != ptr)
        delete [] reinterpret_cast<char *>(ptr);
}

char *InputEngine::ReadBinFile(std::shared_ptr<std::string> file_name, 
    uint32_t *file_size)
{
    std::filebuf *pbuf;
    std::ifstream filestr;
    size_t size;
    filestr.open(file_name->c_str(), std::ios::binary);
    if (!filestr)
        return NULL;

    pbuf = filestr.rdbuf();
    size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    pbuf->pubseekpos(0, std::ios::in);
    char *buffer = new char[size];
    if (NULL == buffer)
        return NULL;

    pbuf->sgetn(buffer, size);
    *file_size = size;

    filestr.close();
    return buffer;
}

HIAI_IMPL_ENGINE_PROCESS("InputEngine", InputEngine, INPUT_ENGINE_INPUT_SIZE)
{
    fprintf(stderr, "InputEngine process start\n");
    std::shared_ptr<std::string> filename = \
        std::static_pointer_cast<std::string>(arg0);
    if (nullptr == filename) {
        HIAI_ENGINE_LOG(this, HIAI_INVALID_INPUT_MSG, "fail to process invalid message");
        return HIAI_INVALID_INPUT_MSG;
    }
    
    uint32_t fsize = 0;
    char *fdata = InputEngine::ReadBinFile(filename, &fsize);
    
    std::shared_ptr<hiai::RawDataBuffer> rdbuf = \
        std::make_shared<hiai::RawDataBuffer>();
    rdbuf->len_of_byte = fsize;
    rdbuf->data.reset((unsigned char *)fdata, DeleteMemoryNew);
    
    hiai::Engine::SendData(0, "RawDataBuffer",
        std::static_pointer_cast<void>(rdbuf));
    
    fprintf(stderr, "InputEngine process success\n");
    return HIAI_OK;
}