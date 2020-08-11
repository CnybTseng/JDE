#ifndef ASSOCIATE_ENGINE_H_

#include <hiaiengine/api.h>

#define ASSOCIATE_ENGINE_INPUT_SIZE  1
#define ASSOCIATE_ENGINE_OUTPUT_SIZE 1

class AssociateEngine : public hiai::Engine
{
public:
    HIAI_DEFINE_PROCESS(ASSOCIATE_ENGINE_INPUT_SIZE, ASSOCIATE_ENGINE_OUTPUT_SIZE)
};

#endif