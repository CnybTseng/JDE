#include <chrono>
#include <iostream>

#include <sys/time.h>

#include "jde.h"

int main(int argc, char *argv[])
{
    bool status = mot::JDE::instance()->init();
    if (!status) {
        std::cout << "build JDE fail" << std::endl;
        return 0;
    }
    
    int loops = 1;
    if (argc > 1) {
        loops = atoi(argv[1]);
    }
    
    float in[320*576*3];
    float out[536*10*18];
    
    float latency = 0;    
    for (int i = 0; i < loops; ++i) {
        auto start = std::chrono::system_clock::now();
        status = mot::JDE::instance()->infer(in, out);
        if (!status) {
            std::cout << "infer JDE fail" << std::endl;
            return 0;
        }
        auto end = std::chrono::system_clock::now();
        latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        latency /= loops;
        std::cout << "latency is " << latency << "ms" << std::endl;
    }
    
    status = mot::JDE::instance()->destroy();
    if (!status) {
        std::cout << "destroy JDE fail" << std::endl;
        return 0;
    }
    
    return 0;
}