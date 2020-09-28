#include <ctime>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <sys/time.h>

#include "jde.h"

int main(int argc, char *argv[])
{
    // Initialize JDE instance.
    bool status = mot::JDE::instance()->init();
    if (!status) {
        std::cout << "build JDE fail" << std::endl;
        return 0;
    }
    
    int loops = 1;
    if (argc > 1) {
        loops = atoi(argv[1]);
    }
    
    // Allocate input buffer.
    mot::DimsX dims0 = mot::JDE::instance()->get_binding_dims(0);
    std::shared_ptr<float> in(new float[dims0.numel()]);
    
    // Allocate output buffer.
    mot::DimsX dims1 = mot::JDE::instance()->get_binding_dims(1);
    std::vector<std::shared_ptr<float>> out(3);
    for (int i = 0; i < out.size(); ++i) {
        mot::DimsX dims = mot::JDE::instance()->get_binding_dims(i + 1);
        out[i] = std::shared_ptr<float>(new float[dims.numel()]);
    }
    
    srand(time(NULL));
    for (int i = 0; i < dims0.numel(); ++i) {
        in.get()[i] = (float)rand() / RAND_MAX;
    }
    
    // Saving input for comparing with pytorch baseline
    std::ofstream ofs("in.bin", std::ios::binary);
    ofs.write(reinterpret_cast<char*>(in.get()), dims0.numel() * sizeof(float));
    ofs.close();
    
    float latency = 0;    
    for (int i = 0; i < loops; ++i) {
#ifndef PROFILE
        auto start = std::chrono::system_clock::now();
        status = mot::JDE::instance()->infer(in, out);
#endif
        if (!status) {
            std::cout << "infer JDE fail" << std::endl;
            return 0;
        }
#ifndef PROFILE
        auto end = std::chrono::system_clock::now();
        latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "latency is " << latency << "ms" << std::endl;
#endif
        // Save output for comparing with pytorch baseline
        if (0 == i) {
            std::ofstream ofs("out.bin", std::ios::binary);
            ofs.write(reinterpret_cast<char*>(out[0].get()), dims1.numel() * sizeof(float));
            ofs.close();
        }
    }   
    
    status = mot::JDE::instance()->destroy();
    if (!status) {
        std::cout << "destroy JDE fail" << std::endl;
        return 0;
    }
    
    return 0;
}