#include <ctime>
#include <string>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <sys/time.h>

#include "jde.h"
#include "utils.h"

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
    std::vector<std::shared_ptr<float>> out(NUM_BINDINGS - 1);
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
        auto start = std::chrono::high_resolution_clock::now();
        status = mot::JDE::instance()->infer(in, out);
        if (!status) {
            std::cout << "infer JDE fail" << std::endl;
            return 0;
        }

        auto end = std::chrono::high_resolution_clock::now();
        latency += std::chrono::duration<float, std::milli>(end - start).count();

        // Saving output for comparing with pytorch baseline
        if (0 == i) {
            for (int i = 0; i < out.size(); ++i) {
                std::ofstream ofs("out" + std::to_string(i) + ".bin", std::ios::binary);
                mot::DimsX dims = mot::JDE::instance()->get_binding_dims(i + 1);
                ofs.write(reinterpret_cast<char*>(out[i].get()), dims.numel() * sizeof(float));
                ofs.close();
            }
        }
    }   
    
    std::cout << "latency is " << latency / loops << "ms" << std::endl;
    status = mot::JDE::instance()->destroy();
    if (!status) {
        std::cout << "destroy JDE fail" << std::endl;
        return 0;
    }
#if PROFILE    
    std::cout << std::endl << mot::profiler << std::endl;
#endif    
    return 0;
}