#include <chrono>
#include <iostream>

#include <sys/time.h>

#include "jde.h"

int main(int argc, char *argv[])
{
    mot::JDE jde;
    bool status = jde.build();
    if (!status) {
        std::cout << "build JDE fail" << std::endl;
        return 0;
    }
    
    int loops = 1;
    if (argc > 1) {
        loops = atoi(argv[1]);
    }
    
    float latency = 0;
    // struct timeval t1, t2;
    // gettimeofday(&t1, NULL);
    auto start = std::chrono::system_clock::now();
    
    for (int i = 0; i < loops; ++i) {
        status = jde.infer();
        if (!status) {
            std::cout << "infer JDE fail" << std::endl;
            return 0;
        }
    }
    
    auto end = std::chrono::system_clock::now();
    latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // gettimeofday(&t2, NULL);
    // latency = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) * 0.001f;
    latency /= loops;
    std::cout << "latency is " << latency << "ms" << std::endl;
    
    status = jde.teardown();
    if (!status) {
        std::cout << "teardown JDE fail" << std::endl;
        return 0;
    }
    
    return 0;
}