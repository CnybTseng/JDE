#include <ctime>
#include <chrono>
#include <fstream>
#include <iostream>
#include "osnet.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage:\n\t./osnet-test "
            "engine_path [weight_path, loops, batch_size, ...]" << std::endl;
        exit(0);
    }
    
    std::string weight_path = "";
    if (argc >= 3) {
        weight_path = argv[2];
    }
    
    int loops = 1;
    if (argc >= 4) {
        loops = atoi(argv[3]);
    }
    
    std::shared_ptr<reid::OSNet> model = std::make_shared<reid::OSNet>();
    if (!model->init(argv[1], weight_path)) {
        std::cerr << "init OSNet fail" << std::endl;
        exit(-1);
    }
    
    int batch_size = model->get_max_batch_size();
    if (argc >= 5) {
        batch_size = std::min<int>(batch_size, atoi(argv[4]));
    }
    
    std::cout << "batch_size: " << batch_size << std::endl;
    mot::DimsX in_dims = model->get_input_dim();
    mot::DimsX out_dims = model->get_output_dim();
    std::shared_ptr<float> in(new float[in_dims.numel() * batch_size]);
    std::shared_ptr<float> out(new float[out_dims.numel() * batch_size]);
    
    // for precision validation
    srand(time(NULL));
    for (int i = 0; i < in_dims.numel() * batch_size; ++i) {
        in.get()[i] = (float)rand() / RAND_MAX;
    }    
    std::ofstream ofs("in.bin", std::ios::binary);
    ofs.write(reinterpret_cast<char*>(in.get()), in_dims.numel() * sizeof(float) * batch_size);
    ofs.close();
    
    if (loops > 1) {
        const int warmup = 100;
        for (int t = 0; t < warmup; ++t) {
            if (!model->forward(in, out, batch_size)) {
                std::cerr << "forward OSNet fail" << std::endl;
            }
        }
        std::cout << "warmup done." << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < loops; ++t) {
        if (!model->forward(in, out, batch_size)) {
            std::cerr << "forward OSNet fail" << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    float latency = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "latency is " << latency / loops << "ms" << std::endl;
    
    ofs = std::ofstream("out.txt");
    for (int i = 0; i < out_dims.numel() * batch_size; ++i) {
        ofs << out.get()[i] << " ";
        if ((i + 1) % 10 == 0) ofs << std::endl;
    }
    ofs.close();
    
    // for precision validation
    ofs = std::ofstream("out.bin", std::ios::binary);
    ofs.write(reinterpret_cast<char*>(out.get()), out_dims.numel() * sizeof(float) * batch_size);
    ofs.close();
    
    if (!model->deinit()) {
        std::cerr << "deinit OSNet fail" << std::endl;
        exit(-1);
    }
    
    return 0;
}