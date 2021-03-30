#include <thread>
#include <cstring>
#include <fstream>
#include <iostream>
#include "tsque.hpp"

template <typename T>
struct arrdel
{
    void operator()(T const *p)
    {
        delete [] p;
    }
};

struct image
{
    image() {}
    image(const unsigned char *_data, int _width, int _height, int _stride)
    : width(_width)
    , height(_height)
    , stride(_stride)
    {
        size_t size = _stride * _height;
        data = std::shared_ptr<unsigned char>(new unsigned char[size], arrdel<unsigned char>());
        if (_data) {
            std:memcpy(data.get(), _data, size);
        }
    };
    int width;
    int height;
    int stride;
    std::shared_ptr<unsigned char> data;
};

void producer(algorithm::tsque<std::shared_ptr<image>> &que)
{
    std::ofstream ofs("send.txt");
    for (int t = 0; t < 100; ++t) {
        unsigned char data[8];
        for (int i = 0; i < 8; ++i) {
            data[i] = (unsigned char)(((float)rand() / RAND_MAX) * 255);
            ofs << (int)data[i] << " ";
        }
        ofs << "\n";
        std::shared_ptr<image> im = std::make_shared<image>(data, 8, 1, 8);
        que.push(im);
    }
    ofs.close();
}

void consumer(algorithm::tsque<std::shared_ptr<image>> &que)
{
    std::ofstream ofs("recv.txt");
    for (int t = 0; t < 200; ++t) {
        std::shared_ptr<image> im;
        if (que.try_pop(im)) {
            unsigned char *p =  reinterpret_cast<unsigned char *>(im.get()->data.get());
            for (int i = 0; i < im.get()->width; ++i) {
                ofs << (int)p[i] << " ";
            }
            ofs << "\n";
        } else {
            std::cout << "pop failed" << "\n";
        }
    }
    ofs.close();
}

int main(int argc, char *argv[])
{
    algorithm::tsque<std::shared_ptr<image>> que;
    std::thread thread1(producer, std::ref(que));
    std::thread thread2(consumer, std::ref(que));
    thread1.join();
    thread2.join();
    exit(0);
}