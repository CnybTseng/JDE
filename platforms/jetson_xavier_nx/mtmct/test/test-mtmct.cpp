#include <chrono>
#include <string>
#include <thread>
#include <ostream>
#include <iostream>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "mtmct.h"

static bool et = false;
static std::condition_variable cvar;
static std::mutex mtx;

/**
 * @brief struct option.
 *  Options for the test program.
 */
struct option
{
    std::vector<unsigned char> cams;    // camera list
    std::map<unsigned char, std::string> urls;  // camera urls.
    friend std::ostream &operator<<(std::ostream &os, const option &opt)
    {
        for (auto item : opt.urls) {
            os << static_cast<int>(item.first) << " : " << item.second << "\n";
        }
        return os;
    }
};

/**
 * @brief Parse values from argv for specified option.
 * @warning After parsing, the parameter i will point to the next option,
 *  or is equal to parameter argc.
 * @param argc Number of arguments.
 * @param argv Argument array.
 * @param i The index of the specified option.
 * @param strs Parsed value strings.
 */
void parsevals(int argc, char *argv[], size_t &i, std::vector<std::string> &strs)
{
    strs.clear();
    while (++i < argc) {
        char *found = strstr(argv[i], "--");
        if (nullptr == found || found > argv[i]) {  // values not begin with "--"
            strs.emplace_back(argv[i]);
        } else {
            break;
        }
    }
}

/**
 * @brief Parse arguments from command line.
 */
bool parseargs(int argc, char *argv[], option &opt)
{
    for (size_t i = 1; i < argc;)
    {
        // Format like: --urls cam1 url1 cam2 url2 ...
        if (0 == strcmp(argv[i], "--urls")) {
            std::vector<std::string> strs;
            parsevals(argc, argv, i, strs);
            if (strs.size() < 2 || 0 != strs.size() % 2) {
                printf("the number of cameras is not equal to the number of urls!\n");
                return false;
            }
            for (size_t j = 0; j < strs.size(); j += 2) {
                opt.cams.emplace_back(atoi(strs[j].c_str()));
                opt.urls[opt.cams.back()] = strs[j + 1];
            }
        } else {
            printf("unrecognied argument: %s\n", argv[i]);
            break;
        }
    }
    return true;
}

/**
 * @brief Test calling algorithm::mtmct::init repeatedly from multiple threads.
 * @warning This is an error operation on module mtmct!
 */
void init(const std::vector<unsigned char> &cams)
{
    std::thread::id tid = std::this_thread::get_id();
    std::cout << "init by thread " << tid << ": " << algorithm::mtmct::inst()->init(cams);
}

/**
 * @brief Simulate push stream module.
 */
void send(unsigned char cam, option opt)
{
    std::cout << "create camera " << static_cast<int>(cam) << "\n";
    cv::VideoCapture vcap(opt.urls[cam]);
    bool opened = vcap.isOpened();
    std::cout << "camera " << static_cast<int>(cam) << " isOpened: " << std::boolalpha << opened << "\n";
    if (!opened) {
        return;
    }
    
    while (1) {
        cv::Mat mat;
        vcap >> mat;
        if (mat.empty()) {
            break;
        }
        // cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_NV12);
        if (!algorithm::mtmct::inst()->impush(cam, mat.data, mat.cols, mat.rows, mat.step)) {
            printf("send%u: failed\n", cam);
        }
    }
}

/**
 * @brief Simulate pull stream module.
 *  Only local trajectories will be processed here.
 */
void receive(unsigned char cam)
{
    ;
}

/**
 * @brief Global trajectories processing module.
 */
void recive()
{
    ;
}

/**
 * @brief Test calling algorithm::mtmct::deinit repeatedly from multiple threads.
 * @warning This is an error operation on module mtmct!
 */
void deinit()
{
    algorithm::mtmct::inst()->deinit();
}

int main(int argc, char *argv[])
{
    // Parse options from command line.
    option opt;
    if (!parseargs(argc, argv, opt)) {
        exit(0);
    }
    std::cout << "options:\n" << opt;

    // This is a correct calling for algorithm::mtmct::init
    if (!algorithm::mtmct::inst()->init(opt.cams)) {
        printf("init failed\n");
        exit(0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Switch between stop and run states.
    // printf("run: %d\n", algorithm::mtmct::inst()->run());
    // std::this_thread::sleep_for(std::chrono::seconds(3));
    // printf("stop: %d\n", algorithm::mtmct::inst()->stop());
    // std::this_thread::sleep_for(std::chrono::seconds(3));
    printf("run: %d\n", algorithm::mtmct::inst()->run());
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Creating and running pull stream threads.
    std::vector<std::thread> tss;
    for (auto url : opt.urls) {
        tss.emplace_back(send, url.first, opt);
    }
    for (size_t i = 0; i < tss.size(); ++i) {
        tss[i].join();
    }

    // Stop all algorithms of mtmct module.
    printf("stop: %d\n", algorithm::mtmct::inst()->stop());
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // This is a correct calling for algorithm::mtmct::deinit
    algorithm::mtmct::inst()->deinit();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    printf("test done.\n");
    exit(0);
}