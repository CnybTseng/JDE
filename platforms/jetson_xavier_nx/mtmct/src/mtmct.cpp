#include <map>
#include <mutex>
#include <chrono>
#include <thread>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <condition_variable>
#include "mtmct.h"
#include "tsque.hpp"

namespace algorithm {

#define BPP_BGR888 3   // bytes per pixel

/**
 * @brief Augmented tracklet structure.
 */
struct augtracklet : public tracklet
{
    unsigned int id;    // target identity
    std::shared_ptr<unsigned char> clip;    // target clip
    std::shared_ptr<float> embed;   // target embedding
};

/**
 * @brief struct image.
 * @warning Only supoort BGR888 or RGB888 format image.
 */
struct image
{
    image(const unsigned char *_data, short _width, short _height, size_t _stride)
    : width(_width), height(_height), stride(_stride)
    {
        assert(width > 0 && height > 0);
        assert(stride == 0 || stride >= BPP_BGR888 * width);
        if (0 == stride) {  // no padding for scan line
            stride = BPP_BGR888 * width;
        }
        size_t size = height * stride;
        data = std::shared_ptr<unsigned char>(new unsigned char[size], [](unsigned char *p){delete [] p;});
        memcpy(data.get(), _data, size);
    }
    std::shared_ptr<unsigned char> data;    // pixel data
    short width;    // horizontal resolution
    short height;   // vertical resolution
    size_t stride;  // byte stride of one scan line
};

/**
 * @brief class thread_wrapper.
 *  The wrapper for the std::thread class.
 */
class thread_wrapper : public std::thread
{
public:
    template<class Function, class ...Args>
    /**
     * @brief Create std::thread with exception handling.
     * @param t The thread need to be created.
     * @param f Callable object to execute in the new thread.
     * @param args Arguments to pass to the new function.
     * @return Return true if success, else return false.
     */
    bool create(std::thread &t, Function &&f, Args &&...args)
    {
        try {
            t = thread(f, args...);
            return true;
        } catch (const std::system_error &e) {
            std::cerr << "caught system error: " << e.code() << ", " << e.what() << "\n";
            return false;
        }
    }
};

/**
 * @brief enum threadstate.
 *  States of thread.
 */
enum threadstate
{
    NEW = 0,    // the thread just has been created
    RUNNING,    // running all algorithms
    STOP,       // stop all algorithms
    EXIT        // exit the thread
};

static std::once_flag init_flag;    // for mtmct::implementor::init()
static std::once_flag deinit_flag;  // for mtmct::implementor::deinit()
static threadstate ts = STOP;    // synchronization signal across threads
static std::condition_variable cv;  // cv for ts
static std::mutex mtx;  // the mutex for cv

/**
 * @brief class mtmct::implementor.
 * @detail This is the actual implementation for the mtmct algorithm.
 */
class mtmct::implementor
{
public:
    bool init(const std::vector<unsigned char> &cams, const char *config);
    bool camadd(unsigned char cam);
    bool camdel(unsigned char cam);
    bool run();
    bool impush(unsigned char cam, const unsigned char *data, short width, short height, size_t stride=0);
    bool tgtpop(std::vector<trajectory> &local, unsigned char cam);
    bool tgtpop(std::vector<trajectory> &global);
    bool stop();
    void deinit();
private:
    void initonce(const std::vector<unsigned char> &cams, const char *config, bool &flag);
    void deinitonce();
    void motexe(unsigned char cam);
    void mtmctexe();
    std::vector<unsigned char> cameras;
    std::map<unsigned char, tsque<std::shared_ptr<image>>> images;  // input images, shared between user and mot
    std::map<unsigned char, tsque<std::shared_ptr<std::vector<augtracklet>>>> tracklets; // local tracklet, shared between mot and mtmct
    std::map<unsigned char, std::vector<trajectory>> lltraj;    // local trajectories, shared between mtmct and user
    std::map<unsigned char, std::thread> motts;  // single camera multiple object tracking threads
    std::thread mtmctt; // multiple targets multiple camera tracking thread
    std::vector<trajectory> gltraj;    // global trajectories, shared between mtmct and user
};

//*******************************************************************
// Actual implementation for mtmct::init()
//*******************************************************************
bool mtmct::implementor::init(const std::vector<unsigned char> &cams, const char *config)
{
    bool flag = false;
    //! What a weird method to solve 'invalid use of non-static member function' error!
    //! https://blog.csdn.net/fengfengdiandia/article/details/82465987
    std::call_once(init_flag, &mtmct::implementor::initonce, this, cams, config, flag);
    return flag;
}

//*******************************************************************
// Thread function for mtmct::implementor::init()
//*******************************************************************
void mtmct::implementor::initonce(const std::vector<unsigned char> &cams, const char *config, bool &flag)
{
    printf("initonce\n");
    if (cams.empty()) {
        return;
    }
    cameras.assign(cams.begin(), cams.end());
    thread_wrapper tw;
    for (size_t i = 0; i < cams.size(); ++i) {
        images[cams[i]] = tsque<std::shared_ptr<image>>();
        tracklets[cams[i]] = tsque<std::shared_ptr<std::vector<augtracklet>>>();
        lltraj[cams[i]] = std::vector<trajectory>();
        if (!tw.create(motts[cams[i]], &mtmct::implementor::motexe, this, cams[i])) {
            return;
        }
        motts[cams[i]].detach();
    }
    if (!tw.create(mtmctt, &mtmct::implementor::mtmctexe, this)) {
        return;
    }
    mtmctt.detach();
    flag = true;
}

//*******************************************************************
// Actual implementation for mtmct::camadd()
//*******************************************************************
bool mtmct::implementor::camadd(unsigned char cam)
{
    if (std::find(cameras.begin(), cameras.end(), cam) != cameras.end()) {
        return false;
    }
    cameras.emplace_back(cam);
    images[cam] = tsque<std::shared_ptr<image>>();
    tracklets[cam] = tsque<std::shared_ptr<std::vector<augtracklet>>>();
    lltraj[cam] = std::vector<trajectory>();
    thread_wrapper tw;
    if (!tw.create(motts[cam], &mtmct::implementor::motexe, this, cam)) {
        return false;
    }
    motts[cam].detach();
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::camdel()
//*******************************************************************
bool mtmct::implementor::camdel(unsigned char cam)
{
    auto iter = std::find(cameras.begin(), cameras.end(), cam);
    if (iter == cameras.end()) {
        return false;
    }
    cameras.erase(iter);
    motts.erase(cam);
    images.erase(cam);
    tracklets.erase(cam);
    lltraj.erase(cam);
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::run()
//*******************************************************************
bool mtmct::implementor::run()
{
    std::lock_guard<std::mutex> lock(mtx);
    ts = RUNNING;
    cv.notify_all();
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::impush()
//*******************************************************************
bool mtmct::implementor::impush(unsigned char cam, const unsigned char *data, short width, short height, size_t stride)
{
    if (std::find(cameras.begin(), cameras.end(), cam) == cameras.end()) {
        return false;
    }
    std::shared_ptr<image> im = std::make_shared<image>(data, width, height, stride);
    images[cam].push(im);
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::tgtpop()
//*******************************************************************
bool mtmct::implementor::tgtpop(std::vector<trajectory> &local, unsigned char cam)
{
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::tgtpop()
//*******************************************************************
bool mtmct::implementor::tgtpop(std::vector<trajectory> &global)
{
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::stop()
//*******************************************************************
bool mtmct::implementor::stop()
{
    std::lock_guard<std::mutex> lock(mtx);
    ts = STOP;
    cv.notify_all();
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::deinit()
//*******************************************************************
void mtmct::implementor::deinit()
{
    std::call_once(deinit_flag, &mtmct::implementor::deinitonce, this);
}

//*******************************************************************
// Thread function for mtmct::implementor::deinit()
//*******************************************************************
void mtmct::implementor::deinitonce()
{
    printf("deinitonce\n");
    std::lock_guard<std::mutex> lock(mtx);
    ts = EXIT;
    cv.notify_all();
}

static void make_random_tracklet(std::vector<augtracklet> &tracks)
{
    
}

//*******************************************************************
// MOT thread execution function.
//*******************************************************************
void mtmct::implementor::motexe(unsigned char cam)
{
    printf("mot%u: create\n", cam);
    unsigned long counter = 0;
    while (1) {
        // Response to synchronized thread state.
        std::unique_lock<std::mutex> lock(mtx);
        if (ts == EXIT) {
            break;
        } else if (ts == STOP) {
            printf("mot%u: stop\n", cam);
            cv.wait(lock, []{return ts != STOP;});
            if (ts == EXIT) {
                break;
            }
            printf("mot%u: running\n", cam);
        }
        lock.unlock();
        // Get image from queue.
        std::shared_ptr<image> im;
        if (!images[cam].try_pop(im)) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));   // 1ms
            continue;
        }
        // printf("mot%u: get %lu images, %dx%d\n", cam, counter++, im.get()->width, im.get()->height);
        // TODO: Generate tracklets.
        std::vector<augtracklet> tracks;
        make_random_tracklet(tracks);
        // Push tracklets to queue.
        tracklets[cam].push(std::make_shared<std::vector<augtracklet>>(tracks));
        std::this_thread::sleep_for(std::chrono::microseconds(10000));  // 10ms
    }
    printf("mot%u: exit\n", cam);
}

//*******************************************************************
// MTMCT thread execution function.
//*******************************************************************
void mtmct::implementor::mtmctexe()
{
    printf("mtmct: create\n");
    while (1) {
        // Response to synchronized thread state.
        std::unique_lock<std::mutex> lock(mtx);
        if (ts == EXIT) {
            break;
        } else if (ts == STOP) {
            printf("mtmct: stop\n");
            cv.wait(lock, []{return ts != STOP;});
            if (ts == EXIT) {
                break;
            }
            printf("mtmct: running\n");
        }
        lock.unlock();
        // TODO: Get tracklets from queue.
        // TODO: Generate local trajectories.
        // TODO: Generate global trajectories.
        // TODO: push trajectories to queue.
        std::this_thread::sleep_for(std::chrono::microseconds(10000));  // 10ms
    }
    printf("mtmct: exit\n");
}

/**
 * @brief Class mtmct is nothing more than a decorator for the algorithm.
 */

//*******************************************************************
// Initialize the pointer to the mtmct self.
//*******************************************************************
mtmct *mtmct::self = nullptr;

//*******************************************************************
// Get the instance of class mtmct.
//*******************************************************************
mtmct *mtmct::inst()
{
    if (!self) {
        self = new mtmct;
    }
    return self;
}

//*******************************************************************
// mtmct initialization.
//*******************************************************************
bool mtmct::init(const std::vector<unsigned char> &cams, const char *config)
{
    return impl->init(cams, config);
}

//*******************************************************************
// Add a camera to device list.
//*******************************************************************
bool mtmct::camadd(unsigned char cam)
{
    return impl->camadd(cam);
}

//*******************************************************************
// Delete a camera from device list.
//*******************************************************************
bool mtmct::camdel(unsigned char cam)
{
    return impl->camdel(cam);
}

//*******************************************************************
// mtmct running.
//*******************************************************************
bool mtmct::run()
{
    return impl->run();
}

//*******************************************************************
// Push image to the image queue.
//*******************************************************************
bool mtmct::impush(unsigned char cam, const unsigned char *data, short width, short height, size_t stride)
{
    return impl->impush(cam, data, width, height, stride);
}

//*******************************************************************
// Pop local targets from trajectory queue.
//*******************************************************************
bool mtmct::tgtpop(std::vector<trajectory> &local, unsigned char cam)
{
    return impl->tgtpop(local, cam);
}

//*******************************************************************
// Pop global targets from trajectory queue.
//*******************************************************************
bool mtmct::tgtpop(std::vector<trajectory> &global)
{
    return impl->tgtpop(global);
}

//*******************************************************************
// mtmct stop running.
//*******************************************************************
bool mtmct::stop()
{
    return impl->stop();
}

//*******************************************************************
// mtmct deinitialization.
//*******************************************************************
void mtmct::deinit()
{
    return impl->deinit();
}

//*******************************************************************
// mtmct constructor.
//*******************************************************************
mtmct::mtmct() : impl(new implementor)
{
}

//*******************************************************************
// mtmct destructor. 
// @warning Must be defined out of line in the implementation file.
//*******************************************************************
mtmct::~mtmct()
{
}

}   // namespace algorithm