#include <map>
#include <mutex>
#include <chrono>
#include <thread>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "mtmct.h"
#include "tsque.hpp"

namespace algorithm {

#define BPP_BGR888 3   // bytes per pixel

/**
 * @brief Augmented tracklet structure.
 * @warning clip and embed are arrays, be carefull when create shared pointer!
 */
struct augtracklet : public tracklet
{
    bool read;  // has been read by tgget or not
    unsigned int id;    // target identity
    std::shared_ptr<unsigned char> clip;    // target clip
    std::shared_ptr<float> embed;   // target embedding
};

/**
 * @brief typedef augtrajectory.
 */
typedef __trajectory<augtracklet> augtrajectory;

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
        unsigned char *ptr = ::new (std::nothrow) unsigned char[size];
        assert(nullptr != ptr);
        data = std::shared_ptr<unsigned char>(ptr, [](unsigned char *p){delete [] p;});
        memcpy(data.get(), _data, size);
    }
    std::shared_ptr<unsigned char> data;    // pixel data array
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
    STOP,       // stop all algorithms
    RUNNING,    // running all algorithms
    EXIT        // exit the thread
};

static std::once_flag init_flag;    // for mtmct::implementor::init()
static std::once_flag deinit_flag;  // for mtmct::implementor::deinit()
static threadstate ts = STOP;    // synchronization signal across threads
static std::condition_variable cv;  // cv for ts
static std::mutex mtx;  // the mutex for cv

/**
 * @brief Camera comparing function.
 */
typedef bool (*camcomp)(camera, camera);

static void make_random_tracklet(std::vector<augtracklet> &tracks);

/**
 * @brief Copy trajectories.
 * @param src The trajectory source.
 * @param dst The trajectory destination.
 * @param compare Camera ID comparing function.
 * @param cam Camera ID.
 * @return Return true if success, else return false.
 */
static bool trajcopy(std::vector<augtrajectory> &src, std::vector<trajectory> &dst, camcomp compare, camera cam=0);

/**
 * @brief class mtmct::implementor.
 * @detail This is the actual implementation for the mtmct algorithm.
 */
class mtmct::implementor
{
public:
    bool init(const std::vector<unsigned char> &cams, const char *config);
    bool camadd(camera cam);
    bool camdel(camera cam);
    bool run();
    bool impush(camera cam, const unsigned char *data, short width, short height, size_t stride=0);
    bool tgget(std::vector<trajectory> &local, camera cam);
    bool tgget(std::vector<trajectory> &global);
    bool stop();
    void deinit();
private:
    void initonce(const std::vector<camera> &cams, const char *config, bool &flag);
    void deinitonce();
    void mottf(camera cam);
    void mtmcttf();
    std::vector<camera> cameras;
    std::map<camera, tsque<std::shared_ptr<image>>> images;  // input images, shared between user and mot
    std::map<camera, tsque<std::shared_ptr<std::vector<augtracklet>>>> tracklets; // local tracklet, shared between mot and mtmct
    std::map<camera, std::vector<augtrajectory>> lltraj;    // local trajectories, shared between mtmct and user
    std::map<camera, std::mutex> llmtx;   // mutex for lltraj
    std::map<camera, std::thread> motts;  // single camera multiple object tracking threads
    std::thread mtmctt; // multiple targets multiple camera tracking thread
    std::vector<augtrajectory> gltraj;    // global trajectories, shared between mtmct and user
    std::mutex glmtx;   // mutex for glmtx
};

//*******************************************************************
// Actual implementation for mtmct::init()
//*******************************************************************
bool mtmct::implementor::init(const std::vector<camera> &cams, const char *config)
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
void mtmct::implementor::initonce(const std::vector<camera> &cams, const char *config, bool &flag)
{
    printf("initonce\n");
    if (cams.empty()) {
        return;
    }
    cameras.assign(cams.begin(), cams.end());
    thread_wrapper tw;
    for (size_t i = 0; i < cams.size(); ++i) {
        if (!tw.create(motts[cams[i]], &mtmct::implementor::mottf, this, cams[i])) {
            return;
        }
        motts[cams[i]].detach();
    }
    if (!tw.create(mtmctt, &mtmct::implementor::mtmcttf, this)) {
        return;
    }
    mtmctt.detach();
    flag = true;
}

//*******************************************************************
// Actual implementation for mtmct::camadd()
//*******************************************************************
bool mtmct::implementor::camadd(camera cam)
{
    if (std::find(cameras.begin(), cameras.end(), cam) != cameras.end()) {
        return false;
    }
    cameras.emplace_back(cam);
    thread_wrapper tw;
    if (!tw.create(motts[cam], &mtmct::implementor::mottf, this, cam)) {
        return false;
    }
    motts[cam].detach();
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::camdel()
//*******************************************************************
bool mtmct::implementor::camdel(camera cam)
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
bool mtmct::implementor::impush(camera cam, const unsigned char *data, short width, short height, size_t stride)
{
    if (std::find(cameras.begin(), cameras.end(), cam) == cameras.end()) {
        return false;
    }
    std::shared_ptr<image> im = std::make_shared<image>(data, width, height, stride);
    images[cam].push(im);
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::tgget()
//*******************************************************************
bool mtmct::implementor::tgget(std::vector<trajectory> &local, camera cam)
{
    if (std::find(cameras.begin(), cameras.end(), cam) == cameras.end()) {
        std::cerr << "The camera is not in device list\n";
        return false;
    }
    std::lock_guard<std::mutex> lock(llmtx[cam]);
    std::vector<augtrajectory> &traj = lltraj[cam];
    if (trajcopy(traj, local, [](camera a, camera b){return a == b;}, cam)) {
        std::cerr << "camera IDs are not matched\n";
        return false;
    }
    return true;
}

//*******************************************************************
// Actual implementation for mtmct::tgget()
//*******************************************************************
bool mtmct::implementor::tgget(std::vector<trajectory> &global)
{
    std::lock_guard<std::mutex> lock(glmtx);
    return trajcopy(gltraj, global, [](camera a, camera b){return true;});
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

//*******************************************************************
// MOT thread function.
//*******************************************************************
void mtmct::implementor::mottf(camera cam)
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
            printf("mot%u: running again\n", cam);
        }
        lock.unlock();
        // Get image from queue.
        std::shared_ptr<image> im;
        if (!images[cam].try_pop(im)) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));   // 1ms
            continue;
        }
        printf("mot%u: get %lu images, %dx%d\n", cam, counter++, im.get()->width, im.get()->height);
        // TODO: Generate local tracklets.
        std::vector<augtracklet> tracks;
        make_random_tracklet(tracks);
        // Push tracklets to queue.
        tracklets[cam].push(std::make_shared<std::vector<augtracklet>>(tracks));
        std::this_thread::sleep_for(std::chrono::microseconds(10000));  // 10ms
    }
    printf("mot%u: exit\n", cam);
}

//*******************************************************************
// MTMCT thread function.
//*******************************************************************
void mtmct::implementor::mtmcttf()
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
            printf("mtmct: running again\n");
        }
        lock.unlock();
        // TODO: Get tracklets from queue and extract better ReID feature in higher resolution.
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
bool mtmct::init(const std::vector<camera> &cams, const char *config)
{
    return impl->init(cams, config);
}

//*******************************************************************
// Add a camera to device list.
//*******************************************************************
bool mtmct::camadd(camera cam)
{
    return impl->camadd(cam);
}

//*******************************************************************
// Delete a camera from device list.
//*******************************************************************
bool mtmct::camdel(camera cam)
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
bool mtmct::impush(camera cam, const unsigned char *data, short width, short height, size_t stride)
{
    return impl->impush(cam, data, width, height, stride);
}

//*******************************************************************
// Get local targets from trajectory queue.
//*******************************************************************
bool mtmct::tgget(std::vector<trajectory> &local, camera cam)
{
    return impl->tgget(local, cam);
}

//*******************************************************************
// Get global targets from trajectory queue.
//*******************************************************************
bool mtmct::tgget(std::vector<trajectory> &global)
{
    return impl->tgget(global);
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

void make_random_tracklet(std::vector<augtracklet> &tracks)
{
    
}

//*******************************************************************
// Copy trajectories.
//*******************************************************************
bool trajcopy(std::vector<augtrajectory> &src, std::vector<trajectory> &dst, camcomp compare, camera cam)
{
    std::vector<bool> news(src.size());
    for (std::vector<trajectory>::size_type i = 0; i < src.size(); ++i) {
        news[i] = true;
    }
    // For exist trajectory, append new tracklet only.
    for (auto &to : dst) {
        if (!compare(to.cam, cam)) {
            return false;
        }
        for (std::vector<trajectory>::size_type i = 0; i < src.size(); ++i) {
            auto &from = src[i];
            if (to.id == from.id) {
                std::list<augtracklet> latests; // latests have not been read yet
                std::list<augtracklet>::reverse_iterator riter;
                for (riter = from.data.rbegin(); riter != from.data.rbegin(); ++riter) {
                    if (riter->read) {
                        break;
                    }
                    latests.emplace_front(*riter);
                    riter->read = true;
                }
                to.data.insert(to.data.end(), latests.begin(), latests.end());
                news[i] = false;
                break;
            }
        }
    }
    // For new trajectory, copy all tracklets.
    for (std::vector<augtrajectory>::size_type i = 0; i < src.size(); ++i) {
        if (!news[i]) {
            continue;
        }
        auto &from = src[i];
        std::list<tracklet> data;
        for (const auto &d : from.data) {
            data.emplace_back(d);   // cast augtracklet to tracklet
        }
        trajectory to = {from.cam, from.id, from.cate, data};
        dst.emplace_back(to);
        for (auto &d : from.data) {
            d.read = true;
        }
    }
    return true;
}

}   // namespace algorithm