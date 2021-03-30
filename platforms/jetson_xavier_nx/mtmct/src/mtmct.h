/********************************************************************
 *
 * @file mtmct.h
 * @brief Multi-targets Multi-cameras tracking.
 * @author CnybTseng
 * @version 1.0.0
 * @date March 11, 2021
 *
 *******************************************************************/

#ifndef _MTMCT_H_
#define _MTMCT_H_

#include <list>
#include <ctime>
#include <memory>
#include <vector>

namespace algorithm {

/**
 * @brief struct tracklet.
 *  The local and global tracklet will share this structure.
 */
struct tracklet
{
    union
    {
        struct
        {
            short x;    // top left x of target in image coordinate system
            short y;    // top left y of target in image coordinate system
            short w;    // target width in image coordinate system
            short h;    // target height in image coordinate system
        } rect; // local tracklet location
        struct
        {
            int x;  // target x in world coordinate system
            int y;  // target y in world coordinate system
        } footprint;    // global tracklet location
    };  // location
    time_t time;    // time for detection
};

/**
 * @brief struct trajectory.
 *  The local and global trajectory will share this structure.
 */
struct trajectory
{
    unsigned char cam;  // camera identity, cam=0 indicates that it's a global trajectory
    unsigned int id;    // target identity
    std::string cate;   // target category
    std::list<tracklet> data;  // tracklets
};

/**
 * @brief class mtmct. This is a singleton class.
 */
class mtmct
{
public:
    /**
     * @brief Get the instance of class mtmct.
     *  You can call methods with algorithm::mtmct::inst()->???
     * @return Return the pointer to the instance.
     */
    static mtmct *inst();
    /**
     * @brief mtmct initialization. Execute only once (EOO).
     * @param cams The vector storing camera IDs. Currently
     *  the maximum number of support cameras is 255. The camera 0 is reserved.
     * @param config Configuration file path. If the parameter is missing
     *  or equal to nullptr, all default parameters will be used.
     * @return Return ture if success, else return false.
     */
    bool init(const std::vector<unsigned char> &cams, const char *config=nullptr);
    /**
     * @brief Add a camera to device list.
     * @warning The maximum number of support cameras is 255.
     *  The camera 0 is reserved.
     * @param cam The camera ID to be added.
     * @return Return ture if success, else return false.
     */
    bool camadd(unsigned char cam);
    /**
     * @brief Delete a camera from device list.
     * @param cam The camera ID to be deleted.
     * @return Return ture if success, else return false.
     */
    bool camdel(unsigned char cam);
    /**
     * @brief mtmct running.
     * @return Return ture if success, else return false.
     */
    bool run();
    /**
     * @brief Push image to the image queue.
     * @warning Only support BGR888 (BGR24) format image.
     *  The maximum size of support image is 32767x32767.
     * @param cam The camera ID for the image.
     * @param data The pointer to image data.
     * @param width The horizontal resolution of image.
     * @param height The vertical resolution of image.
     * @param stride Number of bytes each scanline occupies.
     *  The value should include the padding bytes at the end of each row, if any.
     *  If the parameter is missing or zero, no padding is assumed.
     * @return Return ture if success, else return false.
     */
    bool impush(unsigned char cam, const unsigned char *data, short width, short height, size_t stride=0);
    /**
     * @brief Pop local targets from trajectory queue.
     * @warning If the vector `local` is not empty, this function will update
     *  tracklet queue in fifo manner, it will be faster, the user must keep
     *  the vector alive all the time. If `local` is empty, this function will
     *  copy all tracklets from the beginning.
     * @param local Local trajectories.
     * @param cam Camera ID for trajectory retrieving.
     * @return Return ture if success, else return false.
     */
    bool tgtpop(std::vector<trajectory> &local, unsigned char cam);
    /**
     * @brief Pop global targets from trajectory queue.
     * @warning If the vector `global` is not empty, this function will update
     *  tracklet queue in fifo manner, it will be faster, the user must keep
     *  the vector alive all the time. If `global` is empty, this function will
     *  copy all tracklets from the beginning.
     * @param global Global trajectories.
     * @return Return ture if success, else return false.
     */
    bool tgtpop(std::vector<trajectory> &global);
    /**
     * @brief mtmct stop running.
     * @return Return ture if success, else return false.
     */
    bool stop();
    /**
     * @brief mtmct deinitialization. EOO.
     */
    void deinit();
private:
    mtmct();
    mtmct(const mtmct &other);
    mtmct &operator=(const mtmct &rhs);
    ~mtmct();
    static mtmct *self;
    //! Keep implementation in secrets.
    class implementor;
    std::unique_ptr<implementor> impl;
};

}   // namespace algorithm

#endif  // _MTMCT_H_