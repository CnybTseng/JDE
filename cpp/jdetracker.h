#ifndef JDETRACKER_H
#define JDETRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "trajectory.h"

namespace mot {

class JDETracker
{
private:
    static JDETracker *me;
public:
    static JDETracker *instance(void) {
        if (!me)
            me = new JDETracker();
        return me;
    }
    bool init(void);
    bool update(const cv::Mat &dets);
    void free(void);
private:
    JDETracker(void) : timestamp(0), max_lost_time(30) {}
    ~JDETracker(void) {}
private:
    int timestamp;
    std::vector<mot::Trajectory> tracked_trajectories;
    std::vector<mot::Trajectory> lost_trajectories;
    int max_lost_time;
};

}   // namespace mot

#endif  // JDETRACKER_H