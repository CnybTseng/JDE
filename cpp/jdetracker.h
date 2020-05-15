#ifndef JDETRACKER_H
#define JDETRACKER_H

#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trajectory.h"

typedef std::map<int, int> Match;
typedef std::map<int, int>::iterator MatchIterator;

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
    JDETracker(void) : timestamp(0), max_lost_time(30), lambda(0.98f) {}
    ~JDETracker(void) {}
    cv::Mat motion_distance(const TrajectoryPool &a, const TrajectoryPool &b);
    void linear_assignment(const cv::Mat &cost, float cost_limit, Match &matches,
        std::vector<int> &mismatch_row, std::vector<int> &mismatch_col);
private:
    int timestamp;
    TrajectoryPool tracked_trajectories;
    TrajectoryPool lost_trajectories;
    TrajectoryPool removed_trajectories;
    int max_lost_time;
    float lambda;
};

}   // namespace mot

#endif  // JDETRACKER_H