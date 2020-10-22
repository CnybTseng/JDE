#ifndef JDETRACKER_H
#define JDETRACKER_H

#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

#include "mot.h"
#include "trajectory.h"

namespace mot {

typedef std::map<int, int> Match;
typedef std::map<int, int>::iterator MatchIterator;

struct Track
{
    int id;
    cv::Vec4f ltrb;
};

class JDETracker
{
public:
    static JDETracker *instance(void);
    JDETracker(void);
    virtual ~JDETracker(void) {}
    virtual bool init(void);
    virtual bool update(const cv::Mat &dets, std::vector<Track> &tracks);
    virtual void free(void);
private:
    cv::Mat motion_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b);
    void linear_assignment(const cv::Mat &cost, float cost_limit, Match &matches,
        std::vector<int> &mismatch_row, std::vector<int> &mismatch_col);
    void remove_duplicate_trajectory(TrajectoryPool &a, TrajectoryPool &b, float iou_thresh=0.15f);
private:
    static JDETracker *me;
    int timestamp;
    TrajectoryPool candidates;
    TrajectoryPool tracked_trajectories;
    TrajectoryPool lost_trajectories;
    TrajectoryPool removed_trajectories;
    int max_lost_time;
    float lambda;
};

}   // namespace mot

#endif  // JDETRACKER_H