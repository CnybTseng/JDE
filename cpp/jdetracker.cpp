#include <algorithm>

#include "jdetracker.h"

namespace mot {

JDETracker *JDETracker::me = NULL;

bool JDETracker::init(void)
{
    return true;
}

bool JDETracker::update(const cv::Mat &dets)
{
    ++timestamp;
    std::vector<mot::Trajectory> candidates(dets.rows);
    for (int i = 0; i < dets.rows; ++i)
    {
        cv::Vec4f ltrb;
        float score = dets.at<float>(i, 1);
        const cv::Mat &embedding = dets(cv::Rect(6, i, dets.cols - 6, 1));
        candidates[i] = mot::Trajectory(ltrb, score, embedding);
    }
    
    return 0;
}

void JDETracker::free(void)
{
    
}

}   // namespace mot

#ifdef TEST_JDETRACKER_MODULE

int main()
{
    bool ret = mot::JDETracker::instance()->init();
    mot::JDETracker::instance()->free();
    return 0;
}

#endif